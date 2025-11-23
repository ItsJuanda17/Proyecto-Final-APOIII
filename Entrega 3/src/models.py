"""
Módulo de entrenamiento y evaluación de modelos
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
from typing import Dict, Tuple, Optional
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    cohen_kappa_score, make_scorer, classification_report,
    confusion_matrix
)
from sklearn.base import clone
from xgboost import XGBClassifier


from src.config import CLASS_MAPPING, FINAL_CLASSES, MODEL_CONFIG


# Suprimir warnings de XGBoost sobre parámetros deprecados
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


def consolidate_classes(df: pd.DataFrame, class_col: str = 'action') -> pd.DataFrame:
    """
    Consolida clases similares para mejorar la precisión.

    Args:
        df: DataFrame con columna de clases
        class_col: Nombre de la columna con las clases

    Returns:
        DataFrame con clases consolidadas
    """
    df = df.copy()
    df[class_col] = df[class_col].map(CLASS_MAPPING).fillna(df[class_col])
    return df


def prepare_data(
    features_path: str,
    subject_col: str = 'subject',
    class_col: str = 'action',
    consolidate: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, list]:
    """
    Prepara datos para entrenamiento.

    Returns:
        X, y, groups, label_encoder, class_names
    """
    df = pd.read_csv(features_path)

    if consolidate:
        df = consolidate_classes(df, class_col)

    df = df[df[class_col].isin(FINAL_CLASSES)]

    groups = df[subject_col].astype(str).values
    y_str = df[class_col].astype(str).values

    drop_cols = [c for c in [subject_col, class_col, 'source', 'frame_start', 'frame_end']
                 if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)

    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {class_names}")
    print(f"Sujetos: {len(np.unique(groups))}")

    return X, y, groups, le, class_names


def get_models() -> Dict:
    """Define los modelos a entrenar."""
    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        random_state=MODEL_CONFIG["random_state"]))
        ]),
        "RandomForest": RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=MODEL_CONFIG["random_state"]
        ),
        "XGBoost": XGBClassifier(
            random_state=MODEL_CONFIG["random_state"],
            eval_metric='mlogloss'
        )
    }


def get_param_grids() -> Dict:
    """Define los grids de hiperparámetros."""
    return {
        "SVM_RBF": {
            "clf__C": [1, 3, 10],
            "clf__gamma": ["scale", 0.05, 0.01],
        },
        "RandomForest": {
            "n_estimators": [300, 600],
            "max_depth": [None, 12, 20],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    }


def get_scorers() -> Dict:
    """Define las métricas de evaluación."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "mcc": make_scorer(matthews_corrcoef),
        "kappa": make_scorer(cohen_kappa_score),
    }


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: Optional[Dict] = None,
    param_grids: Optional[Dict] = None,
    scorers: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Entrena modelos con validación Leave-One-Subject-Out.

    Returns:
        DataFrame con resultados de todos los modelos
    """
    if models is None:
        models = get_models()
    if param_grids is None:
        param_grids = get_param_grids()
    if scorers is None:
        scorers = get_scorers()

    logo = LeaveOneGroupOut()
    results = []

    for name, estimator in models.items():
        print(f"\n=== Entrenando {name} ===")

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[name],
            scoring=scorers,
            refit="f1_macro",
            cv=logo.split(X, y, groups=groups),
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X, y)

        best_idx = grid.best_index_
        result = {
            "model": name,
            "best_params": grid.best_params_,
            "best_f1_macro": grid.cv_results_["mean_test_f1_macro"][best_idx],
            "best_accuracy": grid.cv_results_["mean_test_accuracy"][best_idx],
            "best_bal_acc": grid.cv_results_["mean_test_balanced_accuracy"][best_idx],
            "best_precision": grid.cv_results_["mean_test_precision_macro"][best_idx],
            "best_recall": grid.cv_results_["mean_test_recall_macro"][best_idx],
            "best_mcc": grid.cv_results_["mean_test_mcc"][best_idx],
            "best_kappa": grid.cv_results_["mean_test_kappa"][best_idx],
            "best_estimator": grid.best_estimator_,
        }

        print(f"Mejor F1 macro: {result['best_f1_macro']:.4f}")
        print(f"Mejores parámetros: {result['best_params']}")

        results.append(result)

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "best_estimator"}
        for r in results
    ]).sort_values("best_f1_macro", ascending=False)

    return results_df, results


def evaluate_loso(
    best_model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list,
    results: list
) -> Tuple[np.ndarray, Dict]:
    """
    Evalúa el mejor modelo con Leave-One-Subject-Out.

    Returns:
        Matriz de confusión y métricas
    """
    logo = LeaveOneGroupOut()
    model_config = [r for r in results if r["model"] == best_model_name][0]
    best_params = model_config["best_params"]
    base_estimator = get_models()[best_model_name]

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        estimator = clone(base_estimator)

        # Aplicar mejores parámetros
        if best_model_name == "SVM_RBF":
            estimator.set_params(**{
                f"clf__{k.split('__')[-1]}": v
                for k, v in best_params.items()
            })
        else:
            estimator.set_params(**best_params)

        estimator.fit(X.iloc[train_idx], y[train_idx])
        preds = estimator.predict(X.iloc[test_idx])

        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Calcular métricas
    metrics = {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
        "f1_macro": f1_score(y_true_all, y_pred_all, average="macro"),
        "precision_macro": precision_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true_all, y_pred_all),
        "kappa": cohen_kappa_score(y_true_all, y_pred_all),
    }

    print("\n=== Evaluación LOSO Final ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n--- Reporte por clase ---")
    print(classification_report(
        y_true_all,
        y_pred_all,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=3,
        zero_division=0))

    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(class_names)))
    return cm, metrics


def save_model(model, label_encoder: LabelEncoder, path: str):
    """Guarda el modelo y el label encoder."""
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder
    }, path)
    print(f"Modelo guardado en: {path}")


def load_model(path: str) -> Tuple:
    """Carga el modelo y el label encoder."""
    data = joblib.load(path)
    return data['model'], data['label_encoder']
