"""
Script para entrenar el modelo de clasificación de actividades
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings

# Suprimir warnings de XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from src.config import PROJECT_ROOT, POSES_DIR
from src.models import (
    prepare_data, train_models, evaluate_loso, save_model, get_models
)
from src.preprocessing import extract_poses_from_video, clean_pose_dataframe
from src.features import build_frame_features, window_aggregate
import glob


def generate_features_dataset():
    """
    Genera el dataset de características a partir de los archivos parquet.
    """
    print("=== Generando dataset de características ===")
    
    parquet_files = sorted(glob.glob(str(POSES_DIR / "*.parquet")))
    print(f"Encontrados {len(parquet_files)} archivos parquet")
    
    all_windows = []
    
    for pq_path in parquet_files:
        print(f"Procesando: {Path(pq_path).name}")
        
        # Cargar parquet
        df = pd.read_parquet(pq_path)
        
        if len(df) < 32:
            print(f"  Saltando: muy pocos frames ({len(df)})")
            continue
        
        # Preprocesar
        df = clean_pose_dataframe(df)
        
        if len(df) < 32:
            print(f"  Saltando: muy pocos frames después de limpieza ({len(df)})")
            continue
        
        # Extraer características
        df = build_frame_features(df)
        
        # Crear ventanas
        # Extraer subject y action del nombre del archivo
        name = Path(pq_path).stem
        parts = name.split("__")
        subject = parts[0] if len(parts) > 0 else "unknown"
        action = parts[1] if len(parts) > 1 else "unknown"
        
        windows = window_aggregate(df, keep_cols=('fps',))
        
        for win in windows:
            win['subject'] = subject
            win['action'] = action
            win['source'] = Path(pq_path).name
        
        all_windows.extend(windows)
        print(f"  Generadas {len(windows)} ventanas")
    
    # Crear DataFrame
    features_df = pd.DataFrame(all_windows)
    
    # Guardar
    output_path = POSES_DIR / "features_dataset.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nDataset guardado en: {output_path}")
    print(f"Total de ventanas: {len(features_df)}")
    print(f"Distribución por clase:")
    print(features_df['action'].value_counts())
    
    return str(output_path)


def main():
    # Generar dataset si no existe
    features_path = POSES_DIR / "features_dataset.csv"
    if not features_path.exists():
        print("Dataset de características no encontrado. Generando...")
        features_path = generate_features_dataset()
    else:
        print(f"Usando dataset existente: {features_path}")
        features_path = str(features_path)
    
    # Preparar datos
    print("\n=== Preparando datos ===")
    X, y, groups, label_encoder, class_names = prepare_data(
        features_path,
        consolidate=True
    )
    
    # Entrenar modelos
    print("\n=== Entrenando modelos ===")
    results_df, results = train_models(X, y, groups)
    
    print("\n=== Resultados ===")
    print(results_df)
    
    # Evaluar mejor modelo
    best_model_name = results_df.iloc[0]["model"]
    print(f"\n=== Evaluando mejor modelo: {best_model_name} ===")
    
    cm, metrics = evaluate_loso(
        best_model_name, X, y, groups, class_names, results
    )
    
    # Guardar mejor modelo
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    best_model = [r for r in results if r["model"] == best_model_name][0]["best_estimator"]
    model_path = models_dir / "best_model.pkl"
    save_model(best_model, label_encoder, str(model_path))
    
    # Guardar matriz de confusión
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(models_dir / "confusion_matrix.png")
    print(f"\nMatriz de confusión guardada en: {models_dir / 'confusion_matrix.png'}")
    
    # Guardar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(models_dir / "metrics.csv", index=False)
    print(f"Métricas guardadas en: {models_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

