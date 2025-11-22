# 🎯 Guía para Mejorar la Precisión del Modelo

## Problemas Identificados

1. **Desbalance de clases**: Algunas clases tienen muy pocos ejemplos (stand_site: 2, walk_away: 6)
2. **Baja precisión general**: F1 macro ~0.37, Accuracy ~0.57
3. **Falta de datos**: Solo 7 sujetos, algunos con pocas actividades

## Soluciones Implementadas

### 1. Consolidación de Clases ✅

Se agruparon clases similares para reducir el desbalance:

```python
CLASS_MAPPING = {
    "walk_back": "walk",
    "walk_front": "walk",
    "walk_side": "walk",
    "walking_away": "walk",
    "walking_to_camera": "walk",
    "walk_away": "walk",
    "stand_front": "stand",
    "stand_side": "stand",
    "stand_site": "stand",
}
```

**Resultado esperado**: De 11 clases a 4 clases principales, mejorando el balance.

### 2. Feature Engineering Mejorado ✅

Se añadieron características adicionales:

- **Ratios corporales**: `hip_height_ratio`, `body_aspect_ratio`
- **Distancias adicionales**: Entre muñecas, tobillos
- **Altura total del cuerpo**: Para normalización mejorada

### 3. Validación Leave-One-Subject-Out ✅

Asegura que el modelo generalice a nuevos sujetos, no solo a nuevos frames.

## Recomendaciones Adicionales

### 1. Recolectar Más Datos

**Prioridad: ALTA**

- **Más sujetos**: Idealmente 15-20 sujetos diferentes
- **Más variaciones**: Diferentes alturas, pesos, edades
- **Más perspectivas**: Frontal, lateral, 45 grados
- **Más condiciones**: Diferentes iluminaciones, fondos

**Cómo hacerlo**:
```bash
# Usar el script para procesar nuevos videos
python process_videos.py --input ruta/nuevos/videos --output Entrega 2/data/poses
```

### 2. Aumento de Datos (Data Augmentation)

**Prioridad: MEDIA**

- **Espejo horizontal**: Duplicar videos reflejados
- **Variaciones de velocidad**: Acelerar/ralentizar videos
- **Ruido en coordenadas**: Añadir pequeñas variaciones aleatorias a los landmarks
- **Rotaciones menores**: Rotar ligeramente las coordenadas

**Ejemplo de implementación**:
```python
# En src/features.py, añadir función de augmentación
def augment_landmarks(df, mirror=True, noise_std=0.01):
    if mirror:
        # Reflejar coordenadas x
        for i in range(33):
            df[f'x_{i}'] = 1.0 - df[f'x_{i}']
    if noise_std > 0:
        # Añadir ruido gaussiano
        for i in range(33):
            df[f'x_{i}'] += np.random.normal(0, noise_std, len(df))
            df[f'y_{i}'] += np.random.normal(0, noise_std, len(df))
    return df
```

### 3. Ajuste de Hiperparámetros Más Exhaustivo

**Prioridad: MEDIA**

Ampliar los grids de búsqueda:

```python
# En src/models.py
param_grids = {
    "SVM_RBF": {
        "clf__C": [0.1, 1, 3, 10, 30],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1],
    },
    "RandomForest": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 15, 20, 25],
        "max_features": ["sqrt", 0.3, 0.5, 0.7],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}
```

### 4. Reducción de Características

**Prioridad: BAJA**

Después de tener más datos, usar técnicas de selección de características:

- **Feature importance**: Usar importancia de RandomForest/XGBoost
- **Correlación**: Eliminar características altamente correlacionadas
- **PCA**: Reducir dimensionalidad manteniendo varianza

### 5. Modelos Secuenciales

**Prioridad: BAJA** (más complejo)

Para capturar dependencias temporales:

- **LSTM**: Para secuencias de frames
- **CNN 1D**: Para patrones temporales en características
- **Transformer**: Para atención temporal

**Nota**: Requiere más datos y tiempo de entrenamiento.

### 6. Ensambles

**Prioridad: MEDIA**

Combinar predicciones de múltiples modelos:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)
```

### 7. Balanceo de Clases

**Prioridad: MEDIA**

Si después de consolidar aún hay desbalance:

- **SMOTE**: Generar muestras sintéticas de clases minoritarias
- **Undersampling**: Reducir muestras de clases mayoritarias
- **Class weights**: Ajustar pesos en el entrenamiento (ya implementado)

## Plan de Acción Recomendado

### Fase 1: Datos (1-2 semanas)
1. ✅ Consolidar clases (YA HECHO)
2. Recolectar más videos (5-10 sujetos adicionales)
3. Procesar y añadir al dataset

### Fase 2: Mejoras de Modelo (1 semana)
1. Implementar aumentación de datos
2. Ampliar grid de hiperparámetros
3. Re-entrenar modelos

### Fase 3: Optimización (1 semana)
1. Selección de características
2. Ensambles
3. Validación final

## Métricas Objetivo

- **Accuracy**: > 0.75
- **F1 Macro**: > 0.70
- **Balanced Accuracy**: > 0.70
- **Por clase**: Precision y Recall > 0.65 para todas las clases

## Scripts Útiles

### Ver distribución de clases:
```python
import pandas as pd
df = pd.read_csv("Entrega 2/data/poses/features_dataset.csv")
print(df['action'].value_counts())
```

### Verificar calidad de datos:
```python
# Verificar frames válidos por video
summary = pd.read_csv("Entrega 2/data/poses/poses_summary.csv")
print(summary[['video', 'valid_ratio']].sort_values('valid_ratio'))
```

## Notas Finales

- **Paciencia**: Mejorar precisión requiere tiempo y datos
- **Iteración**: Probar una mejora a la vez para entender su impacto
- **Validación**: Siempre usar LOSO para evaluar generalización real
- **Documentación**: Registrar qué cambios mejoran/deterioran resultados

