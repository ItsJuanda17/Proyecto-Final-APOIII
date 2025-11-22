# 🧍‍♀️ Sistema de Análisis de Actividades Humanas

Proyecto Final - Algoritmo y Programación III (APO3)  
Universidad ICESI - Facultad de Ingeniería, Diseño y Ciencias Aplicadas

## 📋 Descripción

Sistema de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, sentarse, ponerse de pie) y realizar un seguimiento de movimientos articulares y posturales en tiempo real usando MediaPipe y modelos de Machine Learning.

## 🎯 Características

- **Detección de poses**: Extracción de 33 landmarks corporales usando MediaPipe
- **Clasificación de actividades**: Reconocimiento de actividades usando modelos supervisados (SVM, Random Forest, XGBoost)
- **Análisis postural**: Cálculo de ángulos articulares, inclinación del tronco y velocidades
- **Interfaz en tiempo real**: Visualización de detecciones usando la cámara o videos

## 🏗️ Estructura del Proyecto

```
Proyecto-Final-APOIII/
│
├── Entrega 1/                    # Primera entrega
│   ├── docs/
│   └── src/
│
├── Entrega 2/                    # Segunda entrega
│   ├── data/
│   │   ├── poses/                # Archivos parquet y CSV
│   │   └── videos/               # Videos de entrenamiento
│   ├── docs/
│   └── Proyecto.ipynb
│
└── Entrega 3/                    # Tercera entrega (Código reorganizado)
    ├── src/                      # Código fuente principal
    │   ├── __init__.py
    │   ├── config.py             # Configuración global
    │   ├── preprocessing.py      # Preprocesamiento de datos
    │   ├── features.py           # Extracción de características
    │   ├── models.py             # Entrenamiento de modelos
    │   ├── inference.py          # Inferencia en tiempo real
    │   └── app.py                # Aplicación principal
    │
    ├── models/                   # Modelos entrenados (generado)
    │   ├── best_model.pkl
    │   ├── confusion_matrix.png
    │   └── metrics.csv
    │
    ├── train.py                  # Script de entrenamiento
    ├── process_videos.py         # Script para procesar videos
    ├── check_setup.py            # Script de verificación
    ├── requirements.txt          # Dependencias
    ├── README.md                 # Este archivo
    ├── INSTRUCCIONES_USO.md      # Guía de uso detallada
    └── MEJORAS_PRECISION.md      # Recomendaciones de mejora
```

## 🚀 Instalación

### Requisitos

- Python 3.8 o superior
- pip

### Pasos

1. **Clonar el repositorio** (o descargar el proyecto)

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 📊 Uso

### 1. Generar Dataset de Características

**⚠️ IMPORTANTE: Ejecuta desde la carpeta `Entrega 3`**

Si ya tienes los archivos `.parquet` con los landmarks extraídos, puedes generar el dataset de características ejecutando:

```bash
cd "Entrega 3"
python train.py
```

Este script:
- Carga los archivos parquet de `../Entrega 2/data/poses/`
- Preprocesa y extrae características
- Genera `features_dataset.csv` en `../Entrega 2/data/poses/`
- Entrena modelos (SVM, Random Forest, XGBoost)
- Evalúa con validación Leave-One-Subject-Out (LOSO)
- Guarda el mejor modelo en `models/best_model.pkl`

### 2. Ejecutar Detección en Tiempo Real

#### Con la cámara web:
```bash
cd "Entrega 3"
python -m src.app --camera 0
```

#### Con un video:
```bash
cd "Entrega 3"
python -m src.app --video ruta/al/video.mp4
```

#### Guardar video procesado:
```bash
cd "Entrega 3"
python -m src.app --video entrada.mp4 --output salida.mp4
```

#### Opciones disponibles:
- `--model`: Ruta al modelo (default: `models/best_model.pkl`)
- `--camera`: Índice de la cámara (default: 0)
- `--video`: Ruta a video para procesar
- `--output`: Ruta para guardar video procesado

#### Controles durante la ejecución:
- `q`: Salir
- `r`: Reiniciar el buffer de frames
- `s`: Guardar screenshot

## 🔧 Mejoras Implementadas

### 1. Reorganización del Código
- ✅ Separación en módulos Python reutilizables
- ✅ Configuración centralizada
- ✅ Código documentado y mantenible

### 2. Mejora de Precisión
- ✅ **Consolidación de clases**: Se agruparon clases similares para reducir el desbalance:
  - Variantes de caminar → `walk`
  - Variantes de estar de pie → `stand`
  - Sentarse se mantiene separado por perspectiva (`sit_front`, `sit_side`)
- ✅ **Feature engineering mejorado**:
  - Ángulos articulares (codos, rodillas, caderas)
  - Inclinación del tronco
  - Velocidades de puntos clave
  - Distancias entre articulaciones
  - Ratios corporales (altura/ancho, altura de cadera)
- ✅ **Validación robusta**: Leave-One-Subject-Out para evitar sobreajuste

### 3. Interfaz en Tiempo Real
- ✅ Visualización de landmarks en video
- ✅ Predicción de actividad en tiempo real
- ✅ Muestra de confianza de la predicción
- ✅ Soporte para cámara y videos

## 📈 Resultados Esperados

Después de consolidar las clases y mejorar las características, se espera:
- **Mejor precisión**: Reducción de clases de 11 a 4 principales
- **Mejor balance**: Distribución más equilibrada de muestras por clase
- **Mejor generalización**: Validación LOSO asegura que el modelo funciona con nuevos sujetos

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
Asegúrate de haber entrenado el modelo primero ejecutando `python train.py` desde la carpeta `Entrega 3`

### Error: "No se pudo abrir la cámara"
- Verifica que la cámara esté conectada
- Prueba con un índice diferente: `--camera 1`
- En Linux, puede requerir permisos: `sudo usermod -a -G video $USER`

### Baja precisión en predicciones
- Verifica que haya suficiente iluminación
- Asegúrate de que la persona esté completamente visible en el frame
- Considera entrenar con más datos

## 📝 Notas Importantes

1. **Ejecutar desde Entrega 3**: Todos los scripts deben ejecutarse desde la carpeta `Entrega 3` para que las rutas funcionen correctamente.

2. **Datos en Google Drive**: Los datos originales están organizados en Google Drive. Asegúrate de tener los archivos `.parquet` en `Entrega 2/data/poses/` antes de entrenar.

2. **Consolidación de clases**: El sistema consolida automáticamente clases similares para mejorar la precisión. Ver `src/config.py` para personalizar el mapeo.

3. **Ventana temporal**: El modelo usa ventanas de 32 frames para hacer predicciones. Se necesita al menos 1 segundo de video a 30 FPS.

## 👥 Integrantes

- Juan David Acevedo - A00399081
- Santiago Santacruz - A00378149
- Esteban Cuellar - A00402548

## 📚 Referencias

- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

## 📄 Licencia

Este proyecto es parte de un trabajo académico de la Universidad ICESI.

---


