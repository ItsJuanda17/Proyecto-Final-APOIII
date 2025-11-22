# 📖 Instrucciones de Uso - Sistema Reorganizado

## 🎉 ¿Qué se ha mejorado?

### 1. Código Reorganizado ✅
- ✅ Separado el notebook monolítico en módulos Python reutilizables
- ✅ Estructura clara: `preprocessing`, `features`, `models`, `inference`
- ✅ Configuración centralizada en `src/config.py`
- ✅ Código documentado y mantenible

### 2. Mejora de Precisión ✅
- ✅ **Consolidación de clases**: De 11 clases a 4 principales
  - `walk_back`, `walk_front`, `walk_side`, `walking_away`, `walking_to_camera` → `walk`
  - `stand_front`, `stand_side`, `stand_site` → `stand`
  - `sit_front`, `sit_side` se mantienen separados
- ✅ **Feature engineering mejorado**: Más características (ratios, distancias adicionales)
- ✅ **Validación robusta**: Leave-One-Subject-Out (LOSO)

### 3. Interfaz en Tiempo Real ✅
- ✅ Aplicación completa con OpenCV
- ✅ Soporte para cámara web y videos
- ✅ Visualización de landmarks y predicciones

## 🚀 Pasos para Usar el Sistema

**⚠️ IMPORTANTE: Todos los comandos deben ejecutarse desde la carpeta `Entrega 3`**

### Paso 1: Instalar Dependencias

```bash
cd "Entrega 3"
pip install -r requirements.txt
```

### Paso 2: Preparar los Datos

Si tus datos están en Google Drive, descárgalos a:
```
Entrega 2/data/poses/*.parquet
```

Si necesitas procesar videos nuevos:

```bash
cd "Entrega 3"
python process_videos.py --input ruta/videos --output ../Entrega 2/data/poses
```

### Paso 3: Entrenar el Modelo

```bash
cd "Entrega 3"
python train.py
```

Este script:
1. Carga los archivos `.parquet` de `Entrega 2/data/poses/`
2. Genera el dataset de características (`features_dataset.csv`)
3. Entrena modelos (SVM, Random Forest, XGBoost)
4. Evalúa con validación LOSO
5. Guarda el mejor modelo en `models/best_model.pkl`

**Tiempo estimado**: 10-30 minutos dependiendo del tamaño del dataset

### Paso 4: Ejecutar Detección en Tiempo Real

#### Opción A: Con la cámara web
```bash
cd "Entrega 3"
python -m src.app --camera 0
```

#### Opción B: Con un video
```bash
cd "Entrega 3"
python -m src.app --video ruta/video.mp4
```

#### Opción C: Guardar video procesado
```bash
cd "Entrega 3"
python -m src.app --video entrada.mp4 --output salida.mp4
```

### Controles durante la ejecución:
- **`q`**: Salir
- **`r`**: Reiniciar el buffer de frames
- **`s`**: Guardar screenshot

## 📊 Estructura de Archivos Generados

Después de ejecutar `train.py`, tendrás:

```
models/
├── best_model.pkl          # Modelo entrenado
├── confusion_matrix.png    # Matriz de confusión
└── metrics.csv             # Métricas de evaluación
```

## 🔍 Verificar Resultados

### Ver métricas del modelo:
```python
import pandas as pd
metrics = pd.read_csv("Entrega 3/models/metrics.csv")
print(metrics)
```

### Ver distribución de clases:
```python
import pandas as pd
df = pd.read_csv("Entrega 2/data/poses/features_dataset.csv")
print(df['action'].value_counts())
```

## ⚠️ Solución de Problemas Comunes

### Error: "Modelo no encontrado"
**Solución**: Ejecuta primero `python train.py` desde la carpeta `Entrega 3` para entrenar el modelo.

### Error: "No se pudo abrir la cámara"
**Soluciones**:
- Verifica que la cámara esté conectada
- Prueba con otro índice: `--camera 1`
- En Linux: `sudo usermod -a -G video $USER` y reinicia sesión

### Baja precisión en predicciones
**Soluciones**:
- Verifica iluminación adecuada
- Asegúrate de que la persona esté completamente visible
- Considera recolectar más datos (ver `MEJORAS_PRECISION.md`)

### Error de características no coinciden
**Solución**: Asegúrate de usar la misma versión de `src/features.py` que se usó para entrenar.

## 📈 Próximos Pasos para Mejorar

1. **Recolectar más datos**: 5-10 sujetos adicionales
2. **Implementar aumentación de datos**: Ver `MEJORAS_PRECISION.md`
3. **Ajustar hiperparámetros**: Ampliar grids de búsqueda
4. **Ensambles**: Combinar múltiples modelos

Ver `MEJORAS_PRECISION.md` para más detalles.

## 🔄 Flujo de Trabajo Completo

```
1. Recolectar videos
   ↓
2. Procesar videos → cd "Entrega 3" && python process_videos.py
   ↓
3. Generar parquet files (en Entrega 2/data/poses/)
   ↓
4. Entrenar modelo → cd "Entrega 3" && python train.py
   ↓
5. Evaluar resultados (métricas, matriz de confusión)
   ↓
6. Usar en tiempo real → cd "Entrega 3" && python -m src.app
```

## 💡 Tips

- **Para desarrollo**: Usa videos cortos primero para probar rápidamente
- **Para producción**: Entrena con todos los datos disponibles
- **Para debugging**: Revisa los logs durante el entrenamiento
- **Para mejor precisión**: Sigue las recomendaciones en `MEJORAS_PRECISION.md`

## 📞 Soporte

Si encuentras problemas:
1. Revisa los mensajes de error en consola
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de que los datos estén en las rutas correctas
4. Consulta `MEJORAS_PRECISION.md` para problemas de precisión

---

¡Listo para usar! 🎉

