# 🧍‍♀️ Proyecto: Análisis de Actividades y Movimiento Humano

## Pregunta(s) de interés

¿Cómo podemos desarrollar una herramienta capaz de **reconocer y analizar actividades humanas básicas** (caminar hacia/desde la cámara, girar, sentarse, ponerse de pie) y realizar un **seguimiento preciso de articulaciones y posturas** a partir de video?

Preguntas específicas:
- ¿Qué tan bien puede el modelo identificar y seguir las articulaciones principales (caderas, rodillas, muñecas, hombros) en condiciones reales?
- ¿Cómo se pueden usar los ángulos articulares y la inclinación del tronco para diferenciar cada actividad?
- ¿Qué características (features) son más útiles para una futura clasificación automática de la actividad?

---

## Tipo de problema

El problema pertenece a la categoría de **Visión por Computadora y Reconocimiento de Acciones Humanas**.  
Más específicamente:
- **Tipo:** problema de *clasificación supervisada secuencial* (cuando se entrene el modelo final).  
- **Etapa actual (Entrega 1):** *recolección y análisis exploratorio de datos (EDA)*.  
- **Entrada:** secuencias de video.  
- **Salida esperada:** archivos de características (landmarks por frame) y métricas de movimiento.

---

## Metodología

1. **Extracción de datos**  
   - Se procesaron los videos usando **MediaPipe Pose** y **OpenCV** en Google Colab.  
   - Se generaron archivos `.parquet` por video, conteniendo coordenadas `(x, y)` y visibilidad de 33 puntos del esqueleto humano.  
   - Los resultados se almacenaron en Google Drive y se resumieron en un CSV con metadatos (frames, fps, resolución, porcentaje de detección, etc.).

2. **Análisis exploratorio (EDA)**  
   - Se evaluó la **cobertura de detección** (% de frames con landmarks válidos).  
   - Se analizaron los **ángulos articulares** (cadera, rodilla) y la **inclinación lateral del tronco** a lo largo del tiempo.  
   - Se generaron visualizaciones de cada video para verificar estabilidad del seguimiento y variación del movimiento.

3. **Estructura del flujo de trabajo**  
   - Carpeta `/videos`: videos originales (.mp4).  
   - Carpeta `/poses`: archivos `.parquet` y resúmenes `.csv` generados automáticamente.  
   - Notebooks de extracción y análisis en `/notebooks`.

4. **Validación técnica**  
   - Cada video genera un reporte automático de detección (`frames`, `miss_frames`, `fps`, `coverage_%`, `vis_mean`).  
   - Se comprobó visualmente que los landmarks coinciden con la posición corporal real.

---

##  Métricas de progreso

Durante esta primera entrega, las métricas se enfocan en **calidad de detección** y **consistencia de datos**:

| Métrica | Descripción |
|----------|--------------|
| `coverage_%` | % de frames con detección válida de pose |
| `vis_mean` | visibilidad promedio de landmarks |
| `miss_frames` | frames sin detección |
| `frames` | cantidad total procesada | 

En etapas posteriores se añadirán:
- *Accuracy / F1-score* de la clasificación de actividad.  
- *Errores angulares medios* (MAE) para validación biomecánica.  

---

## Siguientes pasos

1. **Ingeniería de características**
   - Calcular ángulos relativos, velocidades articulares e inclinaciones promedio.  
   - Generar ventanas temporales (secuencias de N frames) para usar como entrada a modelos ML.

2. **Clasificación automática**
   - Entrenar modelos basados en Random Forest, LSTM o CNN 1D para identificar la acción.  
   - Evaluar con métricas de clasificación (precision, recall, F1).

3. **Interfaz o dashboard**
   - Construir una herramienta que cargue videos y muestre análisis en tiempo real o post-procesado.

4. **Optimización**
   - Ajustar parámetros de MediaPipe (`model_complexity`, `min_detection_confidence`, `min_tracking_confidence`) según la variabilidad de los sujetos.

---

## Estrategias para ampliar el conjunto de datos

Para mejorar la robustez del sistema y cubrir más variabilidad corporal y ambiental:

1. **Recolección propia adicional**
   - Grabar más sujetos (diversas edades, estaturas y contextos).  
   - Usar diferentes perspectivas (frontal, 45°, lateral).  
   - Cambiar la iluminación y el fondo para mejorar generalización.

2. **Uso de videos de dominio público**
   - Incorporar clips libres de derechos (Pexels, Pixabay, Videvo, YouTube Creative Commons).  
   - Seleccionar únicamente videos con el cuerpo completo visible y buena calidad.

3. **Aumento sintético de datos**
   - Aplicar transformaciones: recortes, cambios de brillo, espejo horizontal.  
   - Simular ruido en coordenadas o pequeñas variaciones de cámara.

4. **Anotación automática**
   - Asignar etiquetas de acción (walk, sit, stand, turn) automáticamente según el nombre del archivo o reglas basadas en ángulos.

---

## Consideraciones éticas

La implementación de IA en análisis de movimiento humano implica varios aspectos éticos que deben atenderse:

1. **Privacidad y consentimiento**
   - Todos los participantes deben ser informados y aceptar el uso de sus videos exclusivamente con fines académicos.  
   - Los datos deben almacenarse de manera segura (sin rostros reconocibles si se comparte el dataset).

2. **Sesgo de datos**
   - Evitar conjuntos con una única morfología o contexto (por ejemplo, solo una persona o un entorno).  
   - Incluir diversidad corporal, de género, edad y ropa para que el modelo no discrimine.

3. **Uso responsable**
   - Las técnicas de seguimiento corporal no deben usarse para vigilancia sin consentimiento.  
   - Las salidas del modelo deben interpretarse con precaución: no sustituyen evaluación médica.

4. **Transparencia**
   - Publicar claramente los alcances y limitaciones del sistema.  
   - Documentar cómo se procesan y almacenan los datos (evitar cajas negras).

5. **Sostenibilidad**
   - Priorizar herramientas ligeras y reproducibles (MediaPipe, OpenCV) para minimizar consumo computacional.

---

## Conclusión

Esta primera etapa permitió construir un **pipeline funcional y reproducible** para la extracción de datos de postura a partir de video.  
Se generó un dataset estructurado que servirá como base para entrenar modelos de **clasificación de actividades humanas** y análisis postural.  

---

> **Autores:** 
- Juan David Acevedo
- Esteban Cuellar
- Santiago Santacruz  
> **Curso:** Proyecto Final — Análisis de Movimiento Humano  
> **Fecha:** Octubre 2025
