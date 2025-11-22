"""
Módulo de inferencia en tiempo real
"""
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from collections import deque
from typing import Optional, Tuple
import joblib

from src.config import MEDIAPIPE_CONFIG, FEATURE_CONFIG, NUM_LANDMARKS
from src.preprocessing import clean_pose_dataframe
from src.features import build_frame_features, window_aggregate


class RealTimeActivityDetector:
    """
    Detector de actividades en tiempo real usando MediaPipe y modelo entrenado.
    """
    
    def __init__(
        self,
        model_path: str,
        window_size: int = None,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Ruta al modelo guardado
            window_size: Tamaño de ventana para agregación
            confidence_threshold: Umbral de confianza para predicción
        """
        if window_size is None:
            window_size = FEATURE_CONFIG["window_size"]
        
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Cargar modelo
        data = joblib.load(model_path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.class_names = list(self.label_encoder.classes_)
        
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Buffer para frames
        self.frame_buffer = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=10)
        
        print(f"Modelo cargado: {len(self.class_names)} clases")
        print(f"Clases: {self.class_names}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        """
        Procesa un frame y retorna frame anotado, actividad predicha y confianza.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            frame_annotated, activity, confidence
        """
        # Convertir a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Detectar pose
        results = self.pose.process(rgb)
        
        # Dibujar landmarks
        frame_annotated = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Extraer landmarks
        landmarks = self._extract_landmarks(results, frame.shape[1], frame.shape[0])
        
        if landmarks is not None:
            self.frame_buffer.append(landmarks)
        
        # Predecir si tenemos suficientes frames
        activity = None
        confidence = 0.0
        
        if len(self.frame_buffer) >= self.window_size:
            activity, confidence = self._predict_activity()
            if activity:
                self.prediction_history.append((activity, confidence))
        
        # Mostrar predicción más reciente
        if self.prediction_history:
            latest_activity, latest_conf = self.prediction_history[-1]
            if latest_conf >= self.confidence_threshold:
                self._draw_prediction(frame_annotated, latest_activity, latest_conf)
        
        return frame_annotated, activity, confidence
    
    def _extract_landmarks(
        self,
        results: mp.solutions.pose.Pose,
        width: int,
        height: int
    ) -> Optional[pd.DataFrame]:
        """Extrae landmarks de un frame."""
        if not results.pose_landmarks:
            return None
        
        row = {
            "frame_idx": len(self.frame_buffer),
            "t_sec": len(self.frame_buffer) / 30.0,  # Asumir 30 FPS
            "width": width,
            "height": height,
            "fps": 30.0
        }
        
        for i, landmark in enumerate(results.pose_landmarks.landmark[:NUM_LANDMARKS]):
            row[f"x_{i}"] = landmark.x * width
            row[f"y_{i}"] = landmark.y * height
            row[f"vis_{i}"] = getattr(landmark, "visibility", 0.0)
        
        return pd.DataFrame([row])
    
    def _predict_activity(self) -> Tuple[Optional[str], float]:
        """Predice la actividad usando el buffer de frames."""
        # Convertir buffer a DataFrame
        df = pd.concat(list(self.frame_buffer), ignore_index=True)
        
        # Preprocesar
        df = clean_pose_dataframe(df)
        
        if len(df) < self.window_size:
            return None, 0.0
        
        # Extraer características
        df = build_frame_features(df)
        
        # Crear ventana agregada
        windows = window_aggregate(df, window_size=self.window_size, stride=self.window_size)
        
        if not windows:
            return None, 0.0
        
        # Convertir a DataFrame para predicción
        window_df = pd.DataFrame([windows[-1]])  # Última ventana
        
        # Eliminar columnas no numéricas
        drop_cols = ['frame_start', 'frame_end', 'fps']
        feature_cols = [c for c in window_df.columns if c not in drop_cols]
        X = window_df[feature_cols].select_dtypes(include=[np.number])
        
        try:
            # Asegurar que tenemos las mismas características que el modelo
            # Si el modelo es un Pipeline, obtener los nombres de características esperados
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = self.model.feature_names_in_
                # Reordenar y rellenar características faltantes
                X_aligned = pd.DataFrame(0, index=[0], columns=expected_features)
                for col in X.columns:
                    if col in expected_features:
                        X_aligned[col] = X[col].values[0]
                X = X_aligned
            elif hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                # Para Pipeline, intentar con las características disponibles
                pass
            
            # Predecir
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            activity = self.label_encoder.inverse_transform([pred])[0]
            confidence = float(np.max(proba))
            
            return activity, confidence
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def _draw_prediction(
        self,
        frame: np.ndarray,
        activity: str,
        confidence: float
    ):
        """Dibuja la predicción en el frame."""
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texto
        text = f"Actividad: {activity}"
        conf_text = f"Confianza: {confidence:.2%}"
        
        cv2.putText(frame, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def reset(self):
        """Reinicia el buffer de frames."""
        self.frame_buffer.clear()
        self.prediction_history.clear()
    
    def __del__(self):
        """Libera recursos."""
        if hasattr(self, 'pose'):
            self.pose.close()

