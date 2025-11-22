"""
Módulo de preprocesamiento de datos de poses
"""
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional
import math

from src.config import (
    NUM_LANDMARKS, MEDIAPIPE_CONFIG, PREPROCESSING_CONFIG,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP
)


def extract_poses_from_video(
    video_path: str,
    stride: int = 1,
    **mediapipe_kwargs
) -> pd.DataFrame:
    """
    Extrae landmarks de poses de un video usando MediaPipe.
    
    Args:
        video_path: Ruta al video
        stride: Procesar cada N frames (1 = todos)
        **mediapipe_kwargs: Parámetros adicionales para MediaPipe
        
    Returns:
        DataFrame con landmarks por frame
    """
    config = {**MEDIAPIPE_CONFIG, **mediapipe_kwargs}
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps == 0 or math.isclose(fps, 0.0):
        fps = 24.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    rows = []
    frame_idx = -1
    missed_frames = 0
    
    with mp_pose.Pose(**config) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            
            t_sec = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            
            # Inicializar arrays
            x_coords = np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)
            y_coords = np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)
            visibility = np.zeros(NUM_LANDMARKS, dtype=np.float32)
            
            if result.pose_landmarks:
                for i, landmark in enumerate(result.pose_landmarks.landmark[:NUM_LANDMARKS]):
                    x_coords[i] = landmark.x * width
                    y_coords[i] = landmark.y * height
                    visibility[i] = getattr(landmark, "visibility", 0.0)
            else:
                missed_frames += 1
            
            # Crear fila
            row = {
                "frame_idx": frame_idx,
                "t_sec": float(t_sec),
                "width": width,
                "height": height,
                "fps": float(fps)
            }
            
            for i in range(NUM_LANDMARKS):
                row[f"x_{i}"] = float(x_coords[i])
                row[f"y_{i}"] = float(y_coords[i])
                row[f"vis_{i}"] = float(visibility[i])
            
            rows.append(row)
    
    cap.release()
    return pd.DataFrame(rows)


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza coordenadas x,y a [0,1] para independencia de resolución."""
    df = df.copy()
    w = df['width'].astype(float).replace(0, np.nan)
    h = df['height'].astype(float).replace(0, np.nan)
    
    for i in range(NUM_LANDMARKS):
        df[f'x_{i}'] = df[f'x_{i}'] / w
        df[f'y_{i}'] = df[f'y_{i}'] / h
    
    return df


def filter_low_visibility(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Filtra frames con visibilidad promedio baja."""
    if threshold is None:
        threshold = PREPROCESSING_CONFIG["visibility_threshold"]
    
    vis_cols = [f'vis_{i}' for i in range(NUM_LANDMARKS)]
    if not all(col in df.columns for col in vis_cols):
        # Si no hay visibilidad, usar presencia de coordenadas
        x_cols = [f'x_{i}' for i in range(NUM_LANDMARKS)]
        mask = ~df[x_cols].isna().all(axis=1)
    else:
        mask = df[vis_cols].mean(axis=1) >= threshold
    
    return df.loc[mask].reset_index(drop=True)


def smooth_coordinates(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Suaviza coordenadas usando media móvil."""
    if window is None:
        window = PREPROCESSING_CONFIG["smooth_window"]
    
    df = df.copy()
    for i in range(NUM_LANDMARKS):
        df[f'x_{i}'] = df[f'x_{i}'].rolling(window, center=True, min_periods=1).mean()
        df[f'y_{i}'] = df[f'y_{i}'].rolling(window, center=True, min_periods=1).mean()
    
    return df


def center_by_pelvis(df: pd.DataFrame) -> pd.DataFrame:
    """Centra el esqueleto restando el punto medio de las caderas."""
    df = df.copy()
    mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    
    for i in range(NUM_LANDMARKS):
        df[f'x_{i}'] = df[f'x_{i}'] - mid_hip_x
        df[f'y_{i}'] = df[f'y_{i}'] - mid_hip_y
    
    return df


def clean_pose_dataframe(
    df: pd.DataFrame,
    vis_threshold: float = None,
    smooth_window: int = None,
    use_centering: bool = None
) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento.
    
    Args:
        df: DataFrame con landmarks
        vis_threshold: Umbral de visibilidad
        smooth_window: Ventana para suavizado
        use_centering: Si centrar por pelvis
        
    Returns:
        DataFrame preprocesado
    """
    if vis_threshold is None:
        vis_threshold = PREPROCESSING_CONFIG["visibility_threshold"]
    if smooth_window is None:
        smooth_window = PREPROCESSING_CONFIG["smooth_window"]
    if use_centering is None:
        use_centering = PREPROCESSING_CONFIG["use_centering"]
    
    # 1. Filtrar baja visibilidad
    df = filter_low_visibility(df, threshold=vis_threshold)
    if len(df) == 0:
        return df
    
    # 2. Normalizar coordenadas
    df = normalize_coordinates(df)
    
    # 3. Suavizar
    df = smooth_coordinates(df, window=smooth_window)
    
    # 4. Centrar (opcional)
    if use_centering:
        df = center_by_pelvis(df)
    
    return df

