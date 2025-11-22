"""
Módulo de extracción de características
"""
import numpy as np
import pandas as pd
from typing import List, Dict
import math

from src.config import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    FEATURE_CONFIG
)


def angle_3_points(ax, ay, bx, by, cx, cy):
    """
    Calcula el ángulo ABC en grados [0, 180].
    Vectorizado para trabajar con Series/arrays.
    """
    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by
    
    num = (ba_x * bc_x + ba_y * bc_y)
    den = (np.sqrt(ba_x**2 + ba_y**2) * 
           (np.sqrt(bc_x**2 + bc_y**2) + 1e-9) + 1e-9)
    
    cos_angle = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calculate_joint_angle(df: pd.DataFrame, point_a: int, point_b: int, 
                          point_c: int, output_name: str) -> pd.DataFrame:
    """Calcula el ángulo de una articulación."""
    df = df.copy()
    ax, ay = df[f'x_{point_a}'], df[f'y_{point_a}']
    bx, by = df[f'x_{point_b}'], df[f'y_{point_b}']
    cx, cy = df[f'x_{point_c}'], df[f'y_{point_c}']
    
    df[output_name] = angle_3_points(ax, ay, bx, by, cx, cy)
    return df


def calculate_trunk_inclination(df: pd.DataFrame, output_name: str = 'trunk_incl_deg') -> pd.DataFrame:
    """
    Calcula la inclinación absoluta del tronco (ángulo con vertical).
    """
    df = df.copy()
    mid_sh_x = (df[f'x_{L_SHOULDER}'] + df[f'x_{R_SHOULDER}']) / 2
    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
    mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    
    vx, vy = mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y
    norm = np.sqrt(vx**2 + vy**2) + 1e-9
    cos_angle = np.clip(vy / norm, -1.0, 1.0)
    
    df[output_name] = np.degrees(np.arccos(cos_angle))
    return df


def calculate_trunk_inclination_signed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la inclinación lateral del tronco CON signo (para visualización).
    """
    df = df.copy()
    mid_sh_x = (df[f'x_{L_SHOULDER}'] + df[f'x_{R_SHOULDER}']) / 2
    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
    mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    
    vx, vy = mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y
    # Ángulo respecto a vertical (eje y crece hacia abajo)
    df['trunk_deg'] = np.degrees(np.arctan2(vx, vy))
    return df


def add_velocities(df: pd.DataFrame, points: List[int], fps_col: str = 'fps') -> pd.DataFrame:
    """
    Calcula velocidades para puntos clave.
    
    Args:
        df: DataFrame con landmarks
        points: Lista de índices de landmarks
        fps_col: Nombre de la columna con FPS
        
    Returns:
        DataFrame con columnas de velocidad añadidas
    """
    df = df.copy()
    fps = float(df[fps_col].iloc[0]) if fps_col in df.columns else 30.0
    
    for point_idx in points:
        vx = df[f'x_{point_idx}'].diff() * fps
        vy = df[f'y_{point_idx}'].diff() * fps
        df[f'vel_{point_idx}'] = np.sqrt(vx**2 + vy**2).fillna(0.0)
    
    return df


def calculate_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula distancias entre puntos clave (características adicionales).
    """
    df = df.copy()
    
    # Distancia entre hombros
    df['dist_shoulders'] = np.sqrt(
        (df[f'x_{L_SHOULDER}'] - df[f'x_{R_SHOULDER}'])**2 +
        (df[f'y_{L_SHOULDER}'] - df[f'y_{R_SHOULDER}'])**2
    )
    
    # Distancia entre caderas
    df['dist_hips'] = np.sqrt(
        (df[f'x_{L_HIP}'] - df[f'x_{R_HIP}'])**2 +
        (df[f'y_{L_HIP}'] - df[f'y_{R_HIP}'])**2
    )
    
    # Altura del tronco (hombros a caderas)
    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    df['trunk_height'] = np.abs(mid_sh_y - mid_hip_y)
    
    # Ancho del cuerpo (hombros a caderas en x)
    mid_sh_x = (df[f'x_{L_SHOULDER}'] + df[f'x_{R_SHOULDER}']) / 2
    mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
    df['body_width'] = np.abs(mid_sh_x - mid_hip_x)
    
    # Altura total (cabeza a tobillos) - usando punto medio de tobillos
    # Nota: MediaPipe no tiene cabeza directamente, usamos hombros como proxy superior
    mid_ankle_y = (df[f'y_{L_ANKLE}'] + df[f'y_{R_ANKLE}']) / 2
    df['body_height'] = np.abs(mid_sh_y - mid_ankle_y)
    
    # Ratio altura/ancho del cuerpo
    df['body_aspect_ratio'] = df['body_height'] / (df['body_width'] + 1e-9)
    
    # Distancia entre muñecas (útil para detectar brazos extendidos)
    df['dist_wrists'] = np.sqrt(
        (df[f'x_{L_WRIST}'] - df[f'x_{R_WRIST}'])**2 +
        (df[f'y_{L_WRIST}'] - df[f'y_{R_WRIST}'])**2
    )
    
    # Distancia entre tobillos (útil para detectar postura de pie)
    df['dist_ankles'] = np.sqrt(
        (df[f'x_{L_ANKLE}'] - df[f'x_{R_ANKLE}'])**2 +
        (df[f'y_{L_ANKLE}'] - df[f'y_{R_ANKLE}'])**2
    )
    
    return df


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratios útiles para clasificación.
    """
    df = df.copy()
    
    # Ratio de altura de cadera vs altura total (útil para detectar sentado)
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    mid_ankle_y = (df[f'y_{L_ANKLE}'] + df[f'y_{R_ANKLE}']) / 2
    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
    
    body_height = np.abs(mid_sh_y - mid_ankle_y) + 1e-9
    hip_height = np.abs(mid_hip_y - mid_ankle_y)
    
    df['hip_height_ratio'] = hip_height / body_height
    
    return df


def build_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las características por frame.
    
    Returns:
        DataFrame con características añadidas
    """
    # Ángulos articulares
    df = calculate_joint_angle(df, L_SHOULDER, L_ELBOW, L_WRIST, 'ang_left_elbow')
    df = calculate_joint_angle(df, R_SHOULDER, R_ELBOW, R_WRIST, 'ang_right_elbow')
    df = calculate_joint_angle(df, L_HIP, L_KNEE, L_ANKLE, 'ang_left_knee')
    df = calculate_joint_angle(df, R_HIP, R_KNEE, R_ANKLE, 'ang_right_knee')
    df = calculate_joint_angle(df, L_SHOULDER, L_HIP, L_KNEE, 'ang_left_hip')
    df = calculate_joint_angle(df, R_SHOULDER, R_HIP, R_KNEE, 'ang_right_hip')
    
    # Inclinación del tronco (con y sin signo)
    df = calculate_trunk_inclination_signed(df)
    df = calculate_trunk_inclination(df, 'trunk_incl_deg')
    
    # Velocidades
    df = add_velocities(df, [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP])
    
    # Distancias
    df = calculate_distances(df)
    
    # Ratios
    df = calculate_ratios(df)
    
    return df


def window_aggregate(
    df: pd.DataFrame,
    window_size: int = None,
    stride: int = None,
    keep_cols: tuple = ()
) -> List[Dict]:
    """
    Crea ventanas deslizantes y agrega estadísticas de características.
    
    Args:
        df: DataFrame con características por frame
        window_size: Tamaño de ventana en frames
        stride: Desplazamiento entre ventanas
        keep_cols: Columnas a mantener sin agregar
        
    Returns:
        Lista de diccionarios (una fila por ventana)
    """
    if window_size is None:
        window_size = FEATURE_CONFIG["window_size"]
    if stride is None:
        stride = FEATURE_CONFIG["stride"]
    
    # Identificar columnas de características
    feature_cols = [c for c in df.columns if c.startswith(
        ('ang_', 'trunk_incl_', 'vel_', 'dist_', 'trunk_height', 
         'body_width', 'body_height', 'body_aspect_ratio', 'hip_height_ratio')
    )]
    
    rows = []
    start = 0
    
    while start + window_size <= len(df):
        segment = df.iloc[start:start + window_size]
        
        # Agregar estadísticas
        agg_stats = segment[feature_cols].agg(['mean', 'std', 'min', 'max']).T
        
        row = {
            f"{feat}_{stat}": agg_stats.loc[feat, stat]
            for feat in agg_stats.index
            for stat in ('mean', 'std', 'min', 'max')
        }
        
        # Metadatos
        row['frame_start'] = int(segment['frame_idx'].iloc[0]) if 'frame_idx' in segment else start
        row['frame_end'] = int(segment['frame_idx'].iloc[-1]) if 'frame_idx' in segment else start + window_size - 1
        
        # Columnas adicionales a mantener
        for col in keep_cols:
            if col in segment.columns:
                row[col] = segment[col].iloc[0]
        
        rows.append(row)
        start += stride
    
    return rows

