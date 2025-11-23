"""
Módulo de extracción de características mejorado
Usa landmarks suavizados y centrados preprocesados junto con parámetros de config
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from src.config import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    FEATURE_CONFIG, PREPROCESSING_CONFIG
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
    """Calcula el ángulo de una articulación si no existe."""
    if output_name in df.columns:
        # Ya calculado y preprocesado, no recalcular
        return df

    df = df.copy()
    ax, ay = df[f'x_{point_a}'], df[f'y_{point_a}']
    bx, by = df[f'x_{point_b}'], df[f'y_{point_b}']
    cx, cy = df[f'x_{point_c}'], df[f'y_{point_c}']

    df[output_name] = angle_3_points(ax, ay, bx, by, cx, cy)
    return df


def build_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye características por frame usando landmarks preprocesados.
    Usa ángulos ya calculados si existen para evitar redundancia.
    """
    # Asegurar que los ángulos preprocesados estén presentes; si no, calcular
    angle_cols = {
        'ang_left_elbow': (L_SHOULDER, L_ELBOW, L_WRIST),
        'ang_right_elbow': (R_SHOULDER, R_ELBOW, R_WRIST),
        'ang_left_knee': (L_HIP, L_KNEE, L_ANKLE),
        'ang_right_knee': (R_HIP, R_KNEE, R_ANKLE),
        'ang_left_hip': (L_SHOULDER, L_HIP, L_KNEE),
        'ang_right_hip': (R_SHOULDER, R_HIP, R_KNEE),
    }

    for col, points in angle_cols.items():
        df = calculate_joint_angle(df, *points, output_name=col)

    # Inclinación del tronco con valores preprocesados si existen
    if 'trunk_deg' not in df.columns:
        # calcular signo
        mid_sh_x = (df[f'x_{L_SHOULDER}'] + df[f'x_{R_SHOULDER}']) / 2
        mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
        mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
        mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
        vx, vy = mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y
        df['trunk_deg'] = np.degrees(np.arctan2(vx, vy))

    if 'trunk_incl_deg' not in df.columns:
        vx, vy = mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y
        norm = np.sqrt(vx**2 + vy**2) + 1e-9
        cos_angle = np.clip(vy / norm, -1.0, 1.0)
        df['trunk_incl_deg'] = np.degrees(np.arccos(cos_angle))

    # Velocidades utilizando landmarks suavizados y fps consistente
    df = add_velocities(df,
                       [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP],
                       fps_col='fps')

    # Distancias relevantes
    df = calculate_distances(df)

    # Ratios útiles para clasificación
    df = calculate_ratios(df)

    return df


def add_velocities(df: pd.DataFrame, points: List[int], fps_col: str = 'fps') -> pd.DataFrame:
    """
    Calcula velocidades para puntos clave usando landmarks ya suavizados.
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
    Calcula distancias entre puntos clave.
    """
    df = df.copy()

    df['dist_shoulders'] = np.sqrt(
        (df[f'x_{L_SHOULDER}'] - df[f'x_{R_SHOULDER}'])**2 +
        (df[f'y_{L_SHOULDER}'] - df[f'y_{R_SHOULDER}'])**2
    )

    df['dist_hips'] = np.sqrt(
        (df[f'x_{L_HIP}'] - df[f'x_{R_HIP}'])**2 +
        (df[f'y_{L_HIP}'] - df[f'y_{R_HIP}'])**2
    )

    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2
    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    df['trunk_height'] = np.abs(mid_sh_y - mid_hip_y)

    mid_sh_x = (df[f'x_{L_SHOULDER}'] + df[f'x_{R_SHOULDER}']) / 2
    mid_hip_x = (df[f'x_{L_HIP}'] + df[f'x_{R_HIP}']) / 2
    df['body_width'] = np.abs(mid_sh_x - mid_hip_x)

    mid_ankle_y = (df[f'y_{L_ANKLE}'] + df[f'y_{R_ANKLE}']) / 2
    df['body_height'] = np.abs(mid_sh_y - mid_ankle_y)

    df['body_aspect_ratio'] = df['body_height'] / (df['body_width'] + 1e-9)

    df['dist_wrists'] = np.sqrt(
        (df[f'x_{L_WRIST}'] - df[f'x_{R_WRIST}'])**2 +
        (df[f'y_{L_WRIST}'] - df[f'y_{R_WRIST}'])**2
    )

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

    mid_hip_y = (df[f'y_{L_HIP}'] + df[f'y_{R_HIP}']) / 2
    mid_ankle_y = (df[f'y_{L_ANKLE}'] + df[f'y_{R_ANKLE}']) / 2
    mid_sh_y = (df[f'y_{L_SHOULDER}'] + df[f'y_{R_SHOULDER}']) / 2

    body_height = np.abs(mid_sh_y - mid_ankle_y) + 1e-9
    hip_height = np.abs(mid_hip_y - mid_ankle_y)

    df['hip_height_ratio'] = hip_height / body_height

    return df


def window_aggregate(
    df: pd.DataFrame,
    window_size: int = None,
    stride: int = None,
    keep_cols: tuple = ()
) -> List[Dict]:
    """
    Crea ventanas deslizantes y agrega estadísticas de características.
    """
    if window_size is None:
        window_size = FEATURE_CONFIG["window_size"]
    if stride is None:
        stride = FEATURE_CONFIG["stride"]

    feature_cols = [c for c in df.columns if c.startswith((
        'ang_', 'trunk_incl_', 'vel_', 'dist_',
        'trunk_height', 'body_width', 'body_height',
        'body_aspect_ratio', 'hip_height_ratio'
    ))]

    rows = []
    start = 0

    while start + window_size <= len(df):
        segment = df.iloc[start:start + window_size]

        agg_stats = segment[feature_cols].agg(['mean', 'std', 'min', 'max']).T

        row = {
            f"{feat}_{stat}": agg_stats.loc[feat, stat]
            for feat in agg_stats.index
            for stat in ('mean', 'std', 'min', 'max')
        }

        row['frame_start'] = int(segment['frame_idx'].iloc[0]) if 'frame_idx' in segment else start
        row['frame_end'] = int(segment['frame_idx'].iloc[-1]) if 'frame_idx' in segment else start + window_size - 1

        for col in keep_cols:
            if col in segment.columns:
                row[col] = segment[col].iloc[0]

        rows.append(row)
        start += stride

    return rows
