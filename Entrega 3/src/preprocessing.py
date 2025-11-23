import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional
import math

from src.config import (
    NUM_LANDMARKS, MEDIAPIPE_CONFIG, PREPROCESSING_CONFIG,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE
)

def angle_between_points(a, b, c):
    """
    Calculate the angle at point b formed by segments ab and bc in degrees.
    """
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def compute_joint_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute joint angles such as left/right elbows and knees per frame.
    Adds columns like 'angle_left_elbow', 'angle_right_elbow', 'angle_left_knee', 'angle_right_knee'.
    """
    df = df.copy()
    for side in ['L', 'R']:
        shoulder_idx = L_SHOULDER if side == 'L' else R_SHOULDER
        elbow_idx = L_ELBOW if side == 'L' else R_ELBOW
        wrist_idx = L_WRIST if side == 'L' else R_WRIST
        hip_idx = L_HIP if side == 'L' else R_HIP
        knee_idx = L_KNEE if side == 'L' else R_KNEE
        ankle_idx = L_ANKLE if side == 'L' else R_ANKLE

        angles_elbow = []
        angles_knee = []

        for idx, row in df.iterrows():
            # Extract points for elbow angle
            shoulder = (row[f'x_{shoulder_idx}'], row[f'y_{shoulder_idx}'])
            elbow = (row[f'x_{elbow_idx}'], row[f'y_{elbow_idx}'])
            wrist = (row[f'x_{wrist_idx}'], row[f'y_{wrist_idx}'])
            # Calculate elbow angle
            angle_elbow = angle_between_points(shoulder, elbow, wrist)
            angles_elbow.append(angle_elbow)

            # Extract points for knee angle
            hip = (row[f'x_{hip_idx}'], row[f'y_{hip_idx}'])
            knee = (row[f'x_{knee_idx}'], row[f'y_{knee_idx}'])
            ankle = (row[f'x_{ankle_idx}'], row[f'y_{ankle_idx}'])
            # Calculate knee angle
            angle_knee = angle_between_points(hip, knee, ankle)
            angles_knee.append(angle_knee)

        df[f'angle_{side.lower()}_elbow'] = angles_elbow
        df[f'angle_{side.lower()}_knee'] = angles_knee

    return df

def extract_poses_from_video(
    video_path: str,
    stride: int = 1,
    **mediapipe_kwargs
) -> pd.DataFrame:
    """
    Extract landmarks and extended features from a video using MediaPipe.
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

            # Initialize arrays
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
    df = pd.DataFrame(rows)

    # Apply cleaning and preprocessing pipeline
    df = clean_pose_dataframe(df)

    # Compute extended features (e.g., joint angles)
    df = compute_joint_angles(df)

    return df

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes x,y coords to [0,1] relative to video resolution."""
    df = df.copy()
    w = df['width'].astype(float).replace(0, np.nan)
    h = df['height'].astype(float).replace(0, np.nan)

    for i in range(NUM_LANDMARKS):
        df[f'x_{i}'] = df[f'x_{i}'] / w
        df[f'y_{i}'] = df[f'y_{i}'] / h

    return df

def filter_low_visibility(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Filters out frames with low average visibility."""
    if threshold is None:
        threshold = PREPROCESSING_CONFIG["visibility_threshold"]

    vis_cols = [f'vis_{i}' for i in range(NUM_LANDMARKS)]
    if not all(col in df.columns for col in vis_cols):
        # Fallback: use coordinate presence 
        x_cols = [f'x_{i}' for i in range(NUM_LANDMARKS)]
        mask = ~df[x_cols].isna().all(axis=1)
    else:
        mask = df[vis_cols].mean(axis=1) >= threshold

    return df.loc[mask].reset_index(drop=True)

def smooth_coordinates(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Smooths coordinates using moving average."""
    if window is None:
        window = PREPROCESSING_CONFIG["smooth_window"]

    df = df.copy()
    for i in range(NUM_LANDMARKS):
        df[f'x_{i}'] = df[f'x_{i}'].rolling(window, center=True, min_periods=1).mean()
        df[f'y_{i}'] = df[f'y_{i}'].rolling(window, center=True, min_periods=1).mean()

    return df

def center_by_pelvis(df: pd.DataFrame) -> pd.DataFrame:
    """Centers landmarks by subtracting midpoint of hips for position invariance."""
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
    Full preprocessing pipeline applying visibility filter, normalization, smoothing, and centering.
    """
    if vis_threshold is None:
        vis_threshold = PREPROCESSING_CONFIG["visibility_threshold"]
    if smooth_window is None:
        smooth_window = PREPROCESSING_CONFIG["smooth_window"]
    if use_centering is None:
        use_centering = PREPROCESSING_CONFIG["use_centering"]

    df = filter_low_visibility(df, threshold=vis_threshold)
    
    if len(df) == 0:
        return df

    df = normalize_coordinates(df)
    df = smooth_coordinates(df, window=smooth_window)
    if use_centering:
        df = center_by_pelvis(df)

    return df
