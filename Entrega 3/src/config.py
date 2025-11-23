"""
Configuración global del proyecto
"""

import os
from pathlib import Path

# Rutas base
# Desde Entrega 3/src/config.py: parent = src, parent.parent = Entrega 3, parent.parent.parent = raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Entrega 2" / "data"
POSES_DIR = DATA_DIR / "poses"
VIDEOS_DIR = DATA_DIR / "videos"

# MediaPipe landmarks indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NUM_LANDMARKS = 33

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "smooth_landmarks": True
}

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    "visibility_threshold": 0.3,
    "smooth_window": 5,
    "use_centering": True
}

# Feature extraction parameters
FEATURE_CONFIG = {
    "window_size": 32,  # frames por ventana
    "stride": 16,  # frames de desplazamiento
}

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5
}

# Original detailed classes (no consolidation)
CLASS_MAPPING = {
    "walk_back": "walk_back",
    "walk_front": "walk_front",
    "walk_side": "walk_side",
    "walking_away": "walking_away",
    "walking_to_camera": "walking_to_camera",
    "walk_away": "walk_away",
    "stand_front": "stand_front",
    "stand_side": "stand_side",
    "sit_front": "sit_front",
    "sit_side": "sit_side",
}

# Final classes after no consolidation (same as the original keys)
FINAL_CLASSES = list(CLASS_MAPPING.keys())
