
import sys
import pytest
import cv2
import numpy as np

sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/modules/lanes')
sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/fusion')

from classic import LaneDetectionConfig, LaneDetectionResult, GeometricLaneDetector

@pytest.fixture
def default_config():
    return LaneDetectionConfig(frame_shape=(720, 1280))

@pytest.fixture
def detector(default_config):
    return GeometricLaneDetector(default_config)

@pytest.fixture
def blank_frame(default_config):
    H, W = default_config.frame_shape
    return np.zeros((H, W, 3), dtype=np.uint8)

@pytest.fixture
def lane_frame(default_config):
    H, W = default_config.frame_shape
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)
    for y in range(400, 700, 60):
        cv2.line(frame, (480, y), (380, y+50), (255, 255, 255), 12)
    for y in range(400, H):
        x = int(800 + (y-400)*0.3)
        if 0 <= x < W:
            cv2.circle(frame, (x, y), 6, (0, 200, 255), -1)
    return frame

@pytest.fixture
def wrong_ndim_frame(default_config):
    H, W = default_config.frame_shape
    return np.zeros((H, W), dtype=np.uint8)

@pytest.fixture
def wrong_dtype_frame(default_config):
    H, W = default_config.frame_shape
    return np.zeros((H, W, 3), dtype=np.float32)
