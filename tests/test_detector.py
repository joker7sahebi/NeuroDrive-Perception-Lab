"""
test_detector.py — Unit tests for ObjectDetector (detector.py)
==============================================================
NeuroDrive-Perception-Lab | tests/

Test Coverage
-------------
- BoundingBox          : properties, class_name, repr
- DetectionResult      : fields, repr
- ObjectDetectorConfig : defaults, new strategy fields, repr
- SimpleIoUTracker     : IoU computation, update, tracking, reset
- ObjectDetector init  : type guards, strategy validation, lazy caches
- _mock_detect         : shape, determinism, confidence range
- _apply_nms           : duplicate removal, confidence gate, cap
- detect()             : full pipeline, valid/invalid, error handling
- detect_and_draw()    : output shape, dtype, annotation

Run with:
    pytest tests/test_detector.py -v
"""
from __future__ import annotations

import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/modules/objects')
sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/modules/lanes')
sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/fusion')

from detector import (
    BoundingBox,
    DetectionResult,
    ObjectDetectorConfig,
    ObjectDetector,
    SimpleIoUTracker,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def default_config() -> ObjectDetectorConfig:
    """Default ObjectDetectorConfig — mock strategy, 4 objects."""
    return ObjectDetectorConfig(
        detection_strategy="mock",
        mock_n_objects=4,
    )


@pytest.fixture
def detector(default_config) -> ObjectDetector:
    """ObjectDetector with mock strategy."""
    return ObjectDetector(default_config)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """All-black 720×1280 BGR frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def gray_frame() -> np.ndarray:
    """Gray highway-like BGR frame — same content always = same seed."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)
    return frame


@pytest.fixture
def sample_box() -> BoundingBox:
    """A single valid BoundingBox for geometry tests."""
    return BoundingBox(x1=100, y1=200, x2=300, y2=400,
                       confidence=0.85, class_id=0)


@pytest.fixture
def tracker() -> SimpleIoUTracker:
    """Fresh IoU tracker with default parameters."""
    return SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=5)


# ===========================================================================
# TestBoundingBox
# ===========================================================================

class TestBoundingBox:
    """Tests for BoundingBox dataclass and its derived properties."""

    def test_width(self, sample_box):
        """width == x2 - x1."""
        assert sample_box.width == 200.0

    def test_height(self, sample_box):
        """height == y2 - y1."""
        assert sample_box.height == 200.0

    def test_area(self, sample_box):
        """area == width * height."""
        assert sample_box.area == 40_000.0

    def test_center(self, sample_box):
        """center == midpoint of the box."""
        assert sample_box.center == (200.0, 300.0)

    def test_class_name_car(self, sample_box):
        """class_id 0 → 'car'."""
        assert sample_box.class_name == "car"

    @pytest.mark.parametrize("cid,expected", [
        (0, "car"), (1, "truck"), (2, "pedestrian"),
        (3, "cyclist"), (4, "motorcycle"),
    ])
    def test_class_name_all(self, cid, expected):
        """All valid class_ids map to the correct label."""
        box = BoundingBox(0, 0, 10, 10, 0.9, cid)
        assert box.class_name == expected

    def test_unknown_class_id(self):
        """Out-of-range class_id returns 'unknown_<id>'."""
        box = BoundingBox(0, 0, 10, 10, 0.9, 99)
        assert box.class_name == "unknown_99"

    def test_track_id_default_none(self, sample_box):
        """track_id is None until set by tracker."""
        assert sample_box.track_id is None

    def test_repr_contains_class_name(self, sample_box):
        """repr includes the human-readable class label."""
        assert "car" in repr(sample_box)

    def test_zero_area_box(self):
        """Degenerate box (point) has zero width, height, area."""
        box = BoundingBox(50, 50, 50, 50, 0.9, 0)
        assert box.width == 0.0
        assert box.height == 0.0
        assert box.area == 0.0


# ===========================================================================
# TestObjectDetectorConfig
# ===========================================================================

class TestObjectDetectorConfig:
    """Tests for ObjectDetectorConfig dataclass."""

    def test_default_strategy_is_mock(self):
        """Default detection_strategy must be 'mock'."""
        assert ObjectDetectorConfig().detection_strategy == "mock"

    def test_default_confidence_threshold(self):
        """Default confidence_threshold is 0.5."""
        assert ObjectDetectorConfig().confidence_threshold == 0.5

    def test_default_mock_n_objects(self):
        """Default mock_n_objects is 4."""
        assert ObjectDetectorConfig().mock_n_objects == 4

    def test_rtdetr_model_size_default(self):
        """Default rtdetr_model_size is 'rtdetr-l'."""
        assert ObjectDetectorConfig().rtdetr_model_size == "rtdetr-l"

    def test_rfdetr_model_size_default(self):
        """Default rfdetr_model_size is 'rfdetr_base'."""
        assert ObjectDetectorConfig().rfdetr_model_size == "rfdetr_base"

    def test_grounding_dino_model_default(self):
        """Default grounding_dino_model points to the tiny HF variant."""
        assert "grounding-dino" in ObjectDetectorConfig().grounding_dino_model

    def test_grounding_text_prompt_contains_car(self):
        """Default text prompt must include 'car'."""
        assert "car" in ObjectDetectorConfig().grounding_dino_text_prompt

    def test_yolo_road_classes_contains_car(self):
        """yolo_road_classes must include 'car'."""
        assert "car" in ObjectDetectorConfig().yolo_road_classes

    def test_yolo_class_map_car_is_zero(self):
        """yolo_class_map['car'] == 0."""
        assert ObjectDetectorConfig().yolo_class_map["car"] == 0

    def test_yolo_class_map_person_is_two(self):
        """yolo_class_map['person'] == 2."""
        assert ObjectDetectorConfig().yolo_class_map["person"] == 2

    def test_repr_contains_strategy(self):
        """repr must include the active strategy name."""
        cfg = ObjectDetectorConfig(detection_strategy="yolo")
        assert "yolo" in repr(cfg)

    def test_custom_mock_n_objects(self):
        """mock_n_objects can be overridden at construction time."""
        cfg = ObjectDetectorConfig(mock_n_objects=10)
        assert cfg.mock_n_objects == 10


# ===========================================================================
# TestSimpleIoUTracker
# ===========================================================================

class TestSimpleIoUTracker:
    """Tests for SimpleIoUTracker."""

    def test_iou_perfect_overlap(self, tracker):
        """Two identical boxes → IoU == 1.0."""
        box = BoundingBox(0, 0, 100, 100, 0.9, 0)
        assert tracker._compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self, tracker):
        """Non-overlapping boxes → IoU == 0.0."""
        b1 = BoundingBox(0,   0,  50,  50, 0.9, 0)
        b2 = BoundingBox(100, 100, 200, 200, 0.9, 0)
        assert tracker._compute_iou(b1, b2) == 0.0

    def test_iou_partial_overlap(self, tracker):
        """Partially overlapping boxes → IoU in (0, 1)."""
        b1 = BoundingBox(0,  0, 100, 100, 0.9, 0)
        b2 = BoundingBox(50, 50, 150, 150, 0.9, 0)
        iou = tracker._compute_iou(b1, b2)
        assert 0.0 < iou < 1.0

    def test_iou_zero_area_box(self, tracker):
        """Zero-area box → IoU == 0.0 (no division error)."""
        b1 = BoundingBox(50, 50, 50, 50, 0.9, 0)
        b2 = BoundingBox(0,  0, 100, 100, 0.9, 0)
        assert tracker._compute_iou(b1, b2) == 0.0

    def test_update_assigns_track_ids(self, tracker):
        """All boxes returned by update() must have track_id set."""
        boxes = [BoundingBox(i*100, i*100, i*100+80, i*100+80, 0.9, 0)
                 for i in range(3)]
        result = tracker.update(boxes)
        for box in result:
            assert box.track_id is not None

    def test_update_empty_list(self, tracker):
        """update([]) must return [] without error."""
        assert tracker.update([]) == []

    def test_track_id_persistent_across_frames(self, tracker):
        """Same object slightly moved → same track_id."""
        t = SimpleIoUTracker(iou_threshold=0.1, max_lost_frames=5)
        f1 = [BoundingBox(100, 100, 200, 200, 0.9, 0)]
        f2 = [BoundingBox(105, 105, 205, 205, 0.9, 0)]
        tracked_f1 = t.update(f1)
        tracked_f2 = t.update(f2)
        assert tracked_f1[0].track_id == tracked_f2[0].track_id

    def test_new_object_gets_new_id(self, tracker):
        """Object in a completely different location → new track_id."""
        f1 = [BoundingBox(0,   0,  50,  50, 0.9, 0)]
        f2 = [BoundingBox(600, 400, 700, 500, 0.9, 0)]
        tracked_f1 = tracker.update(f1)
        tracked_f2 = tracker.update(f2)
        assert tracked_f1[0].track_id != tracked_f2[0].track_id

    def test_reset_clears_state(self, tracker):
        """After reset(), tracker behaves as if newly constructed."""
        boxes = [BoundingBox(0, 0, 100, 100, 0.9, 0)]
        tracked = tracker.update(boxes)
        first_id = tracked[0].track_id
        tracker.reset()
        boxes2 = [BoundingBox(0, 0, 100, 100, 0.9, 0)]
        tracked2 = tracker.update(boxes2)
        assert tracked2[0].track_id == first_id  # counter reset to 0


# ===========================================================================
# TestObjectDetectorInit
# ===========================================================================

class TestObjectDetectorInit:
    """Tests for ObjectDetector.__init__."""

    def test_init_mock_success(self, default_config):
        """Constructing with mock strategy must succeed."""
        det = ObjectDetector(default_config)
        assert det is not None

    def test_config_stored_as_config(self, default_config):
        """Attribute must be self.config — never self._cfg."""
        det = ObjectDetector(default_config)
        assert det.config is default_config
        assert not hasattr(det, '_cfg')

    def test_wrong_type_raises_TypeError(self):
        """Non-config argument must raise TypeError."""
        with pytest.raises(TypeError, match="ObjectDetectorConfig"):
            ObjectDetector("not_a_config")  # type: ignore

    def test_invalid_strategy_raises_ValueError(self):
        """Unknown strategy string must raise ValueError."""
        with pytest.raises(ValueError, match="detection_strategy"):
            ObjectDetector(ObjectDetectorConfig(
                detection_strategy="invalid_strategy"
            ))

    def test_all_valid_strategies_accepted(self):
        """All 7 documented strategies must be accepted without error."""
        for strategy in ("mock","yolo","onnx","ensemble",
                         "rtdetr","grounding","rfdetr"):
            det = ObjectDetector(ObjectDetectorConfig(
                detection_strategy=strategy
            ))
            assert det is not None

    def test_model_caches_all_none_initially(self, detector):
        """All lazy model handles must be None before first inference."""
        assert detector._yolo_model        is None
        assert detector._rtdetr_model      is None
        assert detector._grounding_model   is None
        assert detector._rfdetr_model      is None

    def test_repr_contains_strategy(self, detector):
        """repr must expose the active strategy."""
        assert "mock" in repr(detector)

    def test_repr_no_cfg(self, detector):
        """repr must NOT reference self._cfg."""
        assert "_cfg" not in repr(detector)


# ===========================================================================
# TestMockDetect
# ===========================================================================

class TestMockDetect:
    """Tests for ObjectDetector._mock_detect()."""

    def test_returns_list(self, detector, gray_frame):
        """_mock_detect must return a list."""
        assert isinstance(detector._mock_detect(gray_frame), list)

    def test_returns_bounding_boxes(self, detector, gray_frame):
        """Every element must be a BoundingBox."""
        boxes = detector._mock_detect(gray_frame)
        for b in boxes:
            assert isinstance(b, BoundingBox)

    def test_deterministic_same_frame(self, detector, gray_frame):
        """Same frame → identical boxes on repeated calls."""
        b1 = detector._mock_detect(gray_frame)
        b2 = detector._mock_detect(gray_frame)
        assert len(b1) == len(b2)
        for a, b in zip(b1, b2):
            assert a.x1 == b.x1
            assert a.confidence == b.confidence

    def test_different_frames_different_boxes(self, detector,
                                               gray_frame, blank_frame):
        """Different frame content → different box positions."""
        b1 = detector._mock_detect(gray_frame)
        b2 = detector._mock_detect(blank_frame)
        if len(b1) > 0 and len(b2) > 0:
            assert b1[0].x1 != b2[0].x1

    def test_confidence_above_threshold(self, default_config, gray_frame):
        """All mock boxes must have confidence >= config.confidence_threshold."""
        det = ObjectDetector(default_config)
        boxes = det._mock_detect(gray_frame)
        for b in boxes:
            assert b.confidence >= default_config.confidence_threshold

    def test_boxes_within_frame(self, detector, gray_frame):
        """All box coordinates must lie within frame boundaries."""
        H, W = gray_frame.shape[:2]
        boxes = detector._mock_detect(gray_frame)
        for b in boxes:
            assert 0.0 <= b.x1 < W
            assert 0.0 <= b.y1 < H
            assert b.x2 <= W
            assert b.y2 <= H

    def test_no_degenerate_boxes(self, detector, gray_frame):
        """No box with zero or negative width/height should be returned."""
        boxes = detector._mock_detect(gray_frame)
        for b in boxes:
            assert b.width  > 0
            assert b.height > 0


# ===========================================================================
# TestApplyNMS
# ===========================================================================

class TestApplyNMS:
    """Tests for ObjectDetector._apply_nms()."""

    def test_removes_duplicate(self, detector):
        """Two boxes with high IoU → only the higher-confidence one kept."""
        b1 = BoundingBox(100, 100, 300, 300, 0.9, 0)
        b2 = BoundingBox(105, 105, 295, 295, 0.7, 0)
        result = detector._apply_nms([b1, b2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_keeps_non_overlapping(self, detector):
        """Two non-overlapping boxes → both kept."""
        b1 = BoundingBox(0,   0,  100, 100, 0.9, 0)
        b2 = BoundingBox(500, 400, 700, 600, 0.8, 0)
        result = detector._apply_nms([b1, b2])
        assert len(result) == 2

    def test_below_confidence_filtered(self, detector):
        """Box below config.confidence_threshold must be discarded."""
        low_conf = BoundingBox(0, 0, 100, 100,
                               detector.config.confidence_threshold - 0.1, 0)
        result = detector._apply_nms([low_conf])
        assert len(result) == 0

    def test_empty_input(self, detector):
        """Empty input → empty output."""
        assert detector._apply_nms([]) == []

    def test_caps_at_max_detections(self):
        """Output is capped at config.max_detections."""
        cfg = ObjectDetectorConfig(
            detection_strategy="mock",
            max_detections=2,
            confidence_threshold=0.1,
        )
        det = ObjectDetector(cfg)
        boxes = [BoundingBox(i*200, 0, i*200+100, 100, 0.9, 0)
                 for i in range(10)]
        result = det._apply_nms(boxes)
        assert len(result) <= cfg.max_detections

    def test_sorted_by_confidence(self, detector):
        """Output boxes must be sorted descending by confidence."""
        boxes = [
            BoundingBox(i*200, 0, i*200+100, 100,
                        float(0.5 + i * 0.1), 0)
            for i in range(4)
        ]
        result = detector._apply_nms(boxes)
        confs = [b.confidence for b in result]
        assert confs == sorted(confs, reverse=True)


# ===========================================================================
# TestDetect
# ===========================================================================

class TestDetect:
    """Tests for ObjectDetector.detect() — full pipeline."""

    def test_returns_DetectionResult(self, detector, gray_frame):
        """detect() must return a DetectionResult."""
        result = detector.detect(gray_frame)
        assert isinstance(result, DetectionResult)

    def test_valid_on_gray_frame(self, detector, gray_frame):
        """Mock detector always produces boxes → valid=True."""
        result = detector.detect(gray_frame)
        assert result.valid is True

    def test_n_objects_matches_boxes(self, detector, gray_frame):
        """n_objects == len(boxes)."""
        result = detector.detect(gray_frame)
        assert result.n_objects == len(result.boxes)

    def test_confidence_in_range(self, detector, gray_frame):
        """Mean confidence must be in [0, 1]."""
        result = detector.detect(gray_frame)
        assert 0.0 <= result.confidence <= 1.0

    def test_frame_shape_stored(self, detector, gray_frame):
        """DetectionResult.frame_shape must match input frame."""
        result = detector.detect(gray_frame)
        assert result.frame_shape == gray_frame.shape[:2]

    def test_all_boxes_have_track_id(self, detector, gray_frame):
        """Every box in result must have a track_id assigned."""
        result = detector.detect(gray_frame)
        for box in result.boxes:
            assert box.track_id is not None

    def test_wrong_ndim_raises_ValueError(self, detector):
        """2-D array must raise ValueError."""
        bad = np.zeros((720, 1280), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect(bad)

    def test_wrong_dtype_raises_ValueError(self, detector):
        """float32 frame must raise ValueError."""
        bad = np.zeros((720, 1280, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            detector.detect(bad)

    def test_wrong_channels_raises_ValueError(self, detector):
        """Single-channel frame must raise ValueError."""
        bad = np.zeros((720, 1280, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect(bad)

    def test_onnx_strategy_raises_NotImplementedError(self):
        """onnx strategy must raise NotImplementedError on detect()."""
        det = ObjectDetector(ObjectDetectorConfig(
            detection_strategy="onnx"
        ))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            det.detect(frame)

    def test_result_has_all_fields(self, detector, gray_frame):
        """DetectionResult must expose all documented fields."""
        result = detector.detect(gray_frame)
        for field in ("boxes","frame_shape","confidence","valid","n_objects"):
            assert hasattr(result, field)

    @pytest.mark.parametrize("n_obj", [1, 3, 6])
    def test_n_objects_respects_config(self, n_obj, gray_frame):
        """mock_n_objects in config controls how many boxes are generated."""
        cfg = ObjectDetectorConfig(
            detection_strategy="mock",
            mock_n_objects=n_obj,
        )
        det    = ObjectDetector(cfg)
        result = det.detect(gray_frame)
        assert result.n_objects == n_obj


# ===========================================================================
# TestDetectAndDraw
# ===========================================================================

class TestDetectAndDraw:
    """Tests for ObjectDetector.detect_and_draw()."""

    def test_output_shape(self, detector, gray_frame):
        """Output must have the same shape as input."""
        canvas = detector.detect_and_draw(gray_frame)
        assert canvas.shape == gray_frame.shape

    def test_output_dtype(self, detector, gray_frame):
        """Output must be uint8."""
        canvas = detector.detect_and_draw(gray_frame)
        assert canvas.dtype == np.uint8

    def test_frame_not_modified(self, detector, gray_frame):
        """Original frame must not be mutated."""
        original = gray_frame.copy()
        detector.detect_and_draw(gray_frame)
        np.testing.assert_array_equal(gray_frame, original)

    def test_annotations_drawn(self, detector, gray_frame):
        """Canvas must differ from input (annotations were drawn)."""
        canvas = detector.detect_and_draw(gray_frame)
        assert not np.array_equal(canvas, gray_frame)

    def test_wrong_input_raises_ValueError(self, detector):
        """2-D input must raise ValueError."""
        bad = np.zeros((720, 1280), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect_and_draw(bad)
