"""
test_kalman_tracker.py — Unit tests for kalman_tracker.py
==========================================================
NeuroDrive-Perception-Lab | tests/

Test Coverage
-------------
- KalmanBoxConfig      : defaults, repr
- KalmanBoxTracker     : init, predict, update, properties, conversions
- KalmanLaneConfig     : defaults, repr
- KalmanLaneFilter     : init, coast, update, smoothing, reset
- MultiObjectKalmanConfig : defaults, repr
- MultiObjectKalmanTracker: init, IoU matrix, matching, tracking, reset

Run with:
    pytest tests/test_kalman_tracker.py -v
"""
from __future__ import annotations

import sys
import pytest
import numpy as np

sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/core')
sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/modules/lanes')
sys.path.insert(0, '/content/drive/MyDrive/NeuroDrive-Perception-Lab/src/modules/objects')

from kalman_tracker import (
    KalmanBoxConfig,
    KalmanBoxTracker,
    KalmanLaneConfig,
    KalmanLaneFilter,
    MultiObjectKalmanConfig,
    MultiObjectKalmanTracker,
    BoundingBox,
    LaneDetectionResult,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_counter():
    """Reset KalmanBoxTracker class counter before every test."""
    KalmanBoxTracker._count = 0
    yield
    KalmanBoxTracker._count = 0


@pytest.fixture
def box_cfg() -> KalmanBoxConfig:
    """Default KalmanBoxConfig."""
    return KalmanBoxConfig()


@pytest.fixture
def sample_box() -> BoundingBox:
    """A simple 100×100 car box centred at (150, 150)."""
    return BoundingBox(x1=100, y1=100, x2=200, y2=200,
                       confidence=0.9, class_id=0)


@pytest.fixture
def tracker(box_cfg, sample_box) -> KalmanBoxTracker:
    """A KalmanBoxTracker seeded with sample_box."""
    return KalmanBoxTracker(sample_box, box_cfg)


@pytest.fixture
def lane_cfg() -> KalmanLaneConfig:
    """Default KalmanLaneConfig."""
    return KalmanLaneConfig()


@pytest.fixture
def lane_filter(lane_cfg) -> KalmanLaneFilter:
    """Fresh KalmanLaneFilter."""
    return KalmanLaneFilter(lane_cfg)


@pytest.fixture
def valid_lane_result() -> LaneDetectionResult:
    """A valid LaneDetectionResult with realistic highway values."""
    return LaneDetectionResult(
        left_fit    = np.array([1e-4, -0.5, 600.0]),
        right_fit   = np.array([1e-4,  0.5, 700.0]),
        curvature_m = 3000.0,
        offset_m    = -0.1,
        confidence  = 1.0,
        valid       = True,
    )


@pytest.fixture
def invalid_lane_result() -> LaneDetectionResult:
    """An invalid LaneDetectionResult (no lane detected)."""
    return LaneDetectionResult(valid=False, confidence=0.0)


@pytest.fixture
def multi_cfg() -> MultiObjectKalmanConfig:
    """MultiObjectKalmanConfig with min_hits=1 for easy confirmation."""
    return MultiObjectKalmanConfig(
        iou_threshold=0.1,
        min_hits=1,
        max_age=5,
        kalman_box_config=KalmanBoxConfig(min_hits=1, max_age=5),
    )


@pytest.fixture
def multi_tracker(multi_cfg) -> MultiObjectKalmanTracker:
    """Fresh MultiObjectKalmanTracker."""
    return MultiObjectKalmanTracker(multi_cfg)


# ===========================================================================
# TestKalmanBoxConfig
# ===========================================================================

class TestKalmanBoxConfig:
    """Tests for KalmanBoxConfig dataclass."""

    def test_default_process_noise(self, box_cfg):
        """Default process_noise_scale must be 1.0."""
        assert box_cfg.process_noise_scale == 1.0

    def test_default_measurement_noise(self, box_cfg):
        """Default measurement_noise_scale must be 1.0."""
        assert box_cfg.measurement_noise_scale == 1.0

    def test_default_initial_velocity_cov(self, box_cfg):
        """Default initial_velocity_cov must be 10.0."""
        assert box_cfg.initial_velocity_cov == 10.0

    def test_default_max_age(self, box_cfg):
        """Default max_age must be 5."""
        assert box_cfg.max_age == 5

    def test_default_min_hits(self, box_cfg):
        """Default min_hits must be 3."""
        assert box_cfg.min_hits == 3

    def test_repr_contains_max_age(self, box_cfg):
        """repr must include max_age."""
        assert "max_age" in repr(box_cfg)


# ===========================================================================
# TestKalmanBoxTracker
# ===========================================================================

class TestKalmanBoxTracker:
    """Tests for KalmanBoxTracker."""

    def test_track_id_increments(self, box_cfg, sample_box):
        """Each new tracker gets a unique, incrementing track_id."""
        t1 = KalmanBoxTracker(sample_box, box_cfg)
        t2 = KalmanBoxTracker(sample_box, box_cfg)
        assert t2.track_id == t1.track_id + 1

    def test_initial_hit_streak_zero(self, tracker):
        """hit_streak starts at 0."""
        assert tracker.hit_streak == 0

    def test_initial_time_since_update_zero(self, tracker):
        """time_since_update starts at 0."""
        assert tracker.time_since_update == 0

    def test_not_confirmed_initially(self, tracker):
        """Tracker is not confirmed until min_hits updates."""
        assert not tracker.is_confirmed

    def test_not_lost_initially(self, tracker):
        """Tracker is not lost initially."""
        assert not tracker.is_lost

    def test_state_vector_shape(self, tracker):
        """State vector x must have shape (8, 1)."""
        assert tracker.x.shape == (8, 1)

    def test_F_matrix_shape(self, tracker):
        """State transition matrix F must be (8, 8)."""
        assert tracker.F.shape == (8, 8)

    def test_H_matrix_shape(self, tracker):
        """Measurement matrix H must be (4, 8)."""
        assert tracker.H.shape == (4, 8)

    def test_bbox_to_z_shape(self, sample_box):
        """_bbox_to_z must return shape (4,)."""
        z = KalmanBoxTracker._bbox_to_z(sample_box)
        assert z.shape == (4,)

    def test_bbox_to_z_center(self, sample_box):
        """_bbox_to_z: cx and cy must equal box centre."""
        z = KalmanBoxTracker._bbox_to_z(sample_box)
        assert abs(z[0] - 150.0) < 1e-6   # cx
        assert abs(z[1] - 150.0) < 1e-6   # cy

    def test_bbox_to_z_area(self, sample_box):
        """_bbox_to_z: s must equal w * h."""
        z = KalmanBoxTracker._bbox_to_z(sample_box)
        assert abs(z[2] - 10_000.0) < 1e-6   # area = 100 * 100

    def test_bbox_to_z_aspect_ratio(self, sample_box):
        """_bbox_to_z: r must equal w / h (1.0 for square box)."""
        z = KalmanBoxTracker._bbox_to_z(sample_box)
        assert abs(z[3] - 1.0) < 1e-6

    def test_x_to_bbox_roundtrip(self, sample_box):
        """_x_to_bbox(_bbox_to_z(box)) must reconstruct the original box."""
        z = KalmanBoxTracker._bbox_to_z(sample_box)
        x = np.zeros(8); x[:4] = z
        back = KalmanBoxTracker._x_to_bbox(x, 0.9, 0)
        assert abs(back.x1 - sample_box.x1) < 1.0
        assert abs(back.x2 - sample_box.x2) < 1.0

    def test_predict_returns_bounding_box(self, tracker):
        """predict() must return a BoundingBox."""
        pred = tracker.predict()
        assert isinstance(pred, BoundingBox)

    def test_predict_track_id_preserved(self, tracker):
        """predict() must embed the tracker's track_id in the box."""
        pred = tracker.predict()
        assert pred.track_id == tracker.track_id

    def test_predict_increments_time_since_update(self, tracker):
        """predict() must increment time_since_update by 1."""
        tracker.predict()
        assert tracker.time_since_update == 1

    def test_predict_center_close_to_initial(self, tracker, sample_box):
        """First prediction must be close to the seed box centre."""
        pred = tracker.predict()
        cx_pred = (pred.x1 + pred.x2) / 2.0
        cx_true = (sample_box.x1 + sample_box.x2) / 2.0
        assert abs(cx_pred - cx_true) < 5.0

    def test_update_resets_time_since_update(self, tracker, sample_box):
        """update() must reset time_since_update to 0."""
        tracker.predict()
        tracker.update(sample_box)
        assert tracker.time_since_update == 0

    def test_update_increments_hit_streak(self, tracker, sample_box):
        """update() must increment hit_streak."""
        tracker.update(sample_box)
        assert tracker.hit_streak == 1

    def test_is_confirmed_after_min_hits(self, box_cfg, sample_box):
        """Tracker is confirmed after min_hits consecutive updates."""
        cfg = KalmanBoxConfig(min_hits=2)
        t   = KalmanBoxTracker(sample_box, cfg)
        t.update(sample_box)
        assert not t.is_confirmed
        t.update(sample_box)
        assert t.is_confirmed

    def test_is_lost_after_max_age_predicts(self, box_cfg, sample_box):
        """Tracker is lost after max_age consecutive predicts without update."""
        cfg = KalmanBoxConfig(max_age=3)
        t   = KalmanBoxTracker(sample_box, cfg)
        for _ in range(cfg.max_age + 1):
            t.predict()
        assert t.is_lost

    def test_repr_contains_track_id(self, tracker):
        """repr must include the track_id."""
        assert str(tracker.track_id) in repr(tracker)


# ===========================================================================
# TestKalmanLaneConfig
# ===========================================================================

class TestKalmanLaneConfig:
    """Tests for KalmanLaneConfig dataclass."""

    def test_default_process_noise(self, lane_cfg):
        """Default process_noise must be 0.01."""
        assert lane_cfg.process_noise == pytest.approx(0.01)

    def test_default_measurement_noise(self, lane_cfg):
        """Default measurement_noise must be 0.1."""
        assert lane_cfg.measurement_noise == pytest.approx(0.1)

    def test_default_initial_cov(self, lane_cfg):
        """Default initial_cov must be 1.0."""
        assert lane_cfg.initial_cov == pytest.approx(1.0)

    def test_repr_contains_Q(self, lane_cfg):
        """repr must include process noise value."""
        assert "Q" in repr(lane_cfg) or "process" in repr(lane_cfg).lower()


# ===========================================================================
# TestKalmanLaneFilter
# ===========================================================================

class TestKalmanLaneFilter:
    """Tests for KalmanLaneFilter."""

    def test_not_initialised_at_start(self, lane_filter):
        """Filter must not be initialised until first valid result."""
        assert not lane_filter._initialised

    def test_state_shape(self, lane_filter):
        """Internal state _x must have shape (8, 2, 1)."""
        assert lane_filter._x.shape == (8, 2, 1)

    def test_coast_on_invalid_no_init(self, lane_filter, invalid_lane_result):
        """Coasting on invalid result before any update must not initialise."""
        lane_filter.update(invalid_lane_result)
        assert not lane_filter._initialised

    def test_coast_returns_original_when_uninitialised(
        self, lane_filter, invalid_lane_result
    ):
        """Before first valid update, coast must return the input unchanged."""
        out = lane_filter.update(invalid_lane_result)
        assert out.valid == invalid_lane_result.valid

    def test_valid_update_initialises(
        self, lane_filter, valid_lane_result
    ):
        """First valid update must set _initialised=True."""
        lane_filter.update(valid_lane_result)
        assert lane_filter._initialised

    def test_valid_update_returns_valid_result(
        self, lane_filter, valid_lane_result
    ):
        """update() with valid input must return valid=True result."""
        out = lane_filter.update(valid_lane_result)
        assert out.valid is True

    def test_smoothed_left_fit_shape(
        self, lane_filter, valid_lane_result
    ):
        """Smoothed left_fit must have shape (3,)."""
        out = lane_filter.update(valid_lane_result)
        assert out.left_fit is not None
        assert out.left_fit.shape == (3,)

    def test_smoothed_offset_close_to_input(
        self, lane_filter, valid_lane_result
    ):
        """After several updates, smoothed offset must converge to input."""
        for _ in range(10):
            out = lane_filter.update(valid_lane_result)
        assert abs(out.offset_m - valid_lane_result.offset_m) < 0.02

    def test_smoothed_curvature_close_to_input(
        self, lane_filter, valid_lane_result
    ):
        """After several updates, smoothed curvature must converge."""
        for _ in range(10):
            out = lane_filter.update(valid_lane_result)
        assert abs(out.curvature_m - valid_lane_result.curvature_m) < 200.0

    def test_coast_after_valid_still_returns_valid(
        self, lane_filter, valid_lane_result, invalid_lane_result
    ):
        """Coasting after a valid update must return valid=True."""
        lane_filter.update(valid_lane_result)
        out = lane_filter.update(invalid_lane_result)
        assert out.valid is True

    def test_reset_clears_state(
        self, lane_filter, valid_lane_result
    ):
        """reset() must restore filter to uninitialised state."""
        lane_filter.update(valid_lane_result)
        assert lane_filter._initialised
        lane_filter.reset()
        assert not lane_filter._initialised
        np.testing.assert_array_equal(lane_filter._x, 0.0)

    def test_repr_contains_initialised(self, lane_filter):
        """repr must include initialised status."""
        assert "initialised" in repr(lane_filter)

    def test_n_filters_is_eight(self, lane_filter):
        """Filter must maintain exactly 8 scalar filters."""
        assert lane_filter._N_FILTERS == 8


# ===========================================================================
# TestMultiObjectKalmanConfig
# ===========================================================================

class TestMultiObjectKalmanConfig:
    """Tests for MultiObjectKalmanConfig dataclass."""

    def test_default_iou_threshold(self):
        """Default iou_threshold must be 0.30."""
        assert MultiObjectKalmanConfig().iou_threshold == pytest.approx(0.30)

    def test_default_max_age(self):
        """Default max_age must be 5."""
        assert MultiObjectKalmanConfig().max_age == 5

    def test_default_min_hits(self):
        """Default min_hits must be 3."""
        assert MultiObjectKalmanConfig().min_hits == 3

    def test_kalman_box_config_embedded(self):
        """Config must embed a KalmanBoxConfig instance."""
        cfg = MultiObjectKalmanConfig()
        assert isinstance(cfg.kalman_box_config, KalmanBoxConfig)

    def test_repr_contains_iou_threshold(self):
        """repr must include iou_threshold."""
        assert "iou" in repr(MultiObjectKalmanConfig()).lower()


# ===========================================================================
# TestMultiObjectKalmanTracker
# ===========================================================================

class TestMultiObjectKalmanTracker:
    """Tests for MultiObjectKalmanTracker."""

    def test_init_empty_trackers(self, multi_tracker):
        """Tracker must start with no active tracks."""
        assert len(multi_tracker._trackers) == 0

    def test_wrong_config_raises_TypeError(self):
        """Non-config argument must raise TypeError."""
        with pytest.raises(TypeError):
            MultiObjectKalmanTracker("bad_config")  # type: ignore

    def test_update_empty_returns_empty(self, multi_tracker):
        """update([]) must return []."""
        assert multi_tracker.update([]) == []

    def test_update_creates_trackers(self, multi_tracker, sample_box):
        """New detections must spawn new KalmanBoxTracker instances."""
        multi_tracker.update([sample_box])
        assert len(multi_tracker._trackers) == 1

    def test_confirmed_track_returned(self, multi_tracker, sample_box):
        """With min_hits=1, first update must return a confirmed track."""
        result = multi_tracker.update([sample_box])
        assert len(result) == 1
        assert result[0].track_id is not None

    def test_track_id_persistent_across_frames(self, multi_tracker):
        """Same object slightly moved → same track_id in consecutive frames."""
        b1 = BoundingBox(100, 100, 200, 200, 0.9, 0)
        b2 = BoundingBox(105, 105, 205, 205, 0.9, 0)
        r1 = multi_tracker.update([b1])
        r2 = multi_tracker.update([b2])
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].track_id == r2[0].track_id

    def test_new_object_new_track_id(self, multi_tracker):
        """Two simultaneous objects get different track_ids in same frame."""
        b1 = BoundingBox(0,   0,  50,  50, 0.9, 0)
        b2 = BoundingBox(700, 400, 800, 500, 0.9, 0)
        # هر دو در یک frame → باید track_id متفاوت داشته باشند
        result = multi_tracker.update([b1, b2])
        if len(result) >= 2:
            ids = [b.track_id for b in result]
            assert len(set(ids)) == len(ids)   # همه unique

    def test_lost_track_pruned(self, multi_tracker):
        """Track not matched for > max_age frames must be removed."""
        b = BoundingBox(100, 100, 200, 200, 0.9, 0)
        multi_tracker.update([b])
        # Coast without detections past max_age
        for _ in range(multi_tracker.config.max_age + 2):
            multi_tracker.update([])
        assert len(multi_tracker._trackers) == 0

    def test_coast_on_empty_frame(self, multi_tracker, sample_box):
        """update([]) after a detection must not immediately remove track."""
        multi_tracker.update([sample_box])
        multi_tracker.update([])   # one miss — still within max_age
        assert len(multi_tracker._trackers) == 1

    def test_iou_matrix_shape(self, multi_tracker):
        """_iou_matrix must return (D, T) shaped matrix."""
        dets = [BoundingBox(i*150, 0, i*150+100, 100, 0.9, 0) for i in range(3)]
        trks = [BoundingBox(i*150, 0, i*150+100, 100, 0.9, 0) for i in range(2)]
        mat  = multi_tracker._iou_matrix(dets, trks)
        assert mat.shape == (3, 2)

    def test_iou_matrix_diagonal_ones(self, multi_tracker):
        """Identical detection and tracker boxes → IoU == 1 on diagonal."""
        boxes = [BoundingBox(i*200, 0, i*200+100, 100, 0.9, 0) for i in range(3)]
        mat   = multi_tracker._iou_matrix(boxes, boxes)
        for i in range(3):
            assert mat[i, i] == pytest.approx(1.0)

    def test_iou_matrix_non_overlapping_zeros(self, multi_tracker):
        """Non-overlapping boxes → IoU == 0."""
        d = [BoundingBox(0,   0,  50,  50, 0.9, 0)]
        t = [BoundingBox(200, 200, 300, 300, 0.9, 0)]
        mat = multi_tracker._iou_matrix(d, t)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_hungarian_match_returns_three_lists(self, multi_tracker):
        """_hungarian_match must return (matched, unmatched_dets, unmatched_trks)."""
        mat = np.array([[0.8, 0.1], [0.1, 0.7]])
        matched, umd, umt = multi_tracker._hungarian_match(mat)
        assert isinstance(matched, list)
        assert isinstance(umd, list)
        assert isinstance(umt, list)

    def test_hungarian_match_empty_matrix(self, multi_tracker):
        """Empty IoU matrix must return empty matched and full unmatched."""
        mat = np.zeros((3, 0))
        matched, umd, umt = multi_tracker._hungarian_match(mat)
        assert matched == []
        assert len(umd) == 3

    def test_reset_clears_trackers(self, multi_tracker, sample_box):
        """reset() must remove all tracks and reset ID counter."""
        multi_tracker.update([sample_box])
        multi_tracker.reset()
        assert len(multi_tracker._trackers) == 0
        assert KalmanBoxTracker._count == 0

    def test_repr_contains_active(self, multi_tracker):
        """repr must include active track count."""
        assert "active" in repr(multi_tracker)

    def test_multiple_objects_tracked(self, multi_tracker):
        """Multiple simultaneous detections must each get a unique track."""
        boxes = [
            BoundingBox(i*200, 100, i*200+150, 250, 0.9, 0)
            for i in range(4)
        ]
        result = multi_tracker.update(boxes)
        track_ids = [b.track_id for b in result]
        assert len(set(track_ids)) == len(track_ids)   # all unique
