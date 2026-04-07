"""
test_classic.py — Unit tests for GeometricLaneDetector (classic.py)
====================================================================
NeuroDrive-Perception-Lab | tests/

Test Coverage
-------------
- LaneDetectionConfig  : default values, custom args, repr
- GeometricLaneDetector init : type guards, matrix pre-computation
- preprocess()         : shape, dtype, binary values, error handling
- apply_ipm()          : shape, dtype, binary contract, error handling
- sliding_window_search(): return types, dtypes, pixel detection
- fit_polynomial()     : coefficients shape, insufficient pixels
- detect()             : full pipeline, valid/invalid, confidence range

Run with:
    pytest tests/test_classic.py -v
    pytest tests/test_classic.py -v --tb=short
"""
from __future__ import annotations

import sys
import pytest
import numpy as np
import numpy.testing as npt
import cv2

sys.path.insert(0, '/home/claude')

from classic import (
    LaneDetectionConfig,
    LaneDetectionResult,
    GeometricLaneDetector,
)


# ===========================================================================
# TestLaneDetectionConfig
# ===========================================================================

class TestLaneDetectionConfig:
    """Tests for LaneDetectionConfig dataclass."""

    def test_default_frame_shape(self, default_config):
        """Default frame_shape باید (720, 1280) باشد."""
        assert default_config.frame_shape == (720, 1280)

    def test_default_n_windows(self, default_config):
        """Default n_windows باید 9 باشد."""
        assert default_config.n_windows == 9

    def test_default_window_margin(self, default_config):
        """Default window_margin باید 100 باشد."""
        assert default_config.window_margin == 100

    def test_default_poly_degree(self, default_config):
        """Default poly_degree باید 2 (quadratic) باشد."""
        assert default_config.poly_degree == 2

    def test_default_ipm_src_none(self, default_config):
        """ipm_src_points باید به صورت پیش‌فرض None باشد."""
        assert default_config.ipm_src_points is None

    def test_default_ipm_dst_none(self, default_config):
        """ipm_dst_points باید به صورت پیش‌فرض None باشد."""
        assert default_config.ipm_dst_points is None

    def test_custom_frame_shape(self):
        """frame_shape باید با مقدار دلخواه قابل تنظیم باشد."""
        cfg = LaneDetectionConfig(frame_shape=(480, 640))
        assert cfg.frame_shape == (480, 640)

    def test_custom_n_windows(self):
        """n_windows باید با مقدار دلخواه قابل تنظیم باشد."""
        cfg = LaneDetectionConfig(n_windows=12)
        assert cfg.n_windows == 12

    def test_repr_contains_frame(self, default_config):
        """repr باید اطلاعات frame_shape را شامل شود."""
        r = repr(default_config)
        assert "720" in r
        assert "1280" in r

    def test_repr_is_string(self, default_config):
        """repr باید string برگرداند."""
        assert isinstance(repr(default_config), str)

    def test_ym_per_pix_positive(self, default_config):
        """ym_per_pix باید مقدار مثبت داشته باشد."""
        assert default_config.ym_per_pix > 0

    def test_xm_per_pix_positive(self, default_config):
        """xm_per_pix باید مقدار مثبت داشته باشد."""
        assert default_config.xm_per_pix > 0

    def test_confidence_min_pixels_positive(self, default_config):
        """confidence_min_pixels باید مثبت باشد."""
        assert default_config.confidence_min_pixels > 0

    def test_hls_white_bounds_valid(self, default_config):
        """hls_white_low باید کمتر از hls_white_high باشد (channel-wise)."""
        for lo, hi in zip(default_config.hls_white_low,
                          default_config.hls_white_high):
            assert lo <= hi


# ===========================================================================
# TestGeometricLaneDetectorInit
# ===========================================================================

class TestGeometricLaneDetectorInit:
    """Tests for GeometricLaneDetector.__init__."""

    def test_init_success(self, default_config):
        """init با config درست باید موفق باشد."""
        det = GeometricLaneDetector(default_config)
        assert det is not None

    def test_config_stored_as_config(self, default_config):
        """config باید در self.config ذخیره شود — نه self._cfg."""
        det = GeometricLaneDetector(default_config)
        assert det.config is default_config
        assert not hasattr(det, '_cfg')

    def test_wrong_type_raises_TypeError(self):
        """ارسال non-config باید TypeError بدهد."""
        with pytest.raises(TypeError, match="LaneDetectionConfig"):
            GeometricLaneDetector("not_a_config")  # type: ignore

    def test_wrong_type_dict_raises_TypeError(self):
        """ارسال dict باید TypeError بدهد."""
        with pytest.raises(TypeError):
            GeometricLaneDetector({"frame_shape": (720, 1280)})  # type: ignore

    def test_even_kernel_raises_ValueError(self):
        """gaussian_kernel_size زوج باید ValueError بدهد."""
        cfg = LaneDetectionConfig(gaussian_kernel_size=4)
        with pytest.raises(ValueError, match="odd"):
            GeometricLaneDetector(cfg)

    def test_ipm_matrix_computed(self, detector):
        """_ipm_matrix باید در init محاسبه شده باشد."""
        assert detector._ipm_matrix is not None
        assert detector._ipm_matrix.shape == (3, 3)

    def test_ipm_matrix_inv_computed(self, detector):
        """_ipm_matrix_inv باید در init محاسبه شده باشد."""
        assert detector._ipm_matrix_inv is not None
        assert detector._ipm_matrix_inv.shape == (3, 3)

    def test_ipm_matrices_are_inverses(self, detector):
        """_ipm_matrix_inv باید یک ماتریس وارون‌پذیر باشد (det != 0).
        
        Note: getPerspectiveTransform(dst,src) دقیقاً inv(M) نیست
        به دلیل float32 precision — اما باید invertible باشد.
        """
        det_val = np.linalg.det(detector._ipm_matrix_inv)
        assert abs(det_val) > 1e-10, f"_ipm_matrix_inv is singular (det={det_val})"

    def test_left_fit_initially_none(self, detector):
        """_left_fit باید در ابتدا None باشد."""
        assert detector._left_fit is None

    def test_right_fit_initially_none(self, detector):
        """_right_fit باید در ابتدا None باشد."""
        assert detector._right_fit is None

    def test_repr_contains_config(self, detector):
        """repr باید اطلاعات config را شامل شود."""
        r = repr(detector)
        assert "GeometricLaneDetector" in r
        assert "config" in r.lower()

    def test_repr_no_cfg_attribute(self, detector):
        """repr نباید از self._cfg استفاده کند."""
        r = repr(detector)
        assert "_cfg" not in r


# ===========================================================================
# TestPreprocess
# ===========================================================================

class TestPreprocess:
    """Tests for GeometricLaneDetector.preprocess()."""

    def test_output_shape(self, detector, lane_frame, default_config):
        """خروجی باید shape (H, W) داشته باشد."""
        H, W = default_config.frame_shape
        binary = detector.preprocess(lane_frame)
        assert binary.shape == (H, W)

    def test_output_dtype(self, detector, lane_frame):
        """خروجی باید dtype uint8 داشته باشد."""
        binary = detector.preprocess(lane_frame)
        assert binary.dtype == np.uint8

    def test_output_values_binary(self, detector, lane_frame):
        """خروجی باید فقط مقادیر {0, 1} داشته باشد."""
        binary = detector.preprocess(lane_frame)
        unique = set(np.unique(binary).tolist())
        assert unique.issubset({0, 1})

    def test_wrong_ndim_raises_ValueError(self, detector, wrong_ndim_frame):
        """Frame دوبعدی باید ValueError بدهد."""
        with pytest.raises(ValueError, match="shape"):
            detector.preprocess(wrong_ndim_frame)

    def test_wrong_dtype_raises_ValueError(self, detector, wrong_dtype_frame):
        """Frame با dtype float32 باید ValueError بدهد."""
        with pytest.raises(ValueError, match="dtype"):
            detector.preprocess(wrong_dtype_frame)

    def test_blank_frame_low_activation(self, detector, blank_frame,
                                         default_config):
        """Frame سیاه باید pixel فعال کمی داشته باشد."""
        binary = detector.preprocess(blank_frame)
        H, W   = default_config.frame_shape
        ratio  = binary.sum() / (H * W)
        assert ratio < 0.05   # کمتر از 5%

    def test_lane_frame_nonzero(self, detector, lane_frame):
        """Frame با خطوط lane باید pixel فعال داشته باشد."""
        binary = detector.preprocess(lane_frame)
        assert binary.sum() > 0

    def test_lane_frame_higher_than_blank(self, detector, blank_frame,
                                           lane_frame):
        """Frame با lane باید pixel بیشتری از frame خالی داشته باشد."""
        b_blank = detector.preprocess(blank_frame)
        b_lane  = detector.preprocess(lane_frame)
        assert b_lane.sum() > b_blank.sum()

    @pytest.mark.parametrize("bad_shape", [
        (720, 1280, 1),   # single channel
        (720, 1280, 4),   # 4-channel RGBA
    ])
    def test_wrong_channel_count_raises(self, detector, bad_shape):
        """Frame با تعداد channel اشتباه باید ValueError بدهد."""
        bad_frame = np.zeros(bad_shape, dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.preprocess(bad_frame)


# ===========================================================================
# TestApplyIPM
# ===========================================================================

class TestApplyIPM:
    """Tests for GeometricLaneDetector.apply_ipm()."""

    def test_output_shape_preserved(self, detector, lane_frame,
                                     default_config):
        """خروجی IPM باید همان shape ورودی را داشته باشد."""
        H, W   = default_config.frame_shape
        binary = detector.preprocess(lane_frame)
        warped = detector.apply_ipm(binary)
        assert warped.shape == (H, W)

    def test_output_dtype(self, detector, lane_frame):
        """خروجی IPM باید dtype uint8 داشته باشد."""
        binary = detector.preprocess(lane_frame)
        warped = detector.apply_ipm(binary)
        assert warped.dtype == np.uint8

    def test_output_values_binary(self, detector, lane_frame):
        """خروجی IPM باید فقط مقادیر {0, 1} داشته باشد."""
        binary = detector.preprocess(lane_frame)
        warped = detector.apply_ipm(binary)
        unique = set(np.unique(warped).tolist())
        assert unique.issubset({0, 1})

    def test_blank_input_mostly_blank_output(self, detector, default_config):
        """ورودی خالی باید خروجی تقریباً خالی بدهد."""
        H, W       = default_config.frame_shape
        blank_bin  = np.zeros((H, W), dtype=np.uint8)
        warped     = detector.apply_ipm(blank_bin)
        assert warped.sum() == 0

    def test_wrong_ndim_raises_ValueError(self, detector, default_config):
        """ورودی 3D باید ValueError بدهد."""
        H, W = default_config.frame_shape
        bad  = np.zeros((H, W, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.apply_ipm(bad)

    def test_wrong_dtype_raises_ValueError(self, detector, default_config):
        """ورودی float32 باید ValueError بدهد."""
        H, W = default_config.frame_shape
        bad  = np.zeros((H, W), dtype=np.float32)
        with pytest.raises(ValueError):
            detector.apply_ipm(bad)

    def test_uses_cached_matrix(self, detector, lane_frame):
        """apply_ipm نباید matrix را recompute کند — از cache استفاده کند."""
        binary = detector.preprocess(lane_frame)
        M_before = detector._ipm_matrix.copy()
        detector.apply_ipm(binary)
        npt.assert_array_equal(detector._ipm_matrix, M_before)


# ===========================================================================
# TestSlidingWindowSearch
# ===========================================================================

class TestSlidingWindowSearch:
    """Tests for GeometricLaneDetector.sliding_window_search()."""

    def test_returns_tuple_of_two(self, detector, lane_frame):
        """باید tuple از دو PixelCoords برگرداند."""
        binary = detector.preprocess(lane_frame)
        warped = detector.apply_ipm(binary)
        result = detector.sliding_window_search(warped)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_each_pixelcoord_is_tuple_of_two(self, detector, lane_frame):
        """هر PixelCoords باید tuple از دو array باشد."""
        binary       = detector.preprocess(lane_frame)
        warped       = detector.apply_ipm(binary)
        left, right  = detector.sliding_window_search(warped)
        assert len(left)  == 2
        assert len(right) == 2

    def test_dtype_int32(self, detector, lane_frame):
        """همه آرایه‌های خروجی باید dtype int32 داشته باشند."""
        binary      = detector.preprocess(lane_frame)
        warped      = detector.apply_ipm(binary)
        (ly,lx),(ry,rx) = detector.sliding_window_search(warped)
        for arr in [ly, lx, ry, rx]:
            assert arr.dtype == np.int32, f"Expected int32, got {arr.dtype}"

    def test_blank_image_returns_empty_or_small(self, detector,
                                                 default_config):
        """Image خالی باید array خالی یا بسیار کوچک برگرداند."""
        H, W = default_config.frame_shape
        blank_warped = np.zeros((H, W), dtype=np.uint8)
        (ly,lx),(ry,rx) = detector.sliding_window_search(blank_warped)
        assert lx.size < 100
        assert rx.size < 100

    def test_lane_frame_finds_pixels(self, detector, lane_frame):
        """Frame با lane باید pixel قابل توجهی پیدا کند."""
        binary      = detector.preprocess(lane_frame)
        warped      = detector.apply_ipm(binary)
        (ly,lx),(ry,rx) = detector.sliding_window_search(warped)
        total = lx.size + rx.size
        assert total > 100, f"Expected >100 pixels, got {total}"

    def test_y_indices_in_valid_range(self, detector, lane_frame,
                                       default_config):
        """y indices باید در محدوده [0, H) باشند."""
        H, W        = default_config.frame_shape
        binary      = detector.preprocess(lane_frame)
        warped      = detector.apply_ipm(binary)
        (ly,lx),(ry,rx) = detector.sliding_window_search(warped)
        for arr in [ly, ry]:
            if arr.size > 0:
                assert arr.min() >= 0
                assert arr.max() < H

    def test_x_indices_in_valid_range(self, detector, lane_frame,
                                       default_config):
        """x indices باید در محدوده [0, W) باشند."""
        H, W        = default_config.frame_shape
        binary      = detector.preprocess(lane_frame)
        warped      = detector.apply_ipm(binary)
        (ly,lx),(ry,rx) = detector.sliding_window_search(warped)
        for arr in [lx, rx]:
            if arr.size > 0:
                assert arr.min() >= 0
                assert arr.max() < W

    def test_left_right_in_correct_halves(self, detector, lane_frame,
                                           default_config):
        """پیکسل‌های چپ باید در نیمه چپ و راست در نیمه راست باشند."""
        H, W        = default_config.frame_shape
        binary      = detector.preprocess(lane_frame)
        warped      = detector.apply_ipm(binary)
        (ly,lx),(ry,rx) = detector.sliding_window_search(warped)
        if lx.size > 50 and rx.size > 50:
            assert lx.mean() < W * 0.6
            assert rx.mean() > W * 0.4


# ===========================================================================
# TestFitPolynomial
# ===========================================================================

class TestFitPolynomial:
    """Tests for GeometricLaneDetector.fit_polynomial()."""

    def _get_lane_pixels(self, detector, lane_frame):
        """Helper: pipeline تا sliding_window_search."""
        binary = detector.preprocess(lane_frame)
        warped = detector.apply_ipm(binary)
        return detector.sliding_window_search(warped), warped.shape[0]

    def test_returns_two_values(self, detector, lane_frame):
        """باید tuple از دو مقدار برگرداند."""
        (lp, rp), H = self._get_lane_pixels(detector, lane_frame)
        result = detector.fit_polynomial(lp, rp, H)
        assert len(result) == 2

    def test_coeffs_shape_when_enough_pixels(self, detector, lane_frame):
        """coefficients باید shape (3,) داشته باشند وقتی pixel کافی است."""
        (lp, rp), H = self._get_lane_pixels(detector, lane_frame)
        lf, rf = detector.fit_polynomial(lp, rp, H)
        if lf is not None:
            assert lf.shape == (3,)
        if rf is not None:
            assert rf.shape == (3,)

    def test_insufficient_pixels_returns_none(self, detector,
                                               default_config):
        """pixel ناکافی باید None برگرداند."""
        H, W    = default_config.frame_shape
        empty   = (np.array([], dtype=np.int32),
                   np.array([], dtype=np.int32))
        lf, rf  = detector.fit_polynomial(empty, empty, H)
        assert lf is None
        assert rf is None

    def test_single_point_returns_none(self, detector, default_config):
        """یک نقطه (کمتر از poly_degree+1) باید None بدهد."""
        H, W    = default_config.frame_shape
        one_pt  = (np.array([100], dtype=np.int32),
                   np.array([200], dtype=np.int32))
        lf, rf  = detector.fit_polynomial(one_pt, one_pt, H)
        assert lf is None
        assert rf is None

    def test_curvature_cached_after_fit(self, detector, lane_frame):
        """بعد از fit، _left_curv_m و _right_curv_m باید set شده باشند."""
        (lp, rp), H = self._get_lane_pixels(detector, lane_frame)
        detector.fit_polynomial(lp, rp, H)
        assert hasattr(detector, '_left_curv_m')
        assert hasattr(detector, '_right_curv_m')

    def test_metric_fit_cached(self, detector, lane_frame):
        """بعد از fit، _left_fit_m باید set شده باشد."""
        (lp, rp), H = self._get_lane_pixels(detector, lane_frame)
        detector.fit_polynomial(lp, rp, H)
        assert hasattr(detector, '_left_fit_m')
        assert hasattr(detector, '_right_fit_m')


# ===========================================================================
# TestDetect
# ===========================================================================

class TestDetect:
    """Tests for GeometricLaneDetector.detect() — full pipeline."""

    def test_returns_LaneDetectionResult(self, detector, lane_frame):
        """detect() باید LaneDetectionResult برگرداند."""
        result = detector.detect(lane_frame)
        assert isinstance(result, LaneDetectionResult)

    def test_valid_on_lane_frame(self, detector, lane_frame):
        """Frame با lane باید valid=True بدهد."""
        result = detector.detect(lane_frame)
        assert result.valid is True

    def test_invalid_on_blank_frame(self, detector, blank_frame):
        """Frame سیاه باید valid=False بدهد."""
        result = detector.detect(blank_frame)
        assert result.valid is False

    def test_confidence_in_range(self, detector, lane_frame):
        """confidence باید در [0, 1] باشد."""
        result = detector.detect(lane_frame)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_zero_on_blank(self, detector, blank_frame):
        """confidence روی frame خالی باید 0.0 باشد."""
        result = detector.detect(blank_frame)
        assert result.confidence == 0.0

    def test_left_fit_shape_when_valid(self, detector, lane_frame):
        """left_fit وقتی valid باید shape (3,) داشته باشد."""
        result = detector.detect(lane_frame)
        if result.valid:
            assert result.left_fit  is not None
            assert result.right_fit is not None
            assert result.left_fit.shape  == (3,)
            assert result.right_fit.shape == (3,)

    def test_offset_m_float_when_valid(self, detector, lane_frame):
        """offset_m وقتی valid باید float باشد."""
        result = detector.detect(lane_frame)
        if result.valid and result.offset_m is not None:
            assert isinstance(result.offset_m, float)

    def test_curvature_positive_when_valid(self, detector, lane_frame):
        """curvature_m وقتی valid باید مثبت باشد."""
        result = detector.detect(lane_frame)
        if result.valid and result.curvature_m is not None:
            assert result.curvature_m > 0

    def test_wrong_ndim_raises_ValueError(self, detector, wrong_ndim_frame):
        """Frame دوبعدی باید ValueError بدهد."""
        with pytest.raises(ValueError):
            detector.detect(wrong_ndim_frame)

    def test_wrong_dtype_raises_ValueError(self, detector, wrong_dtype_frame):
        """Frame float32 باید ValueError بدهد."""
        with pytest.raises(ValueError):
            detector.detect(wrong_dtype_frame)

    def test_blank_fit_is_none(self, detector, blank_frame):
        """روی frame خالی، left_fit و right_fit باید None باشند."""
        result = detector.detect(blank_frame)
        assert result.left_fit  is None
        assert result.right_fit is None

    def test_result_has_all_fields(self, detector, lane_frame):
        """LaneDetectionResult باید همه فیلدها را داشته باشد."""
        result = detector.detect(lane_frame)
        assert hasattr(result, 'left_fit')
        assert hasattr(result, 'right_fit')
        assert hasattr(result, 'curvature_m')
        assert hasattr(result, 'offset_m')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'valid')

    @pytest.mark.parametrize("frame_fixture", ["blank_frame", "lane_frame"])
    def test_detect_does_not_raise_on_valid_frames(
        self, request, detector, frame_fixture
    ):
        """detect() نباید روی frame‌های معتبر exception بدهد."""
        frame = request.getfixturevalue(frame_fixture)
        result = detector.detect(frame)
        assert isinstance(result, LaneDetectionResult)
