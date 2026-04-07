from __future__ import annotations
import logging, warnings
from numpy.exceptions import RankWarning as _NpRankWarning
from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

GrayImage   = NDArray[np.uint8]
BgrImage    = NDArray[np.uint8]
BinaryImg   = NDArray[np.uint8]
PolyCoeffs  = NDArray[np.float64]
PixelCoords = Tuple[NDArray[np.int32], NDArray[np.int32]]

@dataclass
class LaneDetectionConfig:
    frame_shape: Tuple[int, int] = (720, 1280)
    hls_white_low:   Tuple[int, int, int] = (0,   200,  0)
    hls_white_high:  Tuple[int, int, int] = (180, 255, 55)
    hls_yellow_low:  Tuple[int, int, int] = (15,   30, 115)
    hls_yellow_high: Tuple[int, int, int] = (35,  204, 255)
    canny_low_threshold:  int = 50
    canny_high_threshold: int = 150
    gaussian_kernel_size: int = 5
    ipm_src_points: Optional[NDArray[np.float32]] = field(default=None, repr=False)
    ipm_dst_points: Optional[NDArray[np.float32]] = field(default=None, repr=False)
    n_windows:           int = 9
    window_margin:       int = 100
    min_pixels_recenter: int = 50
    poly_degree: int   = 2
    ym_per_pix:  float = 30.0 / 720.0
    xm_per_pix:  float = 3.7  / 700.0
    confidence_min_pixels: int = 500
    def __repr__(self) -> str:
        return (f"LaneDetectionConfig(frame={self.frame_shape[0]}x{self.frame_shape[1]}, "
                f"windows={self.n_windows}x{self.window_margin*2}px)")

@dataclass
class LaneDetectionResult:
    left_fit:    Optional[PolyCoeffs] = None
    right_fit:   Optional[PolyCoeffs] = None
    curvature_m: Optional[float]      = None
    offset_m:    Optional[float]      = None
    confidence:  float                = 0.0
    valid:       bool                 = False

class GeometricLaneDetector:
    def __init__(self, config: LaneDetectionConfig) -> None:
        if not isinstance(config, LaneDetectionConfig):
            raise TypeError(f"config must be LaneDetectionConfig, got {type(config).__name__!r}.")
        if config.gaussian_kernel_size % 2 == 0:
            raise ValueError(f"gaussian_kernel_size must be odd (got {config.gaussian_kernel_size}).")
        self.config: LaneDetectionConfig = config
        H, W = config.frame_shape
        src = config.ipm_src_points.astype(np.float32) if config.ipm_src_points is not None else \
            np.array([[0.45*W,0.63*H],[0.55*W,0.63*H],[0.85*W,0.95*H],[0.15*W,0.95*H]], dtype=np.float32)
        dst = config.ipm_dst_points.astype(np.float32) if config.ipm_dst_points is not None else \
            np.array([[0.20*W,0.00],[0.80*W,0.00],[0.80*W,float(H)],[0.20*W,float(H)]], dtype=np.float32)
        self._ipm_matrix:     NDArray[np.float64] = cv2.getPerspectiveTransform(src, dst)
        self._ipm_matrix_inv: NDArray[np.float64] = cv2.getPerspectiveTransform(dst, src)
        self._left_fit:    Optional[PolyCoeffs] = None
        self._right_fit:   Optional[PolyCoeffs] = None
        self._left_fit_m:  Optional[PolyCoeffs] = None
        self._right_fit_m: Optional[PolyCoeffs] = None
        self._left_curv_m:  Optional[float] = None
        self._right_curv_m: Optional[float] = None

    def preprocess(self, frame: BgrImage) -> BinaryImg:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"preprocess() expects shape (H, W, 3), got {frame.shape}.")
        if frame.dtype != np.uint8:
            raise ValueError(f"preprocess() expects dtype uint8, got {frame.dtype}.")
        cfg = self.config
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_mask  = cv2.inRange(hls, np.array(cfg.hls_white_low,  dtype=np.uint8),
                                       np.array(cfg.hls_white_high, dtype=np.uint8))
        yellow_mask = cv2.inRange(hls, np.array(cfg.hls_yellow_low, dtype=np.uint8),
                                       np.array(cfg.hls_yellow_high,dtype=np.uint8))
        colour_mask = cv2.bitwise_or(white_mask, yellow_mask)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(cfg.gaussian_kernel_size,cfg.gaussian_kernel_size),0)
        canny_mask = cv2.Canny(blurred, cfg.canny_low_threshold, cfg.canny_high_threshold)
        return (cv2.bitwise_or(colour_mask, canny_mask) > 0).astype(np.uint8)

    def apply_ipm(self, binary_img: BinaryImg) -> BinaryImg:
        if binary_img.ndim != 2 or binary_img.dtype != np.uint8:
            raise ValueError(f"apply_ipm() expects 2-D uint8, got shape={binary_img.shape}.")
        H, W = binary_img.shape
        scaled = (binary_img * 255).astype(np.uint8)
        warped_255 = cv2.warpPerspective(scaled, self._ipm_matrix, (W, H), flags=cv2.INTER_LINEAR)
        return (warped_255 > 127).astype(np.uint8)

    def _lane_histogram(self, warped_img: BinaryImg) -> NDArray[np.int64]:
        H = warped_img.shape[0]
        return warped_img[H // 2:, :].sum(axis=0)

    def sliding_window_search(self, warped_img: BinaryImg) -> Tuple[PixelCoords, PixelCoords]:
        image_height, image_width = warped_img.shape
        histogram      = self._lane_histogram(warped_img)
        midpoint       = histogram.shape[0] // 2
        left_centroid  = int(np.argmax(histogram[:midpoint]))
        right_centroid = int(np.argmax(histogram[midpoint:]) + midpoint)
        window_height  = image_height // self.config.n_windows
        left_y_list, left_x_list, right_y_list, right_x_list = [], [], [], []
        for i in range(self.config.n_windows):
            y_low  = image_height - (i + 1) * window_height
            y_high = image_height - i * window_height
            lx_low  = max(left_centroid  - self.config.window_margin, 0)
            lx_high = min(left_centroid  + self.config.window_margin, image_width)
            left_nz    = warped_img[y_low:y_high, lx_low:lx_high].nonzero()
            left_win_y = left_nz[0] + y_low
            left_win_x = left_nz[1] + lx_low
            left_y_list.append(left_win_y); left_x_list.append(left_win_x)
            if left_win_x.size >= self.config.min_pixels_recenter:
                left_centroid = int(left_win_x.mean())
            rx_low  = max(right_centroid - self.config.window_margin, 0)
            rx_high = min(right_centroid + self.config.window_margin, image_width)
            right_nz    = warped_img[y_low:y_high, rx_low:rx_high].nonzero()
            right_win_y = right_nz[0] + y_low
            right_win_x = right_nz[1] + rx_low
            right_y_list.append(right_win_y); right_x_list.append(right_win_x)
            if right_win_x.size >= self.config.min_pixels_recenter:
                right_centroid = int(right_win_x.mean())
        _empty = np.array([], dtype=np.int32)
        left_y  = np.concatenate(left_y_list).astype(np.int32)  if any(a.size>0 for a in left_y_list)  else _empty.copy()
        left_x  = np.concatenate(left_x_list).astype(np.int32)  if any(a.size>0 for a in left_x_list)  else _empty.copy()
        right_y = np.concatenate(right_y_list).astype(np.int32) if any(a.size>0 for a in right_y_list) else _empty.copy()
        right_x = np.concatenate(right_x_list).astype(np.int32) if any(a.size>0 for a in right_x_list) else _empty.copy()
        return (left_y, left_x), (right_y, right_x)

    def fit_polynomial(self, left_px: PixelCoords, right_px: PixelCoords,
                       image_height: int) -> Tuple[Optional[PolyCoeffs], Optional[PolyCoeffs]]:
        left_y,  left_x  = left_px
        right_y, right_x = right_px
        min_pts = self.config.poly_degree + 1

        def _safe_polyfit(y, x, label):
            if y.size < min_pts:
                return None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", _NpRankWarning)
                    return np.polyfit(y.astype(np.float64), x.astype(np.float64), self.config.poly_degree)
            except _NpRankWarning as e:
                logger.warning("fit_polynomial: RankWarning on %s side — fit=None (%s)", label, e)
                return None

        self._left_fit  = _safe_polyfit(left_y,  left_x,  "left")
        self._right_fit = _safe_polyfit(right_y, right_x, "right")

        self._left_fit_m = self._right_fit_m = None
        if self._left_fit is not None and left_y.size >= min_pts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", _NpRankWarning)
                    self._left_fit_m = np.polyfit(left_y * self.config.ym_per_pix,
                                                  left_x * self.config.xm_per_pix,
                                                  self.config.poly_degree)
            except _NpRankWarning:
                self._left_fit_m = None

        if self._right_fit is not None and right_y.size >= min_pts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", _NpRankWarning)
                    self._right_fit_m = np.polyfit(right_y * self.config.ym_per_pix,
                                                   right_x * self.config.xm_per_pix,
                                                   self.config.poly_degree)
            except _NpRankWarning:
                self._right_fit_m = None

        y_eval_m = (image_height - 1) * self.config.ym_per_pix

        def _curvature(fit_m):
            if fit_m is None: return None
            A, B = float(fit_m[0]), float(fit_m[1])
            denom = abs(2.0 * A)
            if denom < 1e-10: return None
            return (1.0 + (2.0 * A * y_eval_m + B) ** 2) ** 1.5 / denom

        self._left_curv_m  = _curvature(self._left_fit_m)
        self._right_curv_m = _curvature(self._right_fit_m)

        logger.debug("fit_polynomial: left=%d px curv=%.1f m | right=%d px curv=%.1f m",
                     left_x.size,  self._left_curv_m  or float("nan"),
                     right_x.size, self._right_curv_m or float("nan"))
        return self._left_fit, self._right_fit

    def detect(self, frame: BgrImage) -> LaneDetectionResult:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"detect() expects 3-channel uint8, got shape={frame.shape}, dtype={frame.dtype}.")
        binary = self.preprocess(frame)
        warped = self.apply_ipm(binary)
        left_px, right_px   = self.sliding_window_search(warped)
        left_fit, right_fit = self.fit_polynomial(left_px, right_px, warped.shape[0])

        offset_m: Optional[float] = None
        if left_fit is not None and right_fit is not None:
            y_bottom       = warped.shape[0] - 1
            left_base_x    = float(np.polyval(left_fit,  y_bottom))
            right_base_x   = float(np.polyval(right_fit, y_bottom))
            lane_center_x  = (left_base_x + right_base_x) / 2.0
            frame_center_x = frame.shape[1] / 2.0
            offset_m = (frame_center_x - lane_center_x) * self.config.xm_per_pix

        left_count  = left_px[0].size
        right_count = right_px[0].size
        total_px    = left_count + right_count
        confidence  = min(total_px / (2.0 * self.config.confidence_min_pixels), 1.0)
        if left_fit is None or right_fit is None:
            confidence = 0.0

        valid_curvatures = [c for c in (self._left_curv_m, self._right_curv_m) if c is not None]
        curvature_m: Optional[float] = float(np.mean(valid_curvatures)) if valid_curvatures else None

        valid = (left_fit is not None and right_fit is not None
                 and total_px >= self.config.confidence_min_pixels)

        return LaneDetectionResult(left_fit=left_fit, right_fit=right_fit,
                                   curvature_m=curvature_m, offset_m=offset_m,
                                   confidence=confidence, valid=valid)

    def __repr__(self) -> str:
        return f"GeometricLaneDetector(config={self.config!r})"
