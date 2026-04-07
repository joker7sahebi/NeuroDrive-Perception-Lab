"""
deep.py — Deep Learning Lane Detection Module
==============================================
NeuroDrive-Perception-Lab | Phase 2: DL Inference Branch
---------------------------------------------------------
Role in the Hybrid Pipeline
----------------------------
This module implements the **deep learning branch** of the NeuroDrive
hybrid lane detection pipeline.  It is the direct counterpart to
``src/modules/lanes/classic.py`` and shares the same output contract
(:class:`~classic.LaneDetectionResult`) so that the fusion layer
(``src/fusion/lane_tracker.py``) can arbitrate between both detectors
transparently — without knowing which branch produced a given estimate.

Architecture: Row-Anchor Regression (UFLD-style)
-------------------------------------------------
Rather than a dense segmentation model (which requires full-image
inference at every frame), this module uses a **row-anchor regression**
approach inspired by Ultra-Fast Lane Detection (UFLD, ECCV 2020):

1. The image is divided into ``n_row_anchors`` equidistant horizontal
   strips between ``anchor_ratio_start`` and ``anchor_ratio_end``.
2. For each strip the model predicts the **x-position** of the left and
   right lane lines (one scalar per side per strip).
3. The resulting (x, y) point cloud is fitted with a 2nd-degree
   polynomial — the same convention used in ``classic.py`` — yielding
   directly comparable polynomial coefficients for fusion.

This design is inference-efficient:

* Output dimensionality = 2 × n_row_anchors scalars (typically 36 values)
  vs. H × W pixels for a segmentation head.
* Achievable at >30 fps on an automotive-grade CPU (e.g. Intel Atom x7).
* No post-processing pipeline (IPM, sliding window) is needed; the
  spatial structure is baked into the anchor design.

Operating Modes
---------------
mock_mode = True (default)
    Generates synthetic anchor predictions using a parametric highway-
    lane geometry model + Gaussian jitter.  No GPU or trained weights
    required.  Use for unit tests, CI pipelines, and Colab demos.

mock_mode = False
    Runs inference through an ONNX Runtime session loaded from
    ``config.onnx_model_path``.  Intended for production deployment once
    a UFLD-compatible model has been exported.

Shared Interface
----------------
``detect()`` returns a :class:`~classic.LaneDetectionResult` with the
identical fields populated by ``GeometricLaneDetector.detect()``.  The
fusion layer calls both detectors and selects (or blends) the result
with higher ``confidence``.

References
----------
- Qin et al., "Ultra Fast Structure-aware Deep Lane Detection", ECCV 2020.
  https://arxiv.org/abs/2004.11757
- ONNX Runtime Python API: https://onnxruntime.ai/docs/api/python/

Author : Portfolio — Senior ADAS CV Engineer
Target : BMW / Bosch / CARIAD — NeuroDrive-Perception-Lab
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Shared types — import from the classical branch
# ---------------------------------------------------------------------------
# In the installed package these resolve via the src/ namespace.
# For Colab / sys.path usage, ensure classic.py is on the path first.
try:
    from classic import (  # type: ignore[import]
        BgrImage,
        LaneDetectionResult,
        PolyCoeffs,
    )
except ModuleNotFoundError:
    # Graceful fallback: redefine locally so the module is importable
    # in isolation (e.g. during unit-test discovery without classic.py).
    BgrImage   = NDArray[np.uint8]    # type: ignore[misc]
    PolyCoeffs = NDArray[np.float64]  # type: ignore[misc]

    from dataclasses import dataclass as _dc

    @_dc
    class LaneDetectionResult:        # type: ignore[no-redef]
        """Minimal fallback — prefer importing from classic.py."""
        left_fit:    Optional[PolyCoeffs] = None
        right_fit:   Optional[PolyCoeffs] = None
        curvature_m: Optional[float]      = None
        offset_m:    Optional[float]      = None
        confidence:  float                = 0.0
        valid:       bool                 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local type aliases
# ---------------------------------------------------------------------------
Float32Array = NDArray[np.float32]   # normalised model input tensor
AnchorArray  = NDArray[np.float64]   # shape (n_row_anchors, 2): [[x, y], ...]


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeepLaneConfig:
    """Hyperparameter and runtime-config container for :class:`DeepLaneDetector`.

    All scalar constants that would otherwise appear as magic numbers in
    algorithm bodies are centralised here.  Switch ``mock_mode`` off and
    supply ``onnx_model_path`` when deploying to a real ECU.

    Parameters
    ----------
    frame_shape : Tuple[int, int]
        Expected input resolution as (H, W).  Used to compute absolute
        anchor y-positions and for lateral-offset calculation.
    n_row_anchors : int
        Number of horizontal strips used for row-anchor regression.
        Higher values yield a denser point cloud for polynomial fitting
        at the cost of a larger model output tensor.
    anchor_ratio_start : float
        Fractional y-position of the top-most anchor (0 = image top).
        Set to 0.40 to discard sky and far-field noise.
    anchor_ratio_end : float
        Fractional y-position of the bottom-most anchor (1 = image
        bottom).
    poly_degree : int
        Degree of the polynomial fitted to anchor points.  Default 2
        (parabola) matches ``classic.py`` for direct coefficient
        comparison in the fusion layer.
    ym_per_pix : float
        Metres per pixel in the vertical direction.  Must match the
        value in ``LaneDetectionConfig`` so curvature estimates are
        comparable across branches.
    xm_per_pix : float
        Metres per pixel in the horizontal direction.
    confidence_threshold : float
        Minimum per-anchor model confidence for an anchor to be
        included in the polynomial fit.  Anchors below this score are
        treated as unreliable and excluded.
    model_input_size : Tuple[int, int]
        (H, W) to which the frame is resized before ONNX inference.
        Must match the input shape the ONNX model was exported with.
    mock_mode : bool
        If ``True``, :meth:`_mock_predict` is used instead of
        :meth:`_onnx_predict`.  Enables full-pipeline testing without a
        trained model or GPU.
    onnx_model_path : Optional[str]
        Filesystem path to the ONNX model file.  Required when
        ``mock_mode=False``; ignored otherwise.
    mock_noise_std_px : float
        Standard deviation (pixels) of Gaussian noise added to
        synthetic anchor predictions in mock mode.  Simulates the
        natural per-anchor variance of a trained network.
    """

    # Input resolution
    frame_shape:          Tuple[int, int] = (720, 1280)

    # Row-anchor geometry
    n_row_anchors:        int             = 18
    anchor_ratio_start:   float           = 0.40
    anchor_ratio_end:     float           = 1.00

    # Polynomial
    poly_degree:          int             = 2
    ym_per_pix:           float           = 30.0 / 720.0
    xm_per_pix:           float           = 3.7  / 700.0

    # Inference gate
    confidence_threshold: float           = 0.5

    # Model I/O
    model_input_size:     Tuple[int, int] = (288, 800)   # H×W
    mock_mode:            bool            = True
    onnx_model_path:      Optional[str]   = None

    # Mock-mode noise
    mock_noise_std_px:    float           = 5.0

    def __repr__(self) -> str:  # noqa: D105
        mode = "mock" if self.mock_mode else f"onnx:{self.onnx_model_path}"
        return (
            f"DeepLaneConfig("
            f"frame={self.frame_shape[0]}×{self.frame_shape[1]}, "
            f"anchors={self.n_row_anchors}"
            f"[{self.anchor_ratio_start:.2f}→{self.anchor_ratio_end:.2f}], "
            f"poly_degree={self.poly_degree}, "
            f"mode={mode})"
        )


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class DeepLaneDetector:
    """DL lane detector using row-anchor regression (UFLD-style).

    Shares the :class:`~classic.LaneDetectionResult` output interface
    with :class:`~classic.GeometricLaneDetector` so both branches are
    consumed identically by the fusion layer.

    Parameters
    ----------
    config : DeepLaneConfig
        All algorithm and runtime hyperparameters.

    Examples
    --------
    >>> cfg = DeepLaneConfig(mock_mode=True, n_row_anchors=18)
    >>> detector = DeepLaneDetector(cfg)
    >>> result = detector.detect(frame)   # frame: H×W×3 BGR uint8
    >>> print(result.curvature_m, result.offset_m)
    """

    def __init__(self, config: DeepLaneConfig) -> None:
        """Initialise the detector and optionally load the ONNX model.

        In ``mock_mode=True`` no model file is loaded; the detector is
        immediately ready.  In ``mock_mode=False`` an ONNX Runtime
        ``InferenceSession`` is created from ``config.onnx_model_path``.

        Parameters
        ----------
        config : DeepLaneConfig
            Runtime and hyperparameter bundle.

        Raises
        ------
        TypeError
            If ``config`` is not a :class:`DeepLaneConfig` instance.
        ValueError
            If ``mock_mode=False`` and ``onnx_model_path`` is ``None``.
        FileNotFoundError
            If ``mock_mode=False`` and the model file does not exist at
            ``config.onnx_model_path``.
        ImportError
            If ``mock_mode=False`` and ``onnxruntime`` is not installed.
        """
        if not isinstance(config, DeepLaneConfig):
            raise TypeError(
                f"config must be a DeepLaneConfig instance, "
                f"got {type(config).__name__!r}."
            )
        if not config.mock_mode and config.onnx_model_path is None:
            raise ValueError(
                "DeepLaneConfig.onnx_model_path must be set when mock_mode=False."
            )

        self.config: DeepLaneConfig = config

        # ── Pre-compute anchor y-positions ───────────────────────────────
        # Fixed for a given config; computing once at init avoids a
        # repeated linspace call on every frame's hot path.
        H: int = config.frame_shape[0]
        self._anchor_ys: NDArray[np.float64] = np.linspace(
            config.anchor_ratio_start * H,
            config.anchor_ratio_end   * H,
            config.n_row_anchors,
            dtype=np.float64,
        )

        # ── ONNX session (production path only) ──────────────────────────
        self._ort_session = None   # populated below if mock_mode=False

        if not config.mock_mode:
            # Lazy import — onnxruntime is not a hard dependency in mock
            # mode, keeping CI and Colab installs lightweight.
            try:
                import onnxruntime as ort  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for mock_mode=False.\n"
                    "Install with: pip install onnxruntime"
                ) from exc

            import os
            if not os.path.isfile(config.onnx_model_path):  # type: ignore[arg-type]
                raise FileNotFoundError(
                    f"ONNX model not found at path: {config.onnx_model_path!r}"
                )

            self._ort_session = ort.InferenceSession(
                config.onnx_model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.debug(
                "DeepLaneDetector: ONNX session loaded — %r",
                config.onnx_model_path,
            )

        logger.debug("DeepLaneDetector initialised. %r", self.config)

    # ------------------------------------------------------------------
    # Stage 1 — Preprocessing
    # ------------------------------------------------------------------

    def preprocess_dl(self, frame: BgrImage) -> Float32Array:
        """Resize and normalise a BGR frame for DL model inference.

        Prepares the frame for the ONNX model by resizing to
        ``config.model_input_size``, converting to RGB channel order,
        and normalising pixel values to ``[0.0, 1.0]``.

        Parameters
        ----------
        frame : BgrImage
            Raw camera frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        Float32Array
            Normalised image, shape (model_H, model_W, 3), dtype
            ``float32``, values in ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If ``frame`` is not a 3-channel uint8 array.

        Notes
        -----
        **Channel order** — BGR → RGB conversion is applied because UFLD
        and most PyTorch/ONNX models are trained on RGB inputs.

        **Normalisation** — pixel values are divided by 255.0 (the full
        uint8 range).  Channel-wise ImageNet mean/std normalisation is
        deliberately omitted here; it belongs in the ONNX model's
        pre-processing node or a separate configurable transform, not
        in this general-purpose method.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"preprocess_dl() expects a 3-channel uint8 BGR image, "
                f"got shape={frame.shape}, dtype={frame.dtype}."
            )

        model_H, model_W = self.config.model_input_size

        # Resize to model resolution — INTER_LINEAR balances speed and
        # quality for the typical 720p → 288p downscale factor.
        resized: NDArray[np.uint8] = cv2.resize(
            frame,
            (model_W, model_H),       # cv2 convention: (cols, rows)
            interpolation=cv2.INTER_LINEAR,
        )

        # BGR → RGB: UFLD and most ONNX vision models expect RGB input.
        rgb: NDArray[np.uint8] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalise to [0, 1] in float32.
        normalised: Float32Array = rgb.astype(np.float32) / 255.0

        logger.debug(
            "preprocess_dl: %s → %s  dtype=%s",
            frame.shape[:2], normalised.shape[:2], normalised.dtype,
        )
        return normalised

    # ------------------------------------------------------------------
    # Stage 2a — Inference: mock branch
    # ------------------------------------------------------------------

    def _mock_predict(
        self,
        frame: BgrImage,
    ) -> Tuple[AnchorArray, AnchorArray]:
        """Generate synthetic row-anchor predictions without a trained model.

        Produces plausible left/right lane anchor positions using a
        parametric highway-lane geometry model:

        * **Left lane** : x ≈ 0.30 × W at the bottom anchor (near-field),
          converging to x ≈ 0.44 × W at the top anchor (far-field).
        * **Right lane**: x ≈ 0.70 × W at the bottom anchor, converging
          to x ≈ 0.56 × W at the top anchor.

        Gaussian noise (std = ``config.mock_noise_std_px``) is added to
        every anchor to simulate the natural per-prediction variance of a
        trained network.

        Parameters
        ----------
        frame : BgrImage
            Camera frame.  Only ``frame.shape`` is read; pixel content
            is not used in mock mode.

        Returns
        -------
        left_anchors : AnchorArray
            Shape (n_row_anchors, 2) — columns: [x_pixel, y_pixel].
            Row 0 = near-field (bottom of frame), row −1 = far-field.
        right_anchors : AnchorArray
            Shape (n_row_anchors, 2), same ordering.

        Notes
        -----
        The four x-ratio constants (0.30 / 0.70 near, 0.44 / 0.56 far)
        encode standard dual-carriageway lane geometry for a 3.7 m lane
        width in a 1 280 px wide frame.  They are the only intentional
        non-config constants in this method; all other values derive
        from ``self.config``.

        Linear interpolation between the bottom and top x-positions
        approximates the perspective foreshortening of straight lane
        lines — sufficient for validating downstream polynomial fitting
        and curvature stages without model-specific artefacts.
        """
        _H, W = self.config.frame_shape
        n: int = self.config.n_row_anchors

        # t ∈ [0, 1]: 0 = near-field bottom anchor, 1 = far-field top anchor.
        # Parameterises the linear x-convergence toward the vanishing point.
        t: NDArray[np.float64] = np.linspace(0.0, 1.0, n, dtype=np.float64)

        # Standard highway lane x-ratios (dual carriageway, 3.7 m lanes)
        left_x_near:  float = 0.30 * W
        left_x_far:   float = 0.44 * W
        right_x_near: float = 0.70 * W
        right_x_far:  float = 0.56 * W

        # Interpolate from near-field to far-field along t
        left_x:  NDArray[np.float64] = left_x_near  + t * (left_x_far  - left_x_near)
        right_x: NDArray[np.float64] = right_x_near + t * (right_x_far - right_x_near)

        # self._anchor_ys[0] = anchor_ratio_start × H  (far-field / small y)
        # self._anchor_ys[-1] = anchor_ratio_end × H   (near-field / large y)
        # Flip so index 0 corresponds to the near-field anchor (large y),
        # matching the bottom-to-top ordering of classic.py sliding windows.
        anchor_ys_near_first: NDArray[np.float64] = self._anchor_ys[::-1].copy()

        # Gaussian noise simulates model prediction uncertainty
        rng = np.random.default_rng()
        noise_left:  NDArray[np.float64] = rng.normal(
            loc=0.0, scale=self.config.mock_noise_std_px, size=n
        )
        noise_right: NDArray[np.float64] = rng.normal(
            loc=0.0, scale=self.config.mock_noise_std_px, size=n
        )

        left_anchors: AnchorArray = np.column_stack(
            [left_x  + noise_left,  anchor_ys_near_first]
        )
        right_anchors: AnchorArray = np.column_stack(
            [right_x + noise_right, anchor_ys_near_first]
        )

        logger.debug(
            "_mock_predict: left_x∈[%.1f, %.1f]px  "
            "right_x∈[%.1f, %.1f]px  noise_std=%.1fpx",
            left_anchors[:, 0].min(),  left_anchors[:, 0].max(),
            right_anchors[:, 0].min(), right_anchors[:, 0].max(),
            self.config.mock_noise_std_px,
        )
        return left_anchors, right_anchors

    # ------------------------------------------------------------------
    # Stage 2b — Inference: ONNX branch
    # ------------------------------------------------------------------

    def _onnx_predict(
        self,
        frame: BgrImage,
    ) -> Tuple[AnchorArray, AnchorArray]:
        """Run ONNX model inference to obtain row-anchor predictions.

        Preprocesses ``frame`` via :meth:`preprocess_dl`, feeds it
        through the ONNX Runtime session loaded in ``__init__``, and
        decodes the raw output tensor into left/right anchor arrays.

        Parameters
        ----------
        frame : BgrImage
            Raw camera frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        left_anchors : AnchorArray
            Shape (n_row_anchors, 2) — columns: [x_pixel, y_pixel].
        right_anchors : AnchorArray
            Shape (n_row_anchors, 2).

        Raises
        ------
        NotImplementedError
            Always — production ONNX output decoding is model-specific.
            Implement this method once a UFLD ONNX export is available
            and its output tensor layout is known.

        Notes
        -----
        Expected implementation steps (for future contributor):

        1. Call ``self.preprocess_dl(frame)`` → float32 array.
        2. Add batch dimension: ``input_tensor = np.expand_dims(..., 0)``.
        3. Run ``self._ort_session.run(None, {input_name: input_tensor})``.
        4. Decode the output tensor (shape depends on model architecture)
           into two (n_row_anchors, 2) AnchorArrays.
        5. Apply ``config.confidence_threshold`` to filter low-confidence
           anchor predictions before returning.

        See ``docs/onnx_integration.md`` for the expected tensor contract.
        """
        raise NotImplementedError(
            "_onnx_predict() is not yet implemented.\n"
            "Steps to integrate a real UFLD ONNX model:\n"
            "  1. Export weights to ONNX (see docs/onnx_integration.md).\n"
            "  2. Inspect output tensor shape with onnxruntime.\n"
            "  3. Decode the output into AnchorArray of shape "
            "(n_row_anchors, 2) per side.\n"
            "  4. Set config.mock_mode=False and supply onnx_model_path.\n"
            "Until then, use mock_mode=True for full-pipeline validation."
        )

    # ------------------------------------------------------------------
    # Stage 3 — Anchor → Polynomial
    # ------------------------------------------------------------------

    def anchors_to_polynomial(
        self,
        anchors:      AnchorArray,
        image_height: int,
    ) -> Optional[PolyCoeffs]:
        """Fit a polynomial to row-anchor (x, y) point pairs.

        Applies ``numpy.polyfit`` using the same ``x = f(y)`` convention
        as ``classic.py`` so that coefficients from both branches are
        directly interchangeable in the fusion layer.

        Parameters
        ----------
        anchors : AnchorArray
            Array of shape (n_row_anchors, 2) where column 0 is the
            predicted x-position (pixels) and column 1 is the anchor
            y-position (pixels).
        image_height : int
            Full pixel height of the original frame.  Used for logging
            context; not involved in the fit arithmetic.

        Returns
        -------
        Optional[PolyCoeffs]
            Coefficients ``[A, B, C]`` for ``x = A·y² + B·y + C``,
            dtype ``float64``, or ``None`` if the fit fails due to
            insufficient points or numerical rank deficiency.

        Notes
        -----
        **Minimum points**: ``config.poly_degree + 1``.

        **RankWarning handling**: ``numpy.exceptions.RankWarning``
        (NumPy ≥ 2.0) is promoted to an exception via
        ``warnings.catch_warnings`` so that degenerate fits surface as
        ``None`` rather than silently producing garbage coefficients.
        ``np.RankWarning`` is used as a fallback for NumPy 1.x
        compatibility.
        """
        min_pts: int = self.config.poly_degree + 1

        if anchors.shape[0] < min_pts:
            logger.debug(
                "anchors_to_polynomial: insufficient anchors "
                "(%d < %d required) — returning None",
                anchors.shape[0], min_pts,
            )
            return None

        # Column 1 = y (anchor strip y-position in the original frame)
        # Column 0 = x (predicted lane x-position)
        # x = f(y) convention matches classic.py fit_polynomial().
        anchor_y: NDArray[np.float64] = anchors[:, 1].astype(np.float64)
        anchor_x: NDArray[np.float64] = anchors[:, 0].astype(np.float64)

        try:
            with warnings.catch_warnings():
                # RankWarning path: NumPy ≥ 2.0 moved RankWarning into
                # numpy.exceptions; fall back to np.RankWarning for 1.x.
                try:
                    from numpy.exceptions import RankWarning  # NumPy ≥ 2.0
                except ImportError:
                    RankWarning = np.RankWarning  # type: ignore[attr-defined]

                warnings.simplefilter("error", RankWarning)
                coeffs: PolyCoeffs = np.polyfit(
                    anchor_y, anchor_x, self.config.poly_degree
                )

        except Exception as exc:
            logger.warning(
                "anchors_to_polynomial: polyfit failed (%s) — returning None",
                exc,
            )
            return None

        logger.debug(
            "anchors_to_polynomial: %d anchors → "
            "A=%.4e  B=%.4e  C=%.1f  (image_height=%d)",
            anchors.shape[0],
            float(coeffs[0]), float(coeffs[1]), float(coeffs[2]),
            image_height,
        )
        return coeffs

    # ------------------------------------------------------------------
    # Stage 4 — Master Pipeline
    # ------------------------------------------------------------------

    def detect(self, frame: BgrImage) -> LaneDetectionResult:
        """Run the full DL lane detection pipeline on one frame.

        Orchestrates all stages in order::

            _mock_predict | _onnx_predict
                → anchors_to_polynomial (left + right, pixel space)
                → metric-space refit
                → radius-of-curvature calculation
                → lateral-offset calculation
                → LaneDetectionResult

        The output is interface-compatible with
        :class:`~classic.GeometricLaneDetector` so the fusion layer
        can compare or blend results without branch-specific logic.

        Parameters
        ----------
        frame : BgrImage
            Single BGR video frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        LaneDetectionResult
            Fully populated result container.  Check ``result.valid``
            before consuming polynomial coefficients.

        Raises
        ------
        ValueError
            If ``frame`` is not a 3-channel uint8 array.

        Examples
        --------
        >>> detector = DeepLaneDetector(DeepLaneConfig(mock_mode=True))
        >>> result = detector.detect(frame)
        >>> if result.valid:
        ...     print(f"Curvature: {result.curvature_m:.1f} m")
        ...     print(f"Offset:    {result.offset_m:+.3f} m")
        """
        # ── Step 1: Input validation ──────────────────────────────────────
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"detect() expects a 3-channel uint8 BGR image (H, W, 3), "
                f"got shape={frame.shape}, dtype={frame.dtype}."
            )

        # ── Step 2: Inference — branch selected by config ─────────────────
        if self.config.mock_mode:
            left_anchors, right_anchors = self._mock_predict(frame)
        else:
            left_anchors, right_anchors = self._onnx_predict(frame)

        logger.debug(
            "detect: anchors obtained — left=%s  right=%s",
            left_anchors.shape, right_anchors.shape,
        )

        # ── Step 3: Pixel-space polynomial fits ───────────────────────────
        image_height: int = frame.shape[0]

        left_fit:  Optional[PolyCoeffs] = self.anchors_to_polynomial(
            left_anchors,  image_height
        )
        right_fit: Optional[PolyCoeffs] = self.anchors_to_polynomial(
            right_anchors, image_height
        )

        # ── Step 4: Metric-space refit (for physically-meaningful curvature)
        # Re-fit anchor points in metres so the curvature formula yields R
        # in metres — directly comparable to classic.py's output.
        def _metric_fit(
            anchors:    AnchorArray,
            pixel_fit:  Optional[PolyCoeffs],
        ) -> Optional[PolyCoeffs]:
            """Return metric-space polyfit, or None if pixel fit failed."""
            if pixel_fit is None:
                return None
            y_m: NDArray[np.float64] = anchors[:, 1] * self.config.ym_per_pix
            x_m: NDArray[np.float64] = anchors[:, 0] * self.config.xm_per_pix
            min_pts = self.config.poly_degree + 1
            if y_m.size < min_pts:
                return None
            try:
                with warnings.catch_warnings():
                    try:
                        from numpy.exceptions import RankWarning
                    except ImportError:
                        RankWarning = np.RankWarning  # type: ignore[attr-defined]
                    warnings.simplefilter("error", RankWarning)
                    return np.polyfit(y_m, x_m, self.config.poly_degree)
            except Exception:
                return None

        left_fit_m:  Optional[PolyCoeffs] = _metric_fit(left_anchors,  left_fit)
        right_fit_m: Optional[PolyCoeffs] = _metric_fit(right_anchors, right_fit)

        # ── Step 5: Radius of curvature ───────────────────────────────────
        # For x = A·y² + B·y + C:
        #   R = (1 + (2A·y_eval + B)²)^1.5 / |2A|
        # Evaluated at y_eval = bottom of frame in metres, i.e. the
        # position closest to the vehicle — most relevant for steering.
        y_eval_m: float = (image_height - 1) * self.config.ym_per_pix

        def _curvature(fit_m: Optional[PolyCoeffs]) -> Optional[float]:
            if fit_m is None:
                return None
            A, B = float(fit_m[0]), float(fit_m[1])
            denom = abs(2.0 * A)
            if denom < 1e-10:   # near-zero A → effectively straight road
                return None
            return (1.0 + (2.0 * A * y_eval_m + B) ** 2) ** 1.5 / denom

        left_curv_m:  Optional[float] = _curvature(left_fit_m)
        right_curv_m: Optional[float] = _curvature(right_fit_m)

        valid_curvatures: List[float] = [
            c for c in (left_curv_m, right_curv_m) if c is not None
        ]
        curvature_m: Optional[float] = (
            float(np.mean(valid_curvatures)) if valid_curvatures else None
        )

        # ── Step 6: Lateral offset from lane centre ───────────────────────
        # Evaluates both pixel-space polynomial fits at the bottom row,
        # computes the lane centreline, and converts the vehicle's
        # deviation from that centreline to metres.
        # Positive offset → vehicle is to the right of lane centre.
        offset_m: Optional[float] = None
        if left_fit is not None and right_fit is not None:
            y_bottom: int          = image_height - 1
            left_base_x:  float    = float(np.polyval(left_fit,  y_bottom))
            right_base_x: float    = float(np.polyval(right_fit, y_bottom))
            lane_center_x: float   = (left_base_x + right_base_x) / 2.0
            frame_center_x: float  = frame.shape[1] / 2.0
            offset_m = (frame_center_x - lane_center_x) * self.config.xm_per_pix

        # ── Step 7: Confidence score ──────────────────────────────────────
        # DL branch confidence = fraction of expected anchor points that
        # were successfully produced (both sides).  In mock mode this is
        # always 1.0 (all anchors present); in ONNX mode anchors filtered
        # by confidence_threshold may reduce this value.
        # Hard-clamp to 0.0 if either fit failed — mirrors classic.py so
        # the fusion layer can compare scores without branch awareness.
        total_expected: int = self.config.n_row_anchors * 2
        total_used:     int = left_anchors.shape[0] + right_anchors.shape[0]
        confidence: float   = float(total_used) / max(float(total_expected), 1.0)

        if left_fit is None or right_fit is None:
            confidence = 0.0

        # ── Step 8: Validity gate ─────────────────────────────────────────
        valid: bool = (
            left_fit  is not None
            and right_fit is not None
            and confidence > 0.0
        )

        logger.debug(
            "detect: valid=%s  conf=%.3f  curv=%s m  offset=%s m",
            valid,
            confidence,
            f"{curvature_m:.1f}" if curvature_m is not None else "None",
            f"{offset_m:+.3f}"   if offset_m    is not None else "None",
        )

        return LaneDetectionResult(
            left_fit    = left_fit,
            right_fit   = right_fit,
            curvature_m = curvature_m,
            offset_m    = offset_m,
            confidence  = confidence,
            valid       = valid,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return f"DeepLaneDetector(config={self.config!r})"
