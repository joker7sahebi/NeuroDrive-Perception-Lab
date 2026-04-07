"""
lane_fusion.py — Lane Detection Fusion Layer
=============================================
NeuroDrive-Perception-Lab | src/fusion/lane_fusion.py
------------------------------------------------------
Role in the Hybrid Pipeline
----------------------------
This module is the **arbitration layer** that sits between the two lane
detection branches and the downstream vehicle-control / HUD consumers:

    GeometricLaneDetector (classic.py)  ─┐
                                          ├─▶ LaneFusionEngine ─▶ LaneDetectionResult
    DeepLaneDetector      (deep.py)     ─┘

Both upstream detectors expose identical :class:`~classic.LaneDetectionResult`
outputs, so this layer requires no branch-specific logic — it operates purely
on confidence scores, polynomial coefficients, and physical sanity bounds.

Fusion Strategy
---------------
The engine implements a **confidence-weighted arbitration** policy with
four clearly ordered cases:

    Case 1 — Both valid & sane
        Fused coefficients = weighted average of left/right polynomials,
        weights = (confidence × branch_boost).  Curvature and offset are
        likewise weighted.  ``source_label = "fused"``.

    Case 2 — Only classic valid & sane
        Pass-through without modification.  ``source_label = "classic"``.

    Case 3 — Only deep valid & sane
        Pass-through without modification.  ``source_label = "deep"``.

    Case 4 — Neither valid
        Return an empty (invalid) :class:`~classic.LaneDetectionResult`.
        ``source_label = "failed"``.

Physical Sanity Checks
-----------------------
Before any result is accepted, :meth:`LaneFusionEngine._is_result_sane`
validates that curvature and lateral offset are within the bounds
declared in :class:`FusionConfig`.  Results that pass the detector's own
``valid`` flag but contain physically impossible values (e.g. 1 m radius
of curvature at highway speed, or 4 m lateral offset) are treated as
failures.  This is a functional-safety backstop required for ADAS
deployments targeting ISO 26262 ASIL-B.

Integration Notes
-----------------
- :meth:`LaneFusionEngine.run` is the single public entry point for
  frame-level callers; it runs both detectors and fuses in one call.
- :meth:`LaneFusionEngine.fuse` accepts pre-computed results, enabling
  independent unit-testing of each branch before fusion.
- The ``source_label`` return value of ``run()`` is designed for
  downstream telemetry logging and HUD colour-coding (green = fused,
  yellow = single-branch, red = failed).

Author : Portfolio — Senior ADAS CV Engineer
Target : BMW / Bosch / CARIAD — NeuroDrive-Perception-Lab
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Imports from completed branches
# ---------------------------------------------------------------------------
# In the installed package these resolve normally.  In a Colab cell where
# classic.py and deep.py live in the same directory, add that directory to
# sys.path before importing.
try:
    from classic import (       # type: ignore[import]
        GeometricLaneDetector,
        LaneDetectionConfig,
        LaneDetectionResult,
        BgrImage,
        PolyCoeffs,
    )
    from deep import (          # type: ignore[import]
        DeepLaneDetector,
        DeepLaneConfig,
    )
except ModuleNotFoundError:
    # ── Fallback stubs for isolated testing / CI without the full tree ───
    # These mirror the real signatures precisely enough for type-checkers
    # and unit tests to work without importing the full detector modules.
    import dataclasses as _dc

    BgrImage   = NDArray[np.uint8]    # type: ignore[misc]
    PolyCoeffs = NDArray[np.float64]  # type: ignore[misc]

    LaneDetectionResult = _dc.make_dataclass(   # type: ignore[misc,assignment]
        "LaneDetectionResult",
        [
            ("left_fit",    Optional[NDArray[np.float64]], _dc.field(default=None)),
            ("right_fit",   Optional[NDArray[np.float64]], _dc.field(default=None)),
            ("curvature_m", Optional[float],               _dc.field(default=None)),
            ("offset_m",    Optional[float],               _dc.field(default=None)),
            ("confidence",  float,                         _dc.field(default=0.0)),
            ("valid",       bool,                          _dc.field(default=False)),
        ],
    )

    class LaneDetectionConfig:   # type: ignore[no-redef]
        pass

    class GeometricLaneDetector:  # type: ignore[no-redef]
        def detect(self, frame: BgrImage) -> LaneDetectionResult:  # type: ignore[empty-body]
            ...

    class DeepLaneConfig:  # type: ignore[no-redef]
        pass

    class DeepLaneDetector:   # type: ignore[no-redef]
        def detect(self, frame: BgrImage) -> LaneDetectionResult:  # type: ignore[empty-body]
            ...

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source label type alias — documents the four arbitration outcomes
# ---------------------------------------------------------------------------
SourceLabel = str   # Literal["fused", "classic", "deep", "failed"]


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Hyperparameter container for :class:`LaneFusionEngine`.

    All arbitration thresholds and weight boosts are declared here so that
    method bodies remain free of magic numbers.  Modify this dataclass —
    not method bodies — when tuning for a specific vehicle or road domain.

    Parameters
    ----------
    min_confidence_threshold : float
        A branch result with ``confidence`` below this value is treated as
        invalid for fusion purposes, even if the branch itself reports
        ``valid=True``.  Range: [0.0, 1.0].
    classic_weight_boost : float
        Multiplicative factor applied to the classical branch's confidence
        score before weight normalisation.  Set > 1.0 to prefer classical
        results on high-contrast, well-lit highways.  Default 1.0 (neutral).
    deep_weight_boost : float
        Multiplicative factor for the DL branch.  Set > 1.0 to prefer DL
        results under adverse lighting or faded lane markings.
    max_curvature_m : float
        Upper bound on physically plausible lane curvature radius (metres).
        Values above this indicate a near-straight road modelled by an
        ill-conditioned polynomial — treated as a sanity failure.
        50 000 m ≈ curvature of a perfectly straight motorway.
    min_curvature_m : float
        Lower bound on curvature radius.  50 m corresponds to a very sharp
        urban corner; tighter radii are implausible at typical ADAS speeds
        and indicate a fitting artefact.
    max_offset_m : float
        Maximum plausible lateral offset of the ego vehicle from lane centre
        (metres).  Exceeding 3 m implies the vehicle is fully in an adjacent
        lane — almost certainly a detection error.
    frame_shape : Tuple[int, int]
        Expected input resolution (H, W).  Currently reserved for
        future per-resolution scaling of pixel-space sanity bounds.
    """

    min_confidence_threshold: float          = 0.3
    classic_weight_boost:     float          = 1.0
    deep_weight_boost:        float          = 1.0
    max_curvature_m:          float          = 50_000.0
    min_curvature_m:          float          = 50.0
    max_offset_m:             float          = 3.0
    frame_shape:              Tuple[int,int] = (720, 1280)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FusionConfig("
            f"conf_thresh={self.min_confidence_threshold:.2f}, "
            f"boosts=(classic={self.classic_weight_boost:.2f}, "
            f"deep={self.deep_weight_boost:.2f}), "
            f"curv=[{self.min_curvature_m:.0f}, {self.max_curvature_m:.0f}]m, "
            f"max_offset={self.max_offset_m:.1f}m)"
        )


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

class LaneFusionEngine:
    """Confidence-weighted arbitration layer for hybrid lane detection.

    Accepts one :class:`~classic.LaneDetectionResult` from each detector
    branch and applies the fusion strategy described in the module docstring.
    The output is a single :class:`~classic.LaneDetectionResult` suitable
    for direct consumption by steering / HUD / data-logging consumers.

    Parameters
    ----------
    config : FusionConfig
        Arbitration hyperparameters.

    Examples
    --------
    >>> fusion_cfg = FusionConfig(classic_weight_boost=1.2)
    >>> engine     = LaneFusionEngine(fusion_cfg)
    >>> result, label = engine.run(frame, classic_detector, deep_detector)
    >>> print(label, result.curvature_m)
    """

    def __init__(self, config: FusionConfig) -> None:
        """Initialise the fusion engine with validated configuration.

        Parameters
        ----------
        config : FusionConfig
            Arbitration hyperparameters bundle.

        Raises
        ------
        TypeError
            If ``config`` is not a :class:`FusionConfig` instance.
        ValueError
            If any confidence threshold or boost is outside (0, ∞), or if
            ``min_curvature_m >= max_curvature_m``.
        """
        if not isinstance(config, FusionConfig):
            raise TypeError(
                f"config must be a FusionConfig instance, "
                f"got {type(config).__name__!r}."
            )
        if not (0.0 <= config.min_confidence_threshold <= 1.0):
            raise ValueError(
                f"min_confidence_threshold must be in [0, 1], "
                f"got {config.min_confidence_threshold}."
            )
        if config.classic_weight_boost <= 0.0 or config.deep_weight_boost <= 0.0:
            raise ValueError(
                "classic_weight_boost and deep_weight_boost must be > 0."
            )
        if config.min_curvature_m >= config.max_curvature_m:
            raise ValueError(
                f"min_curvature_m ({config.min_curvature_m}) must be "
                f"< max_curvature_m ({config.max_curvature_m})."
            )
        if config.max_offset_m <= 0.0:
            raise ValueError(
                f"max_offset_m must be > 0, got {config.max_offset_m}."
            )

        self.config: FusionConfig = config
        logger.debug("LaneFusionEngine initialised. %r", self.config)

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    def _is_result_sane(self, result: LaneDetectionResult) -> bool:
        """Check whether a detection result is physically plausible.

        A result passes sanity if **all** of the following hold:

        1. ``result.valid`` is ``True``.
        2. ``result.confidence >= config.min_confidence_threshold``.
        3. ``result.curvature_m`` is not ``None`` and lies within
           ``[config.min_curvature_m, config.max_curvature_m]``.
        4. ``result.offset_m`` is not ``None`` and its absolute value
           is ``<= config.max_offset_m``.
        5. Both ``result.left_fit`` and ``result.right_fit`` are not ``None``.

        Parameters
        ----------
        result : LaneDetectionResult
            Candidate detection result from either branch.

        Returns
        -------
        bool
            ``True`` if the result passes all sanity gates, ``False``
            if any gate fails.

        Notes
        -----
        This method is intentionally conservative.  A false-negative
        (rejecting a valid result) is safer than a false-positive
        (accepting a bad polynomial for steering computation) in an
        ISO 26262 ASIL-B pipeline.
        """
        # Gate 1 — branch-level validity flag
        if not result.valid:
            logger.debug("_is_result_sane: FAIL — result.valid=False")
            return False

        # Gate 2 — minimum confidence threshold
        if result.confidence < self.config.min_confidence_threshold:
            logger.debug(
                "_is_result_sane: FAIL — confidence=%.3f < threshold=%.3f",
                result.confidence, self.config.min_confidence_threshold,
            )
            return False

        # Gate 3 — polynomial coefficients present
        if result.left_fit is None or result.right_fit is None:
            logger.debug("_is_result_sane: FAIL — polynomial fit(s) are None")
            return False

        # Gate 4 — curvature within physical bounds
        if result.curvature_m is None:
            logger.debug("_is_result_sane: FAIL — curvature_m is None")
            return False
        if not (
            self.config.min_curvature_m
            <= result.curvature_m
            <= self.config.max_curvature_m
        ):
            logger.debug(
                "_is_result_sane: FAIL — curvature_m=%.1f outside [%.1f, %.1f]",
                result.curvature_m,
                self.config.min_curvature_m,
                self.config.max_curvature_m,
            )
            return False

        # Gate 5 — lateral offset within physical bounds
        if result.offset_m is None:
            logger.debug("_is_result_sane: FAIL — offset_m is None")
            return False
        if abs(result.offset_m) > self.config.max_offset_m:
            logger.debug(
                "_is_result_sane: FAIL — |offset_m|=%.3f > max=%.3f",
                abs(result.offset_m), self.config.max_offset_m,
            )
            return False

        logger.debug(
            "_is_result_sane: PASS — conf=%.3f, curv=%.1f m, offset=%+.3f m",
            result.confidence, result.curvature_m, result.offset_m,
        )
        return True

    # ------------------------------------------------------------------
    # Polynomial blending
    # ------------------------------------------------------------------

    def _weighted_average_fit(
        self,
        fit_a:    PolyCoeffs,
        weight_a: float,
        fit_b:    PolyCoeffs,
        weight_b: float,
    ) -> PolyCoeffs:
        """Compute a normalised weighted average of two polynomial coefficient arrays.

        Each coefficient index is blended independently::

            fit_out[i] = (fit_a[i] * weight_a + fit_b[i] * weight_b)
                         / (weight_a + weight_b)

        This is the standard sensor-fusion blending step for homogeneous
        polynomial representations.  It assumes both fits use the same
        degree and the same ``x = f(y)`` convention as established in
        ``classic.py``.

        Parameters
        ----------
        fit_a : PolyCoeffs
            Coefficient array [A, B, C] from branch A, shape (poly_degree+1,).
        weight_a : float
            Non-negative scalar weight for ``fit_a``.
        fit_b : PolyCoeffs
            Coefficient array [A, B, C] from branch B, same shape.
        weight_b : float
            Non-negative scalar weight for ``fit_b``.

        Returns
        -------
        PolyCoeffs
            Blended coefficient array, same shape as inputs, dtype float64.

        Raises
        ------
        ValueError
            If ``fit_a`` and ``fit_b`` have different shapes, or if the
            combined weight is zero (both weights are zero).
        """
        if fit_a.shape != fit_b.shape:
            raise ValueError(
                f"_weighted_average_fit: shape mismatch — "
                f"fit_a={fit_a.shape}, fit_b={fit_b.shape}."
            )
        total_weight: float = weight_a + weight_b
        if total_weight == 0.0:
            raise ValueError(
                "_weighted_average_fit: combined weight is zero — "
                "at least one weight must be positive."
            )
        blended: PolyCoeffs = (
            fit_a.astype(np.float64) * weight_a
            + fit_b.astype(np.float64) * weight_b
        ) / total_weight
        return blended

    # ------------------------------------------------------------------
    # Core fusion logic
    # ------------------------------------------------------------------

    def fuse(
        self,
        classic_result: LaneDetectionResult,
        deep_result:    LaneDetectionResult,
    ) -> Tuple[LaneDetectionResult, SourceLabel]:
        """Arbitrate between two pre-computed detection results.

        Applies the four-case fusion policy defined in the module docstring.
        In Case 1 (both valid), the fused polynomial coefficients, curvature,
        and offset are all computed as confidence-weighted averages so that
        the stronger signal dominates without completely discarding the other.

        Parameters
        ----------
        classic_result : LaneDetectionResult
            Output of :class:`~classic.GeometricLaneDetector`.
        deep_result : LaneDetectionResult
            Output of :class:`~deep.DeepLaneDetector`.

        Returns
        -------
        result : LaneDetectionResult
            Fused (or pass-through) detection result.
        source_label : SourceLabel
            String identifying which case was triggered:
            ``"fused"`` | ``"classic"`` | ``"deep"`` | ``"failed"``.

        Notes
        -----
        The ``confidence`` of the fused result in Case 1 is the
        weight-normalised average of the two input confidences, capped
        at 1.0.  This means a fused result can never exceed the
        confidence of the stronger branch.
        """
        classic_sane: bool = self._is_result_sane(classic_result)
        deep_sane:    bool = self._is_result_sane(deep_result)

        # ── Case 1: Both valid + sane → weighted average ─────────────────
        if classic_sane and deep_sane:
            w_c: float = classic_result.confidence * self.config.classic_weight_boost
            w_d: float = deep_result.confidence    * self.config.deep_weight_boost
            total_w: float = w_c + w_d

            # Blend polynomial coefficients for left and right lines
            fused_left_fit: PolyCoeffs = self._weighted_average_fit(
                classic_result.left_fit,  w_c,     # type: ignore[arg-type]
                deep_result.left_fit,     w_d,     # type: ignore[arg-type]
            )
            fused_right_fit: PolyCoeffs = self._weighted_average_fit(
                classic_result.right_fit, w_c,     # type: ignore[arg-type]
                deep_result.right_fit,    w_d,     # type: ignore[arg-type]
            )

            # Weighted average of scalar outputs — None-safe via type checking
            # (sanity gates above guarantee these are not None in this branch)
            fused_curvature: float = (
                classic_result.curvature_m * w_c    # type: ignore[operator]
                + deep_result.curvature_m  * w_d    # type: ignore[operator]
            ) / total_w

            fused_offset: float = (
                classic_result.offset_m * w_c       # type: ignore[operator]
                + deep_result.offset_m  * w_d       # type: ignore[operator]
            ) / total_w

            fused_confidence: float = min(
                (classic_result.confidence * w_c + deep_result.confidence * w_d)
                / total_w,
                1.0,
            )

            logger.debug(
                "fuse: Case 1 (FUSED) — w_classic=%.3f, w_deep=%.3f, "
                "conf=%.3f, curv=%.1f m, offset=%+.3f m",
                w_c, w_d, fused_confidence, fused_curvature, fused_offset,
            )

            return (
                LaneDetectionResult(
                    left_fit    = fused_left_fit,
                    right_fit   = fused_right_fit,
                    curvature_m = fused_curvature,
                    offset_m    = fused_offset,
                    confidence  = fused_confidence,
                    valid       = True,
                ),
                "fused",
            )

        # ── Case 2: Only classic valid + sane ────────────────────────────
        if classic_sane:
            logger.debug(
                "fuse: Case 2 (CLASSIC only) — conf=%.3f, "
                "curv=%.1f m, offset=%+.3f m",
                classic_result.confidence,
                classic_result.curvature_m,
                classic_result.offset_m,
            )
            return classic_result, "classic"

        # ── Case 3: Only deep valid + sane ───────────────────────────────
        if deep_sane:
            logger.debug(
                "fuse: Case 3 (DEEP only) — conf=%.3f, "
                "curv=%.1f m, offset=%+.3f m",
                deep_result.confidence,
                deep_result.curvature_m,
                deep_result.offset_m,
            )
            return deep_result, "deep"

        # ── Case 4: Neither valid ─────────────────────────────────────────
        logger.debug(
            "fuse: Case 4 (FAILED) — classic_sane=%s, deep_sane=%s",
            classic_sane, deep_sane,
        )
        return LaneDetectionResult(), "failed"

    # ------------------------------------------------------------------
    # Convenience pipeline runner
    # ------------------------------------------------------------------

    def run(
        self,
        frame:        BgrImage,
        classic_det:  GeometricLaneDetector,
        deep_det:     DeepLaneDetector,
    ) -> Tuple[LaneDetectionResult, SourceLabel]:
        """Run both detectors on ``frame`` and return the fused result.

        This is the single public entry point for frame-level callers.
        It encapsulates the detect → detect → fuse sequence and returns
        both the result and a human-readable source label suitable for
        telemetry logging or HUD colour coding.

        Pipeline::

            frame ─┬─▶ classic_det.detect() ─▶ classic_result ─┐
                   │                                              ├─▶ fuse() ─▶ (result, label)
                   └─▶ deep_det.detect()    ─▶ deep_result    ─┘

        Parameters
        ----------
        frame : BgrImage
            Single BGR camera frame, shape (H, W, 3), dtype ``uint8``.
        classic_det : GeometricLaneDetector
            Instantiated classical detector (already configured).
        deep_det : DeepLaneDetector
            Instantiated DL detector (already configured).

        Returns
        -------
        result : LaneDetectionResult
            Final fused detection result.  Always check ``result.valid``
            before consuming downstream.
        source_label : SourceLabel
            ``"fused"``   — both branches contributed.
            ``"classic"`` — only the classical branch was sane.
            ``"deep"``    — only the DL branch was sane.
            ``"failed"``  — neither branch produced a usable result.

        Raises
        ------
        ValueError
            If ``frame`` is not a 3-channel uint8 BGR image.

        Notes
        -----
        Exceptions raised inside either detector's ``detect()`` call are
        intentionally **not** caught here — they represent programming
        errors (wrong dtype, wrong shape) that should surface immediately
        during integration testing rather than being silently swallowed.

        Examples
        --------
        >>> engine = LaneFusionEngine(FusionConfig())
        >>> result, label = engine.run(frame, classic_det, deep_det)
        >>> if result.valid:
        ...     print(f"[{label}] curvature={result.curvature_m:.0f}m "
        ...           f"offset={result.offset_m:+.3f}m")
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"run() expects a 3-channel uint8 BGR image (H, W, 3), "
                f"got shape={frame.shape}, dtype={frame.dtype}."
            )

        # ── Run both detectors independently ─────────────────────────────
        logger.debug("run: invoking classic detector …")
        classic_result: LaneDetectionResult = classic_det.detect(frame)
        logger.debug(
            "run: classic → valid=%s, conf=%.3f",
            classic_result.valid, classic_result.confidence,
        )

        logger.debug("run: invoking deep detector …")
        deep_result: LaneDetectionResult = deep_det.detect(frame)
        logger.debug(
            "run: deep    → valid=%s, conf=%.3f",
            deep_result.valid, deep_result.confidence,
        )

        # ── Arbitrate ─────────────────────────────────────────────────────
        result, source_label = self.fuse(classic_result, deep_result)

        logger.debug(
            "run: final → source=%r, valid=%s, conf=%.3f",
            source_label, result.valid, result.confidence,
        )

        return result, source_label

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return f"LaneFusionEngine(config={self.config!r})"
