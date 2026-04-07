"""
detector.py — Object Detection & Tracking Module
=================================================
NeuroDrive-Perception-Lab | src/modules/objects/detector.py
-----------------------------------------------------------
Role in the Hybrid Pipeline
----------------------------
This module implements the **object detection and tracking branch** of the
NeuroDrive perception stack.  It operates in parallel with the lane detection
branches and feeds its output to the same fusion / scene-understanding layer.

Architecture: YOLO-style Detection + Simple IoU Tracker
---------------------------------------------------------
Detection
    A YOLO-style single-shot detector regresses axis-aligned bounding boxes
    and class probabilities in a single forward pass.  Non-Maximum Suppression
    (NMS) is applied post-inference to remove duplicate detections.

Tracking
    :class:`SimpleIoUTracker` performs frame-to-frame identity assignment by
    matching new detections to existing tracks via Intersection-over-Union
    (IoU) overlap.  No motion model is used at this stage — Kalman-filter
    prediction will be added in Phase 6.

Operating Modes
---------------
mock_mode = True  (default)
    Synthetic bounding boxes are generated analytically from the frame
    content hash, so identical frames always produce identical detections.
    No GPU, no ONNX runtime required.  Suitable for CI, Colab, and
    integration testing of downstream fusion modules.

mock_mode = False
    A real ONNX YOLOv8/YOLOv5 model is loaded via ``onnxruntime``.
    The ``_onnx_detect`` stub raises :exc:`NotImplementedError` until
    the output tensor → BoundingBox mapping is implemented.

Supported Object Classes
------------------------
    0 = car
    1 = truck
    2 = pedestrian
    3 = cyclist
    4 = motorcycle

References
----------
- Jocher et al., "YOLOv8", Ultralytics 2023.
  https://github.com/ultralytics/ultralytics
- Bewley et al., "Simple Online and Realtime Tracking (SORT)", ICIP 2016.
  https://arxiv.org/abs/1602.00763

Author : Portfolio — Senior ADAS CV Engineer
Target : BMW / Bosch / CARIAD — NeuroDrive-Perception-Lab
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias — consistent with classic.py convention
# ---------------------------------------------------------------------------
BgrImage = NDArray[np.uint8]   # H×W×3, BGR colour order

# ---------------------------------------------------------------------------
# Class metadata — single source of truth; consumed by BoundingBox properties
# and detect_and_draw colour map
# ---------------------------------------------------------------------------
_CLASS_NAMES: Tuple[str, ...] = (
    "car", "truck", "pedestrian", "cyclist", "motorcycle"
)

# BGR colours — one per class_id (car/truck/pedestrian/cyclist/motorcycle)
# Named constants avoid magic RGB tuples scattered across method bodies.
_CLASS_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (255,   0,   0),   # 0 = car         → blue
    (  0, 255,   0),   # 1 = truck        → green
    (  0,   0, 255),   # 2 = pedestrian   → red
    (  0, 255, 255),   # 3 = cyclist      → yellow
    (255,   0, 255),   # 4 = motorcycle   → magenta
)


# ===========================================================================
# BoundingBox
# ===========================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box with class label and optional track identity.

    Coordinates follow the OpenCV image convention:
    origin at top-left, x increases right, y increases down.

    Parameters
    ----------
    x1 : float
        Left edge of the box (pixel column).
    y1 : float
        Top edge of the box (pixel row).
    x2 : float
        Right edge of the box (pixel column).  Must be >= x1.
    y2 : float
        Bottom edge of the box (pixel row).  Must be >= y1.
    confidence : float
        Detection confidence score, range [0, 1].
    class_id : int
        Categorical label index.  Valid values: 0–4.
    track_id : Optional[int]
        Persistent identity assigned by :class:`SimpleIoUTracker`.
        ``None`` until tracking has been run for this box.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    track_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Derived geometry properties                                          #
    # ------------------------------------------------------------------ #

    @property
    def width(self) -> float:
        """Horizontal extent of the box in pixels.

        Returns
        -------
        float
            ``x2 - x1``
        """
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Vertical extent of the box in pixels.

        Returns
        -------
        float
            ``y2 - y1``
        """
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels.

        Returns
        -------
        float
            ``width * height``
        """
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the box.

        Returns
        -------
        Tuple[float, float]
            ``(cx, cy)`` — column and row of the centre pixel.
        """
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def class_name(self) -> str:
        """Human-readable class label.

        Returns
        -------
        str
            One of ``"car"``, ``"truck"``, ``"pedestrian"``,
            ``"cyclist"``, ``"motorcycle"``, or
            ``"unknown_<class_id>"`` for out-of-range IDs.
        """
        if 0 <= self.class_id < len(_CLASS_NAMES):
            return _CLASS_NAMES[self.class_id]
        return f"unknown_{self.class_id}"

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BoundingBox({self.class_name}, "
            f"[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}], "
            f"conf={self.confidence:.2f}, track={self.track_id})"
        )


# ===========================================================================
# DetectionResult
# ===========================================================================

@dataclass
class DetectionResult:
    """Structured output of one :meth:`ObjectDetector.detect` call.

    Parameters
    ----------
    boxes : List[BoundingBox]
        All detected and NMS-filtered objects in this frame.
    frame_shape : Tuple[int, int]
        (H, W) of the source frame — required by downstream consumers
        to convert pixel coordinates to normalised / metric space.
    confidence : float
        Mean detection confidence across all boxes.  ``0.0`` when
        ``boxes`` is empty.
    valid : bool
        ``True`` if at least one box was detected after NMS.
    n_objects : int
        Convenience alias for ``len(boxes)``.
    """

    boxes:       List[BoundingBox]
    frame_shape: Tuple[int, int]
    confidence:  float
    valid:       bool
    n_objects:   int

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DetectionResult(n={self.n_objects}, "
            f"conf={self.confidence:.3f}, valid={self.valid})"
        )


# ===========================================================================
# ObjectDetectorConfig
# ===========================================================================

@dataclass
class ObjectDetectorConfig:
    """Hyperparameter container for :class:`ObjectDetector`.

    All scalar thresholds, sizes, and mock parameters are centralised here.
    Method bodies must reference ``self.config.<field>`` — never embed
    magic numbers in algorithm code.

    Parameters
    ----------
    frame_shape : Tuple[int, int]
        Expected input resolution (H, W) of the raw camera frame.
    confidence_threshold : float
        Minimum detection score to retain a box.  Boxes below this are
        discarded before NMS.  Range [0, 1].
    nms_iou_threshold : float
        IoU overlap threshold for Non-Maximum Suppression.  Two boxes
        with IoU > this value are considered duplicates; the lower-
        confidence box is suppressed.  Range [0, 1].
    max_detections : int
        Hard cap on the number of output boxes after NMS.  The top-N
        by confidence are kept.
    model_input_size : Tuple[int, int]
        (H, W) to which the frame is resized before ONNX inference.
        Must match the resolution used during model training.
    mock_mode : bool
        ``True`` → :meth:`ObjectDetector._mock_detect` (no model).
        ``False`` → :meth:`ObjectDetector._onnx_detect` (ONNX).
    onnx_model_path : Optional[str]
        Path to ``.onnx`` weights file.  Required when ``mock_mode=False``.
    mock_n_objects : int
        Number of synthetic boxes generated per frame in mock mode.
    mock_noise_std_px : float
        Standard deviation (pixels) of Gaussian noise added to mock box
        coordinates, simulating detector jitter.
    class_names : Tuple[str, ...]
        Ordered sequence of class labels.  Index equals ``class_id``.
    """

    frame_shape:          Tuple[int, int]  = (720, 1280)
    confidence_threshold: float            = 0.5
    nms_iou_threshold:    float            = 0.45
    max_detections:       int              = 50
    model_input_size:     Tuple[int, int]  = (640, 640)
    mock_mode:            bool             = True
    onnx_model_path:      Optional[str]    = None
    mock_n_objects:       int              = 4
    mock_noise_std_px:    float            = 10.0
    class_names:          Tuple[str, ...]  = field(
        default_factory=lambda: _CLASS_NAMES
    )

    # ── Strategy routing ─────────────────────────────────────────────────
    detection_strategy: str = "mock"
    # Valid values:
    #   "mock"     → _mock_detect — deterministic synthetic boxes (no model)
    #   "yolo"     → _yolo_detect — YOLOv8 via ultralytics (lazy import)
    #   "onnx"     → _onnx_detect — ONNX Runtime (stub until model wired)
    #   "ensemble" → _ensemble_detect — mock + yolo merged via NMS

    # ── YOLOv8 settings ──────────────────────────────────────────────────
    yolo_model_size: str = "yolov8n"
    # Nano (fastest) → XLarge (most accurate):
    # "yolov8n" | "yolov8s" | "yolov8m" | "yolov8l" | "yolov8x"

    yolo_confidence_threshold: float = 0.30
    # Separate per-frame score gate for YOLO inference; may be set lower
    # than confidence_threshold to allow more candidates into NMS.

    yolo_road_classes: Tuple[str, ...] = field(default_factory=lambda: (
        "car", "truck", "bus", "person",
        "bicycle", "motorcycle", "traffic light", "stop sign",
    ))
    # COCO class names that are relevant for road-scene ADAS; predictions
    # for classes outside this tuple are discarded before NMS.

    yolo_class_map: Dict[str, int] = field(default_factory=lambda: {
        "car":           0,
        "truck":         1,
        "bus":           1,   # mapped to truck (class 1)
        "person":        2,
        "bicycle":       3,
        "motorcycle":    4,
        "traffic light": 0,   # treated as car-class obstacle
        "stop sign":     0,   # treated as car-class obstacle
    })
    # Maps COCO class name → NeuroDrive class_id (0–4).
    # Keys must be a subset of yolo_road_classes.

    # ── RT-DETR settings ─────────────────────────────────────────────────
    rtdetr_model_size: str = "rtdetr-l"
    # "rtdetr-l" (large, balanced) | "rtdetr-x" (xlarge, highest accuracy)
    # RT-DETR is an end-to-end transformer detector — no NMS required
    # internally, but downstream _apply_nms is still applied for
    # consistency with the rest of the pipeline.
    # ultralytics wraps RT-DETR with the same .predict() API as YOLOv8,
    # so the same road-class filter and yolo_class_map are reused.

    # ── Grounding DINO settings ──────────────────────────────────────────
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    # HuggingFace model ID.  "grounding-dino-tiny" is the lightest variant;
    # "grounding-dino-base" offers higher recall at greater compute cost.

    grounding_dino_text_prompt: str = (
        "car . truck . person . bicycle . motorcycle ."
    )
    # Dot-separated object names sent to the text encoder.  Each token
    # corresponds to one target class; the model returns open-vocabulary
    # detections for anything matching these prompts.

    grounding_dino_box_threshold:  float = 0.30
    # Minimum box-regression confidence to retain a predicted box.

    grounding_dino_text_threshold: float = 0.25
    # Minimum text-alignment score for the predicted label.  Setting this
    # lower than box_threshold accepts more label uncertainty while still
    # requiring a confident spatial localisation.

    # ── RF-DETR settings ─────────────────────────────────────────────────
    rfdetr_model_size: str = "rfdetr_base"
    # "rfdetr_base"  — ResNet-50 backbone, fast inference
    # "rfdetr_large" — larger backbone, higher mAP

    rfdetr_confidence_threshold: float = 0.30
    # Per-box confidence gate applied to RF-DETR predictions before NMS.

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ObjectDetectorConfig("
            f"frame={self.frame_shape[0]}×{self.frame_shape[1]}, "
            f"strategy={self.detection_strategy!r}, "
            f"conf_thresh={self.confidence_threshold:.2f}, "
            f"nms_iou={self.nms_iou_threshold:.2f}, "
            f"max_det={self.max_detections})"
        )


# ===========================================================================
# SimpleIoUTracker
# ===========================================================================

class SimpleIoUTracker:
    """Frame-to-frame bounding-box identity tracker using IoU matching.

    Assigns persistent ``track_id`` values to detections by matching each
    new box against the set of active tracks from the previous frame using
    Intersection-over-Union (IoU) overlap.  No motion model is used —
    Kalman-filter prediction is deferred to Phase 6.

    Algorithm (per :meth:`update` call)
    ------------------------------------
    1. For each new detection, compute IoU with every active track.
    2. Assign the track's ``track_id`` to the detection whose IoU exceeds
       ``iou_threshold`` and is the maximum across all candidates
       (greedy matching, highest-IoU-first).
    3. Unmatched detections receive a new unique ``track_id``.
    4. Tracks that have not been matched for more than ``max_lost_frames``
       consecutive frames are pruned from the active set.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU required to associate a detection with a track.
        Range (0, 1].  Lower values allow looser associations across
        larger inter-frame displacements.
    max_lost_frames : int
        Number of consecutive frames a track can remain unmatched before
        it is removed.  Larger values tolerate occlusions at the cost of
        maintaining stale track state longer.
    """

    def __init__(
        self,
        iou_threshold:   float = 0.3,
        max_lost_frames: int   = 5,
    ) -> None:
        """Initialise tracker with empty state.

        Parameters
        ----------
        iou_threshold : float
            Minimum IoU for detection-to-track association.
        max_lost_frames : int
            Frames before a lost track is pruned.
        """
        self._iou_threshold:   float = iou_threshold
        self._max_lost_frames: int   = max_lost_frames

        # Active tracks: {track_id: {"box": BoundingBox, "lost": int}}
        self._tracks: Dict[int, Dict] = {}
        self._next_id: int = 0   # monotonically increasing track counter

        logger.debug(
            "SimpleIoUTracker init: iou_thresh=%.2f, max_lost=%d",
            iou_threshold, max_lost_frames,
        )

    # ------------------------------------------------------------------ #

    def _compute_iou(self, box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute Intersection-over-Union between two bounding boxes.

        IoU = |A ∩ B| / |A ∪ B|

        Parameters
        ----------
        box_a : BoundingBox
            First box.
        box_b : BoundingBox
            Second box.

        Returns
        -------
        float
            IoU value in [0.0, 1.0].  Returns 0.0 when there is no
            overlap or either box has zero area.
        """
        # Intersection rectangle
        inter_x1 = max(box_a.x1, box_b.x1)
        inter_y1 = max(box_a.y1, box_b.y1)
        inter_x2 = min(box_a.x2, box_b.x2)
        inter_y2 = min(box_a.y2, box_b.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0.0:
            return 0.0

        union_area = box_a.area + box_b.area - inter_area

        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area

    # ------------------------------------------------------------------ #

    def update(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """Match detections to active tracks and assign ``track_id`` values.

        Parameters
        ----------
        detections : List[BoundingBox]
            New-frame detections from the detector.  ``track_id`` fields
            are expected to be ``None`` on entry; they are populated
            in-place and also reflected in the returned list.

        Returns
        -------
        List[BoundingBox]
            Same boxes as ``detections``, each with ``track_id`` assigned.
            Order is preserved.

        Notes
        -----
        The greedy matching strategy (highest IoU first) is O(D × T) where
        D = number of detections and T = number of active tracks.  For
        typical ADAS scenes (D, T < 50) this is negligible compared to
        detector inference time.
        """
        if not detections:
            # Increment lost counter for all tracks; prune stale ones
            stale_ids = [
                tid for tid, state in self._tracks.items()
                if state["lost"] >= self._max_lost_frames
            ]
            for tid in stale_ids:
                del self._tracks[tid]
            for state in self._tracks.values():
                state["lost"] += 1
            logger.debug("update: 0 detections — %d tracks remaining", len(self._tracks))
            return detections

        # ── Step 1: Build IoU matrix (D × T) ────────────────────────────
        track_ids:   List[int]          = list(self._tracks.keys())
        track_boxes: List[BoundingBox]  = [self._tracks[t]["box"] for t in track_ids]

        # matched_track[i] = track_id assigned to detections[i], or None
        matched_track: List[Optional[int]] = [None] * len(detections)
        used_tracks:   set                 = set()

        for det_idx, det in enumerate(detections):
            best_iou:      float        = self._iou_threshold   # minimum gate
            best_track_id: Optional[int] = None

            for t_idx, t_box in enumerate(track_boxes):
                if track_ids[t_idx] in used_tracks:
                    continue
                iou = self._compute_iou(det, t_box)
                if iou > best_iou:
                    best_iou      = iou
                    best_track_id = track_ids[t_idx]

            matched_track[det_idx] = best_track_id
            if best_track_id is not None:
                used_tracks.add(best_track_id)

        # ── Step 2: Assign track_ids; create new tracks for unmatched ───
        matched_ids: set = set()
        for det_idx, det in enumerate(detections):
            tid = matched_track[det_idx]
            if tid is not None:
                # Existing track — update box, reset lost counter
                det.track_id = tid
                self._tracks[tid]["box"]  = det
                self._tracks[tid]["lost"] = 0
                matched_ids.add(tid)
            else:
                # New detection — assign fresh track_id
                new_tid = self._next_id
                self._next_id += 1
                det.track_id = new_tid
                self._tracks[new_tid] = {"box": det, "lost": 0}

        # ── Step 3: Increment lost counter; prune stale tracks ───────────
        for tid in list(self._tracks.keys()):
            if tid not in matched_ids:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self._max_lost_frames:
                    del self._tracks[tid]

        logger.debug(
            "update: %d detections, %d active tracks",
            len(detections), len(self._tracks),
        )
        return detections

    def reset(self) -> None:
        """Clear all track state.  Call between independent video sequences.

        Returns
        -------
        None
        """
        self._tracks   = {}
        self._next_id  = 0
        logger.debug("SimpleIoUTracker: state reset")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SimpleIoUTracker("
            f"iou_thresh={self._iou_threshold:.2f}, "
            f"max_lost={self._max_lost_frames}, "
            f"active_tracks={len(self._tracks)})"
        )


# ===========================================================================
# ObjectDetector
# ===========================================================================

class ObjectDetector:
    """YOLO-style object detector with integrated IoU tracker.

    Encapsulates the full per-frame pipeline:
    raw frame → inference (mock or ONNX) → NMS → tracking → DetectionResult.

    Parameters
    ----------
    config : ObjectDetectorConfig
        Full hyperparameter bundle.

    Examples
    --------
    >>> cfg      = ObjectDetectorConfig(mock_mode=True, mock_n_objects=5)
    >>> detector = ObjectDetector(cfg)
    >>> result   = detector.detect(frame)   # frame: H×W×3 BGR uint8
    >>> print(result.n_objects, result.confidence)
    >>> vis = detector.detect_and_draw(frame)
    """

    def __init__(self, config: ObjectDetectorConfig) -> None:
        """Initialise the detector and optionally load the ONNX model.

        Parameters
        ----------
        config : ObjectDetectorConfig
            Hyperparameter bundle.

        Raises
        ------
        TypeError
            If ``config`` is not an :class:`ObjectDetectorConfig` instance.
        ValueError
            If ``mock_mode=False`` and ``onnx_model_path`` is ``None``.
        FileNotFoundError
            If the specified ONNX model file does not exist.
        ImportError
            If ``mock_mode=False`` and ``onnxruntime`` is not installed.
        """
        if not isinstance(config, ObjectDetectorConfig):
            raise TypeError(
                f"config must be an ObjectDetectorConfig instance, "
                f"got {type(config).__name__!r}."
            )

        self.config: ObjectDetectorConfig = config

        # ── ONNX session (live mode only) ────────────────────────────────
        self._ort_session = None

        if not config.mock_mode:
            if config.onnx_model_path is None:
                raise ValueError(
                    "ObjectDetectorConfig.onnx_model_path must be set "
                    "when mock_mode=False."
                )
            import os
            if not os.path.isfile(config.onnx_model_path):
                raise FileNotFoundError(
                    f"ONNX model not found: {config.onnx_model_path!r}"
                )
            try:
                import onnxruntime as ort  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for mock_mode=False. "
                    "Install with: pip install onnxruntime"
                ) from exc
            self._ort_session = ort.InferenceSession(
                config.onnx_model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(
                "ObjectDetector: ONNX model loaded from %r",
                config.onnx_model_path,
            )

        # ── Integrated IoU tracker ───────────────────────────────────────
        # iou_threshold and max_lost_frames are fixed at sensible ADAS
        # defaults here; promote to ObjectDetectorConfig if they need
        # per-deployment tuning in Phase 6.
        self._tracker: SimpleIoUTracker = SimpleIoUTracker(
            iou_threshold=config.nms_iou_threshold,
            max_lost_frames=5,
        )

        # ── YOLOv8 model handle — lazy-loaded on first _yolo_detect call ─
        self._yolo_model = None   # ultralytics.YOLO | None

        # ── Additional model caches — all lazy-loaded on first use ───────
        self._rtdetr_model        = None   # ultralytics.RTDETR | None
        self._grounding_processor = None   # transformers.AutoProcessor | None
        self._grounding_model     = None   # transformers.AutoModelForZeroShotObjectDetection | None
        self._rfdetr_model        = None   # rfdetr.RFDETRBase / RFDETRLarge | None

        # ── Strategy validation ──────────────────────────────────────────
        _valid_strategies = {
            "mock", "yolo", "onnx", "ensemble",
            "rtdetr", "grounding", "rfdetr",
        }
        if config.detection_strategy not in _valid_strategies:
            raise ValueError(
                f"detection_strategy must be one of {_valid_strategies}, "
                f"got {config.detection_strategy!r}."
            )

        logger.debug(
            "ObjectDetector initialised — strategy=%r. %r",
            config.detection_strategy, self.config,
        )

    # ------------------------------------------------------------------ #
    # Mock backend
    # ------------------------------------------------------------------ #

    def _mock_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Generate deterministic synthetic bounding boxes for testing.

        Produces :attr:`~ObjectDetectorConfig.mock_n_objects` boxes whose
        positions are seeded by the frame content hash
        (``np.sum(frame) % 1000``), guaranteeing that the same frame always
        yields the same detections regardless of call order.

        Box geometry
        ------------
        - Centres are uniformly distributed across the frame area.
        - Width and height are independently sampled from
          U[0.05·W, 0.30·W] and U[0.05·H, 0.30·H] respectively,
          approximating the distribution of real vehicle and pedestrian
          boxes in a forward-facing ADAS camera.
        - Gaussian noise (σ = ``config.mock_noise_std_px``) is added to
          all four edge coordinates to simulate detection jitter.
        - All coordinates are clipped to the valid frame extent.

        Parameters
        ----------
        frame : BgrImage
            Raw BGR frame.  Only its shape and sum (for seeding) are used.

        Returns
        -------
        List[BoundingBox]
            Synthetic boxes, length == ``config.mock_n_objects``.
            All ``confidence`` values are at least
            ``config.confidence_threshold``.
        """
        H, W = frame.shape[0], frame.shape[1]
        n    = self.config.mock_n_objects
        n_classes = len(self.config.class_names)

        # Deterministic seed from frame content so the same image always
        # produces the same synthetic detections.  We fold the full 64-bit
        # sum into a 32-bit seed (NumPy Generator requires int, not uint64)
        # using XOR-folding to preserve bit diversity even on simple frames.
        raw_sum: int   = int(np.sum(frame.astype(np.int64)))
        seed:    int   = (raw_sum ^ (raw_sum >> 16)) & 0xFFFF_FFFF
        rng = np.random.default_rng(seed)

        # ── Sample box centres uniformly across the frame ────────────────
        # Centres kept in the central 80% of each axis to avoid boxes that
        # are mostly outside the frame before clipping.
        margin_ratio: float = 0.10
        cx: NDArray[np.float64] = rng.uniform(
            margin_ratio * W, (1.0 - margin_ratio) * W, size=n
        )
        cy: NDArray[np.float64] = rng.uniform(
            margin_ratio * H, (1.0 - margin_ratio) * H, size=n
        )

        # ── Sample box sizes (fraction of frame dimension) ───────────────
        # These ratios approximate real ADAS object size distributions.
        w_min_ratio: float = 0.05
        w_max_ratio: float = 0.30
        h_min_ratio: float = 0.05
        h_max_ratio: float = 0.30

        half_w: NDArray[np.float64] = rng.uniform(
            w_min_ratio * W * 0.5, w_max_ratio * W * 0.5, size=n
        )
        half_h: NDArray[np.float64] = rng.uniform(
            h_min_ratio * H * 0.5, h_max_ratio * H * 0.5, size=n
        )

        # ── Add Gaussian noise to simulate detection jitter ──────────────
        noise: NDArray[np.float64] = rng.normal(
            0.0, self.config.mock_noise_std_px, size=(n, 4)
        )

        # ── Sample class_ids and confidences ─────────────────────────────
        class_ids:   NDArray[np.int64]   = rng.integers(0, n_classes, size=n)
        # Confidence uniformly above config.confidence_threshold
        conf_range: float = 1.0 - self.config.confidence_threshold
        confidences: NDArray[np.float64] = (
            self.config.confidence_threshold
            + rng.uniform(0.0, conf_range, size=n)
        )

        # ── Assemble BoundingBox list ─────────────────────────────────────
        boxes: List[BoundingBox] = []
        for i in range(n):
            x1 = float(np.clip(cx[i] - half_w[i] + noise[i, 0], 0.0, W - 1.0))
            y1 = float(np.clip(cy[i] - half_h[i] + noise[i, 1], 0.0, H - 1.0))
            x2 = float(np.clip(cx[i] + half_w[i] + noise[i, 2], 0.0, W - 1.0))
            y2 = float(np.clip(cy[i] + half_h[i] + noise[i, 3], 0.0, H - 1.0))

            # Skip degenerate boxes (zero width or height after clipping)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
            ))

        logger.debug(
            "_mock_detect: generated %d boxes (seed=%d)", len(boxes), seed
        )
        return boxes

    # ------------------------------------------------------------------ #
    # ONNX backend (stub)
    # ------------------------------------------------------------------ #

    def _onnx_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Run YOLO inference via ONNX Runtime.

        Parameters
        ----------
        frame : BgrImage
            Raw BGR camera frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Detected boxes before NMS, in descending confidence order.

        Raises
        ------
        NotImplementedError
            Always raised until a trained YOLO ONNX model is integrated.
            To enable real inference:

            1. Export YOLOv8/YOLOv5 to ONNX (``model.export(format='onnx')``).
            2. Set ``ObjectDetectorConfig(mock_mode=False,
               onnx_model_path='path/to/model.onnx')``.
            3. Implement the output-tensor → :class:`BoundingBox` mapping
               for your model's specific output head format in this method.
        """
        raise NotImplementedError(
            "_onnx_detect() is not yet implemented. "
            "To enable real ONNX inference:\n"
            "  1. Export a YOLO model: model.export(format='onnx')\n"
            "  2. Set ObjectDetectorConfig(mock_mode=False, "
            "onnx_model_path='path/to/yolo.onnx')\n"
            "  3. Implement the output tensor → BoundingBox mapping "
            "in this method.\n"
            "See: https://docs.ultralytics.com/modes/export/"
        )

    # ------------------------------------------------------------------ #
    # YOLOv8 backend
    # ------------------------------------------------------------------ #

    def _yolo_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Run YOLOv8 inference using the ``ultralytics`` library.

        The model is loaded lazily on the first call and cached in
        ``self._yolo_model``; subsequent calls reuse the loaded weights
        with no I/O overhead.

        Algorithm
        ---------
        1. Lazy-import ``ultralytics`` and load
           ``<config.yolo_model_size>.pt`` on the first invocation.
        2. Run ``model.predict()`` with ``conf=config.yolo_confidence_threshold``
           and ``verbose=False``.
        3. Discard any predicted class name not present in
           ``config.yolo_road_classes``.
        4. Map surviving class names → NeuroDrive ``class_id`` via
           ``config.yolo_class_map``.
        5. Return :class:`BoundingBox` list; NMS is applied downstream.

        Parameters
        ----------
        frame : BgrImage
            BGR frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Raw YOLO detections filtered by road-relevant classes and
            ``config.yolo_confidence_threshold``.  May contain overlapping
            boxes — NMS is applied by the caller.

        Raises
        ------
        ImportError
            If ``ultralytics`` is not installed.
            Install with: ``pip install ultralytics``
        """
        logger.debug(
            "_yolo_detect: strategy=yolo, model_size=%r",
            self.config.yolo_model_size,
        )

        # ── Lazy model load ──────────────────────────────────────────────
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for detection_strategy='yolo'. "
                    "Install with: pip install ultralytics"
                ) from exc

            model_filename: str = self.config.yolo_model_size + ".pt"
            self._yolo_model = YOLO(model_filename)
            logger.info("_yolo_detect: loaded model %r", model_filename)

        # ── Inference ────────────────────────────────────────────────────
        # verbose=False suppresses ultralytics' per-frame console output,
        # which is inappropriate in a production logging environment.
        results = self._yolo_model.predict(
            frame,
            conf=self.config.yolo_confidence_threshold,
            verbose=False,
        )

        boxes: List[BoundingBox] = []

        # ultralytics returns a list of Results (one per image); we pass a
        # single frame so index 0 is the only result.
        for detection in results[0].boxes:
            coco_class_name: str = self._yolo_model.names[
                int(detection.cls[0])
            ]

            # ── Road-class filter ────────────────────────────────────────
            if coco_class_name not in self.config.yolo_road_classes:
                continue

            conf: float = float(detection.conf[0])

            # Belt-and-suspenders confidence gate (ultralytics already
            # filters at predict time, but conf threshold may differ)
            if conf < self.config.yolo_confidence_threshold:
                continue

            # ── Class remapping ──────────────────────────────────────────
            class_id: int = self.config.yolo_class_map.get(coco_class_name, 0)
            if coco_class_name not in self.config.yolo_class_map:
                logger.warning(
                    "_yolo_detect: class %r absent from yolo_class_map — "
                    "defaulting to class_id=0",
                    coco_class_name,
                )

            # ── Extract pixel coordinates (xyxy format) ──────────────────
            x1, y1, x2, y2 = (float(v) for v in detection.xyxy[0])

            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=class_id,
            ))

        logger.debug("_yolo_detect: %d road-relevant boxes", len(boxes))
        return boxes

    # ------------------------------------------------------------------ #
    # Ensemble backend
    # ------------------------------------------------------------------ #

    def _ensemble_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Merge mock and YOLO detections before NMS.

        Demonstrates detection-level sensor fusion: two independent
        hypotheses are pooled so that downstream NMS arbitrates between
        them by spatial IoU, retaining the highest-confidence box in
        each image region.

        This is a simple concatenation ensemble.  Weighted Box Fusion
        (WBF) can replace NMS in Phase 6 for smoother coordinate blending.

        Algorithm
        ---------
        1. ``_mock_detect(frame)`` → ``mock_boxes``
        2. ``_yolo_detect(frame)`` → ``yolo_boxes``
        3. Return ``mock_boxes + yolo_boxes`` — NMS applied by caller.

        Parameters
        ----------
        frame : BgrImage
            BGR frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Combined raw detections from both backends before NMS.
        """
        logger.debug("_ensemble_detect: running mock + yolo backends")

        mock_boxes: List[BoundingBox] = self._mock_detect(frame)
        logger.debug("_ensemble_detect: mock → %d boxes", len(mock_boxes))

        yolo_boxes: List[BoundingBox] = self._yolo_detect(frame)
        logger.debug("_ensemble_detect: yolo → %d boxes", len(yolo_boxes))

        combined: List[BoundingBox] = mock_boxes + yolo_boxes
        logger.debug(
            "_ensemble_detect: %d combined boxes pre-NMS", len(combined)
        )
        return combined

    # ------------------------------------------------------------------ #
    # RT-DETR backend
    # ------------------------------------------------------------------ #

    def _rtdetr_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Run RT-DETR inference via the ``ultralytics`` library.

        RT-DETR (Real-Time Detection Transformer) is an end-to-end
        transformer-based detector that does not require NMS internally.
        The ``ultralytics`` wrapper exposes it through the same
        ``.predict()`` API as YOLOv8, so the road-class filter and
        ``yolo_class_map`` remapping are identical to :meth:`_yolo_detect`.

        The model is loaded lazily on the first call and cached in
        ``self._rtdetr_model``.

        Parameters
        ----------
        frame : BgrImage
            BGR frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Detections filtered by ``config.yolo_road_classes`` and
            ``config.yolo_confidence_threshold``.

        Raises
        ------
        ImportError
            If ``ultralytics`` is not installed.
            Install with: ``pip install ultralytics``
        """
        logger.debug(
            "_rtdetr_detect: strategy=rtdetr, model_size=%r",
            self.config.rtdetr_model_size,
        )

        # ── Lazy model load ──────────────────────────────────────────────
        if self._rtdetr_model is None:
            try:
                from ultralytics import RTDETR  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for detection_strategy='rtdetr'. "
                    "Install with: pip install ultralytics"
                ) from exc

            model_filename: str = self.config.rtdetr_model_size + ".pt"
            self._rtdetr_model = RTDETR(model_filename)
            logger.info("_rtdetr_detect: loaded model %r", model_filename)

        # ── Inference ────────────────────────────────────────────────────
        results = self._rtdetr_model.predict(
            frame,
            conf=self.config.yolo_confidence_threshold,
            verbose=False,
        )

        boxes: List[BoundingBox] = []

        for detection in results[0].boxes:
            coco_class_name: str = self._rtdetr_model.names[
                int(detection.cls[0])
            ]

            # ── Road-class filter (reuses YOLO config) ───────────────────
            if coco_class_name not in self.config.yolo_road_classes:
                continue

            conf: float = float(detection.conf[0])
            if conf < self.config.yolo_confidence_threshold:
                continue

            # ── Class remapping (reuses yolo_class_map) ──────────────────
            class_id: int = self.config.yolo_class_map.get(coco_class_name, 0)
            if coco_class_name not in self.config.yolo_class_map:
                logger.warning(
                    "_rtdetr_detect: class %r absent from yolo_class_map — "
                    "defaulting to class_id=0",
                    coco_class_name,
                )

            x1, y1, x2, y2 = (float(v) for v in detection.xyxy[0])
            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=class_id,
            ))

        logger.debug("_rtdetr_detect: %d road-relevant boxes", len(boxes))
        return boxes

    # ------------------------------------------------------------------ #
    # Grounding DINO backend
    # ------------------------------------------------------------------ #

    def _grounding_label_to_class_id(self, label: str) -> int:
        """Map a Grounding DINO text label to a NeuroDrive class_id.

        Uses substring matching so that labels like ``"a car"`` or
        ``"parked car"`` correctly map to class 0, without requiring an
        exact string match.

        Parameters
        ----------
        label : str
            Raw label string returned by the Grounding DINO model.

        Returns
        -------
        int
            NeuroDrive class_id in [0, 4], or ``0`` (car) as default
            for unrecognised labels.
        """
        # Mapping is declared as a local constant (not in config) because
        # it encodes the semantic relationship between Grounding DINO
        # vocabulary and NeuroDrive class IDs — a fixed ontology, not a
        # tunable threshold.
        _label_map: Dict[str, int] = {
            "car":         0,
            "truck":       1,
            "bus":         1,
            "person":      2,
            "pedestrian":  2,
            "bicycle":     3,
            "cyclist":     3,
            "motorcycle":  4,
        }
        label_lower: str = label.lower().strip()
        for key, cid in _label_map.items():
            if key in label_lower:
                return cid
        logger.debug(
            "_grounding_label_to_class_id: unrecognised label %r → class_id=0",
            label,
        )
        return 0

    def _grounding_dino_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Run zero-shot object detection via Grounding DINO.

        Grounding DINO accepts a free-form text prompt and localises
        all matching objects in the image without class-specific training.
        This enables detection of any road-relevant object expressible in
        natural language, beyond the fixed COCO vocabulary of YOLO.

        Algorithm
        ---------
        1. Lazy-import ``transformers`` and load the processor and model
           from ``config.grounding_dino_model`` (HuggingFace Hub ID).
        2. Convert the BGR frame to a PIL RGB image.
        3. Encode frame + ``config.grounding_dino_text_prompt`` through
           the processor.
        4. Run model inference and apply ``post_process_grounded_object_detection``
           with ``config.grounding_dino_box_threshold`` and
           ``config.grounding_dino_text_threshold``.
        5. Convert predicted boxes from ``[cx, cy, w, h]`` normalised
           format to pixel ``xyxy`` coordinates.
        6. Map label strings to NeuroDrive class_ids via
           :meth:`_grounding_label_to_class_id`.

        Parameters
        ----------
        frame : BgrImage
            BGR frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Zero-shot detections for the text prompt, in pixel xyxy
            coordinates.  NMS is applied downstream.

        Raises
        ------
        ImportError
            If ``transformers`` or ``Pillow`` is not installed.
            Install with: ``pip install transformers Pillow``
        """
        logger.debug(
            "_grounding_dino_detect: strategy=grounding, model=%r",
            self.config.grounding_dino_model,
        )

        # ── Lazy model + processor load ──────────────────────────────────
        if self._grounding_processor is None or self._grounding_model is None:
            try:
                from transformers import (   # type: ignore[import]
                    AutoProcessor,
                    AutoModelForZeroShotObjectDetection,
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for detection_strategy='grounding'. "
                    "Install with: pip install transformers"
                ) from exc

            self._grounding_processor = AutoProcessor.from_pretrained(
                self.config.grounding_dino_model
            )
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_dino_model
            )
            logger.info(
                "_grounding_dino_detect: loaded model %r",
                self.config.grounding_dino_model,
            )

        # ── BGR → PIL RGB ────────────────────────────────────────────────
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for detection_strategy='grounding'. "
                "Install with: pip install Pillow"
            ) from exc

        H: int = frame.shape[0]
        W: int = frame.shape[1]
        # cv2 stores BGR; PIL expects RGB — reverse channel order
        pil_image = Image.fromarray(frame[:, :, ::-1])

        # ── Encode inputs ────────────────────────────────────────────────
        inputs = self._grounding_processor(
            images=pil_image,
            text=self.config.grounding_dino_text_prompt,
            return_tensors="pt",
        )

        # ── Inference ────────────────────────────────────────────────────
        import torch  # type: ignore[import]
        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        # ── Post-process ─────────────────────────────────────────────────
        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.config.grounding_dino_box_threshold,
            text_threshold=self.config.grounding_dino_text_threshold,
            target_sizes=[(H, W)],
        )[0]   # index 0 = single image

        boxes: List[BoundingBox] = []

        for box_tensor, score_tensor, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            conf: float = float(score_tensor)
            if conf < self.config.grounding_dino_box_threshold:
                continue

            # boxes from post_process are already in pixel xyxy — the
            # raw model output is cxcywh normalised, but post_process
            # converts it.  Verify and clip to frame bounds.
            x1 = float(max(0.0, box_tensor[0]))
            y1 = float(max(0.0, box_tensor[1]))
            x2 = float(min(float(W), box_tensor[2]))
            y2 = float(min(float(H), box_tensor[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            class_id: int = self._grounding_label_to_class_id(str(label))
            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=class_id,
            ))

        logger.debug(
            "_grounding_dino_detect: %d boxes detected", len(boxes)
        )
        return boxes

    # ------------------------------------------------------------------ #
    # RF-DETR backend
    # ------------------------------------------------------------------ #

    def _rfdetr_detect(self, frame: BgrImage) -> List[BoundingBox]:
        """Run RF-DETR inference via the ``rfdetr`` package.

        RF-DETR (ResNet Feature-pyramid DETR) is a real-time transformer
        detector that uses a ResNet backbone with a DETR decoder.  The
        ``rfdetr`` package exposes ``RFDETRBase`` and ``RFDETRLarge``
        classes whose ``.predict()`` method accepts a PIL image and returns
        a result object with ``.labels``, ``.boxes`` (xyxy pixel coords),
        and ``.scores``.

        The model is loaded lazily on the first call and cached in
        ``self._rfdetr_model``.

        Parameters
        ----------
        frame : BgrImage
            BGR frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        List[BoundingBox]
            Detections filtered by ``config.rfdetr_confidence_threshold``
            and remapped via ``config.yolo_class_map``.

        Raises
        ------
        ImportError
            If the ``rfdetr`` package or ``Pillow`` is not installed.
            Install with: ``pip install rfdetr Pillow``
        """
        logger.debug(
            "_rfdetr_detect: strategy=rfdetr, model_size=%r",
            self.config.rfdetr_model_size,
        )

        # ── Lazy model load ──────────────────────────────────────────────
        if self._rfdetr_model is None:
            try:
                from rfdetr import RFDETRBase, RFDETRLarge  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "rfdetr is required for detection_strategy='rfdetr'. "
                    "Install with: pip install rfdetr"
                ) from exc

            # Select model class by configured size
            _large_key: str = "rfdetr_large"
            if self.config.rfdetr_model_size == _large_key:
                self._rfdetr_model = RFDETRLarge()
            else:
                self._rfdetr_model = RFDETRBase()

            logger.info(
                "_rfdetr_detect: loaded model size=%r",
                self.config.rfdetr_model_size,
            )

        # ── BGR → PIL RGB ────────────────────────────────────────────────
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for detection_strategy='rfdetr'. "
                "Install with: pip install Pillow"
            ) from exc

        H: int = frame.shape[0]
        W: int = frame.shape[1]
        pil_image = Image.fromarray(frame[:, :, ::-1])   # BGR → RGB

        # ── Inference ────────────────────────────────────────────────────
        result = self._rfdetr_model.predict(
            pil_image,
            threshold=self.config.rfdetr_confidence_threshold,
        )

        boxes: List[BoundingBox] = []

        for label_str, box_coords, score in zip(
            result.labels, result.boxes, result.scores
        ):
            conf: float = float(score)
            if conf < self.config.rfdetr_confidence_threshold:
                continue

            # ── Class remapping via yolo_class_map ───────────────────────
            class_id: int = self.config.yolo_class_map.get(str(label_str), 0)
            if str(label_str) not in self.config.yolo_class_map:
                logger.debug(
                    "_rfdetr_detect: label %r absent from yolo_class_map — "
                    "defaulting to class_id=0",
                    label_str,
                )

            # ── Box coordinates — xyxy pixel format ──────────────────────
            x1 = float(max(0.0,       box_coords[0]))
            y1 = float(max(0.0,       box_coords[1]))
            x2 = float(min(float(W),  box_coords[2]))
            y2 = float(min(float(H),  box_coords[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=class_id,
            ))

        logger.debug("_rfdetr_detect: %d boxes detected", len(boxes))
        return boxes

    # ------------------------------------------------------------------ #
    # Non-Maximum Suppression
    # ------------------------------------------------------------------ #

    def _apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Remove duplicate detections using greedy IoU-based NMS.

        Algorithm
        ---------
        1. Filter boxes below ``config.confidence_threshold``.
        2. Sort remaining boxes by confidence descending.
        3. Iterate: keep the current highest-confidence box; suppress
           any subsequent box whose IoU with it exceeds
           ``config.nms_iou_threshold``.
        4. Truncate to ``config.max_detections``.

        Parameters
        ----------
        boxes : List[BoundingBox]
            Raw detector output, potentially containing duplicates.

        Returns
        -------
        List[BoundingBox]
            Deduplicated boxes, at most ``config.max_detections`` entries,
            sorted by confidence descending.
        """
        # ── Step 1: Confidence gate ──────────────────────────────────────
        filtered: List[BoundingBox] = [
            b for b in boxes
            if b.confidence >= self.config.confidence_threshold
        ]

        if not filtered:
            logger.debug("_apply_nms: no boxes above confidence threshold")
            return []

        # ── Step 2: Sort by confidence descending ────────────────────────
        filtered.sort(key=lambda b: b.confidence, reverse=True)

        # ── Step 3: Greedy suppression ───────────────────────────────────
        # Reuse the tracker's IoU computation for consistency.
        kept:      List[BoundingBox] = []
        suppressed: set              = set()

        for i, box_i in enumerate(filtered):
            if i in suppressed:
                continue
            kept.append(box_i)
            for j in range(i + 1, len(filtered)):
                if j in suppressed:
                    continue
                iou = self._tracker._compute_iou(box_i, filtered[j])
                if iou > self.config.nms_iou_threshold:
                    suppressed.add(j)

        # ── Step 4: Cap output count ─────────────────────────────────────
        result: List[BoundingBox] = kept[: self.config.max_detections]

        logger.debug(
            "_apply_nms: %d → %d boxes (suppressed=%d, capped=%d)",
            len(boxes), len(result),
            len(filtered) - len(kept),
            max(0, len(kept) - self.config.max_detections),
        )
        return result

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #

    def detect(self, frame: BgrImage) -> DetectionResult:
        """Run the full object detection pipeline on one frame.

        Pipeline::

            frame → _mock_detect / _onnx_detect
                  → _apply_nms
                  → tracker.update
                  → DetectionResult

        Parameters
        ----------
        frame : BgrImage
            Single BGR video frame, shape (H, W, 3), dtype ``uint8``.

        Returns
        -------
        DetectionResult
            Populated result container.  ``result.valid`` is ``True``
            when at least one object was detected.

        Raises
        ------
        ValueError
            If ``frame`` is not a 3-channel uint8 BGR image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"detect() expects a 3-channel uint8 BGR image (H, W, 3), "
                f"got shape={frame.shape}, dtype={frame.dtype}."
            )

        H, W = frame.shape[0], frame.shape[1]

        # ── Stage 1: Inference — route by detection_strategy ────────────
        strategy: str = self.config.detection_strategy
        if strategy == "mock":
            raw_boxes: List[BoundingBox] = self._mock_detect(frame)
        elif strategy == "yolo":
            raw_boxes = self._yolo_detect(frame)
        elif strategy == "onnx":
            raw_boxes = self._onnx_detect(frame)
        elif strategy == "ensemble":
            raw_boxes = self._ensemble_detect(frame)
        elif strategy == "rtdetr":
            raw_boxes = self._rtdetr_detect(frame)
        elif strategy == "grounding":
            raw_boxes = self._grounding_dino_detect(frame)
        elif strategy == "rfdetr":
            raw_boxes = self._rfdetr_detect(frame)
        else:
            # Unreachable: __init__ validates strategy at construction time.
            raise ValueError(
                f"Unknown detection_strategy {strategy!r}. "
                "This should have been caught in __init__."
            )

        logger.debug("detect: %d raw boxes from inference", len(raw_boxes))

        # ── Stage 2: NMS ─────────────────────────────────────────────────
        nms_boxes: List[BoundingBox] = self._apply_nms(raw_boxes)
        logger.debug("detect: %d boxes after NMS", len(nms_boxes))

        # ── Stage 3: Tracking ────────────────────────────────────────────
        tracked_boxes: List[BoundingBox] = self._tracker.update(nms_boxes)
        logger.debug("detect: %d boxes after tracking", len(tracked_boxes))

        # ── Stage 4: Assemble result ─────────────────────────────────────
        n_objects: int   = len(tracked_boxes)
        mean_conf: float = (
            float(np.mean([b.confidence for b in tracked_boxes]))
            if tracked_boxes else 0.0
        )
        valid: bool = n_objects > 0

        return DetectionResult(
            boxes       = tracked_boxes,
            frame_shape = (H, W),
            confidence  = mean_conf,
            valid       = valid,
            n_objects   = n_objects,
        )

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def detect_and_draw(self, frame: BgrImage) -> NDArray[np.uint8]:
        """Run detection and render bounding boxes on a copy of the frame.

        Each box is drawn with:
        - A filled rectangle border in the class-specific colour.
        - A label showing ``class_name  conf  #track_id`` in the same
          colour, positioned above the box (or inside if near the top edge).

        Colour map (BGR)
        ----------------
        car=blue, truck=green, pedestrian=red, cyclist=yellow,
        motorcycle=magenta.

        Parameters
        ----------
        frame : BgrImage
            Input BGR frame, shape (H, W, 3), dtype ``uint8``.
            The original array is **not** modified.

        Returns
        -------
        NDArray[np.uint8]
            Annotated BGR frame copy, same shape and dtype as input.

        Raises
        ------
        ValueError
            If ``frame`` is not a 3-channel uint8 image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"detect_and_draw() expects a 3-channel uint8 BGR image, "
                f"got shape={frame.shape}, dtype={frame.dtype}."
            )

        result: DetectionResult = self.detect(frame)
        canvas: NDArray[np.uint8] = frame.copy()

        # Visual constants — kept here as named locals (not in config) since
        # they control rendering aesthetics, not algorithm behaviour.
        box_thickness:    int   = 2      # px
        font_scale:       float = 0.55
        font_thickness:   int   = 1
        font_face:        int   = cv2.FONT_HERSHEY_SIMPLEX
        label_pad_px:     int   = 4      # padding between label text and box
        label_bg_alpha:   float = 0.55   # opacity of label background rectangle

        for box in result.boxes:
            # ── Colour selection ─────────────────────────────────────────
            cid = box.class_id
            color: Tuple[int, int, int] = (
                _CLASS_COLORS_BGR[cid]
                if 0 <= cid < len(_CLASS_COLORS_BGR)
                else (128, 128, 128)   # grey fallback for unknown class_id
            )

            # ── Bounding box ──────────────────────────────────────────────
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(canvas, pt1, pt2, color, box_thickness)

            # ── Label string ──────────────────────────────────────────────
            track_str: str = f"#{box.track_id}" if box.track_id is not None else ""
            label: str     = f"{box.class_name}  {box.confidence:.2f}  {track_str}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font_face, font_scale, font_thickness
            )

            # Position label above the box; clamp to frame top if needed
            label_y_bottom: int = max(int(box.y1) - label_pad_px, text_h + label_pad_px)
            label_y_top:    int = label_y_bottom - text_h - label_pad_px
            label_x_left:   int = int(box.x1)
            label_x_right:  int = int(box.x1) + text_w + label_pad_px

            # Semi-transparent label background for legibility
            bg_roi = canvas[
                max(0, label_y_top) : label_y_bottom + baseline,
                max(0, label_x_left): min(canvas.shape[1], label_x_right),
            ]
            if bg_roi.size > 0:
                overlay = bg_roi.copy()
                overlay[:] = color
                cv2.addWeighted(overlay, label_bg_alpha, bg_roi, 1.0 - label_bg_alpha, 0, bg_roi)

            cv2.putText(
                canvas,
                label,
                (label_x_left, label_y_bottom - baseline),
                font_face,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

        logger.debug(
            "detect_and_draw: rendered %d boxes on frame %s",
            result.n_objects, frame.shape,
        )
        return canvas

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def reset_tracker(self) -> None:
        """Clear all tracker state between independent video sequences.

        Call this when switching to a new clip or camera stream to avoid
        spurious track identity carry-over from the previous sequence.

        Returns
        -------
        None
        """
        self._tracker.reset()
        logger.debug("ObjectDetector: tracker state reset")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ObjectDetector("
            f"strategy={self.config.detection_strategy!r}, "
            f"config={self.config!r})"
        )
