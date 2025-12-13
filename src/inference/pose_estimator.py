"""
Pose Estimation Module for Real-Time Inference.

Provides unified interface for multiple pose estimation backends:
- YOLOv8-Pose: Fast, integrated detection + pose
- RTMPose: High accuracy, SOTA performance
- MediaPipe: Lightweight, good for mobile/edge
- OpenPose: Classic, well-documented

Usage:
    from src.inference import PoseEstimator
    
    # YOLOv8-Pose (recommended for real-time)
    estimator = PoseEstimator(backend='yolov8', model_size='m')
    
    # Process frame
    detections = estimator.estimate(frame)
    # detections: List[Dict] with 'bbox', 'keypoints', 'confidence'
    
    # Batch processing
    batch_detections = estimator.estimate_batch(frames)

References:
- YOLOv8-Pose: https://docs.ultralytics.com/tasks/pose/
- RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- MediaPipe: https://google.github.io/mediapipe/solutions/pose.html
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Keypoint Definitions
# =============================================================================

class KeypointFormat(Enum):
    """Supported keypoint formats."""
    COCO = 'coco'           # 17 keypoints
    COCO_WHOLEBODY = 'coco_wholebody'  # 133 keypoints
    MPII = 'mpii'           # 16 keypoints
    HALPE = 'halpe'         # 26 keypoints
    JTA = 'jta'             # 22 keypoints


# COCO keypoint indices (17 keypoints)
COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# COCO skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

# JTA to COCO mapping (for compatibility)
JTA_TO_COCO = {
    0: 0,   # Head -> nose
    1: 5,   # Shoulders -> left_shoulder (approx)
    2: 6,   # Shoulders -> right_shoulder (approx)
    # ... (complete mapping would be added)
}


@dataclass
class Detection:
    """Single person detection with pose."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    keypoints: np.ndarray     # [num_keypoints, 3] (x, y, confidence)
    confidence: float         # Detection confidence
    track_id: Optional[int] = None  # Tracking ID if available
    
    @property
    def center(self) -> np.ndarray:
        """Get bounding box center."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]
    
    def get_pelvis(self) -> np.ndarray:
        """Get pelvis position (midpoint of hips)."""
        if self.keypoints.shape[0] >= 17:  # COCO format
            left_hip = self.keypoints[11, :2]
            right_hip = self.keypoints[12, :2]
            return (left_hip + right_hip) / 2
        return self.center
    
    def get_keypoint(self, name: str) -> Optional[np.ndarray]:
        """Get keypoint by name."""
        if name in COCO_KEYPOINTS:
            idx = COCO_KEYPOINTS[name]
            if idx < self.keypoints.shape[0]:
                return self.keypoints[idx]
        return None


# =============================================================================
# Base Pose Estimator
# =============================================================================

class BasePoseEstimator:
    """Base class for pose estimators."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        device: str = 'cuda',
    ):
        """
        Initialize pose estimator.
        
        Args:
            confidence_threshold: Minimum detection confidence
            keypoint_threshold: Minimum keypoint confidence
            device: Compute device ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
        self.device = device
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize model (lazy loading)."""
        raise NotImplementedError
    
    def estimate(self, frame: np.ndarray) -> List[Detection]:
        """
        Estimate poses in a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            List of Detection objects
        """
        raise NotImplementedError
    
    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Estimate poses in multiple frames.
        
        Args:
            frames: List of BGR images
        
        Returns:
            List of detection lists
        """
        return [self.estimate(frame) for frame in frames]
    
    def warmup(self, input_size: Tuple[int, int] = (640, 480)):
        """Warmup model with dummy input."""
        if not self._initialized:
            self.initialize()
        
        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        _ = self.estimate(dummy)
        logger.info("Pose estimator warmup complete")


# =============================================================================
# YOLOv8-Pose Estimator
# =============================================================================

class YOLOv8PoseEstimator(BasePoseEstimator):
    """
    YOLOv8-Pose estimator.
    
    Provides integrated detection + pose estimation in a single model.
    Best balance of speed and accuracy for real-time applications.
    
    Model sizes:
    - 'n': Nano (fastest, lowest accuracy)
    - 's': Small
    - 'm': Medium (recommended)
    - 'l': Large
    - 'x': XLarge (slowest, highest accuracy)
    """
    
    def __init__(
        self,
        model_size: str = 'm',
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        device: str = 'cuda',
        half: bool = True,
    ):
        """
        Initialize YOLOv8-Pose.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            model_path: Path to custom model weights
            confidence_threshold: Detection confidence threshold
            keypoint_threshold: Keypoint confidence threshold
            device: Compute device
            half: Use FP16 inference
        """
        super().__init__(confidence_threshold, keypoint_threshold, device)
        
        self.model_size = model_size
        self.model_path = model_path
        self.half = half and device == 'cuda'
    
    def initialize(self):
        """Initialize YOLOv8-Pose model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLOv8 not installed. Install with: pip install ultralytics"
            )
        
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(f'yolov8{self.model_size}-pose.pt')
        
        # Set device
        self.model.to(self.device)
        
        if self.half:
            self.model.model.half()
        
        self._initialized = True
        logger.info(f"YOLOv8-Pose ({self.model_size}) initialized on {self.device}")
    
    def estimate(self, frame: np.ndarray) -> List[Detection]:
        """Estimate poses using YOLOv8-Pose."""
        if not self._initialized:
            self.initialize()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        
        detections = []
        
        if results.keypoints is not None and len(results.keypoints) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            keypoints = results.keypoints.data.cpu().numpy()
            
            for i in range(len(boxes)):
                detection = Detection(
                    bbox=boxes[i],
                    keypoints=keypoints[i],  # [17, 3]
                    confidence=float(confs[i]),
                )
                detections.append(detection)
        
        return detections
    
    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference for multiple frames."""
        if not self._initialized:
            self.initialize()
        
        # Run batch inference
        results_list = self.model(
            frames,
            conf=self.confidence_threshold,
            verbose=False,
        )
        
        all_detections = []
        
        for results in results_list:
            detections = []
            
            if results.keypoints is not None and len(results.keypoints) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                keypoints = results.keypoints.data.cpu().numpy()
                
                for i in range(len(boxes)):
                    detection = Detection(
                        bbox=boxes[i],
                        keypoints=keypoints[i],
                        confidence=float(confs[i]),
                    )
                    detections.append(detection)
            
            all_detections.append(detections)
        
        return all_detections


# =============================================================================
# RTMPose Estimator
# =============================================================================

class RTMPoseEstimator(BasePoseEstimator):
    """
    RTMPose estimator from MMPose.
    
    State-of-the-art accuracy with competitive speed.
    Uses top-down approach: detection first, then pose estimation.
    
    Model variants:
    - 'rtmpose-t': Tiny
    - 'rtmpose-s': Small
    - 'rtmpose-m': Medium
    - 'rtmpose-l': Large
    """
    
    def __init__(
        self,
        model_variant: str = 'rtmpose-m',
        detector: str = 'rtmdet',
        confidence_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        device: str = 'cuda',
    ):
        """
        Initialize RTMPose.
        
        Args:
            model_variant: RTMPose variant
            detector: Person detector ('rtmdet', 'yolov8', 'faster_rcnn')
            confidence_threshold: Detection threshold
            keypoint_threshold: Keypoint threshold
            device: Compute device
        """
        super().__init__(confidence_threshold, keypoint_threshold, device)
        
        self.model_variant = model_variant
        self.detector_type = detector
        self.detector = None
        self.pose_model = None
    
    def initialize(self):
        """Initialize RTMPose model."""
        try:
            from mmpose.apis import init_model as init_pose_model
            from mmpose.apis import inference_topdown
        except ImportError:
            raise ImportError(
                "MMPose not installed. Install with: "
                "pip install -U openmim && mim install mmpose"
            )
        
        # Initialize detector
        self._init_detector()
        
        # Initialize pose model
        # Note: Actual config paths would depend on installation
        pose_config = f'configs/body_2d_keypoint/{self.model_variant}.py'
        pose_checkpoint = f'checkpoints/{self.model_variant}.pth'
        
        try:
            self.pose_model = init_pose_model(
                pose_config,
                pose_checkpoint,
                device=self.device,
            )
        except Exception as e:
            logger.warning(f"Failed to load RTMPose: {e}")
            logger.info("Falling back to YOLOv8-Pose")
            self._fallback_to_yolo()
            return
        
        self._initialized = True
        logger.info(f"RTMPose ({self.model_variant}) initialized on {self.device}")
    
    def _init_detector(self):
        """Initialize person detector."""
        if self.detector_type == 'yolov8':
            try:
                from ultralytics import YOLO
                self.detector = YOLO('yolov8m.pt')
                self.detector.to(self.device)
            except ImportError:
                logger.warning("YOLOv8 not available, using MMDet")
                self.detector_type = 'rtmdet'
        
        if self.detector_type == 'rtmdet':
            try:
                from mmdet.apis import init_detector
                det_config = 'configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'
                det_checkpoint = 'checkpoints/rtmdet_m.pth'
                self.detector = init_detector(
                    det_config, det_checkpoint, device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to init RTMDet: {e}")
    
    def _fallback_to_yolo(self):
        """Fallback to YOLOv8-Pose if RTMPose fails."""
        self._yolo_fallback = YOLOv8PoseEstimator(
            model_size='m',
            device=self.device,
        )
        self._yolo_fallback.initialize()
        self._initialized = True
    
    def estimate(self, frame: np.ndarray) -> List[Detection]:
        """Estimate poses using RTMPose."""
        if not self._initialized:
            self.initialize()
        
        # Check for fallback
        if hasattr(self, '_yolo_fallback'):
            return self._yolo_fallback.estimate(frame)
        
        from mmpose.apis import inference_topdown
        
        # Detect persons
        bboxes = self._detect_persons(frame)
        
        if len(bboxes) == 0:
            return []
        
        # Top-down pose estimation
        pose_results = inference_topdown(
            self.pose_model,
            frame,
            bboxes,
        )
        
        detections = []
        for i, result in enumerate(pose_results):
            keypoints = result.pred_instances.keypoints[0]  # [17, 2]
            scores = result.pred_instances.keypoint_scores[0]  # [17]
            
            # Combine into [17, 3]
            kpts = np.concatenate([
                keypoints,
                scores[:, np.newaxis]
            ], axis=1)
            
            detection = Detection(
                bbox=bboxes[i],
                keypoints=kpts,
                confidence=float(scores.mean()),
            )
            detections.append(detection)
        
        return detections
    
    def _detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in frame."""
        if self.detector_type == 'yolov8' and self.detector is not None:
            results = self.detector(frame, classes=[0], verbose=False)[0]
            if results.boxes is not None:
                return results.boxes.xyxy.cpu().numpy()
            return np.array([])
        
        # MMDet detection
        from mmdet.apis import inference_detector
        result = inference_detector(self.detector, frame)
        
        # Filter person class (class 0 in COCO)
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        
        person_mask = (labels == 0) & (scores > self.confidence_threshold)
        return bboxes[person_mask]


# =============================================================================
# MediaPipe Pose Estimator
# =============================================================================

class MediaPipePoseEstimator(BasePoseEstimator):
    """
    MediaPipe Pose estimator.
    
    Lightweight, runs well on CPU.
    Good for mobile/edge deployment.
    
    Note: MediaPipe processes one person at a time.
    For multi-person, use with a detector.
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        device: str = 'cpu',  # MediaPipe is CPU-optimized
    ):
        """
        Initialize MediaPipe Pose.
        
        Args:
            static_image_mode: Treat each image independently
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            min_detection_confidence: Detection threshold
            min_tracking_confidence: Tracking threshold
            device: Not used (MediaPipe uses CPU)
        """
        super().__init__(min_detection_confidence, 0.3, device)
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_tracking_confidence = min_tracking_confidence
    
    def initialize(self):
        """Initialize MediaPipe Pose."""
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )
        
        self.mp_pose = mp.solutions.pose
        self.model = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        
        self._initialized = True
        logger.info("MediaPipe Pose initialized")
    
    def estimate(self, frame: np.ndarray) -> List[Detection]:
        """Estimate pose using MediaPipe."""
        if not self._initialized:
            self.initialize()
        
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.model.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([
                landmark.x * w,
                landmark.y * h,
                landmark.visibility,
            ])
        keypoints = np.array(keypoints)  # [33, 3]
        
        # Convert to COCO format (17 keypoints)
        coco_keypoints = self._convert_to_coco(keypoints)
        
        # Compute bounding box from keypoints
        valid_pts = keypoints[keypoints[:, 2] > 0.5, :2]
        if len(valid_pts) > 0:
            x1, y1 = valid_pts.min(axis=0)
            x2, y2 = valid_pts.max(axis=0)
            # Add padding
            pad = 20
            bbox = np.array([
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(w, x2 + pad),
                min(h, y2 + pad),
            ])
        else:
            bbox = np.array([0, 0, w, h])
        
        detection = Detection(
            bbox=bbox,
            keypoints=coco_keypoints,
            confidence=float(keypoints[:, 2].mean()),
        )
        
        return [detection]
    
    def _convert_to_coco(self, mp_keypoints: np.ndarray) -> np.ndarray:
        """Convert MediaPipe 33 keypoints to COCO 17 keypoints."""
        # MediaPipe to COCO mapping
        mapping = {
            0: 0,    # nose
            2: 1,    # left_eye_inner -> left_eye
            5: 2,    # right_eye_inner -> right_eye
            7: 3,    # left_ear
            8: 4,    # right_ear
            11: 5,   # left_shoulder
            12: 6,   # right_shoulder
            13: 7,   # left_elbow
            14: 8,   # right_elbow
            15: 9,   # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }
        
        coco_keypoints = np.zeros((17, 3))
        for mp_idx, coco_idx in mapping.items():
            if mp_idx < len(mp_keypoints):
                coco_keypoints[coco_idx] = mp_keypoints[mp_idx]
        
        return coco_keypoints


# =============================================================================
# Unified Pose Estimator Factory
# =============================================================================

class PoseEstimator:
    """
    Unified pose estimator with multiple backend support.
    
    Usage:
        # Default (YOLOv8-Pose)
        estimator = PoseEstimator()
        
        # Specific backend
        estimator = PoseEstimator(backend='rtmpose', model_size='l')
        
        # Estimate poses
        detections = estimator.estimate(frame)
    """
    
    BACKENDS = ['yolov8', 'rtmpose', 'mediapipe']
    
    def __init__(
        self,
        backend: str = 'yolov8',
        model_size: str = 'm',
        confidence_threshold: float = 0.5,
        keypoint_threshold: float = 0.3,
        device: str = 'cuda',
        **kwargs,
    ):
        """
        Initialize pose estimator.
        
        Args:
            backend: Backend to use ('yolov8', 'rtmpose', 'mediapipe')
            model_size: Model size (backend-specific)
            confidence_threshold: Detection confidence threshold
            keypoint_threshold: Keypoint confidence threshold
            device: Compute device
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend.lower()
        
        if self.backend == 'yolov8':
            self.estimator = YOLOv8PoseEstimator(
                model_size=model_size,
                confidence_threshold=confidence_threshold,
                keypoint_threshold=keypoint_threshold,
                device=device,
                **kwargs,
            )
        elif self.backend == 'rtmpose':
            self.estimator = RTMPoseEstimator(
                model_variant=f'rtmpose-{model_size}',
                confidence_threshold=confidence_threshold,
                keypoint_threshold=keypoint_threshold,
                device=device,
                **kwargs,
            )
        elif self.backend == 'mediapipe':
            self.estimator = MediaPipePoseEstimator(
                min_detection_confidence=confidence_threshold,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from {self.BACKENDS}")
    
    def estimate(self, frame: np.ndarray) -> List[Detection]:
        """Estimate poses in frame."""
        return self.estimator.estimate(frame)
    
    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Estimate poses in multiple frames."""
        return self.estimator.estimate_batch(frames)
    
    def warmup(self, input_size: Tuple[int, int] = (640, 480)):
        """Warmup model."""
        self.estimator.warmup(input_size)
    
    def initialize(self):
        """Initialize backend."""
        self.estimator.initialize()


# =============================================================================
# Utility Functions
# =============================================================================

def convert_keypoints(
    keypoints: np.ndarray,
    from_format: KeypointFormat,
    to_format: KeypointFormat,
) -> np.ndarray:
    """
    Convert keypoints between formats.
    
    Args:
        keypoints: Input keypoints [num_keypoints, 2/3]
        from_format: Source format
        to_format: Target format
    
    Returns:
        Converted keypoints
    """
    # Implementation would include conversion matrices
    # for each format pair
    raise NotImplementedError("Keypoint format conversion")


def filter_low_confidence_keypoints(
    keypoints: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Filter out low confidence keypoints.
    
    Args:
        keypoints: [num_keypoints, 3] with confidence
        threshold: Confidence threshold
    
    Returns:
        Filtered keypoints (low conf set to 0)
    """
    filtered = keypoints.copy()
    low_conf_mask = filtered[:, 2] < threshold
    filtered[low_conf_mask, :2] = 0
    filtered[low_conf_mask, 2] = 0
    return filtered


def interpolate_missing_keypoints(
    keypoints: np.ndarray,
    skeleton: List[Tuple[int, int]] = COCO_SKELETON,
) -> np.ndarray:
    """
    Interpolate missing keypoints from neighbors.
    
    Args:
        keypoints: [num_keypoints, 3]
        skeleton: Skeleton connections
    
    Returns:
        Keypoints with interpolated values
    """
    result = keypoints.copy()
    
    for i in range(len(result)):
        if result[i, 2] < 0.1:  # Missing keypoint
            # Find connected keypoints
            neighbors = []
            for conn in skeleton:
                if conn[0] == i and result[conn[1], 2] > 0.5:
                    neighbors.append(conn[1])
                elif conn[1] == i and result[conn[0], 2] > 0.5:
                    neighbors.append(conn[0])
            
            if neighbors:
                # Interpolate from neighbors
                neighbor_pts = result[neighbors, :2]
                result[i, :2] = neighbor_pts.mean(axis=0)
                result[i, 2] = 0.5  # Mark as interpolated
    
    return result


def benchmark_pose_estimator(
    estimator: BasePoseEstimator,
    num_frames: int = 100,
    input_size: Tuple[int, int] = (1280, 720),
) -> Dict[str, float]:
    """
    Benchmark pose estimator performance.
    
    Args:
        estimator: Pose estimator to benchmark
        num_frames: Number of frames to process
        input_size: Input resolution (width, height)
    
    Returns:
        Dictionary with FPS and latency metrics
    """
    # Generate random frames
    frames = [
        np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]
    
    # Warmup
    estimator.warmup(input_size)
    
    # Benchmark
    latencies = []
    
    for frame in frames:
        start = time.perf_counter()
        _ = estimator.estimate(frame)
        latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies)
    
    return {
        'fps': 1.0 / latencies.mean(),
        'latency_mean_ms': latencies.mean() * 1000,
        'latency_std_ms': latencies.std() * 1000,
        'latency_p95_ms': np.percentile(latencies, 95) * 1000,
        'latency_p99_ms': np.percentile(latencies, 99) * 1000,
    }
