"""
Multi-Object Tracking Module for Real-Time Inference.

Provides unified interface for multiple tracking algorithms:
- ByteTrack: Fast, high-performance tracking
- BoT-SORT: State-of-the-art accuracy with appearance features
- DeepSORT: Classic deep learning tracker
- SORT: Simple online tracking

Usage:
    from src.inference import Tracker
    
    # Initialize tracker
    tracker = Tracker(backend='bytetrack')
    
    # Update with detections
    tracks = tracker.update(detections, frame)
    
    # Each track has: id, bbox, keypoints, age, state

References:
- ByteTrack: https://github.com/ifzhang/ByteTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- DeepSORT: https://github.com/nwojke/deep_sort
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


# =============================================================================
# Track State and Data Classes
# =============================================================================

class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1    # Not yet confirmed
    CONFIRMED = 2    # Active, confirmed track
    LOST = 3         # Temporarily lost
    DELETED = 4      # Marked for deletion


@dataclass
class Track:
    """Single tracked object."""
    track_id: int
    bbox: np.ndarray                      # [x1, y1, x2, y2]
    keypoints: Optional[np.ndarray]       # [num_keypoints, 3]
    confidence: float
    state: TrackState = TrackState.TENTATIVE
    age: int = 0                          # Frames since creation
    hits: int = 0                         # Successful associations
    time_since_update: int = 0            # Frames since last update
    velocity: Optional[np.ndarray] = None # Estimated velocity
    features: Optional[np.ndarray] = None # Appearance features
    trajectory: List[np.ndarray] = field(default_factory=list)  # Position history
    
    @property
    def center(self) -> np.ndarray:
        """Get bounding box center."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])
    
    @property
    def tlwh(self) -> np.ndarray:
        """Get bbox in (top-left x, top-left y, width, height) format."""
        return np.array([
            self.bbox[0],
            self.bbox[1],
            self.bbox[2] - self.bbox[0],
            self.bbox[3] - self.bbox[1],
        ])
    
    def get_pelvis(self) -> np.ndarray:
        """Get pelvis position (for trajectory prediction)."""
        if self.keypoints is not None and len(self.keypoints) >= 17:
            left_hip = self.keypoints[11, :2]
            right_hip = self.keypoints[12, :2]
            return (left_hip + right_hip) / 2
        return self.center
    
    def predict_next_position(self) -> np.ndarray:
        """Predict next position based on velocity."""
        if self.velocity is not None:
            return self.center + self.velocity
        return self.center


# =============================================================================
# Kalman Filter for Motion Prediction
# =============================================================================

class KalmanTracker:
    """
    Kalman filter for bounding box tracking.
    
    State: [x, y, s, r, vx, vy, vs]
    - (x, y): center position
    - s: scale (area)
    - r: aspect ratio
    - (vx, vy, vs): velocities
    """
    
    def __init__(self, bbox: np.ndarray):
        """
        Initialize Kalman filter with bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Convert bbox to [x, y, s, r]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_xysr(bbox).reshape(4, 1)
        
        self.time_since_update = 0
        self.history = []
    
    def _bbox_to_xysr(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox to [x, y, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h  # scale (area)
        r = w / (h + 1e-6)  # aspect ratio
        return np.array([x, y, s, r])
    
    def _xysr_to_bbox(self, xysr: np.ndarray) -> np.ndarray:
        """Convert [x, y, s, r] to bbox."""
        x, y, s, r = xysr.flatten()[:4]
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return np.array([
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2,
        ])
    
    def predict(self) -> np.ndarray:
        """Predict next state."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.time_since_update += 1
        self.history.append(self._xysr_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def update(self, bbox: np.ndarray):
        """Update state with measurement."""
        self.time_since_update = 0
        self.history = []
        
        measurement = self._bbox_to_xysr(bbox)
        self.kf.update(measurement)
    
    def get_state(self) -> np.ndarray:
        """Get current bbox state."""
        return self._xysr_to_bbox(self.kf.x)
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate."""
        return self.kf.x[4:6].flatten()


# =============================================================================
# IoU and Distance Functions
# =============================================================================

def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-6)


def compute_iou_matrix(
    bboxes1: np.ndarray,
    bboxes2: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.
    
    Args:
        bboxes1: [N, 4]
        bboxes2: [M, 4]
    
    Returns:
        IoU matrix [N, M]
    """
    n = len(bboxes1)
    m = len(bboxes2)
    
    iou_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])
    
    return iou_matrix


def compute_cost_matrix(
    tracks: List[Track],
    detections: List,
    iou_threshold: float = 0.3,
    use_appearance: bool = False,
    appearance_weight: float = 0.5,
) -> np.ndarray:
    """
    Compute cost matrix for track-detection association.
    
    Args:
        tracks: List of tracks
        detections: List of detections
        iou_threshold: IoU threshold
        use_appearance: Use appearance features
        appearance_weight: Weight for appearance cost
    
    Returns:
        Cost matrix [num_tracks, num_detections]
    """
    num_tracks = len(tracks)
    num_detections = len(detections)
    
    if num_tracks == 0 or num_detections == 0:
        return np.zeros((num_tracks, num_detections))
    
    # Get bboxes
    track_bboxes = np.array([t.bbox for t in tracks])
    det_bboxes = np.array([d.bbox for d in detections])
    
    # IoU-based cost
    iou_matrix = compute_iou_matrix(track_bboxes, det_bboxes)
    iou_cost = 1 - iou_matrix
    
    # Mask out low IoU matches
    iou_cost[iou_matrix < iou_threshold] = 1e6
    
    if use_appearance and all(t.features is not None for t in tracks):
        # Appearance cost (cosine distance)
        track_features = np.array([t.features for t in tracks])
        det_features = np.array([d.features for d in detections if hasattr(d, 'features')])
        
        if len(det_features) == num_detections:
            # Normalize features
            track_features = track_features / (np.linalg.norm(track_features, axis=1, keepdims=True) + 1e-6)
            det_features = det_features / (np.linalg.norm(det_features, axis=1, keepdims=True) + 1e-6)
            
            # Cosine distance
            appearance_cost = 1 - np.dot(track_features, det_features.T)
            
            # Combined cost
            cost_matrix = (1 - appearance_weight) * iou_cost + appearance_weight * appearance_cost
        else:
            cost_matrix = iou_cost
    else:
        cost_matrix = iou_cost
    
    return cost_matrix


# =============================================================================
# Base Tracker
# =============================================================================

class BaseTracker:
    """Base class for multi-object trackers."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep lost track
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of Detection objects
            frame: Current frame (for appearance features)
        
        Returns:
            List of active tracks
        """
        raise NotImplementedError
    
    def _create_track(self, detection) -> Track:
        """Create new track from detection."""
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox.copy(),
            keypoints=detection.keypoints.copy() if detection.keypoints is not None else None,
            confidence=detection.confidence,
            state=TrackState.TENTATIVE,
        )
        self.next_id += 1
        return track
    
    def _update_track(self, track: Track, detection) -> Track:
        """Update track with matched detection."""
        track.bbox = detection.bbox.copy()
        if detection.keypoints is not None:
            track.keypoints = detection.keypoints.copy()
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.trajectory.append(track.center.copy())
        
        # Confirm track if enough hits
        if track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED
        
        return track
    
    def get_active_tracks(self) -> List[Track]:
        """Get confirmed, active tracks."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0


# =============================================================================
# SORT Tracker (Simple Online and Realtime Tracking)
# =============================================================================

class SORTTracker(BaseTracker):
    """
    Simple Online and Realtime Tracking (SORT).
    
    Uses Kalman filter for motion prediction and Hungarian algorithm
    for association. Fast but no appearance features.
    
    Reference: Bewley et al. "Simple Online and Realtime Tracking" (2016)
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        super().__init__(max_age, min_hits, iou_threshold)
        self.kalman_trackers: Dict[int, KalmanTracker] = {}
    
    def update(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracks using SORT algorithm."""
        self.frame_count += 1
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            if track.track_id in self.kalman_trackers:
                kf = self.kalman_trackers[track.track_id]
                track.bbox = kf.predict()
                track.velocity = kf.get_velocity()
        
        # Association
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = compute_cost_matrix(
                self.tracks, detections, self.iou_threshold
            )
            
            # Hungarian algorithm
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # Filter out high-cost matches
            matched_tracks = set()
            matched_dets = set()
            
            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] < 1e5:
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
                    
                    # Update track
                    track = self.tracks[t_idx]
                    detection = detections[d_idx]
                    self._update_track(track, detection)
                    
                    # Update Kalman filter
                    if track.track_id in self.kalman_trackers:
                        self.kalman_trackers[track.track_id].update(detection.bbox)
            
            # Handle unmatched tracks
            for t_idx, track in enumerate(self.tracks):
                if t_idx not in matched_tracks:
                    track.time_since_update += 1
                    track.age += 1
                    if track.time_since_update > self.max_age:
                        track.state = TrackState.DELETED
                    elif track.state == TrackState.CONFIRMED:
                        track.state = TrackState.LOST
            
            # Create new tracks for unmatched detections
            for d_idx, detection in enumerate(detections):
                if d_idx not in matched_dets:
                    track = self._create_track(detection)
                    self.tracks.append(track)
                    self.kalman_trackers[track.track_id] = KalmanTracker(detection.bbox)
        
        elif len(detections) > 0:
            # No existing tracks, create all new
            for detection in detections:
                track = self._create_track(detection)
                self.tracks.append(track)
                self.kalman_trackers[track.track_id] = KalmanTracker(detection.bbox)
        
        else:
            # No detections, age all tracks
            for track in self.tracks:
                track.time_since_update += 1
                track.age += 1
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        
        # Clean up Kalman filters
        active_ids = {t.track_id for t in self.tracks}
        self.kalman_trackers = {
            k: v for k, v in self.kalman_trackers.items() if k in active_ids
        }
        
        return self.get_active_tracks()


# =============================================================================
# ByteTrack
# =============================================================================

class ByteTracker(BaseTracker):
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    
    Associates both high and low confidence detections in two stages.
    Better handling of occlusions and crowded scenes.
    
    Reference: Zhang et al. "ByteTrack" (ECCV 2022)
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
    ):
        """
        Initialize ByteTracker.
        
        Args:
            max_age: Maximum frames to keep lost track
            min_hits: Minimum hits before confirmation
            iou_threshold: IoU threshold
            high_thresh: High confidence threshold
            low_thresh: Low confidence threshold
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.kalman_trackers: Dict[int, KalmanTracker] = {}
    
    def update(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracks using ByteTrack algorithm."""
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Predict track positions
        for track in self.tracks:
            if track.track_id in self.kalman_trackers:
                kf = self.kalman_trackers[track.track_id]
                track.bbox = kf.predict()
                track.velocity = kf.get_velocity()
        
        # First association: high confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        matched_t1, matched_d1, unmatched_t1, unmatched_d1 = self._associate(
            confirmed_tracks, high_dets
        )
        
        # Update matched tracks
        for t_idx, d_idx in zip(matched_t1, matched_d1):
            track = confirmed_tracks[t_idx]
            detection = high_dets[d_idx]
            self._update_track(track, detection)
            if track.track_id in self.kalman_trackers:
                self.kalman_trackers[track.track_id].update(detection.bbox)
        
        # Second association: remaining tracks with low confidence detections
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_t1]
        matched_t2, matched_d2, unmatched_t2, _ = self._associate(
            remaining_tracks, low_dets
        )
        
        # Update matched tracks
        for t_idx, d_idx in zip(matched_t2, matched_d2):
            track = remaining_tracks[t_idx]
            detection = low_dets[d_idx]
            self._update_track(track, detection)
            if track.track_id in self.kalman_trackers:
                self.kalman_trackers[track.track_id].update(detection.bbox)
        
        # Handle unmatched tracks
        truly_unmatched = [remaining_tracks[i] for i in unmatched_t2]
        for track in truly_unmatched:
            track.time_since_update += 1
            track.age += 1
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED
            else:
                track.state = TrackState.LOST
        
        # Third association: tentative tracks with unmatched high detections
        tentative_tracks = [t for t in self.tracks if t.state == TrackState.TENTATIVE]
        unmatched_high_dets = [high_dets[i] for i in unmatched_d1]
        
        matched_t3, matched_d3, unmatched_t3, unmatched_d3 = self._associate(
            tentative_tracks, unmatched_high_dets
        )
        
        for t_idx, d_idx in zip(matched_t3, matched_d3):
            track = tentative_tracks[t_idx]
            detection = unmatched_high_dets[d_idx]
            self._update_track(track, detection)
            if track.track_id in self.kalman_trackers:
                self.kalman_trackers[track.track_id].update(detection.bbox)
        
        # Delete unmatched tentative tracks
        for t_idx in unmatched_t3:
            tentative_tracks[t_idx].state = TrackState.DELETED
        
        # Create new tracks for remaining unmatched detections
        final_unmatched_dets = [unmatched_high_dets[i] for i in unmatched_d3]
        for detection in final_unmatched_dets:
            track = self._create_track(detection)
            self.tracks.append(track)
            self.kalman_trackers[track.track_id] = KalmanTracker(detection.bbox)
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        
        # Clean up
        active_ids = {t.track_id for t in self.tracks}
        self.kalman_trackers = {
            k: v for k, v in self.kalman_trackers.items() if k in active_ids
        }
        
        return self.get_active_tracks()
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Associate tracks with detections."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], [], list(range(len(tracks))), list(range(len(detections)))
        
        cost_matrix = compute_cost_matrix(tracks, detections, self.iou_threshold)
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matched_tracks = []
        matched_dets = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < 1e5:
                matched_tracks.append(t_idx)
                matched_dets.append(d_idx)
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
        
        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets


# =============================================================================
# BoT-SORT (with appearance features)
# =============================================================================

class BoTSORTTracker(ByteTracker):
    """
    BoT-SORT: Robust Associations Multi-Pedestrian Tracking.
    
    Extends ByteTrack with appearance features and camera motion compensation.
    State-of-the-art accuracy on MOT benchmarks.
    
    Reference: Aharon et al. "BoT-SORT" (2022)
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        appearance_weight: float = 0.5,
        feature_extractor: Optional[str] = None,
    ):
        """
        Initialize BoT-SORT.
        
        Args:
            max_age: Maximum age for tracks
            min_hits: Minimum hits for confirmation
            iou_threshold: IoU threshold
            high_thresh: High confidence threshold
            low_thresh: Low confidence threshold
            appearance_weight: Weight for appearance cost
            feature_extractor: Path to feature extractor model
        """
        super().__init__(max_age, min_hits, iou_threshold, high_thresh, low_thresh)
        self.appearance_weight = appearance_weight
        self.feature_extractor = None
        
        if feature_extractor:
            self._init_feature_extractor(feature_extractor)
    
    def _init_feature_extractor(self, model_path: str):
        """Initialize appearance feature extractor."""
        try:
            import torch
            import torchvision.transforms as T
            from torchvision import models
            
            # Simple ResNet feature extractor
            self.feature_model = models.resnet18(pretrained=True)
            self.feature_model.fc = torch.nn.Identity()
            self.feature_model.eval()
            
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((128, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            logger.info("Feature extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize feature extractor: {e}")
    
    def _extract_features(
        self,
        frame: np.ndarray,
        detections: List,
    ) -> List[np.ndarray]:
        """Extract appearance features for detections."""
        if self.feature_model is None:
            return [None] * len(detections)
        
        import torch
        
        features = []
        
        for det in detections:
            # Crop detection
            x1, y1, x2, y2 = det.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(None)
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            # Extract features
            with torch.no_grad():
                img_tensor = self.transform(crop).unsqueeze(0)
                feat = self.feature_model(img_tensor).numpy().flatten()
                features.append(feat)
        
        return features
    
    def update(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracks with appearance features."""
        # Extract features if available
        if frame is not None and self.feature_model is not None:
            features = self._extract_features(frame, detections)
            for det, feat in zip(detections, features):
                det.features = feat
        
        # Use parent ByteTrack update
        return super().update(detections, frame)


# =============================================================================
# Unified Tracker Factory
# =============================================================================

class Tracker:
    """
    Unified tracker with multiple backend support.
    
    Usage:
        tracker = Tracker(backend='bytetrack')
        tracks = tracker.update(detections, frame)
    """
    
    BACKENDS = ['sort', 'bytetrack', 'botsort']
    
    def __init__(
        self,
        backend: str = 'bytetrack',
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        **kwargs,
    ):
        """
        Initialize tracker.
        
        Args:
            backend: Tracking backend
            max_age: Maximum age for tracks
            min_hits: Minimum hits for confirmation
            iou_threshold: IoU threshold
            **kwargs: Backend-specific arguments
        """
        self.backend = backend.lower()
        
        if self.backend == 'sort':
            self.tracker = SORTTracker(max_age, min_hits, iou_threshold)
        elif self.backend == 'bytetrack':
            self.tracker = ByteTracker(
                max_age, min_hits, iou_threshold,
                high_thresh=kwargs.get('high_thresh', 0.6),
                low_thresh=kwargs.get('low_thresh', 0.1),
            )
        elif self.backend == 'botsort':
            self.tracker = BoTSORTTracker(
                max_age, min_hits, iou_threshold,
                high_thresh=kwargs.get('high_thresh', 0.6),
                low_thresh=kwargs.get('low_thresh', 0.1),
                appearance_weight=kwargs.get('appearance_weight', 0.5),
                feature_extractor=kwargs.get('feature_extractor'),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from {self.BACKENDS}")
        
        logger.info(f"Tracker initialized with {backend} backend")
    
    def update(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracks with detections."""
        return self.tracker.update(detections, frame)
    
    def get_active_tracks(self) -> List[Track]:
        """Get active tracks."""
        return self.tracker.get_active_tracks()
    
    def reset(self):
        """Reset tracker."""
        self.tracker.reset()
    
    @property
    def tracks(self) -> List[Track]:
        """Get all tracks."""
        return self.tracker.tracks


# =============================================================================
# Utility Functions
# =============================================================================

def extract_trajectories(
    tracks: List[Track],
    min_length: int = 8,
) -> Dict[int, np.ndarray]:
    """
    Extract trajectories from tracks.
    
    Args:
        tracks: List of tracks
        min_length: Minimum trajectory length
    
    Returns:
        Dictionary of {track_id: trajectory}
    """
    trajectories = {}
    
    for track in tracks:
        if len(track.trajectory) >= min_length:
            trajectories[track.track_id] = np.array(track.trajectory)
    
    return trajectories


def smooth_trajectory(
    trajectory: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Smooth trajectory with moving average.
    
    Args:
        trajectory: [T, 2] trajectory
        window_size: Smoothing window
    
    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = np.zeros_like(trajectory)
    half_window = window_size // 2
    
    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)
        smoothed[i] = trajectory[start:end].mean(axis=0)
    
    return smoothed
