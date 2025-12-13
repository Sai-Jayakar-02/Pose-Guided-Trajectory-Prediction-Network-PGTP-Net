"""
Real-time trajectory prediction with OpenCV visualization.
Combines YOLOv8-pose detection with trajectory prediction model.
"""

import cv2
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import time

from ..models.full_model import AdaptivePoseTrajectoryPredictor, load_model
from ..utils.visualization import (
    draw_skeleton,
    draw_trajectory,
    draw_prediction_with_speed,
    add_info_overlay
)


class RealTimeTrajectoryPredictor:
    """
    Real-time trajectory prediction pipeline.
    
    Pipeline:
    1. YOLOv8-pose: Detect humans and extract pose
    2. Track: Associate detections across frames
    3. Buffer: Maintain history for each person
    4. Predict: Run trajectory model on buffered history
    5. Visualize: Draw predictions on frame
    """
    
    def __init__(
        self,
        model_path: str,
        detector_model: str = 'yolov8m-pose.pt',
        obs_len: int = 8,
        pred_len: int = 12,
        target_fps: float = 2.5,
        device: str = 'cuda',
        conf_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to trained trajectory model
            detector_model: YOLOv8-pose model name/path
            obs_len: Observation length (frames)
            pred_len: Prediction length (frames)
            target_fps: Target FPS for trajectory (2.5 standard)
            device: Compute device
            conf_threshold: Detection confidence threshold
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.target_fps = target_fps
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load trajectory prediction model
        print(f"Loading trajectory model from {model_path}...")
        self.model, self.config = load_model(model_path, device)
        self.model.eval()
        
        # Load YOLOv8-pose detector
        print(f"Loading detector: {detector_model}...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO(detector_model)
        except ImportError:
            print("ERROR: ultralytics not installed. Run: pip install ultralytics")
            self.detector = None
        
        # History buffers for each tracked person
        self.history = defaultdict(lambda: {
            'positions': [],      # World/pixel positions
            'poses': [],          # Skeleton keypoints
            'timestamps': [],     # Frame timestamps
            'last_seen': 0        # Last frame seen
        })
        
        # Frame counter
        self.frame_count = 0
        self.frame_interval = 1  # Process every N frames
        
        # Homography for pixel to world conversion (if available)
        self.homography = None
        
        # FPS calculation
        self.fps_buffer = []
    
    def set_homography(self, H: np.ndarray):
        """
        Set homography matrix for pixel to world coordinate conversion.
        
        Args:
            H: [3, 3] homography matrix
        """
        self.homography = H
    
    def pixel_to_world(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to world coordinates."""
        if self.homography is None:
            # No conversion, return scaled pixels
            return pixel_coords / 100.0  # Rough scale: 100 px = 1m
        
        # Apply homography
        N = len(pixel_coords)
        ones = np.ones((N, 1))
        pixel_homo = np.hstack([pixel_coords, ones])
        world_homo = pixel_homo @ self.homography.T
        world_coords = world_homo[:, :2] / world_homo[:, 2:3]
        return world_coords
    
    def world_to_pixel(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to pixel coordinates."""
        if self.homography is None:
            return world_coords * 100.0
        
        H_inv = np.linalg.inv(self.homography)
        N = len(world_coords)
        ones = np.ones((N, 1))
        world_homo = np.hstack([world_coords, ones])
        pixel_homo = world_homo @ H_inv.T
        pixel_coords = pixel_homo[:, :2] / pixel_homo[:, 2:3]
        return pixel_coords
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame.
        
        Args:
            frame: BGR image
            timestamp: Frame timestamp (optional)
        
        Returns:
            annotated_frame: Frame with visualizations
            predictions: Dictionary of predictions per person
        """
        self.frame_count += 1
        timestamp = timestamp or time.time()
        
        start_time = time.time()
        
        # 1. Detect and track
        detections = self._detect(frame)
        
        # 2. Update history buffers
        self._update_history(detections, timestamp)
        
        # 3. Run predictions for persons with enough history
        predictions = {}
        for track_id, data in self.history.items():
            if len(data['positions']) >= self.obs_len:
                pred = self._predict(track_id)
                if pred is not None:
                    predictions[track_id] = pred
        
        # 4. Visualize
        annotated = self._visualize(frame, detections, predictions)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        self.fps_buffer.append(1.0 / (elapsed + 1e-6))
        if len(self.fps_buffer) > 30:
            self.fps_buffer.pop(0)
        
        # Add info overlay
        info = {
            'FPS': f"{np.mean(self.fps_buffer):.1f}",
            'Tracked': str(len(self.history)),
            'Predicting': str(len(predictions))
        }
        annotated = add_info_overlay(annotated, info)
        
        return annotated, predictions
    
    def _detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8-pose detection.
        
        Args:
            frame: BGR image
        
        Returns:
            List of detections with track_id, bbox, keypoints
        """
        if self.detector is None:
            return []
        
        # Run detection with tracking
        results = self.detector.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    # Get track ID
                    track_id = int(boxes.id[i]) if boxes.id is not None else i
                    
                    # Get bounding box
                    bbox = boxes.xyxy[i].cpu().numpy()
                    
                    # Get keypoints [17, 3] (x, y, conf)
                    kpts = keypoints.data[i].cpu().numpy()
                    
                    # Calculate center position (bottom center of bbox as foot position)
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = bbox[3]  # Bottom of bbox
                    
                    detections.append({
                        'track_id': track_id,
                        'bbox': bbox,
                        'keypoints': kpts,
                        'position': np.array([center_x, center_y])
                    })
        
        return detections
    
    def _update_history(self, detections: List[Dict], timestamp: float):
        """
        Update history buffers with new detections.
        
        Args:
            detections: List of detection dictionaries
            timestamp: Current timestamp
        """
        current_ids = set()
        
        for det in detections:
            track_id = det['track_id']
            current_ids.add(track_id)
            
            # Add to history
            self.history[track_id]['positions'].append(det['position'])
            self.history[track_id]['poses'].append(det['keypoints'])
            self.history[track_id]['timestamps'].append(timestamp)
            self.history[track_id]['last_seen'] = self.frame_count
            
            # Limit history length
            max_len = self.obs_len * 3
            if len(self.history[track_id]['positions']) > max_len:
                self.history[track_id]['positions'] = \
                    self.history[track_id]['positions'][-max_len:]
                self.history[track_id]['poses'] = \
                    self.history[track_id]['poses'][-max_len:]
                self.history[track_id]['timestamps'] = \
                    self.history[track_id]['timestamps'][-max_len:]
        
        # Remove stale tracks
        stale_threshold = 30  # frames
        stale_ids = [
            tid for tid, data in self.history.items()
            if self.frame_count - data['last_seen'] > stale_threshold
        ]
        for tid in stale_ids:
            del self.history[tid]
    
    def _predict(self, track_id: int) -> Optional[Dict]:
        """
        Run trajectory prediction for a single person.
        
        Args:
            track_id: Person track ID
        
        Returns:
            Dictionary with predictions, speeds, uncertainties
        """
        data = self.history[track_id]
        
        # Get last obs_len frames
        positions = np.array(data['positions'][-self.obs_len:])
        poses = np.array(data['poses'][-self.obs_len:])
        
        if len(positions) < self.obs_len:
            return None
        
        # Convert to world coordinates
        positions_world = self.pixel_to_world(positions)
        
        # Normalize trajectory (origin at last position)
        origin = positions_world[-1].copy()
        positions_norm = positions_world - origin
        
        # Normalize pose (pelvis-centered)
        poses_norm = poses.copy()
        if poses.shape[1] >= 1:  # Has pelvis
            pelvis = poses[:, 0:1, :2]
            poses_norm[:, :, :2] = poses[:, :, :2] - pelvis
        
        # Add Z coordinate if needed (2D to 3D)
        if poses_norm.shape[-1] == 2:
            zeros = np.zeros((*poses_norm.shape[:-1], 1))
            poses_norm = np.concatenate([poses_norm, zeros], axis=-1)
        
        # Compute velocity
        velocities = np.diff(positions_norm, axis=0)
        
        # Convert to tensors
        obs_traj = torch.FloatTensor(positions_norm).unsqueeze(0).to(self.device)
        obs_pose = torch.FloatTensor(poses_norm).unsqueeze(0).to(self.device)
        obs_vel = torch.FloatTensor(velocities).unsqueeze(0).to(self.device)
        
        # Run prediction
        with torch.no_grad():
            result = self.model.predict(
                obs_traj=obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_vel
            )
        
        # Extract results
        pred_traj = result['predictions'][0].cpu().numpy()
        speeds = result.get('speeds', None)
        if speeds is not None:
            speeds = speeds[0].cpu().numpy().flatten()
        
        uncertainties = result.get('uncertainties', None)
        if uncertainties is not None:
            uncertainties = uncertainties[0].cpu().numpy()
        
        # Denormalize to world coordinates
        pred_traj_world = pred_traj + origin
        
        # Convert to pixel coordinates
        pred_traj_pixel = self.world_to_pixel(pred_traj_world)
        
        return {
            'trajectory': pred_traj_pixel,
            'trajectory_world': pred_traj_world,
            'speeds': speeds,
            'uncertainties': uncertainties,
            'origin': origin
        }
    
    def _visualize(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        predictions: Dict
    ) -> np.ndarray:
        """
        Visualize detections and predictions.
        
        Args:
            frame: Original frame
            detections: List of detections
            predictions: Dictionary of predictions
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections
        for det in detections:
            track_id = det['track_id']
            keypoints = det['keypoints']
            position = det['position']
            
            # Draw skeleton
            annotated = draw_skeleton(annotated, keypoints[:, :2])
            
            # Draw past trajectory
            if track_id in self.history:
                past_positions = np.array(
                    self.history[track_id]['positions'][-self.obs_len:]
                )
                if len(past_positions) >= 2:
                    annotated = draw_trajectory(
                        annotated, past_positions,
                        color=(0, 255, 0),  # Green
                        thickness=2
                    )
            
            # Draw track ID
            pt = tuple(position.astype(int))
            cv2.putText(
                annotated, f"ID:{track_id}",
                (pt[0] - 20, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        # Draw predictions
        for track_id, pred in predictions.items():
            annotated = draw_prediction_with_speed(
                annotated,
                pred['trajectory'],
                pred['speeds'],
                color=(0, 0, 255),  # Red
                base_radius=4,
                show_time_labels=True,
                fps=self.target_fps
            )
        
        return annotated
    
    def run_on_video(
        self,
        video_source,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None
    ):
        """
        Run prediction on video file or camera.
        
        Args:
            video_source: Video file path or camera index (0)
            output_path: Optional output video path
            display: Whether to display real-time
            max_frames: Maximum frames to process
        """
        # Open video
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps} FPS")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Process frame
                annotated, predictions = self.process_frame(frame)
                
                # Write output
                if writer:
                    writer.write(annotated)
                
                # Display
                if display:
                    cv2.imshow('Trajectory Prediction', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)  # Pause
                
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"Done! Processed {frame_idx} frames.")


def run_realtime_demo(
    model_path: str,
    video_source=0,
    output_path: Optional[str] = None
):
    """
    Convenience function to run real-time demo.
    
    Args:
        model_path: Path to trained model
        video_source: Camera index or video path
        output_path: Optional output video path
    """
    predictor = RealTimeTrajectoryPredictor(
        model_path=model_path,
        detector_model='yolov8m-pose.pt'
    )
    
    predictor.run_on_video(
        video_source=video_source,
        output_path=output_path,
        display=True
    )
