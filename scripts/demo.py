#!/usr/bin/env python3
"""
PGTP-Net Real-Time Demo
Run pose-guided trajectory prediction on webcam or video

Usage:
    python demo.py --webcam
    python demo.py --video path/to/video.mp4
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from pgtpnet.models import PGTPNet, load_model
from pgtpnet.pose_estimation import RTMPoseDetector
from pgtpnet.utils import load_config


# Visualization colors (BGR)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
]


def parse_args():
    parser = argparse.ArgumentParser(description='PGTP-Net Real-Time Demo')
    
    # Input source
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--webcam_id', type=int, default=0, help='Webcam ID')
    parser.add_argument('--video', type=str, default=None, help='Video file path')
    parser.add_argument('--ip_camera', type=str, default=None, help='IP camera URL')
    
    # Model
    parser.add_argument('--model_path', type=str, 
                        default='models/pgtp_jta_best.pt',
                        help='Path to PGTP-Net checkpoint')
    parser.add_argument('--pose_model', type=str,
                        default='models/rtmpose_body8.pth',
                        help='Path to RTMPose checkpoint')
    
    # Visualization
    parser.add_argument('--show_pose', action='store_true', default=True,
                        help='Show pose skeleton')
    parser.add_argument('--show_trajectory', action='store_true', default=True,
                        help='Show trajectory predictions')
    parser.add_argument('--num_futures', type=int, default=8,
                        help='Number of future trajectories to show')
    parser.add_argument('--scale', type=float, default=1.5,
                        help='Trajectory visualization scale')
    
    # Window
    parser.add_argument('--width', type=int, default=1280, help='Window width')
    parser.add_argument('--height', type=int, default=720, help='Window height')
    parser.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')
    
    # Output
    parser.add_argument('--save_video', type=str, default=None,
                        help='Save output video')
    
    return parser.parse_args()


class PersonTracker:
    """Simple IoU-based person tracker."""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections):
        """Update tracks with new detections."""
        # detections: list of (bbox, keypoints, score)
        
        if not detections:
            # Age out tracks
            to_remove = []
            for track_id, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.tracks[track_id]
            return {}
        
        # Match detections to existing tracks using IoU
        matched = {}
        unmatched_dets = list(range(len(detections)))
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = None
            
            for det_idx in unmatched_dets:
                iou = self._compute_iou(track['bbox'], detections[det_idx][0])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                # Update track
                bbox, keypoints, score = detections[best_det_idx]
                track['bbox'] = bbox
                track['keypoints'] = keypoints
                track['score'] = score
                track['age'] = 0
                track['hits'] += 1
                matched[track_id] = track
                unmatched_dets.remove(best_det_idx)
            else:
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, keypoints, score = detections[det_idx]
            self.tracks[self.next_id] = {
                'bbox': bbox,
                'keypoints': keypoints,
                'score': score,
                'age': 0,
                'hits': 1,
            }
            self.next_id += 1
        
        # Return confirmed tracks (hits >= min_hits)
        return {tid: t for tid, t in self.tracks.items() if t['hits'] >= self.min_hits}
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


class TrajectoryBuffer:
    """Buffer for storing trajectory and pose history per person."""
    
    def __init__(self, obs_len=8):
        self.obs_len = obs_len
        self.buffers = {}  # {person_id: deque}
    
    def update(self, person_id, position, keypoints):
        """Add observation for a person."""
        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.obs_len)
        
        self.buffers[person_id].append({
            'position': position,
            'keypoints': keypoints,
        })
    
    def get_observation(self, person_id):
        """Get observation sequence for a person if buffer is full."""
        if person_id not in self.buffers:
            return None
        
        buffer = self.buffers[person_id]
        if len(buffer) < self.obs_len:
            return None
        
        # Extract trajectory and pose sequences
        positions = np.array([obs['position'] for obs in buffer])
        keypoints = np.array([obs['keypoints'] for obs in buffer])
        
        return {
            'trajectory': positions,
            'pose': keypoints,
        }
    
    def cleanup(self, active_ids):
        """Remove buffers for inactive persons."""
        to_remove = [pid for pid in self.buffers if pid not in active_ids]
        for pid in to_remove:
            del self.buffers[pid]


def draw_skeleton(frame, keypoints, color, thickness=2):
    """Draw pose skeleton on frame."""
    # COCO skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    ]
    
    for start, end in skeleton:
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for kp in keypoints:
        if kp[2] > 0.5:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)


def draw_trajectory(frame, past_traj, pred_traj, color, scale=1.5):
    """Draw past and predicted trajectories."""
    # Draw past trajectory (solid line)
    for i in range(len(past_traj) - 1):
        pt1 = (int(past_traj[i][0]), int(past_traj[i][1]))
        pt2 = (int(past_traj[i + 1][0]), int(past_traj[i + 1][1]))
        cv2.line(frame, pt1, pt2, color, int(3 * scale))
        cv2.circle(frame, pt1, int(4 * scale), color, -1)
    
    # Current position (large circle)
    current = past_traj[-1]
    cv2.circle(frame, (int(current[0]), int(current[1])), int(10 * scale), color, -1)
    cv2.circle(frame, (int(current[0]), int(current[1])), int(12 * scale), (255, 255, 255), 2)
    
    # Draw predicted trajectory (dashed line)
    if pred_traj is not None and len(pred_traj) > 0:
        prev_pt = (int(current[0]), int(current[1]))
        for i, pt in enumerate(pred_traj):
            curr_pt = (int(pt[0]), int(pt[1]))
            # Dashed effect
            if i % 2 == 0:
                cv2.line(frame, prev_pt, curr_pt, color, int(2 * scale))
            prev_pt = curr_pt
        
        # Final position marker
        final = pred_traj[-1]
        cv2.circle(frame, (int(final[0]), int(final[1])), int(8 * scale), color, 2)
        cv2.putText(frame, '+4.8s', (int(final[0]) + 10, int(final[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, color, 1)


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print("Loading PGTP-Net model...")
    model, config = load_model(args.model_path, device=device)
    model.eval()
    
    print("Loading pose estimation model...")
    pose_detector = RTMPoseDetector(args.pose_model, device=device)
    
    # Initialize tracker and buffer
    tracker = PersonTracker()
    traj_buffer = TrajectoryBuffer(obs_len=8)
    
    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(args.webcam_id)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.ip_camera:
        cap = cv2.VideoCapture(args.ip_camera)
    else:
        print("Error: Specify --webcam, --video, or --ip_camera")
        return
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (args.width, args.height))
    
    # Fullscreen
    window_name = 'PGTP-Net Demo'
    if args.fullscreen:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\n" + "=" * 50)
    print("PGTP-Net Real-Time Demo")
    print("=" * 50)
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  P - Toggle pose overlay")
    print("  T - Toggle trajectory")
    print("  +/- - Adjust trajectory scale")
    print("=" * 50 + "\n")
    
    # State
    show_pose = args.show_pose
    show_trajectory = args.show_trajectory
    scale = args.scale
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        frame_count += 1
        
        # Detect poses
        detections = pose_detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Update trajectory buffers
        predictions = {}
        for person_id, track in tracks.items():
            # Get center position from bbox
            bbox = track['bbox']
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            
            # Update buffer
            traj_buffer.update(person_id, center, track['keypoints'])
            
            # Get observation and predict
            obs = traj_buffer.get_observation(person_id)
            if obs is not None:
                # Predict trajectory
                with torch.no_grad():
                    obs_traj = torch.tensor(obs['trajectory'], dtype=torch.float32).unsqueeze(0).to(device)
                    obs_pose = torch.tensor(obs['pose'], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    pred = model.sample(
                        obs_traj=obs_traj,
                        obs_pose=obs_pose,
                        num_samples=args.num_futures
                    )
                    predictions[person_id] = {
                        'past': obs['trajectory'],
                        'future': pred[0].cpu().numpy(),  # [K, T, 2]
                    }
        
        # Cleanup inactive buffers
        traj_buffer.cleanup(set(tracks.keys()))
        
        # Draw visualizations
        for person_id, track in tracks.items():
            color = COLORS[person_id % len(COLORS)]
            
            # Draw skeleton
            if show_pose:
                draw_skeleton(frame, track['keypoints'], color)
            
            # Draw trajectory
            if show_trajectory and person_id in predictions:
                pred = predictions[person_id]
                # Draw main prediction
                draw_trajectory(frame, pred['past'], pred['future'][0], color, scale)
                # Draw alternative futures (lighter)
                for k in range(1, min(args.num_futures, len(pred['future']))):
                    alpha_color = tuple(int(c * 0.4) for c in color)
                    draw_trajectory(frame, pred['past'][-1:], pred['future'][k], alpha_color, scale * 0.5)
        
        # Calculate FPS
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Draw info panel
        panel_h, panel_w = 80, 180
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 1)
        
        cv2.putText(frame, 'PGTP-Net', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f'Tracked: {len(tracks)}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Save video
        if writer:
            writer.write(frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'screenshot_{frame_count}.png', frame)
            print(f"Saved screenshot_{frame_count}.png")
        elif key == ord('p'):
            show_pose = not show_pose
        elif key == ord('t'):
            show_trajectory = not show_trajectory
        elif key == ord('+') or key == ord('='):
            scale = min(scale + 0.2, 3.0)
        elif key == ord('-'):
            scale = max(scale - 0.2, 0.5)
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
