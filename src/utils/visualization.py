"""
Visualization utilities for trajectory prediction.
Includes matplotlib plotting and OpenCV real-time visualization.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


# COCO skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Colors for visualization
COLORS = {
    'past_traj': (0, 255, 0),      # Green
    'pred_traj': (0, 0, 255),      # Red
    'skeleton': (255, 255, 0),     # Cyan
    'uncertainty': (255, 0, 255),  # Magenta
}


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    point_radius: int = 3
) -> np.ndarray:
    """
    Draw skeleton on frame.
    
    Args:
        frame: BGR image
        keypoints: [num_joints, 2] or [num_joints, 3] (x, y, conf)
        color: BGR color
        thickness: Line thickness
        point_radius: Keypoint circle radius
    
    Returns:
        Frame with skeleton drawn
    """
    frame = frame.copy()
    
    # Handle 3D keypoints (take only x, y)
    if keypoints.shape[-1] == 3:
        keypoints = keypoints[:, :2]
    
    # Draw connections
    for (i, j) in COCO_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            
            # Skip invalid points
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for pt in keypoints:
        pt = tuple(pt.astype(int))
        if pt[0] > 0 and pt[1] > 0:
            cv2.circle(frame, pt, point_radius, color, -1)
    
    return frame


def draw_trajectory(
    frame: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
    show_points: bool = True,
    fade: bool = False
) -> np.ndarray:
    """
    Draw trajectory on frame.
    
    Args:
        frame: BGR image
        trajectory: [T, 2] pixel coordinates
        color: BGR color
        thickness: Line thickness
        point_radius: Point radius
        show_points: Whether to draw points
        fade: Whether to fade older points
    
    Returns:
        Frame with trajectory drawn
    """
    frame = frame.copy()
    T = len(trajectory)
    
    # Draw lines
    for i in range(T - 1):
        pt1 = tuple(trajectory[i].astype(int))
        pt2 = tuple(trajectory[i + 1].astype(int))
        
        if fade:
            # Fade from transparent to opaque
            alpha = (i + 1) / T
            line_color = tuple(int(c * alpha) for c in color)
        else:
            line_color = color
        
        cv2.line(frame, pt1, pt2, line_color, thickness)
    
    # Draw points
    if show_points:
        for i, pt in enumerate(trajectory):
            pt = tuple(pt.astype(int))
            
            if fade:
                alpha = (i + 1) / T
                pt_color = tuple(int(c * alpha) for c in color)
                radius = int(point_radius * alpha) + 1
            else:
                pt_color = color
                radius = point_radius
            
            cv2.circle(frame, pt, radius, pt_color, -1)
    
    return frame


def draw_prediction_with_speed(
    frame: np.ndarray,
    prediction: np.ndarray,
    speeds: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    base_radius: int = 3,
    show_time_labels: bool = True,
    fps: float = 2.5
) -> np.ndarray:
    """
    Draw predicted trajectory with speed-adaptive visualization.
    
    - Faster movement → larger markers (more spaced out)
    - Slower movement → smaller markers (closer together)
    
    Args:
        frame: BGR image
        prediction: [T, 2] pixel coordinates
        speeds: [T,] speed at each point (m/s)
        color: BGR color
        base_radius: Base point radius
        show_time_labels: Whether to show time labels
        fps: Frames per second
    
    Returns:
        Frame with prediction drawn
    """
    frame = frame.copy()
    T = len(prediction)
    
    # Default speeds if not provided
    if speeds is None:
        speeds = np.ones(T) * 1.5  # Default walking speed
    
    # Normalize speeds for visualization (0.5 to 3.0 m/s typical range)
    speed_normalized = np.clip(speeds / 2.0, 0.3, 2.0)
    
    # Draw connecting lines with fading
    for i in range(T - 1):
        pt1 = tuple(prediction[i].astype(int))
        pt2 = tuple(prediction[i + 1].astype(int))
        
        # Fade line thickness into future
        alpha = 1.0 - (i / T) * 0.7
        line_thickness = max(1, int(2 * alpha))
        line_color = tuple(int(c * alpha) for c in color)
        
        cv2.line(frame, pt1, pt2, line_color, line_thickness)
    
    # Draw points with speed-based sizing
    for i, pt in enumerate(prediction):
        pt_int = tuple(pt.astype(int))
        
        # Radius based on speed (faster = larger)
        radius = int(base_radius * speed_normalized[i])
        radius = max(2, min(radius, 10))
        
        # Color fades into future
        alpha = 1.0 - (i / T) * 0.5
        pt_color = tuple(int(c * alpha) for c in color)
        
        cv2.circle(frame, pt_int, radius, pt_color, -1)
        
        # Time labels every few points
        if show_time_labels and i % 3 == 0 and i > 0:
            time_sec = i / fps
            label = f"{time_sec:.1f}s"
            cv2.putText(
                frame, label,
                (pt_int[0] + 5, pt_int[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, pt_color, 1
            )
    
    return frame


def draw_uncertainty(
    frame: np.ndarray,
    prediction: np.ndarray,
    uncertainties: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw uncertainty ellipses around predictions.
    
    Args:
        frame: BGR image
        prediction: [T, 2] pixel coordinates
        uncertainties: [T, 2] sigma for x and y
        color: BGR color
        alpha: Transparency
    
    Returns:
        Frame with uncertainty visualization
    """
    overlay = frame.copy()
    
    for i, (pt, sigma) in enumerate(zip(prediction, uncertainties)):
        pt = tuple(pt.astype(int))
        
        # Scale sigma to pixels (assuming ~50 px/m)
        sigma_px = sigma * 50
        
        # Draw ellipse
        axes = tuple(np.clip(sigma_px.astype(int), 5, 100))
        cv2.ellipse(overlay, pt, axes, 0, 0, 360, color, -1)
    
    # Blend with original
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def visualize_trajectory(
    past_traj: np.ndarray,
    pred_traj: np.ndarray,
    gt_traj: Optional[np.ndarray] = None,
    title: str = "Trajectory Prediction",
    save_path: Optional[str] = None
):
    """
    Matplotlib visualization of trajectory.
    
    Args:
        past_traj: [obs_len, 2] observed trajectory
        pred_traj: [pred_len, 2] predicted trajectory
        gt_traj: [pred_len, 2] ground truth (optional)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot past trajectory
    ax.plot(past_traj[:, 0], past_traj[:, 1], 'g-o', 
            label='Observed', linewidth=2, markersize=8)
    
    # Plot prediction
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--o',
            label='Predicted', linewidth=2, markersize=6)
    
    # Plot ground truth
    if gt_traj is not None:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-s',
                label='Ground Truth', linewidth=2, markersize=6, alpha=0.7)
    
    # Connect past to prediction
    ax.plot([past_traj[-1, 0], pred_traj[0, 0]],
            [past_traj[-1, 1], pred_traj[0, 1]], 'r--', linewidth=1)
    
    # Mark start and end
    ax.scatter([past_traj[0, 0]], [past_traj[0, 1]], 
               c='green', s=100, marker='*', zorder=5, label='Start')
    ax.scatter([pred_traj[-1, 0]], [pred_traj[-1, 1]], 
               c='red', s=100, marker='X', zorder=5, label='End (Pred)')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_prediction(
    past_traj: np.ndarray,
    predictions: np.ndarray,
    gt_traj: Optional[np.ndarray] = None,
    speeds: Optional[np.ndarray] = None,
    title: str = "Multi-Modal Prediction",
    save_path: Optional[str] = None
):
    """
    Visualize multiple trajectory predictions.
    
    Args:
        past_traj: [obs_len, 2]
        predictions: [K, pred_len, 2] K predictions
        gt_traj: [pred_len, 2] ground truth
        speeds: [K, pred_len] speeds for each prediction
        title: Plot title
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    K = predictions.shape[0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, K))
    
    # Plot past
    ax.plot(past_traj[:, 0], past_traj[:, 1], 'g-o',
            label='Observed', linewidth=3, markersize=10)
    
    # Plot predictions
    for i in range(K):
        alpha = 0.3 + 0.5 * (i / K)
        ax.plot(predictions[i, :, 0], predictions[i, :, 1],
                color=colors[i], linewidth=1.5, alpha=alpha)
        
        # Plot points with speed-based sizing
        if speeds is not None:
            for j, (pt, spd) in enumerate(zip(predictions[i], speeds[i])):
                size = 20 + spd * 30  # Larger for faster
                ax.scatter(pt[0], pt[1], c=[colors[i]], s=size, alpha=alpha)
    
    # Plot ground truth
    if gt_traj is not None:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-s',
                label='Ground Truth', linewidth=2, markersize=8)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create OpenCV video writer.
    
    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: (width, height)
        codec: Video codec
    
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def add_info_overlay(
    frame: np.ndarray,
    info: Dict[str, str],
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Add text overlay with information.
    
    Args:
        frame: BGR image
        info: Dictionary of info to display
        position: Starting position
        font_scale: Font scale
        color: Text color
    
    Returns:
        Frame with overlay
    """
    frame = frame.copy()
    y = position[1]
    
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (position[0], y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2
        )
        y += 25
    
    return frame
