"""
Inference module for real-time trajectory prediction.

Main components:
- RealTimeTrajectoryPredictor: End-to-end real-time inference pipeline
- PoseEstimator: Multi-backend pose estimation (YOLOv8, RTMPose, MediaPipe)
- Tracker: Multi-object tracking (SORT, ByteTrack, BoT-SORT)
- TensorRTOptimizer: TensorRT optimization for maximum performance

Quick Start:
    # Full pipeline (detection + tracking + prediction)
    from src.inference import RealTimeTrajectoryPredictor
    
    predictor = RealTimeTrajectoryPredictor(
        model_path='checkpoints/social_pose_best.pt',
        detector='yolov8m-pose',
    )
    predictor.run_on_video('input.mp4', 'output.mp4')
    
    # Individual components
    from src.inference import PoseEstimator, Tracker
    
    # Pose estimation
    pose_estimator = PoseEstimator(backend='yolov8', model_size='m')
    detections = pose_estimator.estimate(frame)
    
    # Tracking
    tracker = Tracker(backend='bytetrack')
    tracks = tracker.update(detections, frame)
    
    # TensorRT optimization
    from src.inference import TensorRTOptimizer
    
    optimizer = TensorRTOptimizer(model)
    optimizer.optimize('model.trt', input_shape={'input': (1, 8, 2)}, fp16=True)
    output = optimizer.inference(input_data)

Performance Tips:
    1. Use YOLOv8-pose for best speed/accuracy tradeoff
    2. Enable FP16 with TensorRT for 2x speedup
    3. Use ByteTrack for crowded scenes
    4. Batch processing for higher throughput
"""

# Real-time predictor (main pipeline)
from .realtime_predictor import (
    RealTimeTrajectoryPredictor,
    run_realtime_demo,
)

# Pose estimation
from .pose_estimator import (
    # Main interface
    PoseEstimator,
    
    # Backend implementations
    YOLOv8PoseEstimator,
    RTMPoseEstimator,
    MediaPipePoseEstimator,
    BasePoseEstimator,
    
    # Data classes
    Detection,
    KeypointFormat,
    
    # Constants
    COCO_KEYPOINTS,
    COCO_SKELETON,
    
    # Utilities
    filter_low_confidence_keypoints,
    interpolate_missing_keypoints,
    benchmark_pose_estimator,
)

# Tracking
from .tracker import (
    # Main interface
    Tracker,
    
    # Backend implementations
    SORTTracker,
    ByteTracker,
    BoTSORTTracker,
    BaseTracker,
    
    # Data classes
    Track,
    TrackState,
    
    # Kalman filter
    KalmanTracker,
    
    # Utilities
    compute_iou,
    compute_iou_matrix,
    extract_trajectories,
    smooth_trajectory,
)

# TensorRT optimization
from .tensorrt_optimizer import (
    # Main interface
    TensorRTOptimizer,
    
    # Components
    ONNXExporter,
    TensorRTBuilder,
    TensorRTRuntime,
    INT8Calibrator,
    
    # Configuration
    TensorRTConfig,
    
    # Utilities
    check_tensorrt_available,
    get_tensorrt_version,
    compare_pytorch_tensorrt,
    optimize_for_jetson,
)


__all__ = [
    # Main pipeline
    'RealTimeTrajectoryPredictor',
    'run_realtime_demo',
    
    # Pose estimation
    'PoseEstimator',
    'YOLOv8PoseEstimator',
    'RTMPoseEstimator',
    'MediaPipePoseEstimator',
    'BasePoseEstimator',
    'Detection',
    'KeypointFormat',
    'COCO_KEYPOINTS',
    'COCO_SKELETON',
    'filter_low_confidence_keypoints',
    'interpolate_missing_keypoints',
    'benchmark_pose_estimator',
    
    # Tracking
    'Tracker',
    'SORTTracker',
    'ByteTracker',
    'BoTSORTTracker',
    'BaseTracker',
    'Track',
    'TrackState',
    'KalmanTracker',
    'compute_iou',
    'compute_iou_matrix',
    'extract_trajectories',
    'smooth_trajectory',
    
    # TensorRT
    'TensorRTOptimizer',
    'ONNXExporter',
    'TensorRTBuilder',
    'TensorRTRuntime',
    'INT8Calibrator',
    'TensorRTConfig',
    'check_tensorrt_available',
    'get_tensorrt_version',
    'compare_pytorch_tensorrt',
    'optimize_for_jetson',
]
