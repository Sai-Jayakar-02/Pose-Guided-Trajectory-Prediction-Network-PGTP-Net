"""
Utility functions for trajectory prediction.

Modules:
- metrics: ADE, FDE, collision rate, TrajectoryEvaluator
- visualization: matplotlib and OpenCV plotting
- helpers: seed, device, config loading
- logging_utils: TensorBoard, W&B, console logging
- checkpoint: Model save/load, export, SWA averaging

Usage:
    from src.utils import (
        # Metrics
        compute_ade, compute_fde, TrajectoryEvaluator,
        
        # Visualization
        visualize_trajectory, draw_skeleton,
        
        # Helpers
        set_seed, get_device, load_config,
        
        # Logging
        Logger, TensorBoardLogger, WandBLogger, MetricTracker,
        
        # Checkpoints
        CheckpointManager, save_checkpoint, load_checkpoint,
        export_to_onnx, average_checkpoints,
    )
"""

# Metrics
from .metrics import (
    compute_ade,
    compute_fde,
    compute_ade_fde,
    compute_ade_at_time,
    compute_best_of_k,
    compute_collision_rate,
    compute_miss_rate,
    TrajectoryEvaluator,
)

# Visualization
from .visualization import (
    draw_skeleton,
    draw_trajectory,
    draw_prediction_with_speed,
    draw_uncertainty,
    visualize_trajectory,
    visualize_prediction,
    create_video_writer,
    add_info_overlay,
    COCO_SKELETON,
    COLORS,
)

# Helpers
from .helpers import (
    set_seed,
    get_device,
    count_parameters,
    load_config,
    save_config,
    AverageMeter,
    EarlyStopping,
    print_model_summary,
)

# Logging utilities
from .logging_utils import (
    # Main logger
    Logger,
    
    # Backend loggers
    TensorBoardLogger,
    WandBLogger,
    ConsoleLogger,
    FileLogger,
    BaseLogger,
    
    # Utilities
    setup_logging,
    log_model_summary,
    MetricTracker,
)

# Checkpoint utilities
from .checkpoint import (
    # Save/load functions
    save_checkpoint,
    load_checkpoint,
    
    # Checkpoint manager
    CheckpointManager,
    
    # Export functions
    export_to_onnx,
    export_to_torchscript,
    
    # Averaging (SWA)
    average_checkpoints,
    select_best_checkpoints,
    get_checkpoint_metric,
    
    # Weight loading
    load_pretrained_weights,
    freeze_layers,
    
    # Comparison
    compare_checkpoints,
)

__all__ = [
    # Metrics
    'compute_ade',
    'compute_fde',
    'compute_ade_fde',
    'compute_ade_at_time',
    'compute_best_of_k',
    'compute_collision_rate',
    'compute_miss_rate',
    'TrajectoryEvaluator',
    
    # Visualization
    'draw_skeleton',
    'draw_trajectory',
    'draw_prediction_with_speed',
    'draw_uncertainty',
    'visualize_trajectory',
    'visualize_prediction',
    'create_video_writer',
    'add_info_overlay',
    'COCO_SKELETON',
    'COLORS',
    
    # Helpers
    'set_seed',
    'get_device',
    'count_parameters',
    'load_config',
    'save_config',
    'AverageMeter',
    'EarlyStopping',
    'print_model_summary',
    
    # Logging
    'Logger',
    'TensorBoardLogger',
    'WandBLogger',
    'ConsoleLogger',
    'FileLogger',
    'BaseLogger',
    'setup_logging',
    'log_model_summary',
    'MetricTracker',
    
    # Checkpoints
    'save_checkpoint',
    'load_checkpoint',
    'CheckpointManager',
    'export_to_onnx',
    'export_to_torchscript',
    'average_checkpoints',
    'select_best_checkpoints',
    'get_checkpoint_metric',
    'load_pretrained_weights',
    'freeze_layers',
    'compare_checkpoints',
]
