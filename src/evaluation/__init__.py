"""
Evaluation module for trajectory prediction.

Main components:
- TrajectoryMetrics: Comprehensive metrics (ADE, FDE, Best-of-K, etc.)
- Evaluator: Evaluation pipeline for single model
- ModelComparator: Compare multiple models
- AblationStudy: Ablation study support
- ETHUCYBenchmark: Standard ETH/UCY leave-one-out benchmark
- JTABenchmark: JTA dataset benchmark

Usage:
    # Quick evaluation
    from src.evaluation import Evaluator, TrajectoryMetrics
    
    evaluator = Evaluator(model, test_loader, k_samples=20)
    results = evaluator.evaluate()
    
    # ETH/UCY Benchmark
    from src.evaluation import ETHUCYBenchmark
    
    benchmark = ETHUCYBenchmark(
        data_dir='data/raw/eth_ucy',
        model_class=SocialPoseModel,
        config=config,
    )
    results = benchmark.run()
    benchmark.print_results()
    
    # Model comparison
    from src.evaluation import ModelComparator
    
    comparator = ModelComparator(test_loader)
    comparator.add_model('Social-Pose', model1)
    comparator.add_model('Baseline', model2)
    results = comparator.compare()
    print(comparator.get_comparison_table())
"""

# Metrics
from .metrics import (
    # Core metrics
    compute_ade,
    compute_fde,
    compute_ade_fde,
    
    # Best-of-K metrics
    compute_best_of_k,
    compute_best_of_k_ade,
    compute_best_of_k_fde,
    
    # Time-horizon metrics
    compute_ade_at_time,
    compute_fde_at_time,
    
    # Additional metrics
    compute_miss_rate,
    compute_collision_rate,
    compute_self_collision_rate,
    compute_nll,
    compute_kde_nll,
    compute_diversity,
    compute_sample_variance,
    compute_speed_error,
    compute_heading_error,
    
    # Metrics class
    TrajectoryMetrics,
)

# Evaluator
from .evaluator import (
    Evaluator,
    ModelComparator,
    AblationStudy,
)

# Benchmarks
from .benchmark import (
    ETHUCYBenchmark,
    JTABenchmark,
    TrajNetBenchmark,
    run_eth_ucy_benchmark,
    print_benchmark_comparison,
)

__all__ = [
    # Core metrics
    'compute_ade',
    'compute_fde',
    'compute_ade_fde',
    
    # Best-of-K
    'compute_best_of_k',
    'compute_best_of_k_ade',
    'compute_best_of_k_fde',
    
    # Time-horizon
    'compute_ade_at_time',
    'compute_fde_at_time',
    
    # Additional metrics
    'compute_miss_rate',
    'compute_collision_rate',
    'compute_self_collision_rate',
    'compute_nll',
    'compute_kde_nll',
    'compute_diversity',
    'compute_sample_variance',
    'compute_speed_error',
    'compute_heading_error',
    
    # Metrics class
    'TrajectoryMetrics',
    
    # Evaluator
    'Evaluator',
    'ModelComparator',
    'AblationStudy',
    
    # Benchmarks
    'ETHUCYBenchmark',
    'JTABenchmark',
    'TrajNetBenchmark',
    'run_eth_ucy_benchmark',
    'print_benchmark_comparison',
]
