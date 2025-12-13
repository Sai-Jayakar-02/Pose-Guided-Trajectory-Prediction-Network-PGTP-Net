"""
Benchmark Protocols for Trajectory Prediction.

Implements standard benchmark protocols:
- ETH/UCY Leave-One-Out: Train on 4 scenes, test on 1
- JTA Benchmark: Standard train/val/test split
- TrajNet++: Challenge protocol

References:
- ETH/UCY: Pellegrini et al. (2009), Lerner et al. (2007)
- Social-GAN: Gupta et al. (2018)
- Social-LSTM: Alahi et al. (2016)
- Trajectron++: Salzmann et al. (2020)
- TrajNet++: Kothari et al. (2021)

Usage:
    from src.evaluation import ETHUCYBenchmark
    
    benchmark = ETHUCYBenchmark(
        data_dir='data/raw/eth_ucy',
        model_class=SocialPoseModel,
        config=model_config,
    )
    
    results = benchmark.run()
    benchmark.print_results()
    benchmark.save_results('results/eth_ucy_benchmark.json')
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any
from tqdm import tqdm
from datetime import datetime
import logging
from collections import defaultdict

from .evaluator import Evaluator
from .metrics import TrajectoryMetrics, compute_best_of_k

logger = logging.getLogger(__name__)


# =============================================================================
# ETH/UCY Benchmark
# =============================================================================

class ETHUCYBenchmark:
    """
    ETH/UCY Leave-One-Out Benchmark.
    
    Standard protocol:
    - 5 scenes: ETH, Hotel, Univ, Zara1, Zara2
    - Leave-one-out cross-validation
    - Observe 8 frames (3.2s), predict 12 frames (4.8s) at 2.5 FPS
    - Report ADE and FDE per scene and average
    
    Reference metrics (Social-GAN, K=20):
    - ETH: ADE=0.81, FDE=1.52
    - Hotel: ADE=0.72, FDE=1.61
    - Univ: ADE=0.60, FDE=1.26
    - Zara1: ADE=0.34, FDE=0.69
    - Zara2: ADE=0.42, FDE=0.84
    - Average: ADE=0.58, FDE=1.18
    """
    
    SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    
    # Reference results from literature for comparison
    REFERENCE_RESULTS = {
        'Social-LSTM': {
            'eth': {'ADE': 1.09, 'FDE': 2.35},
            'hotel': {'ADE': 0.79, 'FDE': 1.76},
            'univ': {'ADE': 0.67, 'FDE': 1.40},
            'zara1': {'ADE': 0.47, 'FDE': 1.00},
            'zara2': {'ADE': 0.56, 'FDE': 1.17},
        },
        'Social-GAN': {
            'eth': {'ADE': 0.81, 'FDE': 1.52},
            'hotel': {'ADE': 0.72, 'FDE': 1.61},
            'univ': {'ADE': 0.60, 'FDE': 1.26},
            'zara1': {'ADE': 0.34, 'FDE': 0.69},
            'zara2': {'ADE': 0.42, 'FDE': 0.84},
        },
        'Trajectron++': {
            'eth': {'ADE': 0.43, 'FDE': 0.86},
            'hotel': {'ADE': 0.12, 'FDE': 0.19},
            'univ': {'ADE': 0.22, 'FDE': 0.43},
            'zara1': {'ADE': 0.17, 'FDE': 0.32},
            'zara2': {'ADE': 0.12, 'FDE': 0.25},
        },
    }
    
    def __init__(
        self,
        data_dir: str,
        model_class: Type,
        config: Dict,
        k_samples: int = 20,
        obs_len: int = 8,
        pred_len: int = 12,
        batch_size: int = 64,
        num_workers: int = 4,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize ETH/UCY benchmark.
        
        Args:
            data_dir: Path to ETH/UCY data directory
            model_class: Model class to evaluate
            config: Model configuration dictionary
            k_samples: Number of samples for Best-of-K
            obs_len: Observation length (frames)
            pred_len: Prediction length (frames)
            batch_size: Batch size for evaluation
            num_workers: DataLoader workers
            device: Compute device
            checkpoint_dir: Directory containing trained checkpoints
        """
        self.data_dir = Path(data_dir)
        self.model_class = model_class
        self.config = config
        self.k_samples = k_samples
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        self.results = {}
        
        logger.info(f"ETH/UCY Benchmark initialized")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  K samples: {k_samples}")
        logger.info(f"  Sequence: {obs_len} obs -> {pred_len} pred at 2.5 FPS")
    
    def run(
        self,
        scenes: Optional[List[str]] = None,
        progress_bar: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run benchmark on all scenes.
        
        Args:
            scenes: Specific scenes to evaluate (default: all)
            progress_bar: Show progress bar
        
        Returns:
            Dictionary of {scene: metrics}
        """
        scenes = scenes or self.SCENES
        self.results = {}
        
        for scene in scenes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {scene.upper()}")
            logger.info('='*60)
            
            # Load model checkpoint for this scene
            model = self._load_model_for_scene(scene)
            
            # Create test loader
            test_loader = self._create_test_loader(scene)
            
            # Evaluate
            evaluator = Evaluator(
                model=model,
                test_loader=test_loader,
                k_samples=self.k_samples,
                device=self.device,
            )
            
            scene_results = evaluator.evaluate(progress_bar=progress_bar)
            self.results[scene] = scene_results
            
            logger.info(f"  ADE: {scene_results.get('minADE_20', scene_results.get('ADE', 0)):.4f}")
            logger.info(f"  FDE: {scene_results.get('minFDE_20', scene_results.get('FDE', 0)):.4f}")
        
        # Compute average
        self._compute_average()
        
        return self.results
    
    def _load_model_for_scene(self, scene: str) -> torch.nn.Module:
        """Load trained model for a specific test scene."""
        model = self.model_class.from_config(self.config)
        
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f"{scene}_best.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"  Loaded checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"  Checkpoint not found: {checkpoint_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _create_test_loader(self, test_scene: str) -> torch.utils.data.DataLoader:
        """Create test loader for a specific scene."""
        # Import here to avoid circular imports
        from ..data import create_leave_one_out_dataloaders
        
        _, _, test_loader = create_leave_one_out_dataloaders(
            data_dir=str(self.data_dir),
            test_scene=test_scene,
            batch_size=self.batch_size,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            num_workers=self.num_workers,
        )
        
        return test_loader
    
    def _compute_average(self):
        """Compute average metrics across all scenes."""
        if not self.results:
            return
        
        # Skip 'average' if already computed
        scene_results = {k: v for k, v in self.results.items() if k != 'average'}
        
        if not scene_results:
            return
        
        avg_metrics = defaultdict(list)
        
        for scene, metrics in scene_results.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    avg_metrics[key].append(value)
        
        self.results['average'] = {
            key: np.mean(values) for key, values in avg_metrics.items()
        }
    
    def print_results(self, compare_baseline: Optional[str] = None):
        """
        Print benchmark results in formatted table.
        
        Args:
            compare_baseline: Compare against reference baseline (e.g., 'Social-GAN')
        """
        if not self.results:
            print("No results. Run benchmark first.")
            return
        
        print("\n" + "="*70)
        print("ETH/UCY BENCHMARK RESULTS")
        print("="*70)
        
        # Header
        if compare_baseline and compare_baseline in self.REFERENCE_RESULTS:
            print(f"{'Scene':<12} {'ADE':>10} {'FDE':>10} | "
                  f"{'Ref ADE':>10} {'Ref FDE':>10}")
            print("-"*70)
        else:
            print(f"{'Scene':<12} {'ADE':>10} {'FDE':>10}")
            print("-"*34)
        
        # Scene results
        for scene in self.SCENES:
            if scene not in self.results:
                continue
            
            metrics = self.results[scene]
            ade = metrics.get('minADE_20', metrics.get('ADE', 0))
            fde = metrics.get('minFDE_20', metrics.get('FDE', 0))
            
            if compare_baseline and compare_baseline in self.REFERENCE_RESULTS:
                ref = self.REFERENCE_RESULTS[compare_baseline].get(scene, {})
                ref_ade = ref.get('ADE', '-')
                ref_fde = ref.get('FDE', '-')
                
                # Calculate improvement
                if isinstance(ref_ade, float):
                    ade_diff = ((ref_ade - ade) / ref_ade) * 100
                    ade_str = f"{ade:.4f} ({ade_diff:+.1f}%)"
                else:
                    ade_str = f"{ade:.4f}"
                
                if isinstance(ref_fde, float):
                    fde_diff = ((ref_fde - fde) / ref_fde) * 100
                    fde_str = f"{fde:.4f} ({fde_diff:+.1f}%)"
                else:
                    fde_str = f"{fde:.4f}"
                
                print(f"{scene.upper():<12} {ade_str:>18} {fde_str:>18} | "
                      f"{ref_ade:>10} {ref_fde:>10}")
            else:
                print(f"{scene.upper():<12} {ade:>10.4f} {fde:>10.4f}")
        
        # Average
        print("-"*70 if compare_baseline else "-"*34)
        if 'average' in self.results:
            avg = self.results['average']
            ade = avg.get('minADE_20', avg.get('ADE', 0))
            fde = avg.get('minFDE_20', avg.get('FDE', 0))
            print(f"{'AVERAGE':<12} {ade:>10.4f} {fde:>10.4f}")
        
        print("="*70)
    
    def save_results(
        self,
        path: str,
        include_reference: bool = True,
    ):
        """
        Save benchmark results.
        
        Args:
            path: Output path
            include_reference: Include reference baselines
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'k_samples': self.k_samples,
                'obs_len': self.obs_len,
                'pred_len': self.pred_len,
            },
            'results': self.results,
        }
        
        if include_reference:
            output['reference_baselines'] = self.REFERENCE_RESULTS
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def export_latex_table(self, path: str):
        """Export results as LaTeX table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{ETH/UCY Benchmark Results (K=20)}",
            r"\label{tab:eth_ucy_results}",
            r"\begin{tabular}{l|cc|cc|cc}",
            r"\toprule",
            r"& \multicolumn{2}{c|}{Ours} & \multicolumn{2}{c|}{Social-GAN} & \multicolumn{2}{c}{Trajectron++} \\",
            r"Scene & ADE & FDE & ADE & FDE & ADE & FDE \\",
            r"\midrule",
        ]
        
        for scene in self.SCENES:
            our_results = self.results.get(scene, {})
            our_ade = our_results.get('minADE_20', our_results.get('ADE', '-'))
            our_fde = our_results.get('minFDE_20', our_results.get('FDE', '-'))
            
            sgan = self.REFERENCE_RESULTS['Social-GAN'].get(scene, {})
            tpp = self.REFERENCE_RESULTS['Trajectron++'].get(scene, {})
            
            if isinstance(our_ade, float):
                our_ade_str = f"{our_ade:.2f}"
            else:
                our_ade_str = str(our_ade)
            
            if isinstance(our_fde, float):
                our_fde_str = f"{our_fde:.2f}"
            else:
                our_fde_str = str(our_fde)
            
            lines.append(
                f"{scene.upper()} & {our_ade_str} & {our_fde_str} & "
                f"{sgan.get('ADE', '-')} & {sgan.get('FDE', '-')} & "
                f"{tpp.get('ADE', '-')} & {tpp.get('FDE', '-')} \\\\"
            )
        
        # Average
        lines.append(r"\midrule")
        avg = self.results.get('average', {})
        our_ade = avg.get('minADE_20', avg.get('ADE', '-'))
        our_fde = avg.get('minFDE_20', avg.get('FDE', '-'))
        
        if isinstance(our_ade, float):
            our_ade_str = f"\\textbf{{{our_ade:.2f}}}"
        else:
            our_ade_str = str(our_ade)
        
        if isinstance(our_fde, float):
            our_fde_str = f"\\textbf{{{our_fde:.2f}}}"
        else:
            our_fde_str = str(our_fde)
        
        lines.append(f"Average & {our_ade_str} & {our_fde_str} & 0.58 & 1.18 & 0.21 & 0.41 \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"LaTeX table saved to {path}")


# =============================================================================
# JTA Benchmark
# =============================================================================

class JTABenchmark:
    """
    JTA (Joint Track Auto) Benchmark.
    
    Standard protocol:
    - Train/Val/Test split as provided
    - Includes 3D pose annotations
    - Higher resolution and more crowded scenes than ETH/UCY
    """
    
    def __init__(
        self,
        data_dir: str,
        model_class: Type,
        config: Dict,
        k_samples: int = 20,
        obs_len: int = 8,
        pred_len: int = 12,
        batch_size: int = 64,
        num_workers: int = 4,
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize JTA benchmark.
        
        Args:
            data_dir: Path to JTA data directory
            model_class: Model class to evaluate
            config: Model configuration
            k_samples: Number of samples for Best-of-K
            obs_len: Observation length
            pred_len: Prediction length
            batch_size: Batch size
            num_workers: DataLoader workers
            device: Compute device
            checkpoint_path: Path to trained checkpoint
        """
        self.data_dir = Path(data_dir)
        self.model_class = model_class
        self.config = config
        self.k_samples = k_samples
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        self.results = {}
    
    def run(self, progress_bar: bool = True) -> Dict[str, float]:
        """Run JTA benchmark."""
        logger.info("Running JTA Benchmark...")
        
        # Load model
        model = self.model_class.from_config(self.config)
        
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        
        # Create test loader
        from ..data import create_jta_dataloaders
        
        _, _, test_loader = create_jta_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=self.batch_size,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            num_workers=self.num_workers,
        )
        
        # Evaluate
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            k_samples=self.k_samples,
            device=self.device,
        )
        
        self.results = evaluator.evaluate(progress_bar=progress_bar)
        
        return self.results
    
    def print_results(self):
        """Print benchmark results."""
        if not self.results:
            print("No results. Run benchmark first.")
            return
        
        print("\n" + "="*50)
        print("JTA BENCHMARK RESULTS")
        print("="*50)
        
        ade = self.results.get('minADE_20', self.results.get('ADE', 0))
        fde = self.results.get('minFDE_20', self.results.get('FDE', 0))
        
        print(f"  minADE@{self.k_samples}: {ade:.4f}")
        print(f"  minFDE@{self.k_samples}: {fde:.4f}")
        
        # Time horizons
        for key, value in self.results.items():
            if 'ADE@' in key:
                print(f"  {key}: {value:.4f}")
        
        print("="*50)
    
    def save_results(self, path: str):
        """Save benchmark results."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'benchmark': 'JTA',
                'config': {
                    'k_samples': self.k_samples,
                    'obs_len': self.obs_len,
                    'pred_len': self.pred_len,
                },
                'results': self.results,
            }, f, indent=2, default=str)


# =============================================================================
# TrajNet++ Benchmark
# =============================================================================

class TrajNetBenchmark:
    """
    TrajNet++ Challenge Benchmark.
    
    Standardized benchmark with:
    - Multiple interaction categories
    - Synthetic and real data
    - Challenge server evaluation
    
    Reference: Kothari et al. "Human Trajectory Forecasting in Crowds: 
    A Deep Learning Perspective" (2021)
    """
    
    CATEGORIES = [
        'static',      # Non-moving pedestrians
        'linear',      # Linear motion
        'interaction', # Social interactions
        'non_linear',  # Complex motion patterns
    ]
    
    def __init__(
        self,
        data_dir: str,
        model_class: Type,
        config: Dict,
        k_samples: int = 20,
        device: str = 'cuda',
    ):
        """Initialize TrajNet++ benchmark."""
        self.data_dir = Path(data_dir)
        self.model_class = model_class
        self.config = config
        self.k_samples = k_samples
        self.device = device
        
        self.results = {}
    
    def run(self) -> Dict[str, Dict[str, float]]:
        """Run TrajNet++ benchmark."""
        logger.info("Running TrajNet++ Benchmark...")
        logger.warning("TrajNet++ requires specific data format. "
                      "Implement data loading for your setup.")
        
        # Implementation would follow TrajNet++ data format
        # and evaluation protocol
        
        raise NotImplementedError(
            "TrajNet++ benchmark requires specific data format. "
            "See https://github.com/vita-epfl/trajnetplusplustools"
        )
    
    def export_submission(self, output_path: str):
        """Export predictions in TrajNet++ submission format."""
        raise NotImplementedError("Implement submission format export")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_eth_ucy_benchmark(
    model: torch.nn.Module,
    data_dir: str,
    k_samples: int = 20,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run ETH/UCY benchmark.
    
    Args:
        model: Trained model
        data_dir: Path to ETH/UCY data
        k_samples: Number of samples
        device: Compute device
    
    Returns:
        Benchmark results
    """
    from ..data import create_leave_one_out_dataloaders
    
    results = {}
    
    for scene in ETHUCYBenchmark.SCENES:
        logger.info(f"Evaluating on {scene}...")
        
        _, _, test_loader = create_leave_one_out_dataloaders(
            data_dir=data_dir,
            test_scene=scene,
        )
        
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            k_samples=k_samples,
            device=device,
        )
        
        results[scene] = evaluator.evaluate()
    
    # Average
    avg_ade = np.mean([r.get('minADE_20', r.get('ADE', 0)) for r in results.values()])
    avg_fde = np.mean([r.get('minFDE_20', r.get('FDE', 0)) for r in results.values()])
    results['average'] = {'ADE': avg_ade, 'FDE': avg_fde}
    
    return results


def print_benchmark_comparison(
    results: Dict[str, Dict[str, float]],
    model_name: str = "Ours",
):
    """
    Print benchmark comparison with baselines.
    
    Args:
        results: Benchmark results
        model_name: Name of evaluated model
    """
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    # Table header
    header = f"{'Scene':<12} | {model_name:^12} | {'Social-GAN':^12} | {'Trajectron++':^12}"
    print(header)
    print("-"*80)
    
    ref_sgan = ETHUCYBenchmark.REFERENCE_RESULTS['Social-GAN']
    ref_tpp = ETHUCYBenchmark.REFERENCE_RESULTS['Trajectron++']
    
    for scene in ETHUCYBenchmark.SCENES:
        our = results.get(scene, {})
        our_ade = our.get('minADE_20', our.get('ADE', '-'))
        
        sgan_ade = ref_sgan.get(scene, {}).get('ADE', '-')
        tpp_ade = ref_tpp.get(scene, {}).get('ADE', '-')
        
        if isinstance(our_ade, float):
            our_str = f"{our_ade:.3f}"
        else:
            our_str = str(our_ade)
        
        print(f"{scene.upper():<12} | {our_str:^12} | {sgan_ade:^12} | {tpp_ade:^12}")
    
    print("="*80)