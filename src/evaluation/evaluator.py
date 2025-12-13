"""
Evaluation Pipeline for Trajectory Prediction Models.

Provides comprehensive evaluation functionality:
- Single model evaluation on test set
- Multi-model comparison
- Ablation study support
- Visualization generation
- Results export (JSON, CSV, LaTeX tables)

Usage:
    from src.evaluation import Evaluator
    
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        k_samples=20,
        device='cuda'
    )
    
    results = evaluator.evaluate()
    evaluator.save_results('results/evaluation.json')
    evaluator.generate_visualizations('results/viz/')
"""

import torch
import numpy as np
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
from datetime import datetime
import logging

from .metrics import (
    TrajectoryMetrics,
    compute_ade,
    compute_fde,
    compute_best_of_k,
    compute_diversity,
    compute_miss_rate,
    compute_collision_rate,
    compute_speed_error,
    compute_heading_error,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation pipeline for trajectory prediction models.
    
    Supports:
    - Deterministic and stochastic models
    - Best-of-K evaluation
    - Time-horizon analysis
    - Visualization generation
    - Results export
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        k_samples: int = 20,
        time_horizons: List[float] = [1.0, 2.0, 3.0, 4.8],
        fps: float = 2.5,
        device: str = 'cuda',
        stochastic: bool = True,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained trajectory prediction model
            test_loader: Test data loader
            k_samples: Number of samples for Best-of-K evaluation
            time_horizons: Time horizons for ADE@T metrics
            fps: Dataset FPS (2.5 for ETH/UCY)
            device: Compute device
            stochastic: Whether model is stochastic (generates samples)
        """
        self.model = model
        self.test_loader = test_loader
        self.k_samples = k_samples
        self.time_horizons = time_horizons
        self.fps = fps
        self.device = torch.device(device)
        self.stochastic = stochastic
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Metrics tracker
        self.metrics = TrajectoryMetrics(
            k_samples=k_samples,
            time_horizons=time_horizons,
            fps=fps,
        )
        
        # Storage for predictions (for visualization)
        self.all_predictions = []
        self.all_observations = []
        self.all_ground_truth = []
        
        logger.info(f"Evaluator initialized with K={k_samples}, stochastic={stochastic}")
    
    def evaluate(
        self,
        save_predictions: bool = False,
        progress_bar: bool = True,
    ) -> Dict[str, float]:
        """
        Run evaluation on test set.
        
        Args:
            save_predictions: Whether to save predictions for visualization
            progress_bar: Whether to show progress bar
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.metrics.reset()
        self.all_predictions = []
        self.all_observations = []
        self.all_ground_truth = []
        
        self.model.eval()
        
        iterator = tqdm(self.test_loader, desc='Evaluating') if progress_bar else self.test_loader
        
        with torch.no_grad():
            for batch in iterator:
                # Move batch to device
                obs_traj = batch['obs_traj'].to(self.device)
                pred_traj_gt = batch['pred_traj'].to(self.device)
                
                # Get optional inputs
                obs_pose = batch.get('obs_pose')
                if obs_pose is not None:
                    obs_pose = obs_pose.to(self.device)
                
                obs_velocity = batch.get('obs_velocity')
                if obs_velocity is not None:
                    obs_velocity = obs_velocity.to(self.device)
                
                # Generate predictions
                if self.stochastic and hasattr(self.model, 'sample'):
                    # Stochastic model - generate K samples
                    predictions = self.model.sample(
                        obs_traj,
                        obs_pose=obs_pose,
                        obs_velocity=obs_velocity,
                        num_samples=self.k_samples,
                    )  # [batch, K, pred_len, 2]
                    
                    # Rearrange to [K, batch, pred_len, 2]
                    predictions = predictions.permute(1, 0, 2, 3)
                    
                    self.metrics.update(predictions, pred_traj_gt, multi_sample=True)
                else:
                    # Deterministic model
                    predictions, extras = self.model(
                        obs_traj,
                        obs_pose=obs_pose,
                        obs_velocity=obs_velocity,
                    )
                    
                    self.metrics.update(predictions, pred_traj_gt, multi_sample=False)
                
                # Save predictions if requested
                if save_predictions:
                    self.all_predictions.append(predictions.cpu().numpy())
                    self.all_observations.append(obs_traj.cpu().numpy())
                    self.all_ground_truth.append(pred_traj_gt.cpu().numpy())
        
        # Compute final metrics
        results = self.metrics.compute()
        
        logger.info(f"Evaluation complete: {results}")
        
        return results
    
    def evaluate_per_scene(
        self,
        scene_ids: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate and report metrics per scene.
        
        Args:
            scene_ids: List of scene identifiers in batch
        
        Returns:
            Dictionary of {scene_name: metrics}
        """
        # This requires scene information in the batch
        # Implementation depends on dataset structure
        raise NotImplementedError("Per-scene evaluation requires scene IDs in batch")
    
    def compute_extended_metrics(
        self,
        predictions: torch.Tensor,
        gt: torch.Tensor,
        obs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute extended metrics beyond ADE/FDE.
        
        Args:
            predictions: [K, batch, pred_len, 2] or [batch, pred_len, 2]
            gt: [batch, pred_len, 2]
            obs: [batch, obs_len, 2]
        
        Returns:
            Extended metrics dictionary
        """
        results = {}
        
        # Get best prediction if multi-sample
        if predictions.dim() == 4:
            K = predictions.size(0)
            ade_per_k = torch.stack([
                compute_ade(predictions[k], gt, mode='none')
                for k in range(K)
            ], dim=0)
            best_idx = ade_per_k.argmin(dim=0)
            batch_size = gt.size(0)
            best_pred = torch.stack([
                predictions[best_idx[b], b]
                for b in range(batch_size)
            ], dim=0)
            
            # Diversity
            results['diversity'] = compute_diversity(predictions)
        else:
            best_pred = predictions
        
        # Speed error
        results['speed_error'] = compute_speed_error(best_pred, gt)
        
        # Heading error
        results['heading_error'] = compute_heading_error(best_pred, gt)
        
        # Miss rates at different thresholds
        for threshold in [1.0, 2.0, 3.0]:
            results[f'miss_rate_{threshold}m'] = compute_miss_rate(
                best_pred, gt, threshold
            )
        
        return results
    
    def save_results(
        self,
        path: str,
        format: str = 'json',
        include_config: bool = True,
    ):
        """
        Save evaluation results to file.
        
        Args:
            path: Output file path
            format: 'json', 'csv', or 'latex'
            include_config: Whether to include model config
        """
        results = self.metrics.compute()
        
        # Add metadata
        results['timestamp'] = datetime.now().isoformat()
        results['k_samples'] = self.k_samples
        results['stochastic'] = self.stochastic
        
        if include_config and hasattr(self.model, 'config'):
            results['model_config'] = self.model.config
        
        # Save based on format
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format == 'csv':
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        writer.writerow([key, value])
        
        elif format == 'latex':
            self._save_latex_table(path, results)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results saved to {path}")
    
    def _save_latex_table(self, path: str, results: Dict):
        """Save results as LaTeX table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
        ]
        
        # Add metrics
        metric_order = ['minADE_20', 'minFDE_20', 'ADE', 'FDE', 
                        'ADE@1.0s', 'ADE@2.0s', 'ADE@4.8s',
                        'MissRate@2.0m', 'Diversity']
        
        for metric in metric_order:
            if metric in results:
                value = results[metric]
                if isinstance(value, float):
                    lines.append(f"{metric} & {value:.4f} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Trajectory Prediction Results}",
            r"\label{tab:results}",
            r"\end{table}",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    
    def save_predictions(self, path: str):
        """
        Save all predictions to file.
        
        Args:
            path: Output path (.npz format)
        """
        if not self.all_predictions:
            logger.warning("No predictions saved. Run evaluate(save_predictions=True) first.")
            return
        
        np.savez(
            path,
            predictions=np.concatenate(self.all_predictions, axis=1 if self.all_predictions[0].ndim == 4 else 0),
            observations=np.concatenate(self.all_observations, axis=0),
            ground_truth=np.concatenate(self.all_ground_truth, axis=0),
        )
        
        logger.info(f"Predictions saved to {path}")
    
    def generate_visualizations(
        self,
        output_dir: str,
        num_samples: int = 20,
        include_all_samples: bool = False,
    ):
        """
        Generate visualization plots.
        
        Args:
            output_dir: Output directory
            num_samples: Number of samples to visualize
            include_all_samples: Whether to show all K samples
        """
        if not self.all_predictions:
            logger.warning("No predictions saved. Run evaluate(save_predictions=True) first.")
            return
        
        # Import visualization utilities
        try:
            from ..utils.visualization import visualize_trajectory, visualize_prediction
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Visualization requires matplotlib")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Concatenate all data
        all_pred = np.concatenate(self.all_predictions, axis=1 if self.all_predictions[0].ndim == 4 else 0)
        all_obs = np.concatenate(self.all_observations, axis=0)
        all_gt = np.concatenate(self.all_ground_truth, axis=0)
        
        # Sample indices
        total = all_obs.shape[0]
        indices = np.random.choice(total, min(num_samples, total), replace=False)
        
        for i, idx in enumerate(indices):
            obs = all_obs[idx]
            gt = all_gt[idx]
            
            if all_pred.ndim == 4:
                # Multi-sample: [K, batch, pred_len, 2]
                pred = all_pred[:, idx]  # [K, pred_len, 2]
                
                if include_all_samples:
                    visualize_prediction(
                        past_traj=obs,
                        predictions=pred,
                        gt_traj=gt,
                        title=f"Sample {idx}",
                        save_path=os.path.join(output_dir, f"sample_{i:03d}_all.png"),
                    )
                
                # Best sample only
                ade_per_k = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
                best_k = ade_per_k.argmin()
                best_pred = pred[best_k]
            else:
                # Single prediction
                best_pred = all_pred[idx]
            
            visualize_trajectory(
                past_traj=obs,
                pred_traj=best_pred,
                gt_traj=gt,
                title=f"Sample {idx}",
                save_path=os.path.join(output_dir, f"sample_{i:03d}.png"),
            )
        
        logger.info(f"Visualizations saved to {output_dir}")


class ModelComparator:
    """
    Compare multiple trajectory prediction models.
    
    Usage:
        comparator = ModelComparator(test_loader, device='cuda')
        comparator.add_model('Social-Pose', model1)
        comparator.add_model('Baseline', model2)
        
        results = comparator.compare()
        comparator.save_comparison('results/comparison.json')
    """
    
    def __init__(
        self,
        test_loader: torch.utils.data.DataLoader,
        k_samples: int = 20,
        device: str = 'cuda',
    ):
        """
        Initialize comparator.
        
        Args:
            test_loader: Test data loader (shared across models)
            k_samples: Number of samples for Best-of-K
            device: Compute device
        """
        self.test_loader = test_loader
        self.k_samples = k_samples
        self.device = device
        
        self.models = {}
        self.results = {}
    
    def add_model(
        self,
        name: str,
        model: torch.nn.Module,
        stochastic: bool = True,
    ):
        """
        Add model to comparison.
        
        Args:
            name: Model name
            model: Trained model
            stochastic: Whether model is stochastic
        """
        self.models[name] = {
            'model': model,
            'stochastic': stochastic,
        }
        logger.info(f"Added model: {name}")
    
    def compare(
        self,
        progress_bar: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comparison across all models.
        
        Returns:
            Dictionary of {model_name: metrics}
        """
        self.results = {}
        
        for name, model_info in self.models.items():
            logger.info(f"Evaluating {name}...")
            
            evaluator = Evaluator(
                model=model_info['model'],
                test_loader=self.test_loader,
                k_samples=self.k_samples,
                device=self.device,
                stochastic=model_info['stochastic'],
            )
            
            self.results[name] = evaluator.evaluate(progress_bar=progress_bar)
        
        return self.results
    
    def get_comparison_table(self) -> str:
        """
        Get comparison as formatted table.
        
        Returns:
            Formatted string table
        """
        if not self.results:
            return "No results. Run compare() first."
        
        # Get all metrics
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result.keys())
        
        # Filter to numeric metrics
        numeric_metrics = [m for m in all_metrics 
                         if any(isinstance(self.results[n].get(m), (int, float)) 
                               for n in self.results)]
        
        # Build table
        header = ['Model'] + list(self.results.keys())
        lines = [' | '.join(header)]
        lines.append('-' * len(lines[0]))
        
        for metric in sorted(numeric_metrics):
            row = [metric]
            for name in self.results:
                value = self.results[name].get(metric, '-')
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            lines.append(' | '.join(row))
        
        return '\n'.join(lines)
    
    def save_comparison(
        self,
        path: str,
        format: str = 'json',
    ):
        """
        Save comparison results.
        
        Args:
            path: Output path
            format: 'json', 'csv', or 'latex'
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        
        elif format == 'csv':
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                model_names = list(self.results.keys())
                writer.writerow(['Metric'] + model_names)
                
                # Get all metrics
                all_metrics = set()
                for result in self.results.values():
                    all_metrics.update(result.keys())
                
                for metric in sorted(all_metrics):
                    row = [metric]
                    for name in model_names:
                        value = self.results[name].get(metric, '')
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    writer.writerow(row)
        
        elif format == 'latex':
            self._save_latex_comparison(path)
        
        logger.info(f"Comparison saved to {path}")
    
    def _save_latex_comparison(self, path: str):
        """Save comparison as LaTeX table."""
        model_names = list(self.results.keys())
        
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l" + "c" * len(model_names) + "}",
            r"\toprule",
            "Metric & " + " & ".join(model_names) + r" \\",
            r"\midrule",
        ]
        
        # Add metrics
        metrics_order = ['minADE_20', 'minFDE_20', 'ADE@1.0s', 'ADE@2.0s', 
                        'ADE@4.8s', 'MissRate@2.0m', 'Diversity']
        
        for metric in metrics_order:
            values = []
            min_val = float('inf')
            
            for name in model_names:
                val = self.results[name].get(metric, None)
                if isinstance(val, float):
                    values.append(val)
                    if val < min_val:
                        min_val = val
                else:
                    values.append(None)
            
            # Format row with best value bolded
            row_values = []
            for val in values:
                if val is None:
                    row_values.append("-")
                elif val == min_val:
                    row_values.append(f"\\textbf{{{val:.4f}}}")
                else:
                    row_values.append(f"{val:.4f}")
            
            lines.append(f"{metric} & " + " & ".join(row_values) + r" \\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Model Comparison}",
            r"\label{tab:comparison}",
            r"\end{table}",
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


class AblationStudy:
    """
    Conduct ablation studies on model components.
    
    Usage:
        ablation = AblationStudy(
            base_config=config,
            test_loader=test_loader,
            model_class=SocialPoseModel,
        )
        
        # Define ablations
        ablation.add_ablation('No Pose', {'use_pose': False})
        ablation.add_ablation('No Velocity', {'use_velocity': False})
        
        results = ablation.run()
        ablation.save_results('results/ablation.json')
    """
    
    def __init__(
        self,
        base_config: Dict,
        test_loader: torch.utils.data.DataLoader,
        model_class: type,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        k_samples: int = 20,
        device: str = 'cuda',
    ):
        """
        Initialize ablation study.
        
        Args:
            base_config: Base model configuration
            test_loader: Test data loader
            model_class: Model class to instantiate
            train_loader: Training data loader (for training ablated models)
            k_samples: Number of samples for evaluation
            device: Compute device
        """
        self.base_config = base_config.copy()
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.model_class = model_class
        self.k_samples = k_samples
        self.device = device
        
        self.ablations = {}
        self.results = {}
    
    def add_ablation(
        self,
        name: str,
        config_changes: Dict,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Add an ablation configuration.
        
        Args:
            name: Ablation name
            config_changes: Dictionary of config changes
            checkpoint_path: Path to pre-trained checkpoint (optional)
        """
        self.ablations[name] = {
            'config_changes': config_changes,
            'checkpoint_path': checkpoint_path,
        }
    
    def run(
        self,
        train_epochs: int = 100,
        progress_bar: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study.
        
        Args:
            train_epochs: Number of training epochs if training needed
            progress_bar: Show progress bar
        
        Returns:
            Results for each ablation
        """
        self.results = {}
        
        # Evaluate base model
        logger.info("Evaluating base model...")
        base_model = self.model_class.from_config(self.base_config)
        
        # Load checkpoint or train
        # (Simplified - assumes checkpoints are provided)
        
        for name, ablation_info in self.ablations.items():
            logger.info(f"Running ablation: {name}")
            
            # Create modified config
            config = self.base_config.copy()
            self._deep_update(config, ablation_info['config_changes'])
            
            # Create model
            model = self.model_class.from_config(config)
            
            # Load checkpoint if provided
            if ablation_info['checkpoint_path']:
                checkpoint = torch.load(
                    ablation_info['checkpoint_path'],
                    map_location=self.device
                )
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            evaluator = Evaluator(
                model=model,
                test_loader=self.test_loader,
                k_samples=self.k_samples,
                device=self.device,
            )
            
            self.results[name] = evaluator.evaluate(progress_bar=progress_bar)
        
        return self.results
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Deep update dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def save_results(self, path: str):
        """Save ablation results."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'base_config': self.base_config,
                'ablations': {
                    name: info['config_changes']
                    for name, info in self.ablations.items()
                },
                'results': self.results,
            }, f, indent=2, default=str)
        
        logger.info(f"Ablation results saved to {path}")
