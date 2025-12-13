"""
Trajectory Decoder with Speed Adaptation.
Generates future trajectory predictions with adaptive spacing based on predicted speed.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AdaptiveTrajectoryDecoder(nn.Module):
    """
    Decode future trajectory with speed-adaptive output.
    
    HOW IT PREDICTS:
    1. Initialize decoder hidden state from encoder
    2. At each timestep:
       a. Predict position offset (displacement)
       b. Predict speed for this timestep
       c. Use speed to determine temporal weight for loss
    3. Autoregressive: feed prediction back as input
    
    ADAPTIVE SPACING:
    - Model predicts at fixed time intervals (0.4s)
    - Speed prediction allows post-processing to resample
    - Slow walking → points closer together spatially
    - Running → points farther apart spatially
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        output_dim: int = 2,
        pred_len: int = 12,
        predict_speed: bool = True,
        predict_uncertainty: bool = True
    ):
        """
        Args:
            hidden_dim: LSTM hidden dimension
            embedding_dim: Position embedding dimension
            output_dim: Output dimension (2 for x, y)
            pred_len: Number of prediction timesteps
            predict_speed: Whether to predict speed at each step
            predict_uncertainty: Whether to predict uncertainty
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.predict_speed = predict_speed
        self.predict_uncertainty = predict_uncertainty
        
        # Input embedding (previous position)
        self.input_embed = nn.Sequential(
            nn.Linear(output_dim, embedding_dim),
            nn.ReLU()
        )
        
        # LSTM decoder cell
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        
        # Position output head (predicts displacement)
        self.position_head = nn.Linear(hidden_dim, output_dim)
        
        # Speed output head (for adaptive spacing)
        if predict_speed:
            self.speed_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Speed must be positive
            )
        
        # Uncertainty output head (for probabilistic prediction)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_cell: torch.Tensor,
        last_position: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        gt_trajectory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decode future trajectory.
        
        Args:
            encoder_hidden: [batch, hidden_dim] - encoded motion
            encoder_cell: [batch, hidden_dim] - LSTM cell state
            last_position: [batch, 2] - last observed position
            social_context: [batch, pooling_dim] - social pooling output
            teacher_forcing_ratio: Probability of using ground truth during training
            gt_trajectory: [batch, pred_len, 2] - ground truth for teacher forcing
        
        Returns:
            predictions: [batch, pred_len, 2]
            speeds: [batch, pred_len, 1] (if predict_speed)
            uncertainties: [batch, pred_len, 2] (if predict_uncertainty)
        """
        batch_size = encoder_hidden.size(0)
        
        # Initialize hidden states
        hidden = encoder_hidden
        cell = encoder_cell
        
        # If social context provided, incorporate it
        if social_context is not None:
            # Add social context to hidden state
            hidden = hidden + social_context[:, :self.hidden_dim]
        
        # Start from last observed position
        current_pos = last_position
        
        # Storage for outputs
        predictions = []
        speeds = [] if self.predict_speed else None
        uncertainties = [] if self.predict_uncertainty else None
        
        for t in range(self.pred_len):
            # Embed current position
            embedded = self.input_embed(current_pos)
            
            # LSTM step
            hidden, cell = self.lstm_cell(embedded, (hidden, cell))
            
            # Predict displacement from hidden state
            displacement = self.position_head(hidden)
            predicted_pos = current_pos + displacement
            
            # Predict speed if enabled
            if self.predict_speed:
                speed = self.speed_head(hidden)
                speeds.append(speed)
            
            # Predict uncertainty if enabled
            if self.predict_uncertainty:
                # Output log variance, exp to get variance
                log_var = self.uncertainty_head(hidden)
                uncertainty = torch.exp(log_var)
                uncertainties.append(uncertainty)
            
            predictions.append(predicted_pos)
            
            # Prepare next input (teacher forcing or predicted)
            if self.training and gt_trajectory is not None:
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher:
                    current_pos = gt_trajectory[:, t, :]
                else:
                    current_pos = predicted_pos.detach()
            else:
                current_pos = predicted_pos
        
        # Stack outputs
        predictions = torch.stack(predictions, dim=1)  # [B, T, 2]
        
        if self.predict_speed:
            speeds = torch.stack(speeds, dim=1)  # [B, T, 1]
        
        if self.predict_uncertainty:
            uncertainties = torch.stack(uncertainties, dim=1)  # [B, T, 2]
        
        return predictions, speeds, uncertainties


class MultiModalDecoder(nn.Module):
    """
    Multi-modal decoder that generates multiple trajectory hypotheses.
    Uses CVAE-style latent sampling for diversity.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_modes: int = 20,
        pred_len: int = 12
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            latent_dim: Latent space dimension
            num_modes: Number of trajectory modes to generate
            pred_len: Prediction length
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        
        # Latent space encoder (for training with ground truth)
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim + pred_len * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log variance
        )
        
        # Prior network (for inference)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # Cell state projection (to match decoder hidden dim)
        self.cell_proj = nn.Linear(hidden_dim, hidden_dim + latent_dim)
        
        # Trajectory decoder
        self.decoder = AdaptiveTrajectoryDecoder(
            hidden_dim=hidden_dim + latent_dim,
            embedding_dim=64,
            pred_len=pred_len
        )
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_cell: torch.Tensor,
        last_position: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate multiple trajectory hypotheses.
        
        Args:
            encoder_hidden: [batch, hidden_dim]
            encoder_cell: [batch, hidden_dim]
            last_position: [batch, 2]
            gt_trajectory: [batch, pred_len, 2] - for training
            num_samples: Number of samples to generate
        
        Returns:
            predictions: [num_samples, batch, pred_len, 2]
            extras: dict with KL loss, etc.
        """
        batch_size = encoder_hidden.size(0)
        
        if self.training and gt_trajectory is not None:
            # Encode ground truth to get posterior
            gt_flat = gt_trajectory.view(batch_size, -1)
            latent_input = torch.cat([encoder_hidden, gt_flat], dim=-1)
            latent_params = self.latent_encoder(latent_input)
            
            mu_q = latent_params[:, :self.latent_dim]
            log_var_q = latent_params[:, self.latent_dim:]
            
            # Get prior
            prior_params = self.prior(encoder_hidden)
            mu_p = prior_params[:, :self.latent_dim]
            log_var_p = prior_params[:, self.latent_dim:]
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + log_var_q - log_var_p - 
                (torch.exp(log_var_q) + (mu_q - mu_p)**2) / torch.exp(log_var_p),
                dim=-1
            ).mean()
            
            # Sample from posterior
            z = self.reparameterize(mu_q, log_var_q)
            
            # Decode - project cell state to match decoder hidden dim
            decoder_input = torch.cat([encoder_hidden, z], dim=-1)
            decoder_cell = self.cell_proj(encoder_cell)
            predictions, speeds, _ = self.decoder(
                decoder_input, decoder_cell, last_position, 
                gt_trajectory=gt_trajectory, teacher_forcing_ratio=0.5
            )
            
            return predictions.unsqueeze(0), {'kl_loss': kl_loss, 'speeds': speeds}
        
        else:
            # Sample from prior for inference
            prior_params = self.prior(encoder_hidden)
            mu_p = prior_params[:, :self.latent_dim]
            log_var_p = prior_params[:, self.latent_dim:]
            
            # Project cell state once
            decoder_cell = self.cell_proj(encoder_cell)
            
            all_predictions = []
            
            for _ in range(num_samples):
                z = self.reparameterize(mu_p, log_var_p)
                decoder_input = torch.cat([encoder_hidden, z], dim=-1)
                predictions, _, _ = self.decoder(
                    decoder_input, decoder_cell, last_position
                )
                all_predictions.append(predictions)
            
            return torch.stack(all_predictions, dim=0), {}


class GoalConditionedDecoder(nn.Module):
    """
    Goal-conditioned decoder for long-horizon prediction.
    First predicts goals, then generates path to goals.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_goals: int = 5,
        pred_len: int = 12
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_goals: Number of goal candidates
            pred_len: Prediction length
        """
        super().__init__()
        
        self.num_goals = num_goals
        self.pred_len = pred_len
        
        # Goal predictor
        self.goal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_goals * 2)  # x, y for each goal
        )
        
        # Goal probability
        self.goal_prob = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_goals),
            nn.Softmax(dim=-1)
        )
        
        # Path decoder (conditioned on goal)
        self.path_decoder = nn.LSTM(
            input_size=2 + 2,  # current pos + goal
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.position_head = nn.Linear(hidden_dim, 2)
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_cell: torch.Tensor,
        last_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Goal-conditioned prediction.
        
        Args:
            encoder_hidden: [batch, hidden_dim]
            encoder_cell: [batch, hidden_dim]
            last_position: [batch, 2]
        
        Returns:
            predictions: [batch, num_goals, pred_len, 2]
            goals: [batch, num_goals, 2]
            goal_probs: [batch, num_goals]
        """
        batch_size = encoder_hidden.size(0)
        
        # Predict goals
        goal_coords = self.goal_predictor(encoder_hidden)
        goals = goal_coords.view(batch_size, self.num_goals, 2)
        goals = goals + last_position.unsqueeze(1)  # Relative to last position
        
        # Goal probabilities
        goal_probs = self.goal_prob(encoder_hidden)
        
        # Generate path to each goal
        all_paths = []
        
        for g in range(self.num_goals):
            goal = goals[:, g, :]  # [B, 2]
            
            # Initialize path decoder
            current_pos = last_position
            hidden = encoder_hidden.unsqueeze(0)
            cell = encoder_cell.unsqueeze(0)
            
            path = []
            for t in range(self.pred_len):
                # Progress towards goal
                direction_to_goal = goal - current_pos
                
                # Input: current position + goal
                decoder_input = torch.cat([current_pos, goal], dim=-1)
                decoder_input = decoder_input.unsqueeze(1)
                
                output, (hidden, cell) = self.path_decoder(
                    decoder_input, (hidden, cell)
                )
                
                displacement = self.position_head(output.squeeze(1))
                predicted_pos = current_pos + displacement
                
                path.append(predicted_pos)
                current_pos = predicted_pos
            
            all_paths.append(torch.stack(path, dim=1))
        
        predictions = torch.stack(all_paths, dim=1)  # [B, num_goals, T, 2]
        
        return predictions, goals, goal_probs