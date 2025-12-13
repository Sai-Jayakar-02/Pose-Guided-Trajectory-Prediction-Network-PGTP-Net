"""
Trajectory Encoder using LSTM.
Encodes the temporal sequence of past positions to capture motion patterns.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TrajectoryEncoder(nn.Module):
    """
    LSTM encoder for trajectory sequence.
    
    Input: Past trajectory positions [batch, obs_len, 2]
    Output: Hidden state encoding motion history [batch, hidden_dim]
    
    HOW IT LEARNS:
    - The LSTM processes positions sequentially
    - At each timestep, it updates its hidden state based on:
      1. Current position (embedded to higher dimension)
      2. Previous hidden state (memory of past motion)
    - The final hidden state summarizes the entire motion history
    - Backpropagation adjusts weights to minimize prediction error
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Dimension of input (2 for x,y coordinates)
            embedding_dim: Dimension of position embedding
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embed 2D coordinates to higher dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(
        self,
        trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory sequence.
        
        Args:
            trajectory: [batch, seq_len, 2] - (x, y) positions
        
        Returns:
            hidden: [batch, hidden_dim] - encoded motion
            cell: [batch, hidden_dim] - LSTM cell state
        """
        batch_size = trajectory.size(0)
        
        # Embed positions: [B, T, 2] -> [B, T, embedding_dim]
        embedded = self.input_embedding(trajectory)
        
        # Process sequence through LSTM
        # output: [B, T, hidden_dim]
        # hidden: [num_layers, B, hidden_dim]
        # cell: [num_layers, B, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take last layer's hidden state
        hidden = hidden[-1]  # [B, hidden_dim]
        cell = cell[-1]      # [B, hidden_dim]
        
        return hidden, cell
    
    def get_all_hidden_states(
        self,
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Get hidden states at all timesteps (for attention mechanisms).
        
        Args:
            trajectory: [batch, seq_len, 2]
        
        Returns:
            all_hidden: [batch, seq_len, hidden_dim]
        """
        embedded = self.input_embedding(trajectory)
        output, _ = self.lstm(embedded)
        return output


class DisplacementEncoder(nn.Module):
    """
    Alternative encoder that operates on displacements instead of positions.
    Displacements are more informative about motion direction and speed.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(
        self,
        trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory using displacements.
        
        Args:
            trajectory: [batch, seq_len, 2] positions
        
        Returns:
            hidden: [batch, hidden_dim]
            cell: [batch, hidden_dim]
        """
        # Compute displacements: [B, T, 2] -> [B, T-1, 2]
        displacements = trajectory[:, 1:] - trajectory[:, :-1]
        
        # Embed displacements
        embedded = self.embedding(displacements)
        
        # LSTM encoding
        _, (hidden, cell) = self.lstm(embedded)
        
        return hidden[-1], cell[-1]


class BidirectionalTrajectoryEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for trajectory.
    Captures both forward and backward temporal context.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Half for each direction
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Project back to hidden_dim
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional encoding.
        
        Args:
            trajectory: [batch, seq_len, 2]
        
        Returns:
            hidden: [batch, hidden_dim]
            cell: [batch, hidden_dim]
        """
        embedded = self.embedding(trajectory)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        # hidden shape: [2*num_layers, B, hidden_dim//2]
        hidden_forward = hidden[-2]  # Last forward
        hidden_backward = hidden[-1]  # Last backward
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=-1)
        
        cell_forward = cell[-2]
        cell_backward = cell[-1]
        cell_concat = torch.cat([cell_forward, cell_backward], dim=-1)
        
        # Project
        hidden_out = self.output_proj(hidden_concat)
        
        return hidden_out, cell_concat
