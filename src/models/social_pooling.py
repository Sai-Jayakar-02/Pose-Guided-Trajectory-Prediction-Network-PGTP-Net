"""
Social Pooling Module for multi-agent interaction modeling.
Captures collision avoidance, group behavior, and social forces.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SocialPoolingModule(nn.Module):
    """
    Pool information from neighboring pedestrians.
    
    HOW IT MODELS INTERACTION:
    - For each person i, compute relative positions to all others
    - Concatenate relative position with neighbor's hidden state
    - Process through MLP
    - Max-pool across all neighbors to get fixed-size representation
    
    This captures: collision avoidance, group walking, following behavior
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        pooling_dim: int = 256,
        bottleneck_dim: int = 64
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states from encoder
            pooling_dim: Dimension of pooled output
            bottleneck_dim: Bottleneck dimension for relative position encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_dim = pooling_dim
        
        # Encode relative position
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, bottleneck_dim),
            nn.ReLU()
        )
        
        # Process neighbor information (hidden state + relative position)
        self.neighbor_embed = nn.Sequential(
            nn.Linear(hidden_dim + bottleneck_dim, pooling_dim),
            nn.ReLU(),
            nn.Linear(pooling_dim, pooling_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(pooling_dim, pooling_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool social context for each agent.
        
        Args:
            hidden_states: [batch, num_agents, hidden_dim]
            positions: [batch, num_agents, 2] - last observed positions
            mask: [batch, num_agents] - valid agent mask
        
        Returns:
            pooled: [batch, num_agents, pooling_dim]
        """
        batch_size, num_agents, hidden_dim = hidden_states.shape
        
        if num_agents == 1:
            # Single agent, no social pooling needed
            return torch.zeros(
                batch_size, 1, self.pooling_dim,
                device=hidden_states.device
            )
        
        pooled_vectors = []
        
        for i in range(num_agents):
            # Get this agent's position
            agent_pos = positions[:, i:i+1, :]  # [B, 1, 2]
            
            # Compute relative positions to all others
            rel_pos = positions - agent_pos  # [B, N, 2]
            
            # Encode relative positions
            rel_pos_encoded = self.rel_pos_encoder(rel_pos)  # [B, N, bottleneck]
            
            # Concatenate with hidden states
            neighbor_input = torch.cat([hidden_states, rel_pos_encoded], dim=-1)
            
            # Embed neighbor information
            embedded = self.neighbor_embed(neighbor_input)  # [B, N, pooling_dim]
            
            # Create mask to exclude self
            self_mask = torch.ones(num_agents, dtype=torch.bool, device=embedded.device)
            self_mask[i] = False
            
            # Apply additional mask if provided
            if mask is not None:
                combined_mask = self_mask & mask
            else:
                combined_mask = self_mask
            
            # Max pool across neighbors (excluding self)
            if combined_mask.sum() > 0:
                pooled = embedded[:, combined_mask, :].max(dim=1)[0]  # [B, pooling_dim]
            else:
                pooled = torch.zeros(
                    batch_size, self.pooling_dim, 
                    device=embedded.device
                )
            
            pooled_vectors.append(pooled)
        
        pooled_out = torch.stack(pooled_vectors, dim=1)  # [B, N, pooling_dim]
        pooled_out = self.output_proj(pooled_out)
        
        return pooled_out
    
    def forward_single(
        self,
        target_hidden: torch.Tensor,
        target_pos: torch.Tensor,
        neighbor_hidden: torch.Tensor,
        neighbor_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool social context for a single target agent.
        
        Args:
            target_hidden: [batch, hidden_dim] - target's hidden state
            target_pos: [batch, 2] - target's position
            neighbor_hidden: [batch, num_neighbors, hidden_dim]
            neighbor_pos: [batch, num_neighbors, 2]
        
        Returns:
            pooled: [batch, pooling_dim]
        """
        batch_size = target_hidden.size(0)
        num_neighbors = neighbor_hidden.size(1)
        
        if num_neighbors == 0:
            return torch.zeros(
                batch_size, self.pooling_dim,
                device=target_hidden.device
            )
        
        # Compute relative positions
        rel_pos = neighbor_pos - target_pos.unsqueeze(1)  # [B, N, 2]
        
        # Encode
        rel_pos_encoded = self.rel_pos_encoder(rel_pos)
        
        # Combine with neighbor hidden states
        neighbor_input = torch.cat([neighbor_hidden, rel_pos_encoded], dim=-1)
        embedded = self.neighbor_embed(neighbor_input)
        
        # Max pool
        pooled = embedded.max(dim=1)[0]  # [B, pooling_dim]
        pooled = self.output_proj(pooled)
        
        return pooled


class GridBasedPooling(nn.Module):
    """
    Grid-based social pooling (as in Social-LSTM).
    Creates a spatial grid around each agent and pools neighbor hidden states.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        grid_size: int = 8,
        neighborhood_size: float = 4.0
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            grid_size: Size of the grid (grid_size x grid_size)
            neighborhood_size: Physical size of neighborhood (meters)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.cell_size = neighborhood_size / grid_size
        
        # Embedding for grid
        self.grid_embed = nn.Linear(hidden_dim * grid_size * grid_size, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Grid-based pooling.
        
        Args:
            hidden_states: [batch, num_agents, hidden_dim]
            positions: [batch, num_agents, 2]
        
        Returns:
            pooled: [batch, num_agents, hidden_dim]
        """
        batch_size, num_agents, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        pooled_list = []
        
        for i in range(num_agents):
            # Initialize grid
            grid = torch.zeros(
                batch_size, self.grid_size, self.grid_size, hidden_dim,
                device=device
            )
            
            # Get this agent's position
            agent_pos = positions[:, i, :]  # [B, 2]
            
            # Compute relative positions of all neighbors
            rel_pos = positions - agent_pos.unsqueeze(1)  # [B, N, 2]
            
            # Map to grid coordinates
            grid_x = ((rel_pos[:, :, 0] + self.neighborhood_size / 2) / 
                     self.cell_size).long()
            grid_y = ((rel_pos[:, :, 1] + self.neighborhood_size / 2) / 
                     self.cell_size).long()
            
            # Clamp to valid range
            grid_x = grid_x.clamp(0, self.grid_size - 1)
            grid_y = grid_y.clamp(0, self.grid_size - 1)
            
            # Place hidden states in grid (sum if multiple in same cell)
            for j in range(num_agents):
                if j == i:
                    continue
                for b in range(batch_size):
                    gx, gy = grid_x[b, j], grid_y[b, j]
                    grid[b, gx, gy] += hidden_states[b, j]
            
            # Flatten grid and embed
            grid_flat = grid.view(batch_size, -1)  # [B, G*G*H]
            pooled = self.grid_embed(grid_flat)  # [B, H]
            pooled_list.append(pooled)
        
        return torch.stack(pooled_list, dim=1)


class AttentionPooling(nn.Module):
    """
    Attention-based social pooling.
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Position encoding
        self.pos_encoder = nn.Linear(2, hidden_dim // 2)
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention-based pooling.
        
        Args:
            hidden_states: [batch, num_agents, hidden_dim]
            positions: [batch, num_agents, 2]
        
        Returns:
            pooled: [batch, num_agents, hidden_dim]
        """
        batch_size, num_agents, _ = hidden_states.shape
        
        # Encode positions
        pos_encoded = self.pos_encoder(positions)  # [B, N, hidden//2]
        
        # Concatenate hidden states with position encoding
        # Pad position encoding to match hidden_dim
        pos_padded = torch.cat([
            pos_encoded, 
            torch.zeros_like(pos_encoded)
        ], dim=-1)
        
        enhanced = hidden_states + pos_padded
        
        # Self-attention across agents
        attended, _ = self.attention(enhanced, enhanced, enhanced)
        
        # Output projection
        output = self.output_proj(attended)
        
        return output