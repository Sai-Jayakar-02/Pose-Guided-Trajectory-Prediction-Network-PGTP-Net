"""
Pose Transformer Encoder.
Extracts spatiotemporal features from body pose that indicate movement intention.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in 'Attention is All You Need'.
    Adds temporal information to pose embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PoseTransformerEncoder(nn.Module):
    """
    Transformer encoder for pose sequence.
    
    WHY TRANSFORMER FOR POSE:
    - Self-attention captures relationships between ALL joints simultaneously
    - Attention weights reveal which joints matter for prediction
      (Research shows: arms, legs, shoulders have highest attention)
    - Positional encoding preserves temporal ordering
    
    Input: Pose keypoints [batch, obs_len, num_joints, pose_dim]
    Output: Pose features [batch, hidden_dim]
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        """
        Args:
            num_joints: Number of body keypoints (17 for COCO, 22 for JTA)
            input_dim: Dimension of each joint (2 for 2D, 3 for 3D)
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Flatten joints and embed
        # Each frame: [J joints × D coords] -> [hidden_dim]
        self.joint_embedding = nn.Linear(num_joints * input_dim, hidden_dim)
        
        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        pose: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode pose sequence.
        
        Args:
            pose: [batch, seq_len, num_joints, pose_dim]
            mask: Optional attention mask [batch, seq_len]
        
        Returns:
            pose_features: [batch, hidden_dim]
        """
        batch_size, seq_len, num_joints, pose_dim = pose.shape
        
        # Flatten joints: [B, T, J, D] -> [B, T, J*D]
        pose_flat = pose.view(batch_size, seq_len, -1)
        
        # Embed: [B, T, J*D] -> [B, T, hidden_dim]
        embedded = self.joint_embedding(pose_flat)
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)
        
        # Transformer encoding
        # Output: [B, T, hidden_dim]
        encoded = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Take last timestep output (captures most recent pose context)
        # Alternative: mean pooling across time
        output = encoded[:, -1, :]  # [B, hidden_dim]
        
        # Final projection
        output = self.output_proj(output)
        output = self.layer_norm(output)
        
        return output
    
    def forward_with_attention(
        self,
        pose: torch.Tensor
    ) -> tuple:
        """
        Forward pass with attention weights (for visualization/analysis).
        
        Args:
            pose: [batch, seq_len, num_joints, pose_dim]
        
        Returns:
            output: [batch, hidden_dim]
            attention_weights: List of attention matrices
        """
        batch_size, seq_len, num_joints, pose_dim = pose.shape
        
        pose_flat = pose.view(batch_size, seq_len, -1)
        embedded = self.joint_embedding(pose_flat)
        embedded = self.pos_encoder(embedded)
        
        # Manual forward through transformer to capture attention
        attention_weights = []
        x = embedded
        
        for layer in self.transformer.layers:
            # Self-attention with attention weights
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True
            )
            attention_weights.append(attn_weight.detach())
            
            # Rest of transformer layer
            x = layer.norm1(x + attn_output)
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            x = layer.norm2(x + ff_output)
        
        output = self.output_proj(x[:, -1, :])
        output = self.layer_norm(output)
        
        return output, attention_weights


class JointWiseEncoder(nn.Module):
    """
    Alternative pose encoder that processes each joint separately
    before combining with attention.
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # Per-joint temporal encoder
        self.joint_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 4,
            num_layers=1,
            batch_first=True
        )
        
        # Cross-joint attention
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 4,
            num_heads=num_heads // 4,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(num_joints * (hidden_dim // 4), hidden_dim)
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Joint-wise encoding.
        
        Args:
            pose: [batch, seq_len, num_joints, input_dim]
        
        Returns:
            features: [batch, hidden_dim]
        """
        batch_size, seq_len, num_joints, input_dim = pose.shape
        
        # Process each joint's temporal sequence
        joint_features = []
        for j in range(num_joints):
            joint_seq = pose[:, :, j, :]  # [B, T, D]
            _, (h, _) = self.joint_encoder(joint_seq)
            joint_features.append(h[-1])  # [B, hidden//4]
        
        # Stack joint features: [B, J, hidden//4]
        joint_features = torch.stack(joint_features, dim=1)
        
        # Cross-joint attention
        attended, _ = self.joint_attention(
            joint_features, joint_features, joint_features
        )
        
        # Flatten and project
        flattened = attended.view(batch_size, -1)
        output = self.output_proj(flattened)
        
        return output


class SpatialTemporalPoseEncoder(nn.Module):
    """
    Pose encoder with separate spatial and temporal attention.
    First models joint relationships (spatial), then temporal evolution.
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # Joint embedding
        self.joint_embed = nn.Linear(input_dim, hidden_dim // 2)
        
        # Spatial attention (across joints in each frame)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=num_heads // 2,
            batch_first=True
        )
        
        # Temporal attention (across frames)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=num_heads // 2,
            batch_first=True
        )
        
        # Output
        self.output_proj = nn.Linear(hidden_dim // 2, hidden_dim)
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Spatial-temporal encoding.
        
        Args:
            pose: [batch, seq_len, num_joints, input_dim]
        
        Returns:
            features: [batch, hidden_dim]
        """
        B, T, J, D = pose.shape
        
        # Embed joints: [B, T, J, D] -> [B, T, J, hidden//2]
        embedded = self.joint_embed(pose)
        
        # Spatial attention (per frame)
        # Reshape: [B*T, J, hidden//2]
        spatial_in = embedded.view(B * T, J, -1)
        spatial_out, _ = self.spatial_attention(
            spatial_in, spatial_in, spatial_in
        )
        
        # Mean pool across joints: [B*T, hidden//2]
        spatial_pooled = spatial_out.mean(dim=1)
        
        # Reshape for temporal: [B, T, hidden//2]
        temporal_in = spatial_pooled.view(B, T, -1)
        
        # Temporal attention
        temporal_out, _ = self.temporal_attention(
            temporal_in, temporal_in, temporal_in
        )
        
        # Take last frame
        output = self.output_proj(temporal_out[:, -1, :])
        
        return output
