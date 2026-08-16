"""Local Reliability Model (LRM) for ARES V2 Phase 4.

Evaluates token-level/context-level representation stability and predicts prediction
correctness probability P(correct | H_token).
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalReliabilityModel(nn.Module):
    """Local Reliability Model (LRM).

    Features:
    1. LayerNorm standardization (prevents magnitude scale bias).
    2. Linear Bottleneck projection (D=3584 -> bottleneck_dim=64).
    3. Optional Layer Depth Embedding (depth-aware local probing).
    4. Correctness Probability Output Head: P(correct | H_token) in [0, 1].
    """

    def __init__(
        self,
        input_dim: int = 3584,
        bottleneck_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_layer_depth_embedding: bool = True,
        num_layers: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.use_layer_depth_embedding = use_layer_depth_embedding
        self.num_layers = num_layers

        # LayerNorm standardization
        self.layer_norm = nn.LayerNorm(input_dim)

        # Bottleneck projection
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)

        # Layer depth embedding
        if use_layer_depth_embedding:
            self.depth_embedding = nn.Embedding(num_layers, bottleneck_dim)
        else:
            self.depth_embedding = None

        # Hidden MLP
        self.fc_hidden = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Correctness probability head
        self.correctness_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for LRM.

        Args:
            x: Input representations of shape (batch_size, input_dim)
            layer_idx: Optional layer index or tensor for depth conditioning

        Returns:
            Dict containing:
                - 'correctness_prob': Shape (batch_size, 1) in [0, 1]
                - 'failure_risk': Shape (batch_size, 1) = 1.0 - correctness_prob
                - 'bottleneck': Shape (batch_size, bottleneck_dim)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Cast input dtype to match module parameter dtype if mismatched
        target_dtype = self.layer_norm.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        # 1. LayerNorm standardization
        h_norm = self.layer_norm(x)

        # 2. Bottleneck projection
        z = self.bottleneck(h_norm)

        # 3. Add depth embedding if configured
        if self.use_layer_depth_embedding and layer_idx is not None:
            if isinstance(layer_idx, int):
                idx = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx
                idx = max(0, min(idx, self.num_layers - 1))
                layer_tensor = torch.full((x.shape[0],), idx, dtype=torch.long, device=x.device)
            else:
                layer_tensor = layer_idx.to(device=x.device, dtype=torch.long)
                layer_tensor = torch.clamp(layer_tensor, 0, self.num_layers - 1)

            depth_embed = self.depth_embedding(layer_tensor)
            z = z + depth_embed

        # 4. Hidden MLP
        h = self.act(self.fc_hidden(z))
        h = self.dropout(h)

        # 5. Correctness head
        correctness_raw = self.correctness_head(h)
        correctness_prob = torch.sigmoid(correctness_raw)
        failure_risk = 1.0 - correctness_prob

        return {
            "correctness_prob": correctness_prob,
            "failure_risk": failure_risk,
            "bottleneck": z,
        }
