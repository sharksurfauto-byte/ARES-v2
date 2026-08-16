"""Global Reliability Model (GRM) for ARES V2 Phase 4.

Evaluates high-level input feasibility, domain placement, and global query reliability
from normalized intermediate transformer representations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalReliabilityModel(nn.Module):
    """Global Reliability Model (GRM).

    Features:
    1. LayerNorm standardization (prevents magnitude scale bias across network depth).
    2. Linear Bottleneck projection (D=3584 -> bottleneck_dim=128).
    3. Optional Layer Depth Embedding (enables depth-aware probing).
    4. Multi-head output:
       - Domain Classifier (4-class logits: code, general, math, science).
       - Global Feasibility Estimator (R_global in [0, 1]).
    """

    def __init__(
        self,
        input_dim: int = 3584,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        num_domains: int = 4,
        dropout: float = 0.1,
        use_layer_depth_embedding: bool = True,
        num_layers: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
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

        # Hidden representation layer
        self.fc_hidden = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Output heads
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.feasibility_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for GRM.

        Args:
            x: Input representations of shape (batch_size, input_dim)
            layer_idx: Optional layer index or tensor of layer indices for depth conditioning

        Returns:
            Dict containing:
                - 'domain_logits': Shape (batch_size, num_domains)
                - 'domain_probs': Shape (batch_size, num_domains)
                - 'feasibility': Shape (batch_size, 1) in [0, 1]
                - 'bottleneck': Shape (batch_size, bottleneck_dim)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 1. LayerNorm standardization
        h_norm = self.layer_norm(x)

        # 2. Bottleneck projection
        z = self.bottleneck(h_norm)

        # 3. Add depth embedding if configured and available
        if self.use_layer_depth_embedding and layer_idx is not None:
            if isinstance(layer_idx, int):
                # Resolve negative layer index (e.g. -1 -> num_layers - 1)
                idx = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx
                idx = max(0, min(idx, self.num_layers - 1))
                layer_tensor = torch.full((x.shape[0],), idx, dtype=torch.long, device=x.device)
            else:
                # Tensor of layer indices
                layer_tensor = layer_idx.to(device=x.device, dtype=torch.long)
                layer_tensor = torch.clamp(layer_tensor, 0, self.num_layers - 1)

            depth_embed = self.depth_embedding(layer_tensor)
            z = z + depth_embed

        # 4. Hidden MLP
        h = self.act(self.fc_hidden(z))
        h = self.dropout(h)

        # 5. Output heads
        domain_logits = self.domain_head(h)
        domain_probs = F.softmax(domain_logits, dim=-1)
        feasibility_raw = self.feasibility_head(h)
        feasibility = torch.sigmoid(feasibility_raw)

        return {
            "domain_logits": domain_logits,
            "domain_probs": domain_probs,
            "feasibility": feasibility,
            "bottleneck": z,
        }
