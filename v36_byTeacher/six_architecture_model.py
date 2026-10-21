"""GRU used by the six M/G/K architecture ablation.

This module intentionally removes the old motion-polynomial feedback path.
The GRU is a current-frame position refiner: it consumes the current stage
coordinate, current visual uncertainty, temporal UAV embeddings, and its own
hidden state. It outputs a current-frame correction and corrected position.
It never receives the previous/final localization position and never predicts
a future search-center displacement.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import config


@dataclass
class PositionGRUOutput:
    corrected_xy: torch.Tensor
    correction_xy: torch.Tensor
    hidden: torch.Tensor


class PositionRefinementGRU(nn.Module):
    """Current-frame recurrent visual position refiner.

    Inputs per frame:
      - stage_xy: current coordinate produced by the module immediately before G
      - variance_xy: current visual measurement variance (diag x/y)
      - z_uav: current projected UAV embedding
      - previous_z_uav: previous projected UAV embedding
      - hidden: previous GRU hidden state

    Outputs:
      - correction_xy: learned current-frame residual correction
      - corrected_xy: stage_xy + correction_xy
      - hidden: new recurrent state

    Important: corrected_xy is a refinement of the CURRENT incoming measurement.
    It is not previous_final + motion_delta and is never used as a future-motion
    polynomial feedback term.
    """

    def __init__(self, feature_dim=128, hidden_dim=256, dropout=0.0):
        super().__init__()
        embed_dim = int(config.EMBED_DIM)
        self.position_scale_m = float(getattr(config, "POSITION_INPUT_SCALE_M", 1000.0))
        self.variance_scale_m2 = float(getattr(config, "GRU_VARIANCE_SCALE_M2", 100.0))

        def low_dim(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )

        def visual(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )

        self.xy_projector = low_dim(2)
        self.var_projector = low_dim(2)
        self.mean_projector = visual(embed_dim)
        self.diff_projector = visual(embed_dim)
        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.gru.hidden_size, device=device, dtype=dtype)

    def forward_step(
        self,
        stage_xy: torch.Tensor,
        variance_xy: torch.Tensor,
        z_uav: torch.Tensor,
        previous_z_uav: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
    ) -> PositionGRUOutput:
        if previous_z_uav is None:
            previous_z_uav = z_uav
        if hidden is None:
            hidden = self.initial_hidden(z_uav.shape[0], z_uav.device, z_uav.dtype)

        temporal_mean = 0.5 * (z_uav + previous_z_uav)
        first_difference = z_uav - previous_z_uav
        xy_norm = stage_xy.float() / max(self.position_scale_m, 1e-6)
        var_norm = torch.log1p(variance_xy.float().clamp_min(0.0) / max(self.variance_scale_m2, 1e-6))

        recurrent_input = torch.cat(
            [
                self.xy_projector(xy_norm),
                self.var_projector(var_norm),
                self.mean_projector(temporal_mean),
                self.diff_projector(first_difference),
            ],
            dim=1,
        )
        new_hidden = self.gru(recurrent_input, hidden)
        correction_xy = self.position_head(self.dropout(new_hidden))
        corrected_xy = stage_xy.float() + correction_xy
        return PositionGRUOutput(corrected_xy, correction_xy, new_hidden)
