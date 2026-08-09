import math
from dataclasses import dataclass
from typing import Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================================
# Visual retrieval model
# ============================================================================
# This section intentionally keeps the same parameter names used by the existing
# Route-A-only visual checkpoint, so the trained retrieval heads remain usable.
# ============================================================================

class FourierCoordEncoder(nn.Module):
    def __init__(self, num_bands=32, max_freq=16.0):
        super().__init__()
        freqs = torch.logspace(
            0.0,
            math.log10(max_freq),
            steps=num_bands,
        )
        self.register_buffer("freqs", freqs)
        in_dim = 2 + 2 * 2 * num_bands
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, config.EMBED_DIM),
        )

    def forward(self, xy):
        xy = xy.float() / 1000.0
        angles = (
            xy[:, :, None]
            * self.freqs[None, None, :]
            * math.pi
        )
        encoded = torch.cat(
            [
                xy,
                angles.sin().flatten(1),
                angles.cos().flatten(1),
            ],
            dim=1,
        )
        return self.mlp(encoded)


class AllMapGeoCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip, _ = open_clip.create_model_from_pretrained(
            config.BACKBONE_NAME
        )

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        self.use_coord_encoder = bool(
            getattr(config, "USE_COORD_ENCODER", False)
        )

        self.uav_head = nn.Sequential(
            nn.Linear(config.CLIP_DIM, config.CLIP_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.CLIP_DIM, config.EMBED_DIM),
        )

        if self.use_coord_encoder:
            self.coord_encoder = FourierCoordEncoder()
            sat_head_in_dim = (
                config.CLIP_DIM
                + config.EMBED_DIM
            )
        else:
            self.coord_encoder = None
            sat_head_in_dim = config.CLIP_DIM

        self.sat_head = nn.Sequential(
            nn.Linear(
                sat_head_in_dim,
                config.CLIP_DIM,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                config.CLIP_DIM,
                config.EMBED_DIM,
            ),
        )

        self.use_qah_relation = bool(
            getattr(
                config,
                "USE_QAH_MS_RELATION",
                False,
            )
        )

        if self.use_qah_relation:
            relation_dim = int(
                getattr(
                    config,
                    "QAH_RELATION_DIM",
                    32,
                )
            )
            self.qah_relation_head = nn.Sequential(
                nn.Linear(
                    config.EMBED_DIM * 2 + 1,
                    256,
                ),
                nn.GELU(),
                nn.Linear(
                    256,
                    relation_dim,
                ),
            )
        else:
            self.qah_relation_head = None

        self.use_basin_rank_ms = bool(
            getattr(
                config,
                "USE_BASIN_RANK_MS",
                False,
            )
        )

        if self.use_basin_rank_ms:
            self.basin_ranker = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
        else:
            self.basin_ranker = None

        self.logit_scale = nn.Parameter(
            torch.ones([])
            * math.log(1.0 / 0.07)
        )

    @torch.no_grad()
    def encode_clip_image(self, image):
        self.clip.eval()
        return self.clip.encode_image(
            image.float()
        )

    @torch.no_grad()
    def encode_clip_spatial(
        self,
        image,
        output_size=None,
    ):
        output_size = int(
            output_size
            or config.MOTION_SPATIAL_SIZE
        )

        feature = self.encode_clip_image(
            image
        ).unsqueeze(-1).unsqueeze(-1)

        return F.adaptive_avg_pool2d(
            feature.float(),
            (
                output_size,
                output_size,
            ),
        )

    def encode_uav_from_clip(
        self,
        clip_feat,
        yaw=None,
    ):
        embedding = self.uav_head(
            clip_feat.float()
        )
        return F.normalize(
            embedding,
            dim=1,
        )

    def encode_uav(
        self,
        uav,
        yaw=None,
    ):
        return self.encode_uav_from_clip(
            self.encode_clip_image(uav)
        )

    def encode_sat_from_clip(
        self,
        sat_clip_feat,
        xy,
    ):
        if self.use_coord_encoder:
            coord_feat = self.coord_encoder(
                xy.float()
            )
            sat_input = torch.cat(
                [
                    sat_clip_feat.float(),
                    coord_feat,
                ],
                dim=1,
            )
        else:
            sat_input = sat_clip_feat.float()

        embedding = self.sat_head(
            sat_input
        )

        return F.normalize(
            embedding,
            dim=1,
        )

    def encode_relation(
        self,
        z_uav,
        z_sat,
        logits,
    ):
        if self.qah_relation_head is None:
            raise RuntimeError(
                "QAH relation head is disabled"
            )

        pair_product = (
            z_uav.unsqueeze(1)
            * z_sat
        )

        pair_absolute = torch.abs(
            z_uav.unsqueeze(1)
            - z_sat
        )

        pair_logit = logits.unsqueeze(-1)

        pair = torch.cat(
            [
                pair_product,
                pair_absolute,
                pair_logit,
            ],
            dim=-1,
        )

        relation = self.qah_relation_head(
            pair.reshape(
                -1,
                pair.shape[-1],
            )
        )

        return relation.reshape(
            pair.shape[0],
            pair.shape[1],
            -1,
        )


# ============================================================================
# Recurrent + Kalman temporal model
# ============================================================================

@dataclass
class RecurrentKalmanState:
    hidden: torch.Tensor
    kinematic: torch.Tensor
    covariance: torch.Tensor
    previous_frame_id: torch.Tensor


@dataclass
class RecurrentKalmanOutput:
    measurement_xy: torch.Tensor
    measurement_variance: torch.Tensor
    prediction_xy: torch.Tensor
    filtered_xy: torch.Tensor
    kinematic_state: torch.Tensor
    hidden_state: torch.Tensor
    kalman_position_gain: torch.Tensor
    visual_expectation: torch.Tensor
    hardms_xy: torch.Tensor
    final_state: RecurrentKalmanState


class RecurrentKalmanLocalizer(nn.Module):
    """
    Streaming recurrent localizer.

    Each frame:
      1. Explicit constant-acceleration prediction
         p^- = p + v*dt + 0.5*a*dt^2
         v^- = v + a*dt
         a^- = a
      2. GRU consumes visual retrieval evidence + previous motion prediction.
      3. GRU heads output:
           measurement_xy
           measurement_variance (R)
      4. Differentiable Kalman update fuses the predicted physical state and
         the learned visual measurement.
      5. The resulting state [x,y,vx,vy,ax,ay] is carried to the next frame.

    No hard jump rejection and no "N consecutive failures" rule is used.
    """

    def __init__(self):
        super().__init__()

        feature_dim = int(
            config.RNN_FEATURE_DIM
        )
        hidden_dim = int(
            config.RNN_HIDDEN_DIM
        )
        dropout = float(
            config.RNN_DROPOUT
        )

        self.uav_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.sat_projection = nn.Sequential(
            nn.Linear(
                config.EMBED_DIM,
                feature_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        # Numeric features per frame:
        # entropy, top1-top2 margin, max probability,
        # visual_expectation - prediction (x,y),
        # HardMS - prediction (x,y),
        # HardMS - visual_expectation distance,
        # predicted speed,
        # predicted acceleration magnitude.
        self.numeric_projection = nn.Sequential(
            nn.Linear(10, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self.gru = nn.GRUCell(
            feature_dim * 3,
            hidden_dim,
        )

        self.hidden_dropout = nn.Dropout(
            dropout
        )

        # Residual around the continuous probability-weighted visual position.
        self.measurement_residual_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                2,
            ),
        )

        # Predict diagonal measurement variance R_x, R_y.
        self.measurement_variance_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                2,
            ),
        )

        # Learn one positive process-noise variance for each physical state
        # dimension.  This is not a learned trajectory; it describes how much
        # the constant-acceleration model is allowed to drift between frames.
        self.process_variance_raw = nn.Parameter(
            torch.tensor(
                [-2.0, -2.0, -1.5, -1.5, -1.0, -1.0],
                dtype=torch.float32,
            )
        )

        # Begin conservatively: visual measurement residual starts at zero.
        nn.init.zeros_(
            self.measurement_residual_head[-1].weight
        )
        nn.init.zeros_(
            self.measurement_residual_head[-1].bias
        )

        # Initial R near 4 m^2 per axis before learning.
        nn.init.zeros_(
            self.measurement_variance_head[-1].weight
        )
        nn.init.constant_(
            self.measurement_variance_head[-1].bias,
            math.log(math.exp(4.0) - 1.0),
        )

    @staticmethod
    def _entropy(probability):
        count = max(
            int(probability.shape[-1]),
            2,
        )
        return -(
            probability
            * probability.clamp_min(
                1e-8
            ).log()
        ).sum(dim=-1) / math.log(
            float(count)
        )

    @staticmethod
    def _batch_eye(
        batch,
        size,
        device,
        dtype,
    ):
        return torch.eye(
            size,
            device=device,
            dtype=dtype,
        ).unsqueeze(0).expand(
            batch,
            size,
            size,
        )

    def _initial_covariance(
        self,
        batch,
        device,
        dtype,
    ):
        diagonal = torch.tensor(
            [
                config.KALMAN_INIT_POSITION_VAR,
                config.KALMAN_INIT_POSITION_VAR,
                config.KALMAN_INIT_VELOCITY_VAR,
                config.KALMAN_INIT_VELOCITY_VAR,
                config.KALMAN_INIT_ACCELERATION_VAR,
                config.KALMAN_INIT_ACCELERATION_VAR,
            ],
            device=device,
            dtype=dtype,
        )

        return torch.diag(
            diagonal
        ).unsqueeze(0).expand(
            batch,
            6,
            6,
        ).clone()

    def _transition_matrix(
        self,
        dt,
        dtype,
    ):
        batch = int(dt.shape[0])
        device = dt.device

        matrix = self._batch_eye(
            batch,
            6,
            device,
            dtype,
        ).clone()

        dt = dt.to(dtype)
        half_dt2 = 0.5 * dt.square()

        matrix[:, 0, 2] = dt
        matrix[:, 1, 3] = dt

        matrix[:, 0, 4] = half_dt2
        matrix[:, 1, 5] = half_dt2

        matrix[:, 2, 4] = dt
        matrix[:, 3, 5] = dt

        return matrix

    def _process_covariance(
        self,
        dt,
        dtype,
    ):
        batch = int(dt.shape[0])
        variance = F.softplus(
            self.process_variance_raw
        ) + float(
            config.KALMAN_MIN_VARIANCE
        )

        # More elapsed frames -> more accumulated model uncertainty.
        scale = dt.to(
            dtype
        ).clamp_min(1.0).unsqueeze(1)

        diagonal = variance.to(
            dtype
        ).unsqueeze(0) * scale

        return torch.diag_embed(
            diagonal.expand(
                batch,
                -1,
            )
        )

    def _kinematic_predict(
        self,
        state,
        covariance,
        dt,
    ):
        transition = self._transition_matrix(
            dt,
            state.dtype,
        )

        predicted_state = torch.bmm(
            transition,
            state.unsqueeze(-1),
        ).squeeze(-1)

        process = self._process_covariance(
            dt,
            state.dtype,
        )

        predicted_covariance = (
            torch.bmm(
                torch.bmm(
                    transition,
                    covariance,
                ),
                transition.transpose(
                    1,
                    2,
                ),
            )
            + process
        )

        return (
            predicted_state,
            predicted_covariance,
        )

    def _kalman_update(
        self,
        predicted_state,
        predicted_covariance,
        measurement_xy,
        measurement_variance,
    ):
        batch = int(
            predicted_state.shape[0]
        )
        device = predicted_state.device
        dtype = predicted_state.dtype

        # H selects x,y from [x,y,vx,vy,ax,ay].
        observation = torch.zeros(
            batch,
            2,
            6,
            device=device,
            dtype=dtype,
        )
        observation[:, 0, 0] = 1.0
        observation[:, 1, 1] = 1.0

        measurement_covariance = (
            torch.diag_embed(
                measurement_variance.to(
                    dtype
                )
            )
        )

        innovation = (
            measurement_xy
            - predicted_state[:, 0:2]
        )

        hp = torch.bmm(
            observation,
            predicted_covariance,
        )

        innovation_covariance = (
            torch.bmm(
                hp,
                observation.transpose(
                    1,
                    2,
                ),
            )
            + measurement_covariance
        )

        ph_t = torch.bmm(
            predicted_covariance,
            observation.transpose(
                1,
                2,
            ),
        )

        # K = P H^T S^-1.
        # solve(S, PH^T^T)^T is numerically preferable to explicit inverse.
        kalman_gain = torch.linalg.solve(
            innovation_covariance,
            ph_t.transpose(
                1,
                2,
            ),
        ).transpose(
            1,
            2,
        )

        updated_state = (
            predicted_state
            + torch.bmm(
                kalman_gain,
                innovation.unsqueeze(-1),
            ).squeeze(-1)
        )

        identity = self._batch_eye(
            batch,
            6,
            device,
            dtype,
        )

        kh = torch.bmm(
            kalman_gain,
            observation,
        )

        left = identity - kh

        # Joseph covariance update.
        updated_covariance = (
            torch.bmm(
                torch.bmm(
                    left,
                    predicted_covariance,
                ),
                left.transpose(
                    1,
                    2,
                ),
            )
            + torch.bmm(
                torch.bmm(
                    kalman_gain,
                    measurement_covariance,
                ),
                kalman_gain.transpose(
                    1,
                    2,
                ),
            )
        )

        position_gain = torch.stack(
            [
                kalman_gain[:, 0, 0],
                kalman_gain[:, 1, 1],
            ],
            dim=1,
        )

        return (
            updated_state,
            updated_covariance,
            position_gain,
        )

    def _visual_measurement(
        self,
        z_uav,
        z_sat,
        raw_prob,
        centers,
        hardms_xy,
        predicted_state,
        hidden,
    ):
        visual_expectation = (
            raw_prob.unsqueeze(-1)
            * centers
        ).sum(dim=1)

        sat_context = (
            raw_prob.unsqueeze(-1)
            * z_sat
        ).sum(dim=1)

        entropy = self._entropy(
            raw_prob
        )

        top_values = raw_prob.topk(
            k=2,
            dim=1,
        ).values

        margin = (
            top_values[:, 0]
            - top_values[:, 1]
        )

        max_probability = top_values[:, 0]

        predicted_xy = predicted_state[:, 0:2]
        predicted_speed = torch.linalg.norm(
            predicted_state[:, 2:4],
            dim=1,
        )
        predicted_acceleration = torch.linalg.norm(
            predicted_state[:, 4:6],
            dim=1,
        )

        expectation_innovation = (
            visual_expectation
            - predicted_xy
        )

        hardms_innovation = (
            hardms_xy
            - predicted_xy
        )

        hardms_expectation_distance = (
            torch.linalg.norm(
                hardms_xy
                - visual_expectation,
                dim=1,
            )
        )

        numeric = torch.cat(
            [
                entropy.unsqueeze(1),
                margin.unsqueeze(1),
                max_probability.unsqueeze(1),
                expectation_innovation,
                hardms_innovation,
                hardms_expectation_distance.unsqueeze(1),
                predicted_speed.unsqueeze(1),
                predicted_acceleration.unsqueeze(1),
            ],
            dim=1,
        )

        recurrent_input = torch.cat(
            [
                self.uav_projection(
                    z_uav
                ),
                self.sat_projection(
                    sat_context
                ),
                self.numeric_projection(
                    numeric
                ),
            ],
            dim=1,
        )

        hidden = self.gru(
            recurrent_input,
            hidden,
        )

        head_hidden = self.hidden_dropout(
            hidden
        )

        residual = self.measurement_residual_head(
            head_hidden
        )

        measurement_xy = (
            visual_expectation
            + residual
        )

        measurement_variance = (
            F.softplus(
                self.measurement_variance_head(
                    head_hidden
                )
            )
            + float(
                config.KALMAN_MIN_VARIANCE
            )
        ).clamp(
            max=float(
                config.KALMAN_MAX_MEASUREMENT_VAR
            )
        )

        return (
            measurement_xy,
            measurement_variance,
            visual_expectation,
            hidden,
        )

    def initialize_state(
        self,
        z_uav,
        z_sat,
        raw_prob,
        centers,
        hardms_xy,
        frame_id,
    ):
        batch = int(
            z_uav.shape[0]
        )
        device = z_uav.device
        dtype = centers.dtype

        hidden = torch.zeros(
            batch,
            int(config.RNN_HIDDEN_DIM),
            device=device,
            dtype=dtype,
        )

        # At frame 1 there is no previous physical state.  The visual
        # expectation is used only as the reference for the first recurrent
        # measurement.  No GT initialization is used.
        visual_expectation = (
            raw_prob.unsqueeze(-1)
            * centers
        ).sum(dim=1)

        provisional_state = torch.zeros(
            batch,
            6,
            device=device,
            dtype=dtype,
        )
        provisional_state[:, 0:2] = (
            visual_expectation
        )

        (
            measurement_xy,
            measurement_variance,
            visual_expectation,
            hidden,
        ) = self._visual_measurement(
            z_uav,
            z_sat,
            raw_prob,
            centers,
            hardms_xy,
            provisional_state,
            hidden,
        )

        state = torch.zeros(
            batch,
            6,
            device=device,
            dtype=dtype,
        )
        state[:, 0:2] = measurement_xy

        covariance = self._initial_covariance(
            batch,
            device,
            dtype,
        )

        recurrent_state = RecurrentKalmanState(
            hidden=hidden,
            kinematic=state,
            covariance=covariance,
            previous_frame_id=frame_id.long(),
        )

        gain = torch.ones(
            batch,
            2,
            device=device,
            dtype=dtype,
        )

        return (
            recurrent_state,
            measurement_xy,
            measurement_variance,
            visual_expectation,
            state[:, 0:2],
            gain,
        )

    def step(
        self,
        z_uav,
        z_sat,
        raw_prob,
        centers,
        hardms_xy,
        frame_id,
        recurrent_state=None,
    ):
        if recurrent_state is None:
            (
                state,
                measurement_xy,
                measurement_variance,
                visual_expectation,
                prediction_xy,
                gain,
            ) = self.initialize_state(
                z_uav,
                z_sat,
                raw_prob,
                centers,
                hardms_xy,
                frame_id,
            )

            return {
                "state": state,
                "measurement_xy": measurement_xy,
                "measurement_variance": measurement_variance,
                "prediction_xy": prediction_xy,
                "filtered_xy": state.kinematic[:, 0:2],
                "visual_expectation": visual_expectation,
                "kalman_position_gain": gain,
            }

        dt = (
            frame_id.long()
            - recurrent_state.previous_frame_id.long()
        ).float().clamp_min(1.0)

        (
            predicted_state,
            predicted_covariance,
        ) = self._kinematic_predict(
            recurrent_state.kinematic,
            recurrent_state.covariance,
            dt,
        )

        (
            measurement_xy,
            measurement_variance,
            visual_expectation,
            hidden,
        ) = self._visual_measurement(
            z_uav,
            z_sat,
            raw_prob,
            centers,
            hardms_xy,
            predicted_state,
            recurrent_state.hidden,
        )

        (
            updated_state,
            updated_covariance,
            gain,
        ) = self._kalman_update(
            predicted_state,
            predicted_covariance,
            measurement_xy,
            measurement_variance,
        )

        state = RecurrentKalmanState(
            hidden=hidden,
            kinematic=updated_state,
            covariance=updated_covariance,
            previous_frame_id=frame_id.long(),
        )

        return {
            "state": state,
            "measurement_xy": measurement_xy,
            "measurement_variance": measurement_variance,
            "prediction_xy": predicted_state[:, 0:2],
            "filtered_xy": updated_state[:, 0:2],
            "visual_expectation": visual_expectation,
            "kalman_position_gain": gain,
        }

    def forward(
        self,
        z_uav,
        z_sat,
        raw_logits,
        raw_prob,
        centers,
        frame_ids,
        hardms_xy,
        initial_state=None,
    ):
        # raw_logits is accepted for interface compatibility and future
        # extensions.  raw_prob already contains the normalized retrieval
        # evidence used by this recurrent model.
        del raw_logits

        frame_count = int(
            centers.shape[1]
        )

        state = initial_state

        measurement_rows = []
        variance_rows = []
        prediction_rows = []
        filtered_rows = []
        kinematic_rows = []
        hidden_rows = []
        gain_rows = []
        expectation_rows = []

        for time_index in range(
            frame_count
        ):
            result = self.step(
                z_uav[:, time_index],
                z_sat[:, time_index],
                raw_prob[:, time_index],
                centers[:, time_index],
                hardms_xy[:, time_index],
                frame_ids[:, time_index],
                recurrent_state=state,
            )

            state = result["state"]

            measurement_rows.append(
                result["measurement_xy"]
            )
            variance_rows.append(
                result["measurement_variance"]
            )
            prediction_rows.append(
                result["prediction_xy"]
            )
            filtered_rows.append(
                result["filtered_xy"]
            )
            kinematic_rows.append(
                state.kinematic
            )
            hidden_rows.append(
                state.hidden
            )
            gain_rows.append(
                result[
                    "kalman_position_gain"
                ]
            )
            expectation_rows.append(
                result[
                    "visual_expectation"
                ]
            )

        return RecurrentKalmanOutput(
            measurement_xy=torch.stack(
                measurement_rows,
                dim=1,
            ),
            measurement_variance=torch.stack(
                variance_rows,
                dim=1,
            ),
            prediction_xy=torch.stack(
                prediction_rows,
                dim=1,
            ),
            filtered_xy=torch.stack(
                filtered_rows,
                dim=1,
            ),
            kinematic_state=torch.stack(
                kinematic_rows,
                dim=1,
            ),
            hidden_state=torch.stack(
                hidden_rows,
                dim=1,
            ),
            kalman_position_gain=torch.stack(
                gain_rows,
                dim=1,
            ),
            visual_expectation=torch.stack(
                expectation_rows,
                dim=1,
            ),
            hardms_xy=hardms_xy,
            final_state=state,
        )
