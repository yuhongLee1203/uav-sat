import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import RouteDataset, meters_from_latlon
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import ReversibleTopologyRecoveryLSTM

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


ARCHITECTURE_NAME = "ReversibleTopologyRecoveryLSTM_v10"


@dataclass
class RouteCache:
    route_name: str
    frame_ids: torch.Tensor
    gt_xy: torch.Tensor
    uav_clip: torch.Tensor
    image_paths: list

    def __len__(self):
        return int(self.gt_xy.shape[0])


@dataclass
class MissionWaypoint:
    order: int
    frame_index: int
    xy: torch.Tensor


@dataclass
class MissionLeg:
    index: int
    start: MissionWaypoint
    end: MissionWaypoint

    @property
    def start_frame(self):
        return int(self.start.frame_index)

    @property
    def end_frame(self):
        return int(self.end.frame_index)

    @property
    def start_xy(self):
        return self.start.xy

    @property
    def end_xy(self):
        return self.end.xy


@dataclass
class MissionRoute:
    route_name: str
    waypoints: list
    legs: list


@dataclass
class CandidateSet:
    centers: torch.Tensor
    z_sat: torch.Tensor
    raw_logits: torch.Tensor
    valid_mask: torch.Tensor


# =============================================================================
# General / data
# =============================================================================

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cache_dtype():
    if config.FEATURE_CACHE_DTYPE == "float16":
        return torch.float16
    return torch.float32


def parse_frame_id(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(str(value))


def load_mission_route(route_name, origin_lat, origin_lon):
    path = Path(config.WAYPOINT_FILES[route_name])
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = sorted(payload["waypoints"], key=lambda item: int(item["waypoint_order"]))
    waypoints = []

    for item in raw:
        x_m, y_m = meters_from_latlon(
            item["latitude"],
            item["longitude"],
            origin_lat,
            origin_lon,
        )
        waypoints.append(
            MissionWaypoint(
                order=int(item["waypoint_order"]),
                frame_index=int(item.get("frame_index", -1)),
                xy=torch.tensor([x_m, y_m], dtype=torch.float32),
            )
        )

    if len(waypoints) < 2:
        raise RuntimeError("%s: fewer than two waypoints" % route_name)

    legs = []
    for index in range(len(waypoints) - 1):
        legs.append(MissionLeg(index=index, start=waypoints[index], end=waypoints[index + 1]))

    print(
        "%s: %d waypoints -> %d route legs" % (route_name, len(waypoints), len(legs)),
        flush=True,
    )
    print(
        "  " + " -> ".join("W%d[f%d]" % (wp.order, wp.frame_index) for wp in waypoints),
        flush=True,
    )
    return MissionRoute(route_name=route_name, waypoints=waypoints, legs=legs)


@torch.no_grad()
def build_route_cache(route_name, root, visual, device):
    stat = config.VISUAL_CHECKPOINT.stat()
    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "architecture": ARCHITECTURE_NAME,
    }
    cache_path = config.OUTPUT_DIR / "feature_cache" / (route_name + "_uav_clip.pt")

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") == signature:
            print("%s: reuse frozen UAV feature cache" % route_name, flush=True)
            return RouteCache(
                route_name=route_name,
                frame_ids=payload["frame_ids"],
                gt_xy=payload["gt_xy"],
                uav_clip=payload["uav_clip"],
                image_paths=payload["image_paths"],
            )

    dataset = RouteDataset(
        Path(root),
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )

    frame_rows = []
    gt_rows = []
    clip_rows = []
    image_paths = []
    batch_size = int(config.VISUAL_CACHE_BATCH_SIZE)

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[index] for index in range(start, end)]
        uav = torch.stack([item["uav"] for item in items]).to(device)
        clip = visual.encode_uav_clip(uav)
        clip_rows.append(clip.detach().cpu().to(cache_dtype()))
        gt_rows.append(torch.stack([item["xy"].float() for item in items]))

        for item in items:
            frame_rows.append(parse_frame_id(item["frame_id"]))
            image_paths.append(str(item["image_path"]))

        if start == 0 or end == len(dataset) or (start // batch_size) % 10 == 0:
            print("%s backbone cache: %d/%d" % (route_name, end, len(dataset)), flush=True)

    result = RouteCache(
        route_name=route_name,
        frame_ids=torch.tensor(frame_rows, dtype=torch.long),
        gt_xy=torch.cat(gt_rows).float(),
        uav_clip=torch.cat(clip_rows),
        image_paths=image_paths,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "frame_ids": result.frame_ids,
            "gt_xy": result.gt_xy,
            "uav_clip": result.uav_clip,
            "image_paths": result.image_paths,
        },
        cache_path,
    )
    return result


@torch.no_grad()
def build_gallery_embeddings(visual):
    clip_feat = visual.gallery["clip_feat"]
    xy = visual.gallery["xy"]
    rows = []
    batch_size = 4096
    for start in range(0, int(clip_feat.shape[0]), batch_size):
        end = min(start + batch_size, int(clip_feat.shape[0]))
        rows.append(
            visual.model.encode_sat_from_clip(
                clip_feat[start:end].float(),
                xy[start:end].float(),
            )
        )
    result = torch.cat(rows, dim=0)
    print("satellite task embeddings: %s" % (tuple(result.shape),), flush=True)
    return result


# =============================================================================
# Route geometry / visual banks
# =============================================================================

def leg_geometry(leg, device=None, dtype=torch.float32):
    start, end = leg.start_xy, leg.end_xy
    if device is not None:
        start = start.to(device=device, dtype=dtype)
        end = end.to(device=device, dtype=dtype)
    vector = end - start
    length = torch.linalg.norm(vector).clamp_min(1e-6)
    heading = vector / length
    normal = torch.stack([-heading[1], heading[0]])
    return start, end, heading, normal, length


def heading_degrees(leg):
    _, _, heading, _, _ = leg_geometry(leg)
    return float(math.degrees(math.atan2(float(heading[1]), float(heading[0]))))


def route_corridor_mask(xy, leg):
    start, _, heading, normal, length = leg_geometry(leg, device=xy.device, dtype=xy.dtype)
    start_view = start.reshape(1, 2) if xy.ndim == 2 else start.reshape(1, 1, 2)
    relative = xy - start_view
    along = (relative * heading).sum(dim=-1)
    cross = (relative * normal).sum(dim=-1).abs()
    return ((along >= -float(config.ROUTE_ALONG_BACK_PADDING_M)) &
            (along <= length + float(config.ROUTE_ALONG_FORWARD_PADDING_M)) &
            (cross <= float(config.ROUTE_CORRIDOR_HALF_WIDTH_M)))


def route_bank_indices(visual, leg):
    mask = route_corridor_mask(visual.gallery["xy"], leg)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() < 2:
        raise RuntimeError("route corridor has fewer than two SAT patches for W%d->W%d" % (leg.start.order, leg.end.order))
    return indices


def build_route_banks(visual, route):
    banks = {}
    for leg in route.legs:
        indices = route_bank_indices(visual, leg)
        banks[leg.index] = indices
        print("%s W%d->W%d: %d route SAT patches" % (route.route_name, leg.start.order, leg.end.order, int(indices.numel())), flush=True)
    return banks


def forward_projection(xy, reference_xy, leg):
    _, _, heading, _, _ = leg_geometry(leg, device=xy.device, dtype=xy.dtype)
    return ((xy - reference_xy) * heading).sum(dim=-1)


def neighbor_leg_slots(route, active_leg_index):
    return [index if 0 <= index < len(route.legs) else None for index in (active_leg_index - 1, active_leg_index, active_leg_index + 1)]


def route_context(route, active_leg_index, device, dtype):
    leg = route.legs[active_leg_index]
    _, _, heading, normal, _ = leg_geometry(leg, device=device, dtype=dtype)
    return torch.tensor([[float(heading[0]), float(heading[1]), float(normal[0]), float(normal[1]),
                          1.0 if active_leg_index > 0 else 0.0,
                          1.0 if active_leg_index + 1 < len(route.legs) else 0.0]], dtype=dtype, device=device)


def _candidate_from_gallery(visual, gallery_z_sat, z_uav, gallery_indices):
    z_sat = gallery_z_sat[gallery_indices].unsqueeze(0)
    centers = visual.gallery["xy"][gallery_indices].unsqueeze(0)
    raw = visual.model.logit_scale.exp().clamp(max=100.0) * torch.einsum("bd,bnd->bn", z_uav, z_sat)
    return CandidateSet(centers=centers, z_sat=z_sat, raw_logits=raw, valid_mask=torch.ones_like(raw, dtype=torch.bool))


@torch.no_grad()
def build_local_forward_candidate(visual, uav_clip, z_uav, reference_xy, leg, bank_indices, gallery_z_sat):
    candidate = visual.candidate_batch(uav_clip, reference_xy, grid_size=config.GRID_SIZE)
    centers = candidate.centers[0]
    valid = route_corridor_mask(centers, leg) & (forward_projection(centers, reference_xy[0].reshape(1, 2), leg) >= -float(config.FORWARD_HALF_EPS_M))
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() > int(config.FORWARD_CANDIDATE_COUNT):
        distance = torch.linalg.norm(centers[valid_indices] - reference_xy[0].reshape(1, 2), dim=1)
        valid_indices = valid_indices[distance.topk(int(config.FORWARD_CANDIDATE_COUNT), largest=False).indices]
    if valid_indices.numel() > 0:
        return CandidateSet(centers=candidate.centers[:, valid_indices, :], z_sat=candidate.z_sat[:, valid_indices, :],
                            raw_logits=candidate.raw_logits[:, valid_indices],
                            valid_mask=torch.ones((1, int(valid_indices.numel())), dtype=torch.bool, device=candidate.raw_logits.device))
    bank_xy = visual.gallery["xy"][bank_indices]
    projection = forward_projection(bank_xy, reference_xy[0].reshape(1, 2), leg)
    legal = bank_indices[projection >= -float(config.FORWARD_HALF_EPS_M)]
    if legal.numel() == 0:
        legal = bank_indices
    legal_xy = visual.gallery["xy"][legal]
    nearest = legal[torch.linalg.norm(legal_xy - reference_xy, dim=1).argmin().reshape(1)]
    return _candidate_from_gallery(visual, gallery_z_sat, z_uav, nearest)


@torch.no_grad()
def build_topology_recovery_candidate(visual, z_uav, route, banks, active_leg_index, gallery_z_sat):
    pieces = [banks[index] for index in neighbor_leg_slots(route, active_leg_index) if index is not None]
    union = torch.unique(torch.cat(pieces, dim=0))
    z_bank = gallery_z_sat[union]
    logits = visual.model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_bank.t())
    k = min(int(config.RECOVERY_TOPK), int(logits.shape[1]))
    top = logits.topk(k=k, dim=1)
    selected = union[top.indices[0]]
    return CandidateSet(centers=visual.gallery["xy"][selected].unsqueeze(0), z_sat=gallery_z_sat[selected].unsqueeze(0),
                        raw_logits=top.values, valid_mask=torch.ones_like(top.values, dtype=torch.bool))


@torch.no_grad()
def build_leg_evidence(visual, z_uav, route, banks, active_leg_index, gallery_z_sat):
    evidence, valid = [], []
    for leg_index in neighbor_leg_slots(route, active_leg_index):
        if leg_index is None:
            evidence.extend([0.0, 0.0, 0.0, 1.0])
            valid.append(False)
            continue
        indices = banks[leg_index]
        z_bank = gallery_z_sat[indices]
        logits = visual.model.logit_scale.exp().clamp(max=100.0) * (z_uav @ z_bank.t())
        probability = torch.softmax(logits, dim=1)
        k = min(int(config.LEG_EVIDENCE_TOPK), int(logits.shape[1]))
        top = logits.topk(k=k, dim=1).values
        top1 = float(top[0, 0].cpu())
        top2 = float(top[0, 1].cpu()) if top.shape[1] > 1 else top1
        mean_top = float(top.mean().cpu())
        entropy = float((-(probability * probability.clamp_min(1e-8).log()).sum(dim=1) / math.log(max(int(probability.shape[1]), 2)))[0].cpu())
        evidence.extend([math.tanh(top1 / 20.0), math.tanh((top1 - top2) / 10.0), math.tanh(mean_top / 20.0), entropy])
        valid.append(True)
    return torch.tensor([evidence], dtype=z_uav.dtype, device=z_uav.device), torch.tensor([valid], dtype=torch.bool, device=z_uav.device)


def decode_candidate(candidate):
    masked = torch.where(candidate.valid_mask, candidate.raw_logits, torch.full_like(candidate.raw_logits, -1e4))
    index = masked.argmax(dim=1)
    gather = index.reshape(-1, 1, 1).expand(-1, 1, 2)
    xy = candidate.centers.gather(1, gather).squeeze(1)
    return xy, torch.softmax(masked, dim=1), index


def compose_localization(previous_xy, local_xy, recovery_xy, fusion_logits, straight_through):
    probability = torch.softmax(fusion_logits / float(config.BRANCH_SOFTMAX_TEMPERATURE), dim=1)
    hard_index = probability.argmax(dim=1)
    hard = F.one_hot(hard_index, num_classes=int(config.HYPOTHESIS_COUNT)).to(probability.dtype)
    weight = hard + probability - probability.detach() if straight_through else hard
    hypotheses = torch.stack([previous_xy, local_xy, recovery_xy], dim=1)
    return (weight.unsqueeze(-1) * hypotheses).sum(dim=1), hypotheses, probability, hard_index


def candidate_capture(candidate, gt_xy):
    minimum = torch.linalg.norm(candidate.centers - gt_xy[:, None, :], dim=2).min(dim=1).values
    return minimum <= float(config.CANDIDATE_CAPTURE_RADIUS_M)


def training_leg_for_frame(route, frame_id):
    for leg in route.legs:
        if frame_id >= leg.start_frame and frame_id <= leg.end_frame:
            return leg
    return route.legs[-1]


def fusion_target(hypotheses, gt_xy):
    error = torch.linalg.norm(hypotheses.detach() - gt_xy[:, None, :], dim=2)
    hold_error, local_error, recovery_error = error[:, 0], error[:, 1], error[:, 2]
    local_good = local_error <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    recovery_good = recovery_error <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    hold_or_local = torch.where(local_error < hold_error,
                                torch.full_like(hold_error, int(config.HYPOTHESIS_LOCAL), dtype=torch.long),
                                torch.full_like(hold_error, int(config.HYPOTHESIS_HOLD), dtype=torch.long))
    recovery_index = torch.full_like(hold_or_local, int(config.HYPOTHESIS_RECOVERY))
    return torch.where((~local_good) & recovery_good, recovery_index, hold_or_local).long()


def relative_leg_target(true_leg_index, active_leg_index, device):
    value = int(config.LEG_PREVIOUS) if true_leg_index < active_leg_index else int(config.LEG_NEXT) if true_leg_index > active_leg_index else int(config.LEG_CURRENT)
    return torch.tensor([value], dtype=torch.long, device=device)


def apply_leg_choice(active_leg_index, leg_choice, route):
    if int(leg_choice) == int(config.LEG_PREVIOUS):
        return max(0, active_leg_index - 1)
    if int(leg_choice) == int(config.LEG_NEXT):
        return min(len(route.legs) - 1, active_leg_index + 1)
    return active_leg_index


def maybe_perturb_training_leg(active_leg_index, route):
    # Training-only state-noise augmentation. It never uses GT to set the
    # recurrent state; GT is used only to form the supervised leg target.
    if random.random() >= float(config.LEG_TRAIN_PERTURB_PROB):
        return active_leg_index, False
    options = []
    if active_leg_index > 0:
        options.append(active_leg_index - 1)
    if active_leg_index + 1 < len(route.legs):
        options.append(active_leg_index + 1)
    return (random.choice(options), True) if options else (active_leg_index, False)


def train_one_epoch(model, optimizer, visual, gallery_z_sat, cache, route, banks, device, epoch_index):
    model.train()
    hidden, cell = model.initial_state(1, device, torch.float32)
    active_leg_index = 0
    previous_visual_xy = route.legs[0].start_xy.to(device).reshape(1, 2)
    previous_gt = previous_visual_xy.detach()
    previous_z_uav = None
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss, accumulated_steps, logs = None, 0, []
    for sequence_index in range(len(cache)):
        frame_id = int(cache.frame_ids[sequence_index].item())
        true_leg = training_leg_for_frame(route, frame_id)
        active_leg_index, perturbed = maybe_perturb_training_leg(active_leg_index, route)
        active_leg = route.legs[active_leg_index]
        gt_xy = cache.gt_xy[sequence_index:sequence_index + 1].to(device).float()
        uav_clip = cache.uav_clip[sequence_index:sequence_index + 1].to(device).float()
        z_uav = visual.model.encode_uav_from_clip(uav_clip)
        local = build_local_forward_candidate(visual, uav_clip, z_uav, previous_visual_xy.detach(), active_leg, banks[active_leg_index], gallery_z_sat)
        recovery = build_topology_recovery_candidate(visual, z_uav, route, banks, active_leg_index, gallery_z_sat)
        leg_evidence, leg_valid_mask = build_leg_evidence(visual, z_uav, route, banks, active_leg_index, gallery_z_sat)
        context = route_context(route, active_leg_index, device, z_uav.dtype)
        output = model.forward_step(z_uav=z_uav, previous_z_uav=previous_z_uav,
            local_z_sat=local.z_sat, local_raw_logits=local.raw_logits, local_valid_mask=local.valid_mask,
            recovery_z_sat=recovery.z_sat, recovery_raw_logits=recovery.raw_logits, recovery_valid_mask=recovery.valid_mask,
            leg_evidence=leg_evidence, leg_valid_mask=leg_valid_mask, route_context=context, hidden=hidden, cell=cell)
        local_xy, _, _ = decode_candidate(local)
        recovery_xy, _, _ = decode_candidate(recovery)
        current_xy, hypotheses, fusion_probability, _ = compose_localization(previous_visual_xy, local_xy, recovery_xy, output.fusion_logits, True)
        position_loss = F.smooth_l1_loss(current_xy, gt_xy)
        branch_loss = F.cross_entropy(output.fusion_logits, fusion_target(hypotheses, gt_xy))
        leg_target = relative_leg_target(true_leg.index, active_leg_index, device)
        leg_loss = F.cross_entropy(output.leg_logits, leg_target)
        leg_choice = int(output.leg_probability.argmax(dim=1)[0].detach().cpu())
        step_loss = F.smooth_l1_loss(current_xy - previous_visual_xy, gt_xy - previous_gt)
        variance = output.measurement_variance.clamp_min(float(config.KALMAN_R_MIN_VAR))
        variance_nll = 0.5 * (torch.log(variance) + (current_xy - gt_xy) ** 2 / variance).mean()
        loss = float(config.LOSS_POSITION)*position_loss + float(config.LOSS_BRANCH)*branch_loss + float(config.LOSS_LEG_STATE)*leg_loss + float(config.LOSS_STEP)*step_loss + float(config.LOSS_VARIANCE_NLL)*variance_nll
        accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
        accumulated_steps += 1
        p = fusion_probability[0].detach().cpu().numpy(); lp = output.leg_probability[0].detach().cpu().numpy()
        logs.append({"loss":float(loss.detach().cpu()), "position":float(position_loss.detach().cpu()), "branch":float(branch_loss.detach().cpu()), "leg":float(leg_loss.detach().cpu()),
                     "hold":float(p[0]), "local":float(p[1]), "recovery":float(p[2]), "leg_prev":float(lp[0]), "leg_current":float(lp[1]), "leg_next":float(lp[2]),
                     "leg_correct":float(leg_choice == int(leg_target[0].cpu())), "local_capture":float(candidate_capture(local, gt_xy).float().mean().cpu()),
                     "recovery_capture":float(candidate_capture(recovery, gt_xy).float().mean().cpu()), "perturbed":float(perturbed)})
        active_leg_index = apply_leg_choice(active_leg_index, leg_choice, route)
        hidden, cell = output.hidden, output.cell
        previous_z_uav = z_uav.detach(); previous_visual_xy = current_xy; previous_gt = gt_xy.detach()
        if accumulated_steps >= int(config.TBPTT_STEPS) or sequence_index == len(cache)-1:
            normalized = accumulated_loss / float(accumulated_steps)
            if not torch.isfinite(normalized): raise FloatingPointError("non-finite temporal loss")
            normalized.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP_NORM)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            hidden=hidden.detach(); cell=cell.detach(); previous_visual_xy=previous_visual_xy.detach(); previous_z_uav=previous_z_uav.detach(); accumulated_loss=None; accumulated_steps=0
    result = {key:float(np.mean([row[key] for row in logs])) for key in logs[0]}
    result["teacher_ratio"] = 0.0
    return result


def train_temporal_model(model, visual, gallery_z_sat, cache, route, banks, device, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.TEMPORAL_LR), weight_decay=float(config.TEMPORAL_WEIGHT_DECAY))
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("TEMPORAL v10: Route-A only, ZERO teacher from epoch 1", flush=True)
    print("  LOCAL=forward 3x6; RECOVERY=full prev/current/next corridors; LEG=fully reversible", flush=True)
    for epoch_index in range(int(epochs)):
        metrics = train_one_epoch(model, optimizer, visual, gallery_z_sat, cache, route, banks, device, epoch_index)
        payload = {"architecture":ARCHITECTURE_NAME, "model":model.state_dict(), "epoch":epoch_index+1, "training_route":"route_A", "teacher_ratio":0.0,
                   "local_search":"forward 3x6 hard current-image retrieval", "recovery":"previous/current/next full corridors without forward restriction or Pred center",
                   "leg_state":"reversible PREVIOUS/CURRENT/NEXT", "motion_state_input":False, "polynomial_input":False, "predicted_progress_input":False,
                   "predicted_endpoint_distance_input":False, "kalman":"post-model position-only [x,y]", "test_gt_input":False, "test_waypoint_frame_index_input":False}
        temp_path = Path(str(config.TEMPORAL_CHECKPOINT)+".tmp"); torch.save(payload,temp_path); temp_path.replace(config.TEMPORAL_CHECKPOINT)
        print("epoch=%03d/%d loss=%.4f pos=%.4f branch=%.4f leg=%.4f teacher=0.00 H/L/R=%.2f/%.2f/%.2f cap L/R=%.1f/%.1f%% leg P/C/N=%.2f/%.2f/%.2f legAcc=%.1f%% perturb=%.1f%%" %
              (epoch_index+1,int(epochs),metrics["loss"],metrics["position"],metrics["branch"],metrics["leg"],metrics["hold"],metrics["local"],metrics["recovery"],
               metrics["local_capture"]*100,metrics["recovery_capture"]*100,metrics["leg_prev"],metrics["leg_current"],metrics["leg_next"],metrics["leg_correct"]*100,metrics["perturbed"]*100), flush=True)
        print("checkpoint: %s" % config.TEMPORAL_CHECKPOINT, flush=True)


def load_temporal_model(model, device):
    if not config.TEMPORAL_CHECKPOINT.exists(): raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    if checkpoint.get("architecture") != ARCHITECTURE_NAME: raise RuntimeError("temporal checkpoint mismatch: expected %s, got %s" % (ARCHITECTURE_NAME, checkpoint.get("architecture")))
    model.load_state_dict(checkpoint["model"], strict=True); return checkpoint


def make_position_kalman(initial_xy):
    if KalmanFilter is None: raise ImportError("FilterPy is required: pip install filterpy")
    kf=KalmanFilter(dim_x=2, dim_z=2); kf.x=np.asarray(initial_xy,dtype=np.float64).reshape(2); kf.F=np.eye(2); kf.H=np.eye(2)
    kf.P=np.eye(2)*float(config.KALMAN_INIT_POSITION_VAR); kf.Q=np.eye(2)*float(config.KALMAN_Q_POSITION); return kf


def metric_block(prediction, gt):
    prediction=np.asarray(prediction,dtype=np.float64); gt=np.asarray(gt,dtype=np.float64); error=np.linalg.norm(prediction-gt,axis=1)
    if len(prediction)>1:
        pred_step=np.diff(prediction,axis=0); gt_step=np.diff(gt,axis=0); rpe=np.linalg.norm(pred_step-gt_step,axis=1); gt_step_length=np.linalg.norm(gt_step,axis=1)
        jump_threshold=float(np.percentile(gt_step_length,99))+float(config.JUMP_TOLERANCE_M); jump_rate=float((np.linalg.norm(pred_step,axis=1)>jump_threshold).mean()*100)
    else: rpe=np.zeros(1); jump_threshold=0.0; jump_rate=0.0
    return {"MLE_m":float(error.mean()),"MedLE_m":float(np.median(error)),"P90_m":float(np.percentile(error,90)),"P95_m":float(np.percentile(error,95)),
            "ATE_RMSE_m":float(np.sqrt(np.mean(error**2))),"RPE_m":float(rpe.mean()),"JumpRate_pct":jump_rate,"JumpThreshold_m":jump_threshold,
            "LSR@5_pct":float((error<=5).mean()*100),"LSR@10_pct":float((error<=10).mean()*100),"LSR@15_pct":float((error<=15).mean()*100),"LSR@20_pct":float((error<=20).mean()*100),"MaxLE_m":float(error.max())}


@torch.no_grad()
def run_inference(model, visual, gallery_z_sat, cache, route, banks, device, csv_path):
    model.eval(); hidden,cell=model.initial_state(1,device,torch.float32); active_leg_index=0; visual_xy=route.legs[0].start_xy.to(device).reshape(1,2); previous_z_uav=None
    kf=make_position_kalman(visual_xy[0].cpu().numpy()); rows=[]
    for sequence_index in range(len(cache)):
        frame_id=int(cache.frame_ids[sequence_index].item()); active_leg=route.legs[active_leg_index]; previous_xy=visual_xy; active_before=active_leg_index
        uav_clip=cache.uav_clip[sequence_index:sequence_index+1].to(device).float(); z_uav=visual.model.encode_uav_from_clip(uav_clip)
        local=build_local_forward_candidate(visual,uav_clip,z_uav,visual_xy,active_leg,banks[active_leg_index],gallery_z_sat)
        recovery=build_topology_recovery_candidate(visual,z_uav,route,banks,active_leg_index,gallery_z_sat)
        leg_evidence,leg_valid_mask=build_leg_evidence(visual,z_uav,route,banks,active_leg_index,gallery_z_sat); context=route_context(route,active_leg_index,device,z_uav.dtype)
        output=model.forward_step(z_uav=z_uav,previous_z_uav=previous_z_uav,local_z_sat=local.z_sat,local_raw_logits=local.raw_logits,local_valid_mask=local.valid_mask,
            recovery_z_sat=recovery.z_sat,recovery_raw_logits=recovery.raw_logits,recovery_valid_mask=recovery.valid_mask,leg_evidence=leg_evidence,leg_valid_mask=leg_valid_mask,
            route_context=context,hidden=hidden,cell=cell)
        local_xy,_,local_index=decode_candidate(local); recovery_xy,_,recovery_index=decode_candidate(recovery)
        current_visual_xy,_,fusion_probability,branch_index=compose_localization(visual_xy,local_xy,recovery_xy,output.fusion_logits,False)
        leg_probability=output.leg_probability[0]; leg_choice=int(leg_probability.argmax().cpu()); active_leg_index=apply_leg_choice(active_leg_index,leg_choice,route); active_after=active_leg_index; state_delta=active_after-active_before
        kf.predict(); measurement_variance=np.clip(output.measurement_variance[0].cpu().numpy().astype(np.float64),float(config.KALMAN_R_MIN_VAR),float(config.KALMAN_R_MAX_VAR)); kf.R=np.diag(measurement_variance); kf.update(current_visual_xy[0].cpu().numpy().astype(np.float64)); final_xy=np.asarray(kf.x,dtype=np.float64).reshape(2)
        gt_xy=cache.gt_xy[sequence_index].numpy(); visual_np=current_visual_xy[0].cpu().numpy(); local_np=local_xy[0].cpu().numpy(); recovery_np=recovery_xy[0].cpu().numpy(); fp=fusion_probability[0].cpu().numpy(); lp=leg_probability.cpu().numpy()
        rows.append({"sequence_index":int(sequence_index),"frame_id":int(frame_id),"image_path":cache.image_paths[sequence_index],"active_leg_before":int(active_before),"active_leg_after":int(active_after),
            "active_waypoint_from":int(active_leg.start.order),"active_waypoint_to":int(active_leg.end.order),"search_heading_deg":float(heading_degrees(active_leg)),"forward_local_candidate_count":int(local.centers.shape[1]),
            "topology_recovery_candidate_count":int(recovery.centers.shape[1]),"selected_branch":int(branch_index[0].cpu()),"selected_local_index":int(local_index[0].cpu()),"selected_recovery_index":int(recovery_index[0].cpu()),
            "leg_choice":int(leg_choice),"leg_state_delta":int(state_delta),"leg_probability_previous":float(lp[0]),"leg_probability_current":float(lp[1]),"leg_probability_next":float(lp[2]),
            "waypoint_switched_after_frame":int(state_delta!=0),"waypoint_advanced_after_frame":int(state_delta>0),"waypoint_rollback_after_frame":int(state_delta<0),"gt_x":float(gt_xy[0]),"gt_y":float(gt_xy[1]),
            "hold_x":float(previous_xy[0,0].cpu()),"hold_y":float(previous_xy[0,1].cpu()),"local_x":float(local_np[0]),"local_y":float(local_np[1]),"recovery_x":float(recovery_np[0]),"recovery_y":float(recovery_np[1]),
            "fusion_hold":float(fp[0]),"fusion_local":float(fp[1]),"fusion_recovery":float(fp[2]),"visual_x":float(visual_np[0]),"visual_y":float(visual_np[1]),"measurement_var_x":float(measurement_variance[0]),"measurement_var_y":float(measurement_variance[1]),
            "final_x":float(final_xy[0]),"final_y":float(final_xy[1]),"error_visual_m":float(np.linalg.norm(visual_np-gt_xy)),"error_final_m":float(np.linalg.norm(final_xy-gt_xy))})
        hidden,cell=output.hidden,output.cell; previous_z_uav=z_uav; visual_xy=current_visual_xy
    gt=np.asarray([[r["gt_x"],r["gt_y"]] for r in rows],dtype=np.float64); vp=np.asarray([[r["visual_x"],r["visual_y"]] for r in rows],dtype=np.float64); fpred=np.asarray([[r["final_x"],r["final_y"]] for r in rows],dtype=np.float64)
    summary={"ReversibleTopologyVisual":metric_block(vp,gt),"FinalPositionKalman":metric_block(fpred,gt),"LegStateChangeCount":int(sum(r["waypoint_switched_after_frame"] for r in rows)),
             "LegAdvanceCount":int(sum(r["waypoint_advanced_after_frame"] for r in rows)),"LegRollbackCount":int(sum(r["waypoint_rollback_after_frame"] for r in rows)),
             "MeanHoldProbability":float(np.mean([r["fusion_hold"] for r in rows])),"MeanLocalProbability":float(np.mean([r["fusion_local"] for r in rows])),"MeanRecoveryProbability":float(np.mean([r["fusion_recovery"] for r in rows])),
             "MeanLegPreviousProbability":float(np.mean([r["leg_probability_previous"] for r in rows])),"MeanLegCurrentProbability":float(np.mean([r["leg_probability_current"] for r in rows])),"MeanLegNextProbability":float(np.mean([r["leg_probability_next"] for r in rows]))}
    csv_path=Path(csv_path); csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    return summary,rows


def route_catalog(): return {name:Path(root) for name,root in zip(config.ROUTE_NAMES,config.ROUTE_ROOTS)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("train","eval","train_eval"),default="train_eval"); parser.add_argument("--visual-epochs",type=int,default=config.VISUAL_EPOCHS); parser.add_argument("--temporal-epochs",type=int,default=config.TEMPORAL_EPOCHS); parser.add_argument("--reuse-visual",action="store_true"); args=parser.parse_args()
    if args.mode in ("train","train_eval") and int(args.temporal_epochs)<1: raise ValueError("temporal epochs must be >= 1")
    set_seed(config.SEED); device=torch.device(config.DEVICE if torch.cuda.is_available() else "cpu"); config.CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*100,flush=True); print("REVERSIBLE TOPOLOGY RECOVERY LSTM v10",flush=True); print("="*100,flush=True)
    print("LOCAL=forward 3x6. RECOVERY=full previous/current/next corridors. LEG STATE can advance or rollback.",flush=True)
    print("Route-A temporal training is zero-teacher from epoch 1. No endpoint-distance gate, motion vector, polynomial, or ECC heading.",flush=True)
    print("Hard visual XY -> learned-variance position-only Kalman [x,y].",flush=True); print("="*100,flush=True)
    if args.mode in ("train","train_eval"):
        if args.reuse_visual:
            if not config.VISUAL_CHECKPOINT.exists(): raise FileNotFoundError("--reuse-visual requested but visual checkpoint is missing: %s" % config.VISUAL_CHECKPOINT)
            print("reuse Route-A visual checkpoint: %s" % config.VISUAL_CHECKPOINT,flush=True)
        else:
            if config.VISUAL_CHECKPOINT.exists(): config.VISUAL_CHECKPOINT.unlink()
            train_visual_retrieval_a_only(device=device,epochs=int(args.visual_epochs),jitter_m=float(config.LOCAL_PRIOR_JITTER_M),resume=False)
    if not config.VISUAL_CHECKPOINT.exists(): raise FileNotFoundError(config.VISUAL_CHECKPOINT)
    visual=FrozenVisualLocalizer(device); gallery_z_sat=build_gallery_embeddings(visual); model=ReversibleTopologyRecoveryLSTM().to(device); catalog=route_catalog(); route_a=load_mission_route("route_A",visual.origin_lat,visual.origin_lon)
    if args.mode in ("train","train_eval"):
        if config.TEMPORAL_CHECKPOINT.exists(): config.TEMPORAL_CHECKPOINT.unlink()
        cache=build_route_cache("route_A",catalog["route_A"],visual,device); banks=build_route_banks(visual,route_a); train_temporal_model(model,visual,gallery_z_sat,cache,route_a,banks,device,int(args.temporal_epochs))
    else: load_temporal_model(model,device)
    if args.mode in ("eval","train_eval"):
        if args.mode=="eval": load_temporal_model(model,device)
        route_results={}
        for route_name in ("route_B","route_C"):
            route=load_mission_route(route_name,visual.origin_lat,visual.origin_lon); cache=build_route_cache(route_name,catalog[route_name],visual,device); banks=build_route_banks(visual,route); csv_path=config.OUTPUT_DIR/(route_name+"_reversible_topology_frames.csv")
            summary,_=run_inference(model,visual,gallery_z_sat,cache,route,banks,device,csv_path); route_results[route_name]=summary; metric=summary["FinalPositionKalman"]
            print("%s: MLE=%.3fm P90=%.3fm RPE=%.3fm Jump=%.3f%% changes=%d advance=%d rollback=%d H/L/R=%.2f/%.2f/%.2f" % (route_name,metric["MLE_m"],metric["P90_m"],metric["RPE_m"],metric["JumpRate_pct"],summary["LegStateChangeCount"],summary["LegAdvanceCount"],summary["LegRollbackCount"],summary["MeanHoldProbability"],summary["MeanLocalProbability"],summary["MeanRecoveryProbability"]),flush=True)
        payload={"architecture":ARCHITECTURE_NAME,"changes":{"forward_half_local_only":True,"recovery_forward_restriction":False,"recovery_predicted_xy_center":False,"recovery_legs":"previous/current/next","leg_state_reversible":True,"endpoint_distance_gate":False,"observed_motion_input":False,"polynomial_input":False,"teacher_forcing":False,"post_model_position_only_kalman":True},
                 "training":{"route":"route_A","teacher_ratio":0.0,"test_routes_used_for_training":False,"test_gt_input":False,"test_waypoint_frame_index_input":False},"routes":route_results}
        config.OUTPUT_DIR.mkdir(parents=True,exist_ok=True); summary_path=config.OUTPUT_DIR/"robust_tracker_summary.json"; summary_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); print("summary: %s" % summary_path,flush=True)

if __name__=="__main__": main()
