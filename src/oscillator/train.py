"""Train-only normalisation, development selection and clip-held-out evaluation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .model import CadenceNetwork, HarmonicDecoder, decoder_spec, make_decoder
from .preprocess import read_prepared, reconstruct_root
from .runtime import integrate_planar_path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 0.003
    mixture_penalty: float = 0.001
    harmonic_penalty: float = 0.0001
    weight_decay: float = 0.00001
    seed: int = 7
    cadence: bool = False
    device: str = "cpu"
    tau_candidates: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    harmonics: int = 3

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Epochs, batch size and learning rate must be positive")
        if min(self.mixture_penalty, self.harmonic_penalty, self.weight_decay) < 0:
            raise ValueError("Regularisation penalties must be non-negative")
        if not isinstance(self.harmonics, int) or self.harmonics < 1:
            raise ValueError("Harmonic count must be a positive integer")
        if not self.tau_candidates or any(
            t <= 0 or not np.isfinite(t) for t in self.tau_candidates
        ):
            raise ValueError("Smoothing candidates must be positive and finite")


def frame_tensors(clips, names, device):
    arrays = {}
    for key in ("phase", "context", "target"):
        arrays[key] = torch.as_tensor(
            np.concatenate([clips[name][key][clips[name]["valid"]] for name in names]),
            dtype=torch.float32,
            device=device,
        )
    return arrays


@torch.no_grad()
def coordinate_metrics(prediction: np.ndarray, target: np.ndarray):
    error = prediction - target
    return {
        "joint_rmse_rad": float(np.sqrt(np.mean(error[:, :22] ** 2))),
        "root_residual_translation_rmse_m": float(np.sqrt(np.mean(error[:, 22:25] ** 2))),
        "root_residual_rotvec_rmse_rad": float(np.sqrt(np.mean(error[:, 25:] ** 2))),
    }


@torch.no_grad()
def select_smoothing(model, clips, dev_names, candidates):
    """Select a fixed tau using natural cycle-context changes in development clips.

    Only complete, contiguous cycles are used. Score a one-cycle window following each
    non-trivial context change. This is a development proxy for command transitions,
    rather than evidence of performance on arbitrary commanded manoeuvres.
    """
    device = model.target_mean.device
    scores = {}
    transitions = 0
    for tau in candidates:
        errors = []
        transitions = 0
        for name in dev_names:
            clip = clips[name]
            indices = np.flatnonzero(clip["valid"])
            x = torch.as_tensor(clip["context"][indices], dtype=torch.float32, device=device)
            phi = torch.as_tensor(clip["phase"][indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(clip["target"][indices], dtype=torch.float32, device=device)
            targets = (targets - model.target_mean) / model.target_scale
            z_target = model.mixture(x)
            smoothed = torch.empty_like(z_target)
            z = z_target[0].clone()
            transition_mask = np.zeros(len(indices), dtype=bool)
            window_end = -1
            for i, frame in enumerate(indices):
                if i == 0 or frame != indices[i - 1] + 1:
                    z = z_target[i].clone()
                    window_end = -1
                else:
                    change = torch.linalg.vector_norm((x[i] - x[i - 1]) / model.context_scale)
                    if float(change) > 0.05:
                        transitions += 1
                        window_end = frame + int(round(float(x[i, 3]) * float(clip["fps"])))
                    z = z_target[i] + (z - z_target[i]) * np.exp(-1 / float(clip["fps"]) / tau)
                smoothed[i] = z
                transition_mask[i] = frame < window_end
            if transition_mask.any():
                prediction = model.decode(phi, smoothed, normalised=True)
                errors.append((prediction[transition_mask] - targets[transition_mask]).square())
        if not errors:
            raise ValueError("No development context transitions available for smoothing selection")
        scores[str(tau)] = float(torch.cat(errors).mean())
    selected = min(candidates, key=lambda t: scores[str(t)])
    return selected, {
        "tau_z_s": selected,
        "normalised_transition_mse": scores,
        "development_transitions": transitions,
        "protocol": "natural development cycle-context changes; one-cycle windows",
    }


@torch.no_grad()
def evaluate(model, clips, names, cadence=None, skeleton=None):
    device = model.target_mean.device
    reports = {}
    for name in names:
        clip = clips[name]
        mask = clip["valid"]
        phi = torch.as_tensor(clip["phase"][mask], dtype=torch.float32, device=device)
        x = torch.as_tensor(clip["context"][mask], dtype=torch.float32, device=device)
        target = clip["target"][mask]
        prediction = model(phi, x).cpu().numpy()
        position, quat = reconstruct_root(
            prediction[:, 22:], clip["path_pos"][mask], clip["path_yaw"][mask]
        )
        angular_error = (
            Rotation.from_quat(clip["root_rot_xyzw"][mask]).inv() * Rotation.from_quat(quat)
        ).magnitude()
        reconstruction = coordinate_metrics(prediction, target)
        reconstruction.update(
            {
                "world_root_position_rmse_m": float(
                    np.sqrt(np.mean(np.sum((position - clip["root_pos"][mask]) ** 2, axis=1)))
                ),
                "world_root_orientation_rmse_rad": float(np.sqrt(np.mean(angular_error**2))),
            }
        )
        if skeleton is not None:
            from .kinematics import foot_metrics

            reconstruction.update(foot_metrics(skeleton, prediction, position, quat, clip, mask))
        # Reset phase once, then roll uniformly without observed strike corrections.
        # Clip-average context is an oracle condition, explicitly labelled in the report.
        average_context = x.mean(0)
        hz = (
            float(cadence(average_context[:3]))
            if cadence is not None
            else 1 / float(average_context[3])
        )
        average_context = average_context.clone()
        average_context[3] = 1 / hz
        time = clip["time"][mask]
        free_phase = (float(phi[0]) + (time - time[0]) * hz) % 1
        free_phase = torch.as_tensor(free_phase, dtype=torch.float32, device=device)
        free_prediction = model(free_phase, average_context.expand(len(phi), -1)).cpu().numpy()
        command = average_context.cpu().numpy()
        path_start = clip["path_pos"][mask][0]
        yaw_start = float(clip["path_yaw"][mask][0])
        free_paths = [
            integrate_planar_path(path_start, yaw_start, command, float(t - time[0])) for t in time
        ]
        free_root, free_quat = reconstruct_root(
            free_prediction[:, 22:],
            np.stack([p for p, _ in free_paths]),
            np.array([yaw for _, yaw in free_paths]),
        )
        free_angular_error = (
            Rotation.from_quat(clip["root_rot_xyzw"][mask]).inv() * Rotation.from_quat(free_quat)
        ).magnitude()
        phase_error = (free_phase.cpu().numpy() - clip["phase"][mask] + 0.5) % 1 - 0.5
        reports[name] = {
            "frames": int(mask.sum()),
            "reconstruction": reconstruction,
            "uniform_free_rollout": {
                **coordinate_metrics(free_prediction, target),
                "phase_rmse_cycles": float(np.sqrt(np.mean(phase_error**2))),
                "cycle_hz": hz,
                "cadence_source": "learned" if cadence else "oracle_clip_mean",
                "context_source": "oracle_clip_mean",
                "phase_initialisation": "first_valid_observed_phase",
                "root_path_source": "integrated_constant_clip_mean_command",
                "world_root_position_rmse_m": float(
                    np.sqrt(np.mean(np.sum((free_root - clip["root_pos"][mask]) ** 2, axis=1)))
                ),
                "world_root_orientation_rmse_rad": float(np.sqrt(np.mean(free_angular_error**2))),
            },
        }
    return reports


def train(prepared: Path, output: Path, config=None, *, evaluate_test=True):
    config = TrainConfig() if config is None else config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device == "cpu":
        torch.set_num_threads(2)
    metadata, clips = read_prepared(prepared)
    splits = metadata["splits"]
    training = frame_tensors(clips, splits["train"], config.device)
    dev = frame_tensors(clips, splits["dev"], config.device)
    model = HarmonicDecoder(config.harmonics).to(config.device)
    model.set_normalisation(training["context"], training["target"])
    train_target = (training["target"] - model.target_mean) / model.target_scale
    dev_target = (dev["target"] - model.target_mean) / model.target_scale
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(train_target), device=config.device)
        total = 0.0
        for indices in permutation.split(config.batch_size):
            z = model.mixture(training["context"][indices])
            prediction = model.decode(training["phase"][indices], z, normalised=True)
            mse = (prediction - train_target[indices]).square().mean()
            mixture_reg, harmonic_reg = model.regularisation(z)
            loss = (
                mse + config.mixture_penalty * mixture_reg + config.harmonic_penalty * harmonic_reg
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total += float(mse.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            dev_loss = float(
                (model(dev["phase"], dev["context"], normalised=True) - dev_target).square().mean()
            )
        history.append(
            {"epoch": epoch + 1, "train_mse": total / len(train_target), "dev_mse": dev_loss}
        )
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == config.epochs:
            print(
                f"epoch {epoch + 1:4d}: train={history[-1]['train_mse']:.4f} dev={dev_loss:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    cadence = None
    cadence_report = None
    if config.cadence:
        cadence, cadence_report = train_cadence(training, dev, config)
    tau, smoothing_report = select_smoothing(model, clips, splits["dev"], config.tau_candidates)
    # Test clips are evaluated once after development selection; never used for tuning.
    metrics = {
        split: evaluate(model, clips, names, cadence)
        for split, names in splits.items()
        if split != "test" or evaluate_test
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "decoder_spec": decoder_spec(model),
        "decoder_state": best_state,
        "cadence_state": None
        if cadence is None
        else {k: v.detach().cpu() for k, v in cadence.state_dict().items()},
        "cadence_bounds": None if cadence is None else [cadence.min_hz, cadence.max_hz],
        "tau_z_s": tau,
        "joint_names": metadata["joint_names"],
        "preprocessing": metadata,
        "train_config": asdict(config),
    }
    torch.save(checkpoint, output / "model.pt")
    report = {
        "decoder_parameters": model.parameter_count,
        "cadence_parameters": 0
        if cadence is None
        else sum(p.numel() for p in cadence.parameters()),
        "best_epoch": min(history, key=lambda r: r["dev_mse"])["epoch"],
        "best_dev_mse": best_loss,
        "smoothing": smoothing_report,
        "cadence": cadence_report,
        "metrics": metrics,
        "history": history,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_cadence(training, dev, config):
    model = CadenceNetwork().to(config.device)
    model.command_mean.copy_(training["context"][:, :3].mean(0))
    model.command_scale.copy_(training["context"][:, :3].std(0, correction=0).clamp_min(1e-3))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss, best_state, best_epoch = float("inf"), None, None
    for epoch in range(config.epochs):
        # One example per cycle: do not overweight a slower cadence with more frames.
        x = torch.unique(training["context"], dim=0)
        loss = (model(x[:, :3]) - 1 / x[:, 3]).square().mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            dx = torch.unique(dev["context"], dim=0)
            dev_loss = float((model(dx[:, :3]) - 1 / dx[:, 3]).square().mean())
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), {"best_epoch": best_epoch, "dev_cycle_hz_rmse": float(np.sqrt(best_loss))}


def load_checkpoint(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    model = make_decoder(checkpoint.get("decoder_spec", {"kind": "harmonic", "harmonics": 3})).to(
        device
    )
    model.load_state_dict(checkpoint["decoder_state"])
    cadence = None
    if checkpoint["cadence_state"] is not None:
        cadence = CadenceNetwork(*checkpoint["cadence_bounds"]).to(device)
        cadence.load_state_dict(checkpoint["cadence_state"])
        cadence.eval()
    return model.eval(), cadence, checkpoint
