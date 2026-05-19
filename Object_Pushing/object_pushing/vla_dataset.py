from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INSTRUCTION_TEMPLATES = [
    "Push the cylinder toward the target marker.",
    "Move the cylinder to the goal position.",
    "Push the object to the designated spot.",
    "Navigate to the cylinder and push it to the target.",
    "Clear the cylinder by pushing it to the marked location.",
    "Go to the cylinder and move it toward the goal.",
    "Push the round object to where the marker is.",
    "Move the cylinder across the room to the target.",
]

NAVILA_ACTION_RE = re.compile(
    r"^(?:stop|move forward \d+ cm|turn (?:left|right) \d+ degrees?(?:, move forward \d+ cm)?)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ActionRecord:
    frame_idx: int
    forward_cm: float
    rotation_deg: float
    action_label: str


@dataclass(frozen=True)
class MergedAction:
    frame_idx: int
    action_label: str
    merge_count: int
    forward_cm: float
    rotation_deg: float


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_wxyz(quat: tuple[float, float, float, float] | list[float] | np.ndarray) -> float:
    """Return yaw from a quaternion in Isaac/Usd wxyz ordering."""
    w, x, y, z = [float(value) for value in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_action_label(
    forward_cm: float,
    rotation_deg: float,
    *,
    move_threshold_cm: float = 1.0,
    rotation_threshold_deg: float = 1.0,
) -> str:
    """Build a NaVILA-compatible text action label.

    NaVILA's navigation SFT/eval path supports stop, forward, and yaw primitives.
    Lateral and backward displacement are intentionally not labelled here.
    """
    parts: list[str] = []
    if abs(rotation_deg) > rotation_threshold_deg:
        if rotation_deg > 0.0:
            parts.append(f"turn left {abs(rotation_deg):.0f} degrees")
        else:
            parts.append(f"turn right {abs(rotation_deg):.0f} degrees")

    if forward_cm > move_threshold_cm:
        parts.append(f"move forward {forward_cm:.0f} cm")

    if not parts:
        return "stop"
    return ", ".join(parts)


def compute_action_record(
    frame_idx: int,
    pose_before: tuple[float, float, float],
    pose_after: tuple[float, float, float],
    *,
    move_threshold_cm: float = 1.0,
    rotation_threshold_deg: float = 1.0,
) -> ActionRecord:
    """Compute a label from actual robot pose delta.

    Poses are `(x, y, yaw)` in the world frame. Translation is projected onto
    the robot's forward axis at `pose_before`, matching NaVILA-style egocentric
    action labels.
    """
    dx = float(pose_after[0]) - float(pose_before[0])
    dy = float(pose_after[1]) - float(pose_before[1])
    dyaw = normalize_angle(float(pose_after[2]) - float(pose_before[2]))

    forward_cm = (dx * math.cos(pose_before[2]) + dy * math.sin(pose_before[2])) * 100.0
    rotation_deg = math.degrees(dyaw)
    return ActionRecord(
        frame_idx=int(frame_idx),
        forward_cm=float(forward_cm),
        rotation_deg=float(rotation_deg),
        action_label=build_action_label(
            forward_cm,
            rotation_deg,
            move_threshold_cm=move_threshold_cm,
            rotation_threshold_deg=rotation_threshold_deg,
        ),
    )


def validate_navila_action_label(label: str) -> bool:
    return NAVILA_ACTION_RE.match(label.strip()) is not None


def _as_action_record(record: ActionRecord | dict[str, Any]) -> ActionRecord:
    if isinstance(record, ActionRecord):
        return record
    return ActionRecord(
        frame_idx=int(record["frame_idx"]),
        forward_cm=float(record["forward_cm"]),
        rotation_deg=float(record["rotation_deg"]),
        action_label=str(record.get("action_label") or ""),
    )


def _same_merge_direction(
    current: ActionRecord,
    next_record: ActionRecord,
    *,
    move_threshold_cm: float,
    rotation_threshold_deg: float,
) -> bool:
    current_rot = current.rotation_deg
    next_rot = next_record.rotation_deg
    if abs(current_rot) < rotation_threshold_deg and abs(next_rot) < rotation_threshold_deg:
        return current.forward_cm > move_threshold_cm and next_record.forward_cm > move_threshold_cm
    if current_rot > rotation_threshold_deg and next_rot > rotation_threshold_deg:
        return True
    if current_rot < -rotation_threshold_deg and next_rot < -rotation_threshold_deg:
        return True
    return False


def merge_consecutive_actions(
    action_sequence: list[ActionRecord | dict[str, Any]],
    *,
    max_merge: int = 3,
    move_threshold_cm: float = 1.0,
    rotation_threshold_deg: float = 1.0,
) -> list[MergedAction]:
    """Merge up to `max_merge` consecutive same-direction primitives."""
    records = [_as_action_record(record) for record in action_sequence]
    merged: list[MergedAction] = []
    i = 0
    while i < len(records):
        current = records[i]
        accumulated_forward = current.forward_cm
        accumulated_rotation = current.rotation_deg
        merge_count = 1

        while merge_count < max_merge and i + merge_count < len(records):
            next_record = records[i + merge_count]
            if not _same_merge_direction(
                current,
                next_record,
                move_threshold_cm=move_threshold_cm,
                rotation_threshold_deg=rotation_threshold_deg,
            ):
                break
            accumulated_forward += next_record.forward_cm
            accumulated_rotation += next_record.rotation_deg
            merge_count += 1

        merged_label = build_action_label(
            accumulated_forward,
            accumulated_rotation,
            move_threshold_cm=move_threshold_cm,
            rotation_threshold_deg=rotation_threshold_deg,
        )
        merged.append(
            MergedAction(
                frame_idx=current.frame_idx,
                action_label=merged_label,
                merge_count=merge_count,
                forward_cm=float(accumulated_forward),
                rotation_deg=float(accumulated_rotation),
            )
        )
        i += merge_count
    return merged


def sample_history_indices(current_idx: int, *, num_history: int = 7) -> list[int]:
    """Sample previous frame indices with frame 0 always included when possible."""
    current_idx = int(current_idx)
    num_history = max(0, int(num_history))
    if current_idx <= 0 or num_history == 0:
        return []
    if current_idx <= num_history:
        return list(range(current_idx))
    if num_history == 1:
        return [0]

    indices = [0]
    step = (current_idx - 1) / float(num_history - 1)
    for i in range(1, num_history):
        indices.append(int(i * step))
    return indices


def sample_navila_frame_indices(current_idx: int, *, num_history: int = 7) -> list[int]:
    return [*sample_history_indices(current_idx, num_history=num_history), int(current_idx)]


def generate_instruction(rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(INSTRUCTION_TEMPLATES)


def to_pil_rgb(image: Any) -> Image.Image:
    """Convert Isaac camera output, numpy arrays, or PIL images to RGB PIL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def save_episode_dataset(
    save_dir: str | os.PathLike[str],
    episode_id: int,
    frames: list[Any],
    merged_actions: list[MergedAction | dict[str, Any]],
    instruction: str,
    *,
    num_history: int = 7,
    metadata: dict[str, Any] | None = None,
    overwrite_episode: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Save one successful episode and return `(samples, navila_annotations)`."""
    root = Path(save_dir)
    episode_name = f"episode_{int(episode_id):04d}"
    episode_dir = root / episode_name
    frames_dir = episode_dir / "frames"
    if episode_dir.exists() and not overwrite_episode:
        raise FileExistsError(f"Episode directory already exists: {episode_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    for frame_idx, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
        to_pil_rgb(frame).save(frame_path)
        frame_paths.append(_rel(frame_path, root))

    samples: list[dict[str, Any]] = []
    navila_annotations: list[dict[str, Any]] = []
    for merged in merged_actions:
        if not isinstance(merged, MergedAction):
            merged = MergedAction(
                frame_idx=int(merged["frame_idx"]),
                action_label=str(merged["action_label"]),
                merge_count=int(merged.get("merge_count", 1)),
                forward_cm=float(merged.get("forward_cm", 0.0)),
                rotation_deg=float(merged.get("rotation_deg", 0.0)),
            )
        if merged.frame_idx < 0 or merged.frame_idx >= len(frame_paths):
            continue
        history_indices = sample_history_indices(merged.frame_idx, num_history=num_history)
        navila_indices = [*history_indices, merged.frame_idx]
        history_paths = [frame_paths[idx] for idx in history_indices]
        current_path = frame_paths[merged.frame_idx]
        sample_id = f"push_ep{int(episode_id):04d}_step{merged.frame_idx:03d}"
        sample = {
            "id": sample_id,
            "episode_id": int(episode_id),
            "instruction": instruction,
            "history_indices": history_indices,
            "history_frame_paths": history_paths,
            "current_idx": merged.frame_idx,
            "current_frame_path": current_path,
            "action_label": merged.action_label,
            "merge_count": merged.merge_count,
            "forward_cm": merged.forward_cm,
            "rotation_deg": merged.rotation_deg,
        }
        samples.append(sample)
        navila_annotations.append(
            {
                "id": sample_id,
                "video_id": episode_name,
                "frames": [frame_paths[idx] for idx in navila_indices],
                "q": instruction,
                "a": merged.action_label,
                "episode_id": int(episode_id),
                "current_idx": merged.frame_idx,
                "merge_count": merged.merge_count,
            }
        )

    episode_metadata = {
        "episode_id": int(episode_id),
        "instruction": instruction,
        "num_frames": len(frame_paths),
        "num_samples": len(samples),
        "num_history": int(num_history),
    }
    if metadata:
        episode_metadata.update(metadata)

    with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(episode_metadata, f, indent=2)
    with (episode_dir / "samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    return samples, navila_annotations


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    invalid_labels: list[str] = []
    for sample in samples:
        label = str(sample["action_label"])
        action_counts[label] = action_counts.get(label, 0) + 1
        if not validate_navila_action_label(label):
            invalid_labels.append(label)
    return {
        "num_samples": len(samples),
        "action_counts": action_counts,
        "invalid_action_labels": sorted(set(invalid_labels)),
    }


def action_records_to_dict(records: list[ActionRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def merged_actions_to_dict(records: list[MergedAction]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
