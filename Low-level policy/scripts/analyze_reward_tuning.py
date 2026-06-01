#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:  # pragma: no cover
    event_accumulator = None


REWARD_KEYS = [
    "feet_slide",
    "feet_stuck_time_penalty",
    "feet_gait",
    "feet_contact_count_penalty",
    "inter_diagonal_load_balance",
    "joint_mirror",
    "feet_air_time",
    "feet_air_time_variance",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
]

TB_TERMS = [
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/feet_air_time_variance",
    "Episode_Reward/feet_gait",
    "Episode_Reward/feet_contact_count_penalty",
    "Episode_Reward/feet_stuck_time_penalty",
    "Episode_Reward/joint_mirror",
    "Episode_Reward/contact_forces",
    "Episode_Reward/feet_slide",
    "Episode_Reward/inter_diagonal_load_balance",
    "Train/mean_reward",
]

LEGS = ["FL", "FR", "RL", "RR"]


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def find_latest_runs(run_root: Path, top_k: int = 2) -> list[Path]:
    event_files = sorted(run_root.glob("*/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return event_files[:top_k]


def load_run_weights(run_dir: Path) -> dict[str, float]:
    env_yaml = run_dir / "params" / "env.yaml"
    if yaml is None or not env_yaml.exists():
        return {}
    data = yaml.unsafe_load(env_yaml.read_text())
    rewards = data.get("rewards", {})
    weights: dict[str, float] = {}
    for key in REWARD_KEYS:
        node = rewards.get(key)
        if isinstance(node, dict) and "weight" in node:
            weights[key] = float(node["weight"])
    return weights


def load_current_weights(config_path: Path) -> dict[str, float]:
    text = config_path.read_text()
    weights: dict[str, float] = {}
    for key in REWARD_KEYS:
        pattern = re.compile(rf"self\.rewards\.{re.escape(key)}\.weight\s*=\s*([-+0-9.eE]+)")
        match = pattern.search(text)
        if match:
            weights[key] = float(match.group(1))
    return weights


def load_tb_scalars(event_file: Path) -> dict[str, float]:
    if event_accumulator is None:
        return {}
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    result: dict[str, float] = {}
    for term in TB_TERMS:
        if term in tags:
            result[term] = float(ea.Scalars(term)[-1].value)
    return result


def load_csv_metrics(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    n_rows = len(rows)
    time_start = _float(rows[0].get("time_s", 0.0))
    time_end = _float(rows[-1].get("time_s", 0.0))
    dt = (time_end - time_start) / max(n_rows - 1, 1)
    tail_start = 2 * n_rows // 3

    per_leg: dict[str, dict[str, float]] = {}
    per_leg_tail: dict[str, dict[str, float]] = {}

    for leg in LEGS:
        foot = f"{leg}_foot"
        contact = [int(float(row[f"contact_{foot}"])) for row in rows]
        fz = [_float(row[f"W_grf_z_{foot}"]) for row in rows]
        fx = [_float(row[f"W_grf_x_{foot}"]) for row in rows]
        fy = [_float(row[f"W_grf_y_{foot}"]) for row in rows]
        ft = [math.hypot(x, y) for x, y in zip(fx, fy)]
        idx = [i for i, ci in enumerate(contact) if ci]
        ratios = [ft[i] / (abs(fz[i]) + 1e-6) for i in idx if abs(fz[i]) > 1e-6]
        vcols = [f"joint_vel_{leg}HAA", f"joint_vel_{leg}HIP", f"joint_vel_{leg}KNEE"]
        vel_contact = [sum(abs(_float(rows[i][col])) for col in vcols) / 3.0 for i in idx]
        per_leg[leg] = {
            "contact_ratio": sum(contact) / n_rows,
            "mean_fz_all": sum(fz) / n_rows,
            "mean_fz_contact": (sum(fz[i] for i in idx) / len(idx)) if idx else 0.0,
            "mean_ft_contact": (sum(ft[i] for i in idx) / len(idx)) if idx else 0.0,
            "mean_ft_over_mean_fz": (
                (sum(ft[i] for i in idx) / len(idx)) / ((sum(fz[i] for i in idx) / len(idx)) + 1e-6) if idx else 0.0
            ),
            "median_ft_over_fz": statistics.median(ratios) if ratios else 0.0,
            "joint_vel_abs_mean_contact": (sum(vel_contact) / len(vel_contact)) if vel_contact else 0.0,
        }

        tail_rows = rows[tail_start:]
        tail_contact = [int(float(row[f"contact_{foot}"])) for row in tail_rows]
        tail_fz = [_float(row[f"W_grf_z_{foot}"]) for row in tail_rows]
        tail_fx = [_float(row[f"W_grf_x_{foot}"]) for row in tail_rows]
        tail_fy = [_float(row[f"W_grf_y_{foot}"]) for row in tail_rows]
        tail_ft = [math.hypot(x, y) for x, y in zip(tail_fx, tail_fy)]
        tail_idx = [i for i, ci in enumerate(tail_contact) if ci]
        tail_ratios = [tail_ft[i] / (abs(tail_fz[i]) + 1e-6) for i in tail_idx if abs(tail_fz[i]) > 1e-6]
        per_leg_tail[leg] = {
            "contact_ratio": sum(tail_contact) / max(len(tail_contact), 1),
            "mean_fz_contact": (sum(tail_fz[i] for i in tail_idx) / len(tail_idx)) if tail_idx else 0.0,
            "mean_ft_over_mean_fz": (
                (sum(tail_ft[i] for i in tail_idx) / len(tail_idx))
                / ((sum(tail_fz[i] for i in tail_idx) / len(tail_idx)) + 1e-6)
                if tail_idx
                else 0.0
            ),
            "median_ft_over_fz": statistics.median(tail_ratios) if tail_ratios else 0.0,
        }

    patterns = Counter(tuple(int(float(row[f"contact_{leg}_foot"])) for leg in LEGS) for row in rows)
    flrr = [_float(row["W_grf_z_FL_foot"]) + _float(row["W_grf_z_RR_foot"]) for row in rows]
    frrl = [_float(row["W_grf_z_FR_foot"]) + _float(row["W_grf_z_RL_foot"]) for row in rows]

    return {
        "rows": n_rows,
        "duration_s": time_end,
        "dt": dt,
        "per_leg": per_leg,
        "per_leg_tail": per_leg_tail,
        "top_patterns": patterns.most_common(8),
        "inter_diag": {
            "mean_FLRR": sum(flrr) / n_rows,
            "mean_FRRL": sum(frrl) / n_rows,
            "mean_abs_norm_diff": sum(abs(a - b) / (a + b + 1e-6) for a, b in zip(flrr, frrl)) / n_rows,
        },
    }


def detect_issues(metrics: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for leg, stat in metrics["per_leg"].items():
        tail = metrics["per_leg_tail"][leg]
        contact_ratio = stat["contact_ratio"]
        slide_ratio = stat["mean_ft_over_mean_fz"]
        joint_vel = stat["joint_vel_abs_mean_contact"]
        mean_fz_contact = stat["mean_fz_contact"]
        if contact_ratio > 0.75 and slide_ratio > 0.35 and joint_vel < 0.8:
            issues.append(
                f"{leg} stuck-slide support: contact_ratio={contact_ratio:.3f}, ft/fz={slide_ratio:.3f}, joint_vel={joint_vel:.3f}"
            )
        elif contact_ratio > 0.30 and slide_ratio > 0.40 and mean_fz_contact < 120.0:
            issues.append(
                f"{leg} slide-heavy support: contact_ratio={contact_ratio:.3f}, ft/fz={slide_ratio:.3f}, fz_contact={mean_fz_contact:.1f}"
            )
        if contact_ratio < 0.05 and tail["contact_ratio"] < 0.05:
            issues.append(f"{leg} unloaded/unused: contact_ratio={contact_ratio:.3f}, tail={tail['contact_ratio']:.3f}")

    inter_diag = metrics["inter_diag"]["mean_abs_norm_diff"]
    if inter_diag > 0.75:
        issues.append(f"large diagonal-load imbalance: mean_abs_norm_diff={inter_diag:.3f}")
    return issues


def suggest_weight_updates(
    current_weights: dict[str, float], tb_latest: dict[str, float], tb_prev: dict[str, float], issues: list[str]
) -> list[str]:
    suggestions: list[str] = []
    mean_reward = tb_latest.get("Train/mean_reward")
    prev_mean_reward = tb_prev.get("Train/mean_reward")
    feet_slide = tb_latest.get("Episode_Reward/feet_slide")
    stuck = tb_latest.get("Episode_Reward/feet_stuck_time_penalty")

    has_slide_support = any("slide-heavy" in issue or "stuck-slide" in issue for issue in issues)
    has_unloaded = any("unloaded/unused" in issue for issue in issues)

    if has_slide_support and stuck is not None and stuck > -0.02:
        cur = current_weights.get("feet_stuck_time_penalty")
        if cur is not None:
            suggestions.append(f"Increase feet_stuck_time_penalty: {cur:.3f} -> {cur - 0.5:.3f}")
    if has_slide_support and feet_slide is not None:
        cur = current_weights.get("feet_slide")
        if cur is not None and abs(cur) < 1.8:
            suggestions.append(f"Increase feet_slide magnitude: {cur:.3f} -> {cur - 0.3:.3f}")
    if has_unloaded and stuck is not None and stuck < -0.02:
        cur = current_weights.get("feet_contact_count_penalty")
        if cur is not None:
            suggestions.append(f"Increase feet_contact_count_penalty magnitude: {cur:.3f} -> {cur - 0.5:.3f}")
    if mean_reward is not None and prev_mean_reward is not None and mean_reward < prev_mean_reward - 10.0:
        cur = current_weights.get("feet_slide")
        if cur is not None:
            suggestions.append(f"Training reward dropped; ease feet_slide: {cur:.3f} -> {cur + 0.3:.3f}")
    if not suggestions:
        suggestions.append("No strong automatic weight change detected. Keep current weights and gather another run.")
    return suggestions


def print_report(
    csv_metrics: dict[str, Any],
    issues: list[str],
    latest_run: Path | None,
    run_weights: dict[str, float],
    current_weights: dict[str, float],
    tb_latest: dict[str, float],
    tb_prev: dict[str, float],
    suggestions: list[str],
) -> None:
    print("Reward Tuning Report")
    print("=" * 80)
    if latest_run is not None:
        print(f"Latest run: {latest_run.parent}")
    print(f"Rows: {csv_metrics['rows']}, duration_s: {csv_metrics['duration_s']:.3f}, dt: {csv_metrics['dt']:.6f}")
    print()

    print("Current Config Weights")
    for key in REWARD_KEYS:
        if key in current_weights:
            print(f"  {key}: {current_weights[key]:.3f}")
    print()

    if run_weights:
        print("Latest Run Effective Weights")
        for key in REWARD_KEYS:
            if key in run_weights:
                print(f"  {key}: {run_weights[key]:.3f}")
        print()

    print("Per-Foot Metrics")
    for leg in LEGS:
        stat = csv_metrics["per_leg"][leg]
        tail = csv_metrics["per_leg_tail"][leg]
        print(
            f"  {leg}: contact={stat['contact_ratio']:.3f}, fz_contact={stat['mean_fz_contact']:.1f}, "
            f"ft/fz={stat['mean_ft_over_mean_fz']:.3f}, vel_contact={stat['joint_vel_abs_mean_contact']:.3f}, "
            f"tail_contact={tail['contact_ratio']:.3f}"
        )
    print()

    print("Top Contact Patterns")
    for pattern, count in csv_metrics["top_patterns"]:
        ratio = count / csv_metrics["rows"]
        print(f"  {pattern}: {ratio:.3f} ({count})")
    print()

    print("Detected Issues")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  - none")
    print()

    if tb_latest:
        print("TensorBoard Latest")
        for term in TB_TERMS:
            if term in tb_latest:
                latest_value = tb_latest[term]
                if term in tb_prev:
                    print(f"  {term}: {latest_value:.6f} (prev {tb_prev[term]:.6f})")
                else:
                    print(f"  {term}: {latest_value:.6f}")
        print()

    print("Suggested Next Changes")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze locomotion CSV/TensorBoard logs and suggest reward-weight changes.")
    parser.add_argument("--csv-path", type=Path, default=Path("logs/data.csv"))
    parser.add_argument("--run-root", type=Path, default=Path("logs/rsl_rl/MCLrobotics_MCLQuadserial_flat"))
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("source/rl_training/rl_training/tasks/locomotion/config/MCL_Quad_serial/flat_env_cfg.py"),
    )
    args = parser.parse_args()

    runs = find_latest_runs(args.run_root, top_k=2)
    latest_run = runs[0] if runs else None
    prev_run = runs[1] if len(runs) > 1 else None

    csv_metrics = load_csv_metrics(args.csv_path)
    issues = detect_issues(csv_metrics)
    current_weights = load_current_weights(args.config_path)
    run_weights = load_run_weights(latest_run.parent) if latest_run is not None else {}
    tb_latest = load_tb_scalars(latest_run) if latest_run is not None else {}
    tb_prev = load_tb_scalars(prev_run) if prev_run is not None else {}
    suggestions = suggest_weight_updates(current_weights, tb_latest, tb_prev, issues)
    print_report(csv_metrics, issues, latest_run, run_weights, current_weights, tb_latest, tb_prev, suggestions)


if __name__ == "__main__":
    main()
