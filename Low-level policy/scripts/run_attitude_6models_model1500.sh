#!/usr/bin/env bash
set -euo pipefail

# Run 6 attitude ablation models with identical play command conditions and
# save per-model time-series CSV logs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME="MCLrobotics_MCLQuadserial_attitude"
BASE_DIR="logs/rsl_rl/${EXP_NAME}"
OUT_DIR="${BASE_DIR}/comparison_play_model1500_matched_ablation_amp0p1_freq1p2_wait2_36cycles_35s"
TASK="Attitude-MCLQuad-serial"
PITCH_AMP="0.1"
PITCH_FREQ="1.2"
INITIAL_WAIT_S="2.0"
COMMAND_CYCLES="36.0"
COMMAND_DURATION_S="$(awk -v cycles="${COMMAND_CYCLES}" -v freq="${PITCH_FREQ}" 'BEGIN {printf "%.10g", cycles / freq}')"
PLAY_DURATION_S="35.0"
MODEL_CKPT="model_1500.pt"
PYTHON_BIN="${PYTHON_BIN:-/home/teamquad/anaconda3/envs/isaaclab/bin/python}"

MODELS=(
  "pitch_Dyn_NoRand"
  "pitch_Dyn_Rand"
  "pitch_Kinonly_NoRand"
  "pitch_Kinonly_Rand"
  "pitch_Prop_NoRand"
  "pitch_Prop_Rand"
)

mkdir -p "$OUT_DIR"

is_complete_csv() {
  local csv_path="$1"
  if [[ ! -f "$csv_path" ]]; then
    return 1
  fi
  local last_time
  last_time="$(awk -F, 'END {print $2}' "$csv_path")"
  awk -v t="$last_time" -v min_t="$PLAY_DURATION_S" 'BEGIN {exit !(t >= min_t - 0.02)}'
}

for model in "${MODELS[@]}"; do
  run_dir="${BASE_DIR}/${model}"
  if [[ ! -d "$run_dir" ]]; then
    echo "[WARN] Skip missing model dir: $run_dir"
    continue
  fi

  ckpt_path="${run_dir}/${MODEL_CKPT}"
  out_csv="${OUT_DIR}/${model}.csv"
  out_log="${OUT_DIR}/${model}.log"
  hip_scale="1.0"
  tau_mode="none"
  if [[ "$model" == pitch_Dyn_* ]]; then
    # Dynamic ablation: HIP armature/friction = 2Jm/2Bm, tau_comp = Jm/Bm terms.
    hip_scale="2.0"
    tau_mode="dyn"
  elif [[ "$model" == pitch_Prop_* ]]; then
    # Proposed ablation: HIP armature/friction = 2Jm/2Bm, tau_comp = Jm/Bm + Mlink terms.
    hip_scale="2.0"
    tau_mode="prop"
  fi
  rand_scope="base"
  if [[ "$model" == *_Rand ]]; then
    rand_scope="whole"
  fi

  echo "[INFO] Running model: ${model}"
  echo "[INFO]   checkpoint=${ckpt_path}"
  echo "[INFO]   command=wait ${INITIAL_WAIT_S}s, then pitch ${PITCH_AMP}*sin(2*pi*${PITCH_FREQ}*t_active) for ${COMMAND_DURATION_S}s (${COMMAND_CYCLES} cycles)"
  echo "[INFO]   play duration=${PLAY_DURATION_S}s"
  echo "[INFO]   HIP actuator scale=${hip_scale} (relative to Jm/Bm)"
  echo "[INFO]   tau_compensation_mode=${tau_mode}"
  echo "[INFO]   play_rand_scope=${rand_scope}"
  echo "[INFO]   csv=${out_csv}"

  if is_complete_csv "$out_csv"; then
    echo "[INFO]   skip: existing CSV already reaches ${PLAY_DURATION_S}s"
    continue
  fi

  "${PYTHON_BIN}" scripts/play.py \
    --task "${TASK}" \
    --checkpoint "${ckpt_path}" \
    --num_envs 1 \
    --attitude-pitch-amp "${PITCH_AMP}" \
    --attitude-pitch-freq "${PITCH_FREQ}" \
    --traj-initial-stop "${INITIAL_WAIT_S}" \
    --attitude-command-duration-s "${COMMAND_DURATION_S}" \
    --play-duration-s "${PLAY_DURATION_S}" \
    --hip-armature-scale "${hip_scale}" \
    --hip-viscous-friction-scale "${hip_scale}" \
    --tau-compensation-mode "${tau_mode}" \
    --play-rand-scope "${rand_scope}" \
    --log-play-signals-path "${out_csv}" \
    --headless > "${out_log}" 2>&1
done

echo "[INFO] Done. Logs saved under: ${OUT_DIR}"
