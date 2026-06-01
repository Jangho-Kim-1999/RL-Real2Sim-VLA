# Vision RL Command Cheat Sheet

## 1. Low-Level Policy Keyboard Test

Test the frozen low-level velocity policy (`model_4000.pt`) directly with keyboard commands.

```bash
python VR-Robo-main/vrrobo_isaaclab/scripts/keyboard_random_objects.py \
  --task mclquad_seminar_camera_play \
  --num_envs 1

## 2. High-Level Object Pushing Policy Train

python Object_Pushing/scripts/train.py \
  --task mclquad_object_pushing \
  --num_envs 300

## 3. High-Level Object Pushing Policy Play

python Object_Pushing/scripts/play.py \
  --task mclquad_object_pushing_play \
  --num_envs 1 \
  --load_run 2026-05-18_12-14-12 \
  --checkpoint model_100.pt \
  --max_steps 300 \
  --log_dir Object_Pushing/play_logs/eval \
  --log_mat

## 4. NaVILA VLA High-Level Policy
python VR-Robo-main/vrrobo_isaaclab/scripts/rsl_rl/play_gs.py \
  --task mclquad_seminar_camera_play \
  --num_envs 1 \
  --vla \
  --vla_repo NaVILA-main \
  --vla_model_path a8cheng/navila-llama3-8b-8f \
  --vla_instruction "Navigate to the blue target." \
  --max_steps 300
