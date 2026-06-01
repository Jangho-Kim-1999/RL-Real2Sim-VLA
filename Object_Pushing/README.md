# Object Pushing

High-level whole-body position-only pushing task for `MCLQuad` in Isaac Lab.

The task registers:

- `mclquad_object_pushing`
- `mclquad_object_pushing_play`

The high-level policy outputs `(v_x, v_y, w_yaw)` at 5 Hz. These commands are passed to the existing frozen low-level velocity policy:

```text
VR-Robo-main/vrrobo_isaaclab/low_level_policy/model_4000.pt
```

Train:

```bash
python Object_Pushing/scripts/train.py --task mclquad_object_pushing --num_envs 300
```

Play:

```bash
python Object_Pushing/scripts/play.py --task mclquad_object_pushing_play --checkpoint model_1000.pt
```

Run a smoke test with no high-level checkpoint:

```bash
python Object_Pushing/scripts/play.py --task mclquad_object_pushing_play --no_checkpoint --max_steps 100 --log_dir Object_Pushing/play_logs/smoke
```

Play with a trained checkpoint and write CSV plus a MATLAB plot script:

```bash
python Object_Pushing/scripts/play.py --task mclquad_object_pushing_play --load_run <run_folder> --checkpoint <checkpoint.pt> --max_steps 300 --log_dir Object_Pushing/play_logs/eval
```

Also export a `.mat` file when `scipy` is available:

```bash
python Object_Pushing/scripts/play.py --task mclquad_object_pushing_play --load_run <run_folder> --checkpoint <checkpoint.pt> --max_steps 300 --log_dir Object_Pushing/play_logs/eval --log_mat
```

In MATLAB:

```matlab
run('Object_Pushing/play_logs/eval/plot_object_pushing_play_log.m')
```

Collect NaVILA-style VLA fine-tuning data from the trained pushing expert:

```bash
python3 Object_Pushing/scripts/collect_vla_push_data.py \
  --task mclquad_object_pushing_r7_camera_play \
  --load_run <run_folder> \
  --checkpoint <checkpoint.pt> \
  --episodes 5000 \
  --save_dir Object_Pushing/vla_push_dataset
```

The collector saves successful episodes only, including PNG frames, per-episode
metadata, `samples_index.json`, and a NaVILA `vlnce`-style `annotations.json`.
The local NaVILA checkout registers this dataset as `mclquad_push`, so it can be
blended with existing navigation data through `--data_mixture r2r+mclquad_push`
after collection.
