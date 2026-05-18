## Installation

First install Isaac Sim v4.2 and Isaac Lab v1.3.0 following the [NVIDIA Isaac Sim documentation](https://isaac-sim.github.io/IsaacLab/v1.3.0/source/setup/installation/pip_installation.html). Please do not clone the Isaac Lab repository in `./vrrobo_isaaclab` directory. You can clone and install it in any directory you like. We recommend you to install it in `./VR-Robo` directory.

```shell
conda create -n vr-robo-isaaclab python=3.10
conda activate vr-robo-isaaclab
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 --extra-index-url https://pypi.nvidia.com
git clone git@github.com:isaac-sim/IsaacLab.git -b v1.3.0
sudo apt install cmake build-essential
```

In `IsaacLab/source/extensions/omni.isaac.lab_tasks/setup.py`, delete the following two lines to avoid the error of installing `rsl-rl`:
```python
"rsl-rl": ["rsl-rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]
```

Then run:
```shell
cd IsaacLab
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

Then install this repository the required packages. In `./vrrobo_isaaclab` directory, run the following commands:
```shell
python -m pip install -e exts/vrrobo_isaaclab
pip install -e exts/rsl_rl
pip install gymnasium==0.29.0 rpyc timm
```

## Playing
In the first run, it will take a bit long time to build the Isaac Lab extension.
We provide a pre-trained checkpoint for the Go2GS task in `./ckpt/2025-01-11_20-23-09`. You can create a directory `./logs/rsl_rl/unitree_go2_gs` under the `vrrobo_isaaclab` folder and copy the checkpoint to this directory.
```shell
vrrobo_isaaclab
├── logs
│   ├── rsl_rl
│   │   ├── unitree_go2_gs
│   │   │   └── 2025-01-11_20-23-09
│   │   │       ├── params
│   │   │       ├── model_7400.pt
```
And you need to modify `vrrobo_isaaclab/exts/vrrobo_isaaclab/vrrobo_isaaclab/tasks/vrrobo/config/go2/agents/rsl_rl_ppo_cfg.py` line 18 and line 19:
```python
resume = True
load_run = "2025-01-11_20-23-09"
```
Then you can run the following command to play the demo:
```shell
python scripts/rsl_rl/play_gs.py --task go2_gs_play
```

## Training
To train from scratch, you need to modify `vrrobo_isaaclab/exts/vrrobo_isaaclab/vrrobo_isaaclab/tasks/vrrobo/config/go2/agents/rsl_rl_ppo_cfg.py` line 18 and line 19:
```python
resume = False
load_run = ""
```
Then you can run the following command to train the policy:
```shell
python scripts/rsl_rl/train_gs.py --task go2_gs --headless
```

## Training with MCL Quad Serial

The MCL Quad serial task uses the robot USD and pretrained low-level checkpoints copied from
`DEEP-Parallel-Sim2Real-main`.

```shell
python scripts/rsl_rl/train_gs.py --task mclquad_gs --headless
```

For play/evaluation:

```shell
python scripts/rsl_rl/play_gs.py --task mclquad_gs_play
```

To train the MCL Quad high-level policy in `3D_files/3DGUT_USDZ/seminar_plane.usd`
with a front-mounted RGB camera:

```shell
python scripts/rsl_rl/train_gs.py --task mclquad_seminar_camera --headless
```

For play/evaluation:

```shell
python scripts/rsl_rl/play_gs.py --task mclquad_seminar_camera_play
```

To train the MCL Quad high-level policy on flat ground with a single red cone target
and the same front-mounted RGB camera:

```shell
python scripts/rsl_rl/train_gs.py --task mclquad_flat_cone_camera --headless
```

For play/evaluation:

```shell
python scripts/rsl_rl/play_gs.py --task mclquad_flat_cone_camera_play
```

## Playing MCL Quad with NaVILA VLA

The RSL-RL high-level training/play path is unchanged. To swap the high-level policy at play time, use `--vla`.
NaVILA receives the robot front RGB camera frames and returns a text navigation primitive. The adapter converts that
primitive into the same 3D high-level action interface used by the existing low-level velocity controller.

Install the NaVILA runtime into the Isaac Lab Python environment and download the default checkpoint:

```shell
cd vrrobo_isaaclab
NAVILA_REPO=../../NaVILA-main ./scripts/setup_navila_vla.sh
```

Run with NaVILA instead of the trained high-level checkpoint:

```shell
python scripts/rsl_rl/play_gs.py \
  --task mclquad_seminar_camera_play \
  --vla \
  --vla_model_path vla_weights/navila-llama3-8b-8f \
  --vla_load_4bit \
  --vla_target_names "red fire extinguisher,green target,blue block"
```

Use `--vla_instruction "Navigate to the blue block."` to bypass the environment RGB command and force one fixed
language instruction. Drop `--vla_load_4bit` only if the full 8B checkpoint and Isaac Sim fit in GPU memory.

For a finite diagnostic run that saves MATLAB logs, path plots, and one front-camera PNG per simulated second:

```shell
python scripts/rsl_rl/play_gs.py \
  --task mclquad_seminar_camera_play \
  --vla \
  --vla_model_path vla_weights/navila-llama3-8b-8f \
  --vla_load_4bit \
  --vla_target_names "red fire extinguisher,green target,blue block" \
  --log_mat \
  --save_camera_png \
  --camera_png_interval_s 1.0 \
  --max_time_s 60
```

The MATLAB plot script is written next to `play_log.mat` as `plot_play_log.m`.

The default frozen low-level policy is:

```text
low_level_policy/mclquad_flat_conventional_model_9999.pt
```

To switch to the dynamic-conversion checkpoint, edit `policy_dir` in:

```text
exts/vrrobo_isaaclab/vrrobo_isaaclab/tasks/vrrobo/config/mclquad/mclquad_env_cfg.py
```

## Trouble Shooting
If you can't stop the process, run:
```shell
killall -9 pt_main_thread
```
