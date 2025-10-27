## Gaussian SPH Fluid: Physics‑integrated 3D Gaussians for SPH Fluid Dynamics

<p align="center">
  <img src="resources/method_overview.png" alt="Method Overview">
</p>

This repository turns reconstructed 3D Gaussian splats directly into SPH fluid particles, simulates them using a Taichi‑based DFSPH solver, and renders the results with the same Gaussian representation. Colors are obtained from SH coefficients, and isotropic covariances are synthesized per frame for rasterization. In short: what you see is what you simulate.

- **Unified representation**: reconstruction, simulation, and rendering all share 3D Gaussians
- **DFSPH solver**: GPU‑accelerated density‑constraint SPH in Taichi
- **Single‑file configuration**: one JSON controls preprocessing, camera, timing, and SPH
- **Reproducible environment**: Docker image with CUDA/PyTorch/C++ deps preinstalled

---

## Project Structure

```
BlendED-NVIDIA-Track4/
├── _resources/             # figures (method overview)
├── config/                 # per‑scene JSON configs
├── demo/                   # demo images/videos
├── docs/                   # documentation (Docker guide, etc.)
│   └── DOCKER_GUIDE.md
├── gaussian-splatting/     # 3DGS code; includes submodules/
├── model/                  # sample GS models (optional)
├── output/                 # frames/videos/PLY exports
├── particle_filling/       # interior uniform filling utilities
├── scripts/                # helper scripts
├── SPH_Taichi/             # Taichi DFSPH solver (adapted)
├── utils/                  # parameter decode, transforms, camera, render
├── gs_simulation.py        # main entry (preprocess → SPH → render)
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

---

## Repository Clone

```bash
git clone https://github.com/Jinjinjinnn/BlendED-NVIDIA-Track4.git
cd BlendED-NVIDIA-Track4
# (optional) initialize submodules if needed by your git setup
git submodule update --init --recursive
```

---

## Setup (Docker)

On Windows with an NVIDIA GPU, you can be up and running quickly. See `docs/DOCKER_GUIDE.md` for full details and troubleshooting.

1) Build the image (first time only)
```bash
docker-compose build
```

2) Start the container
```bash
docker-compose up -d
```

3) Enter the container and move to the workspace
```bash
docker-compose exec nvidia-track4 /bin/bash
cd /workspace
```

> For logs, pruning, GPU checks, and more helper commands, see `docs/DOCKER_GUIDE.md`.

---

## Quick Start

1) Download a sample GS model (inside the container)
```bash
bash scripts/download_sample_model.sh
```

2) Run simulation + render first frame + optionally make a video
```bash
python gs_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --config ./config/ficus_sph_config.json \
  --render_img --compile_video --white_bg --output_ply
```

- Frames: `./output/<scene_name>/*.png`
- Video: `./output/<scene_name>/output.mp4`
- Initial exports: `init_surface_gaussians.ply`, `filled_particles_init.ply`

More options:
- `--output_path <dir>`: choose result directory root
- `--output_h5`: also export HDF5
- `--debug`: extra logs and intermediate dumps

<img src="./demo/ficus.gif" width="300"/>

---

## JSON Config

Each scene uses a single `.json` that specifies preprocessing, SPH simulation, camera, and timing/exports. See `config/ficus_sph_config.json` and `config/hotdog_sph_config.json` for concrete examples.

### Minimal example

```json
{
  "density0": 1000.0,
  "viscosity": 0.01,
  "surface_tension": 0.01,
  "particleRadius": 0.005,
  "gravitation": [0.0, -9.81, 0.0],
  "shift": true,
  "substep_dt": 0.0001,

  "opacity_threshold": 0.02,
  "rotation_degree": [-90.0],
  "rotation_axis": [0],
  "scale": 1.0,

  "sph_filling": {
    "enabled": true,
    "surface_keep_ratio": 0.4,
    "interior_keep_ratio": 1.0,
    "opacity_threshold": 0.4,
    "boundary": null,
    "k": 8,
    "sigma_scale": 1.0,
    "neighbor_radius_scale": 3.0,
    "iso_radius_factor": 1.0
  },

  "sph_space_vertical_upward_axis": [0, 1, 0],
  "sph_space_viewpoint_center": [0.0, 0.5, 0.0],
  "default_camera_index": -1,
  "move_camera": true,
  "init_azimuthm": -16.3,
  "init_elevation": 13.96,
  "init_radius": 5.11,

  "frame_dt": 0.0333,
  "frame_num": 300
}
```

### Parameter overview

- **SPH simulation**
  - **density0, viscosity, surface_tension, particleRadius**: fluid properties
  - **gravitation**: gravity vector
  - **shift**: center the particle cluster in the simulation domain
  - Optional: **domainStart**, **domainEnd** to fix the domain bounds

- **Timing/exports**
  - **substep_dt**: DFSPH integration substep
  - **frame_dt / frame_num**: frame interval and number of frames

- **Preprocessing**
  - **opacity_threshold**: remove low‑opacity Gaussians
  - **rotation_degree / rotation_axis, scale**: world alignment and scaling
  - Optional: **sim_area** (AABB), **render_uniform_color**, **render_sh_tint/gain/gamma**

- **Interior filling (SPH Filling)**
  - **enabled**, **surface_keep_ratio**, **interior_keep_ratio**
  - **opacity_threshold**, **boundary**, **k**, **sigma_scale**, **neighbor_radius_scale**, **iso_radius_factor**

- **Camera**
  - **sph_space_viewpoint_center**, **sph_space_vertical_upward_axis**
  - **default_camera_index**, **move_camera**, **init_***, **delta_a/e/r**, **show_hint**

---

## References

- T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang. PhysGaussian: Physics‑integrated 3D Gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4389–4398, 2024.

- erizmr. SPH Taichi: A high‑performance implementation of SPH in Taichi. GitHub repository, 2025. Available at: https://github.com/erizmr/SPH_Taichi.


