# 3DGS Training Automation

Automated tool for training 3D Gaussian Splatting models from NeRF format datasets and generating rendered videos.

## Features

- ✅ Runs from host machine with Docker integration
- ✅ One-command workflow: training → rendering → video generation
- ✅ Customizable parameters (iterations, framerate, etc.)
- ✅ Detailed progress output and error handling
- ✅ GPU status verification

---

## Quick Start

### 1. Start Docker Container

```bash
# Start container
docker-compose up -d

# Verify status
docker-compose ps
```

### 2. Run Training

```bash
# Make script executable (first time only)
chmod +x scripts/train_nerf_scene.sh

# Train the lego scene
./scripts/train_nerf_scene.sh lego
```

This single command will:
- Train 3DGS model (30,000 iterations, ~8-10 minutes)
- Render 200 test images (~15 seconds)
- Generate MP4 video (~2 seconds)

### 3. View Results

Generated files:
- `output/lego_trained/point_cloud/iteration_30000/point_cloud.ply` - 3DGS model (77MB)
- `output/lego_trained/test/ours_30000/renders/` - Rendered images (200 frames)
- `output/lego_render.mp4` - Demo video (1.8MB)

---

## Usage

### Basic Syntax

```bash
./scripts/train_nerf_scene.sh <scene_name> [options]
```

### Required Parameters

| Parameter | Description | Examples |
|-----------|-------------|----------|
| `<scene_name>` | NeRF scene name | `lego`, `chair`, `drums`, `hotdog` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i`, `--iterations <num>` | Training iterations | 30000 |
| `-f`, `--framerate <num>` | Video framerate (fps) | 30 |
| `--no-white-bg` | Disable white background | White background enabled |
| `--skip-train` | Skip training step | Don't skip |
| `--skip-render` | Skip rendering step | Don't skip |
| `--skip-video` | Skip video generation | Don't skip |
| `-h`, `--help` | Show help message | - |

---

## Examples

### Example 1: Basic Training

Train lego scene with default parameters:

```bash
./scripts/train_nerf_scene.sh lego
```

**Output:**
- 30,000 iterations (~10 minutes)
- 200 rendered images
- 30fps video

### Example 2: Custom Parameters

Fast training with high framerate video:

```bash
./scripts/train_nerf_scene.sh chair --iterations 10000 --framerate 60
```

### Example 3: Render Only

Render from existing trained model:

```bash
./scripts/train_nerf_scene.sh lego --skip-train
```

### Example 4: Training Only

Train model without rendering:

```bash
./scripts/train_nerf_scene.sh drums --skip-render
```

### Example 5: Batch Processing

Train multiple scenes:

```bash
for scene in lego chair drums hotdog; do
    echo "Training scene: $scene"
    ./scripts/train_nerf_scene.sh $scene
done
```

### Example 6: Black Background

For real-world scenes like fern:

```bash
./scripts/train_nerf_scene.sh fern --no-white-bg
```

---

## Output Structure

```
output/
├── lego_trained/                          # Training output directory
│   ├── point_cloud/
│   │   ├── iteration_7000/
│   │   │   └── point_cloud.ply           # Intermediate checkpoint (61MB)
│   │   └── iteration_30000/
│   │       └── point_cloud.ply           # Final model (77MB) ← For physics simulation
│   ├── test/
│   │   └── ours_30000/
│   │       ├── renders/                   # Rendered images
│   │       │   ├── 00000.png
│   │       │   ├── 00001.png
│   │       │   └── ... (200 images)
│   │       └── gt/                        # Ground truth comparison
│   ├── cameras.json                       # Camera parameters
│   └── input.ply                          # Initial point cloud
└── lego_render.mp4                        # Demo video (1.8MB)
```

---

## Available Scenes

### NeRF Synthetic Dataset

| Scene | Description | Background |
|-------|-------------|------------|
| `lego` | LEGO bulldozer | White |
| `chair` | Chair | White |
| `drums` | Drum set | White |
| `ficus` | Ficus plant | White |
| `hotdog` | Hotdog | White |
| `materials` | Material spheres | White |
| `mic` | Microphone | White |
| `ship` | Ship | White |

---

## Troubleshooting

### Issue 1: Container Not Running

**Error:**
```
✗ Docker容器未运行
ℹ 请先启动容器: docker-compose up -d
```

**Solution:**
```bash
docker-compose up -d
```

### Issue 2: Scene Data Not Found

**Error:**
```
✗ 场景目录不存在: /workspace/data/nerf_synthetic/xxx
```

**Solution:**
```bash
# List available scenes
ls data/nerf_synthetic/

# Or check inside container
docker-compose exec nvidia-track4 ls /workspace/data/nerf_synthetic/
```

### Issue 3: GPU Not Available

**Error:**
```
RuntimeError: No CUDA GPUs are available
```

**Solution:**
```bash
# Rebuild container with GPU support
docker-compose down
docker-compose build
docker-compose up -d

# Verify GPU access
docker-compose exec nvidia-track4 nvidia-smi
```

### Issue 4: Permission Denied

**Error:**
```
permission denied: ./scripts/train_nerf_scene.sh
```

**Solution:**
```bash
chmod +x scripts/train_nerf_scene.sh
```

### Issue 5: CUDA Out of Memory

**Error:**
```
CUDA out of memory
```

**Solution:**
```bash
# Reduce iterations
./scripts/train_nerf_scene.sh lego --iterations 15000
```

---

## Integration with PhysGaussian

The trained models can be directly used for physics simulation:

```bash
# 1. Train model
./scripts/train_nerf_scene.sh lego

# 2. Run physics simulation
python gs_simulation.py \
    --model_path output/lego_trained \
    --config config/lego_sph_config.json \
    --render_img --compile_video
```

---

## Advanced Usage

### Manual Training with Custom Parameters

For fine-grained control, run training commands directly:

```bash
docker-compose exec nvidia-track4 bash -c "
    cd /workspace/gaussian-splatting && \
    python train.py \
        -s /workspace/data/nerf_synthetic/lego \
        -m /workspace/output/lego_custom \
        --iterations 50000 \
        --densify_grad_threshold 0.0002 \
        --white_background \
        --eval
"
```

### Custom Rendering

```bash
docker-compose exec nvidia-track4 bash -c "
    cd /workspace/gaussian-splatting && \
    python render.py \
        -m /workspace/output/lego_trained \
        --skip_train
"
```

---

## Expected Training Times

*Based on NVIDIA RTX 3090 GPU*

| Scene | Iterations | Training | Rendering | Video |
|-------|------------|----------|-----------|-------|
| lego | 30,000 | ~8-10 min | ~15 sec | ~2 sec |
| chair | 30,000 | ~8-10 min | ~15 sec | ~2 sec |
| drums | 30,000 | ~8-10 min | ~15 sec | ~2 sec |

**Note:** Training time is displayed dynamically via tqdm progress bar with real-time ETA.

---

## Quick Command Reference

| Command | Description |
|---------|-------------|
| `./scripts/train_nerf_scene.sh lego` | Full training pipeline |
| `./scripts/train_nerf_scene.sh lego --skip-train` | Render only |
| `./scripts/train_nerf_scene.sh lego -i 10000` | Fast training |
| `./scripts/train_nerf_scene.sh lego -f 60` | 60fps video |
| `./scripts/train_nerf_scene.sh --help` | Show help |

---

## What This Tool Does

This automation tool completes the following workflow:

1. ✅ **Dataset verification** - NeRF Synthetic format with `transforms_train.json` and `transforms_test.json`
2. ✅ **Model training** - 3D Gaussian Splatting training via Docker
3. ✅ **Image rendering** - Generate test view images
4. ✅ **Video creation** - Compile images into MP4 video using ffmpeg

**Key outputs:**
- 📦 `point_cloud.ply` (77MB) - For physics simulation
- 🎬 `render.mp4` (1.8MB) - Demo visualization

---

## License

Based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project.
