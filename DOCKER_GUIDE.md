# Docker Setup Guide for BlendED-NVIDIA-Track4

This guide will walk you through setting up and using Docker for the BlendED-NVIDIA-Track4 project. Docker provides a consistent development environment across different machines, eliminating the need to manually install CUDA, PyTorch, and other dependencies.

## Prerequisites

Before starting, ensure you have the following installed on your Windows machine:

1. **Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop/
   - During installation, make sure to enable WSL2 integration
   - Restart your computer after installation

2. **NVIDIA GPU Drivers**
   - Ensure you have the latest NVIDIA drivers installed
   - Docker will use these drivers to access your GPU

3. **Git** (if not already installed)
   - Download from: https://git-scm.com/download/win

## Initial Setup (One-time only)

### Step 1: Clone the Repository

Open PowerShell or Command Prompt and navigate to your desired directory, then clone the repository:

```powershell
git clone <repository-url>
cd BlendED-NVIDIA-Track4
```

### Step 2: Build the Docker Image

This step creates a Docker image with all necessary dependencies. **This only needs to be done once** (or when Dockerfile changes):

```powershell
docker-compose build
```

**Note:** This process may take 30-60 minutes depending on your internet connection, as it downloads and installs:
- Ubuntu 22.04 base image
- CUDA 11.8 toolkit
- PyTorch 2.0.0 with CUDA support
- COLMAP, Ceres-solver, and other dependencies
- Python packages from requirements.txt
- Gaussian Splatting submodules

**Troubleshooting:** If you encounter build errors:
- Ensure Docker Desktop is running
- Check that you have sufficient disk space (at least 10GB free)

## Daily Usage

### Starting Your Development Environment

1. **Navigate to the project directory:**
   ```powershell
   cd E:\path\to\BlendED-NVIDIA-Track4
   ```

2. **Start the Docker container:**
   ```powershell
   docker-compose up -d
   ```
   This command starts the container in the background (`-d` flag). It should complete in a few seconds.

3. **Access the container:**
   ```powershell
   docker-compose exec nvidia-track4 /bin/bash
   ```
   You should now see a prompt like `user@<container_id>:/home/user$`, indicating you're inside the Ubuntu environment.

4. **Navigate to your project files:**
   ```bash
   cd /workspace
   ls -la
   ```
   You should see all your project files here. Any changes you make in Windows will be reflected here immediately.

### Working Inside the Container

The container provides a complete Ubuntu 22.04 environment with:
- Python 3.10
- CUDA 11.8
- PyTorch 2.0.0 with CUDA support
- All project dependencies pre-installed

You can now run any Python scripts or commands as needed:

```bash
# Example: Run a Python script
python gs_simulation.py

# Example: Install additional packages (if needed)
pip install some-package

# Example: Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Stopping Your Development Environment

1. **Exit the container:**
   ```bash
   exit
   ```
   This returns you to your Windows PowerShell.

2. **Stop the container:**
   ```powershell
   docker-compose down
   ```
   This stops and removes the container, freeing up system resources.

## Restarting After Stopping

If you've stopped the container and want to resume work:

1. **Navigate to project directory:**
   ```powershell
   cd E:\path\to\BlendED-NVIDIA-Track4
   ```

2. **Start container:**
   ```powershell
   docker-compose up -d
   ```

3. **Access container:**
   ```powershell
   docker-compose exec nvidia-track4 /bin/bash
   ```

4. **Navigate to workspace:**
   ```bash
   cd /workspace
   ```

## Useful Docker Commands

### Check Container Status
```powershell
docker-compose ps
```

### View Container Logs
```powershell
docker-compose logs
```

### Remove Everything (if you need to start fresh)
```powershell
docker-compose down
docker-compose build --no-cache
```

### Access Container Without Starting New One
```powershell
docker exec -it nvidia_track4_container /bin/bash
```

## File Synchronization

- **Windows → Container**: Files are automatically synchronized
- **Container → Windows**: Changes made inside the container are immediately visible in Windows
- **No manual copying needed**: The `/workspace` directory in the container is directly linked to your Windows project folder

## Troubleshooting

### Container Won't Start
```powershell
# Check if Docker Desktop is running
# Restart Docker Desktop if needed
docker-compose down
docker-compose up -d
```

### Permission Issues
```powershell
# If you encounter permission errors, try:
docker-compose down
docker-compose up -d --force-recreate
```

### Out of Disk Space
```powershell
# Clean up unused Docker resources:
docker system prune -a
```

### GPU Not Detected
```powershell
# Check if NVIDIA Container Toolkit is installed
# Restart Docker Desktop
# Verify NVIDIA drivers are up to date
```

## Project Structure

```
BlendED-NVIDIA-Track4/
├── Dockerfile              # Docker image configuration
├── docker-compose.yaml    # Container orchestration
├── requirements.txt       # Python dependencies
├── DOCKER_GUIDE.md       # This guide
└── ... (other project files)
```

## Need Help?

If you encounter issues not covered in this guide:

1. Check Docker Desktop is running
2. Ensure NVIDIA drivers are up to date
3. Try restarting Docker Desktop
4. Contact the team for assistance

---

**Remember:** The initial `docker-compose build` takes time, but subsequent `docker-compose up -d` commands are very fast. This setup ensures everyone on the team has an identical development environment.
