#!/bin/bash
################################################################################
# 3DGS Automated Training Script (Host Machine Version)
#
# Purpose: Train 3DGS models from NeRF format data and generate render videos
# Run from: Host machine (controls container via docker-compose)
#
# Usage:
#   ./train_nerf_scene.sh <scene_name> [options]
#   ./train_nerf_scene.sh lego
#   ./train_nerf_scene.sh chair --iterations 20000 --framerate 60
#
# More help: ./train_nerf_scene.sh --help
################################################################################

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default parameters
SCENE_NAME=""
DATA_DIR="/workspace/data/nerf_synthetic"
OUTPUT_DIR="/workspace/output"
ITERATIONS=30000
FRAMERATE=30
WHITE_BG=true
SKIP_TRAIN=false
SKIP_RENDER=false
SKIP_VIDEO=false

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "\n${BOLD}${GREEN}$1${NC}"
    echo -e "${GREEN}$(printf '=%.0s' {1..60})${NC}\n"
}

# Usage instructions
usage() {
    echo -e "${BOLD}3DGS Automated Training Script${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC}"
    echo "    $0 <scene_name> [options]"
    echo ""
    echo -e "${BOLD}Required arguments:${NC}"
    echo "    <scene_name>           NeRF scene name (e.g., lego, chair, drums)"
    echo ""
    echo -e "${BOLD}Optional arguments:${NC}"
    echo "    -i, --iterations <num>     Training iterations (default: 30000)"
    echo "    -f, --framerate <num>      Video framerate (default: 30)"
    echo "    --no-white-bg              Disable white background"
    echo "    --skip-train               Skip training step"
    echo "    --skip-render              Skip rendering step"
    echo "    --skip-video               Skip video generation step"
    echo "    -h, --help                 Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "    # Basic usage"
    echo "    $0 lego"
    echo ""
    echo "    # Custom iterations and framerate"
    echo "    $0 chair --iterations 20000 --framerate 60"
    echo ""
    echo "    # Train only, skip rendering"
    echo "    $0 drums --skip-render"
    echo ""
    echo "    # Render only from existing model"
    echo "    $0 lego --skip-train"
    echo ""
    echo "    # Batch process multiple scenes"
    echo "    for scene in lego chair drums; do"
    echo "        $0 \$scene"
    echo "    done"
    echo ""
    echo -e "${BOLD}Output files:${NC}"
    echo "    output/<scene_name>_trained/point_cloud/iteration_<N>/point_cloud.ply"
    echo "    output/<scene_name>_trained/test/ours_<N>/renders/*.png"
    echo "    output/<scene_name>_render.mp4"
    echo ""
    exit 0
}

# Parse command line arguments
parse_args() {
    if [ $# -eq 0 ]; then
        print_error "Error: Missing scene name argument"
        echo ""
        usage
    fi

    # First argument is scene name (unless --help)
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        usage
    fi

    SCENE_NAME="$1"
    shift

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            -f|--framerate)
                FRAMERATE="$2"
                shift 2
                ;;
            --no-white-bg)
                WHITE_BG=false
                shift
                ;;
            --skip-train)
                SKIP_TRAIN=true
                shift
                ;;
            --skip-render)
                SKIP_RENDER=true
                shift
                ;;
            --skip-video)
                SKIP_VIDEO=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                usage
                ;;
        esac
    done
}

# Check Docker container status
check_container() {
    print_info "Checking Docker container status..."

    if ! docker-compose ps | grep -q nvidia-track4; then
        print_error "Docker container not found"
        print_info "Please start container: docker-compose up -d"
        exit 1
    fi

    if ! docker-compose ps | grep nvidia-track4 | grep -q Up; then
        print_error "Docker container not running"
        print_info "Please start container: docker-compose up -d"
        exit 1
    fi

    print_success "Docker container is running"

    # Check GPU access
    if docker-compose exec -T nvidia-track4 nvidia-smi > /dev/null 2>&1; then
        print_success "GPU access OK"
    else
        print_warning "Warning: GPU may not be available, check Docker GPU configuration"
        print_info "Reference: GPU_SETUP.md"
    fi
}

# Check if scene data exists
check_scene_data() {
    print_info "Checking scene data: $SCENE_NAME"

    local scene_path="$DATA_DIR/$SCENE_NAME"

    # Check inside container
    if ! docker-compose exec -T nvidia-track4 bash -c "[ -d '$scene_path' ]"; then
        print_error "Scene directory not found: $scene_path"
        print_info "Available scenes:"
        docker-compose exec -T nvidia-track4 bash -c "ls -1 $DATA_DIR 2>/dev/null || echo '  (no scenes available)'"
        exit 1
    fi

    # Check required files
    if ! docker-compose exec -T nvidia-track4 bash -c "[ -f '$scene_path/transforms_train.json' ]"; then
        print_error "Missing training config file: transforms_train.json"
        exit 1
    fi

    if ! docker-compose exec -T nvidia-track4 bash -c "[ -f '$scene_path/transforms_test.json' ]"; then
        print_error "Missing test config file: transforms_test.json"
        exit 1
    fi

    print_success "Scene data validation passed"
}

# Train 3DGS model
train_model() {
    if [ "$SKIP_TRAIN" = true ]; then
        print_warning "Skipping training step"
        return 0
    fi

    print_header "[1/3] Training 3DGS Model"

    local scene_path="$DATA_DIR/$SCENE_NAME"
    local model_path="$OUTPUT_DIR/${SCENE_NAME}_trained"

    print_info "Scene: $SCENE_NAME"
    print_info "Iterations: $ITERATIONS"
    print_info "Model output: $model_path"
    echo ""

    # Build training command
    local train_cmd="cd /workspace/gaussian-splatting && python train.py \
        -s '$scene_path' \
        -m '$model_path' \
        --eval \
        --iterations $ITERATIONS \
        --save_iterations 7000 $ITERATIONS \
        --test_iterations 7000 $ITERATIONS"

    if [ "$WHITE_BG" = true ]; then
        train_cmd="$train_cmd --white_background"
    fi

    # Execute training
    if docker-compose exec nvidia-track4 bash -c "$train_cmd"; then
        echo ""
        print_success "Training completed"

        # Show generated model file
        local ply_path="$model_path/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
        if docker-compose exec -T nvidia-track4 bash -c "[ -f '$ply_path' ]"; then
            local size=$(docker-compose exec -T nvidia-track4 bash -c "du -h '$ply_path' | cut -f1")
            print_info "Model file: $ply_path ($size)"
        fi
    else
        print_error "Training failed"
        exit 1
    fi
}

# Render test images
render_images() {
    if [ "$SKIP_RENDER" = true ]; then
        print_warning "Skipping rendering step"
        return 0
    fi

    print_header "[2/3] Rendering Test Images"

    local model_path="$OUTPUT_DIR/${SCENE_NAME}_trained"

    # Check if model exists
    if ! docker-compose exec -T nvidia-track4 bash -c "[ -d '$model_path' ]"; then
        print_error "Model directory not found: $model_path"
        print_info "Please train model first or remove --skip-train option"
        exit 1
    fi

    print_info "Rendering model: $model_path"
    echo ""

    # Execute rendering
    local render_cmd="cd /workspace/gaussian-splatting && python render.py \
        -m '$model_path' \
        --skip_train"

    if docker-compose exec nvidia-track4 bash -c "$render_cmd"; then
        echo ""
        print_success "Rendering completed"

        # Count rendered images
        local renders_pattern="$model_path/test/ours_*/renders/*.png"
        local count=$(docker-compose exec -T nvidia-track4 bash -c "ls $renders_pattern 2>/dev/null | wc -l")
        print_info "Generated images: $count"
    else
        print_error "Rendering failed"
        exit 1
    fi
}

# Create video
create_video() {
    if [ "$SKIP_VIDEO" = true ]; then
        print_warning "Skipping video generation step"
        return 0
    fi

    print_header "[3/3] Creating Video"

    local model_path="$OUTPUT_DIR/${SCENE_NAME}_trained"
    local video_output="$OUTPUT_DIR/${SCENE_NAME}_render.mp4"

    print_info "Framerate: ${FRAMERATE}fps"
    print_info "Output: $video_output"
    echo ""

    # Find renders directory
    local renders_dir=$(docker-compose exec -T nvidia-track4 bash -c "find '$model_path/test' -type d -name 'renders' 2>/dev/null | head -1" | tr -d '\r')

    if [ -z "$renders_dir" ]; then
        print_error "Renders directory not found"
        exit 1
    fi

    # Check if images exist
    local img_count=$(docker-compose exec -T nvidia-track4 bash -c "ls '$renders_dir'/*.png 2>/dev/null | wc -l" | tr -d '\r')
    if [ "$img_count" -eq 0 ]; then
        print_error "No rendered images found"
        exit 1
    fi

    print_info "Found $img_count images"

    # Execute ffmpeg
    local ffmpeg_cmd="ffmpeg -y \
        -framerate $FRAMERATE \
        -pattern_type glob \
        -i '$renders_dir/*.png' \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 23 \
        '$video_output' 2>&1"

    if docker-compose exec nvidia-track4 bash -c "$ffmpeg_cmd" | grep -q "muxing overhead"; then
        echo ""
        print_success "Video generated successfully"

        # Show video size
        local size=$(docker-compose exec -T nvidia-track4 bash -c "du -h '$video_output' 2>/dev/null | cut -f1" | tr -d '\r')
        print_info "Video file: $video_output ($size)"
    else
        print_error "Video generation failed"
        exit 1
    fi
}

# Show final summary
show_summary() {
    print_header "Summary"

    echo -e "${BOLD}Scene:${NC} $SCENE_NAME"
    echo -e "${BOLD}Output directory:${NC} $OUTPUT_DIR"
    echo ""
    echo -e "${BOLD}Generated files:${NC}"

    local model_path="$OUTPUT_DIR/${SCENE_NAME}_trained"
    local ply_file="$model_path/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
    local renders_dir="$model_path/test/ours_$ITERATIONS/renders"
    local video_file="$OUTPUT_DIR/${SCENE_NAME}_render.mp4"

    # Check and display files
    if docker-compose exec -T nvidia-track4 bash -c "[ -f '$ply_file' ]"; then
        local size=$(docker-compose exec -T nvidia-track4 bash -c "du -h '$ply_file' | cut -f1" | tr -d '\r')
        echo -e "  ${GREEN}✓${NC} 3DGS model: $ply_file ($size)"
    fi

    if docker-compose exec -T nvidia-track4 bash -c "[ -d '$renders_dir' ]"; then
        local count=$(docker-compose exec -T nvidia-track4 bash -c "ls '$renders_dir'/*.png 2>/dev/null | wc -l" | tr -d '\r')
        echo -e "  ${GREEN}✓${NC} Rendered images: $renders_dir/ ($count images)"
    fi

    if docker-compose exec -T nvidia-track4 bash -c "[ -f '$video_file' ]"; then
        local size=$(docker-compose exec -T nvidia-track4 bash -c "du -h '$video_file' | cut -f1" | tr -d '\r')
        echo -e "  ${GREEN}✓${NC} Demo video: $video_file ($size)"
    fi

    echo ""
    print_success "All tasks completed!"
}

# Main function
main() {
    # Parse arguments
    parse_args "$@"

    # Print header
    print_header "3DGS Automated Training and Rendering"
    echo -e "${BOLD}Scene name:${NC} $SCENE_NAME"
    echo -e "${BOLD}Iterations:${NC} $ITERATIONS"
    echo -e "${BOLD}Video framerate:${NC} ${FRAMERATE}fps"
    echo -e "${BOLD}White background:${NC} $WHITE_BG"
    echo ""

    # Execute workflow
    check_container
    check_scene_data

    echo ""
    train_model

    echo ""
    render_images

    echo ""
    create_video

    echo ""
    show_summary
}

# Execute main function
main "$@"
