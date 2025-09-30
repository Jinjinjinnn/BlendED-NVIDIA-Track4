import sys
import os
sys.path.append("gaussian-splatting")

import argparse
import json
import math

import torch
import taichi as ti

from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.render_utils import load_params_from_gs
from utils.transformation_utils import (
    generate_rotation_matrices,
    apply_rotations,
    transform2origin,
    shift2center111,
)
from utils.sph_gaussian_converter import SPHGaussianConverter
from SPH_Taichi.config_builder import SimConfig
from SPH_Taichi.particle_system import ParticleSystem
from utils.vis_utils import save_points_ply, save_bbox_obj, save_init_scene_obj, save_init_scene_ply


def load_checkpoint(model_path: str, sh_degree: int = 3, iteration: int = -1) -> GaussianModel:
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")
    gs = GaussianModel(sh_degree)
    gs.load_ply(checkpt_path)
    return gs


def build_domain_lines(domain_end):
    x_max, y_max, z_max = domain_end
    # 8 anchors
    anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
    anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    anchors[1] = ti.Vector([0.0, y_max, 0.0])
    anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    anchors[3] = ti.Vector([x_max, y_max, 0.0])
    anchors[4] = ti.Vector([0.0, 0.0, z_max])
    anchors[5] = ti.Vector([0.0, y_max, z_max])
    anchors[6] = ti.Vector([x_max, 0.0, z_max])
    anchors[7] = ti.Vector([x_max, y_max, z_max])

    indices = ti.field(int, shape=(2 * 12))
    order = [
        0, 1, 0, 2, 1, 3, 2, 3,  # bottom rect
        4, 5, 4, 6, 5, 7, 6, 7,  # top rect
        0, 4, 1, 5, 2, 6, 3, 7,  # vertical
    ]
    for i, v in enumerate(order):
        indices[i] = v
    return anchors, indices


def build_bbox_lines(bboxes):
    # Return a list of (anchors, indices) pairs for each bbox
    lines = []
    for bc in bboxes:
        center = bc["point"]
        size = bc["size"]
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners = [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
        ]
        anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
        for i in range(8):
            anchors[i] = ti.Vector(corners[i])
        indices = ti.field(int, shape=(2 * 12))
        order = [
            0, 1, 0, 2, 1, 3, 2, 3,  # bottom
            4, 5, 4, 6, 5, 7, 6, 7,  # top
            0, 4, 1, 5, 2, 6, 3, 7,  # vertical
        ]
        for i, v in enumerate(order):
            indices[i] = v
        lines.append((anchors, indices))
    return lines


def main():
    parser = argparse.ArgumentParser(description="Visualize SPH init (Gaussians -> particles) with Taichi GUI")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--method", type=int, default=None, help="Override simulationMethod (0:WCSPH,1:IISPH,2:DFSPH)")
    parser.add_argument("--no-gui", action="store_true", help="Save PLY/OBJ and exit (headless)")
    parser.add_argument("--out_dir", type=str, default="vis_init", help="Output directory when --no-gui is set")
    parser.add_argument("--particles-only", action="store_true", help="Save only particle points PLY without any boxes")
    args = parser.parse_args()

    ti.init(arch=ti.cuda, device_memory_GB=2.0)

    # Load config JSON
    with open(args.config, "r") as f:
        config_json = json.load(f)

    preprocessing = config_json.get("preprocessing_params", {})
    camera_params = config_json.get("camera_params", {})
    sph_params = config_json.get("sph_params", {})

    # Optionally override method
    if args.method is not None:
        sph_params["simulationMethod"] = args.method
        config_json["sph_params"] = sph_params

    # Load Gaussians
    gaussians = load_checkpoint(args.model_path)

    # Prepare pipeline and parameters from Gaussians
    class _Pipe:
        def __init__(self):
            self.compute_cov3D_python = True
            self.debug = False
    pipeline = _Pipe()
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # Opacity mask
    opacity_th = preprocessing.get("opacity_threshold", 0.02)
    mask = (init_opacity.squeeze(-1) >= opacity_th).cpu()
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_shs = init_shs[mask, :]

    # Transformations
    rot_mats = generate_rotation_matrices(
        torch.tensor(preprocessing.get("rotation_degree", [0.0])),
        preprocessing.get("rotation_axis", [0]),
    )
    rotated_pos = apply_rotations(init_pos, rot_mats)
    transformed_pos, scale_origin, original_mean_pos = transform2origin(
        rotated_pos, preprocessing.get("scale", 1.0)
    )
    transformed_pos = shift2center111(transformed_pos)

    # Build SPH data (without building a solver if headless)
    gs_num = transformed_pos.shape[0]
    sim_cfg = SimConfig(config_dict=config_json)

    # Diagnostics
    def _aabb(t):
        import numpy as _np
        tnp = t.detach().cpu().numpy()
        return _np.min(tnp, axis=0), _np.max(tnp, axis=0)
    mn, mx = _aabb(transformed_pos)
    print(f"Particles AABB after transform: min {mn}, max {mx}")

    if args.no_gui:
        os.makedirs(args.out_dir, exist_ok=True)
        domain_end = sph_params.get("domainEnd", [2.0, 2.0, 2.0])
        bboxes = [bc for bc in config_json.get("boundary_conditions", []) if bc.get("type") == "bounding_box"]
        if getattr(args, "particles_only", False):
            out_ply = os.path.join(args.out_dir, "particles_only.ply")
            save_points_ply(transformed_pos, out_ply, color=(153, 204, 255))
            print(f"Saved particles-only PLY to: {os.path.abspath(out_ply)}")
            return
        # Preferred: single PLY (points + edges disabled)
        out_ply = os.path.join(args.out_dir, "init_scene.ply")
        save_init_scene_ply(transformed_pos, domain_end, bboxes, out_ply, with_edges=False)
        # Optional: combined OBJ
        out_obj = os.path.join(args.out_dir, "init_scene.obj")
        save_init_scene_obj(transformed_pos, domain_end, bboxes, out_obj)
        print(f"Saved: {os.path.abspath(out_ply)}\nAlso wrote OBJ (optional): {os.path.abspath(out_obj)}")
        return

    # Build SPH system (GGUI on)
    ps = ParticleSystem(sim_cfg, GGUI=True, max_num_particles=gs_num)

    converter = SPHGaussianConverter()
    sph_data, _ = converter.gaussian_to_sph_particles(
        transformed_pos, init_cov, init_opacity, init_shs
    )
    ps.add_particles(object_id=1, **sph_data)

    # Prepare GUI
    window = ti.ui.Window("SPH Init Visualizer", (1280, 720), show_window=True, vsync=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # Camera defaults
    domain_end = sph_params.get("domainEnd", [2.0, 2.0, 2.0])
    cx, cy, cz = [d * 0.5 for d in domain_end]
    camera.position(cx + 3.0, cy + 2.0, cz + 3.0)
    camera.lookat(cx, cy, cz)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(60)

    # Build lines for domain and bboxes
    dom_anchors, dom_indices = build_domain_lines(domain_end)
    bboxes = [bc for bc in config_json.get("boundary_conditions", []) if bc.get("type") == "bounding_box"]
    bbox_lines = build_bbox_lines(bboxes)

    # Main loop
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light((5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0))

        # Update vis buffers from particle system
        ps.copy_to_vis_buffer(invisible_objects=[])
        scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)

        # Draw domain and bboxes
        scene.lines(dom_anchors, indices=dom_indices, color=(0.99, 0.68, 0.28), width=1.0)
        for anchors, indices in bbox_lines:
            scene.lines(anchors, indices=indices, color=(0.28, 0.68, 0.99), width=1.0)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
