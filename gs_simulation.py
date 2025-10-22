import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# SPH dependencies
from SPH_Taichi.sph_engine_utils import *

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
ti_arch = ti.cuda if torch.cuda.is_available() else ti.cpu
mem_frac = float(os.environ.get("TI_MEM_FRAC", "0.4"))

try:
    ti.init(arch=ti_arch, device_memory_fraction=mem_frac)
except Exception as e:
    print(f"[Taichi] init failed with fraction={mem_frac}: {e}")
    ti.init(arch=ti_arch, device_memory_fraction=0.3)

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # scene/output/log directories
    scene_name = os.path.basename(args.config)
    if scene_name.endswith("_config.json"):
        scene_name = scene_name[: -len("_config.json")]
    else:
        scene_name = os.path.splitext(scene_name)[0]

    base_output_dir = args.output_path if args.output_path else os.path.join("./output", scene_name)
    os.makedirs(base_output_dir, exist_ok=True)

    log_dir = os.path.join("./log", scene_name)
    os.makedirs(log_dir, exist_ok=True)

    # load scene config
    print("Loading scene config...")
    (
        cfg,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_all_params_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # rorate and translate object
    if args.debug:
        particle_position_tensor_to_ply(init_pos, os.path.join(log_dir, "init_particles.ply"))
    
    deg_list = preprocessing_params.get("rotation_degree", [])
    axis_list = preprocessing_params.get("rotation_axis", [])
    scale_val = preprocessing_params.get("scale", 1.0)
    sim_area = preprocessing_params.get("sim_area", None)

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(deg_list, device="cuda" if torch.cuda.is_available() else "cpu"),
        axis_list,
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, os.path.join(log_dir, "rotated_particles.ply"))

    # select a sim area and save params of unselected particles
    unselected_pos = unselected_cov = unselected_opacity = unselected_shs = None
    if sim_area is not None:
        assert len(sim_area) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > sim_area[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < sim_area[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, scale_val)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            os.path.join(log_dir, "transformed_particles.ply"),
        )

    # fill particles if needed
    gs_num_surface = transformed_pos.shape[0]
    print(f"[SPH Filling] surface gaussians: {gs_num_surface}")
    device = "cuda:0"
    sph_fill_cfg = preprocessing_params.get("sph_filling", None)

    if sph_fill_cfg is not None and sph_fill_cfg.get("enabled", False):
        seed_pos, shs, opacity, sph_init_cov = sph_uniform_fill_and_seed(
            surface_pos=transformed_pos.to(device=device),
            surface_shs=init_shs.to(device=device),
            surface_cov6=init_cov.to(device=device),
            surface_opacity=init_opacity.to(device=device),
            particle_radius=float(cfg.get_cfg("particleRadius")),
            opacity_threshold=float(sph_fill_cfg.get("opacity_threshold", 0.15)),
            boundary=sph_fill_cfg.get("boundary", None),
            device=device,
            iso_radius_factor=float(sph_fill_cfg.get("iso_radius_factor", 1.0)),
            k=int(sph_fill_cfg.get("k", 8)),
            sigma_scale=float(sph_fill_cfg.get("sigma_scale", 1.0)),
            neighbor_radius_scale=float(sph_fill_cfg.get("neighbor_radius_scale", 3.0)),
        )
        sph_seed_pos = seed_pos
        print(f"[SPH Filling] total particles after filling: {sph_seed_pos.shape[0]}")

        sur_keep = float(sph_fill_cfg.get("surface_keep_ratio", 1.0))
        int_keep = float(sph_fill_cfg.get("interior_keep_ratio", 1.0))

        if sur_keep < 1.0 or int_keep < 1.0:
            Nsurf = int(gs_num_surface)
            Nall  = int(sph_seed_pos.shape[0])
            dev = sph_seed_pos.device
            mask = torch.cat([
                torch.rand(Nsurf, device=dev) < sur_keep,
                torch.rand(Nall - Nsurf, device=dev) < int_keep
            ], dim=0)

            sph_seed_pos = sph_seed_pos[mask]
            shs          = shs[mask]
            opacity      = opacity[mask]
            sph_init_cov = sph_init_cov[mask]
            print(f"[SPH] subsample: {int(mask.sum())} / {Nall}")
    else:
        sph_seed_pos = transformed_pos.to(device=device)
        shs = init_shs
        opacity = init_opacity
        sph_init_cov = torch.zeros((sph_seed_pos.shape[0], 6), device=device)
        sph_init_cov[:gs_num_surface] = init_cov.to(device=device)
        print(f"[SPH Filling] disabled. using surface only: {sph_seed_pos.shape[0]}")

    # shs = init_shs
    # opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    if args.output_ply:
        particle_position_tensor_to_ply(
            transformed_pos,
            os.path.join(base_output_dir, "init_surface_gaussians.ply"),
        )
        particle_position_tensor_to_ply(
            sph_seed_pos,
            os.path.join(base_output_dir, "filled_particles_init.ply"),
        )

    # ============ Initialize SPH (DFSPH) ============
    ctx = initialize_sph_from_positions(cfg, sph_seed_pos, margin_scale=2.0)
    # camera setting (Y-up default is handled in config; keep interface)
    sph_space_viewpoint_center = (
        torch.tensor(camera_params["sph_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    sph_space_vertical_upward_axis = (
        torch.tensor(camera_params["sph_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    # (
    #     viewpoint_center_worldspace,
    #     observant_coordinates,
    # ) = get_center_view_worldspace_and_observant_coordinate(
    #     sph_space_viewpoint_center,
    #     sph_space_vertical_upward_axis,
    #     rotation_matrices,
    #     scale_origin,
    #     original_mean_pos,
    # )

    sph_center_np = np.squeeze(sph_space_viewpoint_center.detach().cpu().numpy(), 0)
    sph_up_np = np.squeeze(sph_space_vertical_upward_axis.detach().cpu().numpy(), 0)
    vertical, h1, h2 = generate_local_coord(sph_up_np)
    viewpoint_center_worldspace = sph_center_np
    observant_coordinates = np.column_stack((h1, h2, vertical))

    cam = camera_params
    current_camera = get_camera_view(
        model_path,
        default_camera_index=cam.get("default_camera_index", 0),
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        show_hint=cam.get("show_hint", False),
        init_azimuthm=cam.get("init_azimuthm", None),
        init_elevation=cam.get("init_elevation", None),
        init_radius=cam.get("init_radius", None),
        move_camera=False,
        current_frame=0,
        delta_a=cam.get("delta_a", None),
        delta_e=cam.get("delta_e", None),
        delta_r=cam.get("delta_r", None),
    )

    pos0 = export_positions_torch(ctx, device="cuda") - ctx["shift"]

    shs0 = shs  # filled 이후 shs_all
    colors0 = convert_SH(shs0, current_camera, gaussians, pos0, None)
    tint = preprocessing_params.get("render_sh_tint", None)
    if tint is not None:
        tint_t = torch.tensor(tint, device=pos0.device, dtype=torch.float32).view(1, 3)
        colors0 = colors0 * tint_t
    gain = float(preprocessing_params.get("render_sh_gain", 1.0))
    gamma = float(preprocessing_params.get("render_sh_gamma", 1.0))
    if gain != 1.0:
        colors0 = colors0 * gain
    if gamma != 1.0:
        colors0 = torch.clamp(colors0, 1e-6, 1.0) ** (1.0 / max(gamma, 1e-6))
    colors0 = colors0.clamp(0.0, 1.0)

    if args.output_ply or args.output_h5:
        directory_to_save = base_output_dir
        os.makedirs(directory_to_save, exist_ok=True)
        save_data_at_frame_sph(
            ctx,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
            time_value=0.0,
            colors_rgb=colors0,
            pos_override=pos0,
        )

    # timings
    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)

    # cache for rendering
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None

    density_err_hist = []
    t = tqdm(range(frame_num))
    for frame in t:
        cam = camera_params
        current_camera = get_camera_view(
            model_path,
            default_camera_index=cam.get("default_camera_index", 0),
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=cam.get("show_hint", False),
            init_azimuthm=cam.get("init_azimuthm", None),
            init_elevation=cam.get("init_elevation", None),
            init_radius=cam.get("init_radius", None),
            move_camera=cam.get("move_camera", False),
            current_frame=frame,
            delta_a=cam.get("delta_a", None),
            delta_e=cam.get("delta_e", None),
            delta_r=cam.get("delta_r", None),
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        if args.render_img and frame == 0:
            pos = export_positions_torch(ctx, device="cuda")
            pos = pos - ctx["shift"]

            shs = shs_render
            opacity = opacity_render

            if preprocessing_params.get("sim_area") is not None:
                pos     = torch.cat([pos,            unselected_pos.to(pos.device)], dim=0)
                shs     = torch.cat([shs,            unselected_shs.to(shs.device)], dim=0)
                opacity = torch.cat([opacity,        unselected_opacity.to(opacity.device)], dim=0)

            n = pos.shape[0]
            if shs.shape[0] != n or opacity.shape[0] != n:
                m = min(n, shs.shape[0], opacity.shape[0])
                pos, shs, opacity = pos[:m], shs[:m], opacity[:m]
                n = m

            # isotropic covariance per current positions
            iso_factor = float(preprocessing_params.get("sph_filling", {}).get("iso_radius_factor", 1.0))
            cov3D = make_isotropic_cov(n, float(cfg.get_cfg("particleRadius")), iso_factor, pos.device, torch.float32)

            uniform = preprocessing_params.get("render_uniform_color", None)
            if uniform is not None:
                colors_precomp = torch.tensor(uniform, device=pos.device, dtype=torch.float32).view(1,3).repeat(pos.shape[0],1)
                shs_arg = None
            else:
                colors_precomp = None
                shs_arg = None  # SH는 convert_SH로 처리
                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, None)
                tint = preprocessing_params.get("render_sh_tint", None)
                if tint is not None:
                    tint_t = torch.tensor(tint, device=pos.device, dtype=torch.float32).view(1,3)
                    colors_precomp = (colors_precomp * tint_t)

                gain = float(preprocessing_params.get("render_sh_gain", 1.0))
                gamma = float(preprocessing_params.get("render_sh_gamma", 1.0))
                if gain != 1.0:
                    colors_precomp = colors_precomp * gain
                if gamma != 1.0:
                    colors_precomp = torch.clamp(colors_precomp, 1e-6, 1.0) ** (1.0 / max(gamma, 1e-6))
                colors_precomp = colors_precomp.clamp(0.0, 1.0)

            rendering, raddi = rasterize(
                means3D=pos, means2D=torch.zeros_like(pos),
                shs=None, colors_precomp=colors_precomp,
                opacities=opacity, scales=None, rotations=None, cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            cv2.imwrite(os.path.join(base_output_dir, f"{0:04d}.png"), 255 * cv2_img)

        for step in range(step_per_frame):
            step_sph(ctx, 1)

        # DFSPH final density error reading + tqdm display
        de = float(getattr(ctx["solver"], "last_avg_density_err", 0.0))
        density_err_hist.append(de)
        t.set_postfix({'dens_err': f'{de:.5e}'})

        if args.output_ply or args.output_h5:
            save_data_at_frame_sph(
                ctx,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
                time_value=(frame + 1) * frame_dt,
                density_error=de,
            )

        if args.render_img:
            # export current positions (simulation space), take only original gaussians
            pos = export_positions_torch(ctx, device="cuda")
            # undo domain-centering shift before world-space inverse transforms
            pos = pos - ctx["shift"]

            shs = shs_render
            opacity = opacity_render

            if preprocessing_params.get("sim_area") is not None:
                pos     = torch.cat([pos,            unselected_pos.to(pos.device)], dim=0)
                shs     = torch.cat([shs,            unselected_shs.to(shs.device)], dim=0)
                opacity = torch.cat([opacity,        unselected_opacity.to(opacity.device)], dim=0)

            n = pos.shape[0]
            if shs.shape[0] != n or opacity.shape[0] != n:
                print(f"[Render] align sizes pos={n}, shs={shs.shape[0]}, opa={opacity.shape[0]}")
                m = min(n, shs.shape[0], opacity.shape[0])
                pos, shs, opacity = pos[:m], shs[:m], opacity[:m]
                n = m

            # isotropic covariance per current positions
            iso_factor = float(preprocessing_params.get("sph_filling", {}).get("iso_radius_factor", 1.0))
            cov3D = make_isotropic_cov(n, float(cfg.get_cfg("particleRadius")), iso_factor, pos.device, torch.float32)

            rot = None
            init_screen_points_full = torch.zeros_like(pos)

            uniform = preprocessing_params.get("render_uniform_color", None)
            if uniform is not None:
                colors_precomp = torch.tensor(uniform, device=pos.device, dtype=torch.float32).view(1,3).repeat(n,1)
            else:
                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                tint = preprocessing_params.get("render_sh_tint", None)
                if tint is not None:
                    tint_t = torch.tensor(tint, device=pos.device, dtype=torch.float32).view(1,3)
                    colors_precomp = (colors_precomp * tint_t)

                gain = float(preprocessing_params.get("render_sh_gain", 1.0))
                gamma = float(preprocessing_params.get("render_sh_gamma", 1.0))
                if gain != 1.0:
                    colors_precomp = colors_precomp * gain
                if gamma != 1.0:
                    colors_precomp = torch.clamp(colors_precomp, 1e-6, 1.0) ** (1.0 / max(gamma, 1e-6))
                colors_precomp = colors_precomp.clamp(0.0, 1.0)

            rendering, raddi = rasterize(
                means3D=pos, means2D=init_screen_points_full,
                shs=None, colors_precomp=colors_precomp,
                opacities=opacity, scales=None, rotations=None, cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            cv2.imwrite(
                os.path.join(base_output_dir, f"{frame + 1:04d}.png"),
                255 * cv2_img,
            )

    if args.debug:
        xs = np.arange(len(density_err_hist))
        plt.figure()
        plt.plot(xs, density_err_hist, lw=1.5)
        plt.xlabel("frame")
        plt.ylabel("DFSPH density error")
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, "density_error.png"), dpi=150)
        plt.close()

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {base_output_dir}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {base_output_dir}/output.mp4"
        )
