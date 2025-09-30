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
from SPH_Taichi.particle_system import ParticleSystem
from SPH_Taichi.config_builder import SimConfig

# MPM dependencies
# from mpm_solver_warp.engine_utils import *
# from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
# import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.sph_gaussian_converter import SPHGaussianConverter

# wp.init()
# wp.config.verify_cuda = True

# We need to initialize Taichi before using any of its functionalities.
# The SPH_Taichi modules rely on Taichi fields, so ti.init() must be called.
import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=4.0)


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
    
    # Set default output path based on config file name if not provided
    if args.output_path is None:
        scene_name = os.path.basename(args.config)
        scene_name = os.path.splitext(scene_name)[0].replace('_config', '')
        args.output_path = os.path.join("output", scene_name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    # The decode_param_json is not fully compatible with the new SPH config.
    # We will load the json manually and then initialize SimConfig.
    with open(args.config, 'r') as f:
        config_json = json.load(f)

    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # Prefer SPH-style config sections when present
    cfg_time = config_json.get("time_params", {})
    cfg_sph = config_json.get("sph_params", {})
    if "frame_dt" in cfg_time:
        time_params["frame_dt"] = cfg_time["frame_dt"]
    if "frame_num" in cfg_time:
        time_params["frame_num"] = cfg_time["frame_num"]
    # Unify substep: use SPH timeStepSize if provided
    if "timeStepSize" in cfg_sph:
        time_params["substep_dt"] = cfg_sph["timeStepSize"]

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = (
        init_opacity.squeeze(-1) >= preprocessing_params["opacity_threshold"]
    ).cpu()
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        # particle_position_tensor_to_ply(
        #     init_pos,
        #     "./log/init_particles.ply",
        # )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        # particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")
        pass

    # Apply particle filling if specified in the config
    if preprocessing_params.get("particle_filling") is not None:
        filling_params = preprocessing_params["particle_filling"]
        if filling_params.get("method") == "possion_disk_sampling":
            print("Applying Possion Disk Sampling for particle filling...")
            filled_particles = possion_disk_sampling(
                gaussians,
                filling_params["radius"],
                filling_params["num_samples_per_candidate"],
                mask=mask,
            )
            # Apply the same rotation to the newly filled particles
            rotated_pos = apply_rotations(filled_particles, rotation_matrices)
            
            # Since we replaced the particles, we need to regenerate other attributes
            # For now, let's create dummy values for cov, opacity, shs
            num_filled = rotated_pos.shape[0]
            # Use a smaller covariance for filled particles to represent them as small spheres
            init_cov = torch.eye(3, device="cuda").unsqueeze(0).repeat(num_filled, 1, 1) * (filling_params["radius"]**2)
            init_opacity = torch.ones(num_filled, 1, device="cuda") * 0.99
            # Use simple white color for filled particles
            init_shs = torch.zeros(num_filled, 1, 3, device="cuda")
            init_shs[:, 0, 0] = 0.28209479177387814 # SH basis for white color
            init_shs[:, 0, 1] = 0.28209479177387814
            init_shs[:, 0, 2] = 0.28209479177387814
            print(f"Filled with {num_filled} particles.")


    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]

        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
    # Map normalized coords (~[-0.5,0.5]) into SPH domain [domainStart, domainEnd]
    sph_params = config_json.get("sph_params", {})
    domain_start = sph_params.get("domainStart", [0.0, 0.0, 0.0])
    domain_end = sph_params.get("domainEnd", [2.0, 2.0, 2.0])
    domain_start_tensor = torch.tensor(domain_start, dtype=transformed_pos.dtype, device="cuda").reshape(1, 3)
    domain_end_tensor = torch.tensor(domain_end, dtype=transformed_pos.dtype, device="cuda").reshape(1, 3)
    domain_size_tensor = domain_end_tensor - domain_start_tensor

    norm01 = transformed_pos + 0.5
    transformed_pos = norm01 * domain_size_tensor + domain_start_tensor

    if args.debug:
        def _aabb(t):
            return torch.min(t, dim=0).values.detach().cpu().numpy(), torch.max(t, dim=0).values.detach().cpu().numpy()
        mn0, mx0 = _aabb(init_pos)
        mn1, mx1 = _aabb(rotated_pos)
        mn2, mx2 = _aabb(transformed_pos)
        print(f"AABB init_pos min/max: {mn0} / {mx0}")
        print(f"AABB rotated_pos min/max: {mn1} / {mx1}")
        print(f"AABB mapped_pos(min/max) to domain {domain_start}~{domain_end}: {mn2} / {mx2}")

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        # particle_position_tensor_to_ply(
        #     transformed_pos,
        #     "./log/transformed_particles.ply",
        # )
        pass
        
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    
    # Init the SPH solver
    print("Initializing SPH solver...")
    # Load the config dictionary into SimConfig
    config = SimConfig(config_dict=config_json) 
    ps = ParticleSystem(config, GGUI=False, max_num_particles=gs_num) # GGUI is False as we are using Gaussian Splatting renderer
    
    # Init the converter
    converter = SPHGaussianConverter()

    # Convert Gaussian particles to SPH format
    sph_data, n_particles = converter.gaussian_to_sph_particles(
        transformed_pos, init_cov, init_opacity, init_shs
    )
    
    # Add particles to the SPH particle system
    ps.add_particles(
        object_id=1, # Use object_id=1 for the main fluid body
        **sph_data
    )

    # Build and initialize the SPH solver
    sph_solver = ps.build_solver()
    sph_solver.initialize()


    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # camera setting (SPH domain -> world)
    # SPH target center: domain center
    sph_center = 0.5 * (domain_start_tensor + domain_end_tensor)
    # Map SPH point to Gaussian world (inverse of earlier mapping)
    def sph_point_to_world(p: torch.Tensor) -> torch.Tensor:
        norm01_r = (p - domain_start_tensor) / domain_size_tensor
        norm_centered = norm01_r - 0.5
        return apply_inverse_rotations(
            undotransform2origin(norm_centered, scale_origin, original_mean_pos),
            rotation_matrices,
        )

    center_world_t = sph_point_to_world(sph_center)
    # SPH up is +Y
    sph_up = torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=center_world_t.dtype)
    up_world_t = sph_point_to_world(sph_center + sph_up)
    world_space_vertical_axis = (up_world_t - center_world_t).squeeze(0)
    viewpoint_center_worldspace = center_world_t.squeeze(0).detach().cpu().numpy()
    vertical, h1, h2 = generate_local_coord(world_space_vertical_axis.detach().cpu().numpy())
    observant_coordinates = np.column_stack((h1, h2, vertical))

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        # save_data_at_frame(
        #     mpm_solver,
        #     directory_to_save,
        #     0,
        #     save_to_ply=args.output_ply,
        #     save_to_h5=args.output_h5,
        # )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = init_opacity
    shs_render = init_shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            # mpm_solver.p2g2p(frame, substep_dt, device=device)
            sph_solver.step()

        if args.output_ply or args.output_h5:
            # save_data_at_frame(
            #     mpm_solver,
            #     directory_to_save,
            #     frame + 1,
            #     save_to_ply=args.output_ply,
            #     save_to_h5=args.output_h5,
            # )
            pass # TODO: Implement saving for SPH particles if needed

        if args.render_img:
            # Export particle data from SPH solver
            sph_positions_np = ps.x.to_numpy()[:n_particles]
            
            # Convert SPH data back to Gaussian format for rendering
            pos, cov3D, opacity, shs = converter.sph_to_gaussian_particles(sph_positions_np)

            # Revert transformations to render in original world space
            norm01_r = (pos - domain_start_tensor) / domain_size_tensor
            norm_centered = norm01_r - 0.5
            pos = apply_inverse_rotations(
                undotransform2origin(
                    norm_centered, scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            # Covariance transformation is complex, for now we use the stored one
            # and only revert scale and rotation
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity, unselected_opacity], dim=0)
                shs = torch.cat([shs, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rotation=None) # Rotation from simulation is not used for color
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )
