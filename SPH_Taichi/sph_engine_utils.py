import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import taichi as ti
import numpy as np
import torch
import h5py

from SPH_Taichi.particle_system import ParticleSystem
from utils.decode_param import compute_sph_domain


def initialize_sph_from_positions(cfg, positions_torch: torch.Tensor, margin_scale: float = 3.0):
    assert hasattr(cfg, "get_cfg")
    assert positions_torch.dim() == 2 and positions_torch.shape[1] == 3

    dom = compute_sph_domain(cfg, positions_torch, margin_scale=margin_scale)
    if hasattr(cfg, "_cfg") and isinstance(cfg._cfg, dict):
        cfg._cfg["domainStart"] = dom["domainStart"]
        cfg._cfg["domainEnd"] = dom["domainEnd"]

    shift = torch.tensor(dom["shift"], device=positions_torch.device, dtype=positions_torch.dtype)
    init_pos = positions_torch + shift

    n = init_pos.shape[0]
    ps = ParticleSystem(cfg, GGUI=False, reserve_particle_num=n)
    # TODO: init_from_positions implement
    ps.init_from_positions(init_pos, density0=cfg.get_cfg("density0"))

    solver = ps.build_solver()
    solver.initialize()

    return {"ps": ps, "solver": solver, "shift": shift, "cfg": cfg}


def step_sph(context: dict, substeps: int = 1):
    solver = context["solver"]
    for _ in range(substeps):
        solver.step()


def export_positions_numpy(context: dict) -> np.ndarray:
    ps = context["ps"]
    n = ps.particle_num[None]
    return ps.x.to_numpy()[:n, :]


def export_positions_torch(context: dict, device: str = None) -> torch.Tensor:
    arr = export_positions_numpy(context)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device=device)
    return t


def particle_position_tensor_to_ply(position, filename: str):
    if isinstance(position, torch.Tensor):
        if position.is_cuda:
            position = position.detach().cpu()
        position = position.numpy()
    pos = np.asarray(position).astype(np.float32)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    writer = ti.tools.PLYWriter(num_vertices=pos.shape[0])
    writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
    writer.export(filename)


def save_data_at_frame_sph(ctx_or_ps, dir_name: str, frame: int,
                           save_to_ply: bool = True, save_to_h5: bool = False,
                           time_value: float = None, density_error: float = None,
                           colors_rgb=None, pos_override=None):
    ps = ctx_or_ps["ps"] if isinstance(ctx_or_ps, dict) else ctx_or_ps
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)

    fullfilename = os.path.join(dir_name, "sim_" + str(frame).zfill(10) + ".h5")

    # positions
    if pos_override is None:
        pos_np = ps.x.to_numpy()[:ps.particle_num[None], :].astype(np.float32)
    else:
        if hasattr(pos_override, "is_cuda"):
            pos_override = pos_override.detach().cpu()
        pos_np = np.asarray(pos_override, dtype=np.float32)

    # colors
    col_np = None
    if colors_rgb is not None:
        if hasattr(colors_rgb, "is_cuda"):
            colors_rgb = colors_rgb.detach().cpu()
        col_np = np.asarray(colors_rgb, dtype=np.float32)
        if col_np.max() > 1.0:
            col_np = np.clip(col_np / 255.0, 0.0, 1.0)

    if save_to_ply:
        writer = ti.tools.PLYWriter(num_vertices=pos_np.shape[0])
        writer.add_vertex_pos(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2])
        if col_np is not None:
            rgb255 = np.clip(col_np * 255.0, 0, 255).astype(np.uint8)
            writer.add_vertex_color(rgb255[:, 0], rgb255[:, 1], rgb255[:, 2])
        writer.export(fullfilename[:-2] + "ply")

    if save_to_h5:
        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        f = h5py.File(fullfilename, "w")
        f.create_dataset("x", data=pos_np.astype(np.float32).T)  # (3,n)
        if col_np is not None:
            f.create_dataset("rgb", data=np.clip(col_np, 0.0, 1.0).astype(np.float32).T)  # (3,n)
        if time_value is None:
            current_time = np.array([[float(frame)]], dtype=np.float32)
        else:
            current_time = np.array([[float(time_value)]], dtype=np.float32)
        f.create_dataset("time", data=current_time)
        if density_error is not None:
            f.create_dataset("density_error", data=np.array([[float(density_error)]], dtype=np.float32))
        f.close()
        print("save simulation data at frame", frame, "to", fullfilename)


def make_isotropic_cov(count, radius, iso_factor, device, dtype):
    r2 = float(radius) * float(iso_factor)
    r2 = r2 * r2
    base = torch.tensor([r2, 0.0, 0.0, r2, 0.0, r2], device=device, dtype=dtype)
    return base.view(1, 6).repeat(int(count), 1)