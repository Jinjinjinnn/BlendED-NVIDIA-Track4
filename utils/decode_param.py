import numpy as np
import torch
import json

class SPHSimConfig:
    def __init__(self, config_dict: dict):
        self._cfg = dict(config_dict or {})

    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist and name not in self._cfg:
            raise KeyError(f"{name} not in SPHSimConfig")
        return self._cfg.get(name, None)


def decode_sph_param_json(json_file: str) -> SPHSimConfig:
    with open(json_file, "r") as f:
        sim_params = json.load(f)

    cfg = {}
    cfg["simulationMethod"] = 4  # DFSPH
    cfg["timeStepSize"] = sim_params.get("timeStepSize", sim_params.get("substep_dt", 1e-4))

    cfg["density0"] = sim_params.get("density0", 1000.0)
    cfg["viscosity"] = sim_params.get("viscosity", 0.01)
    cfg["surface_tension"] = sim_params.get("surface_tension", 0.01)
    cfg["particleRadius"] = sim_params.get("particleRadius", 0.01)

    cfg["gravitation"] = sim_params.get("gravitation", [0.0, -9.81, 0.0])

    if "domainStart" in sim_params:
        cfg["domainStart"] = sim_params["domainStart"]
    if "domainEnd" in sim_params:
        cfg["domainEnd"] = sim_params["domainEnd"]

    cfg["shift"] = sim_params.get("shift", True)

    return SPHSimConfig(cfg)

def decode_pipeline_params_json(json_file):
    import json
    with open(json_file, "r") as f:
        sim_params = json.load(f)

    # time params
    time_params = {
        "substep_dt": sim_params.get("substep_dt", 1e-4),
        "frame_dt": sim_params.get("frame_dt", 1e-2),
        "frame_num": sim_params.get("frame_num", 100),
    }

    # preprocessing params
    preprocessing_params = {
        "opacity_threshold": sim_params.get("opacity_threshold", 0.02),
        "rotation_degree": sim_params.get("rotation_degree", []),
        "rotation_axis": sim_params.get("rotation_axis", []),
        "sim_area": sim_params.get("sim_area", None),
        "scale": sim_params.get("scale", 1.0),
        "render_uniform_color": sim_params.get("render_uniform_color", None),
        "render_sh_tint": sim_params.get("render_sh_tint", None),
        "render_sh_gain": sim_params.get("render_sh_gain", 1.0),
        "render_sh_gamma": sim_params.get("render_sh_gamma", 1.0),
    }

    sph_top = sim_params.get("sph_filling", {}) or {}
    preprocessing_params["sph_filling"] = {
        "enabled": bool(sph_top.get("enabled", False)),
        "opacity_threshold": sph_top.get("opacity_threshold", 0.15),
        "boundary": sph_top.get("boundary", None),
        "k": sph_top.get("k", 8),
        "sigma_scale": sph_top.get("sigma_scale", 1.0),
        "neighbor_radius_scale": sph_top.get("neighbor_radius_scale", 3.0),
        "iso_radius_factor": sph_top.get("iso_radius_factor", 1.0),
    }

    # camera params (Y-up)
    camera_params = {
        "sph_space_viewpoint_center": sim_params.get("sph_space_viewpoint_center", [1.0, 1.0, 1.0]),
        "sph_space_vertical_upward_axis": sim_params.get("sph_space_vertical_upward_axis", [0.0, 1.0, 0.0]),
        "default_camera_index": sim_params.get("default_camera_index", 0),
        "show_hint": sim_params.get("show_hint", False),
        "init_azimuthm": sim_params.get("init_azimuthm", None),
        "init_elevation": sim_params.get("init_elevation", None),
        "init_radius": sim_params.get("init_radius", None),
        "delta_a": sim_params.get("delta_a", None),
        "delta_e": sim_params.get("delta_e", None),
        "delta_r": sim_params.get("delta_r", None),
        "move_camera": sim_params.get("move_camera", False),
    }

    return time_params, preprocessing_params, camera_params


def decode_all_params_json(json_file):
    cfg = decode_sph_param_json(json_file)
    time_params, preprocessing_params, camera_params = decode_pipeline_params_json(json_file)
    return cfg, time_params, preprocessing_params, camera_params


def compute_sph_domain(cfg: SPHSimConfig, positions, margin_scale: float = 3.0):
    if isinstance(positions, torch.Tensor):
        min_pos = positions.min(dim=0).values.detach().cpu().numpy()
        max_pos = positions.max(dim=0).values.detach().cpu().numpy()
    else:
        arr = np.asarray(positions)
        min_pos = arr.min(axis=0)
        max_pos = arr.max(axis=0)

    extent = np.maximum(max_pos - min_pos, 1e-6)

    ds = cfg.get_cfg("domainStart")
    de = cfg.get_cfg("domainEnd")

    if ds is not None and de is not None:
        domain_start = np.asarray(ds, dtype=np.float32)
        domain_end = np.asarray(de, dtype=np.float32)
        domain_size = domain_end - domain_start
        domain_center = domain_start + 0.5 * domain_size
    else:
        domain_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_size = np.maximum(margin_scale * extent, 1e-4)
        domain_end = (domain_start + domain_size).astype(np.float32)
        domain_center = 0.5 * domain_size

    cluster_center = 0.5 * (min_pos + max_pos)
    if cfg.get_cfg("shift", True):
        shift = (domain_center - cluster_center).astype(np.float32)
    else:
        shift = np.zeros(3, dtype=np.float32)

    return {
        "domainStart": domain_start.tolist(),
        "domainEnd": domain_end.tolist(),
        "shift": shift,
    }