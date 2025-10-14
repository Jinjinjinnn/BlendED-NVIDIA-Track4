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