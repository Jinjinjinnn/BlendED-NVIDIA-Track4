import torch


@torch.no_grad()
def uniform_fill_sph(
    surface_pos: torch.Tensor,
    surface_opacity: torch.Tensor,
    particle_radius: float,
    opacity_threshold: float = 0.15,
    boundary=None,
    k: int = 8,
    sigma_scale: float = 1.0,
    neighbor_radius_scale: float = 3.0,
    device: str | None = None,
):

    assert surface_pos.ndim == 2 and surface_pos.shape[1] == 3
    assert surface_opacity.ndim == 2 and surface_opacity.shape[1] == 1
    if device is None:
        device = surface_pos.device
    spacing = float(4.0 * particle_radius)
    sigma = float(sigma_scale * particle_radius)
    r_neighbor = float(neighbor_radius_scale * spacing)

    # bbox
    if boundary is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = boundary
        bbox_min = torch.tensor([xmin, ymin, zmin], device=device, dtype=torch.float32)
        bbox_max = torch.tensor([xmax, ymax, zmax], device=device, dtype=torch.float32)
    else:
        bbox_min = torch.min(surface_pos, dim=0).values
        bbox_max = torch.max(surface_pos, dim=0).values

    # uniform grid
    def ar(vmin, vmax):
        n = max(1, int(torch.ceil((vmax - vmin) / spacing).item()))
        return vmin + torch.arange(n, device=device, dtype=torch.float32) * spacing

    xs = ar(bbox_min[0], bbox_max[0] + 1e-6)
    ys = ar(bbox_min[1], bbox_max[1] + 1e-6)
    zs = ar(bbox_min[2], bbox_max[2] + 1e-6)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (Ng,3)

    Ns = surface_pos.shape[0]
    Ng = grid_pts.shape[0]
    if Ns == 0 or Ng == 0:
        return torch.empty((0, 3), device=device, dtype=surface_pos.dtype)

    kval = min(int(k), Ns)
    two_sigma2 = 2.0 * (sigma * sigma)
    opa_flat = surface_opacity.squeeze(1)  # (Ns,)

    # batch
    B = max(1, int(2e5 // max(Ns, 1)))
    inside_list = []
    for s in range(0, Ng, B):
        e = min(Ng, s + B)
        G = grid_pts[s:e]                                # (b,3)
        d2 = torch.cdist(G, surface_pos, p=2.0).pow(2)   # (b,Ns)

        # optional radius mask
        if r_neighbor > 0.0:
            d2 = torch.where(d2 <= r_neighbor * r_neighbor, d2, torch.full_like(d2, float("inf")))

        d2_topk, idx = torch.topk(d2, k=kval, dim=1, largest=False, sorted=False)  # (b,k)
        w = torch.exp(-d2_topk / two_sigma2) + 1e-12                                # (b,k)
        w_sum = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-8)
        opa_k = opa_flat[idx]                                                       # (b,k)
        opa_acc = (w * opa_k).sum(dim=1, keepdim=True) / w_sum                      # (b,1)
        inside_list.append(opa_acc.squeeze(1) > opacity_threshold)

    inside_mask = torch.cat(inside_list, dim=0)  # (Ng,)
    filled_pos = grid_pts[inside_mask]
    return filled_pos


@torch.no_grad()
def init_filled_particles(
    pos: torch.Tensor,            # (Ns,3)
    shs: torch.Tensor,            # (Ns,C,3)
    cov: torch.Tensor,            # (Ns,6)
    opacity: torch.Tensor,        # (Ns,1)
    new_pos: torch.Tensor,        # (Nn,3)
    particle_radius: float,
    iso_radius_factor: float = 1.0,
):

    assert pos.ndim == 2 and pos.shape[1] == 3
    assert new_pos.ndim == 2 and new_pos.shape[1] == 3
    dev = pos.device
    if new_pos.shape[0] == 0:
        return shs, opacity, cov

    # closest index
    d2 = torch.cdist(new_pos, pos, p=2.0).pow(2)   # (Nn,Ns)
    idx = torch.argmin(d2, dim=1)                  # (Nn,)

    new_shs = shs[idx]                             # (Nn,C,3)
    new_opacity = opacity[idx]                     # (Nn,1)

    # isotrophic cov
    r_iso2 = float(iso_radius_factor * particle_radius) ** 2
    iso_cov6 = torch.tensor([r_iso2, 0.0, 0.0, r_iso2, 0.0, r_iso2], device=dev, dtype=cov.dtype)
    new_cov = iso_cov6.unsqueeze(0).repeat(new_pos.shape[0], 1)  # (Nn,6)

    concat_shs = torch.cat([shs, new_shs], dim=0)                 # (Ns+Nn,C,3)
    concat_opacity = torch.cat([opacity, new_opacity], dim=0)     # (Ns+Nn,1)
    concat_cov6 = torch.cat([cov, new_cov], dim=0)                # (Ns+Nn,6)
    return concat_shs, concat_opacity, concat_cov6


@torch.no_grad()
def sph_uniform_fill_and_seed(
    surface_pos: torch.Tensor,
    surface_shs: torch.Tensor,
    surface_cov6: torch.Tensor,
    surface_opacity: torch.Tensor,
    particle_radius: float,
    opacity_threshold: float = 0.15,
    boundary=None,
    device: str | None = None,
    iso_radius_factor: float = 1.0,
    k: int = 8,
    sigma_scale: float = 1.0,
    neighbor_radius_scale: float = 3.0,
):

    if device is None:
        device = surface_pos.device

    filled = uniform_fill_sph(
        surface_pos=surface_pos.to(device=device),
        surface_opacity=surface_opacity.to(device=device),
        particle_radius=float(particle_radius),
        opacity_threshold=float(opacity_threshold),
        boundary=boundary,
        k=int(k),
        sigma_scale=float(sigma_scale),
        neighbor_radius_scale=float(neighbor_radius_scale),
        device=device,
    )

    shs_all, opacity_all, cov6_all = init_filled_particles(
        pos=surface_pos.to(device=device),
        shs=surface_shs.to(device=device),
        cov=surface_cov6.to(device=device),
        opacity=surface_opacity.to(device=device),
        new_pos=filled,
        particle_radius=float(particle_radius),
        iso_radius_factor=float(iso_radius_factor),
    )

    seed_pos = torch.cat([surface_pos.to(device=device), filled], dim=0)
    return seed_pos, shs_all, opacity_all, cov6_all
