import os
import numpy as np
import torch


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def save_points_ply(points: torch.Tensor | np.ndarray, filepath: str, color: tuple[int, int, int] | None = (255, 255, 255)) -> None:
    """
    Save 3D points to an ASCII PLY file. Optionally assign a single RGB color to all points.
    """
    pts = _to_numpy(points).astype(np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 3, "points must be (N,3)"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if color is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if color is None:
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")
        else:
            r, g, b = [int(max(0, min(255, c))) for c in color]
            for x, y, z in pts:
                f.write(f"{x} {y} {z} {r} {g} {b}\n")


def save_points_ply_colors(points: torch.Tensor | np.ndarray, colors: np.ndarray, filepath: str) -> None:
    """
    Save 3D points with per-vertex RGB colors (uint8) to an ASCII PLY file.
    colors should be shape (N,3), dtype uint8 or convertible.
    """
    pts = _to_numpy(points).astype(np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    assert pts.ndim == 2 and pts.shape[1] == 3, "points must be (N,3)"
    assert cols.shape == (pts.shape[0], 3), "colors must be (N,3)"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def save_bbox_obj(center: list[float] | tuple[float, float, float], size: list[float] | tuple[float, float, float], filepath: str) -> None:
    """
    Save a wireframe bounding box as an OBJ file with 8 vertices and 12 line segments.
    center: (cx, cy, cz)
    size: (sx, sy, sz)
    """
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

    # 8 corners
    corners = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
    ], dtype=np.float32)

    # 12 edges as pairs of vertex indices (1-based for OBJ)
    edges = [
        (1, 2), (1, 3), (2, 4), (3, 4),  # bottom face
        (5, 6), (5, 7), (6, 8), (7, 8),  # top face
        (1, 5), (2, 6), (3, 7), (4, 8),  # vertical edges
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("# Bounding box wireframe\n")
        for v in corners:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a, b in edges:
            f.write(f"l {a} {b}\n")


def save_init_scene_obj(
    particles: torch.Tensor | np.ndarray,
    domain_end: list[float] | tuple[float, float, float],
    bboxes: list[dict],
    filepath: str,
    particles_chunk: int = 2000,
) -> None:
    """
    Save particles and all bounding boxes (including domain) into a single OBJ file.
    - vertices (v): first Np particle vertices, then 8 vertices per bbox
    - points (p): references the first Np vertices (chunked to keep lines reasonable)
    - lines (l): 12 edges per bbox
    Note: some viewers may not show 'p' primitives by default; prefer PLY below for guaranteed point rendering.
    """
    pts = _to_numpy(particles).astype(np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 3, "particles must be (N,3)"

    # Prepare bbox list: domain first, then user bboxes
    cx, cy, cz = [d * 0.5 for d in domain_end]
    sx, sy, sz = domain_end
    all_bboxes = [{"point": [cx, cy, cz], "size": [sx, sy, sz]}]
    if bboxes:
        all_bboxes.extend([bc for bc in bboxes if "point" in bc and "size" in bc])

    # Edge order
    edge_order = [
        (0, 1), (0, 2), (1, 3), (2, 3),  # bottom
        (4, 5), (4, 6), (5, 7), (6, 7),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("# Combined OBJ: particles as points, boxes as lines\n")

        # 1) Write particle vertices
        for x, y, z in pts:
            f.write(f"v {x} {y} {z}\n")
        # 2) Write 'p' statements in chunks
        npart = pts.shape[0]
        if npart > 0:
            start = 1
            while start <= npart:
                end = min(start + particles_chunk - 1, npart)
                indices = " ".join(str(i) for i in range(start, end + 1))
                f.write(f"p {indices}\n")
                start = end + 1

        # 3) Write each bbox corners and edges
        cur_index = npart  # last written index
        for bc in all_bboxes:
            px, py, pz = bc["point"]
            sx, sy, sz = bc["size"]
            hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
            corners = [
                [px - hx, py - hy, pz - hz],
                [px + hx, py - hy, pz - hz],
                [px - hx, py + hy, pz - hz],
                [px + hx, py + hy, pz - hz],
                [px - hx, py - hy, pz + hz],
                [px + hx, py - hy, pz + hz],
                [px - hx, py + hy, pz + hz],
                [px + hx, py + hy, pz + hz],
            ]
            for v in corners:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            base = cur_index + 1
            for a, b in edge_order:
                f.write(f"l {base + a} {base + b}\n")
            cur_index += 8


def save_init_scene_ply(
    particles: torch.Tensor | np.ndarray,
    domain_end: list[float] | tuple[float, float, float],
    bboxes: list[dict],
    filepath: str,
    particle_color: tuple[int, int, int] = (153, 204, 255),
    box_color: tuple[int, int, int] = (255, 180, 64),
    with_edges: bool = False,
) -> None:
    """
    Save a single ASCII PLY containing:
    - vertices: particles (with color), followed by all bbox corner vertices (with box color)
    - edges (optional): 12 edges per box (domain first, then user-defined bboxes)
    Most viewers render points from PLY reliably; edges are optional to avoid artifacts.
    """
    pts = _to_numpy(particles).astype(np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 3, "particles must be (N,3)"

    # Prepare bbox list
    cx, cy, cz = [d * 0.5 for d in domain_end]
    sx, sy, sz = domain_end
    all_bboxes = [{"point": [cx, cy, cz], "size": [sx, sy, sz]}]
    if bboxes:
        all_bboxes.extend([bc for bc in bboxes if "point" in bc and "size" in bc])

    edge_order = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    n_particles = pts.shape[0]
    n_boxes = len(all_bboxes)
    n_box_vertices = 8 * n_boxes
    n_edges = 12 * n_boxes if with_edges else 0

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_particles + n_box_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if with_edges:
            f.write(f"element edge {n_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        f.write("end_header\n")

        # Write particles (with color)
        pr, pg, pb = [int(max(0, min(255, c))) for c in particle_color]
        for x, y, z in pts:
            f.write(f"{x} {y} {z} {pr} {pg} {pb}\n")

        # Write box corners (with box color)
        br, bg, bb = [int(max(0, min(255, c))) for c in box_color]
        base = n_particles
        for bc in all_bboxes:
            px, py, pz = bc["point"]
            sx, sy, sz = bc["size"]
            hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
            corners = [
                [px - hx, py - hy, pz - hz],
                [px + hx, py - hy, pz - hz],
                [px - hx, py + hy, pz - hz],
                [px + hx, py + hy, pz - hz],
                [px - hx, py - hy, pz + hz],
                [px + hx, py - hy, pz + hz],
                [px - hx, py + hy, pz + hz],
                [px + hx, py + hy, pz + hz],
            ]
            for v in corners:
                f.write(f"{v[0]} {v[1]} {v[2]} {br} {bg} {bb}\n")

        if with_edges:
            for i in range(n_boxes):
                box_vertex_base = n_particles + i * 8
                for a, b in edge_order:
                    f.write(f"{box_vertex_base + a} {box_vertex_base + b}\n")
