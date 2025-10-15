import torch
import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from WCSPH import WCSPHSolver
from DFSPH import DFSPHSolver
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace


# TODO: time_value, init_from_positions (in sph_engin_util.py)
@ti.data_oriented
class ParticleSystem:
    def __init__(self, config, GGUI=False, reserve_particle_num: int = 0):
        self.cfg = config
        self.GGUI = GGUI
        # Domain
        ds = self.cfg.get_cfg("domainStart") or [0.0, 0.0, 0.0]
        de = self.cfg.get_cfg("domainEnd") or [1.0, 1.0, 1.0]
        self.domain_start = np.array(ds, dtype=np.float32)
        self.domian_end = np.array(de, dtype=np.float32)
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1
        # # Simulation method
        # self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        # Grid
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.padding = self.grid_size

        # Counters
        self.particle_num = ti.field(int, shape=())
        self.particle_num[None] = 0
        self.particle_max_num = int(reserve_particle_num)

        # Object/rigid bookkeeping (no scene objects now)
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.num_rigid_bodies = 0

        # Grid accumulators
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        # Particle fields
        nmax = self.particle_max_num
        self.object_id = ti.field(dtype=int, shape=nmax)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.m_V = ti.field(dtype=float, shape=nmax)
        self.m = ti.field(dtype=float, shape=nmax)
        self.density = ti.field(dtype=float, shape=nmax)
        self.pressure = ti.field(dtype=float, shape=nmax)
        self.material = ti.field(dtype=int, shape=nmax)
        self.color = ti.Vector.field(3, dtype=int, shape=nmax)
        self.is_dynamic = ti.field(dtype=int, shape=nmax)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=nmax)
            self.density_adv = ti.field(dtype=float, shape=nmax)

        # Buffers for sorting
        self.object_id_buffer = ti.field(dtype=int, shape=nmax)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=nmax)
        self.m_V_buffer = ti.field(dtype=float, shape=nmax)
        self.m_buffer = ti.field(dtype=float, shape=nmax)
        self.density_buffer = ti.field(dtype=float, shape=nmax)
        self.pressure_buffer = ti.field(dtype=float, shape=nmax)
        self.material_buffer = ti.field(dtype=int, shape=nmax)
        self.color_buffer = ti.Vector.field(3, dtype=int, shape=nmax)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=nmax)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=nmax)
            self.density_adv_buffer = ti.field(dtype=float, shape=nmax)

        # Grid ids
        self.grid_ids = ti.field(int, shape=nmax)
        self.grid_ids_buffer = ti.field(int, shape=nmax)
        self.grid_ids_new = ti.field(int, shape=nmax)

        # Optional vis buffers
        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=nmax)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=nmax)


    def build_solver(self):
        solver_type = self.cfg.get_cfg("simulationMethod")
        if solver_type == 0:
            return WCSPHSolver(self)
        elif solver_type == 4:
            return DFSPHSolver(self)
        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_0[p] = x
        self.v[p] = v
        self.density[p] = density
        self.m_V[p] = self.m_V0
        self.m[p] = self.m_V0 * density
        self.pressure[p] = pressure
        self.material[p] = material
        self.is_dynamic[p] = is_dynamic
        self.color[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num


    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)


    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]
    

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    
    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I]-1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for I in ti.grouped(self.grid_ids):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.object_id_buffer[new_index] = self.object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.x_buffer[new_index] = self.x[I]
            self.v_buffer[new_index] = self.v[I]
            self.acceleration_buffer[new_index] = self.acceleration[I]
            self.m_V_buffer[new_index] = self.m_V[I]
            self.m_buffer[new_index] = self.m[I]
            self.density_buffer[new_index] = self.density[I]
            self.pressure_buffer[new_index] = self.pressure[I]
            self.material_buffer[new_index] = self.material[I]
            self.color_buffer[new_index] = self.color[I]
            self.is_dynamic_buffer[new_index] = self.is_dynamic[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor_buffer[new_index] = self.dfsph_factor[I]
                self.density_adv_buffer[new_index] = self.density_adv[I]
        
        for I in ti.grouped(self.x):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.object_id[I] = self.object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.x[I] = self.x_buffer[I]
            self.v[I] = self.v_buffer[I]
            self.acceleration[I] = self.acceleration_buffer[I]
            self.m_V[I] = self.m_V_buffer[I]
            self.m[I] = self.m_buffer[I]
            self.density[I] = self.density_buffer[I]
            self.pressure[I] = self.pressure_buffer[I]
            self.material[I] = self.material_buffer[I]
            self.color[I] = self.color_buffer[I]
            self.is_dynamic[I] = self.is_dynamic_buffer[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor[I] = self.dfsph_factor_buffer[I]
                self.density_adv[I] = self.density_adv_buffer[I]
    

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()
    

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                    task(p_i, p_j, ret)


    @ti.kernel
    def _import_particle_x(self, new_positions: ti.types.ndarray(), n: int):
        for i in range(n):
            for d in ti.static(range(self.dim)):
                self.x[i][d] = new_positions[i, d]


    def export_particle_x_to_torch(self, device: str = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        n = self.particle_num[None]
        np_pos = self.x.to_numpy()[:n, :]  # (n,3)
        t = torch.from_numpy(np_pos)
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t


    def import_particle_x_from_torch(self, t: torch.Tensor):
        assert t.dim() == 2 and t.shape[1] == self.dim, "Expected tensor of shape (N, dim)"
        n_used = self.particle_num[None]
        n = min(t.shape[0], n_used)
        arr = t.detach().to(dtype=torch.float32, device="cpu").numpy()
        self._import_particle_x(arr[:n], n)


    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.x[i]
                self.color_vis_buffer[i] = self.color[i] / 255.0


    @ti.kernel
    def _init_from_positions_kernel(self,
                                    new_positions: ti.types.ndarray(),
                                    n: int,
                                    density0: float,
                                    is_dynamic: int):
        for p in range(n):
            # positions
            for d in ti.static(range(self.dim)):
                self.x[p][d] = new_positions[p, d]
                self.x_0[p][d] = new_positions[p, d]
            # state
            self.v[p] = ti.Vector.zero(float, self.dim)
            self.acceleration[p] = ti.Vector.zero(float, self.dim)
            self.density[p] = density0
            self.m_V[p] = self.m_V0
            self.m[p] = self.m_V0 * density0
            self.pressure[p] = 0.0
            # tags
            self.object_id[p] = 0
            self.material[p] = self.material_fluid
            self.is_dynamic[p] = is_dynamic
            self.color[p] = ti.Vector([255, 255, 255])


    def init_from_positions(self, positions_torch: torch.Tensor, density0: float, is_dynamic: int = 1):
        """
        positions_torch: (N,3) torch.Tensor; 도메인 중심 정렬(shift) 적용된 위치를 넘겨주세요.
        """
        assert positions_torch.dim() == 2 and positions_torch.shape[1] == self.dim
        n = positions_torch.shape[0]
        assert n <= self.particle_max_num, "reserve_particle_num이 부족합니다."

        arr = positions_torch.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        self._init_from_positions_kernel(arr, n, float(density0), int(is_dynamic))
        self.particle_num[None] = n

        self.initialize_particle_system()


    def dump(self, obj_id):
        np_object_id = self.object_id.to_numpy()
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.x.to_numpy()[mask]
        np_v = self.v.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }