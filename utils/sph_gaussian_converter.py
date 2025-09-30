import torch
import numpy as np

class SPHGaussianConverter:
    def __init__(self, default_density=1000.0, default_mass=0.001):
        """
        Initializes the converter.

        Args:
            default_density (float): Default density for new SPH particles.
            default_mass (float): Default mass for new SPH particles.
        """
        self.default_density = default_density
        self.default_mass = default_mass

        # These will store the non-physics related Gaussian attributes
        self.gaussian_attributes = {}

    def gaussian_to_sph_particles(self, gs_positions, gs_covariances, gs_opacities, gs_shs):
        """
        Converts Gaussian splatting particles to a format suitable for the SPH_Taichi solver.

        Args:
            gs_positions (torch.Tensor): Positions of Gaussian particles (N, 3).
            gs_covariances (torch.Tensor): Covariances of Gaussian particles (N, 6).
            gs_opacities (torch.Tensor): Opacities of Gaussian particles (N, 1).
            gs_shs (torch.Tensor): Spherical harmonics coefficients (N, C, 3).

        Returns:
            dict: A dictionary containing particle data for the SPH solver.
            int: The number of particles.
        """
        n_particles = gs_positions.shape[0]

        # Convert torch tensors to numpy arrays, detaching them from the computation graph
        positions_np = gs_positions.detach().cpu().numpy().astype(np.float32)

        # Create basic SPH particle attributes
        sph_data = {
            'new_particles_num': n_particles,
            'new_particles_positions': positions_np,
            'new_particles_velocity': np.zeros((n_particles, 3), dtype=np.float32),
            'new_particle_density': np.full(n_particles, self.default_density, dtype=np.float32),
            'new_particle_pressure': np.zeros(n_particles, dtype=np.float32),
            # In SPH_Taichi, material=1 means fluid
            'new_particles_material': np.ones(n_particles, dtype=np.int32), 
             # is_dynamic=1 means the particle can move
            'new_particles_is_dynamic': np.ones(n_particles, dtype=np.int32),
            # Default color, can be customized
            'new_particles_color': np.array([[153, 204, 255]] * n_particles, dtype=np.int32) 
        }

        # Store Gaussian-specific attributes for rendering later
        # These are not used by the SPH physics simulation
        self.gaussian_attributes = {
            'covariances': gs_covariances.detach(),
            'opacities': gs_opacities.detach(),
            'shs': gs_shs.detach()
        }

        return sph_data, n_particles

    def sph_to_gaussian_particles(self, sph_positions_np):
        """
        Converts updated SPH particle positions back to Gaussian particle format for rendering.
        Other Gaussian attributes are retrieved from storage.

        Args:
            sph_positions_np (np.ndarray): Updated particle positions from the SPH solver (N, 3).

        Returns:
            torch.Tensor: Updated positions for Gaussian rendering.
            torch.Tensor: Original covariances for Gaussian rendering.
            torch.Tensor: Original opacities for Gaussian rendering.
            torch.Tensor: Original spherical harmonics for Gaussian rendering.
        """
        # Convert updated positions back to a torch tensor
        updated_positions = torch.from_numpy(sph_positions_np).to('cuda', dtype=torch.float32)

        # Retrieve the original, unchanged Gaussian attributes
        covariances = self.gaussian_attributes['covariances']
        opacities = self.gaussian_attributes['opacities']
        shs = self.gaussian_attributes['shs']

        return updated_positions, covariances, opacities, shs
