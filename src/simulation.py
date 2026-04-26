import numpy as np
from config import *

class WaveSimulation:
    def __init__(self):
        """Initialize wave simulation with spectrum calculation."""
        self.u = np.fft.fftfreq(GRID_SIZE, d=1.0/GRID_SIZE) * ((2 * np.pi) / L)
        self.uu, self.vv = np.meshgrid(self.u, self.u)  
        self.u_nag = np.sqrt(self.uu**2 + self.vv**2)
        self.mask = self.u_nag > 0
        
        self.P = np.zeros((GRID_SIZE, GRID_SIZE))
        k_m = self.u_nag[self.mask]
        wind_dir_np = np.array(wind_dir)
        wind_dir_np /= np.linalg.norm(wind_dir_np)
        dot = (self.uu[self.mask]/k_m)*wind_dir_np[0] + (self.vv[self.mask]/k_m)*wind_dir_np[1]
        self.P[self.mask] = A * (np.exp(-1 / (k_m * (V_speed**2 / G))**2) / k_m**4) * (dot**2)
        
        self.omega = np.sqrt(G * self.u_nag)
        
        xi_r1, xi_i1 = np.random.randn(2, GRID_SIZE, GRID_SIZE)
        self.h0 = (1 / np.sqrt(2)) * (xi_r1 + 1j * xi_i1) * np.sqrt(self.P)
        
        self.h0_conj = np.conj(self.h0)
    
    def update(self, t):
        """
        Update wave simulation for time t and return height, displacement, and normal data.
        
        Args:
            t (float): Current time
            
        Returns:
            tuple: (height_data, displacement_data, normal_data) as float32 arrays
        """
        h0_t = self.h0 * np.exp(1j * self.omega * t) + self.h0_conj * np.exp(-1j * self.omega * t)
        
        height_data = np.real(np.fft.ifft2(h0_t)) * 10.0
        
        kx_norm = np.where(self.mask, self.uu / self.u_nag, 0) * CHOPPY_FACTOR
        kz_norm = np.where(self.mask, self.vv / self.u_nag, 0) * CHOPPY_FACTOR
        
        dx_disp = -1j * kx_norm * h0_t
        dx_data = np.real(np.fft.ifft2(dx_disp))
        dz_disp = -1j * kz_norm * h0_t
        dz_data = np.real(np.fft.ifft2(dz_disp))
        disp_data = np.stack([dx_data, dz_data], axis=-1)
        
        h_dx = np.roll(height_data, -1, axis=1) - np.roll(height_data, 1, axis=1)
        h_dz = np.roll(height_data, -1, axis=0) - np.roll(height_data, 1, axis=0)
        texel_world = 2.0 * L / GRID_SIZE
        h_dx /= texel_world
        h_dz /= texel_world
        normal_data = np.stack([h_dx, h_dz], axis=-1)
        return height_data.astype('f4'), disp_data.astype('f4'), normal_data.astype('f4')