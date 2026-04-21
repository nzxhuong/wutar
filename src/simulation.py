import numpy as np
from config import *

class WaveSimulation:
    def __init__(self):
        """Initialize wave simulation with spectrum calculation."""
        self.u = np.linspace(-GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE) * (2 * np.pi / L)
        self.uu, self.vv = np.meshgrid(self.u, self.u)  
        self.u_nag = np.sqrt(self.uu**2 + self.vv**2)
        self.mask = self.u_nag > 0
        
        self.P = np.zeros((GRID_SIZE, GRID_SIZE))
        k_m = self.u_nag[self.mask]
        wind_dir_np = np.array(wind_dir)
        dot = (self.uu[self.mask]/k_m)*wind_dir_np[0] + (self.vv[self.mask]/k_m)*wind_dir_np[1]
        self.P[self.mask] = A * (np.exp(-1 / (k_m * (V_speed**2 / G))**2) / k_m**4) * (dot**2)
        
        self.omega = np.sqrt(G * self.u_nag)
        
        xi_r = np.random.randn(GRID_SIZE, GRID_SIZE)
        xi_i = np.random.randn(GRID_SIZE, GRID_SIZE)
        self.h0 = 1 / np.sqrt(2) * (xi_r + 1j * xi_i) * np.sqrt(self.P)
    
    def update(self, t):
        """
        Update wave simulation for time t and return height and displacement data.
        
        Args:
            t (float): Current time
            
        Returns:
            tuple: (height_data, displacement_data) as float32 arrays
        """
        h0_t = self.h0 * np.exp(1j * self.omega * t)
        
        height_data = np.real(np.fft.ifft2(np.fft.ifftshift(h0_t)))
        
        # Calculate displacement for choppy waves
        dx_disp = -1j * (self.uu/self.u_nag) * h0_t
        dx_data = np.real(np.fft.ifft2(np.fft.ifftshift(dx_disp)))
        dz_disp = -1j * (self.vv/self.u_nag) * h0_t
        dz_data = np.real(np.fft.ifft2(np.fft.ifftshift(dz_disp)))
        disp_data = np.stack([dx_data, dz_data], axis=-1)
        
        return height_data.astype('f4'), disp_data.astype('f4')