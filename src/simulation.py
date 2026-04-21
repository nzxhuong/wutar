import numpy as np
from config import *

class WaveSimulation:
    def __init__(self):
        """Initialize wave simulation with JONSWAP spectrum calculation."""
        self.u = np.linspace(-GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE) * (2 * np.pi / L)
        self.uu, self.vv = np.meshgrid(self.u, self.u)  
        self.u_nag = np.sqrt(self.uu**2 + self.vv**2)
        self.mask = self.u_nag > 0
        
        u10 = V_speed
        alpha = ALPHA_COEFF * ((u10**2) / (FETCH * G))**0.22
        omega_p = OMEGA_P_COEFF * ((G**2) / (u10 * FETCH))**(1/3)
        
        self.P = np.zeros((GRID_SIZE, GRID_SIZE))
        k_m = self.u_nag[self.mask]
        omega_m = np.sqrt(G * k_m)
        sigma = np.where(omega_m <= omega_p, 0.07, 0.09)
        r_exponent = np.exp(-((omega_m - omega_p)**2) / (2 * (sigma**2) * (omega_p**2)))
        s_omega = (alpha * G**2 / omega_m**5) * np.exp(-1.25 * (omega_p / omega_m)**4) * (GAMMA**r_exponent)
        dk_domega = 0.5 * np.sqrt(G / k_m)
        p_k = s_omega * dk_domega
        wind_dir_np = np.array(wind_dir)
        wind_norm = np.linalg.norm(wind_dir_np)
        if wind_norm > 0:
            wind_dir_np /= wind_norm
        else:
            wind_dir_np = np.array([1.0, 0.0]) 
        dot_product = (self.uu[self.mask]/k_m)*wind_dir_np[0] + (self.vv[self.mask]/k_m)*wind_dir_np[1]
        self.P[self.mask] = p_k * (dot_product**2)
        
        self.omega = np.sqrt(G * self.u_nag)
        
        xi_r1 = np.random.randn(GRID_SIZE, GRID_SIZE)
        xi_i1 = np.random.randn(GRID_SIZE, GRID_SIZE)
        self.h0 = 1 / np.sqrt(2) * (xi_r1 + 1j * xi_i1) * np.sqrt(self.P)
        
        # Add conjugate spectrum for wave symmetry
        P_neg = np.roll(np.flip(self.P, axis=(0,1)), 1, axis=(0,1))
        xi_r2 = np.random.randn(GRID_SIZE, GRID_SIZE)
        xi_i2 = np.random.randn(GRID_SIZE, GRID_SIZE)
        self.h0_conj = np.conj(1 / np.sqrt(2) * (xi_r2 + 1j * xi_i2) * np.sqrt(P_neg))
    
    def update(self, t):
        """
        Update wave simulation for time t and return height, displacement, and normal data.
        
        Args:
            t (float): Current time
            
        Returns:
            tuple: (height_data, displacement_data, normal_data) as float32 arrays
        """
        h0_t = self.h0 * np.exp(1j * self.omega * t) + self.h0_conj * np.exp(-1j * self.omega * t)
        
        height_data = np.real(np.fft.ifft2(np.fft.ifftshift(h0_t)))
        
        kx_norm = np.where(self.mask, self.uu / self.u_nag, 0)
        kz_norm = np.where(self.mask, self.vv / self.u_nag, 0)
        
        dx_disp = -1j * kx_norm * h0_t
        dx_data = np.real(np.fft.ifft2(np.fft.ifftshift(dx_disp)))
        dz_disp = -1j * kz_norm * h0_t
        dz_data = np.real(np.fft.ifft2(np.fft.ifftshift(dz_disp)))
        disp_data = np.stack([dx_data, dz_data], axis=-1)
        
        grad_x_freq = 1j * self.uu * h0_t
        grad_x_data = np.real(np.fft.ifft2(np.fft.ifftshift(grad_x_freq)))
        grad_z_freq = 1j * self.vv * h0_t
        grad_z_data = np.real(np.fft.ifft2(np.fft.ifftshift(grad_z_freq)))
        normal_data = np.stack([grad_x_data, grad_z_data], axis=-1)
        
        return height_data.astype('f4'), disp_data.astype('f4'), normal_data.astype('f4')