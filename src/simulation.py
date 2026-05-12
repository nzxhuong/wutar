import numpy as np
import torch
import torch.nn.functional as F
from config import *

class WaveSimulation:
    def _get_ker_weight_torch(self, P=6, sigma=1.0):
        k = torch.arange(-P, P + 1, dtype=torch.float32, device=device)
        K, L_mesh = torch.meshgrid(k, k, indexing='ij')
        r = torch.sqrt(K**2 + L_mesh**2)
        q = torch.arange(1, 10001, dtype=torch.float32, device=device) * 0.001
        q3d, r3d = q[:, None, None], r[None, :, :]
        G_val = (q3d**2 * torch.exp(-sigma * q3d**2)
                 * torch.special.bessel_j0(q3d * r3d)).sum(dim=0)
        G_val /= G_val[P, P].clone()
        return G_val.view(1, 1, 2*P+1, 2*P+1)
    
    def _gaussian_kernel_2d(self, size, sigma):
        k = size // 2
        x = torch.arange(-k, k+1).float().to(device)
        g = torch.exp(-x**2 / (2*sigma**2))
        g /= g.sum()
        return (g[:, None] * g[None, :]).view(1, 1, size, size)
    
    def __init__(self):
        """Initialize wave simulation with spectrum calculation on GPU."""
        u_coords = (torch.fft.fftfreq(GRID_SIZE, d=1.0/GRID_SIZE, device=device)
                    * (2 * torch.pi / L))
        uu, vv = torch.meshgrid(u_coords, u_coords, indexing='ij')
        self.uu = uu
        self.vv = vv
        u_nag = torch.sqrt(uu**2 + vv**2)
        self.mask = u_nag > 0
        
        p_spectrum = torch.zeros((GRID_SIZE, GRID_SIZE), device=device)
        k_hat_x = torch.where(self.mask, uu / u_nag, torch.zeros_like(uu))
        k_hat_z = torch.where(self.mask, vv / u_nag, torch.zeros_like(vv))
        self.k_hat_x = k_hat_x
        self.k_hat_z = k_hat_z
        
        wind_dir_tensor = torch.tensor(wind_dir, dtype=torch.float32, device=device)
        wind_dir_tensor /= torch.linalg.norm(wind_dir_tensor)
        dot_product = k_hat_x * wind_dir_tensor[0] + k_hat_z * wind_dir_tensor[1]
        
        L_val = (WIND_SPEED**2) / G
        p_spectrum[self.mask] = (
            A * torch.exp(-1.0 / (u_nag[self.mask] * L_val)**2)
            / (u_nag[self.mask]**4)
        ) * (dot_product[self.mask]**2)
        
        omega_vals = torch.sqrt(G * u_nag)
        self.omega_vals = omega_vals
        
        xi_r1 = torch.randn(GRID_SIZE, GRID_SIZE, device=device)
        xi_i1 = torch.randn(GRID_SIZE, GRID_SIZE, device=device)
        h0 = (1 / np.sqrt(2)) * (xi_r1 + 1j * xi_i1) * torch.sqrt(p_spectrum)
        self.h0 = h0
        self.h0_conj = torch.conj(h0)
        
        self.kernel_tensor = self._get_ker_weight_torch(P=6, sigma=1.0)
        
        self.local_height = torch.zeros((1, 1, GRID_SIZE, GRID_SIZE), device=device)
        self.prev_height = torch.zeros((1, 1, GRID_SIZE, GRID_SIZE), device=device)
        self.obstruction = torch.ones((1, 1, GRID_SIZE, GRID_SIZE), device=device)
        
        self.obs_pos = torch.tensor([GRID_SIZE/2.0, GRID_SIZE/2.0],
                                    dtype=torch.float32, device=device)
        self.obs_vel = torch.zeros(2, dtype=torch.float32, device=device)
        
        self.kernel_2d = self._gaussian_kernel_2d(5, 1.0)
        self.pad_g = self.kernel_2d.shape[-1] // 2
        
        y_g = torch.arange(GRID_SIZE, device=device).float()
        x_g = torch.arange(GRID_SIZE, device=device).float()
        yy0, xx0 = torch.meshgrid(y_g, x_g, indexing='ij')
        self.obstruction[0, 0, (torch.abs(xx0 - self.obs_pos[0]) < 10) & 
                              (torch.abs(yy0 - self.obs_pos[1]) < 25)] = 0.0
        
        denom = 1.0 + ALPHA * DT
        self.C1 = (2.0 - ALPHA * DT) / denom
        self.C2 = 1.0 / denom
        self.C3 = (G * DT**2) / denom
        
        self.obstruction_dirty = True
        self.blured_obstruction = None
        self.source_term = None
        
        # Boat rotation
        self.boat_yaw = 0.0
    
    def update_obstruction(self, active_target=None):
        """Update obstruction position with enhanced boat physics and rotation."""
        pos_changed = False
        cos_y = np.cos(self.boat_yaw)
        sin_y = np.sin(self.boat_yaw)

        if active_target is not None:
            target = torch.tensor(active_target, dtype=torch.float32, device=device)
            diff = target - self.obs_pos
            if torch.linalg.norm(diff) > 0.1:
                tar_yaw = torch.atan2(diff[1], diff[0])
                yaw_diff = tar_yaw - self.boat_yaw
                yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
                self.boat_yaw += yaw_diff.item() * 10.0 * DT
                
                self.obs_pos += diff * DT
                self.obs_vel = diff / DT
                
                vel_magnitude = torch.linalg.norm(self.obs_vel)
                if vel_magnitude > 150.0:
                    self.obs_vel = self.obs_vel * (150.0 / vel_magnitude)
                pos_changed = True
            else:
                self.obs_vel = torch.zeros(2, device=device)
        else:
            if torch.linalg.norm(self.obs_vel) > 0.1:
                self.obs_vel *= 0.99
                self.obs_pos += self.obs_vel * DT
                pos_changed = True
            else:
                self.obs_vel = torch.zeros(2, device=device)

        if pos_changed:
            self.obstruction.fill_(1.0)
            y_g2 = torch.arange(GRID_SIZE, device=device).float()
            x_g2 = torch.arange(GRID_SIZE, device=device).float()
            yy2, xx2 = torch.meshgrid(y_g2, x_g2, indexing='ij')
            
            dx = xx2 - self.obs_pos[1]
            dy = yy2 - self.obs_pos[0]
            dx_rot = dx * cos_y - dy * sin_y
            dy_rot = dx * sin_y + dy * cos_y
            mask2 = (torch.abs(dx_rot) < 10) & (torch.abs(dy_rot) < 25)
            self.obstruction[0, 0, mask2] = 0.0
            self.obstruction_dirty = True

        if self.obstruction_dirty:
            self.blured_obstruction = F.conv2d(
                F.pad(self.obstruction, (self.pad_g, self.pad_g, self.pad_g, self.pad_g), mode='circular'),
                self.kernel_2d, padding=0)
            self.source_term = 1.0 - self.blured_obstruction
            self.obstruction_dirty = False
    
    def update(self, t):
        """
        Update wave simulation for time t and return height, displacement, and normal data.
        
        Args:
            t (float): Current time
            
        Returns:
            tuple: (height_data, displacement_data, normal_data) as float32 arrays
        """
        self.local_height += self.source_term
        self.local_height *= self.blured_obstruction
        self.local_height *= wake_strength
        
        h0_t = (self.h0 * torch.exp(1j * self.omega_vals * t) +
                self.h0_conj * torch.exp(-1j * self.omega_vals * t))
        ambient_torch = torch.fft.ifft2(h0_t).real.unsqueeze(0).unsqueeze(0) * 10.0
        
        self.local_height -= ambient_torch * (1.0 - self.obstruction)
        
        new_h = (self.local_height * self.C1) - (self.prev_height * self.C2) - \
                (self.C3 * F.conv2d(F.pad(self.local_height, (6, 6, 6, 6), mode='circular'), 
                                    self.kernel_tensor))
        self.prev_height, self.local_height = self.local_height, new_h
        
        combined = (ambient_torch * 10.0 + self.local_height) * self.obstruction
        h_field = combined.squeeze()
        
        h_scaled = h_field.contiguous()
        
        disp_fft = torch.stack([-1j * self.k_hat_x * h0_t,
                                -1j * self.k_hat_z * h0_t], dim=0)
        disp = torch.fft.ifft2(disp_fft).real * LAMBDA_CHOP
        dx_tensor = torch.stack([disp[0], disp[1]], dim=-1).contiguous()
        
        return h_scaled, dx_tensor