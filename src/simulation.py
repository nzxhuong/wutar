import numpy as np
import torch
import torch.nn.functional as F
from config import *

class WaveSimulation:
    def _dispersion(self, k_mag):
        """Deep water dispersion relation."""
        return torch.sqrt(G * k_mag)

    def _dispersion_peak(self):
        """JONSWAP peak angular frequency."""
        return 22.0 * (G**2 / (self.WIND_SPEED * self.FETCH)) ** 0.33

    def _tma_correction(self, omega):
        """Shallow water TMA correction factor."""
        omega_h = omega * torch.sqrt(torch.tensor(self.DEPTH / G, device=device))
        correction = torch.where(
            omega_h <= 1.0,
            0.5 * omega_h**2,
            torch.where(
                omega_h < 2.0,
                1.0 - 0.5 * (2.0 - omega_h)**2,
                torch.ones_like(omega_h)
            )
        )
        return correction

    def _jonswap(self, omega, omega_p):
        """JONSWAP frequency spectrum."""
        alpha = 0.076 * (self.WIND_SPEED**2 / (self.FETCH * G))**0.22
        gamma = 3.3

        sigma = torch.where(omega <= omega_p,
                            torch.full_like(omega, 0.07),
                            torch.full_like(omega, 0.09))

        r = torch.exp(-(omega - omega_p)**2 / (2 * sigma**2 * omega_p**2))

        first  = alpha * G**2 / (omega**5)
        second = torch.exp(-1.25 * (omega_p / omega)**4)
        third  = gamma ** r

        return self._tma_correction(omega) * first * second * third

    def _base_spread(self, omega, angle, omega_p):
        """Mitsuyasu directional spreading."""
        ratio = omega / omega_p
        beta = torch.where(
            ratio < 0.95,
            2.61 * ratio**1.3,
            torch.where(
                ratio <= 1.6,
                2.28 * ratio**(-1.3),
                10.0 ** (-0.4 + 0.8393 * torch.exp(-0.567 * torch.log(ratio**2)))
            )
        )
        sech = 1.0 / torch.cosh(beta * angle)
        return beta / (2.0 * torch.tanh(torch.tensor(torch.pi, device=device) * beta)) * sech**2

    def _swell_spread(self, omega, angle, omega_p):
        """Long-period swell directional component."""
        s = 16.0 * torch.tanh(omega_p / omega) * self.SWELL**2
        # normalization: 2^(2s-1)/pi * (Gamma(s+1))^2 / Gamma(2s+1)
        # approximate with a simpler form for GPU
        norm = (2.0**(2*s - 1)) / torch.pi * torch.exp(
            2 * torch.lgamma(s + 1) - torch.lgamma(2*s + 1)
        )
        return norm * torch.abs(torch.cos(angle / 2))**(2 * s)

    def _jonswap_spectrum(self, k_mag, omega, angle, omega_p):
        """Full directional JONSWAP spectrum S(k)."""
        # dω/dk for change of variables from S(ω) to S(k)
        domega_dk = G / (2.0 * torch.sqrt(G * k_mag))

        base   = self._base_spread(omega, angle, omega_p)
        swell  = self._swell_spread(omega, angle, omega_p)

        # Normalize directional spread (approximate integral over -π to π)
        # Unity does this with a loop; we scale by a constant factor
        directional = base * swell

        return self._jonswap(omega, omega_p) * directional * domega_dk / k_mag
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
        self.WIND_SPEED = WIND_SPEED
        self.FETCH = FETCH
        self.DEPTH = DEPTH
        self.SWELL = SWELL

        u_coords = (torch.fft.fftfreq(GRID_SIZE, d=1.0/GRID_SIZE, device=device)
                    * (2 * torch.pi / L))
        uu, vv = torch.meshgrid(u_coords, u_coords, indexing='ij')
        self.uu = uu
        self.vv = vv
        k_mag = torch.sqrt(uu**2 + vv**2)
        self.mask = k_mag > 0

        omega = self._dispersion(torch.where(self.mask, k_mag, torch.ones_like(k_mag)))
        omega_p = self._dispersion_peak()

        wind_angle_rad = torch.tensor(WIND_ANGLE * torch.pi / 180.0, device=device)
        k_angle = torch.atan2(vv, uu)
        rel_angle = k_angle - wind_angle_rad
        rel_angle = (rel_angle + torch.pi) % (2 * torch.pi) - torch.pi

        self.k_hat_x = torch.where(self.mask, uu / k_mag, torch.zeros_like(uu))
        self.k_hat_z = torch.where(self.mask, vv / k_mag, torch.zeros_like(vv))

        p_spectrum = torch.zeros((GRID_SIZE, GRID_SIZE), device=device)
        safe_k = torch.where(self.mask, k_mag, torch.ones_like(k_mag))
        safe_omega = torch.where(self.mask, omega, torch.ones_like(omega))

        p_spectrum[self.mask] = self._jonswap_spectrum(
            safe_k, safe_omega, rel_angle, omega_p
        )[self.mask]

        p_spectrum = torch.where(
            (k_mag > LOW_CUTOFF) & (k_mag < HIGH_CUTOFF),
            p_spectrum,
            torch.zeros_like(p_spectrum)
        )

        delta_k = (2 * torch.pi / L)**2
        xi_r = torch.randn(GRID_SIZE, GRID_SIZE, device=device)
        xi_i = torch.randn(GRID_SIZE, GRID_SIZE, device=device)
        h0 = (1 / torch.sqrt(torch.tensor(2.0, device=device))) \
            * (xi_r + 1j * xi_i) \
            * torch.sqrt(torch.clamp(p_spectrum * delta_k * 2, min=0.0))
        self.h0 = h0
        self.h0_conj = torch.conj(h0)

        self.omega_vals = omega
                
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
        
        self.boat_yaw = 0.0

        self.foam = torch.zeros((GRID_SIZE, GRID_SIZE), device=device)

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
        ambient_torch = torch.fft.ifft2(h0_t, norm="forward").real.unsqueeze(0).unsqueeze(0)
        
        self.local_height -= ambient_torch * (1.0 - self.obstruction)
        
        new_h = (self.local_height * self.C1) - (self.prev_height * self.C2) - \
                (self.C3 * F.conv2d(F.pad(self.local_height, (6, 6, 6, 6), mode='circular'), 
                                    self.kernel_tensor))
        self.prev_height, self.local_height = self.local_height, new_h
        
        combined = (ambient_torch  * DISP_SCALE + self.local_height) * self.obstruction
        h_field = combined.squeeze()
        
        h_scaled = h_field.contiguous()
        
        disp_fft = torch.stack([-1j * self.k_hat_x * h0_t  * LAMBDA_CHOP,
                                -1j * self.k_hat_z * h0_t * LAMBDA_CHOP], dim=0)
        disp = torch.fft.ifft2(disp_fft, norm = 'forward').real * DISP_SCALE
        dx_tensor = torch.stack([disp[0], disp[1]], dim=-1).contiguous()
        j_xx_fft = -self.k_hat_x * self.uu * h0_t
        j_zz_fft = -self.k_hat_z * self.vv * h0_t
        j_xz_fft = -self.k_hat_x * self.vv * h0_t
        j_xx = torch.fft.ifft2(j_xx_fft, norm='forward').real
        j_zz = torch.fft.ifft2(j_zz_fft, norm='forward').real
        j_xz = torch.fft.ifft2(j_xz_fft, norm='forward').real

        jacobian = (1 + LAMBDA_CHOP * j_xx) * (1 + LAMBDA_CHOP * j_zz) - (LAMBDA_CHOP * j_xz)**2
        foam_input = -jacobian + FOAM_INTENSITY
        decay = FOAM_DECAY * DT / torch.clamp(foam_input, 0.5, None)
        accumulation = self.foam - decay
        self.foam = torch.clamp(torch.max(accumulation, foam_input), 0.0, 1.0)

        return h_scaled, dx_tensor, self.foam.contiguous()

