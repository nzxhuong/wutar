import numpy as np
from config import *

class Camera:
    def __init__(self):
        """Initialize camera with default position and orientation."""
        # Camera state
        self.pos = np.array(cam_pos_initial, dtype='f4')
        self.yaw = cam_yaw_initial
        self.pitch = cam_pitch_initial
        
        # Movement vectors
        self.world_up = np.array([0, 1, 0], dtype='f4')
    
    @staticmethod
    def look_at(eye, target, up):
        """Create view matrix using look-at transformation."""
        f = (target - eye)
        f /= np.linalg.norm(f)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        
        res = np.eye(4, dtype='f4')
        res[0, 0] = s[0]; res[0, 1] = s[1]; res[0, 2] = s[2]
        res[1, 0] = u[0]; res[1, 1] = u[1]; res[1, 2] = u[2]
        res[2, 0] = -f[0]; res[2, 1] = -f[1]; res[2, 2] = -f[2]
        res[3, 0] = -np.dot(s, eye)
        res[3, 1] = -np.dot(u, eye)
        res[3, 2] = np.dot(f, eye)
        return res
    
    @staticmethod
    def perspective(fov, aspect, near, far):
        """Create perspective projection matrix."""
        f = 1.0 / np.tan(fov / 2)
        res = np.zeros((4, 4), dtype='f4')
        res[0, 0] = f / aspect
        res[1, 1] = f
        res[2, 2] = (far + near) / (near - far)
        res[2, 3] = -1
        res[3, 2] = (2 * far * near) / (near - far)
        return res
    
    def get_direction_vectors(self):
        """Calculate forward and right vectors from yaw and pitch."""
        forward = np.array([
            np.cos(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch),
            np.sin(self.yaw) * np.cos(self.pitch)
        ], dtype='f4')
        
        right = np.cross(self.world_up, forward)
        right /= np.linalg.norm(right)
        
        return forward, right
    
    def handle_input(self, key):
        """Handle keyboard input to update camera position and rotation."""
        forward, right = self.get_direction_vectors()
        
        if key == ord('w'): self.pos += forward * move_speed
        if key == ord('s'): self.pos -= forward * move_speed
        if key == ord('a'): self.pos -= right * move_speed
        if key == ord('d'): self.pos += right * move_speed
        if key == ord('e'): self.pos[1] += move_speed
        if key == ord('r'): self.pos[1] -= move_speed
        
        if key == ord('i'): self.pitch += rot_speed
        if key == ord('k'): self.pitch -= rot_speed
        if key == ord('j'): self.yaw -= rot_speed
        if key == ord('l'): self.yaw += rot_speed
        
        self.pitch = np.clip(self.pitch, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        return key == ord('q')
    
    def get_view_matrix(self):
        """Get the current view matrix."""
        forward, _ = self.get_direction_vectors()
        target = self.pos + forward
        return self.look_at(self.pos, target, self.world_up)