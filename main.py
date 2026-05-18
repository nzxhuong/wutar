import sys
import numpy as np
import glfw
import moderngl
import torch
sys.path.append('src')

from config import *
from simulation import WaveSimulation
from camera import Camera
from renderer import WaveRenderer
from hand_tracking import HandTracker

if not glfw.init():
    raise RuntimeError("GLFW initialization failed")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

window = glfw.create_window(W, H, "Water", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("GLFW window creation failed")

glfw.make_context_current(window)
glfw.swap_interval(1) 

ctx = moderngl.create_context()

def main():
    simulation = WaveSimulation()
    camera = Camera()
    renderer = WaveRenderer(ctx)
    hand_tracker = HandTracker()
    
    t = 0.0
    
    print("Press Q to quit.")
    
    with torch.no_grad():
        while not glfw.window_should_close(window):
            if int(t / DT) % 60 == 0:
                torch.cuda.synchronize()
            glfw.poll_events()
            
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                break
            
            tip_ndc, hand_target, confirmed = hand_tracker.get_hand_state()
            
            active_target = hand_target
            
            simulation.update_obstruction(active_target)
            
            height_tensor, displacement_tensor, foam_tensor  = simulation.update(t)
            
            model = np.eye(4, dtype='f4')
            view = camera.get_view_matrix()
            proj = Camera.perspective(np.radians(fov_degrees), W/H, near_plane, far_plane)
            renderer.draw_sky(view, proj)
            renderer.draw_ocean(height_tensor, displacement_tensor, foam_tensor, model, view, proj, camera.pos)

            
            obs_pos_cpu = simulation.obs_pos.cpu().numpy()
            world_x = (obs_pos_cpu[1] / GRID_SIZE) * L - (L / 2)
            world_z = (obs_pos_cpu[0] / GRID_SIZE) * L - (L / 2)
            boat_yaw = simulation.boat_yaw
            
            S = np.eye(4, dtype='f4')
            S[0,0] = OBJ_WIDTH / 2
            S[1,1] = 0.5
            S[2,2] = OBJ_HEIGHT / 2
            
            cos_y = np.cos(boat_yaw)
            sin_y = np.sin(boat_yaw)
            R = np.array([
                [cos_y, 0, sin_y, 0],
                [0,     1, 0,     0],
                [-sin_y,0, cos_y, 0],
                [0,     0, 0,     1]
            ], dtype='f4')
            
            T = np.eye(4, dtype='f4')
            T[0,3] = float(world_x)
            T[1,3] = 0.5 
            T[2,3] = float(world_z)
            
            obj_model = T @ R @ S
            renderer.draw_object(obj_model, view, proj)
            
            renderer.blit_to_screen()
            
            renderer.draw_hand_dot(tip_ndc, confirmed)
            
            glfw.swap_buffers(window)
            
            t += DT
    
    hand_tracker.stop()
    renderer.cleanup()
    glfw.terminate()

if __name__ == "__main__":
    main()