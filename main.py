import sys
import numpy as np
import glfw
import moderngl
import torch
sys.path.append('src')

# Import components
from config import *
from simulation import WaveSimulation
from camera import Camera
from renderer import WaveRenderer

# ── GLFW window setup ─────────────────────────────────────────────────────────
if not glfw.init():
    raise RuntimeError("GLFW initialization failed")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

window = glfw.create_window(W, H, "iWave Ocean Viewer", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("GLFW window creation failed")

glfw.make_context_current(window)
glfw.swap_interval(1) 

ctx = moderngl.create_context()

mouse_dragging = False
mouse_target = None

def cursor_pos_callback(window, xpos, ypos):
    global mouse_dragging, mouse_target
    if mouse_dragging:
        grid_x = (xpos / W) * GRID_SIZE
        grid_y = (1.0 - ypos / H) * GRID_SIZE
        mouse_target = [grid_y, grid_x]

def mouse_button_callback(window, button, action, mods):
    global mouse_dragging, mouse_target
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_dragging = True
            xpos, ypos = glfw.get_cursor_pos(window)
            grid_x = (xpos / W) * GRID_SIZE
            grid_y = (1.0 - ypos / H) * GRID_SIZE
            mouse_target = [grid_y, grid_x]
        elif action == glfw.RELEASE:
            mouse_dragging = False
            mouse_target = None

glfw.set_cursor_pos_callback(window, cursor_pos_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)

def main():
    simulation = WaveSimulation()
    camera = Camera()
    renderer = WaveRenderer(ctx)
    
    t = 0.0
    obj_model = np.eye(4, dtype='f4')
    obj_model[0, 0] = 20.0  # Width
    obj_model[1, 1] = 5.0  # Height
    obj_model[2, 2] = 50.0  # Length
    
    print("[ocean] Starting render loop. Press Q to quit.")
    
    with torch.no_grad():
        while not glfw.window_should_close(window):
            glfw.poll_events()
            
            # Check for Q key press
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                break
            
            # Update obstruction based on mouse
            simulation.update_obstruction(mouse_target)
            
            # Update simulation
            height_map, displacement_map, normal_map = simulation.update(t)
            
            # Camera matrices
            model = np.eye(4, dtype='f4')
            view = camera.get_view_matrix()
            proj = Camera.perspective(np.radians(fov_degrees), W/H, near_plane, far_plane)
            
            # Draw ocean
            renderer.draw_ocean(height_map, displacement_map, model, view, proj, camera.pos)
            
            # Draw boat/obstruction object
            # Convert grid position to world position
            obs_pos_cpu = simulation.obs_pos.cpu().numpy()
            world_x = (obs_pos_cpu[1] / GRID_SIZE) * L - (L / 2)
            world_z = (obs_pos_cpu[0] / GRID_SIZE) * L - (L / 2)
            
            obj_model[0, 3] = world_x
            obj_model[2, 3] = world_z
            obj_model[1, 3] = 0.5  # Slightly above water
            
            renderer.draw_object(obj_model, view, proj)
            
            # Blit to screen
            renderer.blit_to_screen()
            
            # Swap buffers
            glfw.swap_buffers(window)
            
            t += DT
    
    # Cleanup
    renderer.cleanup()
    glfw.terminate()

if __name__ == "__main__":
    main()