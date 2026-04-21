import cv2
import numpy as np
import sys
sys.path.append('src')

# Import components
from config import *
from simulation import WaveSimulation
from camera import Camera
from renderer import WaveRenderer

def main():
    simulation = WaveSimulation()
    camera = Camera()
    renderer = WaveRenderer()
    
    t = 0.0
    
    print("Q=Quit")
    
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if camera.handle_input(key):
                break
            
            height_map, displacement_map, normal_map = simulation.update(t)
            
            model = np.eye(4, dtype='f4')
            view = camera.get_view_matrix()
            proj = Camera.perspective(np.radians(fov_degrees), W/H, near_plane, far_plane)
            
            renderer.draw(height_map, displacement_map, normal_map, model, view, proj, camera.pos)
            
            image = renderer.get_image()
            cv2.imshow("Wave", image)
            
            t += animation_speed
    
    finally:
        cv2.destroyAllWindows()
        renderer.cleanup()

if __name__ == "__main__":
    main()