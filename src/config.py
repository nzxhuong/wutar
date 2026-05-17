import torch

GRID_SIZE = 1024
W, H = 1280, 720

L = 1000.0 
G = 9.81    
A = 2e-8    
t = 0.0  
CHOPPY_FACTOR = 1.5   
WIND_SPEED = 30.0

wind_dir = [60.0, 30.0]

DT = 0.025
ALPHA = 0.1
LAMBDA_CHOP = 1.2
DISP_SCALE = 2.5

wake_strength = 1.0

cam_pos_initial = [0.0, 50.0, -250.0]
cam_yaw_initial = 1.56
cam_pitch_initial = -0.2

fov_degrees = 120
near_plane = 0.01
far_plane = 5000.0

light_vector = [0.4, 0.3, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
