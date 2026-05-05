import torch

GRID_SIZE = 256
W, H = 1280, 720

L = 500.0 
G = 9.81    
A = 1.0    
t = 0.0  
CHOPPY_FACTOR = 1.5   
WIND_SPEED = 50.0

wind_dir = [60.0, 30.0]

# iWave physics constants
DT = 0.025
ALPHA = 0.1
LAMBDA_CHOP = 1.5

# Obstruction constants
OBS_RADIUS = 10.0
wake_strength = 0.9

cam_pos_initial = [0.0, 150.0, -250.0]
cam_yaw_initial = 1.56
cam_pitch_initial = -0.4

fov_degrees = 60
near_plane = 0.01
far_plane = 5000.0

light_vector = [0.4, 0.3, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
