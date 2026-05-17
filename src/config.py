import torch

GRID_SIZE = 1024
W, H = 1400, 1050
MESH_SIZE = 4096
L = 500.0 
G = 9.81    
t = 0.0  
WIND_SPEED = 15.0
WIND_ANGLE = 45.0        
FETCH = 1000000.0        
DEPTH = 500.0            
SWELL = 0.5              
LOW_CUTOFF = 0.001       
HIGH_CUTOFF = 9999.0

DT = 0.025
ALPHA = 0.1
LAMBDA_CHOP = 1.5
DISP_SCALE = 2.0

FOAM_INTENSITY = 0.8
FOAM_DECAY = 0.009

wake_strength = 1.0

cam_pos_initial = [0.0, 150.0, -300.0]
cam_yaw_initial = 1.56
cam_pitch_initial = -0.45

fov_degrees = 60
near_plane = 0.01
far_plane = 5000.0

light_vector = [0.4, 0.3, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
