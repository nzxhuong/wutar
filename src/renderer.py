import cv2
import moderngl
import numpy as np
from config import *

class WaveRenderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        
        self._load_shaders()
        
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, 
                                   fragment_shader=self.fragment_shader)
        
        self._setup_geometry()
        
        self._setup_textures()
        
        self._setup_lighting()
    
    def _load_shaders(self):
        """Load vertex and fragment shaders from files."""
        try:
            with open('shaders/vertex.glsl', 'r') as f:
                self.vertex_shader = f.read()
            with open('shaders/fragment.glsl', 'r') as f:
                self.fragment_shader = f.read()
        except FileNotFoundError as e:
            print(f"Error loading shaders: {e}")
            raise e
    
    def _setup_geometry(self):
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        xx, yy = np.meshgrid(x, y)
        vertices = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype('f4')
        
        self.vbo = self.ctx.buffer(vertices)
        
        indices = []
        for r in range(GRID_SIZE - 1):
            for c in range(GRID_SIZE - 1):
                i0 = r * GRID_SIZE + c
                i1 = i0 + 1
                i2 = i0 + GRID_SIZE
                i3 = i2 + 1
                indices.extend([i0, i1, i2, i1, i3, i2])
        
        self.ibo = self.ctx.buffer(np.array(indices, dtype='i4'))
        
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')], 
                                        index_buffer=self.ibo)
    
    def _setup_textures(self):
        self.height_tex = self.ctx.texture((GRID_SIZE, GRID_SIZE), 1, dtype='f4')
        self.height_tex_dx = self.ctx.texture((GRID_SIZE, GRID_SIZE), 2, dtype='f4')
        
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((W, H), 4)], 
            depth_attachment=self.ctx.depth_renderbuffer((W, H))
        )
    
    def _setup_lighting(self):
        light_vec = np.array(light_vector, dtype='f4')
        light_vec /= np.linalg.norm(light_vec)
        self.prog['light_vector'].write(light_vec)
    
    def draw(self, height_data, displacement_data, model_matrix, view_matrix, proj_matrix, cam_pos):
        self.height_tex.write(height_data.tobytes())
        self.height_tex_dx.write(displacement_data.tobytes())
        
        self.fbo.use()
        self.ctx.clear(0.1, 0.1, 0.1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.height_tex.use(location=0)
        self.height_tex_dx.use(location=1)
        
        self.prog['height_map'].value = 0
        self.prog['height_map_dx'].value = 1
        self.prog['model'].write(model_matrix)
        self.prog['view'].write(view_matrix)
        self.prog['proj'].write(proj_matrix)
        self.prog['cam_pos'].write(cam_pos)
        
        self.vao.render(moderngl.TRIANGLES)
    
    def get_image(self):
        raw = self.fbo.read(components=4)
        image = np.frombuffer(raw, dtype='u1').reshape((H, W, 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return np.flipud(image)
    
    def cleanup(self):
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.height_tex.release()
        self.fbo.release()
        self.prog.release()