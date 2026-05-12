import moderngl
import numpy as np
from config import *

class WaveRenderer:
    def __init__(self, ctx):
        """Initialize renderer with existing ModernGL context (from GLFW)."""
        self.ctx = ctx
        
        self._load_shaders()
        
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, 
                                   fragment_shader=self.fragment_shader)
        self.prog_obj = self.ctx.program(vertex_shader=self.obj_vertex_shader,
                                       fragment_shader=self.obj_fragment_shader)
        self.prog_screen = self.ctx.program(vertex_shader=self.screen_vertex_shader,
                                          fragment_shader=self.screen_fragment_shader)
        self.prog_dot = self.ctx.program(vertex_shader=self.dot_vertex_shader,
                                        fragment_shader=self.dot_fragment_shader)
        
        self._setup_geometry()
        self._setup_object_geometry()
        self._setup_screen_quad()
        self._setup_hand_dot()
        self._setup_textures()
        self._setup_lighting()
    
    def _load_shaders(self):
        """Load vertex and fragment shaders from files."""
        try:
            with open('shaders/vertex.glsl', 'r') as f:
                self.vertex_shader = f.read()
            with open('shaders/fragment.glsl', 'r') as f:
                self.fragment_shader = f.read()
            with open('shaders/obj_vertex.glsl', 'r') as f:
                self.obj_vertex_shader = f.read()
            with open('shaders/obj_fragment.glsl', 'r') as f:
                self.obj_fragment_shader = f.read()
            with open('shaders/screen_vertex.glsl', 'r') as f:
                self.screen_vertex_shader = f.read()
            with open('shaders/screen_fragment.glsl', 'r') as f:
                self.screen_fragment_shader = f.read()
            with open('shaders/dot_vertex.glsl', 'r') as f:
                self.dot_vertex_shader = f.read()
            with open('shaders/dot_fragment.glsl', 'r') as f:
                self.dot_fragment_shader = f.read()
        except FileNotFoundError as e:
            print(f"Error loading shaders: {e}")
            raise e
    
    def _setup_geometry(self):
        x = np.linspace(-L/2, L/2, GRID_SIZE)
        y = np.linspace(-L/2, L/2, GRID_SIZE)
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
    
    def _setup_object_geometry(self):
        """Setup boat/obstruction object geometry (cube)."""
        obj_verts = np.array([
            -1,-1, 1,  0,0,1,   1,-1, 1,  0,0,1,   1, 1, 1,  0,0,1,  -1, 1, 1,  0,0,1,
            -1,-1,-1,  0,0,-1, -1, 1,-1,  0,0,-1,  1, 1,-1,  0,0,-1,  1,-1,-1,  0,0,-1,
            -1, 1,-1,  0,1,0,   -1, 1, 1,  0,1,0,   1, 1, 1,  0,1,0,   1, 1,-1,  0,1,0,
            -1,-1,-1,  0,-1,0,  1,-1,-1,  0,-1,0,   1,-1, 1,  0,-1,0,  -1,-1, 1,  0,-1,0,
            -1,-1,-1, -1,0,0,  -1, 1,-1, -1,0,0,  -1, 1, 1, -1,0,0,  -1,-1, 1, -1,0,0,
             1,-1,-1,  1,0,0,   1, 1,-1,  1,0,0,   1, 1, 1,  1,0,0,   1,-1, 1,  1,0,0,
        ], dtype='f4')
        
        obj_indices = np.array([
            0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 
            12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
        ], dtype='i4')
        
        self.obj_vbo = self.ctx.buffer(obj_verts)
        self.obj_ibo = self.ctx.buffer(obj_indices)
        self.obj_vao = self.ctx.vertex_array(self.prog_obj, 
                                             [(self.obj_vbo, '3f 3f', 'in_vert', 'in_norm')], 
                                             index_buffer=self.obj_ibo)
    
    def _setup_screen_quad(self):
        """Setup full-screen quad for FBO blit."""
        screen_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.screen_vbo = self.ctx.buffer(screen_quad)
        self.screen_vao = self.ctx.vertex_array(self.prog_screen, 
                                                [(self.screen_vbo, '2f', 'in_pos')])
    
    def _setup_hand_dot(self):
        """Setup hand dot overlay VAO."""
        dot_data = np.zeros((1, 2), dtype='f4')
        self.dot_vbo = self.ctx.buffer(dot_data)
        self.dot_vao = self.ctx.vertex_array(self.prog_dot, 
                                             [(self.dot_vbo, '2f', 'in_pos')])
    
    def _setup_textures(self):
        self.height_tex = self.ctx.texture((GRID_SIZE, GRID_SIZE), 1, dtype='f4')
        self.height_tex_dx = self.ctx.texture((GRID_SIZE, GRID_SIZE), 2, dtype='f4')
        
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((W, H), 4)], 
            depth_attachment=self.ctx.depth_renderbuffer((W, H))
        )
        self.fbo_color_tex = self.fbo.color_attachments[0]
    
    def _setup_lighting(self):
        light_vec = np.array(light_vector, dtype='f4')
        light_vec /= np.linalg.norm(light_vec)
        self.light_vector = light_vec  # Store as instance attribute
        self.prog['light_vector'].write(light_vec)
        self.prog_obj['light_vector'].write(light_vec)
    
    def draw_ocean(self, height_tensor, displacement_tensor, model_matrix, view_matrix, proj_matrix, cam_pos):
        """Draw ocean surface with zero-copy GPU tensor upload."""
        self.height_tex.write(height_tensor.cpu().numpy().tobytes())
        self.height_tex_dx.write(displacement_tensor.cpu().numpy().tobytes())
        
        self.fbo.use()
        self.ctx.clear(0.05, 0.08, 0.15)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.height_tex.use(location=0)
        self.height_tex_dx.use(location=1)
        
        self.prog['height_map'].value = 0
        self.prog['height_map_dx'].value = 1
        self.prog['WORLD_L'].value = L
        self.prog['model'].write(model_matrix)
        self.prog['view'].write(view_matrix)
        self.prog['proj'].write(proj_matrix)
        self.prog['cam_pos'].write(cam_pos)
        self.prog['light_vector'].write(self.light_vector)
        
        self.vao.render(moderngl.TRIANGLES)
    
    def draw_object(self, obj_model, view_matrix, proj_matrix):
        """Draw the boat/obstruction object."""
        self.prog_obj['model'].write(np.ascontiguousarray(obj_model.T))
        self.prog_obj['view'].write(view_matrix)
        self.prog_obj['proj'].write(proj_matrix)
        self.obj_vao.render(moderngl.TRIANGLES)
    
    def draw_hand_dot(self, tip_ndc, confirmed):
        """Draw hand overlay dot."""
        if tip_ndc is not None:
            dot_data = np.array([[tip_ndc[0], tip_ndc[1]]], dtype='f4')
            self.dot_vbo.write(dot_data.tobytes())
            
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            
            if confirmed:
                self.prog_dot['u_color'].value = (0.1, 0.9, 0.2)
            else:
                self.prog_dot['u_color'].value = (1.0, 0.2, 0.2)
            
            self.dot_vao.render(moderngl.POINTS, vertices=1)
            self.ctx.disable(moderngl.BLEND)
    
    def blit_to_screen(self):
        """Blit FBO to screen (zero-copy)."""
        self.ctx.screen.use()
        self.ctx.clear(0.05, 0.08, 0.15)
        
        self.fbo_color_tex.use(location=0)
        self.prog_screen['screen_tex'].value = 0
        
        self.screen_vao.render(moderngl.TRIANGLE_STRIP)
    
    def cleanup(self):
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.obj_vbo.release()
        self.obj_ibo.release()
        self.obj_vao.release()
        self.screen_vbo.release()
        self.screen_vao.release()
        self.dot_vbo.release()
        self.dot_vao.release()
        self.height_tex.release()
        self.fbo.release()
        self.prog.release()
        self.prog_obj.release()
        self.prog_screen.release()
        self.prog_dot.release()