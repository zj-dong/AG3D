'''
Helpful functions for rendering meshes based on pytorch3d.
'''

import cv2
import torch
import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures

class Renderer():
    def __init__(self, image_size=256,anti_alias=False,real_cam=False):
        super().__init__()

        self.anti_alias = anti_alias

        self.image_size = image_size


        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        R = torch.from_numpy(np.array([[1., 0., 0.],
                                       [0., -1., 0.],
                                       [0., 0., 1.]])).float().unsqueeze(0).to(self.device)


        t = torch.from_numpy(np.array([[0., 0., 2.]])).float().to(self.device)

        self.R = R

        if real_cam:
            self.cameras = FoVPerspectiveCameras(R=R, T=t,device=self.device)
        else:
            self.cameras = FoVOrthographicCameras(R=R, T=t,device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        if anti_alias: image_size = image_size*2
        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=5,blur_radius=1e-6)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_dict(self, mesh_dict, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():


            if 'norm' not in mesh_dict:
                mesh = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None])
                normals = torch.stack(mesh.verts_normals_list())
            else:
                normals = mesh_dict['norm'][None]

            front_light = torch.tensor([0,0,-1]).float().to(mesh_dict['verts'].device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'x' in mode or 'y' in mode:
                mesh = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None])
                normals_coarse = torch.stack(mesh.verts_normals_list())[0]
                normals_fine = mesh_dict['norm']

                cos_dis = (normals_coarse*normals_fine).sum(1, keepdims=True)

                sigma = 0.2
                fine_confidence = 0.5*(cos_dis + 1) # 0~1
                fine_confidence = torch.exp(-(fine_confidence-1)**2/2.0/sigma/sigma)

                fused_n = normals_fine*fine_confidence + normals_coarse*(1-fine_confidence)
                normals_x = fused_n / ((fused_n**2).sum(1, keepdims=True))**0.5
                normals_x = normals_x[None]
                shades_x = (normals_x * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'y' in mode:
                normals_vis_x = normals_x* 0.5 + 0.5 
                mesh_normal = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=normals_vis_x))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'x' in mode:
                mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades_x))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode: 
                assert('color' in mesh_dict)
                mesh_albido = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=mesh_dict['color'][None]))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            if 't' in mode: 
                assert('color' in mesh_dict)
                mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=mesh_dict['color'][None]*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results


    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = torch.tensor([0,0,-1]).float().to(verts.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode: 
                assert(colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            if 't' in mode: 
                assert(colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results


    def render_mesh_pytorch(self, mesh, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            normals = torch.stack(mesh.verts_normals_list())
            front_light = torch.tensor([0,0,-1]).float().to(mesh.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = mesh.clone()
                mesh_normal.textures = Textures(verts_rgb=normals_vis)
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = mesh.clone()
                mesh_shading.textures = Textures(verts_rgb=shades)
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode: 
                image_color = self.renderer(mesh)
                results.append(image_color)
            
            if 't' in mode: 
                image_color = self.renderer(mesh)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results


renderer = Renderer(anti_alias=True,image_size=512)


def render_trimesh(mesh, mode='p', renderer_new=None):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    image = renderer.render_mesh(verts, faces, colors=colors, mode=mode)[0]

    # if renderer_new is None:
    #     image = renderer.render_mesh(verts, faces, colors=colors, mode=mode)[0]
    # else:
    #     image = renderer_new.render_mesh(verts, faces, colors=colors, mode=mode)[0]

    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image


def render(verts, faces, colors=None):
    return renderer.render_mesh(verts, faces, colors)

def render_mesh_dict(mesh, mode='npa',render_new=None):
    if render_new is None:
        image = renderer.render_mesh_dict(mesh, mode)[0]
    else:
        image = render_new.render_mesh_dict(mesh, mode)[0]

    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image


def render_pytorch3d(mesh, mode='npa', renderer_new=None):
    if renderer_new is None:
        image = renderer.render_mesh_pytorch(mesh, mode=mode)[0]
    else:
        image = renderer_new.render_mesh_pytorch(mesh, mode=mode)[0]

    image = (255*image).data.cpu().numpy().astype(np.uint8)

    return image

def render_joint(smpl_jnts, bone_ids, image_size=256):
    marker_sz = 6
    line_wd = 2

    image = np.ones((image_size, image_size,4), dtype=np.uint8)*255 
    smpl_jnts[:,1] += 0.3
    smpl_jnts[:,1] = -smpl_jnts[:,1] 
    smpl_jnts = smpl_jnts[:,:2]*image_size/2 + image_size/2

    for b in bone_ids:
        if b[0]<0 : continue
        joint = smpl_jnts[b[0]]
        cv2.circle(image, joint.astype('int32'), color=(0,0,0,255), radius=marker_sz, thickness=-1)

        joint2 = smpl_jnts[b[1]]
        cv2.circle(image, joint2.astype('int32'), color=(0,0,0,255), radius=marker_sz, thickness=-1)

        cv2.line(image, joint2.astype('int32'), joint.astype('int32'), color=(0,0,0,255), thickness=int(line_wd))

    return image


def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

                        
    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]


    if weights.shape[1]> 30:
        colors= np.concatenate([np.array(cmap.colors)]*3)[:33]



    verts_colors = weights[:,:,None] * colors

    verts_colors = verts_colors.sum(1)

    return verts_colors