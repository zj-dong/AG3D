# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Modified by Zijian Dong for AG3D: Learning to Generate 3D Avatars from 2D Image Collections

"""Generator architectures from the paper
"AG3D: Learning to Generate 3D Avatars from 2D Image Collections"

Code adapted from
"Efficient Geometry-aware 3D Generative Adversarial Networks."""

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
import numpy as np
from tqdm import tqdm
import trimesh

@persistence.persistent_class
class AG3DGenerator(torch.nn.Module):
    
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        is_sr_module = False,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)

        self.neural_rendering_resolution = img_resolution//2
        
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_params=None, cano=False, **synthesis_kwargs):
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)        

        if not cano:
            smpl_params = {
            'betas': c[:,101:],
            'body_pose': c[:,32:101],
            'global_orient': c[:,29:32],
            'transl': c[:,26:29],
            'scale':c[:, 25]
            }
        else:
            smpl_params = None
   
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
    
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, patch_params=patch_params)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, grad_samples, grad_cano_samples, sdf_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, smpl_params, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = neural_rendering_resolution
        
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W)
    
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if self.rendering_kwargs['is_normal']:
            grad_image = grad_samples.permute(0, 2, 1).reshape(N, grad_samples.shape[-1], H, W)
        else:
            grad_image = None

        # Run superresolution to get final image
        rgb_image = feature_image[:, :self.img_channels]

        sr_image = self.superresolution(rgb_image.clone(), feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image':            sr_image, 
                'image_raw':        rgb_image, 
                'image_normal':     grad_image,
                'image_depth':      depth_image,
                'grad_cano':        grad_cano_samples,
                'sdf':              sdf_samples}
                
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, smpl_params=None, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs,smpl_params=smpl_params)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_params=None, cano=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, patch_params=patch_params, cano=cano, **synthesis_kwargs)
    
    def get_mesh(self, z, c, ws=None, voxel_resolution=256, truncation_psi=1, truncation_cutoff=None, update_emas=False, canonical=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        device = z.device

        if not canonical:
            smpl_params = {
                'betas': c[:,101:],
                'body_pose': c[:,32:101],
                'global_orient': c[:,29:32],
                'transl': c[:,26:29],
                'scale':c[:, 25]
                }
        else:
            smpl_params = {
                'betas': c[:,101:]*0,
                'body_pose': c[:,32:101]*0,
                'global_orient': c[:,29:32]*0,
                'transl': c[:,26:29]*0,
                'scale': c[:, 25]*0+1
                }
            smpl_params['transl'][:,1] = 0.3

        if ws is None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        

        smpl_outputs = self.renderer.deformer.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"],
                                       scale = smpl_params["scale"])
        
        face = self.renderer.deformer.smpl_faces
        smpl_verts = smpl_outputs.vertices.float()[0]

        scale = 1.1  # Scale of the padded bbox regarding the tight one.
        verts = smpl_verts.data.cpu().numpy()
        gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
        gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
        gt_scale = (gt_bbox[1] - gt_bbox[0]).max()


        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, smpl_verts=smpl_verts)
        samples = samples.to(device)

        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)

        head = 0
        max_batch = int(1e7)
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.sample_mixed(samples[:, head:head+max_batch], samples[:, head:head+max_batch], ws, truncation_psi=truncation_psi, noise_mode='const', smpl_params=smpl_params)['sdf']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        from utils.shape_utils import convert_sdf_samples_to_ply
        mesh = convert_sdf_samples_to_ply(sigmas, [0, 0, 0], 1, level=0)

        verts = mesh.vertices
        verts = (verts / voxel_resolution - 0.5) * scale
        verts = verts * gt_scale + gt_center

        verts_torch = torch.from_numpy(verts).float().to(device).unsqueeze(0)

        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
  
        weights = self.renderer.deformer.deformer.query_weights(verts_torch).clamp(0,1)[0]
       
        mesh.visual.vertex_colors[:,:3] = weights2colors(weights.data.cpu().numpy()*0.999)*255
        return mesh, weights

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        if sampled_features.ndim == 2:
            x = sampled_features
        else:
            x = sampled_features.flatten(0,1)

        x = self.net(x)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


def create_samples(N=256, smpl_verts=None, cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    verts = smpl_verts.data.cpu().numpy()
    scale = 1.1
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples = (samples / N - 0.5) * scale
    samples = samples * gt_scale + gt_center

    num_samples = N ** 3

    return samples.unsqueeze(0), None, None



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
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors
