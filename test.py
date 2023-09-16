"""generate images and videos of AG3D."""

import os
import click
import json
import tempfile
import copy
import torch


import numpy as np
import imageio
from tqdm import trange
import cv2
import PIL
import trimesh
import glob
import dnnlib
import utils.legacy as legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from utils.mesh_renderer import Renderer, render_trimesh
from utils.visualization_utils import gen_samples, gen_novel_view, calculate_rotation, gen_interp, gen_anim
# #----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--truncation', default=0.7, help='truncation_number', required=True)
@click.option('--pose_dist', help='Pose distribution of training dataset', metavar='[ZIP|DIR]')
@click.option('--res', help='Image resolution', type=int, default=256, metavar='INT', show_default=False)
@click.option('--output_path', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--number', help='Number of generation', metavar='INT', default=1, required=False)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid5k_full', show_default=True)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--type', help='Output type', type=str, default='gen_samples', metavar='STR', show_default=True)
@click.option('--motion_path', help='Motion path of animation', type=str, default=None, metavar='STR', show_default=True)
@click.option('--is_img', help='whether output rendered image', type=bool, default=True)
@click.option('--is_normal', help='whether output rendered normal', type=bool, default=False)
@click.option('--is_img_raw', help='whether output raw rendering', type=bool, default=False)
@click.option('--is_mesh', help='whether output rendered mesh', type=bool, default=False)


def visualization(ctx, network, truncation, pose_dist, res, output_path, number, gpus, verbose, metrics, type, motion_path, is_img, is_normal, is_img_raw, is_mesh):
    
    """generate rendering samples.

    Examples:

    \b
    # pretrained mode in ./model, pose distribution in ./data
    python test.py --metrics=fid5k_full --network=./model/deep_fashion.pkl \\
        --pose_dist=./data/dp_pose_dist.npy --output_path './result/result2' --res=512 --truncation=0.7 --number=100 --type=gen_novel_view

    """
    
    dnnlib.util.Logger(should_flush=True)
    device = torch.device('cuda')

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network, verbose=verbose)

    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    from training.triplane import AG3DGenerator

    print("Reloading Modules!")
    G_new = AG3DGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    args.G = G

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network):
        pkl_dir = os.path.dirname(network)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir


    # Configure torch.
    rank = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    G.rendering_kwargs.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': -0.5,
            'ray_end': 0.5,
            'box_warp': 1.9,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
            'is_normal':True,
    })
    
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])


    # dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
    dataset = np.load(pose_dist)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with torch.no_grad():
        
        zs = torch.randn([number, G.z_dim], device=device)
        
        if 'gen_interp' in type:
            
            zs1 = zs[:int(number/2)]
            zs2 = zs[int(number/2):]
            
            for _i in trange(len(zs1)):
                
                n_steps = 8
                z1 = zs1[_i:_i+1]
                z2 = zs2[_i:_i+1]
                c1 = torch.from_numpy(dataset[np.random.randint(len(dataset))]).pin_memory().to(device).unsqueeze(0)
                c2 = torch.from_numpy(dataset[np.random.randint(len(dataset))]).pin_memory().to(device).unsqueeze(0)
                
                save_path = f'{output_path}/{_i:05}_interp.png'
                
                gen_interp(G, z1, z2, c1, c2, truncation, res, is_img, is_img_raw, is_normal, is_mesh, save_path, n_steps)
        
        for _i in trange(len(zs)):
            
            z = zs[_i:_i+1]

            c = torch.from_numpy(dataset[np.random.randint(len(dataset))]).pin_memory().to(device).unsqueeze(0)
            
            if 'gen_samples' in type:
                
                save_path = f'{output_path}/{_i:05}.png'
                gen_samples(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path)
            
            if 'gen_cano' in type:
                
                save_path = f'{output_path}/{_i:05}_cano.png'
                gen_samples(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, cano=True)
                
            
            if 'gen_novel_view' in type:
                
                save_path = f'{output_path}/{_i:05}.mp4'
                full_rotation = False
                
                if full_rotation:
                    
                    angles = np.array(range(0, 361, 6))
                   
                else:
                    rotation_angle = 40
                    spacing = 8
                    angles = calculate_rotation(rotation_angle, spacing)
                
                save_path = f'{output_path}/{_i:05}.mp4'
                
                gen_novel_view(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, angles)
                
            if 'gen_anim' in type:
                
                save_path = f'{output_path}/{_i:05}_anim.mp4'
                gen_anim(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, motion_path)   
            

#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    visualization() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------