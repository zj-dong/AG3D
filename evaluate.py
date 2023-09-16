"""evaluate fid of AG3D."""

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
@click.option('--data', help='Training dataset', metavar='[ZIP|DIR]')
@click.option('--res', help='Image resolution', type=int, default=256, metavar='INT', show_default=False)
@click.option('--metrics', help='Quality metrics', type=str, default='fid5k_full', show_default=True)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--save_dirs', help='Print optional information',  default=None, metavar='PATH', required=False)



def evaluation(ctx, network, data, res, gpus, verbose, metrics, save_dirs):
    
    """evaluate fid.

    Examples:

    \b
    # pretrained mode in ./model, data in ./data
    python evaluate.py --metrics=fid5k_full --network=./model/deep_fashion.pkl \\
        --data=./data/eva3d_icon.zip  --res=512 

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
    
    
    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.DeepFashionDataset', path=data)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = res
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)


    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))
    

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
            'ray_start': -0.3,
            'ray_end': 0.3,
            'box_warp': 1.9,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
            'is_normal':True,
    })
    
    if rank == 0 and args.verbose:
        print(f'Calculating {metrics}...')
    progress = metric_utils.ProgressMonitor(verbose=args.verbose)
    result_dict = metric_main.calc_metric(metric=metrics, G=G, dataset_kwargs=args.dataset_kwargs,
        num_gpus=args.num_gpus, rank=rank, device=device, progress=progress, save_dir=save_dirs)
    print(result_dict)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    
    evaluation() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------