# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
import PIL.Image
from torch_utils.misc import crop_face
class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, save_dir = None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.save_dir       = save_dir

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float32)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float32)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]
        
        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
            
        if self.capture_mean_cov:
            x64 = x.astype(np.float32)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus

        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
            
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, face=False, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file_img = None
    opts.cache= True
    
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag_img = f'{dataset.name}_img-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_tag_face = f'{dataset.name}_face-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_tag_normal = f'{dataset.name}_normal-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file_img = dnnlib.make_cache_dir_path('gan-metrics', cache_tag_img + '.pkl')
        
        cache_file_face = dnnlib.make_cache_dir_path('gan-metrics', cache_tag_face + '.pkl')
        cache_file_normal = dnnlib.make_cache_dir_path('gan-metrics', cache_tag_normal + '.pkl')
        
        
        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file_img) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file_img), FeatureStats.load(cache_file_face), FeatureStats.load(cache_file_normal)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats1 = FeatureStats(max_items=num_items, **stats_kwargs)
    stats2 = FeatureStats(max_items=num_items, **stats_kwargs)
    stats3 = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    visualization = True
    save_dataset = True
    image_list = []
    normal_list = []
    face_list = []
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
   
    if opts.save_dir is not None and save_dataset is True:
        
        normal_path = os.path.join(opts.save_dir, 'normal')
        image_path = os.path.join(opts.save_dir, 'image')
        face_path = os.path.join(opts.save_dir, 'face')
        pose_path = os.path.join(opts.save_dir, 'pose')
        os.makedirs(normal_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(face_path, exist_ok=True)
        os.makedirs(pose_path, exist_ok=True)
        image_id = 0  
        
    import random
    random.shuffle(item_subset)
    for images, seg, c in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        face_images = crop_face(images, c.cuda(), size=64).to(torch.uint8)
        
        if seg.shape[1] == 1:
            seg = seg.repeat([1,3,1,1])

        normal_images = torch.nn.functional.interpolate(seg, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
        image_list.append(images)
        face_list.append(face_images)
        
        normal_list.append(normal_images)
        
        features = detector(images.to(opts.device), **detector_kwargs)
        face_features = detector(face_images.to(opts.device), **detector_kwargs)
        normal_features = detector(normal_images.to(opts.device), **detector_kwargs)
        stats1.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        stats2.append_torch(face_features, num_gpus=opts.num_gpus, rank=opts.rank)
        stats3.append_torch(normal_features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats1.num_items)

        
        if visualization is True:
            
            save_image_grid(image_list[0], 'img_real.png', [0,255], [8, 8])   
            save_image_grid(face_list[0], 'face_real.png', [0,255], [8, 8]) 
            save_image_grid(normal_list[0],'normal_real.png', [0,255], [8, 8])   

            visualization = False
    # Save to cache.
    if cache_file_img is not None and opts.rank == 0:
        
        os.makedirs(os.path.dirname(cache_file_img), exist_ok=True)
        temp_file1 = cache_file_img + '.' + uuid.uuid4().hex
        temp_file2 = cache_file_face +'.' + uuid.uuid4().hex
        temp_file3 = cache_file_normal + '.' + uuid.uuid4().hex
        
        stats1.save(temp_file1)
        stats2.save(temp_file2)
        stats3.save(temp_file3)       
        os.replace(temp_file1, cache_file_img) # atomic
        os.replace(temp_file2, cache_file_face) # atomic
        os.replace(temp_file3, cache_file_normal) # atomic
        
        
    return stats1, stats2, stats3

#----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, face=False, **stats_kwargs):
    
    if batch_gen is None:
        batch_gen = min(batch_size, 2)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    stats2 = FeatureStats(**stats_kwargs)
    stats3 = FeatureStats(**stats_kwargs)
    
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    visualization = False
    # Main loop.
    with torch.no_grad():

        while not stats.is_full():

            images = []
            face_images = []
            normal_images = []
            for _i in range(batch_size // batch_gen):
                
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                c = next(c_iter).to(opts.device)
                output = G(z=z, c=c, **opts.G_kwargs)
                img = output['image'].detach()
                normal = output['image_normal'].detach()
            
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
                normal = (normal * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
                face_img = crop_face(img, c.cuda(), size=64).to(torch.uint8).detach()
                
                images.append(img)
                face_images.append(face_img)
                normal_images.append(normal)
            
            if visualization is True:
                if opts.save_dir is None:
                    opts.save_dir = 'test.png'
                images_new = torch.cat(images).cpu().numpy()
                face_imgs_new = torch.cat(face_images).cpu().numpy()
                normals_new = torch.cat(normal_images).cpu().numpy()
                save_image_grid(images_new, 'test_image.png', [0,255], [8, 8])   
                save_image_grid(face_imgs_new, 'test_face.png', [0,255], [8, 8]) 
                save_image_grid(normals_new, 'test_normal.png', [0,255], [8, 8])  
                visualization = False
            
            
            images = torch.cat(images)
            normal_images = torch.cat(normal_images)
            face_images = torch.cat(face_images)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            if face_images.shape[1] == 1:
                face_images = face_images.repeat([1, 3, 1, 1])
            if normal_images.shape[1] == 1:
                normal_images = normal_images.repeat([1, 3, 1, 1])
                

            features_images = detector(images, **detector_kwargs)
            features_normals = detector(normal_images, **detector_kwargs)
            features_faces = detector(face_images, **detector_kwargs)
            stats.append_torch(features_images, num_gpus=opts.num_gpus, rank=opts.rank)
            stats2.append_torch(features_faces, num_gpus=opts.num_gpus, rank=opts.rank)
            stats3.append_torch(features_normals, num_gpus=opts.num_gpus, rank=opts.rank)
            
            progress.update(stats.num_items)
   
    return stats, stats2, stats3

#----------------------------------------------------------------------------
