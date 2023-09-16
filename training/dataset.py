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

"""Streaming images and labels from datasets created with dataset_tool.py."""



import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None


label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


label_merger = {(0,255,255):(51,170,221),
                (170,255,85): (85,255,170),
                (255,170,0): (255,255,0),
                (0,119,221): (0,0,85),
                (0,128,0): (0,0,85),
                (255,85,0): (0,0,85)
                }
#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        
        if self._raw_labels is None:

            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
            
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        
        image = self._load_raw_image(self._raw_idx[idx])
        seg = self._load_raw_seg(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            seg = seg[:, :, ::-1]
        return image.copy(), seg.copy(), self.get_label(idx)

    def get_label(self, idx):
        
        label = self._get_raw_labels()[self._raw_idx[idx]]
        pose_list = self._get_raw_labels()

        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class DeepFashionDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set([name for name in self._get_zipfile().namelist() if '_seg' not in name])
        else:
            raise IOError('Path must point to a directory or zip')
        
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION and 'img' in fname)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self._resolution = resolution

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)


    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        
        
        image, normal = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            normal = normal[:, :, ::-1]
            # seg = seg[:, :, ::-1]
        return image.copy(), normal.copy(), self.get_label(idx)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:

            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        
        fname = self._image_fnames[raw_idx].replace('img','normal')
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                normal = pyspng.load(f.read())
            else:
                normal = np.array(PIL.Image.open(f))

        normal_mask = ((normal==255)).all(axis = -1)  #((normal==127) | (normal==128 )).all(axis = -1) 
        normal = (normal / 255.0) * 2 - 1
        normal = normal / (np.linalg.norm(normal, axis=-1,keepdims=True)+1e-8)
        normal = (normal + 1) * 127.5
        normal[normal_mask,:] = 255

        if image.shape[2] == 4:
            image_mask = image[:,:,-1]<0.3*255
            image_alpha = image[:,:,-1]/255.0
            image = image[:,:,:-1]
            normal[image_mask,:] = 255

            image = image * image_alpha[...,None] + 255. * (1-image_alpha[...,None])
            image = image.astype(np.uint8)
            

        if self._resolution and image.shape[0] != self._resolution:
            image = cv2.resize(image, (self._resolution, self._resolution))
        if self._resolution and normal.shape[0] != self._resolution:
            normal = cv2.resize(normal, (self._resolution, self._resolution))

        image = image.transpose(2, 0, 1) # HWC => CHW
        normal = normal.transpose(2, 0, 1) # HWC => CHW

        return image, normal

    def _load_raw_normal(self,raw_idx):
        
        fname = self._image_fnames[raw_idx].replace('img','normal')
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                normal = pyspng.load(f.read())
            else:
                normal = np.array(PIL.Image.open(f))
        if normal.ndim == 2:
            normal = normal[:, :, np.newaxis] # HW => HWC
        
        mask = (normal==255).all(axis = -1)
        normal = (normal / 255.0) * 2 - 1
        
        normal = normal / (np.linalg.norm(normal, axis=-1,keepdims=True)+1e-8)
        normal[mask,:] = 1.0

        normal = (normal + 1) * 127.5

        if self._resolution and normal.shape[0] != self._resolution:
            normal = cv2.resize(normal, (self._resolution, self._resolution))

        normal = normal.transpose(2, 0, 1) # HWC => CHW

        return normal
        
    def _load_raw_seg(self, raw_idx):
        fname = self._image_fnames[raw_idx].replace('.png', '_seg.png')
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        # merge semantic labels
        
        for color_orig in label_merger:
            indices = np.where(np.all(image == color_orig, axis=-1))
            image[indices] = label_merger[color_orig]
        if self._resolution and image.shape[0] != self._resolution:
            image = cv2.resize(image, (self._resolution, self._resolution))
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        
        fname = 'dataset.json'
        
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
 
        labels = dict(labels)
        
        camera_poses = [labels[fname.replace('\\', '/')][0] for fname in self._image_fnames]
        smpl_params = [labels[fname.replace('\\', '/')][1] for fname in self._image_fnames]

        camera_poses = np.array(camera_poses)
        smpl_params = np.array(smpl_params)
        labels = np.concatenate([camera_poses, smpl_params], axis=1)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
 
        return labels