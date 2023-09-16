# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Occupancy Networks
#
# Copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import torch
from torch.utils.cpp_extension import load
sources = ['training/check_sign/mesh_intersection_cuda.cpp','training/check_sign/mesh_intersection_cuda_kernel.cu']
# sources = [hydra.utils.to_absolute_path(s) for s in sources]
mint = load(name='mesh_intersection_cuda',sources=sources)

def _unbatched_check_sign_cuda(verts, faces, points):
    n, _ = points.size()
    points = points.contiguous()
    v1 = torch.index_select(verts, 0, faces[:, 0]).view(-1, 3).contiguous()
    v2 = torch.index_select(verts, 0, faces[:, 1]).view(-1, 3).contiguous()
    v3 = torch.index_select(verts, 0, faces[:, 2]).view(-1, 3).contiguous()

    ints = torch.zeros(n, device=points.device)
    mint.forward_cuda(points, v1, v2, v3, ints)
    contains = ints % 2 == 1

    return contains


def check_sign(verts, faces, points, hash_resolution=512):
    r"""Checks if a set of points is contained inside a mesh. 

    Each batch takes in v vertices, f faces of a watertight trimesh, 
    and p points to check if they are inside the mesh. 
    Shoots a ray from each point to be checked
    and calculates the number of intersections 
    between the ray and triangles in the mesh. 
    Uses the parity of the number of intersections
    to determine if the point is inside the mesh. 

    Args:
        verts (torch.Tensor): vertices of shape (batch_size, num_vertices, 3)
        faces (torch.Tensor): faces of shape (num_faces, 3)
        points (torch.Tensor): points of shape (batch_size, num_points, 3) to check
        hash_resolution (int): resolution used to check the points sign

    Returns:
        (torch.BoolTensor): 
            length p tensor indicating whether each point is 
            inside the mesh 

    Example:
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> verts = torch.tensor([[[0., 0., 0.],
        ...                       [1., 0.5, 1.],
        ...                       [0.5, 1., 1.],
        ...                       [1., 1., 0.5]]], device = device)
        >>> faces = torch.tensor([[0, 3, 1],
        ...                       [0, 1, 2],
        ...                       [0, 2, 3],
        ...                       [3, 2, 1]], device = device)
        >>> axis = torch.linspace(0.1, 0.9, 3, device = device)
        >>> p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
        >>> points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
        >>> points = points.view(1, -1, 3)
        >>> check_sign(verts, faces, points)
        tensor([[ True, False, False, False, False, False, False, False, False, False,
                 False, False, False,  True, False, False, False,  True, False, False,
                 False, False, False,  True, False,  True, False]], device='cuda:0')
    """
    assert verts.device == points.device
    assert faces.device == points.device
    device = points.device

    if not verts.dtype == torch.float32:
        raise TypeError(f"Expected verts entries to be torch.float32 "
                        f"but got {verts.dtype}.")
    if not faces.dtype == torch.int64:
        raise TypeError(f"Expected faces entries to be torch.int64 "
                        f"but got {faces.dtype}.")
    if not points.dtype == torch.float32:
        raise TypeError(f"Expected points entries to be torch.float32 "
                        f"but got {points.dtype}.")
    if not isinstance(hash_resolution, int):
        raise TypeError(f"Expected hash_resolution to be int "
                        f"but got {type(hash_resolution)}.")

    if verts.ndim != 3:
        verts_dim = verts.ndim
        raise ValueError(f"Expected verts to have 3 dimensions " 
                         f"but got {verts_dim} dimensions.")
    if faces.ndim != 2:
        faces_dim = faces.ndim
        raise ValueError(f"Expected faces to have 2 dimensions " 
                         f"but got {faces_dim} dimensions.")
    if points.ndim != 3:
        points_dim = points.ndim
        raise ValueError(f"Expected points to have 3 dimensions " 
                         f"but got {points_dim} dimensions.")

    if verts.shape[2] != 3:
        raise ValueError(f"Expected verts to have 3 coordinates "
                         f"but got {verts.shape[2]} coordinates.")
    if faces.shape[1] != 3:
        raise ValueError(f"Expected faces to have 3 vertices "
                         f"but got {faces.shape[1]} vertices.")
    if points.shape[2] != 3:
        raise ValueError(f"Expected points to have 3 coordinates "
                         f"but got {points.shape[2]} coordinates.")

    xlen = verts[..., 0].max(-1)[0] - verts[..., 0].min(-1)[0]
    ylen = verts[..., 1].max(-1)[0] - verts[..., 1].min(-1)[0]
    zlen = verts[..., 2].max(-1)[0] - verts[..., 2].min(-1)[0]
    maxlen = torch.max(torch.stack([xlen, ylen, zlen]), 0)[0]
    verts = verts / maxlen.view(-1, 1, 1)
    points = points / maxlen.view(-1, 1, 1)

    results = []
    if device.type == 'cuda':
        for i_batch in range(verts.shape[0]):
            contains = _unbatched_check_sign_cuda(verts[i_batch], faces, points[i_batch])
            results.append(contains)
    else:
        for i_batch in range(verts.shape[0]):
            intersector = _UnbatchedMeshIntersector(verts[i_batch], faces, hash_resolution)
            contains = intersector.query(points[i_batch].data.cpu().numpy())
            results.append(torch.tensor(contains).to(device))

    return torch.stack(results)
