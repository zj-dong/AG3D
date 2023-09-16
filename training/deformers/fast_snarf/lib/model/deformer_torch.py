import torch
from torch import einsum
import torch.nn.functional as F
import os

from torch.utils.cpp_extension import load

import fuse_cuda 
import filter_cuda
import precompute_cuda
import numpy as np

# cuda_dir = os.path.join(os.path.dirname(__file__), "../../cuda")
# fuse_kernel = load(name='fuse_cuda',
#                    extra_cuda_cflags=[],
#                    sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
#                             f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel.cu'])
# filter_cuda = load(name='filter',
#                    sources=[f'{cuda_dir}/filter/filter.cpp',
#                             f'{cuda_dir}/filter/filter_kernel.cu'])

# precompute_cuda = load(name='precompute',
#                    sources=[f'{cuda_dir}/precompute/precompute.cpp',
#                             f'{cuda_dir}/precompute/precompute_kernel.cu'])

class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self,  **kwargs):
        super().__init__()

        self.soft_blend = 20

        # self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]

        self.init_bones = [0, 1, 2, 4, 5, 12, 15, 16, 17, 18, 19]
        
        self.init_bones_cuda = torch.tensor(self.init_bones).int()

        # self.global_scale = 1.5
        
        self.global_scale = 1.2
        

    def forward(self, xd, cond, mask, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """

        xc_opt, others = self.search(xd, cond, mask, tfs, eval_mode=True)


        if eval_mode:
            return xc_opt, others


    def precompute(self, tfs):

        b, c, d, h, w = tfs.shape[0], 3, self.resolution//4, self.resolution, self.resolution
        voxel_d = torch.zeros((b,3,d,h,w), device=tfs.device)
        voxel_J = torch.zeros((b,12,d,h,w), device=tfs.device)
        precompute_cuda.precompute(self.lbs_voxel_final, tfs, voxel_d, voxel_J, self.offset_kernel, self.scale_kernel)
        self.voxel_d = voxel_d
        self.voxel_J = voxel_J

    def search(self, xd, cond, mask, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """

        # reshape to [B,?,D] for other functions

        # run broyden without grad
        with torch.no_grad():
            result = self.broyden_cuda(xd, self.voxel_d, self.voxel_J, tfs, mask)

        return result['result'], result

    def broyden_cuda(self,
                    xd_tgt,
                    voxel,
                    voxel_J_inv,
                    tfs,
                    mask,
                    cvg_thresh=2e-4,
                    dvg_thresh=1):
        """
        Args:
            g:     f: (N, 3, 1) -> (N, 3, 1)
            x:     (N, 3, 1)
            J_inv: (N, 3, 3)
        """
        b,n,_ = xd_tgt.shape
        n_init = self.init_bones_cuda.shape[0]

        xc_init_IN = torch.zeros((b,n,n_init,3),device=xd_tgt.device,dtype=torch.float)

        is_valid = mask.expand(b,n,n_init).clone()

        if self.init_bones_cuda.device != xd_tgt.device:
            self.init_bones_cuda = self.init_bones_cuda.to(xd_tgt.device)
        fuse_cuda.fuse_broyden(xc_init_IN, xd_tgt, voxel, voxel_J_inv, tfs, self.init_bones_cuda, True, is_valid, self.offset_kernel, self.scale_kernel, cvg_thresh, dvg_thresh)

        is_valid_new = torch.zeros_like(is_valid)
        filter_cuda.filter(xc_init_IN, is_valid, is_valid_new)

        return {"result": xc_init_IN, 'valid_ids': is_valid_new} #, 'J_inv': J_inv_init_IN}


    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """

        w = self.query_weights(xc, cond, mask=mask)
        # tfs_mask = tfs[:,None].expand(-1, xc.shape[1], -1, -1, -1)[mask,...]
        xd = skinning_mask(xc[mask,:], w[mask,:], tfs, inverse=False)
        return xd

    def switch_to_explicit(self,resolution=32,smpl_verts=None, smpl_faces=None, smpl_weights=None, use_smpl=False):
        
        self.resolution = resolution
        # convert to voxel grid
        device = self.device
    
        b, c, d, h, w = 1, 24, resolution//4, resolution, resolution
        self.ratio = h/d
        grid = create_voxel_grid(d, h, w, device)

        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, smpl_verts.max(dim=1).values], dim=0).to(device)
        
        offset = (gt_bbox[0] + gt_bbox[1])[None,None,:] * 0.5
        scale = (gt_bbox[1] - gt_bbox[0]).max()/2 * self.global_scale

        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

        voxel_d = torch.zeros( (b,3,d,h,w), device=device)
        voxel_J = torch.zeros( (b,16,d,h,w), device=device)

        self.register_buffer('voxel_d', voxel_d)
        self.register_buffer('voxel_J', voxel_J)
        self.register_buffer('offset_kernel', -self.offset)
        scale_kernel = torch.zeros_like(self.offset)
        scale_kernel[...] = 1./self.scale
        scale_kernel[:,:,-1] = scale_kernel[:,:,-1] * self.ratio
        self.register_buffer('scale_kernel', scale_kernel)
        
        def normalize(x):
            x_normalized = (x+self.offset_kernel)*self.scale_kernel
            return x_normalized

        def denormalize(x):
            x_denormalized = x.clone() #/self.global_scale
            x_denormalized[..., -1] = x_denormalized[..., -1]/self.ratio
            x_denormalized *= self.scale
            x_denormalized += self.offset

            return x_denormalized

        self.normalize = normalize
        self.denormalize = denormalize

        grid_denorm = self.denormalize(grid)

        weights = query_weights_smpl(grid_denorm, smpl_verts=smpl_verts.detach().clone(), smpl_weights=smpl_weights.detach().clone()).detach().clone()
        # sdf_smpl = self.query_sdf_smpl(grid_denorm, smpl_verts=smpl_verts.detach().clone(), smpl_faces=smpl_faces.detach().clone(),smpl_weights=smpl_weights.detach().clone()).detach().clone()

        sdf_smpl = torch.tensor(np.load('./model/sdf_smpl.npy')).to(device)

        sdf_smpl[sdf_smpl.abs() > 0.1] = -100000000

        # np.save('sdf_smpl.npy',sdf_smpl.data.cpu().numpy())

        self.register_buffer('lbs_voxel_final', weights.detach())
        self.register_buffer('grid_denorm',grid_denorm)

        self.register_buffer('sdf_smpl',sdf_smpl)

        def query_weights( xc, cond=None, mask=None, mode='bilinear'):
            w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), self.normalize(xc).unsqueeze(2).unsqueeze(2),align_corners=True, mode=mode,padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
            return w

        def query_sdf( xc, cond=None, mask=None, mode='bilinear'):
            sdf = F.grid_sample(self.sdf_smpl.expand(xc.shape[0],-1,-1,-1,-1), self.normalize(xc).unsqueeze(2).unsqueeze(2),align_corners=True, mode=mode,padding_mode='border')
            sdf = sdf.squeeze(-1).squeeze(-1).permute(0,2,1)
            return sdf
    
        self.query_weights = query_weights
        self.query_sdf = query_sdf

    def update_lbs_voxel(self):
        self.lbs_voxel_final = F.softmax( self.lbs_voxel*20,dim=1)
        def query_weights( xc, cond=None, mask=None, mode='bilinear'):
            w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), self.normalize(xc).unsqueeze(2).unsqueeze(2),align_corners=True, mode=mode,padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
            return w

        self.query_weights = query_weights


    def query_sdf_smpl(self, x, smpl_verts, smpl_faces, smpl_weights):
        
        device = x.device

        resolution=128
        b, c, d, h, w = 1, 24, resolution//4, resolution, resolution
        grid = create_voxel_grid(d, h, w, device)
        grid = self.denormalize(grid)

        import trimesh
        mesh = trimesh.Trimesh(vertices=smpl_verts.data.cpu().numpy()[0], faces=smpl_faces.data.cpu().numpy())
        BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
    
        sdf, face_id, uvw = BVH.signed_distance(grid, return_uvw=True, mode='watertight') # [N], [N], [N, 3]

        sdf = sdf.reshape(1,-1,1)
        b, c, d, h, w = 1, 1, resolution//4, resolution, resolution

        sdf = -sdf.permute(0,2,1).reshape(b,c,d,h,w)

        return sdf.detach()

    def skinning_normal(self, xc, normal, tfs, cond=None, mask=None, inverse=False):
        ''' skinning normals
        
        Args:
            x (tensor): canonical points. shape: [B, N, D]
            normal (tensor): canonical normals. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            posed normal (tensor): posed normals. shape: [B, N, D]
            
        '''
        if xc.ndim == 2:
            xc = xc.unsqueeze(0)
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc, cond, mask=mask)
        p_h = F.pad(normal, (0, 1), value=0)
        p_h = torch.einsum('bpn, bnij, bpj->bpi', w, tfs, p_h)

        return p_h[:, :, :3]
    
def skinning_mask(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1

        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (fast_inverse(w_tf)*x_h).sum(-1)

    else:
        w_tf = einsum("pn,nij->pij", w, tfs.squeeze(0))

        x_h = x_h.view(p,1,4).expand(p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :3]

def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    b,p,n = w.shape

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = einsum("bpn,bnij->bpij", w, fast_inverse(tfs))

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        # x_h = (fast_inverse(w_tf)*x_h).sum(-1)
        x_h = (w_tf*x_h).sum(-1)

    else:
        w_tf = einsum("bpn,bnij->bpij", w, tfs)

        x_h = x_h.view(b,p,1,4).expand(b,p,4,4)
        x_h = (w_tf*x_h).sum(-1)

    return x_h[:, :, :3]

def fast_inverse(T):

    shape = T.shape

    T = T.reshape(-1,4,4)
    R = T[:, :3,:3]
    t = T[:, :3,3].unsqueeze(-1)

    R_inv = R.transpose(1,2)
    t_inv = -bmv(R_inv,t)

    T_inv = T
    T_inv[:,:3,:3] = R_inv
    T_inv[:,:3,3] = t_inv.squeeze(-1)
    
    return T_inv.reshape(shape)

def bmv(m, v):
    return (m*v.transpose(-1,-2).expand(-1,3,-1)).sum(-1,keepdim=True)

# def query_weights_smpl(x, smpl_verts, smpl_weights):
    
#     import pytorch3d.ops as ops
#     device = x.device
   
#     distance_batch, index_batch, neighbor_points  = ops.knn_points(x,smpl_verts.detach(),K=1,return_nn=True)

#     index_batch = index_batch[0]

#     skinning_weights = smpl_weights[:,index_batch][:,:,0,:].detach()

#     resolution = 64

#     b, c, d, h, w = 1, 24, resolution//4, resolution, resolution

#     skinning_weights = skinning_weights.permute(0,2,1).reshape(b,c,d,h,w)


#     return skinning_weights.detach()



def create_voxel_grid(d, h, w, device='cuda'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid


def query_weights_smpl(x, smpl_verts, smpl_weights):
    import pytorch3d.ops as ops

    device = smpl_weights.device
    distance_batch, index_batch, neighbor_points  = ops.knn_points(x.to(device),smpl_verts.to(device).detach(),K=10,return_nn=True)

    # neighbor_points = neighbor_points[0]
    distance_batch = distance_batch[0].sqrt().clamp_(0.00003,0.1)
    index_batch = index_batch[0]
    
    # GPU_id = index_batch.get_device()
    # print(GPU_id)
    weights = smpl_weights[0,index_batch]

    # blendshapes = smpl_blendshapes[index_batch].reshape(-1,1,30)
   
    ws=1./distance_batch
    ws=ws/ws.sum(-1,keepdim=True)
    weights = (ws[:,:,None]*weights).sum(1)[None]

    # blendshapes = (ws[:,:,None]*blendshapes).sum(1)[None]

    resolution = 64

    b, c, d, h, w = 1, 24, resolution//4, resolution, resolution
    weights = weights.permute(0,2,1).reshape(b,c,d,h,w)

    # blendshapes = blendshapes.permute(0,2,1).reshape(b,30,d,h,w)

    # weights = F.avg_pool3d(weights, kernel_size=3, stride=1, padding=1)

    # for _ in range(30):
    #     mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
    #         weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
    #         weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0

    #     weights[:,:,1:-1,1:-1,1:-1]=(weights[:,:,1:-1,1:-1,1:-1]-mean)*0.7+mean
    #     sums=weights.sum(1,keepdim=True)
    #     weights=weights/sums

    return weights.detach()#, blendshapes.detach()
