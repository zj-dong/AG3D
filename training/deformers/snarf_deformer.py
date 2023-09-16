'''
Deformer of AG3D adapted from fast-SNARF
'''

from .fast_snarf.lib.model.deformer_torch import ForwardDeformer, skinning
from .smplx import SMPL
import torch
from pytorch3d import ops
import numpy as np

class SNARFDeformer():
    
    def __init__(self, gender) -> None:
        self.body_model = SMPL('./training/deformers/smplx/SMPLX', gender=gender)
        self.deformer = ForwardDeformer()
        
        # threshold for rendering (need to be larger for loose clothing)
        self.threshold = 0.12
        
        self.initialized = False

    def initialize(self, betas):
        
        device = betas.device
        batch_size = 1
        
        # canonical space is defined in t-pose
        body_pose_t = torch.zeros((batch_size, 69), device=device)
        transl = torch.zeros((batch_size, 3), device=device)
        transl[:,1] = 0.3
        global_orient = torch.zeros((batch_size, 3), device=device)
 
        smpl_outputs = self.body_model(betas=betas[:1]*0, body_pose=body_pose_t, transl=transl, global_orient = global_orient)
        self.tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        self.vs_template = smpl_outputs.vertices
        self.smpl_faces = torch.tensor(self.body_model.faces.astype(np.int64), device=device)

        # initialize SNARF
        self.deformer.device = device
        smpl_verts = smpl_outputs.vertices.float().detach().clone()
        #TODO: add batch operation
        smpl_verts = smpl_verts[0][None,:,:]

        self.deformer.switch_to_explicit(resolution=64,
                                         smpl_verts=smpl_verts,
                                         smpl_faces=self.smpl_faces,
                                         smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
                                         use_smpl=True)

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)
        self.deformer.voxel_d = self.deformer.voxel_d.type(self.dtype)
        self.deformer.voxel_J = self.deformer.voxel_J.type(self.dtype)

    def prepare_deformer(self, smpl_params=None):
        
        if smpl_params is None:
            
            pose = torch.zeros((1,69)).cuda()
            pose[:, 2] = torch.pi / 6
            pose[:, 5] = -torch.pi / 6

            smpl_params = {
            'betas': torch.zeros((1,10)).cuda(),
            'body_pose': pose,
            'global_orient': torch.zeros((1,3)).cuda(),
            'transl': torch.zeros((1,)).cuda(),
            }
            
        
        device = smpl_params["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
        
        
        if not self.initialized:
            self.initialize(smpl_params["betas"])
            self.initialized = True
    
        
        smpl_outputs = self.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"],
                                       scale = smpl_params["scale"] if "scale" in smpl_params.keys() else None)
        
        self.smpl_outputs = smpl_outputs

        tfs = (smpl_outputs.A @ self.tfs_inv_t.expand(smpl_outputs.A.shape[0],-1,-1,-1))
        self.deformer.precompute(tfs)
        self.tfs = tfs


    def __call__(self, pts, model, eval_mode=True, render_skinning=False, is_normal=True):

        b, n, _ = pts.shape

        dist_sq, idx, neighbors = ops.knn_points(pts.float(), self.smpl_outputs.vertices.float()[:,::10], K=1)
        mask = dist_sq < self.threshold ** 2

        smpl_nn = False

        if smpl_nn:
            # deformer based on SMPL nearest neighbor search
            k = 3
            dist_sq, idx, neighbors = ops.knn_points(pts,self.smpl_outputs.vertices.float(),K=k,return_nn=True)
            
            dist = dist_sq[0].sqrt().clamp_(0.00001,1.)
            idx = idx[0]
            weights = self.body_model.lbs_weights.clone()[idx]
            mask = dist_sq < 0.02

            ws=1./dist
            ws=ws/ws.sum(-1,keepdim=True)

            weights = (ws[:,:,None].expand(-1,-1,24)*weights).sum(1)[None]

            pts_cano_all = skinning(pts, weights, self.tfs, inverse=True)
            pts_cano_all = pts_cano_all.unsqueeze(2).expand(-1,-1,3,-1)
            others = {"valid_ids": torch.zeros_like(pts_cano_all)[...,0]}
            others["valid_ids"][:,:,0] = mask[...,0]
            others["valid_ids"] = others["valid_ids"].bool()
            
        else:
            # defromer based on fast-SNARF
            with torch.no_grad():
                pts_cano_all, others = self.deformer.forward(pts, cond=None, mask=mask, tfs=self.tfs, eval_mode=eval_mode)

        pts_cano_all = pts_cano_all.reshape(b, n, -1, 3)
        valid = others["valid_ids"].reshape(b, n, -1)

        rgb_cano, sigma_cano, grad_cano, grad_pred_cano = model(pts_cano_all, valid)

        sigma_cano, idx = torch.max(sigma_cano.squeeze(-1), dim=-1)

        pts_cano = torch.gather(pts_cano_all, 2, idx[:, :, None, None].repeat(1,1,1,pts_cano_all.shape[-1]))
        rgb_cano = torch.gather(rgb_cano, 2, idx[:, :, None, None].repeat(1,1,1,rgb_cano.shape[-1]))
        if is_normal:
            grad_cano = torch.gather(grad_cano, 2, idx[:, :, None, None].repeat(1,1,1,grad_cano.shape[-1]))
            grad = self.deformer.skinning_normal(pts_cano.squeeze(2), grad_cano.squeeze(2), self.tfs)
            grad_pred_cano = torch.gather(grad_pred_cano, 2, idx[:, :, None, None].repeat(1,1,1,grad_cano.shape[-1]))
        else:
            grad = None
            grad_pred_cano = None
            
        return rgb_cano, sigma_cano.unsqueeze(-1), grad, grad_pred_cano
