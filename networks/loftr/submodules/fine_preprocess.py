import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
    
    def forward_kp(self, feat_f0, feat_f1, kp1, kp2, data):
        """
        kp1, kp2: [N, 2], float coords in (x, y) format, in image space (not normalized)
        feat_f0, feat_f1: [B, C, H, W], input feature maps
        """
        W = self.W  # local window size
        B, C, H, W_feat = feat_f0.shape
        device = feat_f0.device
        N = kp1.shape[0]

        # Generate sampling grids for local window around each kp
        def get_patch_grid(kp, feat_H, feat_W, patch_W):
            # kp: [N, 2], coords in image scale, (x, y)
            # Output: grid [N, patch_W*patch_W, 2] in normalized [-1, 1] coords
            radius = patch_W // 2
            offsets = torch.stack(torch.meshgrid(
                torch.arange(-radius, radius + 1),
                torch.arange(-radius, radius + 1),
                indexing='ij'
            ), dim=-1).to(device)  # [w, w, 2]
            offsets = offsets.view(-1, 2)  # [w*w, 2]

            # Add offset to each keypoint
            kp_expand = kp[:, None, :] + offsets[None, :, :]  # [N, ww, 2]

            # Normalize to [-1, 1] for grid_sample
            norm_grid = kp_expand.clone()
            norm_grid[..., 0] = (norm_grid[..., 0] / (feat_W - 1)) * 2 - 1  # x
            norm_grid[..., 1] = (norm_grid[..., 1] / (feat_H - 1)) * 2 - 1  # y
            grid = norm_grid[..., [1, 0]]  # convert to (y, x) format
            return grid  # [N, ww, 2]

        # Assume input kp1/kp2 are in full-res image coordinates; need to scale down
        scale_x0 = feat_f0.shape[-1] / data['size0'][0,0]
        scale_y0 = feat_f0.shape[-2] / data['size0'][0,1]
        scale_x1 = feat_f1.shape[-1] / data['size1'][0,0]
        scale_y1 = feat_f1.shape[-2] / data['size1'][0,1]

        kp1_feat = kp1.clone()
        kp2_feat = kp2.clone()
        kp1_feat[:, 0] *= scale_x0
        kp1_feat[:, 1] *= scale_y0
        kp2_feat[:, 0] *= scale_x1
        kp2_feat[:, 1] *= scale_y1

        grid1 = get_patch_grid(kp1_feat, feat_f0.shape[2], feat_f0.shape[3], self.W).float()  # [N, ww, 2]
        grid2 = get_patch_grid(kp2_feat, feat_f1.shape[2], feat_f1.shape[3], self.W).float() 

        # Sample features using grid_sample
        feat_f0_unfold = F.grid_sample(feat_f0, grid1[None], align_corners=True, mode='bilinear')  # [B, C, 1, ww]
        feat_f1_unfold = F.grid_sample(feat_f1, grid2[None], align_corners=True, mode='bilinear')
        feat_f0_unfold = feat_f0_unfold.squeeze(0).permute(1, 2, 0)  # [B, ww, C]
        feat_f1_unfold = feat_f1_unfold.squeeze(0).permute(1, 2, 0)

        # If batch size > 1, you might need to gather by indices to match kp's image
        # Here assume kp1/kp2 from same image or batch==1

        return feat_f0_unfold, feat_f1_unfold
