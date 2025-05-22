# 添加到main_test_lightglue.py中
import torch.optim as optim
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import copy
import os
import tqdm
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import kornia.augmentation as K
from kornia.geometry.transform import warp_perspective
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
import cv2
from PIL import Image
import matplotlib
from losses import NLLLoss
matplotlib.use('Agg')  # 无界面后端，适合在服务器上运行

IMAGE_SIZE = (1024,1024)
kaggle_run = True

def create_spliced_negative_pair(image_paths, img_idx, splice_ratio_range=(0.3, 0.5)):
    """
    创建拼接负样本对 - 将当前图像与另一张随机图像拼接
    整个拼接图像被视为完全不匹配对，拼接区域占原图30%-50%
    
    Args:
        image_paths: 所有图像路径列表
        img_idx: 当前图像索引
        splice_ratio_range: 拼接区域占比范围，默认30%-50%
        
    Returns:
        img1: 原始图像
        img2: 拼接后的图像
        transform: 恒等变换矩阵
    """
    # 随机选择一张不同的图像进行拼接
    other_indices = [i for i in range(len(image_paths)) if i != img_idx]
    if not other_indices:  # 防止只有一张图片的情况
        other_idx = img_idx
    else:
        other_idx = random.choice(other_indices)
    
    # 加载图像
    img1 = load_torch_image(image_paths[img_idx])
    other_img = load_torch_image(image_paths[other_idx])
    
    # 获取图像尺寸
    C, H, W = img1.shape
    
    # 随机决定拼接方向：水平或垂直
    splice_direction = random.choice(['horizontal', 'vertical'])
    
    # 随机决定拼接区域大小（在给定范围内）
    splice_ratio = random.uniform(splice_ratio_range[0], splice_ratio_range[1])
    
    # 创建拼接图像
    img2 = img1.clone()
    
    # 边缘平滑过渡区域宽度
    blend_width = int(min(H, W) * 0.05)  # 5%的图像尺寸作为过渡带
    
    if splice_direction == 'horizontal':
        # 水平拼接：右侧部分替换
        splice_width = int(W * splice_ratio)
        splice_start = W - splice_width
        
        # 创建过渡掩码
        blend_mask = torch.zeros((H, W), device=img1.device)
        for i in range(blend_width):
            # 在接缝处创建渐变过渡带
            pos = splice_start + i - blend_width
            if 0 <= pos < W:
                blend_mask[:, pos] = i / blend_width
        
        # 在接缝右侧全部替换为其他图像
        blend_mask[:, splice_start:] = 1.0
        
        # 将掩码扩展到所有通道
        blend_mask = blend_mask.unsqueeze(0).expand(C, -1, -1)
        
        # 使用blend_mask混合两张图像
        img2 = img1 * (1 - blend_mask) + other_img * blend_mask
        
    else:  # vertical
        # 垂直拼接：下部分替换
        splice_height = int(H * splice_ratio)
        splice_start = H - splice_height
        
        # 创建过渡掩码
        blend_mask = torch.zeros((H, W), device=img1.device)
        for i in range(blend_width):
            # 在接缝处创建渐变过渡带
            pos = splice_start + i - blend_width
            if 0 <= pos < H:
                blend_mask[pos, :] = i / blend_width
        
        # 在接缝下方全部替换为其他图像
        blend_mask[splice_start:, :] = 1.0
        
        # 将掩码扩展到所有通道
        blend_mask = blend_mask.unsqueeze(0).expand(C, -1, -1)
        
        # 使用blend_mask混合两张图像
        img2 = img1 * (1 - blend_mask) + other_img * blend_mask
    
    # 创建恒等变换矩阵
    transform = torch.eye(3)
    
    return img1, img2, transform

class AugmentedImagePairDataset(Dataset):
    def __init__(self, image_paths, size=IMAGE_SIZE,num_pairs_per_image=5, color_change_prob=0.3, copy_paste_prob=0, negative_sample_prob=0):
        self.image_paths = image_paths
        self.num_pairs_per_image = num_pairs_per_image
        self.color_change_prob = color_change_prob
        self.copy_paste_prob = copy_paste_prob  # 新增复制粘贴概率参数
        self.negative_sample_prob = negative_sample_prob  # 新增负样本概率参数
        self.image_size = size
        # 基础变换
        self.color_jitter = K.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0)
        self.perspective = K.RandomPerspective(0.4, p=0.7)
        self.affine = K.RandomAffine(degrees=180, translate=(0, 0), scale=(0.95, 1.2), p=0.7)
        self.blur = K.RandomGaussianBlur((5, 9), (0.1, 2.0), p=0.5)
        
        # 添加区域颜色变换
        self.hue_shift = K.RandomChannelShuffle(p=1.0)  # 通道随机打乱
        
    def apply_regional_color_change(self, img, mask_size=(100, 100)):
        """在图像的随机区域应用颜色变换"""
        H, W = img.shape[-2:]
        # 随机选择区域位置
        x = random.randint(0, W - mask_size[0])
        y = random.randint(0, H - mask_size[1])
        
        # 创建mask
        mask = torch.zeros((1, H, W), device=img.device)
        mask[:, y:y+mask_size[1], x:x+mask_size[0]] = 1.0
        
        # 对选定区域应用颜色变换
        changed_region = self.hue_shift(img[:, :, y:y+mask_size[1], x:x+mask_size[0]])
        img = img.clone()
        img[:, :, y:y+mask_size[1], x:x+mask_size[0]] = changed_region
        
        return img, mask

    def apply_copy_paste(self, img, original_path, mask_size=(400, 400)):
        """
        在图像中随机选择一块区域，复制并应用变换后粘贴到另一位置
        考虑图像的有效区域大小，根据原始图像尺寸调整复制区域大小
        
        Args:
            img: 输入图像张量
            mask_size: 默认掩码大小
            original_path: 原始图像路径，用于获取原始尺寸
        
        Returns:
            修改后的图像，源区域掩码，目标区域掩码
        """
        H, W = img.shape[-2:]
        
        # 获取原始图像的有效区域
        effective_area = None
        
        if original_path is not None:
            try:
                # 打开原始图像获取尺寸
                original_img = Image.open(original_path)
                orig_width, orig_height = original_img.size
                
                # 计算有效区域占总区域的比例
                area_ratio = (orig_width * orig_height) / (H * W)
                
                # 如果原始图像小于目标尺寸，调整复制区域大小
                if area_ratio < 0.8:  # 如果有效区域小于总区域的80%
                    # 根据有效区域比例缩小复制区域
                    adjusted_mask_w = int(mask_size[0] * min(1.0, (orig_width / W) * 1.5))
                    adjusted_mask_h = int(mask_size[1] * min(1.0, (orig_height / H) * 1.5))
                    
                    # 确保区域不会太小
                    mask_w = max(50, min(adjusted_mask_w, W // 3))
                    mask_h = max(50, min(adjusted_mask_h, H // 3))
                    
                    # 如果有效区域太小，使复制区域更小
                    if area_ratio < 0.5:
                        mask_w = min(mask_w, W // 4)
                        mask_h = min(mask_h, H // 4)
                    
                    # 调整复制区域的最大范围为有效区域内
                    effective_area = (
                        max(0, (W - orig_width) // 2),
                        max(0, (H - orig_height) // 2),
                        min(W, (W + orig_width) // 2),
                        min(H, (H + orig_height) // 2)
                    )
                else:
                    # 原始图像接近目标尺寸，使用标准设置
                    mask_w = min(mask_size[0], W // 3)
                    mask_h = min(mask_size[1], H // 3)
            except Exception as e:
                # 如果出现问题，回退到默认值
                mask_w = min(mask_size[0], W // 3)
                mask_h = min(mask_size[1], H // 3)
        else:
            # 没有提供原始路径，使用默认值
            mask_w = min(mask_size[0], W // 3)
            mask_h = min(mask_size[1], H // 3)
        
        # 确定源区域位置（考虑有效区域）
        if effective_area is not None:
            x_min, y_min, x_max, y_max = effective_area
            src_x = random.randint(x_min, max(x_min, x_max - mask_w))
            src_y = random.randint(y_min, max(y_min, y_max - mask_h))
        else:
            src_x = random.randint(0, W - mask_w)
            src_y = random.randint(0, H - mask_h)
        
        # 确定目标区域位置（也考虑有效区域）
        if effective_area is not None:
            x_min, y_min, x_max, y_max = effective_area
            dst_x = random.randint(x_min, max(x_min, x_max - mask_w))
            dst_y = random.randint(y_min, max(y_min, y_max - mask_h))
        else:
            dst_x = random.randint(0, W - mask_w)
            dst_y = random.randint(0, H - mask_h)
        
        # 创建源区域和目标区域掩码
        src_mask = torch.zeros((1, H, W), device=img.device)
        dst_mask = torch.zeros((1, H, W), device=img.device)
        
        src_mask[:, src_y:src_y+mask_h, src_x:src_x+mask_w] = 1.0
        dst_mask[:, dst_y:dst_y+mask_h, dst_x:dst_x+mask_w] = 1.0
        
        # 复制源区域并应用变换
        img_clone = img.clone()
        copied_region = img[:, :, src_y:src_y+mask_h, src_x:src_x+mask_w].clone()
        
        # 对复制区域应用随机变换以增加差异性（代码保持不变）
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            if angle == 90:
                copied_region = copied_region.flip(3).transpose(2, 3)
            elif angle == 180:
                copied_region = copied_region.flip(2).flip(3)
            elif angle == 270:
                copied_region = copied_region.flip(2).transpose(2, 3)
        
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            copied_region = copied_region * brightness_factor
            copied_region = torch.clamp(copied_region, 0, 1)
        
        # 粘贴到目标区域（代码保持不变）
        try:
            c_h, c_w = copied_region.shape[2], copied_region.shape[3]
            d_h, d_w = min(c_h, mask_h), min(c_w, mask_w)
            
            img_clone[:, :, dst_y:dst_y+d_h, dst_x:dst_x+d_w] = copied_region[:, :, :d_h, :d_w]
            
            dst_mask = torch.zeros((1, H, W), device=img.device)
            dst_mask[:, dst_y:dst_y+d_h, dst_x:dst_x+d_w] = 1.0
        except Exception as e:
            img_clone[:, :, dst_y:dst_y+mask_h, dst_x:dst_x+mask_w] = copied_region
        
        return img_clone, src_mask, dst_mask
        
    def __len__(self):
        return len(self.image_paths) * self.num_pairs_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_pairs_per_image
        img_path = self.image_paths[img_idx]
        
                # 决定是否生成拼接负样本
        generate_negative = random.random() < self.negative_sample_prob
        
        is_negative_pair = False
        if generate_negative:
            # 创建拼接负样本
            img1, img2, transform_matrix = create_spliced_negative_pair(self.image_paths, img_idx)
            is_negative_pair = True
            # return {
            #     'image0': img1,
            #     'image1': img2, 
            #     'path': img_path,
            #     'original_idx': img_idx,
            #     'pair_idx': idx % self.num_pairs_per_image,
            #     'transform0': transform_matrix,
            #     'transform1': transform_matrix,
            #     'is_negative_pair': True
            # }
        else:
            # 加载原始图像
            original_img = load_torch_image(img_path, self.image_size)
            img1 = original_img.clone()
            img2 = original_img.clone()
            
        with torch.no_grad():
            img1_batch = img1.unsqueeze(0)
            img2_batch = img2.unsqueeze(0)
            
            # 应用基础变换
            # img1_batch = self.color_jitter(img1_batch)
            img2_batch = self.color_jitter(img2_batch)
            # img1_batch = self.blur(img1_batch)
            img2_batch = self.blur(img2_batch)
            
            # 初始化变换标志和掩码
            apply_color_change = random.random() < self.color_change_prob
            apply_copy_paste = random.random() < self.copy_paste_prob
            
            # 初始化为空tensor
            color_change_mask = torch.zeros((1, img2_batch.shape[2], img2_batch.shape[3]))
            src_copy_mask = torch.zeros((1, img2_batch.shape[2], img2_batch.shape[3]))
            dst_copy_mask = torch.zeros((1, img2_batch.shape[2], img2_batch.shape[3]))
            
            # 随机应用区域颜色变换
            if apply_color_change:
                img2_batch, color_change_mask = self.apply_regional_color_change(img2_batch)
            
            # 随机应用复制粘贴变换
            if apply_copy_paste and not apply_color_change:  # 避免同时应用两种变换
                img2_batch, src_copy_mask, dst_copy_mask = self.apply_copy_paste(img2_batch, original_path=img_path)  # 传递原始图像路径
            
            # 几何变换
            H, W = img1_batch.shape[-2:]
            affine1_params = self.affine.generate_parameters(img1_batch.shape)
            affine1_matrix = self.affine.compute_transformation(
                torch.tensor([[[W/2, H/2]]]), affine1_params, 'bilinear'
            )
            # affine1_matrix = torch.eye(3).unsqueeze(0).to(img1_batch.device)  # 恒等变换矩阵
            perspective_params = self.perspective.generate_parameters(img2_batch.shape)
            perspective_matrix = self.perspective.compute_transformation(img2_batch, perspective_params, 'bilinear')
            
            img1_batch = self.affine(img1_batch, affine1_params)
            img2_batch = self.perspective(img2_batch, perspective_params)
            
            img1 = img1_batch.squeeze(0)
            img2 = img2_batch.squeeze(0)

        # valid_mask1_batch = img1[0] > 0 
        # valid_mask2_batch = img2[0] > 0 
        # # 创建边界区域掩码（有效区域内缩30像素）
        # kernel_size = 61  # 2*30+1
        # padding = 30
        # valid_mask1_inner = F.avg_pool2d(
        #     F.pad(valid_mask1_batch[None,None].float(), (padding, padding, padding, padding), mode='constant', value=0),
        #     kernel_size=kernel_size, stride=1, padding=0
        # )
        # valid_mask2_inner = F.avg_pool2d(
        #     F.pad(valid_mask2_batch[None,None].float(), (padding, padding, padding, padding), mode='constant', value=0),
        #     kernel_size=kernel_size, stride=1, padding=0
        # )
        # valid_mask1_inner = valid_mask1_inner[0,0] > 0.99
        # valid_mask2_inner = valid_mask2_inner[0,0] > 0.99


        valid_mask1_batch = img1[0] > 0 
        valid_mask2_batch = img2[0] > 0 
        # 创建边界区域掩码（有效区域内缩30像素）
        border_shrink = 30

        # 优化方案3：使用OpenCV的形态学操作（仅CPU执行，需要在torch和numpy之间转换）
        # 如果在GPU上，需要先转到CPU
        masks_on_cpu = False
        if valid_mask1_batch.is_cuda:
            valid_mask1_np = valid_mask1_batch.cpu().numpy().astype(np.uint8) * 255
            valid_mask2_np = valid_mask2_batch.cpu().numpy().astype(np.uint8) * 255
            masks_on_cpu = True
        else:
            valid_mask1_np = valid_mask1_batch.numpy().astype(np.uint8) * 255
            valid_mask2_np = valid_mask2_batch.numpy().astype(np.uint8) * 255

        # 创建腐蚀核
        kernel = np.ones((2*border_shrink+1, 2*border_shrink+1), np.uint8)
        # 应用腐蚀操作
        valid_mask1_np_inner = cv2.erode(valid_mask1_np, kernel, iterations=1)
        valid_mask2_np_inner = cv2.erode(valid_mask2_np, kernel, iterations=1)

        # 转回torch
        if masks_on_cpu:
            valid_mask1_inner = torch.from_numpy(valid_mask1_np_inner > 0).to(valid_mask1_batch.device)
            valid_mask2_inner = torch.from_numpy(valid_mask2_np_inner > 0).to(valid_mask2_batch.device)
        else:
            valid_mask1_inner = torch.from_numpy(valid_mask1_np_inner > 0)
            valid_mask2_inner = torch.from_numpy(valid_mask2_np_inner > 0)


        return {
            'image0': img1,
            'image1': img2, 
            'path': img_path,
            'original_idx': img_idx,
            'pair_idx': idx % self.num_pairs_per_image,
            'transform0': affine1_matrix.squeeze(0),
            'transform1': perspective_matrix.squeeze(0),
            'color_change_applied': apply_color_change,
            'color_change_mask': color_change_mask,
            'copy_paste_applied': apply_copy_paste,
            'src_copy_mask': src_copy_mask,
            'dst_copy_mask': dst_copy_mask,
            'valid_mask0_inner': valid_mask1_inner,
            'valid_mask1_inner': valid_mask2_inner,
            'is_negative_pair': is_negative_pair
        }

def load_torch_image(image_path, target_size):
    """
    加载图像为torch张量，使用以下策略处理图像尺寸：
    - 大于目标尺寸的图像：随机裁剪
    - 小于目标尺寸的图像：等比例缩放（保持长宽比），然后在随机位置填充0
    
    Args:
        image_path: 图像文件路径
        target_size: 目标图像尺寸 (width, height)
        
    Returns:
        torch.Tensor: 加载的图像张量，尺寸为target_size
    """
    # 加载原始图像
    img = Image.open(image_path).convert('RGB')
    
    # 获取原始尺寸
    orig_width, orig_height = img.size
    
    # 目标尺寸
    target_width, target_height = target_size
    
    # 随机决定使用裁剪还是缩放（只有当图像足够大时才能裁剪）
    can_crop = orig_width >= target_width and orig_height >= target_height
    use_crop = can_crop and random.random() > 0.5
    
    if 0:
        # 随机裁剪
        max_x = orig_width - target_width
        max_y = orig_height - target_height
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        img = img.crop((x, y, x + target_width, y + target_height))
    else:
        # 等比例缩放，保持长宽比
        # 计算缩放比例
        width_ratio = target_width / orig_width
        height_ratio = target_height / orig_height
        
        # 使用较小的比例进行缩放，确保图像完全在目标尺寸内
        ratio = min(width_ratio, height_ratio)
        # ratio = 1
        
        # 计算缩放后的尺寸
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        
        # 调整大小，保持长宽比
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建黑色背景
        padded_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # 计算可能的粘贴位置范围
        paste_x_max = max(0, target_width - new_width)
        paste_y_max = max(0, target_height - new_height)
        
        # 随机选择粘贴位置
        paste_x = random.randint(0, paste_x_max) if paste_x_max > 0 else 0
        paste_y = random.randint(0, paste_y_max) if paste_y_max > 0 else 0
        
        # 粘贴调整大小后的图像到随机位置
        padded_img.paste(img_resized, (paste_x, paste_y))
        
        img = padded_img
    
    # 转换为torch张量
    img_tensor = TF.to_tensor(img)
    
    return img_tensor

def get_loss(model, pred, data):
    def loss_params(pred, i):
        la, _ = model.log_assignment[i](
            pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
        )
        return {
            "log_assignment": la,
        }
    loss_fn = NLLLoss({
                "gamma": 1.0,
                "fn": "nll",
                "nll_balancing": 0.5,
            })

    sum_weights = 1.0
    nll, gt_weights, loss_metrics = loss_fn(loss_params(pred, -1), data)
    N = pred["ref_descriptors0"].shape[1]
    losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

    losses["confidence"] = 0.0

    # B = pred['log_assignment'].shape[0]
    losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
    for i in range(N - 1):
        params_i = loss_params(pred, i)
        nll, _, _ = loss_fn(params_i, data, weights=gt_weights)

        if model.conf.loss.gamma > 0.0:
            weight = model.conf.loss.gamma ** (N - i - 1)
        else:
            weight = i + 1
        sum_weights += weight
        losses["total"] = losses["total"] + nll * weight

        losses["confidence"] += model.token_confidence[i].loss(
            pred["ref_descriptors0"][:, i],
            pred["ref_descriptors1"][:, i],
            params_i["log_assignment"],
            pred["log_assignment"],
        ) / (N - 1)

        del params_i
    losses["total"] /= sum_weights

    # confidences
    losses["total"] = losses["total"] + losses["confidence"]

    return losses

def get_gt(data, batch_size, device, distance_threshold=5):
    match_matrix = torch.full((batch_size, 1024 + 1, 512 + 1), 0., device=device, dtype=data['keypoints0'][0].dtype)
    
    for b in range(batch_size):
        keypoints0 = data['keypoints0'][b]
        keypoints1 = data['keypoints1'][b]
        is_negative = data['is_negative_pair'][b]
        valid_mask0_inner = data['valid_mask0_inner'][b]
        valid_mask1_inner = data['valid_mask1_inner'][b]
        N0, N1 = keypoints0.shape[0], keypoints1.shape[0]
        
        # # 检查是否为负样本对
        # is_negative = data.get('is_negative_pair', False)
	        # 检查点是否在有效区域内
        kpts0_valid = valid_mask0_inner[
            keypoints0[:, 1].long().clamp(0, valid_mask0_inner.shape[0]-1),
            keypoints0[:, 0].long().clamp(0, valid_mask0_inner.shape[1]-1)
        ]
        kpts1_valid = valid_mask1_inner[
            keypoints1[:, 1].long().clamp(0, valid_mask1_inner.shape[0]-1),
            keypoints1[:, 0].long().clamp(0, valid_mask1_inner.shape[1]-1)
        ]
        
        
        if is_negative:
            # 对于负样本，所有keypoints0都标记为不匹配
            for i in range(N0):
                match_matrix[b, i, N1] = 1.0
            
            # 所有keypoints1都标记为不匹配
            for j in range(N1):
                match_matrix[b, N0, j] = 1.0
                
        else:

            # 检查是否应用了颜色变换，将对应区域的匹配点标记为不匹配
            if data.get('color_change_applied', False) and data.get('color_change_mask') is not None:
                color_mask = data['color_change_mask'][b]
                # 检查keypoints1是否在颜色变换区域内
                kpts_in_changed_region = color_mask[:, keypoints1[:, 1].long(), keypoints1[:, 0].long()].bool()
                
                # 对在颜色变换区域内的关键点，将其视为不匹配
                for j in range(N1):
                    if kpts_in_changed_region[j]:
                        match_matrix[b, N0, j] = 1.0  # 设置为不匹配
                        continue
            
            # 检查是否应用了复制粘贴变换，将源区域和目标区域的匹配点标记为不匹配
            if data.get('copy_paste_applied', False):
                src_mask = data['src_copy_mask'][b]
                dst_mask = data['dst_copy_mask'][b]
                
                # 检查keypoints0是否在源区域内
                kpts0_in_src = src_mask[:, keypoints0[:, 1].long(), keypoints0[:, 0].long()].bool()
                
                # 检查keypoints1是否在源区域或目标区域内
                kpts1_in_src = src_mask[:, keypoints1[:, 1].long(), keypoints1[:, 0].long()].bool()
                kpts1_in_dst = dst_mask[:, keypoints1[:, 1].long(), keypoints1[:, 0].long()].bool()
                
                # 将源区域和目标区域内的点标记为不匹配
                for i in range(N0):
                    if kpts0_in_src[i]:
                        match_matrix[b, i, N1] = 1.0  # 设置为不匹配
                
                for j in range(N1):
                    if kpts1_in_src[j] or kpts1_in_dst[j]:
                        match_matrix[b, N0, j] = 1.0  # 设置为不匹配
            
            # 计算正常区域的匹配
            transform0_inv = torch.inverse(data['transform0'][b].to(device, dtype=keypoints0.dtype))
            transform1 = data['transform1'][b].to(device, dtype=keypoints0.dtype)
            known_transform = torch.matmul(transform1, transform0_inv)
            
            ones = torch.ones((N0, 1), device=device, dtype=keypoints0.dtype)
            kpts0_h = torch.cat([keypoints0, ones], dim=1)
            
            expected_kpts1 = torch.matmul(known_transform, kpts0_h.t()).t()
            expected_kpts1 = expected_kpts1[:, :2] / (expected_kpts1[:, 2:] + 1e-8)
            
            dist_matrix = torch.cdist(expected_kpts1, keypoints1, p=2)
            min_dists, min_indices = dist_matrix.min(dim=1)
            valid_matches = min_dists < distance_threshold
            
            # 设置匹配对，但排除已经被标记为不匹配的点
            for i in range(N0):
                # 如果keypoint0在源复制区域内，跳过，因为已经被标记为不匹配
                if data.get('copy_paste_applied', False) and 'src_copy_mask' in data:
                    if data['src_copy_mask'][b][:, keypoints0[i, 1].long(), keypoints0[i, 0].long()].bool():
                        continue
                        
                if valid_matches[i]:
                    j = min_indices[i]
                    
                    # 检查keypoint1是否在变换区域内
                    in_color_change_region = False
                    in_copy_paste_region = False
                    
                    if data.get('color_change_applied', False) and data.get('color_change_mask') is not None:
                        if data['color_change_mask'][b][:, keypoints1[j, 1].long(), keypoints1[j, 0].long()].bool():
                            in_color_change_region = True
                    
                    if data.get('copy_paste_applied', False):
                        if data['src_copy_mask'][b][:, keypoints1[j, 1].long(), keypoints1[j, 0].long()].bool():
                            in_copy_paste_region = True
                        if data['dst_copy_mask'][b][:, keypoints1[j, 1].long(), keypoints1[j, 0].long()].bool():
                            in_copy_paste_region = True
                    
                    # 如果不在任何特殊区域内，则设置为匹配
                    if not in_color_change_region and not in_copy_paste_region:
                        match_matrix[b, i, j] = 1.0  # 正常匹配
                    else:
                        match_matrix[b, i, N1] = 1.0  # 不匹配
                else:
                    match_matrix[b, i, N1] = 1.0  # 不匹配
        
        match_matrix[b][:-1] *= kpts0_valid.float()[:,None]
        match_matrix[b][:,:-1] *= kpts1_valid.float()[None,:]
    data["gt_matches0"] = (0.5 - match_matrix[:, :, -1]) / 0.5
    data["gt_matches1"] = (0.5 - match_matrix[:, -1, :]) / 0.5
    data["gt_assignment"] = match_matrix
    return data

def compute_batch_transform_loss(model, pred, data, device):
    """
    计算批量的基于已知变换的自监督损失
    
    Args:
        pred: 模型预测结果，包含匹配和置信度
        data: 输入数据，包含特征点和变换矩阵
        device: 计算设备
        
    Returns:
        loss: 损失值
    """
    batch_size = pred['matches0'].shape[0]
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float16 if pred['matches0'].dtype == torch.float16 else torch.float32)
    
    data = get_gt(data, batch_size, device)
    total_loss = get_loss(model, pred,data)['total']

    return torch.mean(total_loss)

def visualize_matches(image0, image1, kpts0, kpts1, matches, confidence, correct_matches=None, path=None):
    """
    可视化两张图像之间的匹配情况
    
    Args:
        image0, image1: 输入图像 (RGB格式)
        kpts0, kpts1: 关键点坐标 [N, 2]
        matches: 匹配索引 [M]，表示kpts0[i]匹配到kpts1[matches[i]]
        confidence: 置信度分数 [M]
        correct_matches: 如果不是None，表示正确匹配的布尔掩码 [M]
        path: 如果不是None，保存可视化结果到指定路径
    """
    # 转换图像为NumPy数组，范围为0-255的uint8
    if isinstance(image0, torch.Tensor):
        image0 = image0.cpu().numpy().transpose(1, 2, 0)
        if image0.shape[2] == 3:  # 如果是RGB图像
            image0 = (image0 * 255).astype(np.uint8)
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy().transpose(1, 2, 0)
        if image1.shape[2] == 3:  # 如果是RGB图像
            image1 = (image1 * 255).astype(np.uint8)
    
    # 创建图像和轴对象
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示图像
    ax[0].imshow(image0)
    ax[1].imshow(image1)
    
    ax[0].set_title('Source Image')
    ax[1].set_title('Target Image')
    
    # 关闭坐标轴
    ax[0].axis('off')
    ax[1].axis('off')
    
    # 绘制关键点
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=5)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=5)
    
    # 创建颜色映射，根据置信度着色
    cmap = cm.jet
    norm = Normalize(vmin=min(confidence), vmax=max(confidence))
    
    # 绘制匹配线
    for i, m in enumerate(matches):
        if m < 0:  # 跳过无匹配的点
            continue
        
        # 获取匹配点坐标
        x1, y1 = kpts0[i]
        x2, y2 = kpts1[m]
        
        # 确定线条颜色
        if correct_matches is not None:
            color = 'green' if correct_matches[i] else 'red'
        else:
            color = cmap(norm(confidence[i]))
        
        # 绘制匹配线
        con = patches.ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax[0], axesB=ax[1], color=color, linewidth=1, alpha=0.8
        )
        fig.add_artist(con)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def check_match_correctness(kpts0, kpts1, matches, transform, threshold=1.5):
    """
    检查匹配是否正确，基于已知的变换矩阵
    
    Args:
        kpts0, kpts1: 关键点坐标 [N, 2]
        matches: 匹配索引 [M]
        transform: 从kpts0到kpts1的变换矩阵 [3, 3]
        threshold: 像素误差阈值，小于该值视为正确匹配
        
    Returns:
        correctness: 布尔掩码 [M]，表示每个匹配是否正确
    """
    valid_matches = matches > -1
    num_valid = valid_matches.sum()
    
    if num_valid == 0:
        return np.array([])
    
    # 提取匹配点
    src_pts = kpts0[valid_matches].detach().cpu().numpy()
    matched_pts = kpts1[matches[valid_matches]].detach().cpu().numpy()
    
    # 转换为齐次坐标
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_h = np.concatenate([src_pts, ones], axis=1)
    
    # 应用变换
    transform_np = transform.detach().cpu().numpy()
    projected_pts = np.dot(transform_np, src_pts_h.T).T
    
    # 归一化
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
    
    # 计算欧氏距离
    errors = np.sqrt(np.sum((projected_pts - matched_pts) ** 2, axis=1))
    
    # 创建正确匹配掩码
    correctness = errors < threshold
    
    # 将结果扩展到原始大小
    full_correctness = np.zeros(len(matches), dtype=bool)
    full_correctness[valid_matches.cpu()] = correctness
    
    return full_correctness

def fine_tune_lightglue(lightglue_matcher, images, feature_dir, device, epochs=5, batch_size=16, num_pairs_per_image=1, learning_rate=1e-5):
    """对LightGlue进行自监督微调，使用批处理和半精度"""
    print(f"开始对LightGlue进行高效自监督微调 ({len(images)}张图像)")
    
    # 获取模型引用
    matcher = copy.deepcopy(lightglue_matcher.model)
    extractor = copy.deepcopy(lightglue_matcher.detector_fine)

    accumulation_steps_ori = len(images)*num_pairs_per_image / (30 * batch_size)
    accumulation_steps = max(1, int(accumulation_steps_ori))  # 确保至少为1
    epochs = max(epochs, int(epochs/accumulation_steps_ori))  # 确保至少为1
    # 冻结特征提取器参数
    for param in extractor.parameters():
        param.requires_grad = False
    
    # 设置LightGlue匹配器为训练模式
    matcher.train()
    
    # 设置优化器，只优化LightGlue匹配器
    optimizer = optim.Adam(matcher.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.1)
    
    # 创建梯度缩放器用于半精度训练
    scaler = GradScaler()
    
    # #遍历所有图像，计算图像的长宽均值
    # image_sizes = []
    # for img_path in images:
    #     img = Image.open(img_path).convert('L')
    #     image_sizes.append((img.size[0], img.size[1]))
    # avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
    # avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
    # avg_size = int(max(avg_width,avg_height)) // 8 * 8 + 32
    # avg_size = avg_size*1.25
    # avg_size = int(avg_size // 8 * 8 + 32)
    # if avg_size > 1600:
    #     avg_size = 1600
    # image_sizes = (avg_size, avg_size)

    # 创建数据集和数据加载器
    dataset = AugmentedImagePairDataset(images, num_pairs_per_image=num_pairs_per_image)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False,
        persistent_workers=False,
        # prefetch_factor=1
    )
    
    # 创建模型保存目录和可视化目录
    model_save_dir = os.path.join(feature_dir, 'fine_tuned_models')
    viz_dir = os.path.join(feature_dir, 'visualizations_train')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 保存原始模型权重以便比较
    original_model_path = os.path.join(model_save_dir, 'original_lightglue.pth')
    torch.save(matcher.state_dict(), original_model_path)
    
    # 微调循环
    best_loss = float('inf')
    best_model_state = None
    
    # 保存一个固定的评估样本，用于跨epoch比较
    eval_batch = None
    
    finetune_num = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        if finetune_num > 15:
            break
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm.tqdm(dataloader)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 保存第一个批次作为评估样本（仅第一个epoch）：
            if epoch == 0 and batch_idx == 0 and eval_batch is None:
                eval_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            finetune_num += 1
            if finetune_num > 15:
                break
            # 将数据移动到设备并转换为半精度
            imgs0 = batch['image0'].to(device)
            imgs1 = batch['image1'].to(device)
            transforms0 = batch['transform0'].to(device)
            transforms1 = batch['transform1'].to(device)

            # 获取批量图像尺寸
            batch_size = imgs0.shape[0]
            
            # 使用半精度提取特征
            with torch.no_grad(), autocast():
                # 批量特征提取
                feats0 = extractor({
                    "image": imgs0,
                    "max_kps":1024
                })
                
                feats1 = extractor({
                    "image": imgs1,
                    "max_kps":512
                })
            
            img_sizes0 = torch.stack([torch.tensor([img.shape[1], img.shape[2]]) for img in imgs0]).to(device)
            img_sizes1 = torch.stack([torch.tensor([img.shape[1], img.shape[2]]) for img in imgs1]).to(device)
            
            # 准备批量匹配输入
            match_data = {
                'keypoints0': feats0['keypoints'],
                'keypoints1': feats1['keypoints'],
                'descriptors0': feats0['descriptors'],
                'descriptors1': feats1['descriptors'],
                'image_size0': img_sizes0,
                'image_size1': img_sizes1,
                'transform0': transforms0,
                'transform1': transforms1,
                'valid_mask0_inner': batch['valid_mask0_inner'].to(device),
                'valid_mask1_inner': batch['valid_mask1_inner'].to(device),
                'is_negative_pair':batch['is_negative_pair']
            }
            
            # 使用半精度进行前向传播和损失计算
            with autocast():
                # 批量前向传播
                pred = matcher(match_data)
                
                # 计算批量损失
                loss = compute_batch_transform_loss(matcher, pred, match_data, device)

                scaled_loss = loss / accumulation_steps

            if loss.item() > 0:
                # 使用梯度缩放器处理反向传播
                scaler.scale(loss).backward()
                
                # 在累积完成后更新参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # 清零梯度
                    
                    # 记录损失
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # 更新进度条
                    progress_bar.set_description(
                        f"Loss: {loss.item():.4f} (累积步数: {(batch_idx + 1) % accumulation_steps}/{accumulation_steps})"
                    )
        
                # 处理最后一个不完整的累积批次
        
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 每个epoch结束后，计算平均损失并更新学习率
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")
            
            # 更新学习率
            scheduler.step()
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = copy.deepcopy(matcher.state_dict())
                
                # # 保存检查点
                # checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch{epoch+1}.pth')
                # torch.save({
                #     'epoch': epoch + 1,
                #     'model_state_dict': matcher.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': best_loss,
                # }, checkpoint_path)
                # print(f"  保存检查点: {checkpoint_path}")
        
        # 在每个epoch结束后，可视化评估样本的匹配结果
        if kaggle_run is False:
            if eval_batch is not None:
                # 将模型临时切换到评估模式
                matcher.eval()
                
                with torch.no_grad(), autocast():
                    # 选择评估批次中的第一个样本
                    viz_img0 = eval_batch['image0'][0:1].to(device)
                    viz_img1 = eval_batch['image1'][0:1].to(device)
                    viz_transform0 = eval_batch['transform0'][0:1].to(device)
                    viz_transform1 = eval_batch['transform1'][0:1].to(device)
                    
                    # 获取图像尺寸
                    viz_img_size0 = torch.tensor([[viz_img0.shape[2], viz_img0.shape[3]]], device=device)
                    viz_img_size1 = torch.tensor([[viz_img1.shape[2], viz_img1.shape[3]]], device=device)
                    
                    # 提取特征
                    viz_feats0 = extractor({
                        "image": viz_img0,
                        "image_size": viz_img_size0,
                        "max_kps":50
                    })
                    
                    viz_feats1 = extractor({
                        "image": viz_img1,
                        "image_size": viz_img_size1,
                        "max_kps":50
                    })
                    
                    # 准备匹配输入
                    viz_data = {
                        'keypoints0': viz_feats0['keypoints'],
                        'keypoints1': viz_feats1['keypoints'],
                        'descriptors0': viz_feats0['descriptors'],
                        'descriptors1': viz_feats1['descriptors'],
                        'image_size0': viz_img_size0,
                        'image_size1': viz_img_size1,
                        'transform0': viz_transform0,
                        'transform1': viz_transform1
                    }
                    
                    # 前向传播
                    viz_pred = matcher(viz_data)
                
                # 获取匹配结果
                kpts0 = viz_feats0['keypoints'][0].cpu().numpy()
                kpts1 = viz_feats1['keypoints'][0].cpu().numpy()
                matches = viz_pred['matches0'][0].cpu().numpy()
                confidence = viz_pred['matching_scores0'][0].cpu().numpy()
                
                # 计算从image0到image1的变换
                transform0_inv = torch.inverse(viz_transform0[0].to(device))
                transform1 = viz_transform1[0].to(device)
                known_transform = torch.matmul(transform1, transform0_inv)
                
                # 检查匹配正确性
                correct_matches = check_match_correctness(
                    viz_feats0['keypoints'][0], 
                    viz_feats1['keypoints'][0], 
                    viz_pred['matches0'][0],
                    known_transform,
                    threshold=3.0
                )
                
                # 可视化并保存结果
                viz_path = os.path.join(viz_dir, f'matches_epoch{epoch+1}.png')
                visualize_matches(
                    viz_img0[0].cpu(), 
                    viz_img1[0].cpu(), 
                    kpts0, kpts1, 
                    matches, 
                    confidence,
                    correct_matches=correct_matches,
                    path=viz_path
                )
                print(f"  保存可视化结果: {viz_path}")
                
                # 恢复训练模式
                matcher.train()

    # 训练结束，加载最佳模型
    if best_model_state is not None:
        matcher.load_state_dict(best_model_state)
    
    # 保存微调后的最终模型
    model_path = os.path.join(model_save_dir, 'lightglue_finetuned.pth')
    torch.save(matcher.state_dict(), model_path)
    print(f"微调完成，模型已保存到 {model_path}")
    
    if kaggle_run is False:
        # 可视化最终模型的结果
        if eval_batch is not None:
            matcher.eval()
            with torch.no_grad():
                # 选择评估批次中的第一个样本
                viz_img0 = eval_batch['image0'][0:1].to(device)
                viz_img1 = eval_batch['image1'][0:1].to(device)
                viz_transform0 = eval_batch['transform0'][0:1].to(device)
                viz_transform1 = eval_batch['transform1'][0:1].to(device)
                
                # 获取图像尺寸
                viz_img_size0 = torch.tensor([[viz_img0.shape[2], viz_img0.shape[3]]], device=device)
                viz_img_size1 = torch.tensor([[viz_img1.shape[2], viz_img1.shape[3]]], device=device)
                
                # 提取特征
                viz_feats0 = extractor({
                    "image": viz_img0,
                    "image_size": viz_img_size0
                })
                
                viz_feats1 = extractor({
                    "image": viz_img1,
                    "image_size": viz_img_size1
                })
                
                # 准备匹配输入
                viz_data = {
                    'keypoints0': viz_feats0['keypoints'],
                    'keypoints1': viz_feats1['keypoints'],
                    'descriptors0': viz_feats0['descriptors'],
                    'descriptors1': viz_feats1['descriptors'],
                    'image_size0': viz_img_size0,
                    'image_size1': viz_img_size1,
                    'transform0': viz_transform0,
                    'transform1': viz_transform1
                }
                
                # 前向传播
                viz_pred = matcher(viz_data)
            
            # 获取匹配结果
            kpts0 = viz_feats0['keypoints'][0].cpu().numpy()
            kpts1 = viz_feats1['keypoints'][0].cpu().numpy()
            matches = viz_pred['matches0'][0].cpu().numpy()
            confidence = viz_pred['matching_scores0'][0].cpu().numpy()
            
            # 计算从image0到image1的变换
            transform0_inv = torch.inverse(viz_transform0[0].to(device))
            transform1 = viz_transform1[0].to(device)
            known_transform = torch.matmul(transform1, transform0_inv)
            
            # 检查匹配正确性
            correct_matches = check_match_correctness(
                viz_feats0['keypoints'][0], 
                viz_feats1['keypoints'][0], 
                viz_pred['matches0'][0],
                known_transform,
                threshold=3.0
            )
            
            # 可视化并保存结果
            viz_path = os.path.join(viz_dir, 'matches_final.png')
            visualize_matches(
                viz_img0[0].cpu(), 
                viz_img1[0].cpu(), 
                kpts0, kpts1, 
                matches, 
                confidence,
                correct_matches=correct_matches,
                path=viz_path
            )
            print(f"保存最终可视化结果: {viz_path}")
        
    # 将模型设为评估模式
    matcher.eval()
    
    return matcher