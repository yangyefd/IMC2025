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

IMAGE_size = (640, 480)  # 图像大小
kaggle_run = True

class AugmentedImagePairDataset(Dataset):
    def __init__(self, image_paths, num_pairs_per_image=5, color_change_prob=0.3):
        self.image_paths = image_paths
        self.num_pairs_per_image = num_pairs_per_image
        self.color_change_prob = color_change_prob
        
        # 基础变换
        self.color_jitter = K.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.7)
        self.perspective = K.RandomPerspective(0.4, p=0.7)
        self.affine = K.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1.2), p=0.7)
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
    def __len__(self):
        return len(self.image_paths) * self.num_pairs_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_pairs_per_image
        img_path = self.image_paths[img_idx]
        
        # 加载原始图像
        original_img = load_torch_image(img_path)
        img1 = original_img.clone()
        img2 = original_img.clone()
        
        with torch.no_grad():
            img1_batch = img1.unsqueeze(0)
            img2_batch = img2.unsqueeze(0)
            
            # 应用基础变换
            img1_batch = self.color_jitter(img1_batch)
            img2_batch = self.color_jitter(img2_batch)
            img1_batch = self.blur(img1_batch)
            img2_batch = self.blur(img2_batch)
            
            # 随机决定是否应用区域颜色变换
            apply_color_change = random.random() < self.color_change_prob
            # 初始化为空tensor而不是None
            color_change_mask = torch.zeros((1, img2_batch.shape[2], img2_batch.shape[3]))
            
            if apply_color_change:
                img2_batch, color_change_mask = self.apply_regional_color_change(img2_batch)
            
            # 几何变换
            H, W = img1_batch.shape[-2:]
            affine1_params = self.affine.generate_parameters(img1_batch.shape)
            affine1_matrix = self.affine.compute_transformation(
                torch.tensor([[[W/2, H/2]]]), affine1_params, 'bilinear'
            )
            
            perspective_params = self.perspective.generate_parameters(img2_batch.shape)
            perspective_matrix = self.perspective.compute_transformation(img2_batch, perspective_params, 'bilinear')
            
            img1_batch = self.affine(img1_batch, affine1_params)
            img2_batch = self.perspective(img2_batch, perspective_params)
            
            img1 = img1_batch.squeeze(0)
            img2 = img2_batch.squeeze(0)
                
        return {
            'image0': img1,
            'image1': img2, 
            'path': img_path,
            'original_idx': img_idx,
            'pair_idx': idx % self.num_pairs_per_image,
            'transform0': affine1_matrix.squeeze(0),
            'transform1': perspective_matrix.squeeze(0),
            'color_change_applied': apply_color_change,  # 添加颜色变换标志
            'color_change_mask': color_change_mask  # 添加颜色变换mask
        }

def load_torch_image(image_path, resize_size=IMAGE_size):
    """加载图像为torch张量"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.resize(img_tensor, resize_size, antialias=True)
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

def get_gt(data, batch_size, device, distance_threshold=2):
    match_matrix = torch.full((batch_size, 1024 + 1, 1024 + 1), 0., device=device, dtype=data['keypoints0'][0].dtype)
    
    for b in range(batch_size):
        keypoints0 = data['keypoints0'][b]
        keypoints1 = data['keypoints1'][b]
        N0, N1 = keypoints0.shape[0], keypoints1.shape[0]
        
        # 如果应用了颜色变换，检查关键点是否在变换区域内
        if data.get('color_change_applied', False) and data.get('color_change_mask') is not None:
            color_mask = data['color_change_mask'][b]
            # 检查keypoints1是否在颜色变换区域内
            kpts_in_changed_region = color_mask[:, keypoints1[:, 1].long(), keypoints1[:, 0].long()].bool()
            
            # 对在颜色变换区域内的关键点，将其视为不匹配
            for j in range(N1):
                if kpts_in_changed_region[j]:
                    match_matrix[b, N0, j] = 1.0  # 设置为不匹配
                    continue
        
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
        
        # 设置匹配对
        for i in range(N0):
            if valid_matches[i]:
                j = min_indices[i]
                # 如果匹配点在颜色变换区域内，则视为不匹配
                if data.get('color_change_applied', False) and data.get('color_change_mask') is not None:
                    if color_mask[:, keypoints1[j, 1].long(), keypoints1[j, 0].long()].bool():
                        match_matrix[b, i, N1] = 1.0  # 不匹配
                        continue
                match_matrix[b, i, j] = 1.0  # 正常匹配
            else:
                match_matrix[b, i, N1] = 1.0  # 不匹配
    
    data["gt_matches0"] = (match_matrix[:, :, -1] - 0.5) / 0.5
    data["gt_matches1"] = (match_matrix[:, -1, :] - 0.5) / 0.5
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

def check_match_correctness(kpts0, kpts1, matches, transform, threshold=3.0):
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

def fine_tune_lightglue(lightglue_matcher, images, feature_dir, device, epochs=5, batch_size=16, num_pairs_per_image=5, learning_rate=1e-5):
    """对LightGlue进行自监督微调，使用批处理和半精度"""
    print(f"开始对LightGlue进行高效自监督微调 ({len(images)}张图像)")
    
    # 获取模型引用
    matcher = lightglue_matcher.model
    extractor = lightglue_matcher.detector

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
    
    # 创建数据集和数据加载器
    dataset = AugmentedImagePairDataset(images, num_pairs_per_image)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1
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
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm.tqdm(dataloader)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 保存第一个批次作为评估样本（仅第一个epoch）：
            if epoch == 0 and batch_idx == 0 and eval_batch is None:
                eval_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
            # 将数据移动到设备并转换为半精度
            imgs0 = batch['image0'].to(device)
            imgs1 = batch['image1'].to(device)
            transforms0 = batch['transform0'].to(device)
            transforms1 = batch['transform1'].to(device)
            
            # 获取批量图像尺寸
            batch_size = imgs0.shape[0]
            img_sizes0 = IMAGE_size
            img_sizes1 = IMAGE_size
            
            optimizer.zero_grad()
            
            # 使用半精度提取特征
            with torch.no_grad(), autocast():
                # 批量特征提取
                feats0 = extractor({
                    "image": imgs0,
                    "image_size": img_sizes0,
                    "max_kps":1024
                })
                
                feats1 = extractor({
                    "image": imgs1,
                    "image_size": img_sizes1,
                    "max_kps":1024
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
                'transform1': transforms1
            }
            
            # 使用半精度进行前向传播和损失计算
            with autocast():
                # 批量前向传播
                pred = matcher(match_data)
                
                # 计算批量损失
                loss = compute_batch_transform_loss(matcher, pred, match_data, device)
            
            if loss.item() > 0:
                # 使用梯度缩放器处理反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
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
                
                # 保存检查点
                checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': matcher.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
                print(f"  保存检查点: {checkpoint_path}")
        
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