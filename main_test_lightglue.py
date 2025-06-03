
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("oldufo/imc2024-packages-lightglue-rerun-kornia")

# print("Path to dataset files:", path)

# Download latest version
# path = kagglehub.dataset_download("eduardtrulls/imc25-utils")

# print("Path to dataset files:", path)
import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from lightglue import match_pair, ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel
import pycolmap
import matplotlib.pyplot as plt
sys.path.append('./imc25-utils/versions/6')
from database import *
from h5_to_db import *
import metric
from typing import Optional, Union
from kornia.feature.laf import get_laf_center, get_laf_orientation, get_laf_scale

from GIMlightglue_match import Lightglue_Matcher
from fine_tune_lightglue import fine_tune_lightglue
# from filter_match import adaptive_match_filtering
from CLIP.clip import clip
from sklearn.cluster import DBSCAN
from data_process.db import *
import pickle

# Device setup
device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')

def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在主程序开始时调用
set_seed(42)

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def get_global_desc(fnames, device=torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('./models/dinov2-pytorch-base-v1')
    model = AutoModel.from_pretrained('./models/dinov2-pytorch-base-v1')
    model = model.eval().to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    return torch.cat(global_descs_dinov2, dim=0)

def get_global_desc_clip(fnames, device=torch.device('cpu')):
    try:
        model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
        print("分簇模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        # 返回一个空的特征向量或默认特征向量
        return torch.zeros((len(fnames), 512)).float()

    model = model.eval().to(device)
    global_descs_dinov2 = []
    
    # 记录处理失败的图像索引
    failed_indices = []
    
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        try:
            key = os.path.splitext(os.path.basename(img_fname_full))[0]
            # 加载并预处理图像
            timg = preprocess(Image.open(img_fname_full)).unsqueeze(0).to(device)

            # 提取特征
            with torch.no_grad():
                features = model.encode_image(timg)
            # 归一化特征
            features = features / features.norm(dim=-1, keepdim=True)

            global_descs_dinov2.append(features.detach().cpu())
        except Exception as e:
            print(f"处理图像 {img_fname_full} 失败: {e}")
            # 记录失败的索引，并使用零向量替代
            failed_indices.append(i)
            # 创建一个零向量作为占位符
            zero_feat = torch.zeros((1, 512), device='cpu').float()
            global_descs_dinov2.append(zero_feat)
    
    # 输出失败图像的数量
    if failed_indices:
        print(f"警告: {len(failed_indices)}张图像处理失败")
    
    return torch.cat(global_descs_dinov2, dim=0).float()

def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs

def get_image_pairs_shortlist(fnames, sim_th=0.6, min_pairs=20, exhaustive_if_less=20, 
                            device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)

    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    mask = dm <= sim_th
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
    return sorted(list(set(matching_list)))

def get_image_pairs_shortlist_clip(fnames, sim_th=0.6, min_pairs=20, exhaustive_if_less=20, 
                            device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc_clip(fnames, device=device)        
    # 计算余弦相似度矩阵 (N x N)
    similarity = torch.mm(descs, descs.t()).detach().cpu().numpy()
    
    # 相似度大于阈值的保留
    mask = similarity >= sim_th
    
    matching_list = []
    ar = np.arange(num_imgs)
    
    for st_idx in range(num_imgs-1):
        # 找出与当前图像相似度大于阈值的所有图像
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        # 如果符合条件的图像太少，选择相似度最高的前min_pairs个
        if len(to_match) < min_pairs:
            to_match = np.argsort(similarity[st_idx])[::-1][:min_pairs+1]  # 降序排列并取前min_pairs+1个
        for idx in to_match:
            if st_idx == idx:  # 跳过自己与自己的匹配
                continue
            # 添加匹配对
            matching_list.append(tuple(sorted((st_idx, idx.item() if hasattr(idx, 'item') else idx))))
    
    # 去重并排序
    return sorted(list(set(matching_list)))

def detect_aliked(img_fnames, feature_dir='.featureout', num_features=4096, 
                 resize_to=1024, device=torch.device('cpu')):
    dtype = torch.float32
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.3, 
                     resize=resize_to).eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            key = key.split('\\')[-1]
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor.extract(image0)
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

def visualize_matches(img1_path, img2_path, kpts1, kpts2, matches, save_path=None, show=False):
    """可视化两张图片的匹配结果
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径  
        kpts1: 第一张图片的特征点 (Nx2)
        kpts2: 第二张图片的特征点 (Nx2)
        matches: 匹配索引 (Mx2)
        save_path: 保存路径,如果为None则显示
        show: 是否显示结果
    """
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 获取原始尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 调整图片大小使其具有相同高度
    height = min(h1, h2, 800)  # 限制最大高度为800像素
    
    # 计算缩放比例
    scale1 = height / h1
    scale2 = height / h2
    
    # 调整图像大小
    img1 = cv2.resize(img1, (int(w1 * scale1), height))
    img2 = cv2.resize(img2, (int(w2 * scale2), height))
    
    # 创建拼接图
    vis = np.hstack([img1, img2])
    
    # 复制关键点并按比例缩放
    kpts1_scaled = kpts1.copy()
    kpts2_scaled = kpts2.copy()
    
    # 应用缩放比例
    kpts1_scaled[:, 0] *= scale1
    kpts1_scaled[:, 1] *= scale1
    kpts2_scaled[:, 0] *= scale2
    kpts2_scaled[:, 1] *= scale2
    
    # 绘制匹配线
    offset = img1.shape[1]
    for idx1, idx2 in matches:
        pt1 = tuple(map(int, kpts1_scaled[idx1]))
        pt2 = tuple(map(int, kpts2_scaled[idx2]))
        pt2 = (pt2[0] + offset, pt2[1])
        cv2.circle(vis, pt1, 2, (0, 255, 0), -1)
        cv2.circle(vis, pt2, 2, (0, 255, 0), -1)
        cv2.line(vis, pt1, pt2, (255, 0, 0), 1)
    
    # 添加匹配数量文本
    cv2.putText(vis, f"Matches: {len(matches)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def detect_person(lightglue_matcher, img_fnames, feature_dir='.featureout', device=torch.device('cpu')):
    #集成方法 ALIke sp各提一半点 2048个
    dtype = torch.float32

    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    mask_lst = []
    mask_dict = {}
    for img_path in tqdm(img_fnames):
        img_fname = img_path.split('/')[-1]
        img_fname = img_fname.split('\\')[-1]
        key = img_fname
        with torch.inference_mode():
            mask, mask_ratio, mask_num = lightglue_matcher.get_person_mask(img_path)
            
            if mask is not None:
                mask_lst.append([key, mask, mask_ratio, mask_num])
                mask_dict[key] = mask
            else:
                mask_dict[key] = np.zeros((0, 0), dtype=np.bool_)
    if len(mask_lst) > 0:
        mask_ratio_sum = 0
        mask_num_sum = 0
        for _, _, mask_ratio, mask_num in mask_lst:
            mask_ratio_sum += mask_ratio
            mask_num_sum += mask_num
        mask_ratio_mean = mask_ratio_sum / len(mask_lst)
        mask_num_mean = mask_num_sum / len(mask_lst)

        if (mask_ratio_mean > 0.15 and abs(mask_num_mean - 1) < 0.5) or len(mask_lst) < 3:
            mask_lst = []
    with h5py.File(f'{feature_dir}/p_mask.h5', mode='w') as f_pmask:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            img_fname = img_fname.split('\\')[-1]
            key = img_fname

            f_pmask[key] = mask_dict[key] if len(mask_lst) > 0 else np.zeros((0, 0), dtype=np.bool_)
            # print(f_pmask[key])
    return

def detect_sp_ensemble(lightglue_matcher, img_fnames, feature_dir='.featureout', num_features=4096, 
                 resize_to=1024, device=torch.device('cpu')):
    #集成方法 ALIke sp各提一半点 2048个
    dtype = torch.float32

    extractor_alike = ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, 
                    resize=resize_to).eval().to(device, dtype)
    
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='w') as f_size,\
         h5py.File(f'{feature_dir}/scale.h5', mode='w') as f_scale,\
         h5py.File(f'{feature_dir}/mask.h5', mode='w') as f_mask:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            img_fname = img_fname.split('\\')[-1]
            key = img_fname
            with torch.inference_mode():
                try:
                # if 1:
                    kpts = np.zeros((num_features*2,2)).astype(np.float32)
                    kpts_refine = np.zeros((num_features*2,2)).astype(np.float32)
                    descs = np.zeros((num_features*2,256)).astype(np.float32)
                    feats0, data = lightglue_matcher.extract(img_path,nms_radius=2,force=True)
                    feats0_kpts = feats0['keypoints0'].reshape(-1, 2).detach().cpu().numpy()
                    kpts[:len(feats0_kpts)] = feats0['keypoints0'].reshape(-1, 2).detach().cpu().numpy()
                    kpts_refine[:len(feats0_kpts)] = feats0['keypoints_refine0'].reshape(-1, 2).detach().cpu().numpy()
                    descs[:len(feats0_kpts)] = feats0['descriptors0'].reshape(len(feats0_kpts), -1).detach().cpu().numpy()

                    image0 = load_torch_image(img_path, device=device).to(dtype)
                    feats0_alike = extractor_alike.extract(image0)
                    feats0_alike_pkts = feats0_alike['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                    feats0_alike_descs = feats0_alike['descriptors'].reshape(len(feats0_alike_pkts), -1).detach().cpu().numpy()
                    #使用分数对点和描述的位置进行排序，分高的放前
                    feats0_alike_pkts_score = feats0_alike['keypoint_scores'].reshape(-1).detach().cpu().numpy()
                    sort_idx = np.argsort(feats0_alike_pkts_score)[::-1]
                    kpts[num_features:num_features+len(feats0_alike_pkts)] = feats0_alike_pkts[sort_idx]
                    descs[num_features:num_features+len(feats0_alike_pkts),:128] = feats0_alike_descs[sort_idx]
                    descs[num_features:num_features+len(feats0_alike_pkts),128:] = feats0_alike_descs[sort_idx]
                    
                    f_kp[key] = kpts
                    f_desc[key] = descs
                    f_size[key] = data['size0'].cpu()
                    f_scale[key] = data['scale0'].cpu()
                    f_mask[key] = np.array([len(feats0_kpts), len(feats0_alike_pkts)])
                except Exception as e:
                    print(f"Error processing image {img_fname}: {e}") 
                    # 处理异常情况，例如记录错误或跳过该图像
                    # 可以选择继续处理下一个图像，或者根据需要进行其他操作
                    continue

    return

def match_with_gimlightglue(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
        h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            key1 = key1.split('\\')[-1]
            key2 = key2.split('\\')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            pred = {}
            pred['keypoints0'] = kp1[None]
            pred['keypoints1'] = kp2[None]
            pred['descriptors0'] = desc1[None]
            pred['descriptors1'] = desc2[None]
            pred['size0'] = torch.from_numpy(f_size[key1][...]).to(device)
            pred['size1'] = torch.from_numpy(f_size[key2][...]).to(device)
            pred['scale0'] = torch.from_numpy(f_scale[key1][...]).to(device)
            pred['scale1'] = torch.from_numpy(f_scale[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lightglue_matcher.match(pred)
            if len(idxs) == 0:
                continue
                
            #  # 应用区域筛选方法
            # filtered_idxs = adaptive_match_filtering(
            #     lightglue_matcher, kp1, kp2, idxs.cpu().numpy(), fname1, fname2, device
            # )
            # # 转回tensor
            # if isinstance(filtered_idxs, np.ndarray):
            #     idxs = torch.from_numpy(filtered_idxs).to(idxs.device)

            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                match_matrix[idx1,idx2] = len(idxs.detach().cpu().numpy().reshape(-1, 2))
                                # 添加可视化
                # if visualize:
                #     vis_dir = os.path.join(feature_dir, 'visualizations')
                #     os.makedirs(vis_dir, exist_ok=True)
                #     save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                #     visualize_matches(fname1, fname2, 
                #                    kp1.cpu().numpy(), 
                #                    kp2.cpu().numpy(),
                #                    idxs.cpu().numpy(),
                #                    save_path)
    return match_matrix

def filter_duplicate_matches(idxs, idxs_alike_adjusted, kp1, kp2, duplicate_threshold=3.0):
    """
    Filter out duplicate ALIKE matches that are too close to SuperPoint matches.
    
    Args:
        idxs: Tensor of shape (N, 2) containing SuperPoint match indices.
        idxs_alike_adjusted: Tensor of shape (M, 2) containing ALIKE match indices.
        kp1: Tensor of shape (K, 2) containing keypoints for image 1.
        kp2: Tensor of shape (K, 2) containing keypoints for image 2.
        duplicate_threshold: Float, pixel distance threshold for duplicates.
    
    Returns:
        Tensor of combined non-duplicate matches.
    """
    # Get coordinates for SuperPoint matches
    sp_coords1 = kp1[idxs[:, 0]]  # Shape: (N, 2)
    sp_coords2 = kp2[idxs[:, 1]]  # Shape: (N, 2)
    
    # Get coordinates for ALIKE matches
    alike_coords1 = kp1[idxs_alike_adjusted[:, 0]]  # Shape: (M, 2)
    alike_coords2 = kp2[idxs_alike_adjusted[:, 1]]  # Shape: (M, 2)
    
    # Compute pairwise distances using broadcasting
    # dist1: Distance between ALIKE points in image 1 and SuperPoint points in image 1
    dist1 = torch.cdist(alike_coords1, sp_coords1, p=2)  # Shape: (M, N)
    dist2 = torch.cdist(alike_coords2, sp_coords2, p=2)  # Shape: (M, N)
    
    # Check for duplicates: both distances must be below threshold
    duplicate_mask = (dist1 < duplicate_threshold) & (dist2 < duplicate_threshold)
    valid_mask = ~torch.any(duplicate_mask, dim=1)  # Shape: (M,)
    
    # Filter non-duplicate ALIKE matches
    filtered_alike_matches = idxs_alike_adjusted[valid_mask]
    
    # Combine matches
    combined_matches = torch.cat([idxs, filtered_alike_matches], dim=0)
    
    return combined_matches

def visualize_clusters(img1_path, img2_path, mkpts1, mkpts2, labels1, labels2, 
                      cluster_centers1, cluster_centers2, cluster_radii1, cluster_radii2,
                      save_path=None, all_kp1=None, all_kp2=None):
    """
    可视化两张图片的聚类结果，包括聚类中心、聚类区域的圆圈和所有特征点
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        mkpts1: 第一张图片的特征点 (Nx2)
        mkpts2: 第二张图片的特征点 (Nx2)
        labels1: 第一张图片特征点的聚类标签 (N)
        labels2: 第二张图片特征点的聚类标签 (N)
        cluster_centers1: 第一张图片的聚类中心 [(x1,y1), (x2,y2), ...]
        cluster_centers2: 第二张图片的聚类中心 [(x1,y1), (x2,y2), ...]
        cluster_radii1: 第一张图片的聚类半径 [r1, r2, ...]
        cluster_radii2: 第二张图片的聚类半径 [r1, r2, ...]
        save_path: 保存路径，如果为None则显示
        all_kp1: 第一张图片的所有特征点 (Mx2)，可以为None
        all_kp2: 第二张图片的所有特征点 (Mx2)，可以为None
    """
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 获取原始尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 调整图片大小使其具有相同高度
    height = min(h1, h2, 800)  # 限制最大高度为800像素
    
    # 计算缩放比例
    scale1 = height / h1
    scale2 = height / h2
    
    # 调整图像大小
    img1 = cv2.resize(img1, (int(w1 * scale1), height))
    img2 = cv2.resize(img2, (int(w2 * scale2), height))
    
    # 创建拼接图
    vis = np.hstack([img1, img2])
    
    # 偏移量
    offset = img1.shape[1]
    
    # 如果提供了所有特征点，则先绘制它们（作为背景）
    if all_kp1 is not None:
        # 确保是numpy数组
        if isinstance(all_kp1, torch.Tensor):
            all_kp1 = all_kp1.cpu().numpy()
        
        # 缩放所有特征点
        all_kp1_scaled = all_kp1.copy()
        all_kp1_scaled[:, 0] *= scale1
        all_kp1_scaled[:, 1] *= scale1
        
        # 绘制所有特征点（淡灰色小点）
        for pt in all_kp1_scaled:
            pt = tuple(map(int, pt))
            cv2.circle(vis, pt, 1, (80, 80, 80), -1)
    
    if all_kp2 is not None:
        # 确保是numpy数组
        if isinstance(all_kp2, torch.Tensor):
            all_kp2 = all_kp2.cpu().numpy()
        
        # 缩放所有特征点
        all_kp2_scaled = all_kp2.copy()
        all_kp2_scaled[:, 0] *= scale2
        all_kp2_scaled[:, 1] *= scale2
        
        # 绘制所有特征点（淡灰色小点）
        for pt in all_kp2_scaled:
            pt = tuple(map(int, (pt[0] + offset / scale2 * scale1, pt[1])))
            cv2.circle(vis, pt, 1, (80, 80, 80), -1)
    
    # 复制关键点并按比例缩放
    mkpts1_scaled = mkpts1.copy()
    mkpts2_scaled = mkpts2.copy()
    
    # 应用缩放比例
    mkpts1_scaled[:, 0] *= scale1
    mkpts1_scaled[:, 1] *= scale1
    mkpts2_scaled[:, 0] *= scale2
    mkpts2_scaled[:, 1] *= scale2
    
    # 所有聚类的颜色映射
    unique_labels1 = np.unique(labels1[labels1 >= 0])
    unique_labels2 = np.unique(labels2[labels2 >= 0])
    num_clusters = max(len(unique_labels1), len(unique_labels2), 1)  # 至少有一种颜色
    
    # 生成随机颜色，但确保对比度足够
    colors = []
    for i in range(num_clusters):
        # 生成HSV颜色以确保彩色和亮度多样性
        h = int(i * 180 / num_clusters) % 180  # 色调均匀分布
        s = 200 + np.random.randint(55)  # 高饱和度
        v = 200 + np.random.randint(55)  # 适中亮度
        bgr_color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    
    # 绘制聚类区域（半透明填充区域）
    for i, (center, radius) in enumerate(zip(cluster_centers1, cluster_radii1)):
        center = (int(center[0] * scale1), int(center[1] * scale1))
        radius = int(radius * scale1)
        color_idx = i % len(colors)
        
        # 创建一个透明图层
        overlay = vis.copy()
        cv2.circle(overlay, center, radius, colors[color_idx], -1)  # 填充圆
        # 添加透明效果
        alpha = 0.2  # 透明度
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    
    # 第二张图片的聚类区域
    for i, (center, radius) in enumerate(zip(cluster_centers2, cluster_radii2)):
        center = (int(center[0] * scale2) + offset, int(center[1] * scale2))
        radius = int(radius * scale2)
        color_idx = i % len(colors)
        
        # 创建一个透明图层
        overlay = vis.copy()
        cv2.circle(overlay, center, radius, colors[color_idx], -1)  # 填充圆
        # 添加透明效果
        alpha = 0.2  # 透明度
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    
    # 绘制特征点和聚类关系
    for i, (pt1, pt2, l1, l2) in enumerate(zip(mkpts1_scaled, mkpts2_scaled, labels1, labels2)):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        pt2 = (pt2[0] + offset, pt2[1])
        
        # 对聚类中的点使用聚类颜色
        if l1 >= 0:
            color_idx = np.where(unique_labels1 == l1)[0][0] % len(colors)
            color = colors[color_idx]
            cv2.circle(vis, pt1, 3, color, -1)
        else:
            # 噪声点为灰色
            cv2.circle(vis, pt1, 2, (128, 128, 128), -1)
            
        if l2 >= 0:
            color_idx = np.where(unique_labels2 == l2)[0][0] % len(colors)
            color = colors[color_idx]
            cv2.circle(vis, pt2, 3, color, -1)
        else:
            # 噪声点为灰色
            cv2.circle(vis, pt2, 2, (128, 128, 128), -1)
        
        # 如果两点都属于聚类，绘制连线
        if l1 >= 0 and l2 >= 0:
            # 找出l1和l2对应的索引
            color_idx1 = np.where(unique_labels1 == l1)[0][0] % len(colors)
            color_idx2 = np.where(unique_labels2 == l2)[0][0] % len(colors)
            
            # 使用混合颜色
            if color_idx1 == color_idx2:
                line_color = colors[color_idx1]
            else:
                # 使用灰色作为不匹配聚类的连线
                line_color = (200, 200, 200)
                
            cv2.line(vis, pt1, pt2, line_color, 1)
    
    # 绘制聚类中心和圆圈边界
    for i, (center, radius) in enumerate(zip(cluster_centers1, cluster_radii1)):
        center = (int(center[0] * scale1), int(center[1] * scale1))
        radius = int(radius * scale1)
        color_idx = i % len(colors)
        cv2.circle(vis, center, 6, colors[color_idx], -1)  # 聚类中心
        cv2.circle(vis, center, radius, colors[color_idx], 2)  # 聚类区域边界
        
        # 添加聚类编号
        cv2.putText(vis, f"{i}", (center[0] + 10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color_idx], 2)
    
    # 第二张图片的聚类中心和圆圈边界
    for i, (center, radius) in enumerate(zip(cluster_centers2, cluster_radii2)):
        center = (int(center[0] * scale2) + offset, int(center[1] * scale2))
        radius = int(radius * scale2)
        color_idx = i % len(colors)
        cv2.circle(vis, center, 6, colors[color_idx], -1)  # 聚类中心
        cv2.circle(vis, center, radius, colors[color_idx], 2)  # 聚类区域边界
        
        # 添加聚类编号
        cv2.putText(vis, f"{i}", (center[0] + 10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color_idx], 2)
    
    # 添加说明文本
    cv2.putText(vis, f"Clusters img1: {len(unique_labels1)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Clusters img2: {len(unique_labels2)}", (offset + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 添加灰色噪声点说明
    cv2.putText(vis, "Gray: Noise points", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    # 添加深灰色背景点说明
    if all_kp1 is not None or all_kp2 is not None:
        cv2.putText(vis, "Dark gray: All features", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

    # 保存图像
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return vis

def filter_clusters_by_match_count(idxs, features_data, key1, key2, cluster_centers1, cluster_centers2, 
                                    cluster_radius1, cluster_radius2, min_matches_per_cluster=5):
    """
    根据聚类中心和半径，计算每个匹配对所属簇，并过滤掉匹配对数量不足阈值的簇
    """
    if isinstance(idxs, torch.Tensor):
        idxs_np = idxs.cpu().numpy()
    else:
        idxs_np = idxs
    
    # 没有聚类时直接返回原始匹配
    if len(cluster_centers1) == 0 or len(cluster_centers2) == 0:
        return idxs
    
    # 获取匹配对坐标
    kp1 = features_data[key1]['kp']
    kp2 = features_data[key2]['kp']
    
    if isinstance(kp1, torch.Tensor):
        kp1 = kp1.cpu().numpy()
    if isinstance(kp2, torch.Tensor):
        kp2 = kp2.cpu().numpy()
    
    # 获取匹配对的坐标
    match_coords1 = kp1[idxs_np[:, 0]]
    match_coords2 = kp2[idxs_np[:, 1]]
    
    # 初始化每个匹配对所属的簇
    match_cluster_ids = -np.ones(len(idxs_np), dtype=int)
    
    # 每个簇的匹配对计数
    cluster_match_counts = {}
    
    # 为每个匹配对分配簇
    for i, (coord1, coord2) in enumerate(zip(match_coords1, match_coords2)):
        # 检查第一张图像中点所属的簇
        cluster1_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers1, cluster_radius1)):
            dist = np.sqrt(np.sum((coord1 - center) ** 2))
            if dist <= radius:
                cluster1_id = c_id
                break
        
        # 检查第二张图像中点所属的簇
        cluster2_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers2, cluster_radius2)):
            dist = np.sqrt(np.sum((coord2 - center) ** 2))
            if dist <= radius:
                cluster2_id = c_id
                break
        
        # 只有当两个点都属于某个簇时，才认为这个匹配对属于一个有效簇
        if cluster1_id >= 0 and cluster2_id >= 0:
            cluster_pair = (cluster1_id, cluster2_id)
            match_cluster_ids[i] = hash(cluster_pair) % 10000000  # 使用哈希值作为簇对的唯一标识
            
            if cluster_pair not in cluster_match_counts:
                cluster_match_counts[cluster_pair] = 0
            cluster_match_counts[cluster_pair] += 1
    
    # 找出满足最小匹配对数量的簇
    valid_cluster_pairs = {pair for pair, count in cluster_match_counts.items() 
                           if count >= min_matches_per_cluster}
    
    # 生成过滤掩码，只保留属于有效簇的匹配对
    valid_mask = np.zeros(len(idxs_np), dtype=bool)
    
    for i, (coord1, coord2) in enumerate(zip(match_coords1, match_coords2)):
        # 再次检查第一张图像中点所属的簇
        cluster1_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers1, cluster_radius1)):
            dist = np.sqrt(np.sum((coord1 - center) ** 2))
            if dist <= radius:
                cluster1_id = c_id
                break
        
        # 再次检查第二张图像中点所属的簇
        cluster2_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers2, cluster_radius2)):
            dist = np.sqrt(np.sum((coord2 - center) ** 2))
            if dist <= radius:
                cluster2_id = c_id
                break
        
        # 如果匹配对属于有效簇，则保留
        if cluster1_id >= 0 and cluster2_id >= 0:
            cluster_pair = (cluster1_id, cluster2_id)
            if cluster_pair in valid_cluster_pairs:
                valid_mask[i] = True
        else:
            # 不属于任何簇的匹配对也保留（可选，视需求而定）
            valid_mask[i] = True
    
    # 应用过滤
    idxs_filter_np = idxs_np[valid_mask]
    
    # 转回原始类型
    if isinstance(idxs, torch.Tensor):
        return torch.tensor(idxs_filter_np, device=idxs.device, dtype=idxs.dtype)
    else:
        return idxs_filter_np

def filter_clusters_by_match_count_with_scores(idxs, scores, features_data, key1, key2, cluster_centers1, cluster_centers2, 
                                    cluster_radius1, cluster_radius2, min_matches_per_cluster=5):
    """
    根据聚类中心和半径，计算每个匹配对所属簇，并过滤掉匹配对数量不足阈值的簇
    同时保留每个匹配对应的分数
    """
    if isinstance(idxs, torch.Tensor):
        idxs_np = idxs.cpu().numpy()
    else:
        idxs_np = idxs
        
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = scores
    
    # 没有聚类时直接返回原始匹配
    if len(cluster_centers1) == 0 or len(cluster_centers2) == 0:
        return idxs, scores
    
    # 获取匹配对坐标
    kp1 = features_data[key1]['kp']
    kp2 = features_data[key2]['kp']
    
    if isinstance(kp1, torch.Tensor):
        kp1 = kp1.cpu().numpy()
    if isinstance(kp2, torch.Tensor):
        kp2 = kp2.cpu().numpy()
    
    # 获取匹配对的坐标
    match_coords1 = kp1[idxs_np[:, 0]]
    match_coords2 = kp2[idxs_np[:, 1]]
    
    # 初始化每个匹配对所属的簇
    match_cluster_ids = -np.ones(len(idxs_np), dtype=int)
    
    # 每个簇的匹配对计数
    cluster_match_counts = {}
    
    # 为每个匹配对分配簇
    for i, (coord1, coord2) in enumerate(zip(match_coords1, match_coords2)):
        # 检查第一张图像中点所属的簇
        cluster1_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers1, cluster_radius1)):
            dist = np.sqrt(np.sum((coord1 - center) ** 2))
            if dist <= radius:
                cluster1_id = c_id
                break
        
        # 检查第二张图像中点所属的簇
        cluster2_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers2, cluster_radius2)):
            dist = np.sqrt(np.sum((coord2 - center) ** 2))
            if dist <= radius:
                cluster2_id = c_id
                break
        
        # 只有当两个点都属于某个簇时，才认为这个匹配对属于一个有效簇
        if cluster1_id >= 0 and cluster2_id >= 0:
            cluster_pair = (cluster1_id, cluster2_id)
            match_cluster_ids[i] = hash(cluster_pair) % 10000000  # 使用哈希值作为簇对的唯一标识
            
            if cluster_pair not in cluster_match_counts:
                cluster_match_counts[cluster_pair] = 0
            cluster_match_counts[cluster_pair] += 1
    
    # 找出满足最小匹配对数量的簇
    valid_cluster_pairs = {pair for pair, count in cluster_match_counts.items() 
                           if count >= min_matches_per_cluster}
    
    # 生成过滤掩码，只保留属于有效簇的匹配对
    valid_mask = np.zeros(len(idxs_np), dtype=bool)
    
    for i, (coord1, coord2) in enumerate(zip(match_coords1, match_coords2)):
        # 再次检查第一张图像中点所属的簇
        cluster1_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers1, cluster_radius1)):
            dist = np.sqrt(np.sum((coord1 - center) ** 2))
            if dist <= radius:
                cluster1_id = c_id
                break
        
        # 再次检查第二张图像中点所属的簇
        cluster2_id = -1
        for c_id, (center, radius) in enumerate(zip(cluster_centers2, cluster_radius2)):
            dist = np.sqrt(np.sum((coord2 - center) ** 2))
            if dist <= radius:
                cluster2_id = c_id
                break
        
        # 如果匹配对属于有效簇，则保留
        if cluster1_id >= 0 and cluster2_id >= 0:
            cluster_pair = (cluster1_id, cluster2_id)
            if cluster_pair in valid_cluster_pairs:
                valid_mask[i] = True
        else:
            # 不属于任何簇的匹配对也保留（可选，视需求而定）
            valid_mask[i] = True
    
    # 应用过滤
    idxs_filter_np = idxs_np[valid_mask]
    scores_filter_np = scores_np[valid_mask]
    
    # 转回原始类型
    if isinstance(idxs, torch.Tensor):
        filtered_idxs = torch.tensor(idxs_filter_np, device=idxs.device, dtype=idxs.dtype)
        filtered_scores = torch.tensor(scores_filter_np, device=scores.device, dtype=scores.dtype)
        return filtered_idxs, filtered_scores
    else:
        return idxs_filter_np, scores_filter_np

def second_match_ensemble(mkpts1, mkpts2, idxs, match_scores, features_data, key1, key2, lg_matcher, device, startidx=4096):
    """二次匹配函数，增加索引映射功能确保结果与原始特征点对应
    
    Args:
        mkpts1, mkpts2: 初次匹配的特征点对
        idxs: 初次匹配的索引对
        match_scores: 初次匹配的分数
        features_data: 特征数据字典
        key1, key2: 图像标识符
        lg_matcher: 匹配器实例
        startidx: 特征起始索引
        
    Returns:
        mapped_idxs: 映射回原始索引的匹配结果
        merged_scores: 合并后的匹配分数
    """

    # 将原始匹配对转换为集合形式，便于快速查找
    orig_idxs = idxs.clone().cpu().numpy()
    orig_matches_set = {(int(idx[0])) for idx in orig_idxs}
    
    # 保存原始匹配分数，用于后续合并
    orig_scores_dict = {}
    for i, (idx1, idx2) in enumerate(orig_idxs):
        orig_scores_dict[(int(idx1), int(idx2))] = match_scores[i].item()

    # 根据图像大小调整eps参数
    img_width = max(features_data[key1]['size'][0][0].item(), features_data[key2]['size'][0][0].item())
    eps = max(18, img_width * 0.03)  # 自适应聚类距离
    min_radius = img_width * 0.15  # 最小半径
    
    db1 = DBSCAN(eps=eps, min_samples=3).fit(mkpts1)
    db2 = DBSCAN(eps=eps, min_samples=3).fit(mkpts2)
    
    labels1 = db1.labels_.copy()
    labels2 = db2.labels_.copy()

    n = len(mkpts1)
    adj = defaultdict(set)

    # 只记录成功聚类的索引
    for i in range(n):
        if labels1[i] != -1:
            adj[f'1_{labels1[i]}'].add(i)
        if labels2[i] != -1:
            adj[f'2_{labels2[i]}'].add(i)

    # DFS 合并
    clusters = []
    visited = set()

    def dfs_rec(i, cluster):
        if i in visited:
            return
        visited.add(i)
        cluster.add(i)

        l1 = labels1[i]
        l2 = labels2[i]

        if l1 != -1:
            for j in adj[f'1_{l1}']:
                dfs(j, cluster)
        if l2 != -1:
            for j in adj[f'2_{l2}']:
                dfs(j, cluster)

    def dfs(i, cluster):
        stack = [i]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.add(curr)
            l1 = labels1[curr]
            l2 = labels2[curr]
            if l1 != -1:
                for j in adj[f'1_{l1}']:
                    if j not in visited:
                        stack.append(j)
            if l2 != -1:
                for j in adj[f'2_{l2}']:
                    if j not in visited:
                        stack.append(j)

    # 初始化最终标签为 -1
    merged_labels = -1 * np.ones(n, dtype=int)

    # 仅合并至少有一边聚类成功的点
    for i in range(n):
        if i not in visited and (labels1[i] != -1 or labels2[i] != -1):
            cluster = set()
            dfs(i, cluster)
            if len(cluster) > 0:
                clusters.append(cluster)

    # 分配新标签
    for new_label, cluster in enumerate(clusters):
        for i in cluster:
            merged_labels[i] = new_label
    # 可以将 merged_labels 应用于 mkpts1 和 mkpts2（它们是一一对应的）
    labels1 = merged_labels.copy()
    labels2 = merged_labels.copy()

    # 提取有效聚类
    valid_clusters1 = np.unique(labels1[labels1 >= 0])
    valid_clusters2 = np.unique(labels2[labels2 >= 0])
    
    # 加载所有特征点
    all_kp1 = features_data[key1]['kp'][startidx:].clone()
    all_kp2 = features_data[key2]['kp'][startidx:].clone()
    all_desc1 = features_data[key1]['desc'][startidx:,:128].clone()
    all_desc2 = features_data[key2]['desc'][startidx:,:128].clone()
    fp_maks1 = features_data[key1]['mask'].clone()[-1]
    fp_maks2 = features_data[key2]['mask'].clone()[-1]
    all_kp1 = all_kp1[:fp_maks1]
    all_kp2 = all_kp2[:fp_maks2]
    all_desc1 = all_desc1[:fp_maks1]
    all_desc2 = all_desc2[:fp_maks2]

    # 为每个聚类创建掩码，判断哪些点在聚类区域内
    all_kp1_np = all_kp1.cpu().numpy()
    all_kp2_np = all_kp2.cpu().numpy()
    
    # 默认所有点都不在区域内
    in_region_mask1 = np.zeros(len(all_kp1), dtype=bool)
    in_region_mask2 = np.zeros(len(all_kp2), dtype=bool)
    
    # 区域扩展系数 - 将聚类区域扩大
    region_expansion = 1.1
    
    cluster_centers1 = []
    cluster_centers2 = []
    cluster_radius1 = []
    cluster_radius2 = []
    # 对每个聚类，找出其中心和半径
    for cluster_id in valid_clusters1:
        cluster_points = mkpts1[labels1 == cluster_id]
        centers = np.mean(cluster_points, axis=0)
        # 计算聚类半径 (最大距离 * 扩展系数)
        distances = np.sqrt(np.sum((cluster_points - centers)**2, axis=1))
        radius = np.max(distances) * region_expansion
        radius = max(radius, min_radius)  # 确保半径至少为最小半径
        
        # 计算所有点到聚类中心的距离，并标记在扩展区域内的点
        all_distances = np.sqrt(np.sum((all_kp1_np - centers)**2, axis=1))
        in_region_mask1 |= (all_distances < radius)
        cluster_centers1.append(centers)
        cluster_radius1.append(radius)
    
    # 对第二张图像重复相同的操作
    for cluster_id in valid_clusters2:
        cluster_points = mkpts2[labels2 == cluster_id]
        centers = np.mean(cluster_points, axis=0)
        distances = np.sqrt(np.sum((cluster_points - centers)**2, axis=1))
        radius = np.max(distances) * region_expansion
        radius = max(radius, min_radius)  # 确保半径至少为最小半径

        all_distances = np.sqrt(np.sum((all_kp2_np - centers)**2, axis=1))
        in_region_mask2 |= (all_distances < radius)
        cluster_centers2.append(centers)
        cluster_radius2.append(radius)
    
    #     # 可视化聚类结果
    # if (len(valid_clusters1) > 0 or len(valid_clusters2) > 0):
    #     # 提取图像路径
    #     images_dir = os.path.dirname(os.path.dirname(features_data[key1]['size'].device.type))
    #     images_dir = '../image-matching-challenge-2025/train/stairs'
    #     img1_path = os.path.join(images_dir, key1)
    #     img2_path = os.path.join(images_dir, key2)
        
    #     # 确保可视化输出目录存在
    #     # vis_dir = os.path.join(os.path.dirname(images_dir), 'visualizations', 'clusters')
    #     vis_dir = './results/featureout/cluster'
    #     os.makedirs(vis_dir, exist_ok=True)
    #     save_path = os.path.join(vis_dir, f'{key1}_{key2}_clusters.png')
    #     if "stairs_split_1_1710453626698.png_stairs_split_1_1710453620694.png_clusters" in save_path:
    #         print("hh")
    #     # 可视化聚类
    #     visualize_clusters(
    #         img1_path, img2_path, 
    #         mkpts1, mkpts2, 
    #         labels1, labels2, 
    #         cluster_centers1, cluster_centers2, 
    #         cluster_radius1, cluster_radius2,
    #         save_path,
    #         all_kp1, all_kp2
    #     )

    # 如果没有有效聚类，返回空结果
    if len(valid_clusters1) == 0 or len(valid_clusters2) == 0:
        return torch.zeros((0,2)), torch.zeros(0)
    else:
        # 使用区域内的特征点进行第二阶段匹配
        region_kp1 = all_kp1[in_region_mask1]
        region_kp2 = all_kp2[in_region_mask2]
        region_desc1 = all_desc1[in_region_mask1]
        region_desc2 = all_desc2[in_region_mask2]
        
        # 关键：记录区域内点与原始点的索引映射关系
        region1_to_original = np.where(in_region_mask1)[0]
        region2_to_original = np.where(in_region_mask2)[0]
        
        # 确保包含原始匹配点
        mkpts1_indices = []
        for pt in mkpts1:
            # 找到与pt最接近的点在all_kp1_np中的索引
            distances = np.sum((all_kp1_np - pt)**2, axis=1)
            closest_idx = np.argmin(distances)
            mkpts1_indices.append(closest_idx)
            
        mkpts2_indices = []
        for pt in mkpts2:
            distances = np.sum((all_kp2_np - pt)**2, axis=1)
            closest_idx = np.argmin(distances)
            mkpts2_indices.append(closest_idx)
        
        # 确保这些索引在掩码中标记为True
        in_region_mask1[mkpts1_indices] = True
        in_region_mask2[mkpts2_indices] = True
        
        # 更新区域内点与原始点的索引映射关系
        region1_to_original = np.where(in_region_mask1)[0]
        region2_to_original = np.where(in_region_mask2)[0]
        
        # 重新获取区域内的特征点
        region_kp1 = all_kp1[in_region_mask1]
        region_kp2 = all_kp2[in_region_mask2]
        region_desc1 = all_desc1[in_region_mask1]
        region_desc2 = all_desc2[in_region_mask2]
        
        # 执行第二阶段匹配
        region_pred = {
            'keypoints0': region_kp1[:3072][None].to(device),
            'keypoints1': region_kp2[:3072][None].to(device),
            'descriptors0': region_desc1[:3072,:128].to(device),
            'descriptors1': region_desc2[:3072,:128].to(device),
            'size0': features_data[key1]['size'].to(device),
            'size1': features_data[key2]['size'].to(device),
            # 'scale0': features_data[key1]['scale'],
            # 'scale1': features_data[key2]['scale'],
        }

        with torch.inference_mode():
            region_dist, region_idxs = lg_matcher(region_pred['descriptors0'].float(), region_pred['descriptors1'].float(),
                KF.laf_from_center_scale_ori(region_pred['keypoints0'].float()),
                KF.laf_from_center_scale_ori(region_pred['keypoints1'].float()))
            region_valid_mask = (region_dist > 0.25)
            region_dist = region_dist[region_valid_mask[:,0]]
            region_idxs = region_idxs[region_valid_mask[:,0]]
            
        # 关键：将区域内的匹配索引映射回原始索引
        if len(region_idxs) > 0:
            # 限制区域匹配的索引范围
            valid_mask = (region_idxs[:, 0] < len(region1_to_original)) & (region_idxs[:, 1] < len(region2_to_original))
            region_idxs = region_idxs[valid_mask]
            region_scores = region_dist[valid_mask]
            
            if len(region_idxs) > 0:
                # 将区域内索引映射回原始索引
                mapped_idxs = torch.zeros_like(region_idxs)
                mapped_idxs[:, 0] = torch.tensor(region1_to_original[region_idxs[:, 0].cpu().numpy()])
                mapped_idxs[:, 1] = torch.tensor(region2_to_original[region_idxs[:, 1].cpu().numpy()])
                mapped_idxs += startidx

                # 转换为numpy进行后续处理
                mapped_idxs_np = mapped_idxs.cpu().numpy()
                
                # 区分重叠匹配和新增匹配
                refined_matches = []
                refined_scores = []
                new_matches = []
                new_scores = []
                
                for i, (idx1, idx2) in enumerate(mapped_idxs_np):
                    idx1, idx2 = int(idx1), int(idx2)
                    if (idx1) in orig_matches_set:
                        refined_matches.append([idx1, idx2])
                        refined_scores.append(region_scores[i].item())
                    else:
                        new_matches.append([idx1, idx2])
                        new_scores.append(region_scores[i].item())
                
                # 保存原始匹配中未被区域匹配覆盖的部分
                preserved_matches = []
                preserved_scores = []
                
                for i, (idx1, idx2) in enumerate(orig_idxs):
                    idx1, idx2 = int(idx1), int(idx2)
                    match_key = (idx1, idx2)
                    if match_key not in {(r[0], r[1]) for r in refined_matches}:
                        preserved_matches.append([idx1, idx2])
                        if match_key in orig_scores_dict:
                            preserved_scores.append(orig_scores_dict[match_key])
                        else:
                            preserved_scores.append(0.5)  # 默认值
                
                # 合并结果：保留的原始匹配 + 精细化匹配 + 新增匹配
                all_matches = np.array(preserved_matches + refined_matches + new_matches)
                all_scores = torch.tensor(preserved_scores + refined_scores + new_scores, device=region_dist.device)
                
                # 按照分数从大到小排序
                if len(all_scores) > 0:
                    sorted_indices = torch.argsort(all_scores, descending=True)
                    merged_idxs = torch.tensor(all_matches, device=sorted_indices.device, dtype=idxs.dtype)[sorted_indices]
                    merged_scores = all_scores[sorted_indices]
                    
                    # 应用聚类过滤并保留对应的分数
                    filtered_idxs, filtered_scores = filter_clusters_by_match_count_with_scores(
                        merged_idxs, merged_scores, features_data, key1, key2, 
                        cluster_centers1, cluster_centers2, cluster_radius1, cluster_radius2
                    )
                    
                    return filtered_idxs, filtered_scores
                
    # 如果找不到合适的区域匹配或区域匹配后没有结果，返回空结果
    return torch.zeros((0,2)), torch.zeros(0)

def match_with_gimlightglue_ensemble(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                                           device=torch.device('cpu'), min_matches=20, batch_size=2, 
                                           tok_limit=1200, match_limit=4096, verbose=True, visualize=True):
    """
    使用批处理方式进行特征匹配，点数不会超过 max_points，但可能小于。
    对于点数相同的匹配对进行批处理，点数不同的匹配对单独处理。

    Args:
        lightglue_matcher: LightGlue 匹配器实例
        img_fnames: 图像文件名列表
        index_pairs: 图像对索引列表
        feature_dir: 特征存储目录
        device: 设备 (CPU/GPU)
        min_matches: 最小匹配数
        batch_size: 批处理大小
        batch_points: 每张图像的最大点数
        verbose: 是否打印详细信息
        visualize: 是否可视化匹配结果
    """
    def lg_forward(
        lg_matcher,
        desc1,
        desc2,
        lafs1,
        lafs2,
    ):
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.
            hw1: Height/width of image.
            hw2: Height/width of image.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.

        """
        keypoints1 = get_laf_center(lafs1)
        keypoints2 = get_laf_center(lafs2)
        dev = lafs1.device

        hw1_ = keypoints1.max(dim=1)[0].squeeze().flip(0)
        hw2_ = keypoints2.max(dim=1)[0].squeeze().flip(0)
 
        ori0 = torch.deg2rad(get_laf_orientation(lafs1).reshape(1, -1))
        ori0[ori0 < 0] += 2.0 * torch.pi
        ori1 = torch.deg2rad(get_laf_orientation(lafs2).reshape(1, -1))
        ori1[ori1 < 0] += 2.0 * torch.pi
        input_dict = {
            "image0": {
                "keypoints": keypoints1,
                "scales": get_laf_scale(lafs1).reshape(1, -1),
                "oris": ori0,
                "lafs": lafs1,
                "descriptors": desc1,
                "image_size": hw1_.flip(0).reshape(-1, 2).to(dev),
            },
            "image1": {
                "keypoints": keypoints2,
                "lafs": lafs2,
                "scales": get_laf_scale(lafs2).reshape(1, -1),
                "oris": ori1,
                "descriptors": desc2,
                "image_size": hw2_.flip(0).reshape(-1, 2).to(dev),
            },
        }
        pred = lg_matcher.matcher(input_dict)
        matches0_batch, mscores0_batch = pred["matches0"], pred["matching_scores0"]
        matches0_batch_lst = []
        mscores0_batch_lst = []
        for idx, matches0 in enumerate(matches0_batch):
            valid = matches0 > -1
            matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)
            matches0_batch_lst.append(matches)
            mscores0_batch_lst.append(mscores0_batch[idx][valid])
        
        return mscores0_batch_lst, matches0_batch_lst
    
    
    # 另外保存到一个字典中，格式为{key1-key2:[idxs,scores]}
    match_dict = {}
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                        "depth_confidence": -1,
                                        "mp": True if 'cuda' in str(device) else False}).eval().to(device)

    # 加载特征数据
    print("加载特征数据...")
    features_data = {}
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
         h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
         h5py.File(f'{feature_dir}/mask.h5', mode='r') as f_mask:
        for img_path in tqdm(img_fnames):
            try:
                key = img_path.split('/')[-1].split('\\')[-1]
                features_data[key] = {
                    'kp': torch.from_numpy(f_kp[key][...]).to(device),
                    'desc': torch.from_numpy(f_desc[key][...]).to(device),
                    'size': torch.from_numpy(f_size[key][...]).to(device),
                    'scale': torch.from_numpy(f_scale[key][...]).to(device),
                    'mask': torch.from_numpy(f_mask[key][...]).to(device)
                }
            except Exception as e:
                print(f"Error loading features for {key}: {e}")
                continue

    # 将图像对按点数分组
    batch_pairs_lst = []
    single_pairs_lst = []
    for pair_idx in index_pairs:
        try:
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1 = fname1.split('/')[-1].split('\\')[-1]
            key2 = fname2.split('/')[-1].split('\\')[-1]
            batch_points = 4096
            num_points10, _  = features_data[key1]['mask']
            num_points20, _  = features_data[key2]['mask']
            if num_points10 == batch_points and num_points20 == batch_points:
                batch_pairs_lst.append(pair_idx)
            else:
                single_pairs_lst.append(pair_idx)
        except Exception as e:
            print(f"Error processing pair {pair_idx}: {e}")
            continue

    run_pairs = 0
    success_pairs = 0
    lg_finetuned = False
    # 批量处理点数相同的图像对
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        # 将图像对分成批次
        num_batches = (len(batch_pairs_lst) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(batch_pairs_lst))
            batch_pairs = batch_pairs_lst[start_idx:end_idx]
            
            batch_data = []
            batch_data_alike = []
            batch_info = []
            
            # 准备批次数据
            for pair_idx in batch_pairs:
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1 = fname1.split('/')[-1].split('\\')[-1]
                key2 = fname2.split('/')[-1].split('\\')[-1]
                try:
                    # 获取图像特征
                    kp1 = features_data[key1]['kp']
                    kp2 = features_data[key2]['kp']
                    desc1 = features_data[key1]['desc']
                    desc2 = features_data[key2]['desc']
                    num_pts_h = len(kp1)
                except Exception as e:
                    print(f"Error loading features for {key1} or {key2}: {e}")
                    continue

                pred = {
                    'keypoints0': kp1[:match_limit][None],
                    'keypoints1': kp2[:match_limit][None],
                    'descriptors0': desc1[:match_limit][None],
                    'descriptors1': desc2[:match_limit][None],
                    'size0': features_data[key1]['size'],
                    'size1': features_data[key2]['size'],
                    'scale0': features_data[key1]['scale'],
                    'scale1': features_data[key2]['scale'],
                }

                
                # pred_alike = {
                #     'keypoints0': kp1[4096:][None],
                #     'keypoints1': kp2[4096:][None],
                #     'descriptors0': desc1[4096:,:128][None],
                #     'descriptors1': desc2[4096:,:128][None],
                #     'size0': features_data[key1]['size'],
                #     'size1': features_data[key2]['size'],
                #     'scale0': features_data[key1]['scale'],
                #     'scale1': features_data[key2]['scale'],
                # }

                batch_data.append(pred)
                # batch_data_alike.append(pred_alike)
                batch_info.append((idx1, idx2, key1, key2, fname1, fname2))
            
            # 批量匹配
            # print(f"处理批次 {batch_idx+1}/{num_batches} ({len(batch_pairs)} 对图像)...")
            
            # 合并批次预测数据
            batch_preds = {
                'keypoints0': torch.cat([data['keypoints0'] for data in batch_data], dim=0).to(device),
                'keypoints1': torch.cat([data['keypoints1'] for data in batch_data], dim=0).to(device),
                'descriptors0': torch.cat([data['descriptors0'] for data in batch_data], dim=0).to(device),
                'descriptors1': torch.cat([data['descriptors1'] for data in batch_data], dim=0).to(device),
                'size0': torch.stack([data['size0'] for data in batch_data], dim=0).to(device),
                'size1': torch.stack([data['size1'] for data in batch_data], dim=0).to(device),
                'scale0': torch.stack([data['scale0'] for data in batch_data], dim=0).to(device),
                'scale1': torch.stack([data['scale1'] for data in batch_data], dim=0).to(device),
            }
            # batch_preds_alike = {
            #     'keypoints0': torch.cat([data['keypoints0'] for data in batch_data_alike], dim=0).to(device),
            #     'keypoints1': torch.cat([data['keypoints1'] for data in batch_data_alike], dim=0).to(device),
            #     'descriptors0': torch.cat([data['descriptors0'] for data in batch_data_alike], dim=0).to(device),
            #     'descriptors1': torch.cat([data['descriptors1'] for data in batch_data_alike], dim=0).to(device),
            #     'size0': torch.stack([data['size0'] for data in batch_data_alike], dim=0).to(device),
            #     'size1': torch.stack([data['size1'] for data in batch_data_alike], dim=0).to(device),
            #     'scale0': torch.stack([data['scale0'] for data in batch_data_alike], dim=0).to(device),
            #     'scale1': torch.stack([data['scale1'] for data in batch_data_alike], dim=0).to(device),
            # }
            # # 合并批次预测数据
            # batch_preds_finetune = {
            #     'keypoints0': torch.cat([data['keypoints0'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'keypoints1': torch.cat([data['keypoints1'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'descriptors0': torch.cat([data['descriptors0'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'descriptors1': torch.cat([data['descriptors1'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'size0': torch.stack([data['size0'] for data in batch_data], dim=0).to(device),
            #     'size1': torch.stack([data['size1'] for data in batch_data], dim=0).to(device),
            #     'scale0': torch.stack([data['scale0'] for data in batch_data], dim=0).to(device),
            #     'scale1': torch.stack([data['scale1'] for data in batch_data], dim=0).to(device),
            # }

            # 批量推理
            with torch.inference_mode():
                batch_dists, batch_idxs = lightglue_matcher.match_batch(batch_preds)
                # batch_dists_fine, batch_idxs_fine = lightglue_matcher.match_batch_finetune(batch_preds_finetune)
                # batch_dists_fine, batch_idxs_fine = lg_forward(lg_matcher, batch_preds_alike['descriptors0'].float(), batch_preds_alike['descriptors1'].float(),
                #         KF.laf_from_center_scale_ori(batch_preds_alike['keypoints0'].float()),
                #         KF.laf_from_center_scale_ori(batch_preds_alike['keypoints1'].float()))
            
            batch_dists_fine = []
            batch_idxs_fine = []
            for i, (idx1, idx2, key1, key2, fname1, fname2) in enumerate(batch_info):
                    kp1 = features_data[key1]['kp']
                    kp2 = features_data[key2]['kp']
                    desc1 = features_data[key1]['desc']
                    desc2 = features_data[key2]['desc']
                    mask1_alike = features_data[key1]['mask'][-1]
                    mask2_alike = features_data[key2]['mask'][-1]
                    pred_alike = {
                        'keypoints0': kp1[4096:][:mask1_alike][None],
                        'keypoints1': kp2[4096:][:mask2_alike][None],
                        'descriptors0': desc1[4096:,:128][:mask1_alike],
                        'descriptors1': desc2[4096:,:128][:mask2_alike],
                    }

                    # 批量推理
                    with torch.inference_mode():
                        dists_finetune, idxs_finetune = lg_matcher(pred_alike['descriptors0'].float(), pred_alike['descriptors1'].float(),
                            KF.laf_from_center_scale_ori(pred_alike['keypoints0'].float()),
                            KF.laf_from_center_scale_ori(pred_alike['keypoints1'].float()))
                        dists_finetune = dists_finetune[:,0]
                        idxs_finetune += 4096
                        dists_finetune_mask = dists_finetune > 0.2
                        idxs_finetune = idxs_finetune[dists_finetune_mask]
                        dists_finetune = dists_finetune[dists_finetune_mask]
                        batch_dists_fine.append(dists_finetune)
                        batch_idxs_fine.append(idxs_finetune)
                    
            # 对 batch_idxs 按照 batch_dists 分数排序并保留最大的 1500 个匹配
            sorted_idxs = []
            sorted_dists = []
            for i in range(len(batch_dists)):
                if len(batch_dists[i]) > 0 or len(batch_dists_fine[i]) > 0:
                    dists = torch.cat([batch_dists[i],batch_dists_fine[i]])
                    idxs = torch.cat([batch_idxs[i],batch_idxs_fine[i]])

                    # dists = batch_dists[i]
                    # idxs = batch_idxs[i]
                    dists, idxs = match_nms(dists, idxs, batch_info[i], features_data, 1)
                    sorted_indices = torch.argsort(dists, descending=True)
                    sorted_dists_one = dists[sorted_indices]
                    sorted_idxs_one = idxs[sorted_indices]
                    top_k = min(tok_limit, len(sorted_dists_one))
                    sorted_idxs.append(sorted_idxs_one[:top_k])
                    sorted_dists.append(sorted_dists_one[:top_k])
                else:
                    sorted_idxs.append([])
                    sorted_dists.append([])

            batch_idxs = sorted_idxs   
            batch_dists = sorted_dists
            # 处理结果
            for i, (idx1, idx2, key1, key2, fname1, fname2) in enumerate(batch_info):
                run_pairs += 1
                if i >= len(batch_idxs) or batch_idxs[i] is None or len(batch_idxs[i]) == 0:
                    continue
                
                idxs = batch_idxs[i]
                match_scores = batch_dists[i]
                
                if verbose:
                    print(f'{key1}-{key2}: {n_matches} matches')
                
                try:
                    if len(idxs) < 800:
                        # # 进行第二阶段匹配
                        mkpts1 = features_data[key1]['kp'][idxs[:,0]]
                        mkpts2 = features_data[key2]['kp'][idxs[:,1]]
                        # 进行第二阶段匹配
                        region_idxs, region_dists = second_match_ensemble(mkpts1.cpu().numpy(), mkpts2.cpu().numpy(), idxs, match_scores, features_data, key1, key2, lg_matcher)
                        print(f'{key1}-{key2}')
                        print("region_dists:", len(idxs), len(region_idxs))
                        idxs = region_idxs[:1500]
                        match_scores = region_dists[:1500]
                except Exception as e:
                    print(f"Error in second match: {e}")
                    continue
                n_matches = len(idxs)
                # 保存匹配结果
                if n_matches >= min_matches:
                    # kpts0 = features_data[key1]['kp'][idxs[:,0]]
                    # kpts1 = features_data[key2]['kp'][idxs[:,1]]
                    # # robust fitting
                    # _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                    #                                 kpts1.cpu().detach().numpy(),
                    #                                 cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                    #                                 confidence=0.999999, maxIters=10000)
                    # mask = mask.ravel() > 0
                    # idxs = idxs[mask]
                    if len(idxs) >= min_matches:
                        group = f_match.require_group(key1)
                        group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                        match_matrix[idx1, idx2] = n_matches
                        
                        match_key = f"{key1}-{key2}"
                        match_dict[match_key] = [idxs.detach().cpu().numpy(), match_scores.detach().cpu().numpy()]
                        
                        success_pairs += 1
                            
                        # 可视化匹配
                        if visualize:
                            vis_dir = os.path.join(feature_dir, 'visualizations')
                            os.makedirs(vis_dir, exist_ok=True)
                            save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                            visualize_matches(
                                fname1, fname2,
                                features_data[key1]['kp'].cpu().numpy(),
                                features_data[key2]['kp'].cpu().numpy(),
                                idxs.cpu().numpy(),
                                save_path
                            )

    with open(os.path.join(feature_dir, 'match_dict.pkl'), 'wb') as f:
        pickle.dump(match_dict, f)

    return match_matrix

def match_with_gimlightglue_ensemble_withfine(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                                           device=torch.device('cpu'), min_matches=20, batch_size=4, 
                                           tok_limit=1200, match_limit=4096, verbose=True, visualize=True):
    """
    使用批处理方式进行特征匹配，点数不会超过 max_points，但可能小于。
    对于点数相同的匹配对进行批处理，点数不同的匹配对单独处理。

    Args:
        lightglue_matcher: LightGlue 匹配器实例
        img_fnames: 图像文件名列表
        index_pairs: 图像对索引列表
        feature_dir: 特征存储目录
        device: 设备 (CPU/GPU)
        min_matches: 最小匹配数
        batch_size: 批处理大小
        batch_points: 每张图像的最大点数
        verbose: 是否打印详细信息
        visualize: 是否可视化匹配结果
    """
    def lg_forward(
        lg_matcher,
        desc1,
        desc2,
        lafs1,
        lafs2,
    ):
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.
            hw1: Height/width of image.
            hw2: Height/width of image.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.

        """
        keypoints1 = get_laf_center(lafs1)
        keypoints2 = get_laf_center(lafs2)
        dev = lafs1.device

        hw1_ = keypoints1.max(dim=1)[0].squeeze().flip(0)
        hw2_ = keypoints2.max(dim=1)[0].squeeze().flip(0)
 
        ori0 = torch.deg2rad(get_laf_orientation(lafs1).reshape(1, -1))
        ori0[ori0 < 0] += 2.0 * torch.pi
        ori1 = torch.deg2rad(get_laf_orientation(lafs2).reshape(1, -1))
        ori1[ori1 < 0] += 2.0 * torch.pi
        input_dict = {
            "image0": {
                "keypoints": keypoints1,
                "scales": get_laf_scale(lafs1).reshape(1, -1),
                "oris": ori0,
                "lafs": lafs1,
                "descriptors": desc1,
                "image_size": hw1_.flip(0).reshape(-1, 2).to(dev),
            },
            "image1": {
                "keypoints": keypoints2,
                "lafs": lafs2,
                "scales": get_laf_scale(lafs2).reshape(1, -1),
                "oris": ori1,
                "descriptors": desc2,
                "image_size": hw2_.flip(0).reshape(-1, 2).to(dev),
            },
        }
        pred = lg_matcher.matcher(input_dict)
        matches0_batch, mscores0_batch = pred["matches0"], pred["matching_scores0"]
        matches0_batch_lst = []
        mscores0_batch_lst = []
        for idx, matches0 in enumerate(matches0_batch):
            valid = matches0 > -1
            matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)
            matches0_batch_lst.append(matches)
            mscores0_batch_lst.append(mscores0_batch[idx][valid])
        
        return mscores0_batch_lst, matches0_batch_lst
    
    
    # 另外保存到一个字典中，格式为{key1-key2:[idxs,scores]}
    match_dict = {}
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                        "depth_confidence": -1,
                                        "mp": True if 'cuda' in str(device) else False}).eval().to(device)

    # 加载特征数据
    print("加载特征数据...")
    features_data = {}
    load_gpu_first = len(img_fnames) < 500
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
         h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
         h5py.File(f'{feature_dir}/mask.h5', mode='r') as f_mask:
        for img_path in tqdm(img_fnames):
            try:
                key = img_path.split('/')[-1].split('\\')[-1]
                if load_gpu_first:
                    features_data[key] = {
                        'kp': torch.from_numpy(f_kp[key][...]).to(device),
                        'desc': torch.from_numpy(f_desc[key][...]).to(device),
                        'size': torch.from_numpy(f_size[key][...]).to(device),
                        'scale': torch.from_numpy(f_scale[key][...]).to(device),
                        'mask': torch.from_numpy(f_mask[key][...]).to(device)
                    }
                else:
                    features_data[key] = {
                        'kp': torch.from_numpy(f_kp[key][...]),
                        'desc': torch.from_numpy(f_desc[key][...]),
                        'size': torch.from_numpy(f_size[key][...]),
                        'scale': torch.from_numpy(f_scale[key][...]),
                        'mask': torch.from_numpy(f_mask[key][...])
                    }
            except Exception as e:
                print(f"Error loading features for {key}: {e}")
                continue

    # 将图像对按点数分组
    batch_pairs_lst = []
    single_pairs_lst = []
    for pair_idx in index_pairs:
        try:
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1 = fname1.split('/')[-1].split('\\')[-1]
            key2 = fname2.split('/')[-1].split('\\')[-1]
            batch_points = 4096
            num_points10, _  = features_data[key1]['mask']
            num_points20, _  = features_data[key2]['mask']
            if num_points10 == batch_points and num_points20 == batch_points:
                batch_pairs_lst.append(pair_idx)
            else:
                single_pairs_lst.append(pair_idx)
        except Exception as e:
            print(f"Error processing pair {pair_idx}: {e}")
            continue

    run_pairs = 0
    success_pairs = 0
    lg_finetuned = False
    # 批量处理点数相同的图像对
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        # 将图像对分成批次
        num_batches = (len(batch_pairs_lst) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(batch_pairs_lst))
            batch_pairs = batch_pairs_lst[start_idx:end_idx]
            
            batch_data = []
            batch_data_alike = []
            batch_info = []
            
            # 准备批次数据
            for pair_idx in batch_pairs:
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1 = fname1.split('/')[-1].split('\\')[-1]
                key2 = fname2.split('/')[-1].split('\\')[-1]
                try:
                    # 获取图像特征
                    kp1 = features_data[key1]['kp']
                    kp2 = features_data[key2]['kp']
                    desc1 = features_data[key1]['desc']
                    desc2 = features_data[key2]['desc']
                    num_pts_h = len(kp1)
                except Exception as e:
                    print(f"Error loading features for {key1} or {key2}: {e}")
                    continue

                pred = {
                    'keypoints0': kp1[:match_limit][None],
                    'keypoints1': kp2[:match_limit][None],
                    'descriptors0': desc1[:match_limit][None],
                    'descriptors1': desc2[:match_limit][None],
                    'size0': features_data[key1]['size'],
                    'size1': features_data[key2]['size'],
                    'scale0': features_data[key1]['scale'],
                    'scale1': features_data[key2]['scale'],
                }

                
                # pred_alike = {
                #     'keypoints0': kp1[4096:][None],
                #     'keypoints1': kp2[4096:][None],
                #     'descriptors0': desc1[4096:,:128][None],
                #     'descriptors1': desc2[4096:,:128][None],
                #     'size0': features_data[key1]['size'],
                #     'size1': features_data[key2]['size'],
                #     'scale0': features_data[key1]['scale'],
                #     'scale1': features_data[key2]['scale'],
                # }

                batch_data.append(pred)
                # batch_data_alike.append(pred_alike)
                batch_info.append((idx1, idx2, key1, key2, fname1, fname2))
            
            # 批量匹配
            # print(f"处理批次 {batch_idx+1}/{num_batches} ({len(batch_pairs)} 对图像)...")
            
            # 合并批次预测数据
            batch_preds = {
                'keypoints0': torch.cat([data['keypoints0'] for data in batch_data], dim=0).to(device),
                'keypoints1': torch.cat([data['keypoints1'] for data in batch_data], dim=0).to(device),
                'descriptors0': torch.cat([data['descriptors0'] for data in batch_data], dim=0).to(device),
                'descriptors1': torch.cat([data['descriptors1'] for data in batch_data], dim=0).to(device),
                'size0': torch.stack([data['size0'] for data in batch_data], dim=0).to(device),
                'size1': torch.stack([data['size1'] for data in batch_data], dim=0).to(device),
                'scale0': torch.stack([data['scale0'] for data in batch_data], dim=0).to(device),
                'scale1': torch.stack([data['scale1'] for data in batch_data], dim=0).to(device),
            }
            # batch_preds_alike = {
            #     'keypoints0': torch.cat([data['keypoints0'] for data in batch_data_alike], dim=0).to(device),
            #     'keypoints1': torch.cat([data['keypoints1'] for data in batch_data_alike], dim=0).to(device),
            #     'descriptors0': torch.cat([data['descriptors0'] for data in batch_data_alike], dim=0).to(device),
            #     'descriptors1': torch.cat([data['descriptors1'] for data in batch_data_alike], dim=0).to(device),
            #     'size0': torch.stack([data['size0'] for data in batch_data_alike], dim=0).to(device),
            #     'size1': torch.stack([data['size1'] for data in batch_data_alike], dim=0).to(device),
            #     'scale0': torch.stack([data['scale0'] for data in batch_data_alike], dim=0).to(device),
            #     'scale1': torch.stack([data['scale1'] for data in batch_data_alike], dim=0).to(device),
            # }
            # # 合并批次预测数据
            # batch_preds_finetune = {
            #     'keypoints0': torch.cat([data['keypoints0'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'keypoints1': torch.cat([data['keypoints1'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'descriptors0': torch.cat([data['descriptors0'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'descriptors1': torch.cat([data['descriptors1'][:,:1024] for data in batch_data], dim=0).to(device),
            #     'size0': torch.stack([data['size0'] for data in batch_data], dim=0).to(device),
            #     'size1': torch.stack([data['size1'] for data in batch_data], dim=0).to(device),
            #     'scale0': torch.stack([data['scale0'] for data in batch_data], dim=0).to(device),
            #     'scale1': torch.stack([data['scale1'] for data in batch_data], dim=0).to(device),
            # }

            # 批量推理
            with torch.inference_mode():
                batch_dists, batch_idxs = lightglue_matcher.match_batch(batch_preds)
                # batch_dists_fine, batch_idxs_fine = lightglue_matcher.match_batch_finetune(batch_preds_finetune)
                # batch_dists_fine, batch_idxs_fine = lg_forward(lg_matcher, batch_preds_alike['descriptors0'].float(), batch_preds_alike['descriptors1'].float(),
                #         KF.laf_from_center_scale_ori(batch_preds_alike['keypoints0'].float()),
                #         KF.laf_from_center_scale_ori(batch_preds_alike['keypoints1'].float()))
            
            batch_dists_fine = []
            batch_idxs_fine = []
            for i, (idx1, idx2, key1, key2, fname1, fname2) in enumerate(batch_info):
                    kp1 = features_data[key1]['kp']
                    kp2 = features_data[key2]['kp']
                    desc1 = features_data[key1]['desc']
                    desc2 = features_data[key2]['desc']
                    mask1_alike = features_data[key1]['mask'][-1]
                    mask2_alike = features_data[key2]['mask'][-1]
                    pred_alike = {
                        'keypoints0': kp1[4096:][:mask1_alike][None].to(device),
                        'keypoints1': kp2[4096:][:mask2_alike][None].to(device),
                        'descriptors0': desc1[4096:,:128][:mask1_alike].to(device),
                        'descriptors1': desc2[4096:,:128][:mask2_alike].to(device),
                    }

                    # 批量推理
                    # with torch.inference_mode(), torch.amp.autocast('cuda'):
                    with torch.inference_mode():
                        dists_finetune, idxs_finetune = lg_matcher(pred_alike['descriptors0'].float(), pred_alike['descriptors1'].float(),
                            KF.laf_from_center_scale_ori(pred_alike['keypoints0'].float()),
                            KF.laf_from_center_scale_ori(pred_alike['keypoints1'].float()))
                        dists_finetune = dists_finetune[:,0]
                        idxs_finetune += 4096
                        dists_finetune_mask = dists_finetune > 0.2
                        idxs_finetune = idxs_finetune[dists_finetune_mask]
                        dists_finetune = dists_finetune[dists_finetune_mask]
                        batch_dists_fine.append(dists_finetune)
                        batch_idxs_fine.append(idxs_finetune)
                    
            # 对 batch_idxs 按照 batch_dists 分数排序并保留最大的 1500 个匹配
            sorted_idxs = []
            sorted_dists = []
            for i in range(len(batch_dists)):
                if len(batch_dists[i]) > 0 or len(batch_dists_fine[i]) > 0:
                    dists = torch.cat([batch_dists[i],batch_dists_fine[i]])
                    idxs = torch.cat([batch_idxs[i],batch_idxs_fine[i]])

                    # dists = batch_dists[i]
                    # idxs = batch_idxs[i]
                    dists, idxs = match_nms(dists, idxs, batch_info[i], features_data, 1)
                    sorted_indices = torch.argsort(dists, descending=True)
                    sorted_dists_one = dists[sorted_indices]
                    sorted_idxs_one = idxs[sorted_indices]
                    top_k = min(tok_limit, len(sorted_dists_one))
                    sorted_idxs.append(sorted_idxs_one[:top_k])
                    sorted_dists.append(sorted_dists_one[:top_k])
                else:
                    sorted_idxs.append([])
                    sorted_dists.append([])

            batch_idxs = sorted_idxs   
            batch_dists = sorted_dists
            # 处理结果
            for i, (idx1, idx2, key1, key2, fname1, fname2) in enumerate(batch_info):
                run_pairs += 1
                if i >= len(batch_idxs) or batch_idxs[i] is None or len(batch_idxs[i]) == 0:
                    continue
                
                idxs = batch_idxs[i]
                match_scores = batch_dists[i]
                
                if verbose:
                    print(f'{key1}-{key2}: {n_matches} matches')
                
                try:
                    if len(idxs) < 800:
                        # # 进行第二阶段匹配
                        mkpts1 = features_data[key1]['kp'][idxs[:,0]]
                        mkpts2 = features_data[key2]['kp'][idxs[:,1]]
                        # 进行第二阶段匹配
                        region_idxs, region_dists = second_match_ensemble(mkpts1.cpu().numpy(), mkpts2.cpu().numpy(), idxs, match_scores, features_data, key1, key2, lg_matcher, device)
                        # print(f'{key1}-{key2}')
                        # print("region_dists:", len(idxs), len(region_idxs))
                        idxs = region_idxs[:1500]
                        match_scores = region_dists[:1500]
                except Exception as e:
                    print(f"Error in second match: {e}")
                    continue
                n_matches = len(idxs)
                # 保存匹配结果
                if n_matches >= min_matches:
                    # kpts0 = features_data[key1]['kp'][idxs[:,0]]
                    # kpts1 = features_data[key2]['kp'][idxs[:,1]]
                    # # robust fitting
                    # _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                    #                                 kpts1.cpu().detach().numpy(),
                    #                                 cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                    #                                 confidence=0.999999, maxIters=10000)
                    # mask = mask.ravel() > 0
                    # idxs = idxs[mask]
                    if len(idxs) >= min_matches:
                        group = f_match.require_group(key1)
                        group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                        match_matrix[idx1, idx2] = n_matches
                        
                        match_key = f"{key1}-{key2}"
                        match_dict[match_key] = [idxs.detach().cpu().numpy(), match_scores.detach().cpu().numpy()]
                        
                        success_pairs += 1

    with open(os.path.join(feature_dir, 'match_dict.pkl'), 'wb') as f:
        pickle.dump(match_dict, f)

    return match_matrix

def match_nms(dists, idxs, batch_info, features_data, radius=3):
    """
    对匹配点对进行非极大值抑制，删除距离过近的冗余匹配
    
    Args:
        dists: 匹配点对的置信度分数
        idxs: 匹配点对索引 (Nx2)
        batch_info: 图像信息元组 (idx1, idx2, key1, key2, fname1, fname2)
        features_data: 特征数据字典
        radius: NMS半径，用于确定匹配点是否过近
        
    Returns:
        filtered_dists: 过滤后的置信度分数
        filtered_idxs: 过滤后的匹配点对索引
    """
    if len(idxs) == 0:
        return dists, idxs
    
    # 获取图像标识符
    _, _, key1, key2, _, _ = batch_info
    
    # 获取匹配点坐标
    kp1 = features_data[key1]['kp']
    kp2 = features_data[key2]['kp']
    
    # 确保数据在CPU上
    if isinstance(dists, torch.Tensor):
        dists = dists.cpu()
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.cpu()
    if isinstance(kp1, torch.Tensor):
        kp1 = kp1.cpu()
    if isinstance(kp2, torch.Tensor):
        kp2 = kp2.cpu()
    
    # 获取匹配点的坐标
    pts1 = kp1[idxs[:, 0]]
    pts2 = kp2[idxs[:, 1]]
    
    # 按置信度排序
    sorted_indices = torch.argsort(dists, descending=True)
    sorted_dists = dists[sorted_indices]
    sorted_idxs = idxs[sorted_indices]
    sorted_pts1 = pts1[sorted_indices]
    sorted_pts2 = pts2[sorted_indices]
    
    # 初始化保留标志
    keep = torch.ones(len(sorted_dists), dtype=torch.bool)
    
    # 执行NMS
    for i in range(len(sorted_dists)):
        if not keep[i]:
            continue
            
        # 计算当前点与其他点的欧氏距离
        dist1 = torch.sqrt(torch.sum((sorted_pts1[i+1:] - sorted_pts1[i].unsqueeze(0))**2, dim=1))
        dist2 = torch.sqrt(torch.sum((sorted_pts2[i+1:] - sorted_pts2[i].unsqueeze(0))**2, dim=1))
        
        # 如果两张图像中的点都在半径内，则标记为抑制
        suppress = (dist1 < radius) & (dist2 < radius)
        keep[i+1:][suppress] = False
    
    # 应用过滤
    filtered_dists = sorted_dists[keep]
    filtered_idxs = sorted_idxs[keep]
    
    # 确保返回与输入相同的设备
    if isinstance(dists, torch.Tensor):
        filtered_dists = filtered_dists.to(dists.device)
        filtered_idxs = filtered_idxs.to(idxs.device)
    
    return filtered_dists, filtered_idxs

def match_with_lightglue(img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=20, verbose=True, visualize=True):
    match_dict = {}
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                              "depth_confidence": -1,
                                              "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            key1, key2 = key1.split('\\')[-1], key2.split('\\')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1, desc2,
                                       KF.laf_from_center_scale_ori(kp1[None]),
                                       KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs) == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                                # 添加可视化
                
                match_key = f"{key1}-{key2}"
                match_dict[match_key] = [idxs.detach().cpu().numpy(), dists.detach().cpu().numpy()]
                        
                if visualize:
                    vis_dir = os.path.join(feature_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    visualize_matches(fname1, fname2, 
                                   kp1.cpu().numpy(), 
                                   kp2.cpu().numpy(),
                                   idxs.cpu().numpy(),
                                   save_path)
    with open(os.path.join(feature_dir, 'match_dict.pkl'), 'wb') as f:
        pickle.dump(match_dict, f)
    return

class PointIndexer:
    def __init__(self, coord_tolerance=1.5):
        """
        coord_tolerance: 匹配坐标时的精度容忍度（单位：像素），例如 0.5 表示将坐标四舍五入到 0.5 像素内。
        """
        self.coord_tolerance = coord_tolerance

        self.image_dict = {}          # image_key -> point3D_id
        self.image_point_index = {}     # image_key -> list of (x, y)，参与匹配的原始点
        self.coord_hash = {}            # (image_key, rounded_x, rounded_y) -> point3D_id
        self.point3D_id_map = {}        # (image_key, x, y) -> point3D_id（原始坐标）

    def _round_coord(self, coord):
        """将浮点坐标按容忍精度归一化（用于索引）"""
        return tuple(np.round(np.array(coord) / self.coord_tolerance).astype(int))

    def process_match(self, key1, key2, mkpts0, mkpts1):
        """
        key1/key2: 图像名（str）
        mkpts0/mkpts1: 匹配点数组 (N,2)，为 numpy array 或 torch tensor
        return: (N,2) 的全局 point3D_id 对
        """
        if isinstance(mkpts0, torch.Tensor):
            mkpts0 = mkpts0.cpu().numpy()
        if isinstance(mkpts1, torch.Tensor):
            mkpts1 = mkpts1.cpu().numpy()

        match_indices = []
        for pt0, pt1 in zip(mkpts0, mkpts1):
            id0 = self._get_or_assign_index(key1, pt0)
            id1 = self._get_or_assign_index(key2, pt1)
            match_indices.append([id0, id1])
        return np.array(match_indices, dtype=np.int32)

    def _get_or_assign_index(self, image_key, coord):
        """
        对于一个图像的某个坐标，查找或分配 point3D_id。
        coord: (x, y)
        """
        rounded = self._round_coord(coord)
        coord = (rounded[0] * self.coord_tolerance, rounded[1] * self.coord_tolerance)
        hash_key = (image_key, rounded[0], rounded[1])

        if image_key not in self.image_dict:
            self.image_dict[image_key] = 0
        if hash_key in self.coord_hash:
            return self.coord_hash[hash_key]
        else:
            self.coord_hash[hash_key] = self.image_dict[image_key]
            self.image_dict[image_key] += 1
            self.image_point_index.setdefault(image_key, []).append((float(coord[0]), float(coord[1])))
            return self.coord_hash[hash_key]

def match_with_gimloftr(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    indexer = PointIndexer()
    with h5py.File(f'{feature_dir}/feat_c.h5', mode='r') as f_c, \
        h5py.File(f'{feature_dir}/feat_f.h5', mode='r') as f_f, \
        h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
        h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            key1 = key1.split('\\')[-1]
            key2 = key2.split('\\')[-1]
            feat_c1 = torch.from_numpy(f_c[key1][...]).to(device)
            feat_c2 = torch.from_numpy(f_c[key2][...]).to(device)
            feat_f1 = torch.from_numpy(f_f[key1][...]).to(device)
            feat_f2 = torch.from_numpy(f_f[key2][...]).to(device)
            pred = {}
            pred['feat_c0'] = feat_c1
            pred['feat_c1'] = feat_c2
            pred['feat_f0'] = feat_f1
            pred['feat_f1'] = feat_f2
            pred['hw0_i'] = torch.from_numpy(f_size[key1][...]).to(device)
            pred['hw1_i'] = torch.from_numpy(f_size[key2][...]).to(device)
            pred['scale0'] = torch.from_numpy(f_scale[key1][...]).to(device)
            pred['scale1'] = torch.from_numpy(f_scale[key2][...]).to(device)
            with torch.inference_mode():
                dists, kpts0, kpts1 = lightglue_matcher.loftr_match(pred)
            
            # if len(idxs) == 0:
            #     continue
            if kpts0 is None or len(kpts0) == 0:
                continue
            
            idxs = indexer.process_match(key1, key2, kpts0.cpu().numpy(), kpts1.cpu().numpy())    
            #  # 应用区域筛选方法
            # filtered_idxs = adaptive_match_filtering(
            #     lightglue_matcher, kp1, kp2, idxs.cpu().numpy(), fname1, fname2, device
            # )
            # # 转回tensor
            # if isinstance(filtered_idxs, np.ndarray):
            #     idxs = torch.from_numpy(filtered_idxs).to(idxs.device)

            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.reshape(-1, 2))
                match_matrix[idx1,idx2] = len(idxs.reshape(-1, 2))
                                # 添加可视化
                # # break
                if visualize:
                    vis_dir = os.path.join(feature_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    visualize_matches(fname1, fname2, 
                                   kpts0.cpu().numpy(), 
                                   kpts1.cpu().numpy(),
                                   np.stack((np.arange(0,len(kpts0)),np.arange(0,len(kpts0))),axis=1),
                                   save_path)

    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for image_key, coords in indexer.image_point_index.items():
            pts = np.array(coords).astype(np.float32)
            # print(f'Image {image_key}: {len(pts)} points')
            f_kp[image_key] = pts
            
    return match_matrix

def remove_matches_from_h5(matches_file_path, pairs_to_remove):
    """
    从匹配的 h5 文件中删除特定的匹配对
    
    参数:
        matches_file_path: 匹配 h5 文件的路径
        pairs_to_remove: 要删除的匹配对列表 [(key1, key2), ...]
    
    返回:
        保存新的 h5 文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(matches_file_path):
        raise FileNotFoundError(f"文件 {matches_file_path} 不存在")
    
    # 将pairs_to_remove转换为集合以加快查找
    pairs_set = set((pair[0], pair[1]) for pair in pairs_to_remove)
    
    # 创建一个临时文件路径
    temp_file_path = matches_file_path + '.temp'
    
    # 读取原始文件并创建新文件
    with h5py.File(matches_file_path, 'r') as src_file, h5py.File(temp_file_path, 'w') as dst_file:
        # 统计匹配对总数
        total_pairs = sum(len(src_file[key1].keys()) for key1 in src_file.keys())
        
        removed_count = 0
        print(f"开始处理匹配文件，共有 {len(src_file.keys())} 个源图像")
        
        # 遍历所有key1
        for key1 in tqdm(src_file.keys()):
            # 为每个key1创建组
            group = dst_file.require_group(key1)
            # 遍历所有key2
            for key2 in src_file[key1].keys():
                # 检查当前匹配对是否在要删除的列表中
                if (key1, key2) in pairs_set:
                    removed_count += 1
                    continue  # 跳过此匹配对
                
                # 复制匹配数据到新文件
                src_file.copy(f"{key1}/{key2}", group)
        
        print(f"成功删除了 {removed_count} 个匹配对，占总数的 {removed_count/total_pairs*100:.2f}%")
    
    # 备份原文件
    backup_file_path = matches_file_path + '.bak'
    os.rename(matches_file_path, backup_file_path)
    
    # 将临时文件重命名为原文件名
    os.rename(temp_file_path, matches_file_path)
    
    print(f"原文件已备份为 {backup_file_path}")
    print(f"新匹配文件已保存为 {matches_file_path}")
    
    return matches_file_path

def import_into_colmap(img_dir, feature_dir='.featureout', database_path='colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
    return

def compare_reconstruction_models(maps1, maps2, images):
    """比较两个重建模型的质量，返回更优的模型"""
    def evaluate_model(maps):
        if not maps:
            return 0, 0, 0, 0, 0
        
        total_registered = 0
        total_clusters = len(maps)
        avg_track_length = 0
        total_3d_points = 0
        avg_reprojection_error = 0
        
        for map_index, cur_map in maps.items():
            total_registered += len(cur_map.images)
            total_3d_points += len(cur_map.points3D)
            
            # 计算平均track长度
            if hasattr(cur_map, 'points3D') and cur_map.points3D:
                track_lengths = [len(point.track.elements) for point in cur_map.points3D.values()]
                if track_lengths:
                    avg_track_length += sum(track_lengths) / len(track_lengths)
                
                # 计算平均重投影误差
                errors = [point.error for point in cur_map.points3D.values()]
                if errors:
                    avg_reprojection_error += sum(errors) / len(errors)
        
        if total_clusters > 0:
            avg_track_length /= total_clusters
            avg_reprojection_error /= total_clusters
            
        return total_registered, total_clusters, avg_track_length, total_3d_points, avg_reprojection_error
    
    reg1, clusters1, track_len1, points1, error1 = evaluate_model(maps1)
    reg2, clusters2, track_len2, points2, error2 = evaluate_model(maps2)
    
    # 改进的评分策略
    # 1. 注册图像数量权重最高
    # 2. track长度很重要，但要防止异常值
    # 3. 聚类数量考虑合理性（1-3个为正常范围）
    # 4. 3D点数量反映重建密度
    # 5. 重投影误差反映精度（越小越好）
    
    # 聚类数量惩罚（1-3个聚类为正常，超过3个给予惩罚）
    cluster_penalty1 = max(0, clusters1 - 3) * 5
    cluster_penalty2 = max(0, clusters2 - 3) * 5
    
    # track长度标准化（防止异常值影响）
    normalized_track1 = min(track_len1, 10)  # 限制最大值为10
    normalized_track2 = min(track_len2, 10)
    
    # 重投影误差标准化（误差越小越好）
    error_score1 = max(0, 10 - error1) if error1 > 0 else 0
    error_score2 = max(0, 10 - error2) if error2 > 0 else 0
    
    score1 = (reg1 * 2 +                    # 注册图像数量（最高权重）
              normalized_track1 * 5 +         # 平均track长度
              points1 * 0.01 +               # 3D点数量
              error_score1 * 5 -             # 重投影误差（转为正向分数）
              cluster_penalty1)              # 聚类数量惩罚
              
    score2 = (reg2 * 2 + 
              normalized_track2 * 5 + 
              points2 * 0.01 + 
              error_score2 * 5 - 
              cluster_penalty2)
    
    print(f"Model 1: {reg1} registered, {clusters1} clusters, track: {track_len1:.2f}, "
          f"points: {points1}, error: {error1:.2f}, score: {score1:.2f}")
    print(f"Model 2: {reg2} registered, {clusters2} clusters, track: {track_len2:.2f}, "
          f"points: {points2}, error: {error2:.2f}, score: {score2:.2f}")
    
    return maps1 if score1 >= score2 else maps2

def incremental_mapping_with_monitoring(database_path, image_path, output_path, mapper_options, images_num,
                                       time_threshold=30.0, max_attempts=2):
    """
    带监控的增量建图，如果运行时间小于阈值则重新建图并比较结果
    
    Args:
        database_path: COLMAP数据库路径
        images_dir: 图像目录
        output_path: 输出路径
        mapper_options: 建图选项
        time_threshold: 时间阈值（秒），小于此值则重新建图
        max_attempts: 最大尝试次数
    
    Returns:
        best_maps: 最优的建图结果
        total_time: 总建图时间
        num_attempts: 实际尝试次数
    """
    best_maps = None
    total_time = 0
    attempt = 1

    images_dir = image_path
    
    print(f"开始第 {attempt} 次建图...")
    
    # 第一次建图
    t_start = time()
    os.makedirs(f"{output_path}_attempt_{attempt}", exist_ok=True)
    
    try:
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=f"{output_path}_attempt_{attempt}",
            options=mapper_options
        )
        first_time = time() - t_start
        total_time += first_time
        best_maps = maps
        
        print(f"第 {attempt} 次建图完成，耗时: {first_time:.2f}秒")
        print(f"注册图像数: {sum(len(m.images) for m in maps.values()) if maps else 0}")
        print(f"聚类数: {len(maps) if maps else 0}")
        
        # 检查是否需要重新建图
        if first_time < time_threshold and attempt < max_attempts and images_num < 60:
            print(f"建图时间 {first_time:.2f}秒 小于阈值 {time_threshold}秒，进行第二次建图...")
            
            attempt += 1
            print(f"开始第 {attempt} 次建图...")
            
            # 修改建图参数以获得不同结果
            modified_options = deepcopy(mapper_options)
            
            # 调整参数策略
            if hasattr(modified_options.mapper, 'init_min_tri_angle'):
                modified_options.mapper.init_min_tri_angle *= 0.8
            if hasattr(modified_options.mapper, 'abs_pose_min_inliers_ratio'):
                modified_options.mapper.abs_pose_min_inliers_ratio *= 0.9
            if hasattr(modified_options.mapper, 'filter_max_reproj_error'):
                modified_options.mapper.filter_max_reproj_error *= 1.1
            
            # 第二次建图
            t_start2 = time()
            os.makedirs(f"{output_path}_attempt_{attempt}", exist_ok=True)
            
            try:
                maps2 = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=f"{output_path}_attempt_{attempt}",
                    options=modified_options
                )
                second_time = time() - t_start2
                total_time += second_time
                
                print(f"第 {attempt} 次建图完成，耗时: {second_time:.2f}秒")
                print(f"注册图像数: {sum(len(m.images) for m in maps2.values()) if maps2 else 0}")
                print(f"聚类数: {len(maps2) if maps2 else 0}")
                
                # 比较两次建图结果
                print("比较两次建图结果...")
                best_maps = compare_reconstruction_models(maps, maps2, None)
                
                # 确定最优结果对应的attempt
                if best_maps == maps:
                    print("第一次建图结果更优")
                    final_output = f"{output_path}_attempt_1"
                else:
                    print("第二次建图结果更优")
                    final_output = f"{output_path}_attempt_2"
                
                # 复制最优结果到最终输出目录
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.copytree(final_output, output_path)
                
            except Exception as e:
                print(f"第二次建图失败: {e}")
                print("使用第一次建图结果")
        else:
            if first_time >= time_threshold:
                print(f"建图时间 {first_time:.2f}秒 >= 阈值 {time_threshold}秒，不进行重新建图")
            else:
                print(f"已达到最大尝试次数 {max_attempts}，停止重新建图")
    
    except Exception as e:
        print(f"第一次建图失败: {e}")
        best_maps = {}
        
    return best_maps, total_time, attempt

# @dataclasses.dataclass
# class Prediction:
#     image_id: str | None
#     dataset: str
#     filename: str
#     cluster_index: int | None = None
#     rotation: np.ndarray | None = None
#     translation: np.ndarray | None = None
@dataclasses.dataclass
class Prediction:
    image_id: Optional[str]  # 或使用 Union[str, None]
    dataset: str
    filename: str 
    cluster_index: Optional[int] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None

# Main processing
is_train = True
is_OneTest = False 
data_dir = '../image-matching-challenge-2025'
workdir = './results'
os.makedirs(workdir, exist_ok=True)

if is_OneTest:
    sample_submission_csv = os.path.join(data_dir, 'train_labels_one.csv' if is_train else 'sample_submission.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv' if is_train else 'sample_submission.csv')

samples = {}
competition_data = pd.read_csv(sample_submission_csv)

for _, row in competition_data.iterrows():
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )


for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')

gc.collect()
max_images = None
datasets_to_process = None

timings = {
    "shortlisting": [],
    "feature_detection": [],
    "feature_matching": [],
    "RANSAC": [],
    "Reconstruction": []
}
mapping_result_strs = []

print(f"Extracting on device {device}")

if is_OneTest:
    dataset_train_test_lst = [
        'ETs_one',
        'stairs_one'
    ]
else:
    dataset_train_test_lst = [
        'ETs',
        'stairs'
        # 'imc2023_heritage'
    ]
    
for dataset, predictions in samples.items():
    if datasets_to_process and dataset not in datasets_to_process:
        print(f'Skipping "{dataset}"')
        continue
    if is_train is True:
        if dataset not in dataset_train_test_lst:
            continue
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    images = [os.path.join(images_dir, p.filename) for p in predictions]
    # images = [
    #     '../image-matching-challenge-2025/train/imc2023_heritage/dioscuri_img_0095.png',
    #     '../image-matching-challenge-2025/train/imc2023_heritage/dioscuri_archive_0003.png'
    # ]
    if max_images is not None:
        images = images[:max_images]

    print(f'\nProcessing dataset "{dataset}": {len(images)} images')
    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
    feature_dir = os.path.join(workdir, 'featureout', dataset)
    os.makedirs(feature_dir, exist_ok=True)

    if 1:
        # try:
        t = time()
        # index_pairs = get_image_pairs_shortlist(images, sim_th=0.3, min_pairs=20, 
        #                                         exhaustive_if_less=20, device=device)
        index_pairs = get_image_pairs_shortlist_clip(images, sim_th=0.76, min_pairs=1, 
                                            exhaustive_if_less=20, device=device)
        timings['shortlisting'].append(time() - t)
        print(f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
        gc.collect()
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete

        # t = time()
        # detect_aliked(images, feature_dir, 4096, device=device)
        # timings['feature_detection'].append(time() - t)
        # print(f'Features detected in {time() - t:.4f} sec')
        
        # t = time()
        # match_with_GIMdkm(dkm_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        # timings['feature_matching'].append(time() - t)
        # print(f'Features matched in {time() - t:.4f} sec')

        lightglue_matcher = Lightglue_Matcher(device=device,num_features=4096)
        
        # t = time()
        # detect_aliked_second(images, feature_dir, 4096, device=device)
        # # detect_person(lightglue_matcher, images, feature_dir, device=device)
        # print(f'person_mask in {time() - t:.4f} sec')

        # index_pairs.append((9,3))
        # index_pairs.append((9,4))
        t = time()
        # detect_aliked(images, feature_dir, 4096, device=device)
        detect_sp_ensemble(lightglue_matcher, images, feature_dir, 4096, device=device)
        timings['feature_detection'].append(time() - t)
        print(f'Features detected in {time() - t:.4f} sec')
        gc.collect()
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete


        # t = time()
        # # detect_aliked(images, feature_dir, 4096, device=device)
        # loftr_feature(lightglue_matcher, images, feature_dir, device=device)
        # timings['feature_detection'].append(time() - t)
        # print(f'Features detected in {time() - t:.4f} sec')
            
        # t = time()
        # match_with_lightglue(images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        # timings['feature_matching'].append(time() - t)
        # print(f'Features matched in {time() - t:.4f} sec')

        # # 3. 微调LightGlue
        # t = time()
        # finetuned_model = fine_tune_lightglue(
        #     lightglue_matcher,
        #     images, 
        #     feature_dir, 
        #     device,
        #     batch_size=4,
        #     epochs=1,
        #     up_limits=15
        # )
        # lightglue_matcher.update_model(finetuned_model)
        # lightglue_matcher.model = finetuned_model
        # print(f'模型微调完成，耗时 {time() - t:.4f} sec')
        

        # from safe_match import match_with_gimlightglue_ensemble_withfine_safe
        t = time()
        # match_matrix = match_with_gimloftr(lightglue_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        match_matrix = match_with_gimlightglue_ensemble_withfine(lightglue_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        # match_matrix = refine_matches(lightglue_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        timings['feature_matching'].append(time() - t)
        print(f'Features matched in {time() - t:.4f} sec')
        print('match_matrix', match_matrix.sum())
        gc.collect()

    if 1:
        from data_process.filter_match import filter_matches_graph, visualize_filtered_matches, visualize_connections
        features_data = {}
        with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc:
            # h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
            # h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
            # h5py.File(f'{feature_dir}/mask.h5', mode='r') as f_mask:
            for img_path in tqdm(images):
                key = img_path.split('/')[-1].split('\\')[-1]
                features_data[key] = {
                    'kp': torch.from_numpy(f_kp[key][...]).to(device),
                    'desc': torch.from_numpy(f_desc[key][...]).to(device),
                    # 'size': torch.from_numpy(f_size[key][...]).to(device),
                    # 'scale': torch.from_numpy(f_scale[key][...]).to(device),
                    # 'mask': torch.from_numpy(f_mask[key][...]).to(device)
                }

        with open(os.path.join(feature_dir, 'match_dict.pkl'), 'rb') as f:
            matches_dict = pickle.load(f)
        cycle_csv_path = os.path.join(feature_dir, 'matches.csv')

        # from train_LR.extract_features import extract_match_features
        # from train_LR.predict import filter_match_with_lr
        # from train_cnn.inference import filter_matches_with_cnn
        # output_csv_path = os.path.join(feature_dir, 'matches_features.csv')
        # 提取特征并保存到CSV
        # df = extract_match_features(matches_dict, features_data, output_csv_path)
        # # cycle_csv_path = None
        # lr_model_path = './results/combined_model/'
        # lr_model_path = './lr_model/LR_PB47'
        # lr_out_csv_path = os.path.join(feature_dir, 'lr_pred.csv')
        # filtered_matches_dict = filter_match_with_lr(matches_dict, features_data, model_dir=lr_model_path,threshold=0.3785,output_csv=lr_out_csv_path)
        # cnn_model_path = './models/0528_best_model.pth'
        # filtered_matches_dict = filter_matches_with_cnn(cnn_model_path, matches_dict, images, threshold=0.01, max_filter_ratio=0.3)
        filtered_matches_dict, cycle_error_data = filter_matches_graph(images, matches_dict, features_data, output_csv=cycle_csv_path, verbose=False)
        
        # # 示例调用
        # key = "stairs_split_1_1710453930259.png"  # 你想作为中心的图像关键字
        # visualize_connections(key, filtered_matches_dict, features_data, images, "connections_viz")

        # # 可视化过滤结果
        # visualize_filtered_matches(images, matches_dict, filtered_matches_dict, features_data, os.path.join(feature_dir, 'graph_results'))
        
        import shutil
        # 备份原始 matches.h5 文件（如果存在）
        matches_h5_path = os.path.join(feature_dir, 'matches.h5')
        if os.path.exists(matches_h5_path):
            backup_path = matches_h5_path + '.bak'
            shutil.copy2(matches_h5_path, backup_path)
            print(f"原始 matches.h5 已备份为 {backup_path}")

        # 将过滤后的匹配结果保存为 matches.h5
        with h5py.File(matches_h5_path, 'w') as f_match:
            for match_key, match_data in filtered_matches_dict.items():
                key1, key2 = match_key.split('-')
                match_indices = match_data  # 获取匹配索引
                
                # 创建key1的组并保存匹配结果
                group = f_match.require_group(key1)
                group.create_dataset(key2, data=match_indices)
                
        print(f"已将过滤后的匹配结果保存至 {matches_h5_path}")

    # exit()
    #删除无用文件
    if os.path.exists(f'{feature_dir}/feat_f.h5'):
        os.remove(f'{feature_dir}/feat_f.h5')
    if os.path.exists(f'{feature_dir}/feat_c.h5'):
        os.remove(f'{feature_dir}/feat_c.h5')

    database_path = os.path.join(feature_dir, 'colmap.db')
    if os.path.isfile(database_path):
        os.remove(database_path)
    gc.collect()

    # matches_file_path = "./results/featureout/ETs_one/matches.h5"
    # # 要删除的匹配对列表
    # pairs_to_remove = [
    #     ("another_et_another_et003.png", "another_et_another_et009.png"),
    #     ("another_et_another_et004.png", "another_et_another_et009.png"),
    # ]
    # # 执行删除操作
    # remove_matches_from_h5(matches_file_path, pairs_to_remove)

    sleep(1)
    import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
    output_path = f'{feature_dir}/colmap_rec_aliked'
    
    t = time()
    pycolmap.match_exhaustive(database_path)
    timings['RANSAC'].append(time() - t)
    print(f'Ran RANSAC in {time() - t:.4f} sec')
    
    # best_pair = find_best_initial_pair(match_matrix, features_data)
    # if best_pair:
    #     mapper_options.init_image_id1 = best_pair[0]
    #     mapper_options.init_image_id2 = best_pair[1]

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 5
    mapper_options.max_num_models = 30
    # mapper_options.mapper.abs_pose_min_num_inliers = 15
    mapper_options.mapper.num_threads = 1
    # if max_pair is not None:
    #     image_id1, image_id2 = pair_id_to_image_pair(max_pair)
    #     mapper_options.init_image_id1 = image_id1
    #     mapper_options.init_image_id2 = image_id2

    os.makedirs(output_path, exist_ok=True)
    t = time()
    # maps = pycolmap.incremental_mapping(database_path=database_path, 
    #                                     image_path=images_dir,
    #                                     output_path=output_path,
    #                                     options=mapper_options)
    
    maps, mapping_time, num_attempts = incremental_mapping_with_monitoring(
        database_path=database_path,
        image_path=images_dir,
        output_path=output_path,
        mapper_options=mapper_options,
        images_num=len(images),
        time_threshold=300.0,  # 30秒阈值，可根据需要调整
        max_attempts=2
    )
    if mapping_time == 0:
        maps = pycolmap.incremental_mapping(database_path=database_path, 
                                    image_path=images_dir,
                                    output_path=output_path,
                                    options=mapper_options)
    

    sleep(1)
    timings['Reconstruction'].append(time() - t)
    print(f'Reconstruction done in  {time() - t:.4f} sec')
    print(maps)

    registered = 0
    for map_index, cur_map in maps.items():
        for index, image in cur_map.images.items():
            prediction_index = filename_to_index[image.name]
            predictions[prediction_index].cluster_index = map_index
            predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
            predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
            registered += 1
    mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
    mapping_result_strs.append(mapping_result_str)
    print(mapping_result_str)
    gc.collect()
    # except Exception as e:
    #     print(e)
    #     mapping_result_str = f'Dataset "{dataset}" -> Failed!'
    #     mapping_result_strs.append(mapping_result_str)
    #     print(mapping_result_str)

print('\nResults')
for s in mapping_result_strs:
    print(s)

print('\nTimings')
for k, v in timings.items():
    print(f'{k} -> total={sum(v):.02f} sec.')

# Create submission file
array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = './submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')

# Compute results for training set
if is_train:
    t = time()
    if is_OneTest:
        final_score, dataset_scores = metric.score(
            gt_csv=os.path.join(data_dir, 'train_labels_one.csv'),
            user_csv=submission_file,
            thresholds_csv=os.path.join(data_dir, 'train_thresholds_one.csv'),
            mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
            inl_cf=0,
            strict_cf=-1,
            verbose=True,
        )
    else:
        final_score, dataset_scores = metric.score(
            gt_csv=os.path.join(data_dir, 'train_labels.csv'),
            user_csv=submission_file,
            thresholds_csv=os.path.join(data_dir, 'train_thresholds.csv'),
            mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
            inl_cf=0,
            strict_cf=-1,
            verbose=True,
        )
    print(f'Computed metric in: {time() - t:.02f} sec.')