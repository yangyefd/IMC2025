
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
from GIMlightglue_match import Lightglue_Matcher
from fine_tune_lightglue import fine_tune_lightglue
# from filter_match import adaptive_match_filtering
from CLIP.clip import clip
# Device setup
device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

# def get_global_desc(fnames, device=torch.device('cpu')):
#     processor = AutoImageProcessor.from_pretrained('./models/dinov2-pytorch-base-v1')
#     model = AutoModel.from_pretrained('./models/dinov2-pytorch-base-v1')
#     model = model.eval().to(device)
#     global_descs_dinov2 = []
#     for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
#         key = os.path.splitext(os.path.basename(img_fname_full))[0]
#         timg = load_torch_image(img_fname_full)
#         with torch.inference_mode():
#             inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
#             outputs = model(**inputs)
#             dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
#         global_descs_dinov2.append(dino_mac.detach().cpu())
#     return torch.cat(global_descs_dinov2, dim=0)

def get_global_desc(fnames, device=torch.device('cpu')):
    model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
    print("分簇模型加载成功")

    model = model.eval().to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        # 加载并预处理图像
        timg = preprocess(Image.open(img_fname_full)).unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            features = model.encode_image(timg)
        # 归一化特征
        features = features / features.norm(dim=-1, keepdim=True)

        global_descs_dinov2.append(features.detach().cpu())
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
    descs = get_global_desc(fnames, device=device)        
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
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, 
                     resize=resize_to).eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
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

def detect_sp(lightglue_matcher, img_fnames, feature_dir='.featureout', num_features=4096, 
                 resize_to=1024, device=torch.device('cpu')):
    dtype = torch.float32
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='w') as f_size,\
         h5py.File(f'{feature_dir}/scale.h5', mode='w') as f_scale:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            img_fname = img_fname.split('\\')[-1]
            key = img_fname
            with torch.inference_mode():
                feats0, data = lightglue_matcher.extract(img_path)
                kpts = feats0['keypoints0'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors0'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
                f_size[key] = data['size0'].cpu()
                f_scale[key] = data['scale0'].cpu()
    return

def detect_sp_ensemble(lightglue_matcher, img_fnames, feature_dir='.featureout', num_features=4096, 
                 resize_to=1024, device=torch.device('cpu')):
    #集成方法 ALIke sp各提一半点 2048个
    dtype = torch.float32

    extractor_alike = ALIKED(max_num_keypoints=num_features//2, detection_threshold=0.01, 
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
                kpts = np.zeros((num_features,2))
                descs = np.zeros((num_features,256))
                feats0, data = lightglue_matcher.extract(img_path)
                feats0_kpts = feats0['keypoints0'].reshape(-1, 2).detach().cpu().numpy()
                kpts[:len(feats0_kpts)] = feats0['keypoints0'].reshape(-1, 2).detach().cpu().numpy()
                descs[:len(feats0_kpts)] = feats0['descriptors0'].reshape(len(feats0_kpts), -1).detach().cpu().numpy()

                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0_alike = extractor_alike.extract(image0)
                feats0_alike_pkts = feats0_alike['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                kpts[num_features//2:num_features//2+len(feats0_alike_pkts)] = feats0_alike_pkts
                descs[num_features//2:num_features//2+len(feats0_alike_pkts),:128] = feats0_alike['descriptors'].reshape(len(feats0_alike_pkts), -1).detach().cpu().numpy()

                f_kp[key] = kpts
                f_desc[key] = descs
                f_size[key] = data['size0'].cpu()
                f_scale[key] = data['scale0'].cpu()
                f_mask[key] = np.array([len(feats0_kpts), len(feats0_alike_pkts)])

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

def match_with_gimlightglue_ensemble(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                            "depth_confidence": -1,
                                            "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
        h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale, \
        h5py.File(f'{feature_dir}/mask.h5', mode='r') as f_mask, \
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
            fp_maks1 = np.array(f_mask[key1])
            fp_maks2 = np.array(f_mask[key2])
            num_pts = len(kp1)
            pred = {}
            pred['keypoints0'] = kp1[:num_pts//2][:fp_maks1[0]][None]
            pred['keypoints1'] = kp2[:num_pts//2][:fp_maks2[0]][None]
            pred['descriptors0'] = desc1[:num_pts//2][:fp_maks1[0]][None]
            pred['descriptors1'] = desc2[:num_pts//2][:fp_maks2[0]][None]
            pred['size0'] = torch.from_numpy(f_size[key1][...]).to(device)
            pred['size1'] = torch.from_numpy(f_size[key2][...]).to(device)
            pred['scale0'] = torch.from_numpy(f_scale[key1][...]).to(device)
            pred['scale1'] = torch.from_numpy(f_scale[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lightglue_matcher.match(pred)
                _, idxs_alike = lg_matcher(desc1[num_pts//2:,:128][:fp_maks1[1]].float(), desc2[num_pts//2:,:128][:fp_maks2[1]].float(),
                        KF.laf_from_center_scale_ori(kp1[num_pts//2:][:fp_maks1[1]][None].float()),
                        KF.laf_from_center_scale_ori(kp2[num_pts//2:][:fp_maks2[1]][None].float()))
                idxs_alike += num_pts//2
                idxs = torch.cat([idxs, idxs_alike], dim=0)
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

def match_with_gimlightglue_batch(lightglue_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), batch_size=8, min_matches=15, verbose=True, visualize=True):
    """使用批处理方式进行特征匹配，大幅提高速度"""
    match_matrix = np.zeros((len(img_fnames), len(img_fnames)), dtype=np.int32)
    
    # 批量加载特征数据
    print("加载特征数据...")
    features_data = {}
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size, \
         h5py.File(f'{feature_dir}/scale.h5', mode='r') as f_scale:
         
        for img_path in tqdm(img_fnames):
            key = img_path.split('/')[-1].split('\\')[-1]
            features_data[key] = {
                'kp': torch.from_numpy(f_kp[key][...]).to(device),
                'desc': torch.from_numpy(f_desc[key][...]).to(device),
                'size': torch.from_numpy(f_size[key][...]).to(device),
                'scale': torch.from_numpy(f_scale[key][...]).to(device),
            }
    
    # 批量处理
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        # 将图像对分成批次
        num_batches = (len(index_pairs) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(index_pairs))
            batch_pairs = index_pairs[start_idx:end_idx]
            
            batch_data = []
            batch_info = []
            
            # 准备批次数据
            for pair_idx in batch_pairs:
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1 = fname1.split('/')[-1].split('\\')[-1]
                key2 = fname2.split('/')[-1].split('\\')[-1]
                
                # 获取图像特征
                kp1 = features_data[key1]['kp']
                kp2 = features_data[key2]['kp']
                desc1 = features_data[key1]['desc']
                desc2 = features_data[key2]['desc']
                
                pred = {
                    'keypoints0': kp1[None],
                    'keypoints1': kp2[None],
                    'descriptors0': desc1[None],
                    'descriptors1': desc2[None],
                    'size0': features_data[key1]['size'],
                    'size1': features_data[key2]['size'],
                    'scale0': features_data[key1]['scale'],
                    'scale1': features_data[key2]['scale'],
                }
                
                batch_data.append(pred)
                batch_info.append((idx1, idx2, key1, key2, fname1, fname2))
            
            # 批量匹配
            print(f"处理批次 {batch_idx+1}/{num_batches} ({len(batch_pairs)} 对图像)...")
            
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
            
            # 批量推理
            with torch.inference_mode():
                batch_dists, batch_idxs = lightglue_matcher.match_batch(batch_preds)
            
            # 处理结果
            for i, (idx1, idx2, key1, key2, fname1, fname2) in enumerate(batch_info):
                if i >= len(batch_idxs) or batch_idxs[i] is None or len(batch_idxs[i]) == 0:
                    continue
                
                idxs = batch_idxs[i]
                n_matches = len(idxs)
                
                if verbose:
                    print(f'{key1}-{key2}: {n_matches} matches')
                
                # 保存匹配结果
                if n_matches >= min_matches:
                    group = f_match.require_group(key1)
                    group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                    match_matrix[idx1, idx2] = n_matches
                    
                    # # 可视化匹配
                    # if visualize:
                    #     vis_dir = os.path.join(feature_dir, 'visualizations')
                    #     os.makedirs(vis_dir, exist_ok=True)
                    #     save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    #     visualize_matches(
                    #         fname1, fname2,
                    #         features_data[key1]['kp'].cpu().numpy(),
                    #         features_data[key2]['kp'].cpu().numpy(),
                    #         idxs.cpu().numpy(),
                    #         save_path
                    #     )
    
    return match_matrix

def match_with_lightglue(img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):
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
                if visualize:
                    vis_dir = os.path.join(feature_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    visualize_matches(fname1, fname2, 
                                   kp1.cpu().numpy(), 
                                   kp2.cpu().numpy(),
                                   idxs.cpu().numpy(),
                                   save_path)
    return

def match_with_GIMdkm(dkm_matcher, img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):

    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)

            with torch.inference_mode():
                dists, idxs = dkm_matcher.match(kp1, kp2, fname1, fname2)
            if len(idxs) == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')

            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                                # 添加可视化
                if visualize:
                    vis_dir = os.path.join(feature_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    visualize_matches(fname1, fname2, 
                                   kp1.cpu().numpy(), 
                                   kp2.cpu().numpy(),
                                   idxs.cpu().numpy(),
                                   save_path)
            # break
    return

def import_into_colmap(img_dir, feature_dir='.featureout', database_path='colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
    return

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

    # try:
    t = time()
    # index_pairs = get_image_pairs_shortlist(images, sim_th=0.3, min_pairs=20, 
    #                                         exhaustive_if_less=20, device=device)
    index_pairs = get_image_pairs_shortlist_clip(images, sim_th=0.75, min_pairs=1, 
                                        exhaustive_if_less=20, device=device)
    timings['shortlisting'].append(time() - t)
    print(f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
    gc.collect()

    # t = time()
    # detect_aliked(images, feature_dir, 4096, device=device)
    # timings['feature_detection'].append(time() - t)
    # print(f'Features detected in {time() - t:.4f} sec')
    
    # t = time()
    # match_with_GIMdkm(dkm_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
    # timings['feature_matching'].append(time() - t)
    # print(f'Features matched in {time() - t:.4f} sec')

    lightglue_matcher = Lightglue_Matcher(device=device,num_features=2048)
    t = time()
    # detect_aliked(images, feature_dir, 4096, device=device)
    detect_sp_ensemble(lightglue_matcher, images, feature_dir, 4096, device=device)
    timings['feature_detection'].append(time() - t)
    print(f'Features detected in {time() - t:.4f} sec')
        
    # t = time()
    # match_with_lightglue(images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
    # timings['feature_matching'].append(time() - t)
    # print(f'Features matched in {time() - t:.4f} sec')

    # # 3. 微调LightGlue
    # t = time()
    # fine_tuned_matcher = fine_tune_lightglue(
    #     lightglue_matcher,
    #     images, 
    #     feature_dir, 
    #     device,
    #     batch_size=8,
    #     epochs=5
    # )
    # lightglue_matcher.update_model(fine_tuned_matcher)
    # print(f'模型微调完成，耗时 {time() - t:.4f} sec')
    

    t = time()
    match_matrix = match_with_gimlightglue_ensemble(lightglue_matcher, images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
    timings['feature_matching'].append(time() - t)
    print(f'Features matched in {time() - t:.4f} sec')
    print('match_matrix', match_matrix.sum())

    database_path = os.path.join(feature_dir, 'colmap.db')
    if os.path.isfile(database_path):
        os.remove(database_path)
    gc.collect()

    sleep(1)
    import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
    output_path = f'{feature_dir}/colmap_rec_aliked'
    
    t = time()
    pycolmap.match_exhaustive(database_path)
    timings['RANSAC'].append(time() - t)
    print(f'Ran RANSAC in {time() - t:.4f} sec')
    
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 25
    os.makedirs(output_path, exist_ok=True)
    t = time()
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