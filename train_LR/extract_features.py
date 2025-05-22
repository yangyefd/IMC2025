import os
import numpy as np
import pandas as pd
import torch
import pickle
import re
from collections import defaultdict
from tqdm import tqdm


def build_match_graph(match_dict):
    """
    构建匹配图，统计每个图像的连接信息
    
    Args:
        match_dict: 匹配字典，包含匹配对和匹配分数 {key1-key2: [idxs, scores]}
    
    Returns:
        node_stats: 每个节点(图像)的统计信息
        {
            key: {
                'connections': 连接数量,
                'matches_count': [与每个连接的匹配点数量],
                'matches_scores': [与每个连接的匹配分数]
            }
        }
    """
    # 初始化图结构
    graph = defaultdict(list)
    node_stats = defaultdict(lambda: {'connections': 0, 'matches_count': [], 'matches_scores': []})
    
    # 构建图
    for match_key, match_data in match_dict.items():
        # 跳过包含outliers的匹配对
        if 'outliers' in match_key:
            continue
            
        key1, key2 = match_key.split('-')
        match_indices = match_data[0]
        match_scores = match_data[1]
        
        num_matches = len(match_indices)
        mean_score = np.mean(match_scores) if len(match_scores) > 0 else 0
        
        # 添加连接
        graph[key1].append((key2, num_matches, mean_score))
        graph[key2].append((key1, num_matches, mean_score))
    
    # 计算每个节点的统计信息
    for key, connections in graph.items():
        node_stats[key]['connections'] = len(connections)
        node_stats[key]['matches_count'] = [c[1] for c in connections]
        node_stats[key]['matches_scores'] = [c[2] for c in connections]
    
    return node_stats


def extract_match_features(match_dict, features_data, output_csv_path):
    """
    提取匹配对的特征，用于训练LR分类器
    
    Args:
        match_dict: 匹配字典，包含匹配对和匹配分数 {key1-key2: [idxs, scores]}
        features_data: 特征数据，包含特征点和描述子 {key: {'kp': kp, 'desc': desc, ...}}
        output_csv_path: 输出CSV文件路径
        
    Returns:
        pandas DataFrame: 包含提取的特征
    """
    features_list = []
    
    # 首先构建匹配图，计算每个图像的统计信息
    print("构建匹配图并计算节点统计信息...")
    node_stats = build_match_graph(match_dict)
    
    print("提取匹配对特征...")
    for match_key, match_data in tqdm(match_dict.items()):
        # 跳过包含outliers的匹配对
        if 'outliers' in match_key:
            continue
            
        key1, key2 = match_key.split('-')
        
        # 提取场景名称作为标签
        scene_name1 = extract_scene_name(key1)
        scene_name2 = extract_scene_name(key2)
        
        # 如果场景名称相同，则标签为1，否则为0
        label = 1 if scene_name1 == scene_name2 else 0
        
        # 获取匹配索引和分数
        match_indices = match_data[0]
        match_scores = match_data[1]
        
        # 如果匹配数量很少，可能不是有效的匹配对
        if len(match_indices) < 5:
            continue
            
        # 计算匹配数量
        num_matches = len(match_indices)
        
        # 提取特征点和描述子
        kp1 = features_data[key1]['kp']
        kp2 = features_data[key2]['kp']
        desc1 = features_data[key2]['desc']
        desc2 = features_data[key2]['desc']
        
        # 确保转为numpy进行计算
        if isinstance(kp1, torch.Tensor):
            kp1 = kp1.detach().cpu().numpy()
        if isinstance(kp2, torch.Tensor):
            kp2 = kp2.detach().cpu().numpy()
        if isinstance(desc1, torch.Tensor):
            desc1 = desc1.detach().cpu().numpy()
        if isinstance(desc2, torch.Tensor):
            desc2 = desc2.detach().cpu().numpy()
            
        # 1. 匹配分数统计特征
        score_mean = np.mean(match_scores) if len(match_scores) > 0 else 0
        score_std = np.std(match_scores) if len(match_scores) > 0 else 0
        score_min = np.min(match_scores) if len(match_scores) > 0 else 0
        score_max = np.max(match_scores) if len(match_scores) > 0 else 0
        score_median = np.median(match_scores) if len(match_scores) > 0 else 0
        score_q25 = np.percentile(match_scores, 25) if len(match_scores) > 0 else 0
        score_q75 = np.percentile(match_scores, 75) if len(match_scores) > 0 else 0
        
        # 2. 匹配点坐标统计特征
        matched_kp1 = kp1[match_indices[:, 0]]
        matched_kp2 = kp2[match_indices[:, 1]]
        
        # 计算匹配点的空间分布
        kp1_x_mean = np.mean(matched_kp1[:, 0]) if len(matched_kp1) > 0 else 0
        kp1_y_mean = np.mean(matched_kp1[:, 1]) if len(matched_kp1) > 0 else 0
        kp2_x_mean = np.mean(matched_kp2[:, 0]) if len(matched_kp2) > 0 else 0
        kp2_y_mean = np.mean(matched_kp2[:, 1]) if len(matched_kp2) > 0 else 0
        
        kp1_x_std = np.std(matched_kp1[:, 0]) if len(matched_kp1) > 0 else 0
        kp1_y_std = np.std(matched_kp1[:, 1]) if len(matched_kp1) > 0 else 0
        kp2_x_std = np.std(matched_kp2[:, 0]) if len(matched_kp2) > 0 else 0
        kp2_y_std = np.std(matched_kp2[:, 1]) if len(matched_kp2) > 0 else 0
        
        # 计算两张图像中匹配点的位置差异
        kp_x_mean_diff = abs(kp1_x_mean - kp2_x_mean)
        kp_y_mean_diff = abs(kp1_y_mean - kp2_y_mean)
        kp_x_std_diff = abs(kp1_x_std - kp2_x_std)
        kp_y_std_diff = abs(kp1_y_std - kp2_y_std)
        
        # 3. 匹配点的空间分布 - 分析覆盖区域和密度
        if len(matched_kp1) > 0:
            kp1_x_range = np.max(matched_kp1[:, 0]) - np.min(matched_kp1[:, 0])
            kp1_y_range = np.max(matched_kp1[:, 1]) - np.min(matched_kp1[:, 1])
            kp1_area = kp1_x_range * kp1_y_range if kp1_x_range > 0 and kp1_y_range > 0 else 0
            kp1_density = num_matches / kp1_area if kp1_area > 0 else 0
        else:
            kp1_x_range, kp1_y_range, kp1_area, kp1_density = 0, 0, 0, 0
            
        if len(matched_kp2) > 0:
            kp2_x_range = np.max(matched_kp2[:, 0]) - np.min(matched_kp2[:, 0])
            kp2_y_range = np.max(matched_kp2[:, 1]) - np.min(matched_kp2[:, 1])
            kp2_area = kp2_x_range * kp2_y_range if kp2_x_range > 0 and kp2_y_range > 0 else 0
            kp2_density = num_matches / kp2_area if kp2_area > 0 else 0
        else:
            kp2_x_range, kp2_y_range, kp2_area, kp2_density = 0, 0, 0, 0
        
        area_ratio = kp1_area / kp2_area if kp2_area > 0 else 0
        
        # 4. 计算匹配点平均距离变化
        if len(matched_kp1) >= 2 and len(matched_kp2) >= 2:
            # 计算每张图像中所有点对之间的距离
            distances_1 = []
            distances_2 = []
            num_pairs_to_sample = min(1000, len(matched_kp1) * (len(matched_kp1) - 1) // 2)
            
            if num_pairs_to_sample > 500:
                # 如果点对太多，随机采样一部分
                from itertools import combinations
                import random
                pairs = list(combinations(range(len(matched_kp1)), 2))
                sampled_pairs = random.sample(pairs, num_pairs_to_sample)
                
                for i, j in sampled_pairs:
                    dist1 = np.sqrt(np.sum((matched_kp1[i] - matched_kp1[j]) ** 2))
                    dist2 = np.sqrt(np.sum((matched_kp2[i] - matched_kp2[j]) ** 2))
                    distances_1.append(dist1)
                    distances_2.append(dist2)
            else:
                for i in range(len(matched_kp1)):
                    for j in range(i + 1, len(matched_kp1)):
                        dist1 = np.sqrt(np.sum((matched_kp1[i] - matched_kp1[j]) ** 2))
                        dist2 = np.sqrt(np.sum((matched_kp2[i] - matched_kp2[j]) ** 2))
                        distances_1.append(dist1)
                        distances_2.append(dist2)
            
            distances_1 = np.array(distances_1)
            distances_2 = np.array(distances_2)
            
            # 计算距离比率和差异
            distance_ratios = distances_2 / distances_1
            distance_ratios = distance_ratios[~np.isnan(distance_ratios) & ~np.isinf(distance_ratios)]
            
            if len(distance_ratios) > 0:
                distance_ratio_mean = np.mean(distance_ratios)
                distance_ratio_std = np.std(distance_ratios)
                distance_ratio_consistency = 1.0 - min(distance_ratio_std / (distance_ratio_mean + 1e-8), 1.0)
            else:
                distance_ratio_mean = 0
                distance_ratio_std = 0
                distance_ratio_consistency = 0
        else:
            distance_ratio_mean = 0
            distance_ratio_std = 0
            distance_ratio_consistency = 0
            
        # 5. 描述子特征
        # 提取匹配点的描述子，计算其相似度和一致性
        if hasattr(match_indices, 'dtype') and len(match_indices) > 0:
            # 描述子相似度计算中添加安全检查
            try:
                matched_desc1 = desc1[match_indices[:, 0]]
                matched_desc2 = desc2[match_indices[:, 1]]
                # 确保没有零向量导致归一化问题
                norm1 = np.linalg.norm(matched_desc1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(matched_desc2, axis=1, keepdims=True)
                
                # 避免除以零
                valid_indices = (norm1 > 1e-8).flatten() & (norm2 > 1e-8).flatten()
                
                if np.any(valid_indices):
                    matched_desc1_norm = matched_desc1[valid_indices] / norm1[valid_indices]
                    matched_desc2_norm = matched_desc2[valid_indices] / norm2[valid_indices]
                    cosine_sims = np.sum(matched_desc1_norm * matched_desc2_norm, axis=1)
                    
                    desc_sim_mean = np.mean(cosine_sims)
                    desc_sim_std = np.std(cosine_sims)
                else:
                    desc_sim_mean = 0
                    desc_sim_std = 0
            except (IndexError, ValueError, TypeError, ZeroDivisionError) as e:
                desc_sim_mean = 0
                desc_sim_std = 0
        else:
            desc_sim_mean = 0
            desc_sim_std = 0
        
        # 7. 匹配分布统计
        # 分析匹配对的分布特性
        if len(match_scores) > 0:
            score_skew = calculate_skewness(match_scores)
            score_kurtosis = calculate_kurtosis(match_scores)
        else:
            score_skew = 0
            score_kurtosis = 0
            
        # 9. 匹配强度比例
        # 检查最强匹配和最弱匹配之间的比例
        if len(match_scores) >= 2:
            top_10_pct = int(max(1, len(match_scores) * 0.1))
            bottom_10_pct = int(max(1, len(match_scores) * 0.1))
            
            sorted_scores = np.sort(match_scores)
            top_scores_mean = np.mean(sorted_scores[-top_10_pct:])
            bottom_scores_mean = np.mean(sorted_scores[:bottom_10_pct])
            
            score_top_bottom_ratio = top_scores_mean / bottom_scores_mean if bottom_scores_mean > 0 else 0
        else:
            score_top_bottom_ratio = 0
        
        # 10. 新增：图像连接信息统计
        # 获取key1和key2的连接信息
        key1_stats = node_stats[key1]
        key2_stats = node_stats[key2]
        
        # key1的连接统计
        key1_connections = key1_stats['connections']
        key1_matches_mean = np.mean(key1_stats['matches_count']) if key1_stats['matches_count'] else 0
        key1_matches_std = np.std(key1_stats['matches_count']) if key1_stats['matches_count'] else 0
        key1_scores_mean = np.mean(key1_stats['matches_scores']) if key1_stats['matches_scores'] else 0
        key1_scores_std = np.std(key1_stats['matches_scores']) if key1_stats['matches_scores'] else 0
        
        # key2的连接统计
        key2_connections = key2_stats['connections']
        key2_matches_mean = np.mean(key2_stats['matches_count']) if key2_stats['matches_count'] else 0
        key2_matches_std = np.std(key2_stats['matches_count']) if key2_stats['matches_count'] else 0
        key2_scores_mean = np.mean(key2_stats['matches_scores']) if key2_stats['matches_scores'] else 0
        key2_scores_std = np.std(key2_stats['matches_scores']) if key2_stats['matches_scores'] else 0
        
        # 计算当前匹配与平均匹配的比率
        key1_matches_ratio = num_matches / key1_matches_mean if key1_matches_mean > 0 else 0
        key2_matches_ratio = num_matches / key2_matches_mean if key2_matches_mean > 0 else 0
        key1_scores_ratio = score_mean / key1_scores_mean if key1_scores_mean > 0 else 0
        key2_scores_ratio = score_mean / key2_scores_mean if key2_scores_mean > 0 else 0
        
        # 收集特征
        feature_dict = {
            # 基本信息
            'key1': key1,
            'key2': key2,
            'label': label,
            'scene1': scene_name1,
            'scene2': scene_name2,
            
            # 1. 匹配数量和分数
            'num_matches': num_matches,
            'score_mean': score_mean,
            'score_std': score_std,
            'score_min': score_min,
            'score_max': score_max,
            'score_median': score_median,
            'score_q25': score_q25,
            'score_q75': score_q75,
            'score_top_bottom_ratio': score_top_bottom_ratio,
            
            # 2. 匹配点空间统计
            'kp1_x_mean': kp1_x_mean,
            'kp1_y_mean': kp1_y_mean,
            'kp2_x_mean': kp2_x_mean,
            'kp2_y_mean': kp2_y_mean,
            'kp1_x_std': kp1_x_std,
            'kp1_y_std': kp1_y_std,
            'kp2_x_std': kp2_x_std,
            'kp2_y_std': kp2_y_std,
            'kp_x_mean_diff': kp_x_mean_diff,
            'kp_y_mean_diff': kp_y_mean_diff,
            'kp_x_std_diff': kp_x_std_diff,
            'kp_y_std_diff': kp_y_std_diff,
            
            # 3. 匹配点空间分布
            'kp1_area': kp1_area,
            'kp2_area': kp2_area,
            'kp1_density': kp1_density,
            'kp2_density': kp2_density,
            'area_ratio': area_ratio,
            
            # 4. 距离变化统计
            'distance_ratio_mean': distance_ratio_mean,
            'distance_ratio_std': distance_ratio_std,
            'distance_ratio_consistency': distance_ratio_consistency,
            
            # 5. 描述子统计
            'desc_sim_mean': desc_sim_mean,
            'desc_sim_std': desc_sim_std,
            
            # 7. 分数分布统计
            'score_skew': score_skew,
            'score_kurtosis': score_kurtosis,
            
            # 10. 新增：图像连接信息统计
            'key1_connections': key1_connections,
            'key1_matches_mean': key1_matches_mean,
            'key1_matches_std': key1_matches_std, 
            'key1_scores_mean': key1_scores_mean,
            'key1_scores_std': key1_scores_std,
            'key2_connections': key2_connections,
            'key2_matches_mean': key2_matches_mean,
            'key2_matches_std': key2_matches_std,
            'key2_scores_mean': key2_scores_mean,
            'key2_scores_std': key2_scores_std,
            'key1_matches_ratio': key1_matches_ratio,
            'key2_matches_ratio': key2_matches_ratio,
            'key1_scores_ratio': key1_scores_ratio,
            'key2_scores_ratio': key2_scores_ratio,
        }
        
        features_list.append(feature_dict)
    
    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    
    print(f"已提取 {len(df)} 个匹配对的特征，保存至 {output_csv_path}")
    # print(f"正样本数量: {df['label'].sum()}, 负样本数量: {len(df) - df['label'].sum()}")
    
    return df


def extract_scene_name(key):
    """从图像名称中提取场景名称"""

    # 如果没有匹配到任何模式，返回整个文件名（不包括扩展名）
    return "_".join(key.split('_')[:-1])


def calculate_skewness(data):
    """计算偏度 (skewness)"""
    if len(data) < 3:
        return 0
        
    n = len(data)
    mean = np.mean(data)
    m3 = np.sum((data - mean) ** 3) / n
    m2 = np.sum((data - mean) ** 2) / n
    return m3 / (m2 ** 1.5) if m2 > 0 else 0


def calculate_kurtosis(data):
    """计算峰度 (kurtosis)"""
    if len(data) < 4:
        return 0
        
    n = len(data)
    mean = np.mean(data)
    m4 = np.sum((data - mean) ** 4) / n
    m2 = np.sum((data - mean) ** 2) / n
    return m4 / (m2 ** 2) - 3 if m2 > 0 else 0


def main():
    """主函数，用于测试特征提取"""
    # 设置文件路径
    feature_dir = './results/featureout/ETs'  # 根据实际路径调整
    match_dict_path = os.path.join(feature_dir, 'match_dict.pkl')
    output_csv_path = os.path.join(feature_dir, 'match_features.csv')
    
    # 加载match_dict
    print(f"加载匹配字典 {match_dict_path}...")
    with open(match_dict_path, 'rb') as f:
        match_dict = pickle.load(f)
    
    # 加载features_data
    print("加载特征数据...")
    features_data = {}
    import h5py
    with h5py.File(os.path.join(feature_dir, 'keypoints_coarse.h5'), mode='r') as f_kp, \
         h5py.File(os.path.join(feature_dir, 'descriptors.h5'), mode='r') as f_desc, \
         h5py.File(os.path.join(feature_dir, 'size.h5'), mode='r') as f_size, \
         h5py.File(os.path.join(feature_dir, 'scale.h5'), mode='r') as f_scale, \
         h5py.File(os.path.join(feature_dir, 'mask.h5'), mode='r') as f_mask:
        for key in tqdm(f_kp.keys()):
            features_data[key] = {
                'kp': torch.from_numpy(f_kp[key][...]),
                'desc': torch.from_numpy(f_desc[key][...]),
                'size': torch.from_numpy(f_size[key][...]),
                'scale': torch.from_numpy(f_scale[key][...]),
                'mask': torch.from_numpy(f_mask[key][...])
            }
    
    # 提取特征并保存到CSV
    df = extract_match_features(match_dict, features_data, output_csv_path)
    
    # 显示一些统计信息
    print("\n特征统计:")
    print(df.describe())
    
    # 检查类别不平衡问题
    label_counts = df['label'].value_counts()
    print("\n标签分布:")
    print(label_counts)
    print(f"正负样本比例: {label_counts[1] / label_counts[0] if 0 in label_counts else 'NA'}")


if __name__ == "__main__":
    main()