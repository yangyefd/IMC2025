'''
frames: list of frames path, can use to get the image
matches_dict: list of matches {key1-key2: [idxs, idx1, idx2, matchscores]}
features_data: list of features data, kp, descriptor...
'''
def visualize_filtered_matches(frames, original_matches_dict, filtered_matches_dict, features_data, output_dir=None):
    """
    可视化匹配过滤效果，将保留的匹配对用绿色表示，过滤掉的匹配对用红色表示。
    
    Args:
        frames: 图像路径列表
        original_matches_dict: 原始匹配字典 {key1-key2: matches}
        filtered_matches_dict: 过滤后的匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        output_dir: 输出目录，如果为None则直接显示
        
    Returns:
        None
    """
    import cv2
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每对匹配
    for key_pair in original_matches_dict.keys():
        if key_pair not in original_matches_dict:
            continue
            
        key1, key2 = key_pair.split('-')
        
        # 找到对应的图像路径
        img1_path = None
        img2_path = None
        for frame in frames:
            if key1 in frame:
                img1_path = frame
            if key2 in frame:
                img2_path = frame
        
        if img1_path is None or img2_path is None:
            print(f"无法找到 {key_pair} 对应的图像文件")
            continue
        
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"无法读取图像: {img1_path} 或 {img2_path}")
            continue
            
        # 获取原始匹配
        original_matches = original_matches_dict[key_pair][0]
        
        # 获取过滤后的匹配
        filtered_matches = []
        if key_pair in filtered_matches_dict:
            filtered_matches = filtered_matches_dict[key_pair]
        
        # 将过滤后的匹配转换为集合，便于判断是否保留
        filtered_matches_set = set()
        for m in filtered_matches:
            filtered_matches_set.add((int(m[0]), int(m[1])))
        
        # 获取特征点
        kp1 = features_data[key1]['kp']
        kp2 = features_data[key2]['kp']
        
        # 准备绘图
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建拼接图像
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        
        # 计算保留和过滤的匹配对数量
        kept_count = 0
        removed_count = 0
        
        # 绘制匹配线
        for m in original_matches:
            idx1, idx2 = int(m[0]), int(m[1])
            
            # 判断是否保留
            is_kept = (idx1, idx2) in filtered_matches_set
            
            # 更新计数
            if is_kept:
                kept_count += 1
                color = (0, 255, 0)  # 绿色表示保留的匹配
            else:
                removed_count += 1
                color = (0, 0, 255)  # 红色表示过滤掉的匹配
            
            # 获取坐标点
            pt1 = (int(kp1[idx1][0]), int(kp1[idx1][1]))
            pt2 = (int(kp2[idx2][0]) + w1, int(kp2[idx2][1]))
            
            # 绘制线和点
            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
        
        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"保留的匹配: {kept_count}, 过滤的匹配: {removed_count}"
        cv2.putText(vis, text, (10, 30), font, 1, (255, 255, 255), 2)
        
        # 输出结果
        output_path = os.path.join(output_dir, f"{key_pair}_filtered.jpg")
        cv2.imwrite(output_path, vis)
        print(f"保存可视化结果至: {output_path}")

def filter_matches_graph(frames, matches_dict, features_data, threshold=0.6, verbose=True):
    """
    基于图结构的匹配过滤函数，去除不一致的匹配。
    
    Args:
        frames: 图像路径列表
        matches_dict: 匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        threshold: 过滤阈值
        verbose: 是否输出详细统计信息
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
    """
    import numpy as np
    import cv2
    from collections import defaultdict
    import networkx as nx
    from sklearn.cluster import DBSCAN

    # 记录每个阶段的匹配数量
    stats = defaultdict(dict)
    
    # 新字典去除matches_dict中的分数
    filtered_matches_dict = {}
    for key_pair, matches in matches_dict.items():
        if len(matches) > 0:
            filtered_matches_dict[key_pair] = matches[0]
            # 记录初始匹配数
            stats["初始"][key_pair] = len(matches[0])
    
    # 构建图像之间的连接图
    G = nx.Graph()
    match_stats = defaultdict(int)
    
    # 1. 创建图结构并统计初始匹配情况
    for key_pair, matches in filtered_matches_dict.items():
        key1, key2 = key_pair.split('-')
        if len(matches) > 0:
            G.add_edge(key1, key2, weight=len(matches))
            match_stats[(key1, key2)] = len(matches)
    
    # 2. 使用循环一致性直接过滤不一致的匹配
    # 为每个匹配对创建一个掩码，初始所有点都保留
    match_masks = {}
    for key_pair in filtered_matches_dict:
        match_masks[key_pair] = np.ones(len(filtered_matches_dict[key_pair]), dtype=bool)
    
    cycle_filtered_count = 0
    cycle_total_checked = 0
    
    # 查找所有长度为3的循环
    for cycle in nx.cycle_basis(G):
        if len(cycle) == 3:
            # 获取三对匹配关系
            keys = cycle
            pairs = [
                (keys[0], keys[1]),
                (keys[1], keys[2]),
                (keys[2], keys[0])
            ]
            
            # 检查所有匹配对是否存在
            valid_cycle = True
            for key1, key2 in pairs:
                pair_key = f"{key1}-{key2}"
                reverse_key = f"{key2}-{key1}"
                if pair_key not in filtered_matches_dict and reverse_key not in filtered_matches_dict:
                    valid_cycle = False
                    break
            
            if not valid_cycle:
                continue
            
            # 构建匹配索引查找表
            match_maps = []
            pair_keys = []
            
            for key1, key2 in pairs:
                pair_key = f"{key1}-{key2}"
                reverse_key = f"{key2}-{key1}"
                
                if pair_key in filtered_matches_dict:
                    matches = filtered_matches_dict[pair_key]
                    is_reverse = False
                    pair_keys.append(pair_key)
                else:
                    matches = filtered_matches_dict[reverse_key]
                    is_reverse = True
                    pair_keys.append(reverse_key)
                
                # 创建从idx1到idx2的映射以及索引映射（用于掩码）
                if is_reverse:
                    # 调整索引顺序
                    match_map = {int(m[1]): int(m[0]) for m in matches}
                else:
                    match_map = {int(m[0]): int(m[1]) for m in matches}
                
                match_maps.append(match_map)
            
            # 为每个匹配对准备检查记录
            checked_indices = [set(), set(), set()]
            consistent_indices = [set(), set(), set()]
            
            # 检查第一对中的每个匹配点
            for idx, (idx1, idx2) in enumerate(filtered_matches_dict[pair_keys[0]]):
                idx1, idx2 = int(idx1), int(idx2)
                
                # 尝试形成一个循环
                if idx2 in match_maps[1]:
                    idx3 = match_maps[1][idx2]
                    if idx3 in match_maps[2]:
                        # 这个点确实在三张图像中都有对应，可以进行闭环检查
                        idx1_cycle = match_maps[2][idx3]
                        
                        # 将这个匹配标记为已检查
                        checked_indices[0].add(idx)
                        
                        # 找到第二对和第三对中对应的索引
                        for idx_2, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[1]]):
                            if int(m1) == idx2 and int(m2) == idx3:
                                checked_indices[1].add(idx_2)
                                break
                                
                        for idx_3, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[2]]):
                            if int(m1) == idx3 and int(m2) == idx1_cycle:
                                checked_indices[2].add(idx_3)
                                break
                        
                        # 检查循环是否一致
                        if idx1_cycle == idx1:
                            # 循环一致，标记为一致的匹配
                            consistent_indices[0].add(idx)
                            
                            # 找到第二对和第三对中对应的索引
                            for idx_2, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[1]]):
                                if int(m1) == idx2 and int(m2) == idx3:
                                    consistent_indices[1].add(idx_2)
                                    break
                                    
                            for idx_3, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[2]]):
                                if int(m1) == idx3 and int(m2) == idx1_cycle:
                                    consistent_indices[2].add(idx_3)
                                    break
            
            # 更新掩码 - 只将不一致的匹配设为False
            for i in range(3):
                for idx in checked_indices[i]:
                    if idx not in consistent_indices[i]:
                        match_masks[pair_keys[i]][idx] = False
            
            # 统计信息
            for i in range(3):
                cycle_total_checked += len(checked_indices[i])
                cycle_filtered_count += len(checked_indices[i]) - len(consistent_indices[i])
                
                if verbose and checked_indices[i]:
                    consist_ratio = len(consistent_indices[i]) / len(checked_indices[i]) * 100
                    print(f"{pair_keys[i]}: 循环检查了 {len(checked_indices[i])} 个匹配, 一致: {len(consistent_indices[i])} ({consist_ratio:.1f}%)")
    
    # 应用循环一致性掩码
    for key_pair in filtered_matches_dict:
        if key_pair in match_masks:
            original_count = len(filtered_matches_dict[key_pair])
            filtered_matches_dict[key_pair] = filtered_matches_dict[key_pair][match_masks[key_pair]]
            filtered_count = original_count - len(filtered_matches_dict[key_pair])
            
            if verbose and filtered_count > 0:
                print(f"{key_pair}: 循环一致性过滤移除 {filtered_count} 对匹配 ({filtered_count/original_count*100:.1f}%)，剩余 {len(filtered_matches_dict[key_pair])} 对")
    
    if verbose and cycle_total_checked > 0:
        print(f"循环一致性总计检查: {cycle_total_checked} 对匹配, 过滤: {cycle_filtered_count} 对 ({cycle_filtered_count/cycle_total_checked*100:.1f}%)")
    
    # 保存循环一致性检查后的状态
    for key_pair in filtered_matches_dict:
        stats["循环一致性检查后"][key_pair] = len(filtered_matches_dict[key_pair])
    
    # 后续代码保持不变...
    # 4. 结合空间一致性检查 (RANSAC)
    ransac_filtered_count = 0
    ransac_total_count = 0
    
    for key_pair in list(filtered_matches_dict.keys()):
        if len(filtered_matches_dict[key_pair]) < 5:
            continue
        
        key1, key2 = key_pair.split('-')
        matches = filtered_matches_dict[key_pair]
        
        # 记录RANSAC前的匹配数
        before_count = len(matches)
        ransac_total_count += before_count
        
        # 获取匹配点坐标
        kp1 = features_data[key1]['kp']
        kp2 = features_data[key2]['kp']
        
        matched_pts1 = kp1[matches[:, 0]]
        matched_pts2 = kp2[matches[:, 1]]
        
        # 使用RANSAC计算几何模型 (基础矩阵)
        if isinstance(matched_pts1, np.ndarray) and isinstance(matched_pts2, np.ndarray):
            try:
                _, mask = cv2.findFundamentalMat(
                    matched_pts1, matched_pts2, 
                    cv2.USAC_MAGSAC, 1.0, 0.999, 10000
                )
                
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    
                    # 应用RANSAC筛选
                    mask = mask.ravel().astype(bool)
                    filtered_matches_dict[key_pair] = matches[mask]
                    
                    # 统计过滤数量
                    after_count = len(filtered_matches_dict[key_pair])
                    removed = before_count - after_count
                    ransac_filtered_count += removed
                    
                    if verbose and removed > 0:
                        print(f"{key_pair}: RANSAC过滤移除 {removed} 对匹配 ({removed/before_count*100:.1f}%)，剩余 {after_count} 对")
            except Exception as e:
                if verbose:
                    print(f"{key_pair}: RANSAC计算出错 - {e}")
    
    # 保存RANSAC后的状态
    for key_pair in filtered_matches_dict:
        stats["RANSAC过滤后"][key_pair] = len(filtered_matches_dict[key_pair])
    
    if verbose and ransac_total_count > 0:
        print(f"RANSAC总计过滤: {ransac_filtered_count}/{ransac_total_count} 对匹配 ({ransac_filtered_count/ransac_total_count*100:.1f}%)")
    
    # 5. 检查匹配点分布 - 使用DBSCAN聚类
    cluster_filtered_count = 0
    cluster_total_count = 0
    
    # for key_pair in list(filtered_matches_dict.keys()):
    #     if len(filtered_matches_dict[key_pair]) < 5:
    #         continue
        
    #     key1, key2 = key_pair.split('-')
    #     matches = filtered_matches_dict[key_pair]
        
    #     # 记录聚类前的匹配数
    #     before_count = len(matches)
    #     cluster_total_count += before_count
        
    #     # 获取匹配点坐标
    #     kp1 = features_data[key1]['kp'].cpu().numpy()
    #     kp2 = features_data[key2]['kp'].cpu().numpy()
        
    #     matched_pts1 = kp1[matches[:, 0]]
    #     matched_pts2 = kp2[matches[:, 1]]
        
    #     # 检查匹配点分布 - 使用DBSCAN聚类
    #     try:
    #         # 分析匹配点在两张图像中的分布
    #         db1 = DBSCAN(eps=30, min_samples=3).fit(matched_pts1)
    #         db2 = DBSCAN(eps=30, min_samples=3).fit(matched_pts2)
            
    #         labels1 = db1.labels_
    #         labels2 = db2.labels_
            
    #         # 计算有效聚类数
    #         n_clusters1 = len(set(labels1)) - (1 if -1 in labels1 else 0)
    #         n_clusters2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
            
    #         if verbose:
    #             print(f"{key_pair}: 聚类分析 - 图1: {n_clusters1}个聚类, 图2: {n_clusters2}个聚类")
            
    #         # 当一张图像中的点形成多个聚类而另一张只有一个聚类时，
    #         # 可能是将不同物体的相似部分错误匹配在一起
    #         if max(n_clusters1, n_clusters2) > 1 and min(n_clusters1, n_clusters2) == 1:
    #             if key_pair not in suspicious_pairs:
    #                 suspicious_pairs.append(key_pair)
    #                 if verbose:
    #                     print(f"{key_pair}: 聚类不平衡，标记为可疑")
            
    #         # 如果聚类分析表明存在多个不一致的组，尝试保留主要聚类
    #         if n_clusters1 > 1 or n_clusters2 > 1:
    #             # 找出最大的聚类对
    #             cluster_pairs = {}
    #             for i in range(len(labels1)):
    #                 if labels1[i] == -1 or labels2[i] == -1:
    #                     continue
    #                 pair = (labels1[i], labels2[i])
    #                 if pair not in cluster_pairs:
    #                     cluster_pairs[pair] = 0
    #                 cluster_pairs[pair] += 1
                
    #             if cluster_pairs:
    #                 main_pair = max(cluster_pairs.items(), key=lambda x: x[1])[0]
                    
    #                 # 筛选属于主要聚类对的匹配
    #                 mask = np.zeros(len(matches), dtype=bool)
    #                 for i in range(len(labels1)):
    #                     if (labels1[i], labels2[i]) == main_pair:
    #                         mask[i] = True
                    
    #                 # 应用聚类筛选
    #                 filtered_matches_dict[key_pair] = matches[mask]
                    
    #                 # 统计过滤数量
    #                 after_count = len(filtered_matches_dict[key_pair])
    #                 removed = before_count - after_count
    #                 cluster_filtered_count += removed
                    
    #                 if verbose and removed > 0:
    #                     print(f"{key_pair}: 聚类过滤移除 {removed} 对匹配 ({removed/before_count*100:.1f}%)，剩余 {after_count} 对")
    #     except Exception as e:
    #         if verbose:
    #             print(f"{key_pair}: 聚类分析出错 - {e}")
    
    # # 保存聚类分析后的状态
    # for key_pair in filtered_matches_dict:
    #     stats["聚类分析后"][key_pair] = len(filtered_matches_dict[key_pair])
    
    # if verbose and cluster_total_count > 0:
    #     print(f"聚类分析总计过滤: {cluster_filtered_count}/{cluster_total_count} 对匹配 ({cluster_filtered_count/cluster_total_count*100:.1f}%)")
    
    # ----- 移除了描述子距离过滤部分 -----
    
    # 7. 处理可疑匹配对
    # suspicious_removed_count = 0
    # suspicious_reduced_count = 0
    # suspicious_pairs_limit = 10
    # for key_pair in suspicious_pairs:
    #     if key_pair in filtered_matches_dict:
    #         before_count = len(filtered_matches_dict[key_pair])
            
    #         # 可选：完全移除或减少匹配数量
    #         if len(filtered_matches_dict[key_pair]) > 10:
    #             # 保留最好的少量匹配
    #             filtered_matches_dict[key_pair] = filtered_matches_dict[key_pair][:10]
    #             after_count = len(filtered_matches_dict[key_pair])
    #             suspicious_reduced_count += 1
                
    #             if verbose:
    #                 print(f"{key_pair}: 可疑匹配，减少至 {after_count} 对 (从 {before_count} 对)")
    #         else:
    #             del filtered_matches_dict[key_pair]
    #             suspicious_removed_count += 1
                
    #             if verbose:
    #                 print(f"{key_pair}: 可疑匹配，完全移除 {before_count} 对")
    
        # 1. 限制每对图像的最大匹配点数量
    limited_count = 0
    for key_pair in list(filtered_matches_dict.keys()):
        matches = filtered_matches_dict[key_pair]
        if len(matches) > 2000:
            # 排序后只保留最好的max_matches个匹配（假设匹配已按质量排序）
            filtered_matches_dict[key_pair] = matches[:2000]
            limited_count += 1
            
            if verbose:
                print(f"{key_pair}: 限制匹配数量从 {len(matches)} 到 {2000}")


    # 保存最终状态
    for key_pair in list(filtered_matches_dict.keys()):
        stats["最终"][key_pair] = len(filtered_matches_dict[key_pair])
    
    # 对已删除的匹配对，在最终状态中标记为0
    for key_pair in matches_dict:
        if key_pair not in filtered_matches_dict:
            stats["最终"][key_pair] = 0
    
    if verbose:
        print(f"\n匹配过滤统计摘要:")
        print(f"共处理 {len(matches_dict)} 对匹配")
        print(f"保留匹配: {len(filtered_matches_dict)} 对\n")
        
        # 打印每个阶段的匹配数量变化
        print(f"各阶段匹配数量变化:")
        # 获取所有key_pair
        all_pairs = set()
        for stage in stats:
            all_pairs.update(stats[stage].keys())
        
        # 打印表头
        stages = list(stats.keys())
        header = "匹配对        "
        for stage in stages:
            header += f" | {stage}"
        print(header)
        print("-" * len(header))
        
        # 按初始匹配数量排序
        sorted_pairs = sorted(all_pairs, key=lambda x: stats["初始"].get(x, 0), reverse=True)
        
        # 打印每对的统计
        for pair in sorted_pairs:
            line = f"{pair:12}"
            for stage in stages:
                value = stats[stage].get(pair, "-")
                line += f" | {value:6}"
            print(line)
    
    return filtered_matches_dict

def check_cycle_consistency(keys, matches_dict, features_data):
    """
    检查三元环的匹配一致性，只考虑三张图像中共同可见的特征点
    """
    import numpy as np
    
    # 获取三对匹配
    pairs = [
        (keys[0], keys[1]),
        (keys[1], keys[2]),
        (keys[2], keys[0])
    ]
    
    # 检查所有匹配对是否存在
    for key1, key2 in pairs:
        pair_key = f"{key1}-{key2}"
        reverse_key = f"{key2}-{key1}"
        if pair_key not in matches_dict and reverse_key not in matches_dict:
            return 0.0  # 缺失匹配对，一致性为0
    
    # 构建匹配索引查找表
    match_maps = []
    for key1, key2 in pairs:
        pair_key = f"{key1}-{key2}"
        if pair_key in matches_dict:
            matches = matches_dict[pair_key]
        else:
            reverse_key = f"{key2}-{key1}"
            matches = matches_dict[reverse_key]
            # 调整索引顺序
            matches = np.fliplr(matches)
            
        # 创建从idx1到idx2的映射
        match_map = {int(m[0]): int(m[1]) for m in matches}
        match_maps.append(match_map)
    
    # 检查循环一致性
    consistent_count = 0
    total_checked = 0  # 只计算我们实际检查的点数
    
    # 取第一对的匹配点
    first_matches = list(match_maps[0].items())
    
    for idx1, idx2 in first_matches:
        # 只检查可能形成闭环的点
        if idx2 in match_maps[1]:
            idx3 = match_maps[1][idx2]
            if idx3 in match_maps[2]:
                # 这个点确实在三张图像中都有对应，可以进行闭环检查
                total_checked += 1
                idx1_cycle = match_maps[2][idx3]
                if idx1_cycle == idx1:
                    consistent_count += 1
    
    # 如果没有可检查的完整循环，返回0
    if total_checked == 0:
        return 1
        
    # 只计算实际检查过的点的一致性比例
    return consistent_count / total_checked

def filter_matches_semantic(frames, matches_dict, features_data, threshold=0.6):
    """
    基于语义相似度的匹配过滤函数
    
    Args:
        frames: 图像路径列表
        matches_dict: 匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        threshold: 相似度阈值
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
    """
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    
    # 尝试导入CLIP模型（如果可用）
    try:
        from CLIP.clip import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
        semantic_available = True
    except:
        semantic_available = False
        print("无法加载CLIP模型，跳过语义相似度检查")
    
    filtered_matches_dict = matches_dict.copy()
    
    if not semantic_available:
        return filtered_matches_dict
    
    # 提取图像区域的语义特征
    for key_pair in list(filtered_matches_dict.keys()):
        key1, key2 = key_pair.split('-')
        matches = filtered_matches_dict[key_pair]
        
        if len(matches) < 10:
            continue
            
        # 找到frame路径
        frame1 = None
        frame2 = None
        for frame in frames:
            if key1 in frame:
                frame1 = frame
            if key2 in frame:
                frame2 = frame
                
        if frame1 is None or frame2 is None:
            continue
            
        # 加载图像
        try:
            img1 = Image.open(frame1)
            img2 = Image.open(frame2)
            
            # 获取匹配点坐标
            kp1 = features_data[key1]['kp']
            kp2 = features_data[key2]['kp']
            
            matched_pts1 = kp1[matches[:, 0]]
            matched_pts2 = kp2[matches[:, 1]]
            
            # 计算匹配点的凸包并提取区域
            if len(matched_pts1) > 3:
                hull1 = cv2.convexHull(matched_pts1.astype(np.int32))
                hull2 = cv2.convexHull(matched_pts2.astype(np.int32))
                
                # 创建掩码
                mask1 = np.zeros(img1.size[::-1], dtype=np.uint8)
                mask2 = np.zeros(img2.size[::-1], dtype=np.uint8)
                
                cv2.fillPoly(mask1, [hull1], 255)
                cv2.fillPoly(mask2, [hull2], 255)
                
                # 计算边界框
                x1, y1, w1, h1 = cv2.boundingRect(hull1)
                x2, y2, w2, h2 = cv2.boundingRect(hull2)
                
                # 裁剪区域
                region1 = np.array(img1)[y1:y1+h1, x1:x1+w1]
                region2 = np.array(img2)[y2:y2+h2, x2:x2+w2]
                
                # 转回PIL图像
                region1_pil = Image.fromarray(region1)
                region2_pil = Image.fromarray(region2)
                
                # 计算CLIP特征
                with torch.no_grad():
                    region1_features = model.encode_image(preprocess(region1_pil).unsqueeze(0).to(device))
                    region2_features = model.encode_image(preprocess(region2_pil).unsqueeze(0).to(device))
                    
                    # 正则化
                    region1_features = region1_features / region1_features.norm(dim=-1, keepdim=True)
                    region2_features = region2_features / region2_features.norm(dim=-1, keepdim=True)
                    
                    # 计算相似度
                    similarity = (region1_features @ region2_features.T).item()
                    
                    # 低相似度可能表示虚假匹配
                    if similarity < threshold:
                        # 根据相似度调整保留的匹配点数量
                        keep_ratio = max(0.1, similarity / threshold)
                        keep_count = max(5, int(len(matches) * keep_ratio))
                        filtered_matches_dict[key_pair] = matches[:keep_count]
        except Exception as e:
            print(f"处理{key_pair}时出错: {e}")
            continue
    
    return filtered_matches_dict

def adaptive_match_filtering(matcher, kp1, kp2, matches, img1_path, img2_path, device, min_matches=15):
    """
    使用多种策略自适应地过滤匹配对
    
    Args:
        matcher: LightGlue匹配器
        kp1, kp2: 两张图片的特征点
        matches: 原始匹配结果 (Nx2)
        img1_path, img2_path: 图像路径
        device: 设备
        min_matches: 最小匹配数
        
    Returns:
        filtered_matches: 过滤后的匹配索引 (Mx2)
    """
    import numpy as np
    import cv2
    import torch
    from sklearn.cluster import DBSCAN
    
    if len(matches) <= min_matches:
        return matches  # 匹配数量已经很少，不再过滤
    
    # 转换为numpy数组以便处理
    if isinstance(matches, torch.Tensor):
        matches_np = matches.cpu().numpy()
    else:
        matches_np = matches
        
    if isinstance(kp1, torch.Tensor):
        kp1_np = kp1.cpu().numpy()
    else:
        kp1_np = kp1
        
    if isinstance(kp2, torch.Tensor):
        kp2_np = kp2.cpu().numpy()
    else:
        kp2_np = kp2
    
    # 1. 首先应用RANSAC过滤 (基础矩阵)
    try:
        pts1 = kp1_np[matches_np[:, 0]]
        pts2 = kp2_np[matches_np[:, 1]]
        
        # 使用MAGSAC算法，较为鲁棒
        _, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.USAC_MAGSAC,
            1.0, 0.999, 10000
        )
        
        if mask is not None:
            mask = mask.ravel().astype(bool)
            matches_np = matches_np[mask]
    except Exception as e:
        print(f"RANSAC过滤出错: {e}")
    
    # 如果匹配数太少，返回RANSAC结果
    if len(matches_np) <= min_matches:
        return matches_np
    
    # 2. 空间一致性检查 - 使用DBSCAN聚类检测异常匹配
    try:
        pts1 = kp1_np[matches_np[:, 0]]
        pts2 = kp2_np[matches_np[:, 1]]
        
        # 动态确定eps参数
        pts_range1 = np.max(pts1, axis=0) - np.min(pts1, axis=0)
        pts_range2 = np.max(pts2, axis=0) - np.min(pts2, axis=0)
        avg_range = (np.mean(pts_range1) + np.mean(pts_range2)) / 2
        eps = max(20, avg_range * 0.05)  # 自适应聚类距离
        
        # 对两张图的点分别进行聚类
        db1 = DBSCAN(eps=eps, min_samples=3).fit(pts1)
        db2 = DBSCAN(eps=eps, min_samples=3).fit(pts2)
        
        labels1 = db1.labels_
        labels2 = db2.labels_
        
        # 检查聚类一致性
        cluster_map = {}
        for i in range(len(labels1)):
            if labels1[i] == -1 or labels2[i] == -1:
                continue  # 跳过噪声点
                
            cluster_pair = (labels1[i], labels2[i])
            if cluster_pair not in cluster_map:
                cluster_map[cluster_pair] = 0
            cluster_map[cluster_pair] += 1
        
        # 保留主要聚类对的匹配
        valid_mask = np.zeros(len(matches_np), dtype=bool)
        for i in range(len(labels1)):
            cluster_pair = (labels1[i], labels2[i])
            if cluster_pair in cluster_map and cluster_map[cluster_pair] >= 3:
                valid_mask[i] = True
        
        # 如果过滤后的匹配太少，保留所有非噪声点
        if np.sum(valid_mask) < min_matches:
            valid_mask = (labels1 != -1) & (labels2 != -1)
            
        matches_np = matches_np[valid_mask]
    except Exception as e:
        print(f"聚类分析出错: {e}")
    
    # 3. 主方向一致性检查
    try:
        if len(matches_np) > min_matches:
            pts1 = kp1_np[matches_np[:, 0]]
            pts2 = kp2_np[matches_np[:, 1]]
            
            # 计算匹配点之间的方向向量
            directions = []
            for i in range(len(pts1)-1):
                for j in range(i+1, len(pts1)):
                    # 第一张图中两点的方向
                    dir1 = pts1[j] - pts1[i]
                    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-10)
                    
                    # 第二张图中对应两点的方向
                    dir2 = pts2[j] - pts2[i]
                    dir2 = dir2 / (np.linalg.norm(dir2) + 1e-10)
                    
                    # 计算方向相似度 (点积)
                    sim = np.dot(dir1, dir2)
                    directions.append(sim)
            
            # 分析方向一致性
            directions = np.array(directions)
            mean_dir = np.mean(directions)
            std_dir = np.std(directions)
            
            # 方向一致性低，可能是错误匹配
            if mean_dir < 0.6 or std_dir > 0.4:
                # 减少保留的匹配数量
                keep_count = max(min_matches, len(matches_np) // 2)
                matches_np = matches_np[:keep_count]
    except Exception as e:
        print(f"方向一致性检查出错: {e}")
    
    # 如果是torch.Tensor，转回tensor
    if isinstance(matches, torch.Tensor):
        return torch.tensor(matches_np, device=matches.device, dtype=matches.dtype)
    else:
        return matches_np