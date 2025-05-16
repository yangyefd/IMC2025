def cluster_based_filtering(kpts0, kpts1, matches, min_cluster_size=5):
    """
    使用聚类方法过滤匹配点
    
    Args:
        kpts0, kpts1: 关键点坐标
        matches: 匹配索引
        min_cluster_size: 最小聚类大小
        
    Returns:
        过滤后的匹配
    """
    if len(matches) < min_cluster_size:
        return matches
    
    # 提取匹配点坐标
    src_pts = kpts0[matches[:, 0]]
    dst_pts = kpts1[matches[:, 1]]
    
    # 计算匹配点的运动向量
    motions = dst_pts - src_pts
    
    # 使用DBSCAN聚类
    clustering = DBSCAN(eps=10.0, min_samples=min_cluster_size).fit(motions)
    labels = clustering.labels_
    
    # 找出最大聚类
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    
    if len(counts) > 0:
        largest_cluster = unique_labels[np.argmax(counts)]
        filtered_matches = matches[labels == largest_cluster]
        return filtered_matches
    else:
        return matches  # 没有找到足够大的聚类
    
def adaptive_score_filtering(match_matrix, match_scores, quantile=0.7):
    """
    基于匹配分数的自适应阈值过滤
    
    Args:
        match_matrix: 匹配矩阵
        match_scores: 匹配分数
        quantile: 分数阈值百分位数
        
    Returns:
        过滤后的匹配矩阵
    """
    filtered_match_matrix = match_matrix.copy()
    
    for i in range(match_matrix.shape[0]):
        for j in range(match_matrix.shape[1]):
            if match_matrix[i, j] > 0:
                # 获取当前匹配对的分数
                scores = match_scores[(i, j)]
                
                if len(scores) > 0:
                    # 计算自适应阈值
                    threshold = np.quantile(scores, quantile)
                    
                    # 计算有效匹配数量
                    valid_count = np.sum(scores >= threshold)
                    
                    # 更新匹配矩阵
                    filtered_match_matrix[i, j] = valid_count
    
    return filtered_match_matrix

def check_local_consistency(kpts0, kpts1, matches, radius=50, min_neighbors=3):
    """
    检查局部邻域内匹配点的空间一致性
    
    Args:
        kpts0, kpts1: 关键点坐标
        matches: 匹配索引
        radius: 局部邻域半径
        min_neighbors: 最小一致邻居数量
        
    Returns:
        有效匹配的掩码
    """
    src_pts = kpts0[matches[:, 0]]
    dst_pts = kpts1[matches[:, 1]]
    
    # 计算距离矩阵
    dist_src = cdist(src_pts, src_pts)
    dist_dst = cdist(dst_pts, dst_pts)
    
    # 检查每个匹配点的局部一致性
    consistent_mask = np.zeros(len(matches), dtype=bool)
    
    for i in range(len(matches)):
        # 找出源图像中的局部邻居
        neighbors_src = np.where(dist_src[i] < radius)[0]
        
        if len(neighbors_src) < min_neighbors:
            continue
            
        # 计算邻居间的相对距离
        rel_dist_src = dist_src[i, neighbors_src]
        rel_dist_dst = dist_dst[i, neighbors_src]
        
        # 计算相对距离比率
        ratio = rel_dist_dst / (rel_dist_src + 1e-8)
        
        # 判断比率是否接近1（一致的变换）
        consistent_neighbors = np.sum((0.7 < ratio) & (ratio < 1.3))
        
        consistent_mask[i] = consistent_neighbors >= min_neighbors
    
    return consistent_mask

def optimize_match_matrix(match_matrix, features_data, img_fnames, 
                         geometric_check=True, consistency_check=True, 
                         cluster_filtering=True, min_matches=10):
    """
    优化匹配矩阵，去除错误匹配
    
    Args:
        match_matrix: 原始匹配矩阵
        features_data: 特征数据
        img_fnames: 图像文件路径
        geometric_check: 是否进行几何验证
        consistency_check: 是否进行一致性检查
        cluster_filtering: 是否进行聚类过滤
        min_matches: 最小保留匹配数
        
    Returns:
        优化后的匹配矩阵
    """
    optimized_matrix = match_matrix.copy()
    
    # 遍历所有匹配对
    for i in range(match_matrix.shape[0]):
        for j in range(i+1, match_matrix.shape[1]):
            if match_matrix[i, j] < min_matches:
                continue
                
            # 提取特征点和匹配
            key1 = os.path.basename(img_fnames[i])
            key2 = os.path.basename(img_fnames[j])
            
            kpts1 = features_data[key1]['kp'].cpu().numpy()
            kpts2 = features_data[key2]['kp'].cpu().numpy()
            
            # 获取匹配
            matches = get_matches(i, j, match_matrix)  # 根据实际情况实现
            
            filtered_matches = matches
            
            # 1. 几何验证
            if geometric_check and len(matches) >= 8:
                filtered_matches = geometric_verification(kpts1, kpts2, filtered_matches)
            
            # 2. 聚类过滤
            if cluster_filtering and len(filtered_matches) >= min_matches:
                filtered_matches = cluster_based_filtering(
                    kpts1, kpts2, filtered_matches, min_cluster_size=min_matches//2)
            
            # 更新匹配矩阵
            optimized_matrix[i, j] = len(filtered_matches)
            optimized_matrix[j, i] = len(filtered_matches)
            
    return optimized_matrix