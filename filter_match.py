import sklearn
import numpy as np
import cv2

def filter_matches_by_region_consistency(kpts0, kpts1, matches, img1_path, img2_path, device):
    """基于区域一致性筛选匹配点对"""
    # 1. 提取原始匹配点对
    if len(matches) < 8:  # 至少需要8对点来估计基础矩阵
        return matches
        
    src_pts = kpts0[matches[:, 0]]
    dst_pts = kpts1[matches[:, 1]]
    
    # 2. 使用DBSCAN进行空间聚类，找出密集匹配区域
    from sklearn.cluster import DBSCAN
    
    # 在源图像空间进行聚类
    clustering = DBSCAN(eps=50, min_samples=5).fit(src_pts.cpu().numpy())
    labels = clustering.labels_
    
    # 3. 对每个聚类区域验证几何一致性
    filtered_matches = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # 跳过噪声点
            continue
            
        # 获取当前聚类的匹配点
        cluster_mask = labels == cluster_id
        cluster_matches = matches[cluster_mask]
        
        if len(cluster_matches) < 8:
            continue
            
        # 提取该区域的匹配点
        cluster_src = src_pts[cluster_mask]
        cluster_dst = dst_pts[cluster_mask]
        
        # 计算该区域的基础矩阵或单应性矩阵
        F, inliers = cv2.findFundamentalMat(
            cluster_src.cpu().numpy(),
            cluster_dst.cpu().numpy(),
            cv2.FM_RANSAC,
            3.0,  # 像素误差阈值
            0.99  # 置信度
        )
        
        # 如果没有找到有效的变换模型，跳过
        if F is None or inliers is None:
            continue
            
        # 将内点添加到过滤后的匹配中
        inliers = inliers.ravel().astype(bool)
        filtered_matches.append(cluster_matches[inliers])
    
    if len(filtered_matches) == 0:
        return matches  # 如果过滤后没有匹配点，返回原始匹配
        
    # 合并所有有效的匹配点
    filtered_matches = np.vstack(filtered_matches)
    return filtered_matches

def filter_matches_by_patch_similarity(kpts0, kpts1, matches, img1_path, img2_path, patch_size=21, threshold=0.8):
    """基于局部区域外观相似性筛选匹配点对"""
    # 加载原始图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return matches
    
    # 转换关键点为整数坐标
    kpts0_np = kpts0.cpu().numpy().astype(int)
    kpts1_np = kpts1.cpu().numpy().astype(int)
    
    # 半径
    radius = patch_size // 2
    
    # 计算每对匹配点的图像块相似度
    similarities = []
    valid_matches = []
    
    for idx, (i, j) in enumerate(matches):
        # 获取关键点坐标
        x1, y1 = kpts0_np[i]
        x2, y2 = kpts1_np[j]
        
        # 确保点不在边界上
        if (x1 < radius or y1 < radius or x1 >= img1.shape[1] - radius or y1 >= img1.shape[0] - radius or
            x2 < radius or y2 < radius or x2 >= img2.shape[1] - radius or y2 >= img2.shape[0] - radius):
            continue
        
        # 提取图像块
        patch1 = img1[y1-radius:y1+radius+1, x1-radius:x1+radius+1]
        patch2 = img2[y2-radius:y2+radius+1, x2-radius:x2+radius+1]
        
        # 计算相似度（使用归一化互相关）
        similarity = cv2.matchTemplate(patch1, patch2, cv2.TM_CCORR_NORMED)[0, 0]
        
        if similarity > threshold:
            valid_matches.append(idx)
            similarities.append(similarity)
    
    if len(valid_matches) == 0:
        return np.zeros((0,2))  # 如果过滤后没有匹配点，返回原始匹配
        
    return matches[valid_matches]

def filter_matches_by_color_similarity(kpts0, kpts1, matches, img1_path, img2_path, patch_size=51, threshold=0.7):
    """基于颜色相似性筛选匹配点对"""
    # 加载原始彩色图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return matches
    
    # 转换关键点为整数坐标
    kpts0_np = kpts0.cpu().numpy().astype(int)
    kpts1_np = kpts1.cpu().numpy().astype(int)
    
    # 半径
    radius = patch_size // 2
    
    # 计算每对匹配点的颜色相似度
    valid_matches = []
    
    for idx, (i, j) in enumerate(matches):
        # 获取关键点坐标
        x1, y1 = kpts0_np[i]
        x2, y2 = kpts1_np[j]
        
        # 确保点不在边界上
        if (x1 < radius or y1 < radius or x1 >= img1.shape[1] - radius or y1 >= img1.shape[0] - radius or
            x2 < radius or y2 < radius or x2 >= img2.shape[1] - radius or y2 >= img2.shape[0] - radius):
            continue
        
        # 提取彩色图像块
        patch1 = img1[y1-radius:y1+radius+1, x1-radius:x1+radius+1]
        patch2 = img2[y2-radius:y2+radius+1, x2-radius:x2+radius+1]
        cv2.imwrite('patch1.png', patch1)
        cv2.imwrite('patch2.png', patch2)

        # 计算颜色直方图
        hist1_b = cv2.calcHist([patch1], [0], None, [8], [0, 256])
        hist1_g = cv2.calcHist([patch1], [1], None, [8], [0, 256])
        hist1_r = cv2.calcHist([patch1], [2], None, [8], [0, 256])
        
        hist2_b = cv2.calcHist([patch2], [0], None, [8], [0, 256])
        hist2_g = cv2.calcHist([patch2], [1], None, [8], [0, 256])
        hist2_r = cv2.calcHist([patch2], [2], None, [8], [0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1_b, hist1_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1, cv2.NORM_MINMAX)
        
        # 比较直方图 (巴氏距离越小表示越相似)
        sim_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_BHATTACHARYYA)
        sim_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_BHATTACHARYYA)
        sim_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_BHATTACHARYYA)
        
        sim_count = 3
        if np.isnan(sim_b):
            sim_b = 0.0
            sim_count -= 1
        if np.isnan(sim_g):
            sim_g = 0.0
            sim_count -= 1
        if np.isnan(sim_r):
            sim_r = 0.0
            sim_count -= 1
        # 计算平均相似度 (1 - 距离，使得值越大表示越相似)
        if sim_count == 0:
            color_similarity = 0.0
        else:
            color_similarity = 1.0 - (sim_b + sim_g + sim_r) / sim_count
        

        # 综合考虑颜色和结构相似度
        combined_similarity = color_similarity
        
        if combined_similarity > threshold:
            valid_matches.append(idx)
    
    if len(valid_matches) == 0 or len(valid_matches) < 8:
        return np.zeros((0,2))  # 如果过滤后匹配点太少，返回原始匹配
        
    return matches[valid_matches]

def advanced_match_filtering(lightglue_matcher, kpts0, kpts1, matches, img1_path, img2_path, device):
    """综合区域分析的高级匹配点筛选"""
    # 1. 首先使用RANSAC进行初步筛选
    if len(matches) < 8:
        return matches
        
    # 转换匹配点格式
    src_pts = kpts0[matches[:, 0]].cpu().numpy()
    dst_pts = kpts1[matches[:, 1]].cpu().numpy()
    
    # RANSAC筛选
    H, inliers_H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    F, inliers_F = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3.0, 0.99)
    
    # 使用更适合场景的模型
    if H is not None and inliers_H is not None and inliers_H.sum() > 10:
        inliers = inliers_H.ravel().astype(bool)
    elif F is not None and inliers_F is not None and inliers_F.sum() > 10:
        inliers = inliers_F.ravel().astype(bool)
    else:
        return matches  # 如果两种方法都失败，返回原始匹配
    
    # 2. 加载图像计算区域相似度
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return matches[inliers]
    
    # 3. 对每个内点匹配计算区域相似度
    filtered_matches = []
    patch_size = 31  # 更大的区域
    radius = patch_size // 2
    
    for idx in np.where(inliers)[0]:
        i, j = matches[idx]
        x1, y1 = src_pts[idx].astype(int)
        x2, y2 = dst_pts[idx].astype(int)
        
        # 确保点不在边界上
        if (x1 < radius or y1 < radius or x1 >= img1.shape[1] - radius or y1 >= img1.shape[0] - radius or
            x2 < radius or y2 < radius or x2 >= img2.shape[1] - radius or y2 >= img2.shape[0] - radius):
            continue
        
        # 提取图像块
        patch1 = cv2.cvtColor(img1[y1-radius:y1+radius+1, x1-radius:x1+radius+1], cv2.COLOR_BGR2GRAY)
        patch2 = cv2.cvtColor(img2[y2-radius:y2+radius+1, x2-radius:x2+radius+1], cv2.COLOR_BGR2GRAY)
        
        # 计算相似度
        similarity = cv2.matchTemplate(patch1, patch2, cv2.TM_CCORR_NORMED)[0, 0]
        
        if similarity > 0.75:  # 相似度阈值
            filtered_matches.append(idx)
    
    if len(filtered_matches) < 8:
        return matches[inliers]  # 如果过滤后匹配点太少，返回RANSAC内点
    
    return matches[filtered_matches]

def adaptive_match_filtering_ori(lightglue_matcher, kpts0, kpts1, matches, img1_path, img2_path, device):
    """根据匹配点数量自适应调整筛选策略"""
    # 如果匹配点太少，完全不筛选
    if len(matches) < 30 or len(matches) > 300:
        return matches
        
    # 如果匹配点数量很多，只做轻量级筛选或不筛选
    if len(matches) > 200:
        # 对于大量匹配，只需要移除明显异常值
        src_pts = kpts0[matches[:, 0]].cpu().numpy()
        dst_pts = kpts1[matches[:, 1]].cpu().numpy()
        
        # 使用较宽松的RANSAC参数
        F, inliers = cv2.findFundamentalMat(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=8.0,  # 非常宽松的阈值
            confidence=0.9
        )
        
        # 如果RANSAC失败或保留点太少，返回原始匹配
        if F is None or inliers is None or inliers.sum() < len(matches) * 0.7:
            return matches
            
        # 只移除明显的离群点
        inliers = inliers.ravel().astype(bool)
        return matches[inliers]
    
    # 对于中等数量的匹配，使用标准RANSAC
    elif len(matches) > 50:
        src_pts = kpts0[matches[:, 0]].cpu().numpy()
        dst_pts = kpts1[matches[:, 1]].cpu().numpy()
        
        # 标准RANSAC参数
        F, inliers = cv2.findFundamentalMat(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            confidence=0.95
        )
        
        if F is None or inliers is None or inliers.sum() < 15:
            return matches
            
        inliers = inliers.ravel().astype(bool)
        return matches[inliers]
    
    # 对于较少数量的匹配，可以尝试更严格的筛选
    else:
        # 使用您原来的advanced_match_filtering方法
        return advanced_match_filtering(lightglue_matcher, kpts0, kpts1, matches, img1_path, img2_path, device)
    
def adaptive_match_filtering(lightglue_matcher, kpts0, kpts1, matches, img1_path, img2_path, device):
    """根据匹配点数量自适应调整筛选策略"""
    # 如果匹配点太少，完全不筛选
    if len(matches) > 150:
        return matches
        
    return filter_matches_by_color_similarity(kpts0, kpts1, matches, img1_path, img2_path)