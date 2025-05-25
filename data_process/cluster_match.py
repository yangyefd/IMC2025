import torch
import os
from tqdm import tqdm 
import h5py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import pickle
from collections import defaultdict

def cluster_and_crop_matches(img_fnames, matches_data, feature_dir='.featureout', 
                            top_cluster_percentage=0.85, device=torch.device('cpu'),
                            output_dir='./cropped_regions', visualize=True):
    """
    Cluster matching points between image pairs, extract the top clusters containing 
    most matching points, and crop the outer boundary of these clusters.
    
    Args:
        img_fnames: List of image file paths
        matches_data: Dictionary of matches {key1-key2: [idxs, scores]} or h5py file path
        feature_dir: Directory containing feature data
        top_cluster_percentage: Percentage of matching points to include (0.8-0.9 recommended)
        device: Computation device
        output_dir: Directory to save cropped regions
        visualize: Whether to visualize the clustering results
        
    Returns:
        crop_regions: Dictionary mapping image keys to their crop regions {key: [x_min, y_min, x_max, y_max]}
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Load feature data
    print("Loading feature data...")
    features_data = {}
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
         h5py.File(f'{feature_dir}/size.h5', mode='r') as f_size:
        for img_path in tqdm(img_fnames):
            key = img_path.split('/')[-1].split('\\')[-1]
            features_data[key] = {
                'kp': torch.from_numpy(f_kp[key][...]).to(device),
                'desc': torch.from_numpy(f_desc[key][...]).to(device),
                'size': torch.from_numpy(f_size[key][...]).to(device),
            }
    
    # Load matches data if it's a file path
    if isinstance(matches_data, str):
        with open(matches_data, 'rb') as f:
            matches_dict = pickle.load(f)
    else:
        matches_dict = matches_data
    
    # Dictionary to store crop regions for each image
    crop_regions = {}
    image_match_points = defaultdict(list)
    
    # First pass: gather all matching points for each image
    print("Gathering matching points...")
    for match_key, match_data in matches_dict.items():
        key1, key2 = match_key.split('-')
        match_idxs = match_data[0]  # Nx2 array of indices
        match_scores = match_data[1] if len(match_data) > 1 else None
        
        # Get keypoints for both images
        kp1 = features_data[key1]['kp'].cpu().numpy()
        kp2 = features_data[key2]['kp'].cpu().numpy()
        
        # Extract matched keypoints
        for idx1, idx2 in match_idxs:
            pt1 = kp1[idx1]
            pt2 = kp2[idx2]
            
            # Store matching points with their source pair
            image_match_points[key1].append((pt1, match_key))
            image_match_points[key2].append((pt2, match_key))
    
    # Second pass: cluster and extract crop regions
    print("Clustering matching points and determining crop regions...")
    for key, points_with_pairs in tqdm(image_match_points.items()):
        # Extract just the points for clustering
        points = np.array([p[0] for p in points_with_pairs])
        
        # Adapt eps based on image size
        img_width = features_data[key]['size'][0][0].item()
        img_height = features_data[key]['size'][0][1].item()
        eps = max(20, img_width * 0.03)  # Adaptive clustering distance
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=5).fit(points)
        labels = db.labels_
        
        # Count points in each cluster
        unique_labels = np.unique(labels)
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels if label != -1}
        
        # Sort clusters by size (descending)
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total points in clusters (excluding noise points with label -1)
        total_clustered_points = sum(cluster_sizes.values())
        
        # Select top clusters that contain at least top_cluster_percentage of points
        selected_clusters = []
        cumulative_points = 0
        
        for label, size in sorted_clusters:
            selected_clusters.append(label)
            cumulative_points += size
            if cumulative_points / total_clustered_points >= top_cluster_percentage:
                break
        
        # If no clusters were found, use all points
        if not selected_clusters:
            # Use bounding box of all points
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])
            
            # Add padding (10% of image size)
            x_pad = img_width * 0.1
            y_pad = img_height * 0.1
            
            crop_regions[key] = [
                max(0, x_min - x_pad),
                max(0, y_min - y_pad),
                min(img_width, x_max + x_pad),
                min(img_height, y_max + y_pad)
            ]
        else:
            # Get points from selected clusters
            mask = np.isin(labels, selected_clusters)
            selected_points = points[mask]
            
            # Compute bounding box of selected points
            x_min = np.min(selected_points[:, 0])
            y_min = np.min(selected_points[:, 1])
            x_max = np.max(selected_points[:, 0])
            y_max = np.max(selected_points[:, 1])
            
            # Add padding (10% of image size)
            x_pad = img_width * 0.1
            y_pad = img_height * 0.1
            
            crop_regions[key] = [
                max(0, x_min - x_pad),
                max(0, y_min - y_pad),
                min(img_width, x_max + x_pad),
                min(img_height, y_max + y_pad)
            ]
        
        # Visualize clustering results
        if visualize:
            # Find the original image path
            img_path = None
            for fname in img_fnames:
                if key in fname:
                    img_path = fname
                    break
            
            if img_path:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Create a visualization image
                vis = img.copy()
                
                # Draw all match points with small dots
                for pt, _ in points_with_pairs:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (80, 80, 80), -1)
                
                # Use different colors for each cluster
                colors = []
                for i in range(len(unique_labels)):
                    h = int(i * 180 / len(unique_labels)) % 180
                    s = 200 + np.random.randint(55)
                    v = 200 + np.random.randint(55)
                    bgr_color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0]
                    colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
                
                # Draw points colored by cluster
                for i, (pt, _) in enumerate(points_with_pairs):
                    label = labels[i]
                    if label != -1:  # Not noise
                        color_idx = np.where(unique_labels == label)[0][0] % len(colors)
                        color = colors[color_idx]
                        # Larger circles for clustered points
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, color, -1)
                
                # Draw selected cluster points with bigger dots
                for i, (pt, _) in enumerate(points_with_pairs):
                    label = labels[i]
                    if label in selected_clusters:
                        color_idx = np.where(unique_labels == label)[0][0] % len(colors)
                        color = colors[color_idx]
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
                
                # Draw bounding box of crop region
                x_min, y_min, x_max, y_max = crop_regions[key]
                cv2.rectangle(vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                
                # Add text showing cluster statistics
                cv2.putText(vis, f"Total points: {len(points)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Selected clusters: {len(selected_clusters)}/{len(cluster_sizes)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Points in selected clusters: {cumulative_points}/{total_clustered_points}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save visualization
                save_path = os.path.join(vis_dir, f'{key}_clusters.png')
                cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                # Also save the cropped image
                cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                crop_path = os.path.join(output_dir, f'{key}_cropped.png')
                cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    
    # Save crop regions to a file for later use
    with open(os.path.join(output_dir, 'crop_regions.pkl'), 'wb') as f:
        pickle.dump(crop_regions, f)
    
    print(f"Processed {len(crop_regions)} images. Cropped regions saved to {output_dir}")
    return crop_regions

def apply_crops_to_matching(extractor, img_fnames, crop_regions, feature_dir='.featureout', 
                           output_dir='./cropped_matching', device=torch.device('cpu'), 
                           use_cropped_images=True):
    """
    Apply the determined crop regions to the images and extract features from these cropped regions
    for improved matching. 可以直接使用预先裁剪好的图像，避免重复裁剪操作。
    
    Args:
        img_fnames: List of image file paths
        crop_regions: Dictionary mapping image keys to crop regions {key: [x_min, y_min, x_max, y_max]}
        feature_dir: Directory containing original feature data
        output_dir: Directory to save new features extracted from cropped images
        device: Computation device
        use_cropped_images: Whether to use pre-cropped images if available
        
    Returns:
        None (saves new feature files in output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有预先裁剪好的图像
    cropped_dir = os.path.dirname(os.path.dirname(crop_regions.get('__path__', '')))
    if not os.path.isdir(cropped_dir):
        cropped_dir = os.path.dirname(os.path.abspath(output_dir))
    
    # 获取所有可能的裁剪图像路径
    cropped_images = {}
    if use_cropped_images:
        for root, _, files in os.walk(cropped_dir):
            for file in files:
                if file.endswith('_cropped.png'):
                    key = file.replace('_cropped.png', '')
                    cropped_images[key] = os.path.join(root, file)
    
    # 处理每张图像
    with h5py.File(f'{output_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{output_dir}/descriptors.h5', mode='w') as f_desc, \
         h5py.File(f'{output_dir}/size.h5', mode='w') as f_size, \
         h5py.File(f'{output_dir}/mask.h5', mode='w') as f_mask:
        
        for img_path in tqdm(img_fnames, desc="处理图像特征"):
            key = img_path.split('/')[-1].split('\\')[-1]
            
            # 如果这个图像有裁剪区域，使用它
            if key in crop_regions:
                # 尝试使用预先裁剪好的图像
                if key in cropped_images and use_cropped_images:
                    cropped_img_path = cropped_images[key]
                    cropped_img = cv2.imread(cropped_img_path)
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    x_min, y_min, x_max, y_max = crop_regions[key]
                else:
                    # 如果没有预裁剪图像，读取原图并裁剪
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告: 无法读取图像 {img_path}，跳过")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 应用裁剪
                    x_min, y_min, x_max, y_max = crop_regions[key]
                    cropped_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                # 转换为张量用于特征提取
                try:
                    cropped_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    cropped_tensor = cropped_tensor.to(device)
                    
                    # 从裁剪图像中提取特征
                    with torch.inference_mode():
                        feats = extractor.extract(cropped_tensor)
                        
                        # 获取关键点并调整回原始图像坐标
                        kpts = feats['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                        kpts[:, 0] += x_min  # 添加裁剪偏移
                        kpts[:, 1] += y_min
                        
                        descs = feats['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                        
                        # 保存到特征文件
                        f_kp[key] = kpts
                        f_desc[key] = descs
                        f_size[key] = [[cropped_img.shape[1], cropped_img.shape[0]]]  # 宽度,高度
                        f_mask[key] = [len(kpts)]  # 记录有效特征点数量
                except Exception as e:
                    print(f"处理图像 {key} 时出错: {e}")
                    # 使用原始特征作为备选
                    try:
                        with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as orig_kp, \
                             h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as orig_desc, \
                             h5py.File(f'{feature_dir}/size.h5', mode='r') as orig_size, \
                             h5py.File(f'{feature_dir}/mask.h5', mode='r') as orig_mask:
                            
                            if key in orig_kp:
                                f_kp[key] = orig_kp[key][...]
                                f_desc[key] = orig_desc[key][...]
                                f_size[key] = orig_size[key][...]
                                f_mask[key] = orig_mask[key][...] if key in orig_mask else [len(orig_kp[key])]
                    except Exception as e2:
                        print(f"无法加载 {key} 的原始特征: {e2}")
            else:
                # 如果没有裁剪区域，使用原始特征
                try:
                    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as orig_kp, \
                         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as orig_desc, \
                         h5py.File(f'{feature_dir}/size.h5', mode='r') as orig_size, \
                         h5py.File(f'{feature_dir}/mask.h5', mode='r') as orig_mask:
                        
                        if key in orig_kp:
                            f_kp[key] = orig_kp[key][...]
                            f_desc[key] = orig_desc[key][...]
                            f_size[key] = orig_size[key][...]
                            f_mask[key] = orig_mask[key][...] if key in orig_mask else [len(orig_kp[key])]
                except Exception as e:
                    print(f"无法加载 {key} 的原始特征: {e}")
    
    print(f"从裁剪区域提取的特征已保存到 {output_dir}")