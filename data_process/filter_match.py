def ransac_filter(matches, keypoints1, keypoints2, threshold=0.7):
    # 提取匹配点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    # RANSAC估计基础矩阵
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    # 计算内点比例
    inlier_ratio = np.sum(mask) / len(mask)
    
    # 如果内点比例低于阈值，认为是误匹配
    if inlier_ratio < threshold:
        return [], 0
    
    # 返回内点匹配和内点比例
    inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    return inliers, inlier_ratio

def bow_clustering(images, n_clusters=None):
    # 提取SIFT特征
    sift = cv2.SIFT_create()
    descriptors = []
    
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        descriptors.append(des)
    
    # 构建词袋模型
    all_des = np.vstack(descriptors)
    k = n_clusters if n_clusters else min(int(np.sqrt(len(images))), 10)
    
    # K-means聚类创建词典
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, vocabulary = cv2.kmeans(all_des.astype(np.float32), k, None, 
                                       criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 计算每张图像的BoW直方图
    histograms = []
    for des in descriptors:
        hist = np.zeros(k)
        for d in des:
            # 找到最近的视觉词
            dist = np.linalg.norm(vocabulary - d.reshape(1, -1), axis=1)
            idx = np.argmin(dist)
            hist[idx] += 1
        histograms.append(hist / np.sum(hist))  # 归一化
    
    # 对图像聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, scene_labels, _ = cv2.kmeans(np.array(histograms).astype(np.float32), 
                                  min(len(images)//5 + 1, 10), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    return scene_labels

def graph_consistency(matches_dict, threshold=0.6):
    # 构建匹配图
    G = nx.Graph()
    
    # 添加节点和边
    for (i, j), matches in matches_dict.items():
        weight = len(matches)  # 匹配点数量作为边权重
        G.add_edge(i, j, weight=weight, matches=matches)
    
    # 检测极大连通分量 (可能对应不同场景)
    connected_components = list(nx.connected_components(G))
    
    # 对每个连通分量进行一致性检查
    filtered_matches = {}
    for component in connected_components:
        subgraph = G.subgraph(component)
        
        # 移除权重较低的边 (可能是误匹配)
        edges_to_remove = []
        for u, v, data in subgraph.edges(data=True):
            # 计算相对权重
            max_weight = max([d['weight'] for _, _, d in subgraph.edges(data=True)])
            if data['weight'] / max_weight < threshold:
                edges_to_remove.append((u, v))
        
        # 更新过滤后的匹配
        for u, v in subgraph.edges():
            if (u, v) not in edges_to_remove and (v, u) not in edges_to_remove:
                filtered_matches[(u, v)] = subgraph[u][v]['matches']
    
    return filtered_matches

def semantic_similarity_filter(images, matches_dict, threshold=0.75):
    # 加载预训练模型
    model = tf.keras.applications.ResNet50(include_top=False, pooling='avg')
    
    # 提取图像特征
    features = {}
    for idx, img in enumerate(images):
        # 预处理
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(img_rgb, axis=0)
        )
        # 提取特征
        features[idx] = model.predict(img_array).flatten()
    
    # 过滤匹配对
    filtered_matches = {}
    for (i, j), matches in matches_dict.items():
        # 计算余弦相似度
        similarity = np.dot(features[i], features[j]) / (
            np.linalg.norm(features[i]) * np.linalg.norm(features[j])
        )
        
        if similarity >= threshold:
            filtered_matches[(i, j)] = matches
    
    return filtered_matches

def integrated_filter(images, initial_matches):
    # 1. 使用RANSAC过滤明显的误匹配
    ransac_filtered = {}
    for (i, j), matches in initial_matches.items():
        inliers, ratio = ransac_filter(matches, keypoints[i], keypoints[j])
        if ratio > 0.5:  # 保守阈值
            ransac_filtered[(i, j)] = inliers
    
    # 2. 场景聚类
    scene_labels = bow_clustering(images)
    
    # 3. 去除跨场景匹配
    scene_filtered = {}
    for (i, j), matches in ransac_filtered.items():
        if scene_labels[i] == scene_labels[j]:
            scene_filtered[(i, j)] = matches
    
    # 4. 图一致性检查
    graph_filtered = graph_consistency(scene_filtered)
    
    # 5. 语义相似度检查
    semantic_filtered = semantic_similarity_filter(images, graph_filtered)
    
    # 6. 回环检测与优化
    loops = loop_closure_detection(images, semantic_filtered)
    
    # 返回最终过滤结果
    return semantic_filtered, scene_labels, loops