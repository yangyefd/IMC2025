import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

class FeatureClassifier(nn.Module):
    """
    用于分类图像特征向量的神经网络模型
    """
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=2):
        super(FeatureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class PairDataset(Dataset):
    """
    图像对数据集
    """
    def __init__(self, feature_pairs, labels):
        self.feature_pairs = feature_pairs  # 列表[(feature1, feature2), ...]
        self.labels = labels  # 列表[0或1, ...]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feat1, feat2 = self.feature_pairs[idx]
        # 将两个特征向量连接在一起
        combined_feature = torch.cat([feat1, feat2])
        return combined_feature, self.labels[idx]

def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    vec1_norm = vec1 / vec1.norm(dim=0)
    vec2_norm = vec2 / vec2.norm(dim=0)
    return torch.dot(vec1_norm, vec2_norm)

def prepare_initial_training_data(match_dict, features, match_threshold=400):
    """
    根据匹配数量准备初始训练数据
    
    Args:
        match_dict: 包含所有匹配关系的字典 {(idx1, idx2): [matches, ...]}
        features: 所有图像的特征向量 {img_idx: feature_vector}
        match_threshold: 匹配对数量的阈值，大于此值被视为正样本
    
    Returns:
        训练数据和标签
    """
    positive_pairs = []
    negative_pairs = []
    labels = []
    
    # 创建图像匹配数量字典
    img_match_count = defaultdict(lambda: defaultdict(int))
    for (idx1, idx2), matches in match_dict.items():
        # 记录匹配数量
        img_match_count[idx1][idx2] = len(matches)
        img_match_count[idx2][idx1] = len(matches)
    
    # 创建所有图像的索引集合
    all_indices = set(features.keys())
    
    # 为每个图像寻找正样本和负样本
    for img_idx in all_indices:
        # 找出与当前图像匹配最多的图像作为正样本
        matched_imgs = [(other_idx, count) for other_idx, count in img_match_count[img_idx].items()]
        matched_imgs.sort(key=lambda x: x[1], reverse=True)
        
        # 选择匹配数量超过阈值的作为正样本
        for other_idx, count in matched_imgs:
            if count > match_threshold:
                positive_pairs.append((features[img_idx], features[other_idx]))
                labels.append(1)  # 1表示正样本
        
        # 选择没有匹配的图像作为负样本
        unmatched_imgs = list(all_indices - set(img_match_count[img_idx].keys()) - {img_idx})
        # 最多选择与正样本相同数量的负样本
        negative_count = min(len(positive_pairs), len(unmatched_imgs))
        
        if negative_count > 0:
            random_negative_indices = random.sample(unmatched_imgs, negative_count)
            for neg_idx in random_negative_indices:
                negative_pairs.append((features[img_idx], features[neg_idx]))
                labels.append(0)  # 0表示负样本
    
    # 合并正负样本
    all_pairs = positive_pairs + negative_pairs
    
    return all_pairs, labels

def train_classifier(model, train_loader, val_loader, device, 
                    epochs=10, learning_rate=0.001, patience=5):
    """
    训练分类器模型
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
        epochs: 训练轮数
        learning_rate: 学习率
        patience: 早停耐心值
    
    Returns:
        训练好的模型
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_classifier.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_classifier.pth'))
    return model

def predict_pairs(model, features, device, batch_size=128):
    """
    预测所有图像对的匹配概率
    
    Args:
        model: 训练好的模型
        features: 所有图像的特征向量 {img_idx: feature_vector}
        device: 训练设备
        batch_size: 批处理大小
        
    Returns:
        预测结果 {(idx1, idx2): probability}
    """
    model.eval()
    predictions = {}
    
    # 获取所有图像索引
    all_indices = list(features.keys())
    total_pairs = len(all_indices) * (len(all_indices) - 1) // 2
    
    with torch.no_grad():
        batch_features = []
        batch_pairs = []
        
        with tqdm(total=total_pairs, desc="Predicting") as pbar:
            for i in range(len(all_indices)):
                for j in range(i + 1, len(all_indices)):
                    idx1, idx2 = all_indices[i], all_indices[j]
                    
                    # 组合特征向量
                    combined_feature = torch.cat([features[idx1], features[idx2]])
                    batch_features.append(combined_feature)
                    batch_pairs.append((idx1, idx2))
                    
                    # 达到批处理大小时进行预测
                    if len(batch_features) == batch_size:
                        batch_tensor = torch.stack(batch_features).to(device)
                        outputs = model(batch_tensor)
                        probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率
                        
                        # 存储预测结果
                        for k in range(len(batch_pairs)):
                            predictions[batch_pairs[k]] = probs[k].item()
                        
                        # 清空批处理
                        batch_features = []
                        batch_pairs = []
                        pbar.update(batch_size)
            
            # 处理剩余的数据
            if batch_features:
                batch_tensor = torch.stack(batch_features).to(device)
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                for k in range(len(batch_pairs)):
                    predictions[batch_pairs[k]] = probs[k].item()
                
                pbar.update(len(batch_features))
    
    return predictions

def create_high_conf_matchdict(predictions, threshold=0.8):
    """
    根据预测结果创建高置信度匹配字典
    
    Args:
        predictions: 预测结果 {(idx1, idx2): probability}
        threshold: 概率阈值
        
    Returns:
        高置信度匹配字典 {(idx1, idx2): 1}
    """
    high_conf_matches = {}
    for pair, prob in predictions.items():
        if prob >= threshold:
            high_conf_matches[pair] = 1  # 1 作为占位符，实际上我们只关心键
    
    return high_conf_matches

def prepare_new_training_data(match_dict, high_conf_matches, features):
    """
    根据之前的匹配和高置信度匹配准备新的训练数据
    
    Args:
        match_dict: 原始匹配字典
        high_conf_matches: 高置信度匹配字典
        features: 所有图像的特征向量
        
    Returns:
        新的训练数据和标签
    """
    positive_pairs = []
    negative_pairs = []
    labels = []
    
    # 原始正样本
    for (idx1, idx2), matches in match_dict.items():
        if len(matches) > 0:  # 有匹配点就视为正样本
            positive_pairs.append((features[idx1], features[idx2]))
            labels.append(1)
    
    # 高置信度预测的正样本（去除已经在原始匹配中的部分）
    for (idx1, idx2) in high_conf_matches.keys():
        if (idx1, idx2) not in match_dict and (idx2, idx1) not in match_dict:
            positive_pairs.append((features[idx1], features[idx2]))
            labels.append(1)
    
    # 所有索引
    all_indices = list(features.keys())
    
    # 创建已匹配图像对的集合
    matched_pairs = set()
    for (idx1, idx2) in match_dict.keys():
        matched_pairs.add((min(idx1, idx2), max(idx1, idx2)))
    
    for (idx1, idx2) in high_conf_matches.keys():
        matched_pairs.add((min(idx1, idx2), max(idx1, idx2)))
    
    # 创建负样本（未匹配的图像对）
    for i in range(len(all_indices)):
        for j in range(i + 1, len(all_indices)):
            idx1, idx2 = all_indices[i], all_indices[j]
            if (min(idx1, idx2), max(idx1, idx2)) not in matched_pairs:
                negative_pairs.append((features[idx1], features[idx2]))
                labels.append(0)
                
                # 控制负样本数量，避免数据不平衡
                if len(negative_pairs) >= len(positive_pairs):
                    break
        if len(negative_pairs) >= len(positive_pairs):
            break
    
    # 合并正负样本
    all_pairs = positive_pairs + negative_pairs
    
    return all_pairs, labels

def cluster_images_by_similarity(features, threshold=0.75):
    """
    根据特征向量的相似度将图像分成簇
    
    Args:
        features: 所有图像的特征向量 {img_idx: feature_vector}
        threshold: 相似度阈值
        
    Returns:
        图像簇 {cluster_id: [img_idx, ...]}
    """
    # 将特征向量组织成矩阵
    indices = list(features.keys())
    feature_matrix = torch.stack([features[idx] for idx in indices])
    
    # 归一化特征向量
    feature_matrix = nn.functional.normalize(feature_matrix, p=2, dim=1)
    
    # 计算相似度矩阵 (余弦相似度)
    similarity_matrix = torch.mm(feature_matrix, feature_matrix.t()).numpy()
    
    # 用DBSCAN聚类
    # 将相似度转换为距离 (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # 应用DBSCAN
    db = DBSCAN(eps=1-threshold, min_samples=2, metric='precomputed')
    cluster_labels = db.fit_predict(distance_matrix)
    
    # 组织聚类结果
    clusters = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:  # 忽略噪声点
            clusters[cluster_id].append(indices[i])
    
    return clusters

def visualize_clusters(clusters, features, output_file="clusters_visualization.png"):
    """
    可视化聚类结果
    
    Args:
        clusters: 图像簇 {cluster_id: [img_idx, ...]}
        features: 所有图像的特征向量
        output_file: 输出文件名
    """
    # 将特征向量压缩到2D空间
    from sklearn.decomposition import PCA
    
    # 收集所有特征向量
    all_indices = []
    feature_vectors = []
    for cluster_id, indices in clusters.items():
        for idx in indices:
            all_indices.append(idx)
            feature_vectors.append(features[idx].numpy())
    
    # 如果没有聚类结果，则退出
    if not feature_vectors:
        print("没有找到聚类结果，无法可视化")
        return
    
    # 应用PCA降维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(np.stack(feature_vectors))
    
    # 可视化
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    
    for i, (cluster_id, indices) in enumerate(clusters.items()):
        # 找出该簇中图像的索引位置
        idx_positions = [all_indices.index(idx) for idx in indices]
        
        # 绘制该簇中的点
        plt.scatter(
            reduced_features[idx_positions, 0],
            reduced_features[idx_positions, 1],
            s=100, 
            color=colors[i],
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )
        
        # 添加图像索引标签
        for pos, idx in zip(idx_positions, indices):
            plt.annotate(
                str(idx),
                (reduced_features[pos, 0], reduced_features[pos, 1]),
                fontsize=8
            )
    
    plt.title('Image Clusters Visualization', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"聚类可视化已保存到 {output_file}")

def train_iterative_classifier(match_dict, features, 
                              max_iterations=5, 
                              device='cuda', 
                              initial_threshold=400,
                              confidence_threshold=0.8,
                              batch_size=64,
                              epochs=10):
    """
    迭代训练分类器
    
    Args:
        match_dict: 初始匹配字典
        features: 所有图像的特征向量
        max_iterations: 最大迭代次数
        device: 训练设备
        initial_threshold: 初始匹配阈值
        confidence_threshold: 置信度阈值
        batch_size: 批处理大小
        epochs: 每次迭代的训练轮数
        
    Returns:
        最终的分类器和预测结果
    """
    # 确保设备可用
    device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
    print(f"使用设备: {device}")
    
    # 确定特征维度
    feature_dim = next(iter(features.values())).shape[0]
    
    # 初始化模型
    model = FeatureClassifier(input_dim=feature_dim*2)  # 两个特征向量拼接
    
    # 准备初始训练数据
    print("准备初始训练数据...")
    initial_pairs, initial_labels = prepare_initial_training_data(match_dict, features, match_threshold=initial_threshold)
    
    # 如果没有足够的训练数据，降低阈值重试
    if len(initial_pairs) < 10:
        print(f"初始训练数据不足，降低阈值至 {initial_threshold//2} 重试...")
        initial_pairs, initial_labels = prepare_initial_training_data(
            match_dict, features, match_threshold=initial_threshold//2)
    
    print(f"初始训练数据: {len(initial_pairs)} 对 (正样本: {sum(initial_labels)}, 负样本: {len(initial_labels) - sum(initial_labels)})")
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(initial_pairs))
    indices = list(range(len(initial_pairs)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_pairs = [initial_pairs[i] for i in train_indices]
    train_labels = [initial_labels[i] for i in train_indices]
    
    val_pairs = [initial_pairs[i] for i in val_indices]
    val_labels = [initial_labels[i] for i in val_indices]
    
    # 创建数据加载器
    train_dataset = PairDataset(train_pairs, train_labels)
    val_dataset = PairDataset(val_pairs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始训练
    print("\n开始初始训练...")
    model = train_classifier(model, train_loader, val_loader, device, epochs=epochs)
    
    # 迭代训练
    high_conf_matches = {}
    for iteration in range(max_iterations):
        print(f"\n开始第 {iteration+1}/{max_iterations} 次迭代...")
        
        # 预测所有图像对
        print("预测所有图像对的匹配概率...")
        predictions = predict_pairs(model, features, device, batch_size=batch_size*2)
        
        # 找出高置信度匹配
        print("筛选高置信度匹配...")
        new_high_conf_matches = create_high_conf_matchdict(predictions, threshold=confidence_threshold)
        
        # 检查是否有新的高置信度匹配
        new_matches = set(new_high_conf_matches.keys()) - set(high_conf_matches.keys())
        print(f"新发现的高置信度匹配: {len(new_matches)}")
        
        if not new_matches:
            print("没有新的高置信度匹配，停止迭代")
            break
        
        # 更新高置信度匹配
        high_conf_matches.update(new_high_conf_matches)
        print(f"当前高置信度匹配总数: {len(high_conf_matches)}")
        
        # 准备新的训练数据
        print("准备新的训练数据...")
        new_pairs, new_labels = prepare_new_training_data(match_dict, high_conf_matches, features)
        print(f"新训练数据: {len(new_pairs)} 对 (正样本: {sum(new_labels)}, 负样本: {len(new_labels) - sum(new_labels)})")
        
        # 分割新的训练集和验证集
        train_size = int(0.8 * len(new_pairs))
        indices = list(range(len(new_pairs)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_pairs = [new_pairs[i] for i in train_indices]
        train_labels = [new_labels[i] for i in train_indices]
        
        val_pairs = [new_pairs[i] for i in val_indices]
        val_labels = [new_labels[i] for i in val_indices]
        
        # 创建数据加载器
        train_dataset = PairDataset(train_pairs, train_labels)
        val_dataset = PairDataset(val_pairs, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 重新训练模型
        print("重新训练模型...")
        model = train_classifier(model, train_loader, val_loader, device, epochs=epochs)
    
    # 最终进行一次预测
    final_predictions = predict_pairs(model, features, device, batch_size=batch_size*2)
    
    return model, final_predictions

def self_supervised_clustering(match_dict_path, feature_path, 
                               output_dir='./model_output',
                               device='cuda',
                               clustering_threshold=0.75):
    """
    自监督式图像聚类主函数
    
    Args:
        match_dict_path: 匹配字典路径
        feature_path: 特征向量路径
        output_dir: 输出目录
        device: 训练设备
        clustering_threshold: 聚类相似度阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载匹配字典和特征向量
    print(f"加载匹配字典: {match_dict_path}")
    with open(match_dict_path, 'rb') as f:
        match_dict = pickle.load(f)
    
    print(f"加载特征向量: {feature_path}")
    with open(feature_path, 'rb') as f:
        features_data = pickle.load(f)
    
    # 确保特征向量是PyTorch张量
    features = {}
    for idx, feature in features_data.items():
        if not isinstance(feature, torch.Tensor):
            features[idx] = torch.tensor(feature, dtype=torch.float32)
        else:
            features[idx] = feature.float()
    
    print(f"处理 {len(features)} 个图像的特征向量")
    
    # 训练迭代分类器
    model, predictions = train_iterative_classifier(
        match_dict, 
        features, 
        device=device
    )
    
    # 保存模型和预测结果
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_classifier.pth'))
    
    with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    
    # 进行聚类
    print(f"\n基于特征相似度进行聚类 (阈值: {clustering_threshold})...")
    clusters = cluster_images_by_similarity(features, threshold=clustering_threshold)
    
    print(f"发现 {len(clusters)} 个图像簇:")
    for cluster_id, indices in clusters.items():
        print(f"  簇 {cluster_id}: {len(indices)} 个图像 - {indices}")
    
    # 可视化聚类结果
    visualization_path = os.path.join(output_dir, 'clusters_visualization.png')
    visualize_clusters(clusters, features, output_file=visualization_path)
    
    # 保存聚类结果
    with open(os.path.join(output_dir, 'clusters.pkl'), 'wb') as f:
        pickle.dump(clusters, f)
    
    print(f"分析完成，所有结果已保存到 {output_dir}")
    return clusters

if __name__ == "__main__":
    # 示例用法
    match_dict_path = './match_dict.pkl'
    feature_path = './features.pkl'
    
    clusters = self_supervised_clustering(
        match_dict_path,
        feature_path,
        output_dir='./model_output',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        clustering_threshold=0.75
    )