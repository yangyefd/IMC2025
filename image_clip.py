import os
from CLIP.clip import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
    
print("模型加载成功")

# 图像文件夹路径
image_folder = "../image-matching-challenge-2025/train/ETs"
print(f"处理文件夹: {image_folder}")

# 获取所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
num_images = len(image_files)
print(f"找到 {num_images} 张图像")

# 如果图像太多，显示警告
if num_images > 100:
    print(f"警告: 文件夹包含 {num_images} 张图像，计算所有两两相似度需要进行 {num_images*(num_images-1)//2} 次比较。")
    response = input("是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消操作")
        exit()

# 提取所有图像的特征
image_features = {}
print("提取图像特征...")
for image_file in tqdm(image_files):
    image_path = os.path.join(image_folder, image_file)
    try:
        # 加载并预处理图像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.encode_image(image)
            
        # 归一化特征
        features = features / features.norm(dim=-1, keepdim=True)
        
        # 存储特征
        image_features[image_file] = features
    except Exception as e:
        print(f"处理 {image_file} 时出错: {e}")

# 计算所有图像对之间的相似度
similarities = {}
print("计算图像相似度...")
for i, img1 in enumerate(tqdm(image_files)):
    for j, img2 in enumerate(image_files):
        # 只计算上三角矩阵，避免重复计算
        if j <= i:
            continue
            
        # 获取特征
        features1 = image_features.get(img1)
        features2 = image_features.get(img2)
        
        if features1 is None or features2 is None:
            continue
            
        # 计算余弦相似度
        similarity = torch.cosine_similarity(features1, features2, dim=-1).item()
        
        # 存储结果
        similarities[(img1, img2)] = similarity

# 创建相似度矩阵
similarity_matrix = np.zeros((num_images, num_images))
for i, img1 in enumerate(image_files):
    for j, img2 in enumerate(image_files):
        if i == j:
            # 对角线上的值（自己与自己比较）设为1
            similarity_matrix[i, j] = 1.0
        elif j > i:
            # 上三角
            similarity_matrix[i, j] = similarities.get((img1, img2), 0.0)
        else:
            # 下三角（对称）
            similarity_matrix[i, j] = similarities.get((img2, img1), 0.0)

# 保存结果到CSV文件
results_dir = "similarity_results"
os.makedirs(results_dir, exist_ok=True)

# 保存相似度矩阵
matrix_file = os.path.join(results_dir, "similarity_matrix.csv")
pd.DataFrame(
    similarity_matrix, 
    index=image_files, 
    columns=image_files
).to_csv(matrix_file)
print(f"相似度矩阵已保存到: {matrix_file}")

# 保存所有配对的相似度
pairs_file = os.path.join(results_dir, "image_pairs_similarity.csv")
with open(pairs_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image1', 'Image2', 'Similarity'])
    for (img1, img2), similarity in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([img1, img2, similarity])
print(f"图像对相似度已保存到: {pairs_file}")

# 可视化相似度矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix, 
    xticklabels=image_files,
    yticklabels=image_files,
    cmap='viridis',
    annot=False
)
plt.title('图像相似度矩阵 (CLIP特征余弦相似度)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'similarity_heatmap.png'), dpi=300)
print(f"热力图已保存到: {os.path.join(results_dir, 'similarity_heatmap.png')}")

# 输出高相似度的图像对
threshold = 0.7
high_similarity_pairs = [(img1, img2, sim) for (img1, img2), sim in similarities.items() if sim > threshold]
high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n相似度高于 {threshold} 的图像对:")
for img1, img2, sim in high_similarity_pairs[:20]:  # 只显示前20对
    print(f"{img1} - {img2}: {sim:.4f}")

print(f"\n发现 {len(high_similarity_pairs)} 对高相似度图像对")