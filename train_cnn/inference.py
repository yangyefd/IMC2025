import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json

class MatchPairDataset(Dataset):
    """匹配对数据集，按需加载"""
    
    def __init__(self, matches_dict, frames, target_size=800):
        self.matches_dict = matches_dict
        self.frames = frames
        self.target_size = target_size
        self.match_keys = list(matches_dict.keys())
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.match_keys)
    
    def __getitem__(self, idx):
        match_key = self.match_keys[idx]
        key1, key2 = match_key.split('-')
        
        # 动态加载图像（不预加载）
        img_path1 = [item for item in self.frames if key1 in item][0]
        img_path2 = [item for item in self.frames if key2 in item][0]
        
        img1 = self.load_and_preprocess_image(img_path1)
        img2 = self.load_and_preprocess_image(img_path2)
        
        # 转换为tensor
        img1_tensor = self.transform(Image.fromarray(img1))
        img2_tensor = self.transform(Image.fromarray(img2))
        
        combined = torch.cat([img1_tensor, img2_tensor], dim=0)
        
        return combined, match_key
    
    def load_and_preprocess_image(self, image_path):
        """加载和预处理图像"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            
            return self.resize_keep_ratio(img, self.target_size)
        except Exception as e:
            print(f"加载图像错误 {image_path}: {e}")
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
    
    def resize_keep_ratio(self, image, target_size):
        """保持比例调整图像尺寸"""
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        result = np.zeros((target_size, target_size), dtype=np.uint8)
        start_h = (target_size - new_h) // 2
        start_w = (target_size - new_w) // 2
        result[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        
        return result
    
class SiameseNet(nn.Module):
    """孪生网络架构用于图像匹配（灰度图像版本）"""
    
    def __init__(self, input_channels=2, num_classes=1):  # 改为2通道
        super(SiameseNet, self).__init__()
        
        # 特征提取backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # 分类
        output = self.classifier(features)
        
        return output

def filter_matches_with_cnn(model_path, matches_dict, frames, threshold=0.5, max_filter_ratio=0.2,
                                     batch_size=16, target_size=800, device=None, num_workers=0):
    """内存优化版本的CNN过滤函数"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    print(f"加载CNN模型: {model_path}")
    
    # 加载模型
    model = SiameseNet(input_channels=2, num_classes=1).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.eval()
    
    # 创建数据集和数据加载器
    dataset = MatchPairDataset(matches_dict, frames, target_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    filtered_matches = {}
    
    # 收集所有匹配对的分数
    all_scores = []
    all_keys = []
    print(f"开始推理，共 {len(dataset)} 个匹配对...")
    
    with torch.no_grad():
        for batch_data, batch_keys in tqdm(dataloader):
            # 移动到设备
            batch_data = batch_data.to(device, non_blocking=True)
            
            # 模型推理
            outputs = model(batch_data).squeeze()
            
            # 处理单个样本的情况
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # 应用阈值过滤
            # predictions = (outputs > threshold).cpu().numpy()
            scores = outputs.cpu().numpy()
            
            # 收集所有分数和keys
            all_scores.extend(scores.tolist())
            all_keys.extend(batch_keys)

            # # 保存通过过滤的匹配对
            # for pred, score, key in zip(predictions, scores, batch_keys):
            #     if pred:
            #         original_data = matches_dict[key]
            #         filtered_matches[key] = [
            #             original_data[0] if len(original_data) > 0 else [],
            #             original_data[1] if len(original_data) > 1 else []
            #         ]
            
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 将分数和keys转换为numpy数组以便处理
    all_scores = np.array(all_scores)
    all_keys = np.array(all_keys)
    
    # 动态过滤策略
    # 1. 首先使用阈值过滤
    threshold_mask = all_scores > threshold
    threshold_pass_count = np.sum(threshold_mask)
    total_count = len(all_scores)
    threshold_pass_rate = threshold_pass_count / total_count
    threshold_fail_rate = 1 - threshold_pass_rate  # 没通过阈值的比例

    print(f"阈值 {threshold} 过滤结果: {threshold_pass_count}/{total_count} ({threshold_pass_rate*100:.1f}%)")
    print(f"没通过阈值的比例: {threshold_fail_rate*100:.1f}%")

    if threshold_fail_rate > max_filter_ratio:  # 如果没通过阈值的匹配对超过20%
        print("没通过阈值的匹配对超过20%，认为分类器可能存在泛化性问题...")
        print("只过滤分数最低的20%...")
        # 去除分数最低的20%
        sorted_indices = np.argsort(all_scores)  # 从低到高排序
        remove_count = int(total_count * max_filter_ratio)  # 要移除的数量
        remove_indices = sorted_indices[:remove_count]  # 最低的20%
        
        final_mask = np.ones(len(all_scores), dtype=bool)
        final_mask[remove_indices] = False
        
        print(f"移除最低分的 {remove_count} 个匹配对")
    else:  # 如果没通过阈值的匹配对不超过20%
        print("没通过阈值的匹配对不超过20%，认为分类器可信...")
        print("将不通过阈值的全部去除...")
        # 使用阈值过滤，保留通过阈值的匹配对
        final_mask = threshold_mask
        
        removed_count = total_count - threshold_pass_count
        print(f"移除未通过阈值的 {removed_count} 个匹配对")
    
    # 根据最终掩码过滤匹配对
    filtered_matches = {}
    
    for i, (score, key) in enumerate(zip(all_scores, all_keys)):
        if final_mask[i]:
            original_data = matches_dict[key]
            filtered_matches[key] = [
                original_data[0] if len(original_data) > 0 else [],
                original_data[1] if len(original_data) > 1 else []
            ]
        else:
            print(f"移除匹配对 {key}，CNN分数: {score:.4f}")

    print(f"过滤完成: {len(filtered_matches)}/{len(matches_dict)} 匹配对通过CNN过滤")
    print(f"保留率: {len(filtered_matches)/len(matches_dict)*100:.1f}%")
    
    return filtered_matches

def main():
    """主函数，用于测试CNN推理"""
    parser = argparse.ArgumentParser(description='CNN模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='CNN模型路径')
    parser.add_argument('--matches_file', type=str, required=True, help='匹配字典pickle文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--output_file', type=str, default='filtered_matches.pkl', help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='过滤阈值')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--target_size', type=int, default=800, help='目标图像尺寸')
    
    args = parser.parse_args()
    
    # 加载匹配字典
    import pickle
    print(f"加载匹配字典: {args.matches_file}")
    with open(args.matches_file, 'rb') as f:
        matches_dict = pickle.load(f)
    
    print(f"原始匹配对数量: {len(matches_dict)}")
    
    # 构建图像路径字典
    frames = {}
    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                key = os.path.splitext(file)[0]
                frames[key] = os.path.join(root, file)
    
    print(f"找到图像文件: {len(frames)} 个")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 执行CNN过滤
    filtered_matches = filter_matches_with_cnn(
        model_path=args.model_path,
        matches_dict=matches_dict,
        frames=frames,
        threshold=args.threshold,
        batch_size=args.batch_size,
        target_size=args.target_size,
        device=device
    )
    
    # 保存结果
    print(f"保存过滤结果: {args.output_file}")
    with open(args.output_file, 'wb') as f:
        pickle.dump(filtered_matches, f)
    
    # 输出统计信息
    print("\n=== 过滤统计 ===")
    print(f"原始匹配对: {len(matches_dict)}")
    print(f"过滤后匹配对: {len(filtered_matches)}")
    print(f"保留率: {len(filtered_matches)/len(matches_dict)*100:.1f}%")
    
    # 分析CNN分数分布
    if filtered_matches:
        cnn_scores = [data['cnn_score'] for data in filtered_matches.values() if 'cnn_score' in data]
        if cnn_scores:
            print(f"CNN分数统计:")
            print(f"  平均分: {np.mean(cnn_scores):.4f}")
            print(f"  最高分: {np.max(cnn_scores):.4f}")
            print(f"  最低分: {np.min(cnn_scores):.4f}")


if __name__ == '__main__':
    main()