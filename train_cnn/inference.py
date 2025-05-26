import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import argparse
import json
from tqdm import tqdm
from train import SiameseNet, ImagePairDataset


def load_trained_model(model_path, device):
    """加载训练好的模型"""
    model = SiameseNet(input_channels=6, num_classes=1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    return model


def predict_batch(model, dataloader, device, threshold=0.5):
    """批量预测"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='预测中'):
            images = images.to(device)
            
            outputs = model(images).squeeze()
            probs = outputs.cpu().numpy()
            preds = (outputs > threshold).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def evaluate_model(model_path, csv_file, image_dir, output_dir, threshold=0.5, batch_size=16):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = load_trained_model(model_path, device)
    
    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImagePairDataset(
        csv_file=csv_file,
        image_base_dir=image_dir,
        target_size=800,
        transform=transform,
        augment=False  # 推理时不使用数据增强
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 预测
    probs, preds, labels = predict_batch(model, dataloader, device, threshold)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    # 打印结果
    print(f"评估结果 (阈值: {threshold}):")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果到CSV
    df = pd.read_csv(csv_file)
    df['predicted_prob'] = probs
    df['predicted_label'] = preds
    df['correct'] = (preds == labels).astype(int)
    
    results_path = os.path.join(output_dir, 'prediction_results.csv')
    df.to_csv(results_path, index=False)
    print(f"预测结果已保存至: {results_path}")
    
    # 保存评估指标
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def filter_matches_with_cnn(model_path, matches_dict, features_data, threshold=0.5, 
                           image_base_dir=None, batch_size=16):
    """
    使用训练好的CNN模型筛选匹配对
    
    Args:
        model_path: 训练好的模型路径
        matches_dict: 匹配字典 {key1-key2: [idxs, scores]}
        features_data: 特征数据字典（可选，主要用于兼容）
        threshold: 分类阈值
        image_base_dir: 图像基础目录
        batch_size: 批次大小
    
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = load_trained_model(model_path, device)
    
    # 准备数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建临时数据集
    match_pairs = []
    for match_key in matches_dict.keys():
        if 'outliers' not in match_key:
            key1, key2 = match_key.split('-')
            match_pairs.append({'key1': key1, 'key2': key2, 'label': 1})  # 临时标签
    
    if not match_pairs:
        return {}
    
    # 创建临时CSV
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df_temp = pd.DataFrame(match_pairs)
    df_temp.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    try:
        # 创建数据集和数据加载器
        dataset = ImagePairDataset(
            csv_file=temp_csv.name,
            image_base_dir=image_base_dir,
            target_size=800,
            transform=transform,
            augment=False
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # 预测
        probs, preds, _ = predict_batch(model, dataloader, device, threshold)
        
        # 筛选匹配对
        filtered_matches_dict = {}
        kept_count = 0
        total_count = 0
        
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            match_key = f"{match_pairs[i]['key1']}-{match_pairs[i]['key2']}"
            if match_key in matches_dict:
                total_count += 1
                if pred == 1:  # 预测为有效匹配
                    filtered_matches_dict[match_key] = matches_dict[match_key]
                    kept_count += 1
        
        print(f"CNN筛选结果: 保留 {kept_count}/{total_count} 个匹配对 ({kept_count/total_count*100:.1f}%)")
        
    finally:
        # 清理临时文件
        os.unlink(temp_csv.name)
    
    return filtered_matches_dict


def main():
    parser = argparse.ArgumentParser(description='CNN模型推理和评估')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--csv_file', type=str, required=True, help='测试数据CSV文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    # 评估模型
    metrics = evaluate_model(
        args.model_path, 
        args.csv_file, 
        args.image_dir, 
        args.output_dir, 
        args.threshold, 
        args.batch_size
    )
    
    print("\n评估完成!")


if __name__ == '__main__':
    main()