import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class ImagePairDataset(Dataset):
    """图像对数据集"""
    
    def __init__(self, csv_file, image_base_dir, target_size=800, transform=None, augment=False):
        """
        Args:
            csv_file: CSV文件路径，包含key1, key2, label列
            image_base_dir: 图像基础目录
            target_size: 目标尺寸
            transform: 变换函数
            augment: 是否进行数据增强
        """
        self.df = pd.read_csv(csv_file)
        self.image_base_dir = image_base_dir
        self.target_size = target_size
        self.transform = transform
        self.augment = augment
        
        # 数据增强变换（适用于灰度图像）
        self.augment_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 移除saturation和hue
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
        ])
        
        print(f"数据集加载完成: {len(self.df)} 个样本")
        print(f"正样本: {self.df['label'].sum()}, 负样本: {len(self.df) - self.df['label'].sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def find_image_path(self, key):
        """根据key查找图像路径"""

        path = os.path.join(self.image_base_dir, key)
        if os.path.exists(path):
            return path
        
        # 在子目录中递归查找
        for root, dirs, files in os.walk(self.image_base_dir):
            for dir in dirs:
                path = os.path.join(root, dir, key)
                if os.path.exists(path):
                    return path
           
        return None
    
    def resize_keep_ratio(self, image, target_size):
        """保持长宽比的resize，并满足倍数限制"""
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(target_size / w, target_size / h)
        
        # 计算新尺寸，确保是32的倍数（适合CNN下采样）
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整到32的倍数
        new_w = ((new_w + 31) // 32) * 32
        new_h = ((new_h + 31) // 32) * 32
        
        # 限制最大尺寸
        new_w = min(new_w, target_size)
        new_h = min(new_h, target_size)
        
        # resize图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 如果需要，创建正方形图像（padding）
        if new_w != target_size or new_h != target_size:
            # 创建黑色背景（灰度图像）
            if len(resized.shape) == 2:  # 灰度图
                padded = np.zeros((target_size, target_size), dtype=np.uint8)
            else:  # 彩色图，但这种情况应该不会发生
                padded = np.zeros((target_size, target_size, resized.shape[2]), dtype=np.uint8)
            
            # 计算居中位置
            start_h = (target_size - new_h) // 2
            start_w = (target_size - new_w) // 2
            
            # 放置图像
            padded[start_h:start_h+new_h, start_w:start_w+new_w] = resized
            return padded
        
        return resized
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key1, key2, label = row['key1'], row['key2'], row['label']
        
        # 查找图像路径
        img1_path = self.find_image_path(key1)
        img2_path = self.find_image_path(key2)
        
        if img1_path is None or img2_path is None:
            # 如果找不到图像，返回黑色图像（灰度）
            img1 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            img2 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        else:
            # 加载图像并转换为灰度
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                img1 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
                img2 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                # 调整尺寸
                img1 = self.resize_keep_ratio(img1, self.target_size)
                img2 = self.resize_keep_ratio(img2, self.target_size)
        
        # 转换为PIL图像用于transforms
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        
        # 数据增强
        if self.augment and self.transform:
            # 对两张图像应用相同的随机变换
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            img1_pil = self.augment_transforms(img1_pil)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            img2_pil = self.augment_transforms(img2_pil)
        
        # 应用基础变换
        if self.transform:
            img1_tensor = self.transform(img1_pil)
            img2_tensor = self.transform(img2_pil)
        else:
            img1_tensor = transforms.ToTensor()(img1_pil)
            img2_tensor = transforms.ToTensor()(img2_pil)
        
        # 拼接成两个通道（2通道灰度图像）
        combined = torch.cat([img1_tensor, img2_tensor], dim=0)  # Shape: [2, H, W]
        
        return combined, torch.tensor(label, dtype=torch.float32)


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


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduce=False)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 收集预测和标签
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 收集预测、概率和标签
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # 计算其他指标
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, precision, recall, f1, auc, all_labels, all_preds, all_probs


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()


def plot_roc_curve(y_true, y_probs, save_path=None):
    """绘制ROC曲线"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #plt.show()
    except Exception as e:
        print(f"无法绘制ROC曲线: {e}")


def save_model(model, optimizer, epoch, loss, save_path):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"模型已保存至: {save_path}")


def load_model(model, optimizer, load_path, device):
    """加载模型"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"模型已从 {load_path} 加载，epoch: {epoch}, loss: {loss}")
    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description='训练CNN图像匹配分类器（灰度图像版本）')
    parser.add_argument('--csv_file', type=str, default='E:\\yey\\work\\IMC2025\\IMC2025\\results\\featureout\\ETs\\matches_features.csv', help='训练数据CSV文件路径')
    parser.add_argument('--image_dir', type=str, default='E:\\yey\\work\\IMC2025\\image-matching-challenge-2025\\train', help='图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./cnn_model_output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--target_size', type=int, default=800, help='目标图像尺寸')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--use_focal_loss', action='store_true', help='使用Focal Loss')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换（灰度图像标准化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图像只需要一个通道的标准化
    ])
    
    # 加载数据集
    print("加载数据集...")
    full_dataset = ImagePairDataset(
        csv_file=args.csv_file,
        image_base_dir=args.image_dir,
        target_size=args.target_size,
        transform=transform,
        augment=True
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建验证集（不使用数据增强）
    val_dataset.dataset.augment = False
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 创建模型（2通道输入用于灰度图像对）
    model = SiameseNet(input_channels=2, num_classes=1).to(device)
    
    # 损失函数和优化器
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_model(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print("开始训练...")
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, y_true, y_pred, y_probs = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')
        print(f'  Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_model(model, optimizer, epoch, val_loss, best_model_path)
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_model(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_model(model, optimizer, args.epochs-1, val_losses[-1], final_model_path)
    
    # 绘制训练历史
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # 绘制ROC曲线
    plot_roc_curve(
        y_true, y_probs,
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型和结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()