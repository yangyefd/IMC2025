"""
CNN训练示例脚本
使用LR分类器生成的CSV数据训练CNN模型
"""

import os
import subprocess
import sys

def run_training():
    """运行CNN训练"""
    
    # 配置参数
    csv_file = "../train_LR/results/combined_model/combined_data.csv"  # LR生成的CSV文件
    image_dir = "../images"  # 图像文件夹，需要根据实际路径调整
    output_dir = "./cnn_model_output"
    
    # 训练参数
    batch_size = 8  # 根据GPU内存调整
    epochs = 50
    learning_rate = 0.001
    target_size = 800
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--csv_file", csv_file,
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--target_size", str(target_size),
        "--use_focal_loss"  # 使用Focal Loss处理类别不平衡
    ]
    
    print("开始训练CNN模型...")
    print(f"命令: {' '.join(cmd)}")
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        print("请先运行LR训练生成CSV数据文件")
        return
    
    # 检查图像目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        print("请设置正确的图像目录路径")
        return
    
    # 运行训练
    try:
        subprocess.run(cmd, check=True)
        print("\n训练完成!")
        print(f"模型和结果保存在: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")


def run_evaluation():
    """运行模型评估"""
    
    # 配置参数
    model_path = "./cnn_model_output/best_model.pth"
    csv_file = "../train_LR/results/combined_model/combined_data.csv"
    image_dir = "../images"
    output_dir = "./evaluation_results"
    threshold = 0.5
    
    # 构建评估命令
    cmd = [
        sys.executable, "inference.py",
        "--model_path", model_path,
        "--csv_file", csv_file,
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--threshold", str(threshold)
    ]
    
    print("开始评估CNN模型...")
    print(f"命令: {' '.join(cmd)}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    # 运行评估
    try:
        subprocess.run(cmd, check=True)
        print("\n评估完成!")
        print(f"评估结果保存在: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"评估失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CNN训练和评估")
    parser.add_argument("action", choices=["train", "eval"], help="执行的动作")
    
    args = parser.parse_args()
    
    if args.action == "train":
        run_training()
    elif args.action == "eval":
        run_evaluation()