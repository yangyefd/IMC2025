# 读取model 和 scaler展示
import pickle
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_info(model_dir, verbose=True):
    """
    加载并显示模型信息
    
    Args:
        model_dir: 模型文件目录路径
        verbose: 是否打印详细信息
        
    Returns:
        model: 加载的模型
        scaler: 加载的缩放器
        feature_names: 特征名称列表
        threshold: 最佳阈值 (如果存在)
    """
    results = {}
    
    # 检查是否是高级模型目录
    is_advanced = os.path.exists(os.path.join(model_dir, 'best_single_model.pkl'))
    
    # 根据目录类型确定文件路径
    if is_advanced:
        model_path = os.path.join(model_dir, 'best_single_model.pkl')
        ensemble_model_path = os.path.join(model_dir, 'ensemble_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'selected_features.pkl')
        threshold_path = os.path.join(model_dir, 'ensemble_best_thresholds.txt')
    else:
        model_path = os.path.join(model_dir, 'lr_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    # 加载模型
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        results['model'] = model
        if verbose:
            print(f"\n{'='*60}\n加载模型: {os.path.basename(model_path)}")
            print(f"模型类型: {type(model).__name__}")
            
           # 显示模型初始化参数
            print("\n模型初始化参数:")
            model_params = {
                'penalty': getattr(model, 'penalty', 'none'),
                'C': getattr(model, 'C', 'unknown'),
                'solver': getattr(model, 'solver', 'unknown'),
                'max_iter': getattr(model, 'max_iter', 'unknown'),
                'class_weight': getattr(model, 'class_weight', None),
                'random_state': getattr(model, 'random_state', None),
                'fit_intercept': getattr(model, 'fit_intercept', True)
            }
            
            for param, value in model_params.items():
                print(f"  {param}: {value}")
            
            # 显示训练后属性
            print("\n训练后属性:")
            if hasattr(model, 'coef_'):
                print(f"  coef_: {model.coef_.tolist()}")
                print(f"  intercept_: {model.intercept_}")
            
            if hasattr(model, 'classes_'):
                print(f"  classes_: {model.classes_}")
            
            if hasattr(model, 'n_features_in_'):
                print(f"  n_features_in_: {model.n_features_in_}")

            
    else:
        print(f"警告: 未找到模型文件 {model_path}")
        results['model'] = None
    
    # 加载缩放器
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        results['scaler'] = scaler
        if verbose:
            print(f"\n{'='*60}\n加载缩放器: {os.path.basename(scaler_path)}")
            print(f"缩放器类型: {type(scaler).__name__}")
            if hasattr(scaler, 'mean_'):
                print(f"均值形状: {scaler.mean_.shape}")
                print(f"均值: {scaler.mean_.tolist()}")
            if hasattr(scaler, 'scale_'):
                print(f"缩放系数形状: {scaler.scale_.shape}")
                print(f"缩放系数: {scaler.scale_.tolist()}")
    else:
        print(f"警告: 未找到缩放器文件 {scaler_path}")
        results['scaler'] = None
    
    print("scaler.mean_:", scaler.mean_.sum())
    print("scaler.scale_:", scaler.scale_.sum())
    print("logreg.coef_:", model.coef_.sum())

    return results

if __name__ == "__main__":
    # 模型目录路径
    model_dir = './results/combined_model/'  # 替换为实际路径
    # 加载模型信息
    results = load_model_info(model_dir)