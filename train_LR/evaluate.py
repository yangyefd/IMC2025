# 评估给定阈值的LR分类器在各个场景的FAFR性能

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
import glob
from tqdm import tqdm
import logging
import sys
import re
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lr_evaluation.log')
    ]
)

logger = logging.getLogger(__name__)


def extract_scene_name(filename):
    """
    从文件名提取场景名称
    例如：
    - another_et_another_et002.png -> another_et_another_et002
    - another_et_another_165464.png -> another_et_another
    
    Args:
        filename: 文件名
        
    Returns:
        str: 提取的场景名称
    """
    # 去除扩展名
    name = filename.split('.')[0]
    
    # 分割文件名
    parts = name.split('_')
    
    # 如果最后一部分是含有多个数字的字符串（4位以上），则移除
    if len(parts) > 0 and re.match(r'.*\d{4,}', parts[-1]):
        return '_'.join(parts[:-1])
    
    # 否则保留完整的名称（可能包含少量数字）
    return name


def load_model(model_dir):
    """
    加载模型、缩放器和特征名称
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        model: 加载的模型
        scaler: 加载的缩放器
        feature_names: 特征名称
        threshold: 分类阈值
    """
    # 判断是否是高级模型目录
    is_advanced = os.path.exists(os.path.join(model_dir, 'best_single_model.pkl'))
    
    if is_advanced:
        model_path = os.path.join(model_dir, 'best_single_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'selected_features.pkl')
        threshold_path = os.path.join(model_dir, 'ensemble_best_thresholds.txt')
    else:
        model_path = os.path.join(model_dir, 'lr_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        threshold_path = os.path.join(model_dir, 'best_threshold.txt')
    
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model = joblib.load(model_path)
    
    # 加载缩放器
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到缩放器文件: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # 加载特征名称
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"未找到特征名称文件: {feature_names_path}")
    feature_names = joblib.load(feature_names_path)
    
    # 加载阈值
    threshold = 0.5  # 默认阈值
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_content = f.readlines()
        
        for line in threshold_content:
            if '最佳F1分数的阈值' in line or '最优阈值' in line:
                try:
                    threshold = float(line.split(':')[-1].strip())
                    break
                except:
                    pass
    
    logger.info(f"已加载模型: {model_path}")
    logger.info(f"已加载缩放器: {scaler_path}")
    logger.info(f"已加载特征名称: {feature_names_path}")
    logger.info(f"使用分类阈值: {threshold}")
    
    return model, scaler, feature_names, threshold


def find_csv_files(base_dir):
    """
    查找所有matches_features.csv或matches_features_relabeled.csv文件
    
    Args:
        base_dir: 基础目录
        
    Returns:
        list: CSV文件路径列表
    """
    csv_files = []
    
    # 优先查找relabeled文件
    relabeled_files = glob.glob(os.path.join(base_dir, '**/matches_features_relabeled.csv'), recursive=True)
    if relabeled_files:
        csv_files.extend(relabeled_files)
        logger.info(f"找到 {len(relabeled_files)} 个重新标记的CSV文件")
    
    # 如果没有relabeled文件，则查找原始文件
    if not csv_files:
        original_files = glob.glob(os.path.join(base_dir, '**/matches_features.csv'), recursive=True)
        csv_files.extend(original_files)
        logger.info(f"找到 {len(original_files)} 个原始CSV文件")
    
    return csv_files


def evaluate_scene(df, model, scaler, feature_names, threshold, frr_warning_threshold=0.1):
    """
    评估单个场景的FAFR性能
    
    Args:
        df: DataFrame，包含特征和标签
        model: 模型
        scaler: 缩放器
        feature_names: 特征名称
        threshold: 分类阈值
        frr_warning_threshold: FRR警告阈值
        
    Returns:
        dict: 包含FAR/FRR/精确率/召回率等指标的字典
    """
    # 确保所有特征都存在
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        for feature in missing_features:
            df[feature] = 0  # 用0填充缺失的特征
    
    # 准备特征矩阵
    X = df[feature_names].values
    y_true = df['label'].values
    
    # 缩放特征
    X_scaled = scaler.transform(X)
    
    # 预测概率和类别
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # 计算指标
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (fn + tp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 判断是否需要警告
    warning = frr > frr_warning_threshold
    
    return {
        'total_samples': len(df),
        'positive_samples': sum(y_true),
        'negative_samples': len(y_true) - sum(y_true),
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'far': far,
        'frr': frr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'warning': warning
    }


def analyze_scores_by_scene(df, model, scaler, feature_names):
    """
    分析不同场景的分数分布
    
    Args:
        df: DataFrame，包含特征和标签
        model: 模型
        scaler: 缩放器
        feature_names: 特征名称
        
    Returns:
        tuple: (positive_scores, negative_scores) 按场景分组的分数
    """
    # 提取场景信息
    if 'scene1' not in df.columns or 'scene2' not in df.columns:
        df['scene1'] = df['key1'].apply(extract_scene_name)
        df['scene2'] = df['key2'].apply(extract_scene_name)
    
    # 确保所有特征都存在
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        for feature in missing_features:
            df[feature] = 0  # 用0填充缺失的特征
    
    # 准备特征矩阵
    X = df[feature_names].values
    
    # 缩放特征
    X_scaled = scaler.transform(X)
    
    # 预测概率
    df['score'] = model.predict_proba(X_scaled)[:, 1]
    
    # 区分同场景和不同场景匹配
    df['same_scene'] = (df['scene1'] == df['scene2']).astype(int)
    
    # 按场景分组计算分数
    scene_scores = {}
    
    # 处理正样本（同场景）
    positive_data = df[df['same_scene'] == 1]
    positive_scores = {}
    
    for scene, group in positive_data.groupby('scene1'):
        positive_scores[scene] = group['score'].values
    
    # 处理负样本（不同场景）
    negative_data = df[df['same_scene'] == 0]
    negative_scores = {}
    
    for (scene1, scene2), group in negative_data.groupby(['scene1', 'scene2']):
        scene_pair = f"{scene1}-{scene2}"
        negative_scores[scene_pair] = group['score'].values
    
    return positive_scores, negative_scores


def plot_score_distributions(positive_scores, negative_scores, output_dir=None):
    """
    绘制分数分布图
    
    Args:
        positive_scores: 按场景分组的正样本分数
        negative_scores: 按场景分组的负样本分数
        output_dir: 输出目录
    """
    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 合并所有分数
    all_positive = np.concatenate([scores for scores in positive_scores.values()])
    all_negative = np.concatenate([scores for scores in negative_scores.values()])
    
    # 绘制整体分布
    plt.figure(figsize=(12, 8))
    sns.histplot(all_positive, kde=True, stat="density", label="同场景匹配", color="green", alpha=0.6)
    sns.histplot(all_negative, kde=True, stat="density", label="不同场景匹配", color="red", alpha=0.6)
    plt.title("同场景与不同场景匹配分数分布")
    plt.xlabel("分数")
    plt.ylabel("密度")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "overall_distribution.png"), dpi=300, bbox_inches="tight")
    # else:
    #     plt.show()
    
    # 绘制各场景分布
    for scene, scores in positive_scores.items():
        if len(scores) < 10:  # 样本太少，跳过
            continue
            
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, kde=True, stat="density", color="green", alpha=0.8)
        plt.title(f"场景: {scene} 的匹配分数分布")
        plt.xlabel("分数")
        plt.ylabel("密度")
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"scene_{scene}_distribution.png"), dpi=300, bbox_inches="tight")
            plt.close()
    
    # 绘制热力图展示场景间匹配情况
    scene_matrix = defaultdict(lambda: defaultdict(list))
    
    # 收集所有场景
    scenes = set(positive_scores.keys())
    
    # 填充同场景匹配的平均分数
    for scene, scores in positive_scores.items():
        scene_matrix[scene][scene] = np.mean(scores) if len(scores) > 0 else 0
    
    # 填充不同场景匹配的平均分数
    for scene_pair, scores in negative_scores.items():
        if '-' in scene_pair:
            scene1, scene2 = scene_pair.split('-')
            scenes.add(scene1)
            scenes.add(scene2)
            scene_matrix[scene1][scene2] = np.mean(scores) if len(scores) > 0 else 0
    
    # 转换为DataFrame进行可视化
    scenes = sorted(list(scenes))
    matrix_data = np.zeros((len(scenes), len(scenes)))
    
    for i, scene1 in enumerate(scenes):
        for j, scene2 in enumerate(scenes):
            if isinstance(scene_matrix[scene1][scene2], list):
                matrix_data[i, j] = np.mean(scene_matrix[scene1][scene2]) if scene_matrix[scene1][scene2] else 0
            else:
                matrix_data[i, j] = scene_matrix[scene1][scene2]
    
    # 绘制热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix_data, annot=False, fmt=".2f", cmap="coolwarm", 
                xticklabels=scenes, yticklabels=scenes)
    plt.title("场景间匹配平均分数热力图")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "scene_heatmap.png"), dpi=300, bbox_inches="tight")
    # else:
    #     plt.show()


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="评估LR分类器在各个场景的FAFR性能")
    parser.add_argument("--model_dir", type=str, default="./results/combined_model",
                        help="模型目录路径")
    parser.add_argument("--data_dir", type=str, default="./results/featureout",
                        help="特征CSV所在目录")
    parser.add_argument("--threshold", type=float, default=0.29,
                        help="分类阈值，不指定则使用模型默认阈值")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="结果输出目录")
    parser.add_argument("--frr_warning", type=float, default=0.1,
                        help="FRR警告阈值")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 配置文件日志
    if args.output_dir:
        file_handler = logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # 加载模型
    try:
        model, scaler, feature_names, model_threshold = load_model(args.model_dir)
        # 如果指定了阈值，则使用指定的阈值
        threshold = args.threshold if args.threshold is not None else model_threshold
        logger.info(f"使用分类阈值: {threshold}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    
    # 查找特征CSV文件
    csv_files = find_csv_files(args.data_dir)
    if not csv_files:
        logger.error(f"未在{args.data_dir}下找到特征CSV文件")
        return
    
    # 按场景评估
    results = {}
    all_data = pd.DataFrame()
    
    for csv_file in tqdm(csv_files, desc="处理特征文件"):
        try:
            # 加载数据
            df = pd.read_csv(csv_file)
            
            # 如果没有scene列，添加场景信息
            if 'scene1' not in df.columns or 'scene2' not in df.columns:
                df['scene1'] = df['key1'].apply(extract_scene_name)
                df['scene2'] = df['key2'].apply(extract_scene_name)
            
            # 如果没有label列，根据scene生成label
            if 'label' not in df.columns:
                df['label'] = (df['scene1'] == df['scene2']).astype(int)
            
            # 添加到全部数据
            all_data = pd.concat([all_data, df], ignore_index=True)
            
            # 按场景分组评估
            for (scene1, scene2), group in df.groupby(['scene1', 'scene2']):
                # 只评估同场景匹配
                if scene1 != scene2:
                    continue
                    
                scene = scene1
                scene_results = evaluate_scene(
                    group, model, scaler, feature_names, threshold, args.frr_warning
                )
                
                results[scene] = scene_results
                
                # 如果FRR超过警告阈值，记录警告
                if scene_results['warning']:
                    logger.warning(
                        f"场景 {scene} 的FRR = {scene_results['frr']:.4f}, "
                        f"超过警告阈值 {args.frr_warning}"
                    )
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {e}")
    
    # 合并所有场景结果
    all_results = evaluate_scene(all_data, model, scaler, feature_names, threshold)
    
    # 输出总体结果
    logger.info("\n--- 总体评估结果 ---")
    logger.info(f"总样本数: {all_results['total_samples']}")
    logger.info(f"正样本数: {all_results['positive_samples']} ({all_results['positive_samples']/all_results['total_samples']*100:.1f}%)")
    logger.info(f"负样本数: {all_results['negative_samples']} ({all_results['negative_samples']/all_results['total_samples']*100:.1f}%)")
    logger.info(f"FAR: {all_results['far']:.4f}")
    logger.info(f"FRR: {all_results['frr']:.4f}")
    logger.info(f"精确率: {all_results['precision']:.4f}")
    logger.info(f"召回率: {all_results['recall']:.4f}")
    logger.info(f"F1分数: {all_results['f1']:.4f}")
    
    # 输出每个场景的结果
    logger.info("\n--- 各场景评估结果 ---")
    scene_results_list = []
    
    for scene, result in results.items():
        logger.info(f"\n场景: {scene}")
        logger.info(f"  样本数: {result['total_samples']}")
        logger.info(f"  FAR: {result['far']:.4f}")
        logger.info(f"  FRR: {result['frr']:.4f}")
        logger.info(f"  精确率: {result['precision']:.4f}")
        logger.info(f"  召回率: {result['recall']:.4f}")
        logger.info(f"  F1分数: {result['f1']:.4f}")
        
        scene_results_list.append({
            'scene': scene,
            'samples': result['total_samples'],
            'far': result['far'],
            'frr': result['frr'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1']
        })
    
    # 保存结果到CSV
    if args.output_dir:
        scene_results_df = pd.DataFrame(scene_results_list)
        scene_results_df.to_csv(os.path.join(args.output_dir, 'scene_results.csv'), index=False)
        logger.info(f"\n各场景结果已保存到: {os.path.join(args.output_dir, 'scene_results.csv')}")
    
    # 分析分数分布
    logger.info("\n分析分数分布...")
    positive_scores, negative_scores = analyze_scores_by_scene(all_data, model, scaler, feature_names)
    
    # 绘制分数分布图
    plot_score_distributions(positive_scores, negative_scores, args.output_dir)
    logger.info(f"分数分布图已保存到: {args.output_dir}")
    
    # 计算一些额外的统计信息
    scene_stats_df = pd.DataFrame(scene_results_list)
    logger.info("\n--- 场景性能统计 ---")
    logger.info(f"平均FAR: {scene_stats_df['far'].mean():.4f} ± {scene_stats_df['far'].std():.4f}")
    logger.info(f"平均FRR: {scene_stats_df['frr'].mean():.4f} ± {scene_stats_df['frr'].std():.4f}")
    logger.info(f"平均F1分数: {scene_stats_df['f1'].mean():.4f} ± {scene_stats_df['f1'].std():.4f}")
    
    high_frr_scenes = scene_stats_df[scene_stats_df['frr'] > args.frr_warning]
    if not high_frr_scenes.empty:
        logger.warning(f"\n发现 {len(high_frr_scenes)} 个FRR较高的场景:")
        for _, row in high_frr_scenes.iterrows():
            logger.warning(f"  场景: {row['scene']}, FRR: {row['frr']:.4f}")
    
    logger.info("\n评估完成！")


if __name__ == "__main__":
    main()
