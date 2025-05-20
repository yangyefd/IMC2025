# 评估给定阈值的LR分类器在各个场景的FAFR性能

import os
import numpy as np
import pandas as pd
import joblib
import glob
from tqdm import tqdm
import logging
import sys
import re
import argparse
from collections import defaultdict

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
    """从文件名提取场景名称"""
    # 去除扩展名
    name = filename.split('.')[0]
    name = "_".join(name.split('_')[:-1])  # 处理路径分隔符
    
    # 分割文件名
    parts = name.split('_')
    
    # 如果最后一部分是含有多个数字的字符串（4位以上），则移除
    if len(parts) > 0 and re.match(r'.*\d{4,}', parts[-1]):
        return '_'.join(parts[:-1])
    
    # 否则保留完整的名称（可能包含少量数字）
    return name


def load_model(model_dir):
    """加载模型、缩放器和特征名称"""
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
    logger.info(f"默认分类阈值: {threshold}")
    
    return model, scaler, feature_names, threshold


def find_csv_files(base_dir):
    """查找所有matches_features.csv或matches_features_relabeled.csv文件"""
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


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估LR分类器在各个场景的FAFR性能")
    parser.add_argument("--model_dir", type=str, default="./results/combined_model",
                        help="模型目录路径")
    parser.add_argument("--data_dir", type=str, default="./results/featureout",
                        help="特征CSV所在目录")
    parser.add_argument("--threshold", type=float, default=0.9,
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
    
    # 加载所有数据
    all_data = pd.DataFrame()
    for csv_file in tqdm(csv_files, desc="加载数据"):
        try:
            df = pd.read_csv(csv_file)
            # 如果没有scene列，添加场景信息
            if 'scene1' not in df.columns or 'scene2' not in df.columns:
                df['scene1'] = df['key1'].apply(extract_scene_name)
                df['scene2'] = df['key2'].apply(extract_scene_name)
            
            # 如果没有label列，根据scene生成label
            df['label'] = (df['scene1'] == df['scene2']).astype(int)
                
            all_data = pd.concat([all_data, df], ignore_index=True)
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {e}")
    
    # 确保所有特征都存在
    missing_features = [f for f in feature_names if f not in all_data.columns]
    if missing_features:
        for feature in missing_features:
            all_data[feature] = 0  # 用0填充缺失的特征

    # 准备特征矩阵
    X = all_data[feature_names].values
    
    # 缩放特征
    X_scaled = scaler.transform(X)
    
    # 预测概率和类别
    all_data['pred_prob'] = model.predict_proba(X_scaled)[:, 1]
    all_data['pred'] = (all_data['pred_prob'] >= threshold).astype(int)
    
    # 初始化场景统计
    scene_stats = defaultdict(lambda: {'total': 0, 'fa': 0, 'fr': 0})
    
    # 处理同场景匹配（计算FR）
    same_scene = all_data[all_data['scene1'] == all_data['scene2']]
    for _, row in same_scene.iterrows():
        scene = row['scene1']
        scene_stats[scene]['total'] += 1
        # 如果预测错误（假阴性），增加FR计数
        if row['pred'] == 0 and row['label'] == 1:
            scene_stats[scene]['fr'] += 1
    
    # 处理不同场景匹配（计算FA）
    diff_scene = all_data[all_data['scene1'] != all_data['scene2']]
    for _, row in diff_scene.iterrows():
        scene1 = row['scene1']
        scene2 = row['scene2']
        # 同时增加两个场景的比对总数
        scene_stats[scene1]['total'] += 1
        scene_stats[scene2]['total'] += 1
        # 如果预测错误（假阳性），为两个场景都增加FA
        if row['pred'] == 1 and row['label'] == 0:
            scene_stats[scene1]['fa'] += 1
            scene_stats[scene2]['fa'] += 1
    
    # 计算总体统计
    all_total = sum(stat['total'] for stat in scene_stats.values())
    all_fa = sum(stat['fa'] for stat in scene_stats.values())
    all_fr = sum(stat['fr'] for stat in scene_stats.values())
    all_far = all_fa / all_total if all_total > 0 else 0
    all_frr = all_fr / len(same_scene) if len(same_scene) > 0 else 0
    
    # 打印总体结果
    logger.info("\n--- 总体评估结果 ---")
    logger.info(f"总匹配对数: {all_total}")
    logger.info(f"FA数: {all_fa}")
    logger.info(f"FR数: {all_fr}")
    logger.info(f"FAR: {all_far:.6f}")
    logger.info(f"FRR: {all_frr:.6f}")
    
    # 打印各场景结果
    logger.info("\n--- 各场景评估结果 ---")
    logger.info(f"{'场景':<20} {'匹配对总数':<10} {'FA数':<6} {'FR数':<6} {'FAR':<10} {'FRR':<10}")
    
    # 存储超过警告阈值的场景
    warning_scenes = []
    
    # 计算并打印每个场景的统计数据
    for scene, stats in sorted(scene_stats.items()):
        # 计算FAR和FRR
        far = stats['fa'] / stats['total'] if stats['total'] > 0 else 0
        # 计算FRR需要找出该场景的同场景匹配数
        same_scene_count = len(same_scene[same_scene['scene1'] == scene])
        frr = stats['fr'] / same_scene_count if same_scene_count > 0 else 0
        
        logger.info(f"{scene:<20} {stats['total']:<10} {stats['fa']:<6} {stats['fr']:<6} {far:<10.6f} {frr:<10.6f}")
        
        # 检查是否超过警告阈值
        if frr > args.frr_warning:
            warning_scenes.append((scene, frr))
    
    # 打印超过警告阈值的场景
    if warning_scenes:
        logger.info("\n--- FRR超过警告阈值的场景 ---")
        for scene, frr in sorted(warning_scenes, key=lambda x: x[1], reverse=True):
            logger.warning(f"场景: {scene}, FRR: {frr:.6f}, 超过警告阈值 {args.frr_warning}")
    
    # 保存结果到CSV
    if args.output_dir:
        results_list = []
        for scene, stats in scene_stats.items():
            same_scene_count = len(same_scene[same_scene['scene1'] == scene])
            far = stats['fa'] / stats['total'] if stats['total'] > 0 else 0
            frr = stats['fr'] / same_scene_count if same_scene_count > 0 else 0
            
            results_list.append({
                'scene': scene,
                'total_matches': stats['total'],
                'fa_count': stats['fa'],
                'fr_count': stats['fr'], 
                'far': far,
                'frr': frr
            })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(os.path.join(args.output_dir, 'scene_results.csv'), index=False)
        logger.info(f"\n结果已保存到: {os.path.join(args.output_dir, 'scene_results.csv')}")
    
    logger.info("\n评估完成！")


if __name__ == "__main__":
    main()