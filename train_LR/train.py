import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from tqdm import tqdm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_data(csv_path):
    """
    加载CSV文件中的匹配特征数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        pandas DataFrame: 特征数据
    """
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"数据加载完成，共 {len(df)} 个样本")
    
    # 检查数据
    print(f"正样本数量: {df['label'].sum()}")
    print(f"负样本数量: {len(df) - df['label'].sum()}")
    print(f"特征数量: {len(df.columns) - 5}")  # 减去key1, key2, label, scene1, scene2
    
    return df


def preprocess_data(df):
    """
    预处理数据：移除不需要的列，处理缺失值，标准化特征
    
    Args:
        df: 原始数据DataFrame
        
    Returns:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称列表
        scaler: 标准化器
    """
    # 复制数据，避免修改原始数据
    df = df.copy()
    
    # 填补缺失值
    df = df.fillna(0)
    
    # 移除非特征列
    non_feature_cols = ['key1', 'key2', 'label', 'scene1', 'scene2']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # 检查无穷值和NaN值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # 分离特征和标签
    X = df[feature_cols].values
    y = df['label'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler


def analyze_data(df, output_dir=None):
    """
    分析数据并可视化
    
    Args:
        df: 数据DataFrame
        output_dir: 输出目录，用于保存图表
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 类别分布
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', data=df)
    plt.title('类别分布')
    plt.xlabel('标签 (0: 负样本, 1: 正样本)')
    plt.ylabel('样本数量')
    
    # 在柱状图上添加具体数值
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 2. 特征相关性分析
    non_feature_cols = ['key1', 'key2', 'label', 'scene1', 'scene2']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # 计算与标签的相关性
    correlations = df[feature_cols + ['label']].corr()['label'].sort_values(ascending=False)
    
    print("与标签相关性最高的10个特征:")
    print(correlations.head(11))  # 包括标签本身
    
    # 绘制前15个最相关特征的条形图
    plt.figure(figsize=(12, 8))
    correlations_without_label = correlations.drop('label')
    top_features = correlations_without_label.abs().sort_values(ascending=False).head(15).index
    sns.barplot(x=correlations_without_label[top_features].values, y=top_features)
    plt.title('与标签相关性最高的15个特征')
    plt.xlabel('相关系数')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'top_correlations.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 3. 正负样本特征分布对比
    top_5_features = correlations_without_label.abs().sort_values(ascending=False).head(5).index
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_5_features, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=df, x=feature, hue='label', common_norm=False)
        plt.title(f'特征分布: {feature}')
        
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    # plt.show()


def train_lr_model(X, y, feature_names=None, cv=5, output_dir=None):
    """
    训练逻辑回归模型，使用交叉验证选择最佳参数
    
    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称
        cv: 交叉验证折数
        output_dir: 输出目录
        
    Returns:
        best_model: 训练好的最佳模型
        feature_importances: 特征重要性
        thresholds_df: 不同阈值的精确率和召回率表
    """
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 设置网格搜索参数
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced'],
    }
    
    # 网格搜索
    print("执行网格搜索...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型和参数
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"最佳参数: {best_params}")
    print(f"交叉验证最佳F1分数: {grid_search.best_score_:.4f}")
    
    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    
    print("\n测试集评估结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC: {ap:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # ROC曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend(loc='best')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # PR曲线
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_curve, precision_curve, label=f'PR曲线 (AP = {ap:.4f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', label=f'随机 (AP = {sum(y_test)/len(y_test):.4f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('PR 曲线')
    plt.legend(loc='best')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    if feature_names is not None:
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(best_model.coef_[0])
        })
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        print("\n前20个最重要的特征:")
        print(feature_importances.head(20))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 10))
        top_features = feature_importances.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('特征重要性 (Top 20)')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    # 计算阈值表
    print("\n计算不同分类阈值下的性能指标...")
    thresholds = np.arange(0, 1.01, 0.01)
    threshold_metrics = []

    for threshold in tqdm(thresholds, desc="计算阈值表"):
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh, labels=[0, 1]).ravel()
        
        # 计算FAR和FRR
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # 负样本被误判为正样本的比例
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 正样本被误判为负样本的比例
        
        # 计算标准指标
        accuracy = accuracy_score(y_test, y_pred_thresh)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold, 
            'FAR': far,
            'FRR': frr,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })
    
    # 转换为DataFrame
    thresholds_df = pd.DataFrame(threshold_metrics)
    
    # 找出最佳F1阈值
    best_f1_idx = thresholds_df['f1_score'].idxmax()
    best_f1_threshold = thresholds_df.loc[best_f1_idx, 'threshold']
    print(f"最佳F1分数的阈值: {best_f1_threshold:.2f}, "
          f"精确率: {thresholds_df.loc[best_f1_idx, 'precision']:.4f}, "
          f"召回率: {thresholds_df.loc[best_f1_idx, 'recall']:.4f}, "
          f"F1: {thresholds_df.loc[best_f1_idx, 'f1_score']:.4f}")
    
    # 绘制P-R曲线与阈值
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_df['threshold'], thresholds_df['precision'], 'b-', label='精确率')
    plt.plot(thresholds_df['threshold'], thresholds_df['recall'], 'r-', label='召回率')
    plt.plot(thresholds_df['threshold'], thresholds_df['f1_score'], 'g-', label='F1分数')
    plt.axvline(x=best_f1_threshold, color='k', linestyle='--', label=f'最佳F1阈值 = {best_f1_threshold:.2f}')
    plt.xlabel('分类阈值')
    plt.ylabel('分数')
    plt.title('不同阈值下的精确率、召回率和F1分数')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'threshold_curve.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 如果有输出目录，保存阈值表
    if output_dir:
        thresholds_path = os.path.join(output_dir, 'threshold_table.csv')
        thresholds_df.to_csv(thresholds_path, index=False)
        print(f"阈值表已保存至: {thresholds_path}")
    
    return best_model, feature_importances, thresholds_df


def save_model(model, scaler, feature_names, output_dir):
    """
    保存模型和相关组件
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        feature_names: 特征名称
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存模型
    model_path = os.path.join(output_dir, 'lr_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存标准化器
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存至: {scaler_path}")
    
    # 保存特征名称
    feature_names_path = os.path.join(output_dir, 'feature_names.pkl')
    joblib.dump(feature_names, feature_names_path)
    print(f"特征名称已保存至: {feature_names_path}")


def main():
    # 设置路径
    data_dir = './results/featureout/stairs'  # 根据实际路径调整
    csv_path = os.path.join(data_dir, 'matches_features.csv')
    output_dir = os.path.join(data_dir, 'lr_model')
    
    # 加载数据
    df = load_data(csv_path)
    
    # 平衡数据集（可选）
    # 如果类别不平衡严重，可以考虑平衡数据
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]
    print(f"正样本: {len(pos_samples)}, 负样本: {len(neg_samples)}")
    
    # 如果负样本比正样本多很多，可以考虑降采样
    if len(neg_samples) > 3 * len(pos_samples):
        # 负样本降采样，使负样本数量为正样本的2倍
        neg_samples = neg_samples.sample(n=min(len(neg_samples), 2*len(pos_samples)), random_state=42)
        # 合并数据
        df_balanced = pd.concat([pos_samples, neg_samples])
        # 打乱数据
        df_balanced = shuffle(df_balanced, random_state=42)
        print(f"平衡后数据集: {len(df_balanced)} 样本，正样本: {sum(df_balanced['label'])}, 负样本: {len(df_balanced) - sum(df_balanced['label'])}")
        df = df_balanced
    
    # 数据分析
    analyze_data(df, output_dir)
    
    # 预处理数据
    X, y, feature_names, scaler = preprocess_data(df)
    
    # 训练模型并获取阈值表
    model, feature_importances, thresholds_df = train_lr_model(X, y, feature_names, output_dir=output_dir)
    
    # 保存模型
    save_model(model, scaler, feature_names, output_dir)
    
    # 保存特征重要性
    feature_importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    print(f"特征重要性已保存至: {os.path.join(output_dir, 'feature_importances.csv')}")
    
    # 保存最佳阈值信息
    best_threshold = thresholds_df.loc[thresholds_df['f1_score'].idxmax(), 'threshold']
    with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f"最佳F1分数的阈值: {best_threshold:.4f}\n")
        best_metrics = thresholds_df.loc[thresholds_df['f1_score'].idxmax()]
        f.write(f"精确率: {best_metrics['precision']:.4f}\n")
        f.write(f"召回率: {best_metrics['recall']:.4f}\n")
        f.write(f"F1分数: {best_metrics['f1_score']:.4f}\n")
        f.write(f"准确率: {best_metrics['accuracy']:.4f}\n")
    print(f"最佳阈值信息已保存至: {os.path.join(output_dir, 'best_threshold.txt')}")


if __name__ == "__main__":
    main()