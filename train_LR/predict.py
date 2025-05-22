def filter_match_with_lr(matches_dict, features_data, model_dir='./results/featureout/stairs/lr_model', threshold=None, output_csv=None):
    """
    使用训练好的LR分类器对匹配对进行筛选
    
    Args:
        matches_dict: 匹配字典 {key1-key2: [idxs, scores]} 
        features_data: 特征数据字典 {key: {'kp': kp, 'desc': desc, ...}}
        model_dir: LR模型及相关文件目录
        threshold: 分类阈值，若为None则使用最佳F1阈值
        output_csv: 输出特征和预测分数的CSV文件路径
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典 {key1-key2: idxs}
    """
    import os
    import numpy as np
    import pandas as pd
    import joblib
    import torch
    from collections import defaultdict
    from train_LR.extract_features import extract_match_features, build_match_graph
    import tempfile
    
    print("使用LR分类器筛选匹配对...")
    
    # 1. 加载模型和相关文件
    try:
        model_path = os.path.join(model_dir, 'lr_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        
        # 如果没有指定阈值，加载最佳阈值
        if threshold is None:
            threshold_path = os.path.join(model_dir, 'best_threshold.txt')
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    first_line = f.readline().strip()
                    threshold = float(first_line.split(':')[-1].strip())
            else:
                threshold = 0.5  # 默认阈值
        
        print(f"已加载LR模型，使用分类阈值: {threshold}")
    except Exception as e:
        print(f"加载LR模型失败: {e}")
        return matches_dict  # 如果模型加载失败，返回原始匹配
    
    # 2. 提取特征
    features_csv_path = output_csv if output_csv else tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    temp_file_created = not output_csv
    
    try:
        # 提取每个匹配对的特征
        df = extract_match_features(matches_dict, features_data, features_csv_path)
        
        # 确保提取的特征与模型使用的特征一致
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            print(f"警告: 缺少以下特征: {missing_features}")
            # 添加缺失特征，填充为0
            for f in missing_features:
                df[f] = 0
        
        # 3. 使用模型进行预测
        X = df[feature_names].values
        
        # 标准化特征
        X_scaled = scaler.transform(X)
        
        nan_rows = np.isnan(X_scaled).any(axis=1)
        nan_row_indices = np.where(nan_rows)[0]
        if len(nan_row_indices) > 0:
            X_scaled[nan_row_indices] = 0

        # 获取预测概率
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # 使用阈值进行分类
        y_pred = (y_prob >= threshold).astype(int)
        
        # 将预测概率和预测结果添加到DataFrame
        df['pred_prob'] = y_prob
        df['prediction'] = y_pred
        
        # 如果指定了输出路径，保存特征和预测结果
        if output_csv:
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 保存合并后的数据到CSV
            df.to_csv(output_csv, index=False)
            print(f"特征数据和预测结果已保存至: {output_csv}")
        
        # 4. 根据预测结果过滤匹配对
        filtered_matches_dict = {}
        total_matches = 0
        kept_matches = 0
        
        for i, (key_pair, prob, pred) in enumerate(zip(df['key1'] + '-' + df['key2'], y_prob, y_pred)):
            if pred == 1:  # 预测为正类（有效匹配）
                if key_pair in matches_dict:
                    filtered_matches_dict[key_pair] = matches_dict[key_pair]
                    kept_matches += len(matches_dict[key_pair][0])
            
            if key_pair in matches_dict:
                total_matches += len(matches_dict[key_pair][0])
        
        # 5. 输出统计信息
        positive_count = sum(y_pred)
        print(f"LR分类结果: 总匹配对数 {len(df)}, 预测为有效的匹配对数 {positive_count} ({positive_count/len(df)*100:.1f}%)")
        print(f"保留匹配点数: {kept_matches}/{total_matches} ({kept_matches/total_matches*100:.1f}%)")
        
    except Exception as e:
        print(f"使用LR模型筛选失败: {e}")
        import traceback
        traceback.print_exc()
        filtered_matches_dict = matches_dict  # 出错时返回原始匹配
    finally:
        # 只删除临时文件，不删除用户指定的输出文件
        if temp_file_created and os.path.exists(features_csv_path):
            os.remove(features_csv_path)
    
    return filtered_matches_dict