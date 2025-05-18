'''
frames: list of frames path, can use to get the image
matches_dict: list of matches {key1-key2: [idxs, idx1, idx2, matchscores]}
features_data: list of features data, kp, descriptor...
'''
import numpy as np
import cv2
from collections import defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN
import copy

def visualize_filtered_matches(frames, original_matches_dict, filtered_matches_dict, features_data, output_dir=None):
    """
    可视化匹配过滤效果，将保留的匹配对用绿色表示，过滤掉的匹配对用红色表示。
    
    Args:
        frames: 图像路径列表
        original_matches_dict: 原始匹配字典 {key1-key2: matches}
        filtered_matches_dict: 过滤后的匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        output_dir: 输出目录，如果为None则直接显示
        
    Returns:
        None
    """
    import cv2
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每对匹配
    for key_pair in original_matches_dict.keys():
        if key_pair not in original_matches_dict:
            continue
            
        key1, key2 = key_pair.split('-')
        
        # 找到对应的图像路径
        img1_path = None
        img2_path = None
        for frame in frames:
            if key1 in frame:
                img1_path = frame
            if key2 in frame:
                img2_path = frame
        
        if img1_path is None or img2_path is None:
            print(f"无法找到 {key_pair} 对应的图像文件")
            continue
        
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"无法读取图像: {img1_path} 或 {img2_path}")
            continue
            
        # 获取原始匹配
        original_matches = original_matches_dict[key_pair][0]
        
        # 获取过滤后的匹配
        filtered_matches = []
        if key_pair in filtered_matches_dict:
            filtered_matches = filtered_matches_dict[key_pair]
        
        # 将过滤后的匹配转换为集合，便于判断是否保留
        filtered_matches_set = set()
        for m in filtered_matches:
            filtered_matches_set.add((int(m[0]), int(m[1])))
        
        # 获取特征点
        kp1 = features_data[key1]['kp']
        kp2 = features_data[key2]['kp']
        
        # 准备绘图
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建拼接图像
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        
        # 计算保留和过滤的匹配对数量
        kept_count = 0
        removed_count = 0
        
        # 绘制匹配线
        for m in original_matches:
            idx1, idx2 = int(m[0]), int(m[1])
            
            # 判断是否保留
            is_kept = (idx1, idx2) in filtered_matches_set
            
            # 更新计数
            if is_kept:
                kept_count += 1
                color = (0, 255, 0)  # 绿色表示保留的匹配
            else:
                removed_count += 1
                color = (0, 0, 255)  # 红色表示过滤掉的匹配
            
            # 获取坐标点
            pt1 = (int(kp1[idx1][0]), int(kp1[idx1][1]))
            pt2 = (int(kp2[idx2][0]) + w1, int(kp2[idx2][1]))
            
            # 绘制线和点
            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
        
        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"保留的匹配: {kept_count}, 过滤的匹配: {removed_count}"
        cv2.putText(vis, text, (10, 30), font, 1, (255, 255, 255), 2)
        
        # 输出结果
        output_path = os.path.join(output_dir, f"{key_pair}_filtered.jpg")
        cv2.imwrite(output_path, vis)
        print(f"保存可视化结果至: {output_path}")

def visualize_connections(key_center, filtered_matches_dict, features_data, frames, output_dir=None):
    """
    可视化以指定图像为中心的匹配连接关系
    
    Args:
        key_center: 中心图像的关键字
        filtered_matches_dict: 过滤后的匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        frames: 图像路径列表
        output_dir: 输出目录，默认为None时创建在当前目录
        
    Returns:
        None (保存或显示图像)
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import os
    from PIL import Image
    import cv2
    
    if output_dir is None:
        output_dir = "connection_viz"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建连接图
    G = nx.Graph()
    
    # 添加节点
    connected_keys = set()
    connected_keys.add(key_center)
    
    # 寻找与中心图像相连的所有图像
    for key_pair, matches in filtered_matches_dict.items():
        key1, key2 = key_pair.split('-')
        
        if key1 == key_center:
            connected_keys.add(key2)
            match_count = len(matches)
            G.add_edge(key1, key2, weight=match_count)
        elif key2 == key_center:
            connected_keys.add(key1)
            match_count = len(matches)
            G.add_edge(key2, key1, weight=match_count)
    
    # 如果没有连接，则返回
    if len(connected_keys) <= 1:
        print(f"图像 {key_center} 没有匹配关系")
        return
    
    # 找出所有图像之间的连接关系
    for key_pair, matches in filtered_matches_dict.items():
        key1, key2 = key_pair.split('-')
        if key1 in connected_keys and key2 in connected_keys:
            match_count = len(matches)
            G.add_edge(key1, key2, weight=match_count)
    
    # 为图像节点添加位置和缩略图
    node_images = {}
    positions = {}
    
    # 获取中心图像并调整大小
    center_img_path = None
    for frame in frames:
        if key_center in frame:
            center_img_path = frame
            break
    
    if center_img_path:
        center_img = Image.open(center_img_path)
        # 确保中心图像不会太大
        center_img.thumbnail((100, 100))
        node_images[key_center] = np.array(center_img)
        positions[key_center] = (0, 0)  # 中心位置
    
    # 计算其他节点的位置和图像
    num_connected = len(connected_keys) - 1
    radius = 300  # 圆的半径
    angle_step = 2 * np.pi / num_connected
    
    i = 0
    for key in connected_keys:
        if key == key_center:
            continue
        
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[key] = (x, y)
        
        # 获取图像路径
        img_path = None
        for frame in frames:
            if key in frame:
                img_path = frame
                break
        
        if img_path:
            img = Image.open(img_path)
            img.thumbnail((80, 80))  # 连接节点图像稍小
            node_images[key] = np.array(img)
        
        i += 1
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制边（带宽度）
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        max_weight = max([d['weight'] for _, _, d in G.edges(data=True)])
        width = 1 + 10 * weight / max_weight  # 根据权重归一化线宽
        
        # 连接图
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        
        # 绘制连接线，宽度表示匹配点数量
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=width)
        
        # 在连接线上显示匹配点数量
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        plt.text(mid_x, mid_y, str(weight), fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 绘制节点与图像
    for key, (x, y) in positions.items():
        # 在节点位置显示图像
        if key in node_images:
            img = node_images[key]
            img_height, img_width = img.shape[:2]
            
            # 创建图像显示区域
            image_box = plt.Rectangle((x - img_width/2, y - img_height/2), 
                                      img_width, img_height, 
                                      fill=False, edgecolor='gray')
            plt.gca().add_patch(image_box)
            
            # 显示图像
            plt.imshow(img, extent=(x - img_width/2, x + img_width/2, 
                                    y - img_height/2, y + img_height/2))
        
        # 图像下方显示图像名称
        plt.text(x, y - 50, key, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # 如果是中心节点，添加额外标记
        if key == key_center:
            circle = plt.Circle((x, y), 110, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(circle)
    
    # 设置坐标轴
    plt.xlim(min([p[0] for p in positions.values()]) - 200, 
             max([p[0] for p in positions.values()]) + 200)
    plt.ylim(min([p[1] for p in positions.values()]) - 200, 
             max([p[1] for p in positions.values()]) + 200)
    
    # 设置标题
    plt.title(f'图像 {key_center} 的匹配连接关系')
    plt.axis('off')  # 隐藏坐标轴
    
    # 添加图例
    plt.figtext(0.02, 0.02, "连接线宽度表示匹配点数量", fontsize=10)
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{key_center}_connections.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"连接关系图已保存至: {output_path}")
    
    # 创建一个饼图展示匹配点数量分布
    match_counts = {}
    total_matches = 0
    
    for u, v, data in G.edges(data=True):
        if u == key_center:
            match_counts[v] = data['weight']
            total_matches += data['weight']
        elif v == key_center:
            match_counts[u] = data['weight']
            total_matches += data['weight']
    
    if match_counts:
        plt.figure(figsize=(10, 6))
        labels = [f"{key} ({count})" for key, count in match_counts.items()]
        sizes = [count for count in match_counts.values()]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'{key_center} 匹配点分布 (总计: {total_matches})')
        
        # 保存饼图
        pie_path = os.path.join(output_dir, f"{key_center}_match_distribution.png")
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        print(f"匹配点分布图已保存至: {pie_path}")
    
    # 为连接的图像对创建匹配可视化缩略图
    for key in connected_keys:
        if key == key_center:
            continue
        
        key_pair = f"{key_center}-{key}"
        reverse_key_pair = f"{key}-{key_center}"
        
        if key_pair in filtered_matches_dict:
            matches = filtered_matches_dict[key_pair]
        elif reverse_key_pair in filtered_matches_dict:
            matches = filtered_matches_dict[reverse_key_pair]
        else:
            continue
        
        # 找到对应的图像路径
        img1_path = None
        img2_path = None
        
        for frame in frames:
            if key_center in frame:
                img1_path = frame
            if key in frame:
                img2_path = frame
        
        if img1_path is None or img2_path is None:
            continue
        
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            continue
        
        # 调整图像大小以便并排显示
        max_height = 400
        scale1 = max_height / img1.shape[0]
        scale2 = max_height / img2.shape[0]
        
        img1_resized = cv2.resize(img1, (int(img1.shape[1] * scale1), max_height))
        img2_resized = cv2.resize(img2, (int(img2.shape[1] * scale2), max_height))
        
        # 获取特征点
        kp1 = features_data[key_center]['kp']
        kp2 = features_data[key]['kp']
        
        # 创建拼接图像
        h1, w1 = img1_resized.shape[:2]
        h2, w2 = img2_resized.shape[:2]
        
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1_resized
        vis[:h2, w1:w1+w2] = img2_resized
        
        # 绘制匹配线 (最多显示30条，避免图像过于混乱)
        display_count = min(len(matches), 30)
        for i in range(display_count):
            idx1, idx2 = int(matches[i][0]), int(matches[i][1])
            
            # 获取调整后的坐标点
            pt1 = (int(kp1[idx1][0] * scale1), int(kp1[idx1][1] * scale1))
            pt2 = (int(kp2[idx2][0] * scale2) + w1, int(kp2[idx2][1] * scale2))
            
            # 绘制线和点
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 255, 0), -1)
        
        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f"{key_center}-{key}: {len(matches)} 匹配点", 
                   (10, 30), font, 1, (255, 255, 255), 2)
        
        # 保存匹配可视化
        match_viz_path = os.path.join(output_dir, f"{key_center}_{key}_match.jpg")
        cv2.imwrite(match_viz_path, vis)
        print(f"匹配可视化已保存至: {match_viz_path}")
    
    plt.close('all')
    return G  # 返回图对象，便于进一步分析

def visualize_cycle_consistency(cycle_name, frames, matches_dict, features_data, output_dir=None):
    """
    可视化三元组中的循环一致性检查。
    
    Args:
        cycle_name: 三元组名称，格式为 "img1-img2-img3"
        frames: 图像路径列表
        matches_dict: 匹配字典
        features_data: 特征数据字典
        output_dir: 输出目录
    """
    import cv2
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    # 解析三元组中的图像名称
    keys = cycle_name.split('-')
    if len(keys) != 3:
        raise ValueError(f"Invalid cycle name: {cycle_name}")
    
    # 获取图像路径
    img_paths = {}
    for key in keys:
        for frame in frames:
            if key in frame:
                img_paths[key] = frame
                break
    
    if len(img_paths) != 3:
        raise ValueError("Cannot find all images in frames")
    
    # 读取图像
    images = {k: cv2.imread(str(p)) for k, p in img_paths.items()}
    
    # 创建一个大画布，3x2布局
    fig = plt.figure(figsize=(20, 10))
    
    # 第一行：显示三张原始图像
    for i, key in enumerate(keys):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB))
        plt.title(key)
        plt.axis('off')
    
    # 第二行：显示三对匹配关系
    pairs = [
        (keys[0], keys[1]),
        (keys[1], keys[2]),
        (keys[2], keys[0])
    ]
    
    # 收集所有匹配以便统计
    cycle_matches = []
    matched_points = {k: set() for k in keys}
    
    # 为每对图像创建匹配可视化
    for i, (key1, key2) in enumerate(pairs):
        pair_key = f"{key1}-{key2}"
        reverse_key = f"{key2}-{key1}"
        
        if pair_key in matches_dict:
            matches = matches_dict[pair_key]
            is_reverse = False
        elif reverse_key in matches_dict:
            matches = matches_dict[reverse_key]
            is_reverse = True
        else:
            continue
            
        # 获取特征点
        kp1 = features_data[key1]['kp']
        kp2 = features_data[key2]['kp']
        
        # 收集匹配信息
        cycle_matches.append({
            'key1': key1,
            'key2': key2,
            'matches': matches,
            'is_reverse': is_reverse
        })
        
        # 收集匹配点
        for m in matches:
            if is_reverse:
                matched_points[key1].add(int(m[1]))
                matched_points[key2].add(int(m[0]))
            else:
                matched_points[key1].add(int(m[0]))
                matched_points[key2].add(int(m[1]))
        
        # 创建拼接图像
        h1, w1 = images[key1].shape[:2]
        h2, w2 = images[key2].shape[:2]
        vis = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = images[key1]
        vis[:h2, w1:w1+w2] = images[key2]
        
        plt.subplot(2, 3, i+4)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"{key1}-{key2}: {len(matches)} matches")
        
        # 绘制匹配线（最多显示50条，避免过于混乱）
        display_matches = matches[:50] if len(matches) > 50 else matches
        for m in display_matches:
            if is_reverse:
                pt1 = tuple(map(int, kp2[m[0]]))
                pt2 = tuple(map(int, kp1[m[1]]))
            else:
                pt1 = tuple(map(int, kp1[m[0]]))
                pt2 = tuple(map(int, kp2[m[1]]))
            
            pt2 = (pt2[0] + w1, pt2[1])  # 调整第二张图中点的x坐标
            
            # 绘制匹配线和点
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=0.5, alpha=0.5)
            plt.plot(pt1[0], pt1[1], 'r.', markersize=3)
            plt.plot(pt2[0], pt2[1], 'r.', markersize=3)
        
        plt.axis('off')
    
    # 添加一些统计信息
    info_text = f"Cycle: {cycle_name}\n"
    for pair in cycle_matches:
        info_text += f"{pair['key1']}-{pair['key2']}: {len(pair['matches'])} matches\n"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10)
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"cycle_{cycle_name.replace('.png','')}.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_cycle_consistency_with_tracks(cycle_name, frames, matches_dict, features_data, output_dir=None, top_n=10):
    """
    可视化三元组中距离最大的 top_n 个点的追踪路径。
    """
    import cv2
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # 解析三元组中的图像名称
    keys = cycle_name.split('-')
    if len(keys) != 3:
        raise ValueError(f"Invalid cycle name: {cycle_name}")
    
    # 获取图像路径和读取图像
    img_paths = {}
    for key in keys:
        for frame in frames:
            if key in frame:
                img_paths[key] = frame
                break
    images = {k: cv2.imread(str(p)) for k, p in img_paths.items()}
    
    # 构建匹配链
    pairs = [(keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])]
    match_maps = []
    pair_keys = []
    is_reverse_lst = []
    for key1, key2 in pairs:
        pair_key = f"{key1}-{key2}"
        reverse_key = f"{key2}-{key1}"
        
        if pair_key in matches_dict:
            matches = matches_dict[pair_key]
            is_reverse = False
            pair_keys.append(pair_key)
        else:
            matches = matches_dict[reverse_key]
            is_reverse = True
            pair_keys.append(reverse_key)
        is_reverse_lst.append(is_reverse)
        # 创建正向或反向的匹配映射
        if is_reverse:
            match_map = {int(m[1]): int(m[0]) for m in matches}
        else:
            match_map = {int(m[0]): int(m[1]) for m in matches}
        match_maps.append(match_map)
    
    # 计算循环误差
    matches0 = np.array(matches_dict[pair_keys[0]], dtype=int)
    if is_reverse_lst[0]:
        idx1_arr = matches0[:, 1]
        idx2_arr = matches0[:, 0]
    else:
        idx1_arr = matches0[:, 0]
        idx2_arr = matches0[:, 1]
        
    # 计算完整循环
    idx3_arr = np.array([match_maps[1].get(idx2, -1) for idx2 in idx2_arr])
    valid_1 = idx3_arr != -1
    idx1_cycle_arr = np.array([match_maps[2].get(idx3, -1) if valid else -1
                              for idx3, valid in zip(idx3_arr, valid_1)])
    valid_2 = idx1_cycle_arr != -1
    valid = valid_1 & valid_2
    
    # 计算误差距离
    kp1 = np.array(features_data[keys[0]]['kp'])
    pt1 = kp1[idx1_arr[valid]]
    pt1_cycle = kp1[idx1_cycle_arr[valid]]
    distances = np.linalg.norm(pt1 - pt1_cycle, axis=1)
    
    # 获取距离最大的 top_n 个点的索引
    top_indices = np.argsort(distances)[-top_n:][::-1]
    
    # 创建图像
    plt.figure(figsize=(20, 8))
    
    # 计算图像拼接
    max_h = max(img.shape[0] for img in images.values())
    total_w = sum(img.shape[1] for img in images.values())
    canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # 拼接图像
    current_w = 0
    x_offsets = {}
    for k, img in images.items():
        h, w = img.shape[:2]
        canvas[:h, current_w:current_w+w] = img
        x_offsets[k] = current_w
        current_w += w
    
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
    # 为top_n个点分配不同颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, top_n))
    
    # 绘制追踪路径
    for color_idx, idx in enumerate(top_indices):
        # 获取原始点索引
        idx1 = idx1_arr[valid][idx]
        idx2 = idx2_arr[valid][idx]
        idx3 = idx3_arr[valid][idx]
        idx1_cycle = idx1_cycle_arr[valid][idx]
        
        # 获取所有点的坐标
        pt1 = kp1[idx1]
        pt2 = np.array(features_data[keys[1]]['kp'][idx2])
        pt3 = np.array(features_data[keys[2]]['kp'][idx3])
        pt1_c = kp1[idx1_cycle]
        
        # 调整x坐标以适应拼接图像
        pts = np.array([
            [pt1[0] + x_offsets[keys[0]], pt1[1]],
            [pt2[0] + x_offsets[keys[1]], pt2[1]],
            [pt3[0] + x_offsets[keys[2]], pt3[1]],
            [pt1_c[0] + x_offsets[keys[0]], pt1_c[1]]
        ])
        
        # 绘制路径
        plt.plot(pts[:, 0], pts[:, 1], '-', color=colors[color_idx], 
                linewidth=2, label=f'Error: {distances[idx]:.1f}px')
        plt.plot(pts[:, 0], pts[:, 1], 'o', color=colors[color_idx], 
                markersize=8)
        
        # 添加编号标签
        plt.text(pts[0, 0], pts[0, 1]-10, f'{color_idx+1}', 
                color=colors[color_idx], fontsize=12, ha='center')
    
    plt.title(f'Top {top_n} Largest Cycle Errors')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    
    # 保存或显示结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"cycle_{cycle_name}_tracks.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def filter_matches_graph(frames, matches_dict, features_data_t, threshold=0.6, distance_threshold=10.0, inlier_ratio_threshold=0.8, verbose=True, output_csv=None):
    """
    基于图结构的匹配过滤函数，去除不一致的匹配及靠近不一致匹配的点对。
    
    Args:
        frames: 图像路径列表
        matches_dict: 匹配字典 {key1-key2: matches} idxs:matches[0] match_scores:matches[1]
        features_data: 特征数据字典
        threshold: 循环一致性过滤阈值
        distance_threshold: 空间邻近过滤的距离阈值（像素）
        verbose: 是否输出详细统计信息
        output_csv: 保存循环误差统计的CSV文件路径
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
    """
    import numpy as np
    import pandas as pd
    import os
    import csv
    import copy
    from collections import defaultdict
    import networkx as nx

    features_data_t = copy.deepcopy(features_data_t)
    features_data = {k: {"kp":v['kp'].cpu().numpy()} for k, v in features_data_t.items()}
    
    # 记录每个阶段的匹配数量
    stats = defaultdict(dict)
        
    # 新字典去除matches_dict中的分数
    filtered_matches_dict = {}
    filtered_scores_dict = {}
    for key_pair, matches in matches_dict.items():
        if len(matches) > 0:
            filtered_matches_dict[key_pair] = matches[0]
            filtered_scores_dict[key_pair] = matches[1]
            # 记录初始匹配数
            stats["初始"][key_pair] = len(matches[0])
    
    # # 过滤匹配
    # filtered_matches_dict = filter_matches_comprehensive(
    #     filtered_matches_dict,
    #     features_data,
    #     filtered_scores_dict,
    #     min_matches=15
    # )

    stats = defaultdict(dict)
    for key_pair, matches in filtered_matches_dict.items():
        if len(matches) > 0:
            # 记录初始匹配数
            stats["初始"][key_pair] = len(matches[0])
    

    # return filtered_matches_dict, None
    # 构建图像之间的连接图
    G = nx.Graph()
    match_stats = defaultdict(int)
    
    # 1. 创建图结构并统计初始匹配情况
    for key_pair, matches in filtered_matches_dict.items():
        key1, key2 = key_pair.split('-')
        if len(matches) > 0:
            G.add_edge(key1, key2, weight=len(matches))
            match_stats[(key1, key2)] = len(matches)
    
    # 2. 使用循环一致性直接过滤不一致的匹配，并去除邻近点
    # 为每个匹配对创建一个掩码，初始所有点都保留
    match_masks = {}
    for key_pair in filtered_matches_dict:
        match_masks[key_pair] = np.ones(len(filtered_matches_dict[key_pair]), dtype=bool)
    
    cycle_filtered_count = 0
    cycle_total_checked = 0
    proximity_filtered_count = 0  # 新增：统计因邻近过滤移除的匹配数
    
    # 创建用于记录循环误差的数据结构
    cycle_error_data = []
    
    # 查找所有长度为3的循环
    for cycle in nx.cycle_basis(G):
        if len(cycle) == 3:
            # 获取三对匹配关系
            keys = cycle
            pairs = [
                (keys[0], keys[1]),
                (keys[1], keys[2]),
                (keys[2], keys[0])
            ]
            
            # 三元组名称
            cycle_name = f"{keys[0]}-{keys[1]}-{keys[2]}"
            
            # 检查所有匹配对是否存在
            valid_cycle = True
            for key1, key2 in pairs:
                pair_key = f"{key1}-{key2}"
                reverse_key = f"{key2}-{key1}"
                if pair_key not in filtered_matches_dict and reverse_key not in filtered_matches_dict:
                    valid_cycle = False
                    break
            
            if not valid_cycle:
                continue
            
            # 构建匹配索引查找表
            match_maps = []
            pair_keys = []
            is_reverse_lst = []
            
            # 记录每对之间的匹配数量
            pair_match_counts = []
            
            for key1, key2 in pairs:
                pair_key = f"{key1}-{key2}"
                reverse_key = f"{key2}-{key1}"
                
                if pair_key in filtered_matches_dict:
                    matches = filtered_matches_dict[pair_key]
                    is_reverse = False
                    pair_keys.append(pair_key)
                else:
                    matches = filtered_matches_dict[reverse_key]
                    is_reverse = True
                    pair_keys.append(reverse_key)
                
                is_reverse_lst.append(is_reverse)
                pair_match_counts.append(len(matches))
                
                # 创建从idx1到idx2的映射以及索引映射（用于掩码）
                if is_reverse:
                    match_map = {int(m[1]): int(m[0]) for m in matches}
                else:
                    match_map = {int(m[0]): int(m[1]) for m in matches}
                
                match_maps.append(match_map)
            
            # 为每个匹配对准备检查记录
            checked_indices = [set(), set(), set()]
            consistent_indices = [set(), set(), set()]
            inconsistent_points = [defaultdict(list), defaultdict(list), defaultdict(list)]  # 存储不一致点的坐标
            
            # === Step 1: 批量读取第一对匹配 ===
            matches0 = np.array(filtered_matches_dict[pair_keys[0]], dtype=int)  # shape: (N, 2)
            if is_reverse_lst[0]:
                idx1_arr = matches0[:, 1]
                idx2_arr = matches0[:, 0]
            else:
                idx1_arr = matches0[:, 0]
                idx2_arr = matches0[:, 1]

            # === Step 2: 映射 idx2 -> idx3 和 idx3 -> idx1_cycle ===
            idx3_arr = np.array([match_maps[1].get(idx2, -1) for idx2 in idx2_arr])
            valid_1 = idx3_arr != -1

            idx1_cycle_arr = np.array([match_maps[2].get(idx3, -1) if valid else -1
                                    for idx3, valid in zip(idx3_arr, valid_1)])
            valid_2 = idx1_cycle_arr != -1

            valid = valid_1 & valid_2
            valid_idx = np.nonzero(valid)[0]

            # === Step 3: 加载关键点坐标 ===
            kp1 = np.array(features_data[keys[0]]['kp'])  # 第1张图像的关键点坐标
            pt1 = kp1[idx1_arr[valid]]
            pt1_cycle = kp1[idx1_cycle_arr[valid]]

            # === Step 4: 距离检查（是否一致）并记录循环误差 ===
            same_index = idx1_arr[valid] == idx1_cycle_arr[valid]
            distances = np.linalg.norm(pt1 - pt1_cycle, axis=1)
            close_enough = distances < 5
            is_consistent = same_index | close_enough
            
            # 计算循环误差均值
            if len(distances) > 0:
                mean_cycle_error = np.mean(distances)
                #another_et_another_et005.png-another_et_another_et004.png-another_et_another_et001.png
                # 如果循环误差过大，移除该三元组中匹配数量最少的匹配对

                # # 在filter_matches_graph函数中添加调用
                # # 在处理循环时添加：
                # if 'another_et_another_et005.png-another_et_another_et004.png-another_et_another_et001.png' in cycle_name:
                #     visualize_cycle_consistency(cycle_name, frames, filtered_matches_dict, features_data, 
                #                             output_dir='cycle_visualization')
                    
                #     # 添加轨迹可视化
                #     visualize_cycle_consistency_with_tracks(cycle_name, frames, filtered_matches_dict, 
                #                                         features_data, output_dir='cycle_visualization')
    

                if mean_cycle_error > 25:
                    # 找出三对匹配中数量最少的一对
                    min_match_count = min(pair_match_counts)
                    min_match_idx = pair_match_counts.index(min_match_count)
                    pair_to_remove = pair_keys[min_match_idx]
                    
                    if verbose:
                        print(f"三元组 {cycle_name} 循环误差过大 ({mean_cycle_error:.2f} > 30)，移除最弱匹配对 {pair_to_remove} (匹配数: {min_match_count})")
                    
                    # 将该匹配对从 filtered_matches_dict 中移除
                    if pair_to_remove in filtered_matches_dict:
                        del filtered_matches_dict[pair_to_remove]
                        
                        # 更新统计信息
                        stats["循环误差过滤"][pair_to_remove] = 0
                        
                        # 更新图结构，移除边
                        key1, key2 = pair_to_remove.split('-')
                        if G.has_edge(key1, key2):
                            G.remove_edge(key1, key2)
                            
                        # 由于已经移除了匹配对，这个三元组不需要进行后续的一致性检查
                        # 直接跳过当前循环的后续步骤
                        continue
                
                                # 记录循环误差数据
                
                cycle_error_data.append({
                    'cycle': cycle_name,
                    'mean_error': mean_cycle_error,
                    'matches_1': pair_match_counts[0],
                    'matches_2': pair_match_counts[1],
                    'matches_3': pair_match_counts[2],
                    'checked_points': len(distances),
                    'consistent_points': np.sum(is_consistent)
                })
            consistent_valid_idx = valid_idx[is_consistent]
            inconsistent_valid_idx = valid_idx[~is_consistent]

            # === Step 5: 为第二对和第三对构建查找表 ===
            match_dict_1 = {(int(m1), int(m2)): i for i, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[1]])}
            match_dict_2 = {(int(m1), int(m2)): i for i, (m1, m2) in enumerate(filtered_matches_dict[pair_keys[2]])}

            # === Step 6: 处理一致匹配 ===
            for i in consistent_valid_idx:
                idx1 = idx1_arr[i]
                idx2 = idx2_arr[i]
                idx3 = idx3_arr[i]
                idx1_cycle = idx1_cycle_arr[i]

                checked_indices[0].add(i)
                consistent_indices[0].add(i)

                idx_2 = match_dict_1.get((idx2, idx3))
                if idx_2 is not None:
                    checked_indices[1].add(idx_2)
                    consistent_indices[1].add(idx_2)

                idx_3 = match_dict_2.get((idx3, idx1_cycle))
                if idx_3 is not None:
                    checked_indices[2].add(idx_3)
                    consistent_indices[2].add(idx_3)

            # === Step 7: 处理不一致匹配 ===
            kp2 = np.array(features_data[keys[1]]['kp'])
            kp3 = np.array(features_data[keys[2]]['kp'])

            for i in inconsistent_valid_idx:
                idx1 = idx1_arr[i]
                idx2 = idx2_arr[i]
                idx3 = idx3_arr[i]

                checked_indices[0].add(i)
                
                # 添加对应的坐标
                inconsistent_points[0][pair_keys[0]].append(kp1[idx1])
                inconsistent_points[1][pair_keys[1]].append(kp2[idx2])
                inconsistent_points[2][pair_keys[2]].append(kp3[idx3])
            
            # 更新掩码 - 只将不一致的匹配设为False
            for i in range(3):
                for idx in checked_indices[i]:
                    if idx not in consistent_indices[i]:
                        match_masks[pair_keys[i]][idx] = False
            
            # 统计信息
            for i in range(3):
                cycle_total_checked += len(checked_indices[i])
                cycle_filtered_count += len(checked_indices[i]) - len(consistent_indices[i])
                
                if verbose and checked_indices[i]:
                    consist_ratio = len(consistent_indices[i]) / len(checked_indices[i]) * 100
                    print(f"{pair_keys[i]}: 循环检查了 {len(checked_indices[i])} 个匹配, 一致: {len(consistent_indices[i])} ({consist_ratio:.1f}%)")
            
            # 3. 移除靠近不一致匹配的点对
            if distance_threshold > 0:
                for i, pair_key in enumerate(pair_keys):
                    if pair_key in inconsistent_points[i] and len(inconsistent_points[i][pair_key]) > 0:
                        key1, key2 = pair_key.split('-')
                        matches = filtered_matches_dict[pair_key]
                        kp1 = features_data[key1]['kp']
                        kp2 = features_data[key2]['kp']
                        matched_pts1 = kp1[matches[:, 0]]
                        matched_pts2 = kp2[matches[:, 1]]
                        
                        inconsistent_pts = np.array(inconsistent_points[i][pair_key])
                        
                        # 计算所有匹配点与所有不一致点的距离（向量化）
                        distances1 = np.linalg.norm(matched_pts1[:, np.newaxis] - inconsistent_pts[np.newaxis, :], axis=2)
                        distances2 = np.linalg.norm(matched_pts2[:, np.newaxis] - inconsistent_pts[np.newaxis, :], axis=2)
                        
                        # 找到距离小于阈值的匹配点（在任一图像中）
                        proximity_mask = np.any(distances1 < distance_threshold, axis=1) | np.any(distances2 < distance_threshold, axis=1)
                        
                        # 更新掩码：仅保留未标记为不一致且不在邻近范围的匹配
                        original_mask = match_masks[pair_key].copy()
                        match_masks[pair_key] = match_masks[pair_key] & ~proximity_mask
                        
                        # 统计因邻近过滤移除的匹配数
                        removed_count = np.sum(original_mask & proximity_mask)
                        proximity_filtered_count += removed_count
                        
                        if verbose and removed_count > 0:
                            print(f"{pair_key}: 邻近过滤移除 {removed_count} 对匹配")
    
    # 将循环误差数据保存到CSV文件
    if output_csv and cycle_error_data:
        # 创建目录（如果不存在）
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 将数据转换为DataFrame并保存为CSV
        df = pd.DataFrame(cycle_error_data)
        df.to_csv(output_csv, index=False)
        
        if verbose:
            print(f"循环误差统计已保存至: {output_csv}")
            
        # 打印一些汇总统计
        if verbose:
            print("\n循环误差统计摘要:")
            print(f"总三元组数量: {len(df)}")
            print(f"平均循环误差: {df['mean_error'].mean():.2f} 像素")
            print(f"最大循环误差: {df['mean_error'].max():.2f} 像素")
            print(f"最小循环误差: {df['mean_error'].min():.2f} 像素")
    
    # 应用循环一致性和邻近过滤掩码
    for key_pair in filtered_matches_dict:
        if key_pair in match_masks:
            original_count = len(filtered_matches_dict[key_pair])
            filtered_matches_dict[key_pair] = filtered_matches_dict[key_pair][match_masks[key_pair]]
            filtered_count = original_count - len(filtered_matches_dict[key_pair])
            
            if verbose and filtered_count > 0:
                print(f"{key_pair}: 循环一致性及邻近过滤移除 {filtered_count} 对匹配 ({filtered_count/original_count*100:.1f}%)，剩余 {len(filtered_matches_dict[key_pair])} 对")
    
    if verbose and cycle_total_checked > 0:
        print(f"循环一致性总计检查: {cycle_total_checked} 对匹配, 过滤: {cycle_filtered_count} 对 ({cycle_filtered_count/cycle_total_checked*100:.1f}%)")
    if verbose and proximity_filtered_count > 0:
        print(f"邻近过滤总计移除: {proximity_filtered_count} 对匹配")
    
    # 保存循环一致性和邻近过滤后的状态
    for key_pair in filtered_matches_dict:
        stats["循环一致性检查后"][key_pair] = len(filtered_matches_dict[key_pair])
    
    # 4. 移除匹配点集中在直线附近的匹配对
    if 0:
        line_removed_pairs = 0
        for key_pair in list(filtered_matches_dict.keys()):
            if len(filtered_matches_dict[key_pair]) < 5:  # 匹配点太少无法可靠拟合直线
                continue
            
            key1, key2 = key_pair.split('-')
            matches = filtered_matches_dict[key_pair]
            kp1 = features_data[key1]['kp']
            kp2 = features_data[key2]['kp']
            matched_pts1 = kp1[matches[:, 0]]  # 形状: (M, 2)
            matched_pts2 = kp2[matches[:, 1]]  # 形状: (M, 2)
            
            # 使用RANSAC拟合直线，检查两张图像的点分布
            for pts, img_name in [(matched_pts1, key1), (matched_pts2, key2)]:
                try:
                    # 将点转换为float32以兼容cv2.fitLine
                    pts = pts.astype(np.float32)
                    pts = pts.reshape(-1, 1, 2)  # 转换为 (N, 1, 2) 格式
                    # 使用RANSAC拟合直线
                    # distType=cv2.DIST_L2表示使用最小二乘法，param=0表示自动选择
                    # reps=0.01表示点到直线的距离阈值，aeps=0.01表示角度阈值
                    # 使用 cv2.fitLine 拟合直线
                    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

                    # 假设 pts 是 (N, 1, 2) 的点集，vx, vy, x0, y0 是拟合直线的参数
                    # 提取点坐标
                    pts_array = pts[:, 0, :]  # 形状 (N, 2)，即 [px, py]

                    # 计算点到直线上某点的向量 (dx, dy)
                    deltas = pts_array - np.array([x0, y0])[None,:,0]  # 形状 (N, 2)

                    # 计算点到直线的距离（叉积的模）
                    distances = np.abs(deltas[:, 0] * vy - deltas[:, 1] * vx)  # 形状 (N,)

                    # 统计距离小于 10 像素的点数量
                    threshold = 10.0  # 距离阈值（像素）
                    inliers = sum(1 for dist in distances if dist < threshold)

                    
                    # 计算内点比例
                    if len(pts) > 0:
                        inlier_ratio = np.sum(inliers) / len(pts)
                    else:
                        inlier_ratio = 0.0
                    
                    if verbose:
                        print(f"{key_pair} 在图像 {img_name} 的内点比例: {inlier_ratio:.2f}")
                    
                    # 如果内点比例超过阈值，移除整个匹配对
                    if inlier_ratio > inlier_ratio_threshold:
                        if verbose:
                            print(f"{key_pair}: 在图像 {img_name} 的匹配点过于集中在直线附近 (内点比例 {inlier_ratio:.2f} > {inlier_ratio_threshold})，移除整个匹配对")
                        del filtered_matches_dict[key_pair]
                        line_removed_pairs += 1
                        stats["直线过滤后"][key_pair] = 0
                        break  # 任一图像的点集中在直线即可移除，无需检查另一张图像
                    
                except Exception as e:
                    if verbose:
                        print(f"{key_pair}: 直线拟合出错 (图像 {img_name}) - {e}")
                    continue
        
        if verbose and line_removed_pairs > 0:
            print(f"直线过滤总计移除: {line_removed_pairs} 个匹配对")
        
        # 保存直线过滤后的状态
        for key_pair in filtered_matches_dict:
            stats["直线过滤后"][key_pair] = len(filtered_matches_dict[key_pair])
        

    # 后续代码保持不变...
    # 1. 限制每对图像的最大匹配点数量
    limited_count = 0
    for key_pair in list(filtered_matches_dict.keys()):
        matches = filtered_matches_dict[key_pair]
        if len(matches) > 2000:
            # 排序后只保留最好的max_matches个匹配（假设匹配已按质量排序）
            filtered_matches_dict[key_pair] = matches[:2000]
            limited_count += 1
            
            if verbose:
                print(f"{key_pair}: 限制匹配数量从 {len(matches)} 到 {2000}")

    # 保存最终状态
    for key_pair in list(filtered_matches_dict.keys()):
        stats["最终"][key_pair] = len(filtered_matches_dict[key_pair])
    
    # 对已删除的匹配对，在最终状态中标记为0
    for key_pair in matches_dict:
        if key_pair not in filtered_matches_dict:
            stats["最终"][key_pair] = 0

    # 保存循环误差过滤状态的初始化
    for key_pair in filtered_matches_dict:
        stats["循环误差过滤"][key_pair] = len(filtered_matches_dict[key_pair])
        
    if verbose:
        print(f"\n匹配过滤统计摘要:")
        print(f"共处理 {len(matches_dict)} 对匹配")
        print(f"保留匹配: {len(filtered_matches_dict)} 对\n")
        
        # 打印每个阶段的匹配数量变化
        print(f"各阶段匹配数量变化:")
        all_pairs = set()
        for stage in stats:
            all_pairs.update(stats[stage].keys())
        
        stages = list(stats.keys())
        header = "匹配对        "
        for stage in stages:
            header += f" | {stage}"
        print(header)
        print("-" * len(header))
        
        sorted_pairs = sorted(all_pairs, key=lambda x: stats["初始"].get(x, 0), reverse=True)
        
        for pair in sorted_pairs:
            line = f"{pair:12}"
            for stage in stages:
                value = stats[stage].get(pair, "-")
                line += f" | {value:6}"
            print(line)
    
    return filtered_matches_dict, cycle_error_data

def enhanced_filter_matches_graph(frames, matches_dict, features_data_t, 
                                 score_threshold=0.6, cycle_error_threshold=5.0,
                                 max_iterations=10, verbose=True):
    """
    增强型匹配过滤函数，基于三元组循环一致性进行匹配过滤，然后以高质量三元组作为种子扩展匹配组。
    
    Args:
        frames: 图像路径列表
        matches_dict: 匹配字典 {key1-key2: [match_indices, match_scores]}
        features_data_t: 特征数据字典
        score_threshold: 匹配分数阈值，高于此分数的匹配被认为是可靠的
        cycle_error_threshold: 循环误差阈值（像素），低于此值的循环被认为是一致的
        max_iterations: 最大迭代次数
        verbose: 是否输出详细统计信息
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
        cycle_error_data: 循环误差数据
    """
    import numpy as np
    import networkx as nx
    from collections import defaultdict
    import copy
    import heapq
    
    # 复制特征数据和匹配字典
    features_data_t = copy.deepcopy(features_data_t)
    features_data = {k: {"kp": v['kp'].cpu().numpy()} for k, v in features_data_t.items()}
    matches_dict_copy = copy.deepcopy(matches_dict)
    
    # 统计信息
    stats = defaultdict(dict)
    for key_pair, matches in matches_dict_copy.items():
        if len(matches) > 0:
            stats["初始"][key_pair] = len(matches[0])
    
    # 分离匹配索引和匹配分数
    filtered_matches_dict = {}
    filtered_scores_dict = {}
    for key_pair, matches in matches_dict_copy.items():
        if len(matches) > 0:
            filtered_matches_dict[key_pair] = matches[0]
            filtered_scores_dict[key_pair] = matches[1]
    
    # 1. 构建图像连接图
    G = nx.Graph()
    for key_pair, matches in filtered_matches_dict.items():
        key1, key2 = key_pair.split('-')
        if len(matches) > 0:
            # 使用匹配数量作为权重
            G.add_edge(key1, key2, weight=len(matches))
    
    # 2. 计算所有三元组的循环误差
    cycle_quality_map = {}  # {cycle_name: {"error": float, "pairs": [pair1, pair2, pair3]}}
    
    # 查找所有长度为3的循环
    for cycle in nx.cycle_basis(G):
        if len(cycle) == 3:
            # 获取三对匹配关系
            keys = cycle
            pairs = [
                (keys[0], keys[1]),
                (keys[1], keys[2]),
                (keys[2], keys[0])
            ]
            
            cycle_name = f"{keys[0]}-{keys[1]}-{keys[2]}"
            
            # 检查所有匹配对是否存在
            valid_cycle = True
            pair_keys = []
            
            for key1, key2 in pairs:
                pair_key = f"{key1}-{key2}"
                reverse_key = f"{key2}-{key1}"
                
                if pair_key in filtered_matches_dict:
                    pair_keys.append(pair_key)
                elif reverse_key in filtered_matches_dict:
                    pair_keys.append(reverse_key)
                else:
                    valid_cycle = False
                    break
            
            if not valid_cycle:
                continue
            
            # 计算循环误差
            mean_cycle_error = calculate_cycle_error(keys, pair_keys, filtered_matches_dict, features_data)
            
            if mean_cycle_error is not None:
                cycle_quality_map[cycle_name] = {
                    "error": mean_cycle_error,
                    "pairs": pair_keys,
                    "nodes": keys
                }
    
    # 3. 按循环误差排序
    sorted_cycles = sorted(cycle_quality_map.items(), key=lambda x: x[1]["error"])
    
    # 4. 迭代筛选过程
    reliable_matches = set()  # 存储可靠匹配对
    processed_cycles = set()  # 已处理的循环
    
    # 如果没有循环，则返回原始匹配
    if not sorted_cycles:
        if verbose:
            print("未发现有效循环，返回原始匹配")
        return filtered_matches_dict, None
    
    # 依次处理每个循环三元组作为种子
    for cycle_name, cycle_data in sorted_cycles:
        # 如果循环已处理，跳过
        if cycle_name in processed_cycles:
            continue
        
        # 如果循环误差太大，跳过
        if cycle_data["error"] > cycle_error_threshold * 2:
            continue
        
        # 将该循环标记为已处理
        processed_cycles.add(cycle_name)
        
        # 如果循环误差小于阈值，将其包含的匹配对加入可靠匹配集合
        if cycle_data["error"] <= cycle_error_threshold:
            for pair in cycle_data["pairs"]:
                reliable_matches.add(pair)
            
            # 以该三元组为种子，扩展匹配组
            current_group = set(cycle_data["nodes"])  # 当前组中的节点
            current_pairs = set(cycle_data["pairs"])  # 当前组中的匹配对
            
            # 迭代扩展，尝试添加与当前组构成循环的匹配对
            for _ in range(max_iterations):
                expanded = False
                
                # 1. 先寻找能与当前组中的节点构成新三元组的匹配对
                potential_cycles = []
                
                for node1 in current_group:
                    for node2 in current_group:
                        if node1 != node2:
                            # 检查所有与这两个节点相连的其他节点
                            for node3 in G.nodes():
                                if node3 not in current_group and G.has_edge(node1, node3) and G.has_edge(node2, node3):
                                    new_cycle = [node1, node2, node3]
                                    new_cycle_name = "-".join(sorted(new_cycle))
                                    
                                    if new_cycle_name in cycle_quality_map and new_cycle_name not in processed_cycles:
                                        cycle_error = cycle_quality_map[new_cycle_name]["error"]
                                        if cycle_error <= cycle_error_threshold:
                                            potential_cycles.append((cycle_error, new_cycle_name))
                
                # 按循环误差排序
                potential_cycles.sort()
                
                # 尝试添加最佳循环
                if potential_cycles:
                    _, best_cycle_name = potential_cycles[0]
                    best_cycle_data = cycle_quality_map[best_cycle_name]
                    
                    # 将新循环的节点和匹配对添加到当前组
                    current_group.update(best_cycle_data["nodes"])
                    for pair in best_cycle_data["pairs"]:
                        current_pairs.add(pair)
                        reliable_matches.add(pair)
                    
                    processed_cycles.add(best_cycle_name)
                    expanded = True
                
                # 2. 如果没有找到合适的循环，尝试添加与当前组中节点的高分数匹配
                if not expanded:
                    best_score = -1
                    best_pair = None
                    
                    for node in current_group:
                        for neighbor in G.neighbors(node):
                            if neighbor not in current_group:
                                pair_key = f"{node}-{neighbor}"
                                reverse_key = f"{neighbor}-{node}"
                                
                                if pair_key in filtered_matches_dict:
                                    key_to_check = pair_key
                                elif reverse_key in filtered_matches_dict:
                                    key_to_check = reverse_key
                                else:
                                    continue
                                
                                # 计算该匹配对的平均分数
                                if key_to_check in filtered_scores_dict:
                                    avg_score = np.mean(filtered_scores_dict[key_to_check])
                                    if avg_score > best_score and avg_score >= score_threshold:
                                        best_score = avg_score
                                        best_pair = key_to_check
                    
                    # 如果找到高分数匹配，添加到可靠匹配集合
                    if best_pair is not None:
                        node1, node2 = best_pair.split('-')
                        reliable_matches.add(best_pair)
                        current_pairs.add(best_pair)
                        current_group.add(node1)
                        current_group.add(node2)
                        expanded = True
                
                # 如果无法继续扩展，退出循环
                if not expanded:
                    break
    
    # 5. 构建最终过滤结果
    final_matches_dict = {}
    for pair in reliable_matches:
        if pair in filtered_matches_dict:
            final_matches_dict[pair] = filtered_matches_dict[pair]
    
    # 统计信息
    stats["最终"] = {pair: len(final_matches_dict[pair]) for pair in final_matches_dict}
    
    if verbose:
        print(f"\n增强型匹配过滤统计摘要:")
        print(f"共处理 {len(matches_dict_copy)} 对匹配")
        print(f"可靠匹配: {len(final_matches_dict)} 对")
        print(f"过滤比例: {(len(matches_dict_copy) - len(final_matches_dict)) / len(matches_dict_copy) * 100:.1f}%")
        
        # 打印每个阶段的匹配数量变化
        print(f"\n各阶段匹配数量变化:")
        all_pairs = set()
        for stage in stats:
            all_pairs.update(stats[stage].keys())
        
        stages = list(stats.keys())
        header = "匹配对        "
        for stage in stages:
            header += f" | {stage}"
        print(header)
        print("-" * len(header))
        
        sorted_pairs = sorted(all_pairs)
        for pair in sorted_pairs:
            line = f"{pair:12}"
            for stage in stages:
                value = stats[stage].get(pair, "-")
                line += f" | {value:6}"
            print(line)
    
    return final_matches_dict, cycle_quality_map


def calculate_cycle_error(nodes, pair_keys, matches_dict, features_data):
    """
    计算三元组的平均循环误差
    
    Args:
        nodes: 三元组中的节点 [node1, node2, node3]
        pair_keys: 三元组中的匹配对键 [pair1, pair2, pair3]
        matches_dict: 匹配字典
        features_data: 特征数据字典
        
    Returns:
        mean_cycle_error: 平均循环误差
    """
    import numpy as np
    
    # 创建匹配索引查找表
    match_maps = []
    is_reverse_lst = []
    
    for i, pair_key in enumerate(pair_keys):
        key1, key2 = pair_key.split('-')
        matches = matches_dict[pair_key]
        
        # 判断是否需要反转匹配
        if key1 == nodes[i] and key2 == nodes[(i+1)%3]:
            is_reverse = False
        else:
            is_reverse = True
        
        is_reverse_lst.append(is_reverse)
        
        # 创建从idx1到idx2的映射
        if is_reverse:
            match_map = {int(m[1]): int(m[0]) for m in matches}
        else:
            match_map = {int(m[0]): int(m[1]) for m in matches}
        
        match_maps.append(match_map)
    
    # 批量读取第一对匹配
    matches0 = np.array(matches_dict[pair_keys[0]], dtype=int)
    if is_reverse_lst[0]:
        idx1_arr = matches0[:, 1]
        idx2_arr = matches0[:, 0]
    else:
        idx1_arr = matches0[:, 0]
        idx2_arr = matches0[:, 1]
    
    # 映射 idx2 -> idx3 和 idx3 -> idx1_cycle
    idx3_arr = np.array([match_maps[1].get(idx2, -1) for idx2 in idx2_arr])
    valid_1 = idx3_arr != -1
    
    idx1_cycle_arr = np.array([match_maps[2].get(idx3, -1) if valid else -1
                              for idx3, valid in zip(idx3_arr, valid_1)])
    valid_2 = idx1_cycle_arr != -1
    
    valid = valid_1 & valid_2
    
    if not np.any(valid):
        return None
    
    # 加载关键点坐标
    kp1 = np.array(features_data[nodes[0]]['kp'])
    pt1 = kp1[idx1_arr[valid]]
    pt1_cycle = kp1[idx1_cycle_arr[valid]]
    
    # 计算距离和循环误差均值
    distances = np.linalg.norm(pt1 - pt1_cycle, axis=1)
    
    if len(distances) > 0:
        return np.mean(distances)
    else:
        return None
    
def filter_matches_comprehensive(matches_dict, features_data, scores_dict, min_matches=15):
    """
    综合过滤策略，包含:
    1. 基于聚类的离群点检测
    2. 基于图连通性的稀疏匹配过滤
    3. 基于局部结构一致性的过滤
    4. 基于匹配分数的自适应阈值过滤
    
    Args:
        matches_dict: {pair_key: matches} 匹配字典
        features_data: 特征点信息
        scores_dict: {pair_key: scores} 匹配分数
        min_matches: 最小保留匹配数
    """
    filtered_dict = matches_dict.copy()
    
    # 1. 构建图结构分析连通性
    G = nx.Graph()
    for pair_key, matches in matches_dict.items():
        key1, key2 = pair_key.split('-')
        G.add_edge(key1, key2, weight=len(matches))
    
    # 2. 对每对匹配进行过滤
    for pair_key in list(filtered_dict.keys()):
        matches = filtered_dict[pair_key]
        if len(matches) < min_matches:
            continue
            
        key1, key2 = pair_key.split('-')
        kpts1 = features_data[key1]['kp']
        kpts2 = features_data[key2]['kp']
        scores = scores_dict[pair_key]
        
        # 2.1 聚类过滤
        filtered_indices = cluster_based_filtering(
            kpts1[matches[:, 0]], 
            kpts2[matches[:, 1]], 
            scores
        )
        
        if len(filtered_indices) < min_matches:
            del filtered_dict[pair_key]
            continue
            
        # # 2.2 局部结构一致性检查
        # filtered_indices = check_local_structure(
        #     kpts1[matches[filtered_indices, 0]],
        #     kpts2[matches[filtered_indices, 1]],
        #     scores[filtered_indices]
        # )
        
        # if len(filtered_indices) < min_matches:
        #     del filtered_dict[pair_key]
        #     continue
            
        # 2.3 自适应分数阈值过滤
        score_threshold = adaptive_score_threshold(scores[filtered_indices])
        final_indices = filtered_indices[scores[filtered_indices] > score_threshold]
        
        if len(final_indices) < min_matches:
            del filtered_dict[pair_key]
        else:
            filtered_dict[pair_key] = matches[final_indices]
    
    return filtered_dict

def cluster_based_filtering(pts1, pts2, scores, eps=None, min_samples=3):
    """基于DBSCAN聚类的离群点检测"""
    # 计算匹配点位移向量
    motions = pts2 - pts1
    
    # 自适应确定eps
    if eps is None:
        pts_range = np.max(motions, axis=0) - np.min(motions, axis=0)
        eps = np.mean(pts_range) * 0.1
    
    # DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(motions)
    labels = clustering.labels_
    
    # 找出最大聚类
    if len(np.unique(labels)) > 1:  # 存在多个聚类
        # 结合分数权重选择主要聚类
        clusters = []
        for label in range(labels.max() + 1):
            mask = (labels == label)
            avg_score = np.mean(scores[mask])
            size = np.sum(mask)
            clusters.append((label, size * avg_score))
        
        main_cluster = max(clusters, key=lambda x: x[1])[0]
        return np.where(labels == main_cluster)[0]
    else:
        return np.arange(len(pts1))

def check_local_structure(pts1, pts2, scores, radius=30):
    """检查局部结构一致性"""
    valid_indices = []
    
    for i in range(len(pts1)):
        # 找出局部邻域内的点
        dist1 = np.linalg.norm(pts1 - pts1[i], axis=1)
        neighbors1 = np.where(dist1 < radius)[0]
        
        dist2 = np.linalg.norm(pts2 - pts2[i], axis=1)
        neighbors2 = np.where(dist2 < radius)[0]
        
        if len(neighbors1) < 3 or len(neighbors2) < 3:
            continue
            
        # 计算局部结构相似度
        structure1 = compute_local_structure(pts1[neighbors1])
        structure2 = compute_local_structure(pts2[neighbors2])
        
        if structure_similarity(structure1, structure2) > 0.7:
            valid_indices.append(i)
    
    return np.array(valid_indices)

def compute_local_structure(pts):
    """计算点集的局部结构特征"""
    # 使用相对角度和距离比作为结构特征
    center = np.mean(pts, axis=0)
    vectors = pts - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    distances = np.linalg.norm(vectors, axis=1)
    
    # 归一化距离
    distances = distances / np.max(distances)
    
    return np.column_stack([angles, distances])

def structure_similarity(s1, s2):
    """计算两个局部结构的相似度"""
    if len(s1) != len(s2):
        return 0
    
    # 计算角度差异和距离比差异
    angle_diff = np.min(np.abs(s1[:, 0][:, None] - s2[:, 0]))
    dist_diff = np.mean(np.abs(s1[:, 1] - s2[:, 1]))
    
    return np.exp(-(angle_diff + dist_diff))

def adaptive_score_threshold(scores):
    """自适应确定分数阈值"""
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score - 0.5 * std_score

def check_cycle_consistency(keys, matches_dict, features_data):
    """
    检查三元环的匹配一致性，只考虑三张图像中共同可见的特征点
    """
    import numpy as np
    
    # 获取三对匹配
    pairs = [
        (keys[0], keys[1]),
        (keys[1], keys[2]),
        (keys[2], keys[0])
    ]
    
    # 检查所有匹配对是否存在
    for key1, key2 in pairs:
        pair_key = f"{key1}-{key2}"
        reverse_key = f"{key2}-{key1}"
        if pair_key not in matches_dict and reverse_key not in matches_dict:
            return 0.0  # 缺失匹配对，一致性为0
    
    # 构建匹配索引查找表
    match_maps = []
    for key1, key2 in pairs:
        pair_key = f"{key1}-{key2}"
        if pair_key in matches_dict:
            matches = matches_dict[pair_key]
        else:
            reverse_key = f"{key2}-{key1}"
            matches = matches_dict[reverse_key]
            # 调整索引顺序
            matches = np.fliplr(matches)
            
        # 创建从idx1到idx2的映射
        match_map = {int(m[0]): int(m[1]) for m in matches}
        match_maps.append(match_map)
    
    # 检查循环一致性
    consistent_count = 0
    total_checked = 0  # 只计算我们实际检查的点数
    
    # 取第一对的匹配点
    first_matches = list(match_maps[0].items())
    
    for idx1, idx2 in first_matches:
        # 只检查可能形成闭环的点
        if idx2 in match_maps[1]:
            idx3 = match_maps[1][idx2]
            if idx3 in match_maps[2]:
                # 这个点确实在三张图像中都有对应，可以进行闭环检查
                total_checked += 1
                idx1_cycle = match_maps[2][idx3]
                if idx1_cycle == idx1:
                    consistent_count += 1
    
    # 如果没有可检查的完整循环，返回0
    if total_checked == 0:
        return 1
        
    # 只计算实际检查过的点的一致性比例
    return consistent_count / total_checked

def filter_matches_semantic(frames, matches_dict, features_data, threshold=0.6):
    """
    基于语义相似度的匹配过滤函数
    
    Args:
        frames: 图像路径列表
        matches_dict: 匹配字典 {key1-key2: matches}
        features_data: 特征数据字典
        threshold: 相似度阈值
        
    Returns:
        filtered_matches_dict: 过滤后的匹配字典
    """
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    
    # 尝试导入CLIP模型（如果可用）
    try:
        from CLIP.clip import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
        semantic_available = True
    except:
        semantic_available = False
        print("无法加载CLIP模型，跳过语义相似度检查")
    
    filtered_matches_dict = matches_dict.copy()
    
    if not semantic_available:
        return filtered_matches_dict
    
    # 提取图像区域的语义特征
    for key_pair in list(filtered_matches_dict.keys()):
        key1, key2 = key_pair.split('-')
        matches = filtered_matches_dict[key_pair]
        
        if len(matches) < 10:
            continue
            
        # 找到frame路径
        frame1 = None
        frame2 = None
        for frame in frames:
            if key1 in frame:
                frame1 = frame
            if key2 in frame:
                frame2 = frame
                
        if frame1 is None or frame2 is None:
            continue
            
        # 加载图像
        try:
            img1 = Image.open(frame1)
            img2 = Image.open(frame2)
            
            # 获取匹配点坐标
            kp1 = features_data[key1]['kp']
            kp2 = features_data[key2]['kp']
            
            matched_pts1 = kp1[matches[:, 0]]
            matched_pts2 = kp2[matches[:, 1]]
            
            # 计算匹配点的凸包并提取区域
            if len(matched_pts1) > 3:
                hull1 = cv2.convexHull(matched_pts1.astype(np.int32))
                hull2 = cv2.convexHull(matched_pts2.astype(np.int32))
                
                # 创建掩码
                mask1 = np.zeros(img1.size[::-1], dtype=np.uint8)
                mask2 = np.zeros(img2.size[::-1], dtype=np.uint8)
                
                cv2.fillPoly(mask1, [hull1], 255)
                cv2.fillPoly(mask2, [hull2], 255)
                
                # 计算边界框
                x1, y1, w1, h1 = cv2.boundingRect(hull1)
                x2, y2, w2, h2 = cv2.boundingRect(hull2)
                
                # 裁剪区域
                region1 = np.array(img1)[y1:y1+h1, x1:x1+w1]
                region2 = np.array(img2)[y2:y2+h2, x2:x2+w2]
                
                # 转回PIL图像
                region1_pil = Image.fromarray(region1)
                region2_pil = Image.fromarray(region2)
                
                # 计算CLIP特征
                with torch.no_grad():
                    region1_features = model.encode_image(preprocess(region1_pil).unsqueeze(0).to(device))
                    region2_features = model.encode_image(preprocess(region2_pil).unsqueeze(0).to(device))
                    
                    # 正则化
                    region1_features = region1_features / region1_features.norm(dim=-1, keepdim=True)
                    region2_features = region2_features / region2_features.norm(dim=-1, keepdim=True)
                    
                    # 计算相似度
                    similarity = (region1_features @ region2_features.T).item()
                    
                    # 低相似度可能表示虚假匹配
                    if similarity < threshold:
                        # 根据相似度调整保留的匹配点数量
                        keep_ratio = max(0.1, similarity / threshold)
                        keep_count = max(5, int(len(matches) * keep_ratio))
                        filtered_matches_dict[key_pair] = matches[:keep_count]
        except Exception as e:
            print(f"处理{key_pair}时出错: {e}")
            continue
    
    return filtered_matches_dict

def adaptive_match_filtering(matcher, kp1, kp2, matches, img1_path, img2_path, device, min_matches=15):
    """
    使用多种策略自适应地过滤匹配对
    
    Args:
        matcher: LightGlue匹配器
        kp1, kp2: 两张图片的特征点
        matches: 原始匹配结果 (Nx2)
        img1_path, img2_path: 图像路径
        device: 设备
        min_matches: 最小匹配数
        
    Returns:
        filtered_matches: 过滤后的匹配索引 (Mx2)
    """
    import numpy as np
    import cv2
    import torch
    from sklearn.cluster import DBSCAN
    
    if len(matches) <= min_matches:
        return matches  # 匹配数量已经很少，不再过滤
    
    # 转换为numpy数组以便处理
    if isinstance(matches, torch.Tensor):
        matches_np = matches.cpu().numpy()
    else:
        matches_np = matches
        
    if isinstance(kp1, torch.Tensor):
        kp1_np = kp1.cpu().numpy()
    else:
        kp1_np = kp1
        
    if isinstance(kp2, torch.Tensor):
        kp2_np = kp2.cpu().numpy()
    else:
        kp2_np = kp2
    
    # 1. 首先应用RANSAC过滤 (基础矩阵)
    try:
        pts1 = kp1_np[matches_np[:, 0]]
        pts2 = kp2_np[matches_np[:, 1]]
        
        # 使用MAGSAC算法，较为鲁棒
        _, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.USAC_MAGSAC,
            1.0, 0.999, 10000
        )
        
        if mask is not None:
            mask = mask.ravel().astype(bool)
            matches_np = matches_np[mask]
    except Exception as e:
        print(f"RANSAC过滤出错: {e}")
    
    # 如果匹配数太少，返回RANSAC结果
    if len(matches_np) <= min_matches:
        return matches_np
    
    # 2. 空间一致性检查 - 使用DBSCAN聚类检测异常匹配
    try:
        pts1 = kp1_np[matches_np[:, 0]]
        pts2 = kp2_np[matches_np[:, 1]]
        
        # 动态确定eps参数
        pts_range1 = np.max(pts1, axis=0) - np.min(pts1, axis=0)
        pts_range2 = np.max(pts2, axis=0) - np.min(pts2, axis=0)
        avg_range = (np.mean(pts_range1) + np.mean(pts_range2)) / 2
        eps = max(20, avg_range * 0.05)  # 自适应聚类距离
        
        # 对两张图的点分别进行聚类
        db1 = DBSCAN(eps=eps, min_samples=3).fit(pts1)
        db2 = DBSCAN(eps=eps, min_samples=3).fit(pts2)
        
        labels1 = db1.labels_
        labels2 = db2.labels_
        
        # 检查聚类一致性
        cluster_map = {}
        for i in range(len(labels1)):
            if labels1[i] == -1 or labels2[i] == -1:
                continue  # 跳过噪声点
                
            cluster_pair = (labels1[i], labels2[i])
            if cluster_pair not in cluster_map:
                cluster_map[cluster_pair] = 0
            cluster_map[cluster_pair] += 1
        
        # 保留主要聚类对的匹配
        valid_mask = np.zeros(len(matches_np), dtype=bool)
        for i in range(len(labels1)):
            cluster_pair = (labels1[i], labels2[i])
            if cluster_pair in cluster_map and cluster_map[cluster_pair] >= 3:
                valid_mask[i] = True
        
        # 如果过滤后的匹配太少，保留所有非噪声点
        if np.sum(valid_mask) < min_matches:
            valid_mask = (labels1 != -1) & (labels2 != -1)
            
        matches_np = matches_np[valid_mask]
    except Exception as e:
        print(f"聚类分析出错: {e}")
    
    # 3. 主方向一致性检查
    try:
        if len(matches_np) > min_matches:
            pts1 = kp1_np[matches_np[:, 0]]
            pts2 = kp2_np[matches_np[:, 1]]
            
            # 计算匹配点之间的方向向量
            directions = []
            for i in range(len(pts1)-1):
                for j in range(i+1, len(pts1)):
                    # 第一张图中两点的方向
                    dir1 = pts1[j] - pts1[i]
                    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-10)
                    
                    # 第二张图中对应两点的方向
                    dir2 = pts2[j] - pts2[i]
                    dir2 = dir2 / (np.linalg.norm(dir2) + 1e-10)
                    
                    # 计算方向相似度 (点积)
                    sim = np.dot(dir1, dir2)
                    directions.append(sim)
            
            # 分析方向一致性
            directions = np.array(directions)
            mean_dir = np.mean(directions)
            std_dir = np.std(directions)
            
            # 方向一致性低，可能是错误匹配
            if mean_dir < 0.6 or std_dir > 0.4:
                # 减少保留的匹配数量
                keep_count = max(min_matches, len(matches_np) // 2)
                matches_np = matches_np[:keep_count]
    except Exception as e:
        print(f"方向一致性检查出错: {e}")
    
    # 如果是torch.Tensor，转回tensor
    if isinstance(matches, torch.Tensor):
        return torch.tensor(matches_np, device=matches.device, dtype=matches.dtype)
    else:
        return matches_np