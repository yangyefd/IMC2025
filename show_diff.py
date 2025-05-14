import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def normalize_filename(filename):
    """
    归一化文件名，处理像 a_b_matches.png 和 b_a_matches.png 这样的情况
    """
    if '_matches.png' in filename:
        # 对于形如 a_b_matches.png 的文件名
        parts = filename.replace('_matches.png', '').split('_')
        if len(parts) >= 2:
            # 取出图像名称
            names = []
            current_name = ""
            for part in parts:
                if part.startswith('stairs') or current_name:
                    if not current_name:
                        current_name = part
                    else:
                        current_name += "_" + part
                        if ".png" in part:
                            names.append(current_name)
                            current_name = ""
            
            # 如果成功提取出两个名称，进行排序后重新组合
            if len(names) >= 2:
                sorted_names = sorted(names)
                return '_'.join(sorted_names) + '_matches.png'
    
    # 对于其他情况，直接返回原文件名
    return filename

def get_file_mapping(folder1, folder2):
    """
    获取两个文件夹中文件的映射关系
    """
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # 创建归一化的文件名映射
    norm_to_actual1 = {normalize_filename(f): f for f in files1}
    norm_to_actual2 = {normalize_filename(f): f for f in files2}
    
    # 获取所有归一化后的文件名
    all_norm_files = set(norm_to_actual1.keys()) | set(norm_to_actual2.keys())
    
    # 创建映射
    file_mapping = []
    for norm_file in all_norm_files:
        actual_file1 = norm_to_actual1.get(norm_file, None)
        actual_file2 = norm_to_actual2.get(norm_file, None)
        file_mapping.append((norm_file, actual_file1, actual_file2))
    
    return file_mapping

def merge_images(folder1, folder2, output_folder, max_width=1200, max_height=800):
    """
    合并两个文件夹中的相同图像并保存到输出文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取文件映射
    file_mapping = get_file_mapping(folder1, folder2)
    
    print(f"找到 {len(file_mapping)} 对需要比较的图像")
    
    for norm_file, file1, file2 in tqdm(file_mapping):
        # 读取第一张图像，如果不存在则创建黑图
        if file1:
            img1 = cv2.imread(os.path.join(folder1, file1))
            if img1 is None:
                print(f"无法读取图像: {os.path.join(folder1, file1)}")
                continue
        else:
            # 我们需要等待知道第二张图像的大小才能创建黑图
            img1 = None
        
        # 读取第二张图像，如果不存在则创建黑图
        if file2:
            img2 = cv2.imread(os.path.join(folder2, file2))
            if img2 is None:
                print(f"无法读取图像: {os.path.join(folder2, file2)}")
                continue
        else:
            # 我们需要等待知道第一张图像的大小才能创建黑图
            img2 = None
        
        # 如果两张图像都不存在，则跳过
        if img1 is None and img2 is None:
            print(f"两个文件夹中都没有找到图像: {norm_file}")
            continue
        
        # 如果其中一张图像不存在，则创建相同大小的黑图
        if img1 is None:
            img1 = np.zeros_like(img2)
        elif img2 is None:
            img2 = np.zeros_like(img1)
        
        # 调整两张图像为相同大小
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 确定目标尺寸
        target_w = min(max(w1, w2), max_width)
        target_h = min(max(h1, h2), max_height)
        
        # 调整图像大小
        if w1 > target_w or h1 > target_h:
            scale = min(target_w / w1, target_h / h1)
            new_w, new_h = int(w1 * scale), int(h1 * scale)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if w2 > target_w or h2 > target_h:
            scale = min(target_w / w2, target_h / h2)
            new_w, new_h = int(w2 * scale), int(h2 * scale)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 获取调整后的尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建合并图像
        merged_img = np.zeros((h1 + h2 + 20, max(w1, w2), 3), dtype=np.uint8)
        
        # 添加图像到合并图像
        merged_img[:h1, :w1] = img1
        merged_img[h1+20:h1+20+h2, :w2] = img2
        
        # 添加分隔线
        cv2.line(merged_img, (0, h1+10), (max(w1, w2), h1+10), (255, 255, 255), 2)
        
        # 添加文件名标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # 添加文件夹1的标签
        folder1_label = f"Folder1: {file1 if file1 else 'Not found'}"
        cv2.putText(merged_img, folder1_label, (10, h1 - 10), font, font_scale, (0, 255, 255), font_thickness)
        
        # 添加文件夹2的标签
        folder2_label = f"Folder2: {file2 if file2 else 'Not found'}"
        cv2.putText(merged_img, folder2_label, (10, h1 + 40), font, font_scale, (0, 255, 255), font_thickness)
        
        # 保存合并图像
        output_filename = norm_file.replace('.png', '_comparison.png')
        cv2.imwrite(os.path.join(output_folder, output_filename), merged_img)
    
    print(f"合并完成，结果保存在: {output_folder}")

def main(A_path,B_path,output_path):
    parser = argparse.ArgumentParser(description='比较两个文件夹中的相同图像')
    # parser.add_argument('folder1', help='第一个文件夹路径')
    # parser.add_argument('folder2', help='第二个文件夹路径')
    # parser.add_argument('--output', '-o', default='comparison_results', help='输出文件夹路径')
    parser.add_argument('--max-width', type=int, default=1200, help='最大宽度')
    parser.add_argument('--max-height', type=int, default=800, help='最大高度')
    
    args = parser.parse_args()
    
    args.folder1, args.folder2, args.output = A_path, B_path, output_path

    if not os.path.exists(args.folder1):
        print(f"错误: 文件夹不存在: {args.folder1}")
        return
    
    if not os.path.exists(args.folder2):
        print(f"错误: 文件夹不存在: {args.folder2}")
        return
    
    merge_images(args.folder1, args.folder2, args.output, args.max_width, args.max_height)

if __name__ == "__main__":
    A_path = "E:\\yey\\work\\IMC2025\\IMC2025\\results\\featureout\\stairs\\visualizations"  # 替换为实际路径
    B_path = "E:\\yey\\work\\IMC2025\\IMC2025\\results\\featureout_mr\\stairs\\visualizations"  # 替换为实际路径
    output_path = "E:\\yey\\work\\IMC2025\\IMC2025\\results\\compare"  # 替换为实际路径

    main(A_path,B_path,output_path)