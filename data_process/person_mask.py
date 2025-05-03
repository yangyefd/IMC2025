
import cv2
import numpy as np
from pathlib import Path
# 支持的图像扩展名
image_extensions = ('.jpg', '.jpeg', '.png')

def calculate_texture_variance(image, mask):
    """计算掩码区域的颜色方差"""
    masked_image = image[mask]
    if masked_image.size == 0:
        return 0
    variance = np.var(masked_image, axis=0).mean()
    return variance

def detect_colors(image, mask):
    """检测掩码区域内的肉色和雕塑颜色占比"""
    import numpy as np
    import cv2
    
    # 确保掩码是二值化的uint8类型
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # 确保掩码是二值化的（只有0和1或0和255的值）
    if np.max(mask) > 1:
        # 如果掩码值大于1，假设它是0-255范围的，将其转换为二值掩码
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # 将图像转换为HSV颜色空间
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 创建掩码区域内的HSV图像
    # 创建掩码副本以避免修改原始掩码
    mask_3d = np.stack([mask, mask, mask], axis=-1)
    # 提取掩码区域内的像素
    masked_pixels = image_hsv[mask > 0]
    
    # 如果掩码筛选后为空，返回0
    if masked_pixels.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 确保 masked_pixels 是二维数组，每行是一个HSV像素
    if len(masked_pixels.shape) == 1:
        masked_pixels = masked_pixels.reshape(-1, 3)
    
    # 肉色范围 (HSV) - 更严格的肉色范围以避免误识别
    skin_lower = np.array([0, 60, 100], dtype=np.uint8)
    skin_upper = np.array([17, 140, 230], dtype=np.uint8)
    
    # 军绿色范围 (HSV)
    olive_lower = np.array([40, 50, 50], dtype=np.uint8)
    olive_upper = np.array([80, 255, 180], dtype=np.uint8)
    
    # 青铜色范围 (HSV) - 扩大范围以更好地检测铜雕塑
    bronze_lower = np.array([10, 40, 40], dtype=np.uint8)
    bronze_upper = np.array([35, 200, 220], dtype=np.uint8)
    
    # 大理石色范围 (HSV)
    marble_lower = np.array([0, 0, 180], dtype=np.uint8)
    marble_upper = np.array([25, 40, 255], dtype=np.uint8)
    
    # 石灰色/灰色范围 (HSV，常见于石质雕塑或建筑)
    stone_lower = np.array([0, 0, 40], dtype=np.uint8)
    stone_upper = np.array([179, 60, 180], dtype=np.uint8)
    
    # 计算每种颜色的像素数量
    skin_pixels = np.sum(np.all((masked_pixels >= skin_lower) & (masked_pixels <= skin_upper), axis=1))
    olive_pixels = np.sum(np.all((masked_pixels >= olive_lower) & (masked_pixels <= olive_upper), axis=1))
    bronze_pixels = np.sum(np.all((masked_pixels >= bronze_lower) & (masked_pixels <= bronze_upper), axis=1))
    marble_pixels = np.sum(np.all((masked_pixels >= marble_lower) & (masked_pixels <= marble_upper), axis=1))
    stone_pixels = np.sum(np.all((masked_pixels >= stone_lower) & (masked_pixels <= stone_upper), axis=1))
    
    # 计算颜色占比 - 使用掩码中的总像素数作为分母
    total_pixels = masked_pixels.shape[0]  # 掩码区域内的总像素数
    
    skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0.0
    olive_ratio = olive_pixels / total_pixels if total_pixels > 0 else 0.0
    bronze_ratio = bronze_pixels / total_pixels if total_pixels > 0 else 0.0
    marble_ratio = marble_pixels / total_pixels if total_pixels > 0 else 0.0
    stone_ratio = stone_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return skin_ratio, olive_ratio, bronze_ratio, marble_ratio, stone_ratio

def is_person(image, mask, conf_score, variance_threshold=500, skin_threshold=0.03, sculpture_threshold=0.5, conf_threshold=0.7):
    """严格判断是否为人类，减少误判"""
    # 计算掩码面积占比
    # total_pixels = image.shape[0] * image.shape[1]
    # mask_area = np.sum(mask)
    # mask_ratio = mask_area / total_pixels if total_pixels > 0 else 0.0
    # if mask_ratio > 0.4:
    #     print(f"Mask area too large ({mask_ratio:.2f} > 0.5)")
    #     return False

    # 检查置信度
    if conf_score < conf_threshold:
        # print(f"Confidence too low ({conf_score:.2f} < {conf_threshold})")
        return False

    # 检测颜色
    skin_ratio, olive_ratio, bronze_ratio, marble_ratio, stone_ratio = detect_colors(image, mask)
    # print(f"Skin ratio: {skin_ratio:.3f}, Bronze ratio: {bronze_ratio:.3f}, Marble ratio: {marble_ratio:.3f}, Stone ratio: {stone_ratio:.3f}")


    # 计算纹理方差
    variance = calculate_texture_variance(image, mask)
    if variance < variance_threshold:
        # print(f"Variance too low ({variance:.2f} < {variance_threshold})")
        return False

    # 判断逻辑：严格要求肉色占比高，且无明显雕塑颜色
    if skin_ratio > skin_threshold and  olive_ratio < sculpture_threshold and bronze_ratio < sculpture_threshold and marble_ratio < sculpture_threshold and stone_ratio < sculpture_threshold:
        return True  # 高肉色占比，无雕塑颜色，判定为人类
    return False  # 其他情况一律判定为非人类

def person_mask(image_path, model):

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        # print(f"Failed to load image: {image_path}")
        return None, 0, 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    
    # 使用YOLOv8-Seg进行实例分割
    results = model(image_rgb, conf=0.7, classes=[0])  # COCO类0表示人，置信度阈值0.5
    
    # 获取分割结果
    person_masks = []
    if results[0].masks is not None:
        for mask, score in zip(results[0].masks.data, results[0].boxes.conf):
            # 将掩码调整为与原始图像相同的分辨率
            mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if not is_person(image_rgb, mask_resized, score.item()):  # 阈值可调，雕塑通常方差低
                continue

            person_masks.append(mask_resized)

    if not person_masks:
        # print(f"No people detected in {image_path}")
        return None, 0, 0

    # 合并所有人的掩码
    combined_mask = np.zeros((original_height, original_width), dtype=bool)
    for mask in person_masks:
        combined_mask |= mask  # 按位或操作合并掩码

    # 创建透明背景的分割结果
    combined_segmented_image = np.zeros_like(image_rgb)
    combined_segmented_image[combined_mask] = image_rgb[combined_mask]

    mask_ratio = np.sum(combined_mask) / (original_height * original_width)
    # # 保存合并后的分割结果
    # combined_output_filename = os.path.join(output_path, f"{Path(image_path).stem}_combined.png")
    # cv2.imwrite(combined_output_filename, cv2.cvtColor(combined_segmented_image, cv2.COLOR_RGB2BGR))
    # print(f"Saved combined segmented image to {combined_output_filename}")

    return combined_mask, mask_ratio, len(person_masks)  # 返回合并后的分割结果
