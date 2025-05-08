import sqlite3
import numpy as np
import pycolmap
import cv2

def get_image_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM images")
    image_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return image_ids

def get_match_pairs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, data FROM matches")
    results = cursor.fetchall()
    conn.close()
    return results

def pair_id_to_image_pair(pair_id):
    # 与 colmap 源码中的 pair_id 算法一致
    image_id2 = pair_id % (2**32)
    image_id1 = (pair_id - image_id2) // (2**32)
    return image_id1, image_id2
def read_keypoints(db_path, image_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)

def read_matches(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, data FROM matches")
    pairs = cursor.fetchall()
    conn.close()
    return pairs  # 你之后可以用 pair_id_to_image_pair 解析
def pair_id_to_image_pair(pair_id):
    image_id2 = pair_id % (2 ** 31)
    image_id1 = pair_id // (2 ** 31)
    return image_id1, image_id2

def write_matches(db_path, image_id1, image_id2, matches):
    rows = matches.shape[0] 
    cols = matches.shape[1]
    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    data = matches.astype(np.uint32).tobytes()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO matches (pair_id, data, rows, cols) VALUES (?, ?, ?, ?)", (pair_id, data, rows, cols))
    conn.commit()
    conn.close()

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 + image_id2 * (2 ** 31)
def delete_matches(db_path, image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    pair_id = image_id1 + image_id2 * (2 ** 31)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
    conn.commit()
    conn.close()
# 用法示例
def delete_two_view_geometry(db_path, image_id1, image_id2):
    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
        conn.commit()
# def write_two_view_geometry(db_path, image_id1, image_id2, F, inlier_matches):
#     assert F.shape == (3, 3)
#     F = F.astype(np.float32)
#     inlier_matches = inlier_matches.astype(np.uint32)

#     pair_id = image_ids_to_pair_id(image_id1, image_id2)

#     # 写入前删除旧记录
#     delete_matches(db_path, image_id1, image_id2)
#     delete_two_view_geometry(db_path, image_id1, image_id2)

#     # 写入匹配对
#     write_matches(db_path, image_id1, image_id2, inlier_matches)

#     # 写入几何信息
#     config = 2  # pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED = 2
#     num_inliers = inlier_matches.shape[0]
#     F_blob = F.tobytes()
#     inlier_blob = inlier_matches.tobytes()

#     with sqlite3.connect(db_path) as conn:
#         conn.execute("""
#             INSERT OR REPLACE INTO two_view_geometries 
#             (pair_id, rows, cols, config, F, E, H, inlier_matches, num_inliers, score) 
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#             (pair_id, 3, 3, config, F_blob, None, None, inlier_blob, num_inliers, 0.0))
#         conn.commit()

def write_two_view_geometry(db_path, image_id1, image_id2, F, inlier_matches):
    assert F.shape == (3, 3)
    F = F.astype(np.float32)
    inlier_matches = inlier_matches.astype(np.uint32)

    pair_id = image_ids_to_pair_id(image_id1, image_id2)

    # 写入前删除旧记录
    delete_matches(db_path, image_id1, image_id2)
    delete_two_view_geometry(db_path, image_id1, image_id2)

    # 写入匹配对
    write_matches(db_path, image_id1, image_id2, inlier_matches)

  
def read_keypoints(db_path, image_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"No keypoints found for image_id {image_id}")
        data = row[0]
        keypoints = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)  # COLMAP stores 6D keypoints
        return keypoints
def read_matches_from_db(db_path):
    matches_data = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT pair_id, data FROM matches")
        for pair_id, blob in cursor.fetchall():
            if blob is None:
                continue  # 跳过空值
            match_array = np.frombuffer(blob, dtype=np.uint32).reshape(-1, 2)
            matches_data.append((pair_id, match_array))
    return matches_data

def prosac_lo_ransac_fundamental(points1, points2, confidence_scores, threshold=1.0, max_iter=1000):
 
    # Step 1: 按置信分数进行排序
    sorted_indices = np.argsort(-confidence_scores)  # 从高到低排序
    points1_sorted = points1[sorted_indices]
    points2_sorted = points2[sorted_indices]

    best_fundamental_matrix = None
    best_inliers_mask = None

    for i in range(max_iter):
        # Step 2: 实现 PROSAC 优先级采样
        sample_size = min(8 + i // 10, len(points1_sorted))  # 动态扩展采样范围
        sampled_indices = np.random.choice(sample_size, 8, replace=False)  # 基础矩阵至少需要 8 个点

        sampled_points1 = points1_sorted[sampled_indices]
        sampled_points2 = points2_sorted[sampled_indices]

        # Step 3: 使用 OpenCV 的 RANSAC 方法估算基础矩阵
        fundamental_matrix, mask = cv2.findFundamentalMat(sampled_points1, sampled_points2, cv2.FM_RANSAC, threshold)

        # Step 4: 如果拟合成功，计算内点数量
        if fundamental_matrix is not None:
            errors = compute_reprojection_error(points1_sorted, points2_sorted, fundamental_matrix)
            inliers_mask = errors < threshold
            inliers = np.sum(inliers_mask)

            # Step 5: 保存最优基础矩阵
            if best_inliers_mask is None or inliers > np.sum(best_inliers_mask):
                best_fundamental_matrix = fundamental_matrix
                best_inliers_mask = inliers_mask

    # Step 6: LO-RANSAC 局部优化：使用内点重新拟合模型
    if best_inliers_mask is not None:
        refined_points1 = points1_sorted[best_inliers_mask]
        refined_points2 = points2_sorted[best_inliers_mask]
        best_fundamental_matrix, _ = cv2.findFundamentalMat(refined_points1, refined_points2, cv2.FM_RANSAC)

    return best_fundamental_matrix, best_inliers_mask

def compute_reprojection_error(points1, points2, fundamental_matrix):
    """
    计算点对在基础矩阵拟合上的重投影误差
    """
    points1_homo = np.hstack((points1, np.ones((points1.shape[0], 1))))  # 转为齐次坐标
    points2_homo = np.hstack((points2, np.ones((points2.shape[0], 1))))  # 转为齐次坐标

    # 计算重投影误差：对称距离
    errors1 = np.abs(np.sum(points2_homo * np.dot(fundamental_matrix, points1_homo.T).T, axis=1))
    errors2 = np.abs(np.sum(points1_homo * np.dot(fundamental_matrix.T, points2_homo.T).T, axis=1))
    errors = errors1 + errors2

    return errors
