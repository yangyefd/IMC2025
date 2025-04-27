
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("oldufo/imc2024-packages-lightglue-rerun-kornia")

# print("Path to dataset files:", path)

# Download latest version
# path = kagglehub.dataset_download("eduardtrulls/imc25-utils")

# print("Path to dataset files:", path)
import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from lightglue import match_pair, ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel
import pycolmap
import matplotlib.pyplot as plt
sys.path.append('./imc25-utils/versions/6')
from database import *
from h5_to_db import *
import metric

# Device setup
device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def get_global_desc(fnames, device=torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('./models/dinov2-pytorch-base-v1')
    model = AutoModel.from_pretrained('./models/dinov2-pytorch-base-v1')
    model = model.eval().to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    return torch.cat(global_descs_dinov2, dim=0)

def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs

def get_image_pairs_shortlist(fnames, sim_th=0.6, min_pairs=20, exhaustive_if_less=20, 
                            device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    mask = dm <= sim_th
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
    return sorted(list(set(matching_list)))

def detect_aliked(img_fnames, feature_dir='.featureout', num_features=4096, 
                 resize_to=1024, device=torch.device('cpu')):
    dtype = torch.float32
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, 
                     resize=resize_to).eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor.extract(image0)
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

def visualize_matches(img1_path, img2_path, kpts1, kpts2, matches, save_path=None):
    """可视化两张图片的匹配结果
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径  
        kpts1: 第一张图片的特征点 (Nx2)
        kpts2: 第二张图片的特征点 (Nx2)
        matches: 匹配索引 (Mx2)
        save_path: 保存路径,如果为None则显示
    """
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 调整图片大小使其具有相同高度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = min(h1, h2)
    img1 = cv2.resize(img1, (int(w1*height/h1), height))
    img2 = cv2.resize(img2, (int(w2*height/h2), height))
    
    # 创建拼接图
    vis = np.hstack([img1, img2])
    
    # 绘制匹配线
    offset = img1.shape[1]
    for idx1, idx2 in matches:
        pt1 = tuple(map(int, kpts1[idx1]))
        pt2 = tuple(map(int, kpts2[idx2]))
        pt2 = (pt2[0] + offset, pt2[1])
        cv2.circle(vis, pt1, 2, (0,255,0), -1)
        cv2.circle(vis, pt2, 2, (0,255,0), -1)
        cv2.line(vis, pt1, pt2, (255,0,0), 1)

    # 显示或保存
    # plt.figure(figsize=(20,10))
    # plt.imshow(vis)
    # plt.axis('off')
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # plt.savefig(save_path)
        # plt.close()
    # else:
    #     plt.show()

def match_with_lightglue(img_fnames, index_pairs, feature_dir='.featureout', 
                        device=torch.device('cpu'), min_matches=15, verbose=True, visualize=True):
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                              "depth_confidence": -1,
                                              "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1, desc2,
                                       KF.laf_from_center_scale_ori(kp1[None]),
                                       KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs) == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print(f'{key1}-{key2}: {n_matches} matches')
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                                # 添加可视化
                if visualize:
                    vis_dir = os.path.join(feature_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f'{key1}_{key2}_matches.png')
                    visualize_matches(fname1, fname2, 
                                   kp1.cpu().numpy(), 
                                   kp2.cpu().numpy(),
                                   idxs.cpu().numpy(),
                                   save_path)
    return

def import_into_colmap(img_dir, feature_dir='.featureout', database_path='colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
    return

@dataclasses.dataclass
class Prediction:
    image_id: str | None
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

# Main processing
is_train = False
data_dir = '/mnt/e/yey/work/IMC2025/image-matching-challenge-2025'
workdir = './results'
os.makedirs(workdir, exist_ok=True)

sample_submission_csv = os.path.join(data_dir, 'train_labels.csv' if is_train else 'sample_submission.csv')
samples = {}
competition_data = pd.read_csv(sample_submission_csv)
for _, row in competition_data.iterrows():
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )

for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')

gc.collect()
max_images = None
datasets_to_process = None

timings = {
    "shortlisting": [],
    "feature_detection": [],
    "feature_matching": [],
    "RANSAC": [],
    "Reconstruction": []
}
mapping_result_strs = []

print(f"Extracting on device {device}")
for dataset, predictions in samples.items():
    if datasets_to_process and dataset not in datasets_to_process:
        print(f'Skipping "{dataset}"')
        continue
    
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    images = [os.path.join(images_dir, p.filename) for p in predictions]
    if max_images is not None:
        images = images[:max_images]

    print(f'\nProcessing dataset "{dataset}": {len(images)} images')
    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
    feature_dir = os.path.join(workdir, 'featureout', dataset)
    os.makedirs(feature_dir, exist_ok=True)

    try:
        t = time()
        index_pairs = get_image_pairs_shortlist(images, sim_th=0.3, min_pairs=20, 
                                              exhaustive_if_less=20, device=device)
        timings['shortlisting'].append(time() - t)
        print(f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
        gc.collect()

        t = time()
        detect_aliked(images, feature_dir, 4096, device=device)
        timings['feature_detection'].append(time() - t)
        print(f'Features detected in {time() - t:.4f} sec')
        
        t = time()
        match_with_lightglue(images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        timings['feature_matching'].append(time() - t)
        print(f'Features matched in {time() - t:.4f} sec')

        database_path = os.path.join(feature_dir, 'colmap.db')
        if os.path.isfile(database_path):
            os.remove(database_path)
        gc.collect()
        sleep(1)
        import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
        output_path = f'{feature_dir}/colmap_rec_aliked'
        
        t = time()
        pycolmap.match_exhaustive(database_path)
        timings['RANSAC'].append(time() - t)
        print(f'Ran RANSAC in {time() - t:.4f} sec')
        
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 3
        mapper_options.max_num_models = 25
        os.makedirs(output_path, exist_ok=True)
        t = time()
        maps = pycolmap.incremental_mapping(database_path=database_path, 
                                          image_path=images_dir,
                                          output_path=output_path,
                                          options=mapper_options)
        sleep(1)
        timings['Reconstruction'].append(time() - t)
        print(f'Reconstruction done in  {time() - t:.4f} sec')
        print(maps)

        registered = 0
        for map_index, cur_map in maps.items():
            for index, image in cur_map.images.items():
                prediction_index = filename_to_index[image.name]
                predictions[prediction_index].cluster_index = map_index
                predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
                predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
                registered += 1
        mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)
        gc.collect()
    except Exception as e:
        print(e)
        mapping_result_str = f'Dataset "{dataset}" -> Failed!'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)

print('\nResults')
for s in mapping_result_strs:
    print(s)

print('\nTimings')
for k, v in timings.items():
    print(f'{k} -> total={sum(v):.02f} sec.')

# Create submission file
array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')

# Compute results for training set
if is_train:
    t = time()
    final_score, dataset_scores = metric.score(
        gt_csv='/kaggle/input/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='/kaggle/input/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    print(f'Computed metric in: {time() - t:.02f} sec.')