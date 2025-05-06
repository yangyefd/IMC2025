
import cv2
import torch
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from os.path import join
from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue
from ultralytics import YOLO
from data_process.person_mask import person_mask
from networks.loftr.loftr import LoFTR
from networks.loftr.misc import lower_config
from networks.loftr.config import get_cfg_defaults


DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def fast_make_matching_figure(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def fast_make_matching_overlay(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.line(out, (x0, y0 + sh), (x1 + margin + w0, y1 + sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale


def preprocess_loftr(image: np.ndarray, grayscale: bool = False, resize_max: int = 640, dfactor: int = 8):
    """
    预处理图像以适配 LoFTR 模型。
    - 如果图像尺寸超过 resize_max，则调整大小。
    - 确保图像尺寸可以被 dfactor 整除。

    Args:
        image (np.ndarray): 输入图像。
        grayscale (bool): 是否为灰度图像。
        resize_max (int): 最大尺寸限制。
        dfactor (int): 尺寸必须是 dfactor 的倍数。

    Returns:
        torch.Tensor: 预处理后的图像。
        np.ndarray: 缩放比例。
    """
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]  # (宽, 高)
    scale = np.array([1.0, 1.0])

    # 如果图像尺寸超过 resize_max，则调整大小
    if resize_max:
        max_dim = max(size)
        if max_dim > resize_max:
            scale = resize_max / max_dim
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    # 如果是灰度图像，调整维度
    if grayscale:
        assert image.ndim == 2, f"Expected 2D grayscale image, got shape {image.shape}"
        image = image[None]  # 添加通道维度
    else:
        image = image.transpose((2, 0, 1))  # HxWxC 转为 CxHxW

    # 转换为 PyTorch 张量并归一化到 [0, 1]
    image = torch.from_numpy(image / 255.0).float()

    # 确保图像尺寸可以被 dfactor 整除
    size_new = tuple(map(
        lambda x: int(x // dfactor * dfactor),
        image.shape[-2:]
    ))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]

    return image, scale


def compute_geom(data,
                 ransac_method=DEFAULT_RANSAC_METHOD,
                 ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                 ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
                 ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
                 ) -> dict:

    mkpts0 = data["mkpts0_f"].cpu().detach().numpy()
    mkpts1 = data["mkpts1_f"].cpu().detach().numpy()

    if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
        return {}

    h1, w1 = data["hw0_i"]

    geo_info = {}

    F, inliers = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if F is not None:
        geo_info["Fundamental"] = F.tolist()

    H, _ = cv2.findHomography(
        mkpts1,
        mkpts0,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if H is not None:
        geo_info["Homography"] = H.tolist()
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            mkpts0.reshape(-1, 2),
            mkpts1.reshape(-1, 2),
            F,
            imgSize=(w1, h1),
        )
        geo_info["H1"] = H1.tolist()
        geo_info["H2"] = H2.tolist()

    return geo_info


def wrap_images(img0, img1, geo_info, geom_type):
    img0 = img0[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]
    img1 = img1[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]

    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape

    rectified_image0 = img0
    rectified_image1 = None
    H = np.array(geo_info["Homography"])
    F = np.array(geo_info["Fundamental"])

    title = []
    if geom_type == "Homography":
        rectified_image1 = cv2.warpPerspective(
            img1, H, (img0.shape[1], img0.shape[0])
        )
        title = ["Image 0", "Image 1 - warped"]
    elif geom_type == "Fundamental":
        H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
        rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
        title = ["Image 0 - warped", "Image 1 - warped"]
    else:
        print("Error: Unknown geometry type")

    fig = plot_images(
        [rectified_image0.squeeze(), rectified_image1.squeeze()],
        title,
        dpi=300,
    )

    img = fig2im(fig)

    plt.close(fig)

    return img


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi:
        size:
        pad:
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    figsize = (size * n, size * 6 / 5) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)

    return fig


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.buffer_rgba(), dtype="u1")
    # noinspection PyArgumentList
    im = buf_ndarray.reshape(h, w, 4)
    return im


def get_padding_size(image, h, w):
    orig_width = image.shape[3]
    orig_height = image.shape[2]
    aspect_ratio = w / h

    new_width = max(orig_width, int(orig_height * aspect_ratio))
    new_height = max(orig_height, int(orig_width / aspect_ratio))

    pad_height = new_height - orig_height
    pad_width = new_width - orig_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return orig_width, orig_height, pad_left, pad_right, pad_top, pad_bottom

class Lightglue_Matcher():
    def __init__(self, device, num_features=4096):
        self.model = None
        self.detector = None
        self.device = device
                # load model
        ckpt = None
        model = None

        ckpt = 'gim_lightglue_100h.ckpt'
        checkpoints_path_loftr = './models/gim_loftr_50h.ckpt'

        detector = SuperPoint({
            'max_num_keypoints': num_features,
            'force_num_keypoints': False,
            'detection_threshold': 0.0,
            'nms_radius': 5,
            "refinement_radius": 0,
            'trainable': False,
        })
        model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })


        model_loftr = LoFTR(lower_config(get_cfg_defaults())['loftr'])

        state_dict_loft = torch.load(checkpoints_path_loftr, map_location='cpu')
        if 'state_dict' in state_dict_loft.keys(): state_dict_loft = state_dict_loft['state_dict']
        model_loftr.load_state_dict(state_dict_loft)


        # 加载YOLOv8-Seg模型
        model_yolo = YOLO("./models/yolov8n-seg.pt")  # 使用轻量级分割模型

        # weights path
        checkpoints_path = join('models', ckpt)

        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        detector.load_state_dict(state_dict)

        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        self.detector = detector.eval().to(device)
        self.model = model.eval().to(device)
        self.model_yolo = model_yolo.eval().to(device)
        self.model_loftr = model_loftr.eval().to(device)

    def loftr_extract(self, img_path0, resize=False):
        device = self.device
        image0_ori = read_image(img_path0)
        if resize:
            image0, scale0 = preprocess_loftr(image0_ori)
        else:
            image0, scale0 = preprocess(image0_ori)

        image0 = image0.to(device)[None]
        scale0 = torch.tensor(scale0).to(device)[None]

        data = {}
        data.update(dict(image0=image0))

        size0 = torch.tensor(data["image0"].shape[-2:])

        data.update(dict(size0=size0))
        data.update(dict(scale0=scale0))


        with torch.no_grad():
            feat_c0, feat_f0 = self.model_loftr.extract({
                "image0": data["image0"],
            })
        return feat_c0, feat_f0, data

    def loftr_match(self, data):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.model_loftr.forward_match(data)
        kpts0 = data['mkpts0_f']
        kpts1 = data['mkpts1_f']
        b_ids = data['m_bids']
        mconf = data['mconf']

        return mconf, kpts0, kpts1

    def loftr_refine(self, data):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.model_loftr.forward_fine(data)
        kpts0 = data['mkpts0_c']
        kpts1 = data['mkpts1_c']
        mconf = data['mconf']

        return mconf, kpts0, kpts1

    def extract(self, img_path0):
        device = self.device
        gray0 = read_image(img_path0, grayscale=True)
        gray0, scale0 = preprocess(gray0, grayscale=True)

        gray0 = gray0.to(device)[None]
        scale0 = torch.tensor(scale0).to(device)[None]

        data = {}
        data.update(dict(gray0=gray0))

        size0 = torch.tensor(data["gray0"].shape[-2:][::-1])[None]

        data.update(dict(size0=size0))
        data.update(dict(scale0=scale0))

        pred = {}
        with torch.no_grad():
            pred.update({k + '0': v for k, v in self.detector({
                "image": data["gray0"],
            }).items()})
        pred['keypoints0'] = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        pred['keypoints_refine0'] = torch.cat([kp * s for kp, s in zip(pred['keypoints_refine0'], data['scale0'][:, None])])
        
        return pred, data

    def get_person_mask(self, img_path0):
        return person_mask(img_path0, self.model_yolo)
    
    # 修改Lightglue_Matcher类，添加更新模型的方法
    def update_model(self, new_model):
        """用微调后的模型更新当前模型"""
        self.model = new_model.eval().to(self.device)

    def match(self, pred_in):
        pred = {}
        kpts0 = torch.cat([kp / s for kp, s in zip(pred_in['keypoints0'], pred_in['scale0'][:, None])])
        kpts1 = torch.cat([kp / s for kp, s in zip(pred_in['keypoints1'], pred_in['scale1'][:, None])])
        pred['keypoints0'] = kpts0[None].float().to(self.device)
        pred['keypoints1'] = kpts1[None].float().to(self.device)
        pred['descriptors0'] = pred_in['descriptors0'].float().to(self.device)
        pred['descriptors1'] = pred_in['descriptors1'].float().to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred.update(self.model({**pred, 
                                **{'image_size0': pred_in['size0'],
                                    'image_size1': pred_in['size1']}}))

        matches = pred['matches'][0]
        mconf = pred['scores'][0]
        return mconf, matches 

    def match_batch(self, pred_in):
        pred = {}
        kpts0 = pred_in['keypoints0'] / pred_in['scale0']
        kpts1 = pred_in['keypoints1'] / pred_in['scale1']
        pred['keypoints0'] = kpts0.float().to(self.device)
        pred['keypoints1'] = kpts1.float().to(self.device)
        pred['descriptors0'] = pred_in['descriptors0'].float().to(self.device)
        pred['descriptors1'] = pred_in['descriptors1'].float().to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred.update(self.model({**pred, 
                                **{'image_size0': pred_in['size0'][:,0],
                                    'image_size1': pred_in['size1'][:,0]}}))

        matches = pred['matches']
        mconf = pred['scores']
        return mconf, matches 