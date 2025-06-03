# IMC2025 图像匹配挑战解决方案

## 概述

这是一个用于IMC2025（Image Matching Challenge 2025）的完整解决方案，实现了基于深度学习的图像匹配和三维重建pipeline。该解决方案结合了多种先进的特征检测和匹配算法，取得了第八名的成绩。

## 主要特性

### 特征检测
- **SuperPoint + ALIKED集成**: 结合SuperPoint和ALIKED特征检测器，提取更丰富的特征点。其中superpoint采用原图尺寸，当图像大于4096时resize到4096，alike则统一resize到1024.

### 特征匹配
- **LightGlue匹配器**: 基于学习的特征匹配算法
- **二次匹配策略**: 对初次匹配结果进行聚类分析和区域扩展匹配
- **匹配过滤**: 使用图论方法和循环一致性检查过滤错误匹配

### 图像检索和对选择
- **CLIP特征**: 使用CLIP模型进行图像全局特征提取
- **相似度阈值**: 基于余弦相似度进行图像对筛选

### 三维重建
- **COLMAP集成**: 使用COLMAP进行增量式三维重建
- **多次重建比较**: 自动进行多次重建并选择最优结果
- **重建质量评估**: 基于注册图像数、轨迹长度、重投影误差等指标评估重建质量

## 环境依赖

```bash
pip install torch torchvision torchaudio
pip install kornia
pip install lightglue
pip install transformers
pip install opencv-python
pip install pycolmap
pip install scikit-learn
pip install clip-by-openai
```

## 项目结构

```
IMC2025/
├── main_test_lightglue.py          # 主程序文件
├── GIMlightglue_match.py           # LightGlue匹配器实现
├── fine_tune_lightglue.py          # LightGlue微调模块
├── CLIP/                           # CLIP模型相关
├── data_process/                   # 数据处理模块
├── models/                         # 预训练模型
├── results/                        # 结果输出目录
└── imc25-utils/                    # IMC2025工具包
```

## 使用方法

### 1. 数据准备
将IMC2025数据集放置在以下目录结构：
```
../image-matching-challenge-2025/
├── train/
│   ├── ETs/
│   ├── stairs/
│   └── ...
└── test/
```

### 2. 模型准备
下载所需的预训练模型：
- DINOv2模型：`./models/dinov2-pytorch-base-v1`
- CLIP模型：`./models/ViT-B-32.pt`
- gimlightglue：`./models/gim_lightglue_100h.ckpt`

### 3. 关键参数配置

```python
# 设备配置
device = K.utils.get_cuda_device_if_available(0)

# 特征检测参数
num_features = 4096      # 最大特征点数
resize_to = 1024         # 图像resize尺寸

# 匹配参数
sim_th = 0.76           # 图像相似度阈值
min_pairs = 1           # 最小匹配对数
min_matches = 20        # 最小匹配点数

# 批处理参数
batch_size = 4          # 批处理大小
tok_limit = 1200        # 最大token限制
```

## 算法流程

### 1. 图像对选择
- 使用CLIP提取全局特征
- 计算图像间余弦相似度
- 基于相似度阈值选择候选图像对

### 2. 特征检测
- SuperPoint检测结构化特征点
- ALIKED检测补充特征点
- 特征点数量控制和优化

### 3. 特征匹配
- LightGlue进行初始匹配
- 批处理提高匹配效率
- 基于聚类的二次匹配
- 匹配点NMS去重

### 4. 匹配过滤
- 图论一致性检查
- 循环一致性验证
- 基于几何约束的过滤

### 5. 三维重建
- COLMAP数据库构建
- 增量式重建
- 多次重建比较
- 最优结果选择

## 性能优化

### GPU加速
- 特征检测和匹配全程GPU加速
- 批处理减少GPU-CPU数据传输
- 内存管理和垃圾回收

### 算法优化
- 自适应参数调整
- 智能图像对选择
- 多阶段匹配策略
- 并行处理支持

## 评估指标

- **注册图像比例**: 成功重建的图像数量占比
- **轨迹长度**: 3D点的平均观测次数
- **重投影误差**: 重建精度指标
- **聚类数量**: 重建场景的连通性

## 链接

1. gimlightglue：https://github.com/xuelunshen/gim
2. clip：https://github.com/openai/CLIP

## 许可证

本项目采用MIT许可证。