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

### 5. 三维重建
- COLMAP数据库构建
- 增量式重建
- 多次重建比较
- 最优结果选择

## 链接

1. gimlightglue：https://github.com/xuelunshen/gim
2. clip：https://github.com/openai/CLIP

## 作者

本项目由 [yangye] 开发完成。

## 许可证

本项目采用Apache License 2.0许可证。详情请参见[LICENSE](LICENSE)文件。

```
Copyright 2025 [yangye]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```