# IMC2025 Image Matching Challenge Solution

[中文 README](readme.zh.md) | [English README](readme.md)
## Overview

This is a complete solution for IMC2025 (Image Matching Challenge 2025), implementing a deep learning-based image matching and 3D reconstruction pipeline. The solution combines multiple state-of-the-art feature detection and matching algorithms, achieving 8th place in the competition.

## Key Features

### Feature Detection
- **SuperPoint + ALIKED Integration**: Combines SuperPoint and ALIKED feature detectors to extract richer feature points. SuperPoint uses original image size (resized to 4096 when larger), while ALIKED uniformly resizes to 1024.

### Feature Matching
- **LightGlue Matcher**: Learning-based feature matching algorithm
- **Two-stage Matching Strategy**: Performs clustering analysis and region extension matching on initial results
- **Match Filtering**: Uses graph theory methods and cycle consistency checks to filter incorrect matches

### Image Retrieval and Pair Selection
- **CLIP Features**: Uses CLIP model for global image feature extraction
- **Similarity Threshold**: Image pair selection based on cosine similarity

### 3D Reconstruction
- **COLMAP Integration**: Uses COLMAP for incremental 3D reconstruction
- **Multiple Reconstruction Comparison**: Automatically performs multiple reconstructions and selects optimal results
- **Reconstruction Quality Assessment**: Evaluates reconstruction quality based on registered image count, track length, reprojection error, etc.

## Dependencies

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

## Project Structure

```
IMC2025/
├── main_test_lightglue.py          # Main program file
├── GIMlightglue_match.py           # LightGlue matcher implementation
├── fine_tune_lightglue.py          # LightGlue fine-tuning module
├── CLIP/                           # CLIP model related
├── data_process/                   # Data processing modules
├── models/                         # Pre-trained models
├── results/                        # Output directory
└── imc25-utils/                    # IMC2025 utilities
```

## Usage

### 1. Data Preparation
Place the IMC2025 dataset in the following directory structure:
```
../image-matching-challenge-2025/
├── train/
│   ├── ETs/
│   ├── stairs/
│   └── ...
└── test/
```

### 2. Model Preparation
Download the required pre-trained models:
- DINOv2 model: `./models/dinov2-pytorch-base-v1`
- CLIP model: `./models/ViT-B-32.pt`
- gimlightglue: `./models/gim_lightglue_100h.ckpt`

### 3. Key Parameter Configuration

```python
# Device configuration
device = K.utils.get_cuda_device_if_available(0)

# Feature detection parameters
num_features = 4096      # Maximum number of features
resize_to = 1024         # Image resize dimension

# Matching parameters
sim_th = 0.76           # Image similarity threshold
min_pairs = 1           # Minimum number of pairs
min_matches = 20        # Minimum number of matches

# Batch processing parameters
batch_size = 4          # Batch size
tok_limit = 1200        # Maximum token limit
```

## Algorithm Pipeline

### 1. Image Pair Selection
- Extract global features using CLIP
- Calculate cosine similarity between images
- Select candidate image pairs based on similarity threshold

### 2. Feature Detection
- SuperPoint detects structured feature points
- ALIKED detects complementary feature points
- Feature point count control and optimization

### 3. Feature Matching
- Initial matching with LightGlue
- Batch processing for improved matching efficiency
- Clustering-based second-stage matching
- Match point NMS deduplication

### 4. Match Filtering
- Graph consistency checks
- Cycle consistency verification

### 5. 3D Reconstruction
- COLMAP database construction
- Incremental reconstruction
- Multiple reconstruction comparison
- Optimal result selection

## Links

1. gimlightglue: https://github.com/xuelunshen/gim
2. clip: https://github.com/openai/CLIP

## License

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