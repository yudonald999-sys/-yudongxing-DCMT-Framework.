# DCMT-Framework

**Dynamic Cross-Modal Tokenization: Integrating Human Chunking Mechanisms into Multimodal LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This repository contains the implementation code, evaluation scripts, and dataset for the paper:

> **Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs**  
> Dongxing Yu  
> School of Education, Sanda University  
> *Information* (2025)

The Dynamic Cross-Modal Tokenization (DCMT) framework introduces adaptive boundary detection, hierarchical representations, and cross-modal alignment mechanisms inspired by human cognitive chunking processes.

## Key Features

- **Adaptive Boundary Detection**: Context-sensitive token boundary adjustment based on semantic coherence
- **Hierarchical Representation**: Multi-level transformer encoders with bidirectional connections
- **Cross-Modal Alignment**: Contrastive learning objectives for visual-linguistic correspondence
- **CMCE Dataset**: Cross-Modal Chunking Evaluation benchmark for assessing multimodal integration

## Repository Structure

```
DCMT-Framework/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── data/
│   └── CMCE/                   # Cross-Modal Chunking Evaluation dataset
│       ├── README.md           # Dataset documentation
│       ├── train.json          # Training split
│       ├── val.json            # Validation split
│       ├── test.json           # Test split
│       └── annotations/        # Human annotations
├── src/
│   ├── models/
│   │   ├── dcmt.py             # Main DCMT model
│   │   ├── boundary_detector.py # Adaptive boundary detection
│   │   ├── hierarchical.py     # Hierarchical representation
│   │   └── alignment.py        # Cross-modal alignment
│   ├── utils/
│   │   ├── tokenizer.py        # Dynamic tokenization utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── data_loader.py      # Data loading utilities
│   └── training/
│       ├── train.py            # Training script
│       └── config.py           # Training configurations
├── evaluation/
│   ├── evaluate.py             # Main evaluation script
│   ├── benchmark_vqa.py        # VQA evaluation
│   ├── benchmark_gqa.py        # GQA evaluation
│   └── human_alignment.py      # Human-model alignment analysis
├── measurements/
│   ├── README.md               # Measurement protocols
│   ├── eye_tracking.py         # Eye-tracking analysis
│   ├── chunking_metrics.py     # Chunking measurement tools
│   └── statistical_tests.py    # Statistical analysis scripts
├── supplementary/
│   ├── statistical_details.csv # Raw statistical values
│   └── model_outputs.json      # Model prediction samples
├── figures/
│   └── ...                     # Publication figures
└── docs/
    ├── INSTALLATION.md         # Detailed installation guide
    ├── USAGE.md                # Usage examples
    └── REPRODUCTION.md         # Steps to reproduce results
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU support)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yudongxing/DCMT-Framework.git
cd DCMT-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Usage

### Training

```bash
python src/training/train.py \
    --config configs/dcmt_base.yaml \
    --data_path data/CMCE \
    --output_dir outputs/
```

### Evaluation

```bash
# Evaluate on CMCE benchmark
python evaluation/evaluate.py \
    --model_path checkpoints/dcmt_best.pt \
    --dataset CMCE \
    --output results/

# Evaluate on VQA v2
python evaluation/benchmark_vqa.py \
    --model_path checkpoints/dcmt_best.pt
```

### Inference

```python
from src.models.dcmt import DCMTModel

# Load model
model = DCMTModel.from_pretrained("checkpoints/dcmt_best.pt")

# Process image-text pair
image = load_image("path/to/image.jpg")
text = "A red car parked next to a person"

# Get predictions with dynamic tokenization
output = model(image, text)
print(output.chunks)  # Cross-modal chunk boundaries
print(output.alignment)  # Visual-textual alignment scores
```

## Model Parameters

| Component             | Parameter          | Value   | Description                  |
| --------------------- | ------------------ | ------- | ---------------------------- |
| Base Transformer      | d_model            | 768     | Model dimensionality         |
|                       | n_heads            | 12      | Number of attention heads    |
|                       | n_layers           | 12      | Number of transformer layers |
|                       | d_ff               | 3072    | Feed-forward dimension       |
| Vision Encoder        | patch_size         | 16×16   | Image patch dimensions       |
|                       | img_size           | 224×224 | Input image dimensions       |
| Adaptive Tokenization | boundary_threshold | 0.5     | Boundary detection threshold |
|                       | learning_rate      | 1e-4    | Adam optimizer learning rate |

## Results

### Performance on Benchmarks

| Model           | VQA (%)  | Complex Scene (%) | GQA (%)  | CMCE (%) |
| --------------- | -------- | ----------------- | -------- | -------- |
| BLIP-2          | 78.3     | 69.4              | 63.7     | 58.9     |
| Flamingo        | 80.1     | 72.6              | 65.2     | 62.3     |
| GPT-4V          | 86.5     | 79.8              | 72.4     | 68.7     |
| **DCMT (Ours)** | **94.3** | **85.1**          | **77.9** | **82.4** |

### Statistical Significance

| Task                      | Improvement | p-value | Cohen's d |
| ------------------------- | ----------- | ------- | --------- |
| Visual Question Answering | +7.8%       | p<0.001 | 0.82      |
| Complex Scene Description | +5.3%       | p<0.01  | 0.65      |
| Cross-modal Retrieval     | +4.7%       | p<0.001 | 0.71      |
| Visual Reasoning          | +5.7%       | p<0.001 | 0.78      |

## CMCE Dataset

The Cross-Modal Chunking Evaluation (CMCE) dataset contains 10,000 image-text pairs with human-annotated chunk boundaries. See [data/CMCE/README.md](data/CMCE/README.md) for details.

### Dataset Statistics

- **Total samples**: 10,000 image-text pairs
- **Training/Validation/Test split**: 7,000 / 1,500 / 1,500
- **Average chunks per image**: 4.2 ± 1.3
- **Average chunks per text**: 3.8 ± 1.1
- **Cross-modal alignment rate**: 72%
- **Inter-annotator agreement (Fleiss' κ)**: 0.78

## Citation

```bibtex
@article{yu2025adaptive,
  title={Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs},
  author={Yu, Dongxing},
  journal={Information},
  year={2025},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Dongxing Yu
- **Email**: yudongxing@sandau.edu.cn
- **Institution**: School of Education, Sanda University, Shanghai, China

## Acknowledgments

This work was supported by the Shanghai Municipal Education Commission Educational Science Planning (Grant No. C2023035) for the project "Theoretical Construction and Exploratory Application of the Educational Metaverse".
