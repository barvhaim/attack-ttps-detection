# 🎯 MITRE ATT&CK TTPs Classification

## 🚀 Installation

### 📋 Prerequisites
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and environment manager ⚡
- Python 3.11 or higher 🐍

### ⚙️ Setup

1. **Install uv**:
   https://github.com/astral-sh/uv?tab=readme-ov-file#installation

2. **Download pre-trained models**:
   ```bash
   mkdir -p scibert_multi_label_model
   wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/config.json -O scibert_multi_label_model/config.json
   wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/pytorch_model.bin -O scibert_multi_label_model/pytorch_model.bin
   ```

## 🏃‍♂️ Usage

### 🏋️ Training

To train the model:
```bash
uv run train.py
```

### 🔮 Prediction

To make predictions using a trained model:
```bash
uv run predict.py
```

