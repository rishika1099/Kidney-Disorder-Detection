# 🏥 Deep Learning based Auto-Diagnosis of Kidney Disorders

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Motivation](#-motivation)
- [Features](#-features)
- [Results](#-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Future Work](#-future-work)
- [References](#-references)
- [Team](#-team)

---

## 🔍 Overview

This project leverages **Deep Learning** and **Transfer Learning** techniques to automate the diagnosis of kidney disorders from CT scan images. Our Modified VGG19 model achieves an impressive **99.2% accuracy** in classifying kidney abnormalities into four categories:

- 🟢 **Normal** - Healthy kidney
- 🔵 **Cyst** - Fluid-filled sacs
- 🟡 **Stone** - Kidney stones (Nephrolithiasis)
- 🔴 **Tumor** - Abnormal cell growth

---

## 💡 Motivation

The global healthcare system faces a critical challenge:
- 🏥 **Overwhelming patient load** on nephrologists
- ⏰ **Long waiting times** for diagnosis
- 🌍 **Limited nephrologist workforce** worldwide
- ⚠️ **Delayed diagnosis** can lead to irreversible damage

Our solution: **Automate the detection process** to reduce workload on doctors and provide faster, more accurate diagnoses for patients.

---

## ✨ Features

- 🤖 **Automated Classification** - Classifies CT scans into 4 categories (Normal, Stone, Cyst, Tumor)
- 🎯 **High Accuracy** - 99.2% accuracy with 99% precision and recall
- ⚡ **Fast Processing** - Quick analysis of medical images
- 🔄 **Transfer Learning** - Leverages pre-trained VGG19 and ResNet50 models
- 📊 **Comprehensive Metrics** - ROC curves, AUC scores, confusion matrices
- 🖼️ **Image Processing** - Advanced preprocessing including grayscale conversion and augmentation

---

## 📈 Results

### 🏆 Model Performance

| Model | Accuracy | Precision | Recall | AUC Score |
|-------|----------|-----------|--------|-----------|
| **Modified VGG19** | **99.2%** | **99%** | **99%** | **0.992** |
| Modified ResNet50 | 98.5% | 99% | 98% | 0.988 |

### 📊 Classification Report (Modified VGG19)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cyst | 0.99 | 0.99 | 0.99 | 801 |
| Normal | 0.99 | 0.99 | 0.99 | 801 |
| Stone | 0.98 | 0.99 | 0.99 | 795 |
| Tumor | 1.00 | 1.00 | 1.00 | 801 |

### 📉 Key Insights

✅ **No Overfitting** - Training and validation loss curves converge smoothly  
✅ **Balanced Performance** - High metrics across all classes  
✅ **Clinical Reliability** - Suitable for real-world medical applications

---

## 🏗️ Architecture

### Modified VGG19 Architecture

```
Input (224×224×3)
    ↓
[VGG19 Pre-trained Layers - Frozen]
    ↓ (16 Conv Layers + 5 MaxPool)
    ↓
Flatten Layer
    ↓
Dense(4096) + ReLU + Dropout(0.5)
    ↓
Dense(1024) + ReLU + Dropout(0.5)
    ↓
Dense(4) + Softmax
    ↓
Output [Normal, Cyst, Stone, Tumor]
```

### Key Components:
- 🧱 **16 Convolutional Layers** - Feature extraction
- 🏊 **5 MaxPooling Layers** - Dimensionality reduction
- 🎲 **Dropout Layers** - Prevents overfitting
- 🎯 **Softmax Activation** - Multi-class classification

---

## 📦 Dataset

**Source:** [CT Kidney Dataset on Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

### Dataset Statistics:
- 🖼️ **Image Type:** CT Scan (Abdomen & Urogram)
- 📊 **Classes:** 4 (Normal, Cyst, Stone, Tumor)
- 🔄 **Augmentation:** Applied to balance dataset
- 🎨 **Preprocessing:** Grayscale conversion, normalization, resizing to 224×224

---

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.7 or above
python --version

# Minimum Hardware Requirements
# RAM: 8GB or above
# CPU: Intel Core i3 or above
```

### Setup

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/kidney-disorder-diagnosis.git
cd kidney-disorder-diagnosis
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Required Libraries**
```bash
pip install tensorflow==2.x
pip install pandas numpy
pip install pillow
pip install matplotlib seaborn
pip install scikit-learn
```

---

## 🚀 Usage

### Training the Model

```python
# Open the Jupyter Notebook
jupyter notebook VGG19.ipynb

# Or for ResNet50
jupyter notebook ResNet50.ipynb
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('modified_vgg19.h5')

# Load and preprocess image
img = Image.open('path/to/ct_scan.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']
result = classes[np.argmax(prediction)]

print(f"Predicted: {result}")
```

---

## 📊 Performance Metrics

### 🎯 Accuracy Comparison with Existing Methods

#### Kidney Stone Detection
| Model | Accuracy | Recall | Precision |
|-------|----------|--------|-----------|
| **Our VGG19** | **99.2%** | **99%** | **99%** |
| ResNet50 [14] | 98% | 100% | 96.5% |
| XResNet101 [14] | 98% | 98.8% | 97.6% |
| VGG19 [14] | 98% | 99.4% | 97.6% |
| SVM [18] | 97.7% | 95.7% | 100% |

#### Kidney Abnormality Classification
| Model | Accuracy | Recall | Precision |
|-------|----------|--------|-----------|
| **Our VGG19** | **99.2%** | **99%** | **99%** |
| Fuzzy KNN [17] | 96.68% | 98.4% | 95.8% |
| GOA + ANN [6] | 95.83% | 91.66% | 97.22% |

### 📉 Loss Curves
- ✅ Training loss flattens to stability
- ✅ Minimal gap between training and validation loss
- ✅ No signs of overfitting

---

## 🔮 Future Work

- 🌐 **Larger Dataset** - Validate on more diverse patient data
- 👥 **Demographic Features** - Incorporate age, gender, ethnicity
- 🧬 **Genetic Information** - Add genetic sequence features
- 🏥 **Clinical Integration** - Deploy in real hospital systems
- 📱 **Mobile App** - Develop user-friendly mobile interface
- 🔗 **MRI Scanner Integration** - Direct connection for instant diagnosis

---

## 📁 Project Structure

```
kidney-disorder-diagnosis/
│
├── VGG19.ipynb                 # VGG19 model training notebook
├── ResNet50.ipynb              # ResNet50 model training notebook
├── Capstone_Report.pdf         # Detailed project report
├── Capstone_Poster.pdf         # Project poster
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── data/                       # Dataset directory
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/                     # Saved models
│   ├── modified_vgg19.h5
│   └── modified_resnet50.h5
│
└── results/                    # Results and visualizations
    ├── confusion_matrices/
    ├── roc_curves/
    └── loss_curves/
```
---

