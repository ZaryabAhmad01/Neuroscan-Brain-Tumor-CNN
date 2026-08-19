<div align="center">

# 🧠 Brain Tumor Detection using CNN

### Deep Learning based MRI Classification System

Classifies brain MRI scans into **4 categories** — Glioma, Meningioma, Pituitary Tumor, and No Tumor — using a Convolutional Neural Network built from scratch in TensorFlow/Keras.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#-license)

<img src="https://img.shields.io/badge/Test%20Accuracy-89.00%25-success?style=flat-square" />
<img src="https://img.shields.io/badge/Classes-4-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Params-5.1M-orange?style=flat-square" />

</div>

---

## 📌 Overview

This project uses a custom **4-block Convolutional Neural Network** to classify brain MRI scans into four categories. It covers the full deep learning pipeline — from loading and preprocessing image data, to training, evaluating, and running inference on new MRI images, with clear visual diagnostics (accuracy/loss curves, confusion matrix, per-class metrics, and sample prediction grids).

> ⚠️ **Disclaimer:** This project is built for educational and research purposes only. It is **not** a certified medical diagnostic tool and should never be used as a substitute for professional medical advice.

---

## 🗂️ Dataset

The model is trained on an MRI brain tumor dataset organized into `Training` and `Testing` folders, inferred automatically from directory structure.

| Split | Images | Classes |
|---|---|---|
| Training | 5,600 | glioma, meningioma, notumor, pituitary |
| Testing / Validation | 1,600 | glioma, meningioma, notumor, pituitary |

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Images are resized to **224 × 224** and pixel values normalized to **[0, 1]**.

---
<img width="634" height="417" alt="image" src="https://github.com/user-attachments/assets/491aa09f-f96b-455c-9db1-69ed14b6c698" />
<img width="270" height="80" alt="image" src="https://github.com/user-attachments/assets/3e9429db-7476-4a78-85ec-336f102312fa" />
<img width="301" height="102" alt="image" src="https://github.com/user-attachments/assets/830372a6-8169-4798-8abe-ca7c77bf17a6" />
<img width="1237" height="489" alt="image" src="https://github.com/user-attachments/assets/26939905-7bb0-4302-ba24-6bd13e4daff3" />
<img width="461" height="239" alt="image" src="https://github.com/user-attachments/assets/31ab7323-892a-417d-a3f6-cb25a0bc0d28" />

## 🏗️ Model Architecture

A sequential CNN with 4 convolutional blocks of increasing depth, followed by a dense classifier head with dropout regularization.

```
Input (224, 224, 3)
        │
   Conv2D(32, 3x3) → ReLU → MaxPooling2D(2x2)
        │
   Conv2D(64, 3x3) → ReLU → MaxPooling2D(2x2)
        │
   Conv2D(128, 3x3) → ReLU → MaxPooling2D(2x2)
        │
   Conv2D(256, 3x3) → ReLU → MaxPooling2D(2x2)
        │
      Flatten
        │
   Dropout(0.5)
        │
   Dense(128, ReLU)
        │
   Dropout(0.3)
        │
   Dense(4, Softmax)
```

| Property | Value |
|---|---|
| Total Parameters | 5,107,652 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |
| Epochs | 20 |
| Batch Size | 32 |

---

## 📊 Results

### Training Performance

| Metric | Final Value |
|---|---|
| Training Accuracy | 98.45% |
| Validation Accuracy | 89.00% |
| Test Accuracy | **89.00%** |
| Test Loss | 1.19 |

<div align="center">
<i>Training vs Validation Accuracy/Loss curves show strong convergence with mild overfitting after ~epoch 10.</i>
</div>

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Glioma | 0.89 | 0.78 | 0.83 | 400 |
| Meningioma | 0.84 | 0.89 | 0.86 | 400 |
| No Tumor | 0.89 | 0.99 | 0.94 | 400 |
| Pituitary | 0.94 | 0.90 | 0.92 | 400 |
| **Weighted Avg** | **0.89** | **0.89** | **0.89** | 1600 |

### Per-Class Accuracy

| Class | Accuracy |
|---|---|
| No Tumor | 🟩 99.50% |
| Pituitary | 🟩 89.75% |
| Meningioma | 🟨 88.50% |
| Glioma | 🟨 78.25% |

### Confusion Matrix

```
                Predicted
             glio  meni  notu  pitu
Actual glio   313    41    35    11
       meni    20   354    13    13
       notu     0     2   398     0
       pitu    19    22     0   359
```

The model is most confident distinguishing "No Tumor" cases (99.5% recall) and shows the most confusion between **glioma** and **meningioma** — a pattern that mirrors the real diagnostic difficulty clinicians face with these two tumor types.

---

## 🔍 Sample Inference Output

```
==================================================
  🧠 BRAIN TUMOR DETECTION RESULT
==================================================
  ✅ Tumor detected: NO
  🏷️  Classification: HEALTHY
  🎯 Confidence: 100.0%

  📊 Detailed Probabilities:
     glioma      : 0.0%
     meningioma  : 0.0%
     notumor     : 100.0% ████████████████████
     pituitary   : 0.0%
==================================================
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/brain-tumor-cnn.git
cd brain-tumor-cnn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.15.0
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Usage

1. **Download the dataset** and place it in the following structure:
   ```
   dataset/Training/<class_name>/*.jpg
   dataset/Testing/<class_name>/*.jpg
   ```

2. **Open and run the notebook:**
   ```bash
   jupyter notebook Brain_Tumor_cnn_02.ipynb
   ```

3. **Run all cells** in order to:
   - Load & preprocess the MRI dataset
   - Build and compile the CNN
   - Train the model for 20 epochs
   - Evaluate on the test set (accuracy, confusion matrix, classification report)
   - Run predictions on sample/new MRI images

---

## 📁 Project Structure

```
brain-tumor-cnn/
├── Brain_Tumor_cnn_02.ipynb   # Main notebook (data → training → evaluation → inference)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── assets/                    # (optional) sample images, plots for README
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data Handling | `image_dataset_from_directory`, `tf.data` pipeline |
| Evaluation | scikit-learn (classification report, confusion matrix) |
| Visualization | Matplotlib, Seaborn |

---

## 🔮 Future Improvements

- [ ] Use transfer learning (ResNet50 / EfficientNet / VGG16) to boost glioma/meningioma separation
- [ ] Add data augmentation to reduce overfitting seen after epoch 10
- [ ] Add Grad-CAM visualizations to highlight tumor regions
- [ ] Deploy as a Streamlit / Flask web app for interactive predictions
- [ ] Convert to TFLite for lightweight/edge deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](../../issues) or open a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

Made with 🧠 and TensorFlow

</div>
