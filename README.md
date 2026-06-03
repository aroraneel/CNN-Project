# Fashion AI — CNN Clothing Classifier

A complete end-to-end deep learning project that classifies clothing items using a Convolutional Neural Network trained on the Fashion-MNIST dataset.

## 🚀 Live Demo

**[fashion-ai-mnist.streamlit.app](https://fashion-ai-mnist.streamlit.app)**

---

## 📌 Overview

This project builds a CNN from scratch, trains it on 60,000 Fashion-MNIST images, and deploys it as an interactive web application. Users can draw a clothing sketch, upload a photo, or pick a sample image — the model predicts what clothing item it is in real time.

---

## 🧠 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.03% |
| Validation Accuracy | 94.26% |
| Kaggle Public Score | 0.938 |
| Kaggle Private Score | 0.935 |

---

## 🏗 Model Architecture

| Layer | Details |
|-------|---------|
| Input | 28 × 28 grayscale image |
| Data Augmentation | Random Flip + Rotation |
| Conv Block 1 | 64 filters, 3×3, BatchNorm, MaxPool, SpatialDropout |
| Conv Block 2 | 128 filters, 3×3, BatchNorm, MaxPool, SpatialDropout |
| Conv Block 3 | 256 filters, 3×3, BatchNorm |
| Conv Block 4 | 256 filters, 3×3, BatchNorm, MaxPool, SpatialDropout |
| Conv Block 5 | 256 filters, 3×3, BatchNorm |
| Global Average Pooling | — |
| Dense | 512 neurons, ReLU, Dropout |
| Output | 10 classes, Softmax |

**Training:** EarlyStopping + ReduceLROnPlateau, batch size 512, up to 70 epochs

---

## 👗 10 Classes

| # | Class |
|---|-------|
| 0 | 👕 T-shirt/Top |
| 1 | 👖 Trouser |
| 2 | 🧥 Pullover |
| 3 | 👗 Dress |
| 4 | 🧥 Coat |
| 5 | 👡 Sandal |
| 6 | 👔 Shirt |
| 7 | 👟 Sneaker |
| 8 | 👜 Bag |
| 9 | 👢 Ankle Boot |

---

## 📁 Project Structure

```
end-to-end-fashion-mnist/
├── data/
│   └── raw/                    # Raw dataset files
├── logs/                       # Training logs
├── models/
│   └── fashion_cnn_model.h5    # Trained model weights
├── notebooks/
│   └── project.ipynb           # Training notebook
├── src/
│   ├── __init__.py
│   └── predict.py              # Prediction helper functions
├── .streamlit/
│   └── config.toml             # App theme config
├── app.py                      # Streamlit web app
├── Dockerfile                  # Docker deployment config
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

---

## 🌐 App Features

- **3 Pages** — Home, Classify, Result
- **3 Input Modes** — Draw on canvas, Upload photo, Pick sample image
- **Real-time prediction** with confidence scores
- **All 10 class probabilities** shown as bar chart
- Fully responsive — works on laptop and mobile

---

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/aroraneel/end-to-end-fashion-mnist.git
cd end-to-end-fashion-mnist

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🐳 Run with Docker

```bash
docker build -t fashion-ai .
docker run -p 8501:8501 fashion-ai
```

Then open: **http://localhost:8501**

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model training |
| Streamlit | Web application |
| streamlit-drawable-canvas | Drawing input |
| Pillow | Image processing |
| NumPy | Data manipulation |
| Python 3.11 | Runtime |

---

## 📊 Dataset

**Fashion-MNIST** by Zalando Research

- 60,000 training images
- 10,000 test images
- 28×28 grayscale
- 10 clothing categories

---

## 👤 Author

**Neel Arora** — BCA Undergraduate | Data Science & Machine Learning