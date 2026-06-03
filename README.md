# 👗 Fashion MNIST CNN Classifier

A Convolutional Neural Network that classifies clothing images from the Fashion-MNIST dataset. Draw a sketch, upload a photo, or pick a sample — the model predicts what clothing item it is!

## 🚀 Live Demo
[Click here to try it]() ← add your Streamlit Cloud URL after deployment

## 📁 Project Structure

    CNN Project/
    ├── data/
    │   └── raw/
    ├── logs/
    ├── models/
    │   └── fashion_cnn_model.h5
    ├── notebooks/
    │   └── project.ipynb
    ├── src/
    │   ├── __init__.py
    │   └── predict.py
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    ├── setup.py
    └── .streamlit/
        └── config.toml

## 🧠 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.03% |
| Validation Accuracy | 94.26% |
| Kaggle Public Score | 0.938 |
| Kaggle Private Score | 0.935 |

## 🏃 Run Locally

    git clone https://github.com/aroraneel/CNN-Project.git
    cd CNN-Project
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## 🐳 Run with Docker

    docker build -t fashion-mnist-classifier .
    docker run -p 8501:8501 fashion-mnist-classifier

Then open: http://localhost:8501

## 📦 10 Classes
T-shirt/Top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle Boot

## 🛠 Tech Stack
- TensorFlow / Keras
- Streamlit
- Python 3.11

## 👤 Author
**Neel Arora** — BCA Undergraduate | Data Science & ML Enthusiast