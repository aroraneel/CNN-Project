# Fashion-MNIST Image Classification using Convolutional Neural Networks

## Overview
This project implements a deep learning model using **Convolutional Neural Networks (CNN)** to classify clothing images from the Fashion-MNIST dataset. The goal of the project is to build a CNN architecture capable of recognizing different types of fashion items from grayscale images.

The model was developed using **TensorFlow/Keras** and trained using several optimization techniques such as **Batch Normalization, Data Augmentation, Spatial Dropout, and Learning Rate Scheduling** to improve generalization and model performance.

The final model achieved **over 94% validation accuracy** and strong performance on the Kaggle test leaderboard.

---

## Objectives
- Build a CNN model from scratch for image classification  
- Train and evaluate the model on the Fashion-MNIST dataset  
- Apply deep learning techniques to improve model accuracy  
- Analyze model performance using learning curves and confusion matrix  
- Visualize learned filters and neuron activations  

---

## Dataset
The dataset used in this project is **Fashion-MNIST**, which contains grayscale images of clothing items.

Dataset Characteristics:

| Split | Images |
|------|------|
| Training | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |
| Total | 70,000 |

Each image has:
- Size: **28 × 28 pixels**
- Channel: **1 (grayscale)**
- Classes: **10 clothing categories**

Classes include:

- T-shirt / Top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle Boot  

---

## Methodology

### Data Preprocessing
- Loaded Fashion-MNIST dataset
- Normalized pixel values
- Applied data augmentation techniques
- Split data into training, validation, and test sets

### Model Architecture
A **5-layer Convolutional Neural Network** was implemented with the following structure:

- Input Layer (28×28 grayscale image)
- Data Augmentation (Random Flip + Rotation)
- Convolution Layer 1 – 64 filters
- Max Pooling + Spatial Dropout
- Convolution Layer 2 – 128 filters
- Max Pooling + Spatial Dropout
- Convolution Layer 3 – 256 filters
- Convolution Layer 4 – 256 filters
- Max Pooling + Spatial Dropout
- Convolution Layer 5 – 256 filters
- Global Average Pooling
- Dense Layer (512 neurons)
- Output Layer (Softmax – 10 classes)

---

## Model Performance

| Metric | Value |
|------|------|
| Best Training Accuracy | 96.03% |
| Best Validation Accuracy | 94.18% |
| Validation Accuracy | 94.26% |
| Kaggle Public Score | 0.93800 |
| Kaggle Private Score | 0.93528 |

The small difference between training and validation accuracy indicates that the model generalizes well without significant overfitting.

---

## Model Evaluation
The model performance was evaluated using multiple metrics and visualization techniques including:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training vs Validation Accuracy Curve
- Training vs Validation Loss Curve

The confusion matrix analysis shows that **Trouser and Bag classes were easiest to classify**, while **Shirt was the most challenging due to visual similarity with T-shirt and Pullover**.

---

## Visualization and Interpretability

### Filter Visualization
Filters from the first convolution layer were visualized to understand the patterns learned by the network. The filters capture features such as edges, blobs, and texture patterns.

### Guided Backpropagation
Guided backpropagation was applied to visualize which pixels activate neurons in deeper convolution layers. This helps interpret how the CNN focuses on important image regions.

---

## Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

---

## Applications
- Image classification systems
- Computer vision research
- Retail product recognition
- Automated clothing categorization
- Deep learning experimentation

---

## Author
**Neel Arora**  
BCA Undergraduate | Data Science & Machine Learning Enthusiast