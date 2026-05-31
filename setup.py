from setuptools import setup, find_packages

setup(
    name="fashion_mnist_classifier",
    version="1.0.0",
    author="Neel Arora",
    description="Fashion MNIST Image Classification using CNN",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit==1.35.0",
        "tensorflow-cpu==2.16.1",
        "streamlit-drawable-canvas==0.9.3",
        "Pillow==10.3.0",
        "numpy==1.26.4",
    ],
)