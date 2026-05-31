import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

def load_model(model_path: str = "models/fashion_cnn_model.h5"):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to model input tensor."""
    img = image.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)

def predict(model, image: Image.Image):
    """Return predicted class name and confidence."""
    tensor = preprocess_image(image)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs