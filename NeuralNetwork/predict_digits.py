"""
predict_digits.py
------------------
Reloads the trained CNN (and MLP, for comparison) from ./models/ and predicts
the digit in every image found in ./sekiller/ (put your 5 handwritten digit
photos/drawings there — .jpg, .jpeg, or .png).

Folder layout expected (relative to this script):

    project_folder/
        predict_digits.py   <- this file
        models/
            cnn.weights.h5
            mlp.weights.h5
        sekiller/
            <your 5 images here>

Usage:
    python predict_digits.py
"""
import glob
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "sekiller")


def build_cnn():
    """Must exactly match the architecture used during training."""
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(10, activation="softmax", name="output"),
    ], name="cnn")


def build_mlp():
    """Must exactly match the architecture used during training."""
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.Dense(128, activation="relu", name="dense2"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(10, activation="softmax", name="output"),
    ], name="mlp")


def preprocess_image(path):
    """Load a real photographed/drawn digit image and turn it into a
    (28, 28) float32 array in [0, 1], matching MNIST's format:
      1. Load
      2. Convert to grayscale
      3. Resize to 28x28
      4. Invert if needed (photographed ink-on-paper is dark-on-light;
         MNIST is light-on-dark -- auto-detected by mean brightness)
      5. Normalize to [0, 1]
    """
    raw = Image.open(path)
    gray = raw.convert("L")
    resized = gray.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized).astype("float32")

    if arr.mean() > 127:
        arr = 255.0 - arr

    return arr / 255.0


def load_models():
    missing = []
    cnn_path = os.path.join(MODELS_DIR, "cnn.weights.h5")
    mlp_path = os.path.join(MODELS_DIR, "mlp.weights.h5")
    if not os.path.exists(cnn_path):
        missing.append(cnn_path)
    if not os.path.exists(mlp_path):
        missing.append(mlp_path)
    if missing:
        raise FileNotFoundError(
            "Missing model weight file(s):\n  " + "\n  ".join(missing) +
            "\n\nRun the training notebook first (it saves into ./models/)."
        )

    cnn = build_cnn()
    cnn.load_weights(cnn_path)

    mlp = build_mlp()
    mlp.load_weights(mlp_path)

    return cnn, mlp


def main():
    print(f"Looking for images in: {IMAGES_DIR}")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Folder not found: {IMAGES_DIR}\n"
            f"Create a 'sekiller' folder next to this script and put your "
            f"5 handwritten digit images in it."
        )

    paths = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    )

    if not paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png images found in {IMAGES_DIR}")

    print(f"Found {len(paths)} image(s).\n")

    print("Loading trained models from ./models/ ...")
    cnn, mlp = load_models()
    print("Models loaded.\n")

    print(f"{'File':<30} {'CNN pred':>10} {'CNN conf':>10} {'MLP pred':>10} {'MLP conf':>10}")
    print("-" * 74)

    for path in paths:
        fname = os.path.basename(path)
        processed = preprocess_image(path)
        batch = processed[np.newaxis, ..., np.newaxis]

        cnn_probs = cnn.predict(batch, verbose=0)[0]
        mlp_probs = mlp.predict(batch, verbose=0)[0]

        cnn_pred, cnn_conf = int(np.argmax(cnn_probs)), float(np.max(cnn_probs))
        mlp_pred, mlp_conf = int(np.argmax(mlp_probs)), float(np.max(mlp_probs))

        print(f"{fname:<30} {cnn_pred:>10} {cnn_conf:>9.1%} {mlp_pred:>10} {mlp_conf:>9.1%}")


if __name__ == "__main__":
    main()
