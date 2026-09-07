"""
app.py
------
Streamlit demo for the CNN trained in the mandatory task (cnn_vs_mlp_mnist).

Loads the saved CNN weights (no retraining happens here), lets the user draw
a digit on an in-browser canvas, preprocesses that drawing to exactly match
the training-time pipeline, and shows the predicted digit plus a top-3
probability breakdown.

Run with:
    streamlit run app.py
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "cnn.weights.h5")
LOG_PATH = os.path.join(SCRIPT_DIR, "logs", "predictions_log.csv")

CANVAS_SIZE = 280  # 10x the 28x28 target -> clean downscale, easy to draw on


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session, never retrained)
# ---------------------------------------------------------------------------
def build_cnn():
    """Must exactly match the architecture used during training
    (see cnn_vs_mlp_mnist.ipynb, section 3)."""
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


@st.cache_resource(show_spinner="Loading trained CNN...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = build_cnn()
    model.load_weights(MODEL_PATH)
    return model


# ---------------------------------------------------------------------------
# Preprocessing -- MUST mirror the training-time pipeline exactly:
#   1. Grayscale
#   2. Resize to 28x28
#   3. Auto-invert so digit ends up light-on-dark, like MNIST
#   4. Normalize pixel values to [0, 1]
# This is the same logic used in predict_digits.py (the CLI sibling of this
# app), so both entry points agree on what the model actually sees.
# ---------------------------------------------------------------------------
def preprocess_canvas(image_data: np.ndarray) -> np.ndarray:
    rgba = image_data.astype("uint8")
    img = Image.fromarray(rgba, mode="RGBA").convert("L")
    resized = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized).astype("float32")

    if arr.mean() > 127:
        arr = 255.0 - arr

    return arr / 255.0


def predict(model, processed: np.ndarray):
    batch = processed[np.newaxis, ..., np.newaxis]
    probs = model.predict(batch, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    return probs, top3_idx


def log_prediction(pred_digit: int, confidence: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted_digit": pred_digit,
        "confidence": round(confidence, 4),
    }])
    header = not os.path.exists(LOG_PATH)
    row.to_csv(LOG_PATH, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Digit Classifier Demo", page_icon="\u270f\ufe0f", layout="centered")
st.title("Handwritten Digit Classifier")
st.caption("Draw a single digit (0-9) below. The CNN was trained once, offline, and is only reloaded here.")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

model = load_model()

if model is None:
    st.error(
        f"Model weights not found at `models/cnn.weights.h5`.\n\n"
        f"Run the training notebook first, then copy its `models/cnn.weights.h5` "
        f"output into this app's `models/` folder."
    )
    st.stop()

col_canvas, col_controls = st.columns([2, 1])

with col_canvas:
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        key=f"canvas_{st.session_state.canvas_key}",
    )

with col_controls:
    live_mode = st.checkbox("Live prediction", value=False, help="Predict continuously as you draw (bonus feature).")
    predict_clicked = st.button("Predict", use_container_width=True, disabled=live_mode)
    if st.button("Clear", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()
    log_enabled = st.checkbox("Log predictions", value=False, help="Append each prediction to logs/predictions_log.csv (bonus feature).")

has_drawing = (
    canvas_result.image_data is not None
    and canvas_result.image_data[:, :, :3].sum() > 0
)

should_predict = has_drawing and (predict_clicked or live_mode)

if should_predict:
    processed = preprocess_canvas(canvas_result.image_data)
    probs, top3_idx = predict(model, processed)
    pred_digit = int(top3_idx[0])
    pred_conf = float(probs[pred_digit])

    st.divider()
    result_col, chart_col = st.columns([1, 2])

    with result_col:
        st.metric("Predicted digit", pred_digit, f"{pred_conf:.1%} confidence")
        st.image(processed, caption="What the model sees (28x28)", width=140, clamp=True)

    with chart_col:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        labels = [str(i) for i in top3_idx]
        values = [probs[i] for i in top3_idx]
        bars = ax.barh(labels[::-1], values[::-1], color="#3182bd")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Top-3 predictions")
        for bar, v in zip(bars, values[::-1]):
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)
        st.pyplot(fig)

    if log_enabled:
        log_prediction(pred_digit, pred_conf)
        st.caption("Logged to `logs/predictions_log.csv`.")
elif not has_drawing:
    st.info("Draw a digit on the canvas, then click **Predict** (or enable **Live prediction**).")

with st.expander("Preprocessing details (must match training exactly)"):
    st.markdown(
        """
        1. Canvas RGBA -> grayscale (`convert("L")`)
        2. Resize to **28x28** with LANCZOS resampling
        3. Auto-invert if the image is mostly light (canvas is black-on-white,
           MNIST is white-on-black) — same rule as `predict_digits.py`
        4. Normalize pixel values from `[0, 255]` to `[0, 1]`
        5. Add batch and channel dims -> shape `(1, 28, 28, 1)` before `model.predict`
        """
    )
