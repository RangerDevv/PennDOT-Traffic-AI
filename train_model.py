"""
PennDOT Lane Congestion Detector - Model Training Script
Trains a MobileNetV2-based classifier on the pennDOT Model Dataset,
then exports a quantized INT8 TFLite model and a C header file for
deployment on the ESP32-S3-EYE.

Usage:
    python train_model.py

Requirements:
    pip install tensorflow tf-keras pillow numpy scikit-learn matplotlib
"""

import os
import logging

# Force TF to use the legacy keras 2.x API via tf_keras.
# This is necessary for TFLite conversion to work with TF 2.16+.
os.environ["TF_USE_LEGACY_KERAS"]   = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"]      = "3"


import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# tf_keras is keras 2.x packaged for TF 2.16+ – fully TFLite compatible
import tf_keras as keras
from tf_keras import layers

# Try to import tensorflow-addons for augmentation
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    tfa = None
    TFA_AVAILABLE = False

from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIRS  = [Path("pennDOT Model Dataset"), Path("Sample Dataset")]
OUTPUT_TFLITE = Path("lane_congestion_model.tflite")
OUTPUT_HEADER = Path("lane_congestion_detector/model_data.h")

IMG_HEIGHT = 48
IMG_WIDTH  = 48
BATCH_SIZE = 8
EPOCHS     = 120
SEED       = 42

# Classes must match folder names exactly (order fixes label indices)
CLASS_NAMES = [
    "Left Lane Congestion",
    "Right Lane Congestion",
]

# Crop to bottom 50 % of image before resizing – that's where the road is
CROP_FRACTION = 0.50

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str, label: int):
    """Load, crop road region, resize and normalise a JPEG image."""
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_jpeg(raw, channels=3)
    shape = tf.shape(img)
    h     = shape[0]
    # Crop bottom CROP_FRACTION rows (where road / cars live)
    crop_start = tf.cast(tf.cast(h, tf.float32) * (1.0 - CROP_FRACTION), tf.int32)
    img   = img[crop_start:, :, :]
    img   = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img   = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_brightness(img, 0.20)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.75, 1.25)
    img = tf.image.random_hue(img, 0.05)
    # NOTE: random_flip_left_right is intentionally omitted — flipping
    # "Left Lane Congestion" produces a "Right Lane Congestion" scene but
    # keeps the wrong label, which destroys left/right discrimination.


    # Random small rotation (±10 degrees) if tfa is available
    if TFA_AVAILABLE:
        angle = tf.random.uniform([], -0.17, 0.17)  # radians ≈ ±10°
        img = tfa.image.rotate(img, angle, fill_mode='reflect')

        # Random perspective transform (mild)
        def random_perspective(img):
            d = 0.06 * tf.cast(tf.shape(img)[0], tf.float32)
            pts1 = tf.constant([[0,0],[IMG_WIDTH,0],[IMG_WIDTH,IMG_HEIGHT],[0,IMG_HEIGHT]], dtype=tf.float32)
            pts2 = pts1 + tf.random.uniform([4,2], -d, d)
            return tfa.image.transform_ops.matrices_to_flat_transforms(
                tfa.image.transform_ops.get_perspective_transform(pts1, pts2)
            )
        img = tfa.image.transform(img, random_perspective(img), fill_mode='reflect')

    # Random zoom / crop-and-resize
    crop_frac = tf.random.uniform([], 0.80, 1.0)
    crop_h = tf.cast(tf.cast(IMG_HEIGHT, tf.float32) * crop_frac, tf.int32)
    crop_w = tf.cast(tf.cast(IMG_WIDTH, tf.float32) * crop_frac, tf.int32)
    img = tf.image.random_crop(img, [crop_h, crop_w, 3])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Random Gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.03)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def build_dataset():
    paths, labels = [], []
    for dataset_dir in DATASET_DIRS:
        for idx, cls in enumerate(CLASS_NAMES):
            folder = dataset_dir / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                paths.append(str(img_path))
                labels.append(idx)

    paths  = np.array(paths)
    labels = np.array(labels)
    print(f"Total images found: {len(paths)}")
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {np.sum(labels == idx)}")

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=labels)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f"Class weights: {class_weights}")

    # Stratified split: 80 % train / 20 % val
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=SEED
    )
    print(f"Train: {len(tr_p)}  Val: {len(va_p)}")

    def make_ds(p, l, augment_flag=False):
        ds = tf.data.Dataset.from_tensor_slices((p, l))
        ds = ds.map(load_and_preprocess,
                    num_parallel_calls=tf.data.AUTOTUNE)
        if augment_flag:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(len(p), seed=SEED)
            # Repeat small dataset so each epoch sees more augmented variants
            ds = ds.repeat(8)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    return make_ds(tr_p, tr_l, augment_flag=True), make_ds(va_p, va_l), class_weights


def build_model():
    """Tiny CNN designed for fast INT8 inference on ESP32-S3."""
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)           # 24x24x8

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)           # 12x12x16

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)           # 6x6x32

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)  # 32

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train(model, train_ds, val_ds, class_weights):
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
    ]

    print("\n── Training ──")
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=cb,
                        class_weight=class_weights)
    return history


def plot_history(history):
    acc  = history.history["accuracy"]
    vacc = history.history["val_accuracy"]
    plt.figure(figsize=(8, 4))
    plt.plot(acc,  label="Train Acc")
    plt.plot(vacc, label="Val Acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training_history.png")


# ── Representative dataset for INT8 quantisation ─────────────────────────────
_rep_images = None

def _load_rep_images():
    global _rep_images
    if _rep_images is not None:
        return
    imgs = []
    for dataset_dir in DATASET_DIRS:
        for cls in CLASS_NAMES:
            folder = dataset_dir / cls
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.jpg"))[:5]:
                raw  = tf.io.read_file(str(p))
                img  = tf.image.decode_jpeg(raw, channels=3)
                h    = tf.shape(img)[0]
                cs   = tf.cast(tf.cast(h, tf.float32) * (1.0 - CROP_FRACTION), tf.int32)
                img  = img[cs:, :, :]
                img  = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
                img  = tf.cast(img, tf.float32) / 255.0
                imgs.append(img.numpy())
    _rep_images = np.array(imgs, dtype=np.float32)


def representative_data_gen():
    _load_rep_images()
    for img in _rep_images:
        yield [img[np.newaxis, ...]]


# ── TFLite conversion ─────────────────────────────────────────────────────────

def convert_to_tflite(model):
    # tf_keras (keras 2.x) is fully compatible with TFLiteConverter.from_keras_model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    OUTPUT_TFLITE.write_bytes(tflite_model)
    print(f"TFLite model saved → {OUTPUT_TFLITE}  ({len(tflite_model):,} bytes)")
    return tflite_model


# ── C header generation ───────────────────────────────────────────────────────

def generate_header(tflite_bytes: bytes):
    OUTPUT_HEADER.parent.mkdir(parents=True, exist_ok=True)

    hex_values = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    class_list = ", ".join(f'"{c}"' for c in CLASS_NAMES)
    num_classes = len(CLASS_NAMES)

    header = f"""\
// AUTO-GENERATED by train_model.py — DO NOT EDIT MANUALLY
// Model: MobileNetV2-0.35 INT8 quantised
// Input: {IMG_WIDTH}x{IMG_HEIGHT} RGB  |  Classes: {num_classes}

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>
#include <pgmspace.h>

// ── Model binary ─────────────────────────────────────────────────────────────
// __attribute__((aligned(8))): required 8-byte alignment for flatbuffers.
// PROGMEM: keep array in flash (not RAM) on ESP32.
const unsigned char g_model_data[] __attribute__((aligned(8))) PROGMEM = {{
  {hex_values}
}};
const unsigned int g_model_data_len = {len(tflite_bytes)};

// ── Class labels ─────────────────────────────────────────────────────────────
constexpr int   kNumClasses   = {num_classes};
constexpr int   kImageWidth   = {IMG_WIDTH};
constexpr int   kImageHeight  = {IMG_HEIGHT};
constexpr float kCropFraction = {CROP_FRACTION}f;  // use bottom fraction of frame

const char* const kClassNames[kNumClasses] = {{
  {class_list}
}};

#endif  // MODEL_DATA_H
"""
    OUTPUT_HEADER.write_text(header)
    print(f"C header saved → {OUTPUT_HEADER}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"TensorFlow {tf.__version__}")
    print(f"Classes: {CLASS_NAMES}")

    train_ds, val_ds, class_weights = build_dataset()
    model = build_model()

    history = train(model, train_ds, val_ds, class_weights)
    plot_history(history)

    # Save Keras model
    keras_path = "lane_congestion_model.keras"
    model.save(keras_path)
    print(f"Keras model saved → {keras_path}")

    # Convert & export
    tflite_bytes = convert_to_tflite(model)
    generate_header(tflite_bytes)

    print("\nDone!  Flash the Arduino sketch in lane_congestion_detector/")


if __name__ == "__main__":
    main()
