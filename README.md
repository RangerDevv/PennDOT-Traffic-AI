# PennDOT Lane Congestion Detector – ESP32-S3-EYE

Classifies live camera frames into four road states using a quantised
MobileNetV2 model running entirely on the ESP32-S3-EYE (no cloud needed).

| Class | Description |
|---|---|
| Both Lane Congestion | Vehicles/objects blocking both lanes |
| Left Lane Congestion | Congestion in the left lane only |
| No Lane Congestion | Road is clear |
| Right Lane Congestion | Congestion in the right lane only |

---

## Repository layout

```
PennDOT-Traffic-AI/
├── pennDOT Model Dataset/         # Training images (4 class folders)
├── Sample Dataset/                # Extra sample images
├── train_model.py                 # Step 1 – train & export model
└── lane_congestion_detector/
    ├── lane_congestion_detector.ino   # Arduino sketch
    └── model_data.h                   # Generated C header (model weights)
```

---

## Quick-start

### 1. Train the model (host PC)

```bash
# Install Python dependencies
pip install tensorflow pillow numpy scikit-learn matplotlib

# Train MobileNetV2, quantise to INT8, write model_data.h
python train_model.py
```

The script will:
- Load images from `pennDOT Model Dataset/`
- Apply data augmentation (brightness, contrast, horizontal flip)
- Run a two-phase training: frozen base → fine-tune top 30 layers
- Save `lane_congestion_model.keras` and `lane_congestion_model.tflite`
- Overwrite `lane_congestion_detector/model_data.h` with the embedded model

### 2. Flash the ESP32-S3-EYE

#### Arduino IDE setup

1. Add the ESP32 board package (if not already):  
   File → Preferences → Additional boards manager URLs:  
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`

2. Install boards: Tools → Board → Boards Manager → search **ESP32** → install
   **esp32 by Espressif Systems** ≥ 2.0.11

3. Install the TFLite library:  
   Sketch → Include Library → Manage Libraries → search **TensorFlowLite_ESP32** → install

4. Board settings:
   | Setting | Value |
   |---|---|
   | Board | ESP32S3 Dev Module |
   | Flash Mode | QIO 80MHz |
   | Flash Size | 8MB (64Mb) |
   | Partition Scheme | Minimal SPIFFS (1.9MB APP / 300KB SPIFFS) |
   | PSRAM | OPI PSRAM |
   | Upload Speed | 921600 |

5. Open `lane_congestion_detector/lane_congestion_detector.ino` and upload.

#### PlatformIO (alternative)

```ini
[env:esp32s3eye]
platform  = espressif32
board     = esp32-s3-devkitc-1
framework = arduino
board_build.psram     = opi
board_build.flash_mode = qio
board_build.partitions = min_spiffs.csv
lib_deps  = tanakamasayuki/TensorFlowLite_ESP32
```

### 3. Monitor results

Open Serial Monitor at **115200 baud**. Every 2 seconds you'll see:

```
─────────────────────────────────
  Both Lane Congestion       2.4%
  Left Lane Congestion       5.1%
  No Lane Congestion         1.8%
  Right Lane Congestion     90.7%
→ RESULT: Right Lane Congestion  (90.7%)
```

---

## How it works

1. The OV2640 camera captures a 320×240 RGB565 frame.
2. The **bottom 50 %** of the frame is cropped (where the road surface appears).
3. The crop is nearest-neighbour resized to **96×96** and quantised to INT8 to match the model's input tensor.
4. TFLite Micro invokes the model on the ESP32-S3's CPU.
5. The INT8 output logits are dequantised to float probabilities and the highest-confidence class is printed over UART.

> **Tip:** Point the camera so that both lanes fill the lower half of the frame,
> matching the perspective used in the training images.

---

## Tuning

| Constant | Location | Description |
|---|---|---|
| `kConfidenceThreshold` | `.ino` | Minimum confidence to report a class (default 0.65) |
| `kInferenceIntervalMs` | `.ino` | Delay between inferences in ms (default 2000) |
| `kCropFraction` | `model_data.h` | Fraction of frame height to keep from bottom (default 0.5) |
| `CROP_FRACTION` | `train_model.py` | Must match `kCropFraction` |
| `IMG_HEIGHT/WIDTH` | `train_model.py` | Input resolution (default 96×96) |
