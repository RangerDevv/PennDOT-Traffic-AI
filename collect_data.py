"""
PennDOT Data Collector – Rapidly capture labeled training images from ESP32-S3-EYE.

Shows the live camera feed in a window. Press a key to save the current
frame into the correct class folder:

    B  →  Both Lane Congestion
    L  →  Left Lane Congestion
    N  →  No Lane Congestion
    R  →  Right Lane Congestion
    Q  →  Quit

Images are saved to 'pennDOT Model Dataset/<class>/' with a timestamp filename.

Usage:
    python collect_data.py [ESP32_IP]
    python collect_data.py 192.168.1.42

Requirements:
    pip install opencv-python requests
"""

import sys
import time
from pathlib import Path

import cv2
import requests
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
ESP32_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.1"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
DATASET_DIR = Path("pennDOT Model Dataset")

CLASS_KEYS = {
    ord('b'): "Both Lane Congestion",
    ord('l'): "Left Lane Congestion",
    ord('n'): "No Lane Congestion",
    ord('r'): "Right Lane Congestion",
}

# Ensure all class folders exist
for name in CLASS_KEYS.values():
    (DATASET_DIR / name).mkdir(parents=True, exist_ok=True)


def grab_frame():
    """Grab a single JPEG from /capture and decode it."""
    try:
        r = requests.get(CAPTURE_URL, timeout=3)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img, r.content
    except requests.RequestException:
        pass
    return None, None


def main():
    print(f"Connecting to ESP32 at {ESP32_IP}...")
    print("Keys:  B=Both  L=Left  N=No congestion  R=Right  Q=Quit")
    print(f"Saving to: {DATASET_DIR.resolve()}\n")

    # Count existing images per class
    for name in CLASS_KEYS.values():
        count = len(list((DATASET_DIR / name).glob("*.jpg")))
        print(f"  {name}: {count} images")
    print()

    saved_total = 0
    cv2.namedWindow("PennDOT Data Collector", cv2.WINDOW_NORMAL)

    while True:
        img, jpeg_bytes = grab_frame()
        if img is None:
            cv2.putText(
                np.zeros((240, 320, 3), dtype=np.uint8),
                "No connection...", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            cv2.waitKey(500)
            continue

        # Draw help overlay
        display = img.copy()
        labels = ["B:Both", "L:Left", "N:None", "R:Right", "Q:Quit"]
        for i, lbl in enumerate(labels):
            cv2.putText(display, lbl, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display, f"Saved: {saved_total}", (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("PennDOT Data Collector", display)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break

        if key in CLASS_KEYS:
            class_name = CLASS_KEYS[key]
            ts = int(time.time() * 1000)
            filename = DATASET_DIR / class_name / f"{ts}.jpg"
            filename.write_bytes(jpeg_bytes)
            saved_total += 1
            count = len(list((DATASET_DIR / class_name).glob("*.jpg")))
            print(f"  ✓ Saved → {class_name} ({count} total)")

    cv2.destroyAllWindows()
    print(f"\nDone! Saved {saved_total} new images.")
    print("Run 'python train_model.py' to retrain with the new data.")


if __name__ == "__main__":
    main()
