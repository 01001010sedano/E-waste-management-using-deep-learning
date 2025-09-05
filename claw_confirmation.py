import cv2
import time
import numpy as np
import datetime

def capture_image(camera, delay=0.2, flush_count=0):
    """Capture a single image from the camera after a short delay to allow settling. No buffer flush."""
    time.sleep(delay)
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    return frame.copy()

def item_present(image, model, confidence_threshold=0.5, target_classes=None):
    """
    Run YOLO detection on the image. Return True if any target item is detected above threshold.
    If target_classes is None, any detection counts as present.
    """
    results = model(image, verbose=False)
    detected_any = False
    total_detections = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            print(f"[DEBUG] Detected: {label} (conf: {conf:.2f})")  # DEBUG
            total_detections += 1
            if conf < confidence_threshold:
                continue
            if (target_classes is None) or (label in target_classes):
                print(f"[DEBUG] Item present: {label} (conf: {conf:.2f})")  # DEBUG
                detected_any = True
    print(f"[DEBUG] Total detections in image: {total_detections}")  # DEBUG
    return detected_any

def confirm_claw_grip(camera, model, max_retries=3, confidence_threshold=0.5, target_classes=None, verbose=True):
    """
    Attempt to confirm that the item has been gripped by comparing before/after images.
    Returns True if grip succeeded, False if failed after retries.
    Now requires 2 consecutive frames with no detection to confirm grip.
    """
    consecutive_no_detection = 0
    for attempt in range(1, max_retries+1):
        if verbose:
            print(f"[Claw Confirmation] Attempt {attempt}...")
        # Capture before grip
        if verbose:
            print("  Capturing BEFORE grip image...")
        img_before = capture_image(camera)
        if verbose:
            print("  Please close the claw now (in main sequence)...")
        time.sleep(0.5)  # Give time for claw to close in main sequence
        # Capture after grip
        if verbose:
            print("  Capturing AFTER grip image...")
        img_after = capture_image(camera)
        # Save after-grip image for debugging
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        after_path = f"claw_confirm_after_attempt{attempt}_{timestamp}.jpg"
        cv2.imwrite(after_path, img_after)
        print(f"[DEBUG] Saved after-grip image: {after_path}")
        # Check if item is still present after grip
        present_after = item_present(img_after, model, confidence_threshold, target_classes)
        if not present_after:
            consecutive_no_detection += 1
            if consecutive_no_detection >= 2:
                if verbose:
                    print("  Grip confirmed: item is gone from pickup area (2x no detection).")
                return True
            else:
                if verbose:
                    print("  No item detected, but waiting for another confirmation frame...")
                continue
        else:
            consecutive_no_detection = 0
            if verbose:
                print("  Grip failed: item still present. Retrying...")
            time.sleep(0.5)  # Wait before retry
    if verbose:
        print("[Claw Confirmation] Failed to grip item after max retries.")
    return False 