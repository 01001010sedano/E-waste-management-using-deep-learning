from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("/home/amen2022/pso-1/train32/weights/best.pt")

toxicity_map = {
    "Battery": "Red",
    "PCB": "Yellow",
    "Sensor": "Yellow",
    "Cables": "Green",
    "USB flash drive": "Green"
}   

# Use a higher performance backend if available
cv2.setUseOptimized(True)

# Initialize webcam (0 = default)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # lower res = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)  # cap frame rate if needed

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🎥 Camera ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received.")
        break

    # Run YOLO prediction
    results = model.predict(source=frame, conf=0.4, iou=0.4, show=False, verbose=False)

    # Draw bounding boxes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        classification = toxicity_map.get(label, "Green")
        print(f"Detected: {label} | Confidence: {conf:.2f} | Classification: {classification}")

        color = (0, 255, 0) if classification == "Green" else \
                (0, 255, 255) if classification == "Yellow" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({classification})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 E-Waste Detection", frame)

    # Wait ~33ms per frame = ~30fps. Press 'q' to exit
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
