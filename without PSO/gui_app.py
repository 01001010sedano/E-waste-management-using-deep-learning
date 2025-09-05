import tkinter as tk
import cv2
import threading
import time
import os
import json
import pandas as pd
import shared_state
from ultralytics import YOLO
from queue import Queue, Empty
from threading import Event
import numpy as np
from shared_state import global_frame, frame_lock, latest_raw_frame, last_annotated_frame, set_status_message
from sortation import toxicity_map, bin_mapping  # Import the mapping from sortation
from robotic_arm import RoboticArm
from tkinter import messagebox
from PIL import Image, ImageTk
from report_generator import generate_toxicity_report_graph



# Create the main Tkinter window
root = tk.Tk()
root.title("E-Waste Detection System")
root.geometry("600x680")
root.resizable(False, False)

# Initialize camera
print("\n🎥 Initializing camera...")

def initialize_camera():
    """Initialize camera with cross-platform support for Windows and Linux/Raspberry Pi"""
    import platform
    
    system = platform.system()
    
    if system == "Windows":
        # Windows-specific initialization
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW avoids long delays on Windows
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera on Windows (device 0)")
    else:
        # Linux/Raspberry Pi initialization
        # Try different video devices in order of preference
        devices_to_try = [0, 1, 2]  # Try /dev/video0, /dev/video1, /dev/video2
        
        cap = None
        for device in devices_to_try:
            try:
                cap = cv2.VideoCapture(device)
                if cap.isOpened():
                    print(f"✅ Camera opened successfully on device {device}")
                    break
                else:
                    cap.release()
            except Exception as e:
                print(f"Failed to open device {device}: {e}")
                if cap:
                    cap.release()
        
        if not cap or not cap.isOpened():
            raise RuntimeError("Failed to open camera on any available device")
    
    return cap

shared_state.camera = initialize_camera()

# Add a video feed label
video_label = tk.Label(root)
video_label.pack()

def update_video_feed():
    with frame_lock:
        frame_to_show = shared_state.current_display_frame.copy() if shared_state.current_display_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (400, 300))
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.config(image=img)
    video_label.image = img

    root.after(30, update_video_feed)

def detection_thread_func(model, cap, detection_queue, stop_event, detection_pause_event):
    """Continuously process frames and enqueue detections, pausing when requested."""
    CONFIDENCE_THRESHOLD = 0.6  # Only process detections with confidence > 60%
    
    while not stop_event.is_set():
        if detection_pause_event.is_set():
            time.sleep(0.05)
            continue
            
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Process frame at consistent size
        process_frame = cv2.resize(frame, (640, 480))
        results = model(process_frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                
                # Only process high confidence detections
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                    
                class_id = int(box.cls[0])
                detected_class = model.names[class_id]
                print(f"[DEBUG] YOLO detected: {detected_class}")
                
                toxicity_level = toxicity_map.get(detected_class, toxicity_map["default"])
                bin_number = bin_mapping[toxicity_level]
                print(f"[DEBUG] Mapped toxicity_level: {toxicity_level} → bin {bin_number}")
                
                # Update status message
                set_status_message(f"Detected: {detected_class} ({conf:.2f})", "black")
                
                # Add detection to queue with detection timestamp
                detection_time = time.time()
                detection_queue.put((detected_class, toxicity_level, bin_number, conf, detection_time))
                
                # Draw detection for display with correct colors
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)  # Default green for non-toxic
                if toxicity_level == "highly_toxic":
                    color = (0, 0, 255)  # Red for highly toxic
                elif toxicity_level == "mildly_toxic":
                    color = (0, 165, 255)  # Orange for mildly toxic
                
                cv2.rectangle(process_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{detected_class} ({conf:.2f}) - {toxicity_level} (Bin {bin_number})"
                cv2.putText(process_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Update shared state frame
        with frame_lock:
            shared_state.current_display_frame = process_frame.copy()

def arm_thread_func(arm, detection_queue, stop_event, detection_pause_event):
    """Handle arm movements based on detections from the queue."""
    from sortation import pick_up_item, drop_to_bin, move_to_pickup, move_to_bin
    
    while not stop_event.is_set():
        try:
            # Wait for detection with timeout
            detected_class, toxicity_level, bin_number, conf, detection_time = detection_queue.get(timeout=1.0)
            
            print(f"\n🎯 Sorting {detected_class}...")
            print(f"⏱️ Detection time: {detection_time}")
            
            # Pause detection during arm movement
            detection_pause_event.set()
            
            # Move to pickup position first (if not already there)
            print("🔄 Moving to pickup position...")
            move_to_pickup(arm)
            
            # Perform sorting sequence
            print("🤏 Picking up item...")
            pick_up_item(arm)
            
            print(f"📦 Moving to bin {bin_number}...")
            move_to_bin(arm, bin_number)
            
            print(f"🗑️ Dropping item into bin {bin_number}...")
            drop_to_bin(arm, bin_number)
            
            print("🔄 Returning to pickup position...")
            move_to_pickup(arm)
            
            # Calculate total elapsed time from detection to completion
            completion_time = time.time()
            total_elapsed_time = completion_time - detection_time
            
            result = {
                "item": detected_class,
                "detection_time": detection_time,
                "completion_time": completion_time,
                "total_time_seconds": total_elapsed_time,
                "confidence": conf,
                "toxicity_level": toxicity_level,
                "bin_number": bin_number
            }
            print("📊 Sorting Results:")
            print(json.dumps(result, indent=2))
            
            # Resume detection
            detection_pause_event.clear()
            
        except Empty:
            continue
        except Exception as e:
            print(f"Error in arm thread: {e}")
            detection_pause_event.clear()

def run_yolo_detection():
    def detection_loop():
        try:
            print("\n🤖 Loading YOLOv8 model...")
            model = YOLO('train32/weights/best.pt')
            model.conf = 0.6
            print("✅ Model loaded successfully!")
            
            # Use the camera instance from shared_state
            if not shared_state.camera or not shared_state.camera.isOpened():
                print("Error: Camera not initialized")
                return
            
            print("[INFO] Starting detection loop.")
            
            # Initialize detection system
            stop_event = Event()
            detection_queue = Queue()
            detection_pause_event = Event()
            detection_pause_event.clear()  # Allow detection
            
            # Create arm instance
            arm = RoboticArm()
            
            # Start threads
            detection_thread = threading.Thread(target=detection_thread_func, 
                                              args=(model, shared_state.camera, detection_queue, stop_event, detection_pause_event))
            arm_thread = threading.Thread(target=arm_thread_func, 
                                        args=(arm, detection_queue, stop_event, detection_pause_event))
            
            detection_thread.start()
            arm_thread.start()
            
            # Keep the detection running
            while True:
                time.sleep(1)
                
        except Exception as e:
            error_message = str(e)
            root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during detection:\n{error_message}"))

    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()


def export_to_excel():
    json_file = "summary.json"
    excel_file = "summary.xlsx"
    try:
        if not os.path.exists(json_file):
            messagebox.showwarning("Warning", "No summary.json file found!")
            return

        with open(json_file, "r") as file:
            data = json.load(file)

        if not isinstance(data, list) or len(data) == 0:
            messagebox.showwarning("Warning", "No data available to export!")
            return

        df = pd.DataFrame(data)
        df.to_excel(excel_file, index=False)
        messagebox.showinfo("Success", f"Summary exported successfully to {excel_file}!")
        if os.name == "nt":
            os.startfile(os.path.abspath(excel_file))
        elif os.name == "posix":
            os.system(f'xdg-open "{os.path.abspath(excel_file)}"')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during export:\n{e}")

def export_unidentified_to_excel():
    json_file = "unidentified/unidentified.json"
    excel_file = "unidentified/unidentified.xlsx"
    try:
        if not os.path.exists(json_file):
            messagebox.showwarning("Warning", "No unidentified.json file found!")
            return

        with open(json_file, "r") as file:
            data = json.load(file)

        if not isinstance(data, list) or len(data) == 0:
            messagebox.showwarning("Warning", "No unidentified data to export!")
            return

        df = pd.DataFrame(data)
        df.to_excel(excel_file, index=False)
        messagebox.showinfo("Success", f"Unidentified log exported to {excel_file}!")
        if os.name == "nt":
            os.startfile(os.path.abspath(excel_file))
        elif os.name == "posix":
            os.system(f'xdg-open "{os.path.abspath(excel_file)}"')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

def update_camera_feed():
    while True:
        ret, frame = shared_state.camera.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        with frame_lock:
            latest_raw_frame[:] = frame.copy()
        time.sleep(0.03)

def refresh_display():
    with frame_lock:
        frame_to_show = last_annotated_frame if last_annotated_frame is not None else latest_raw_frame.copy()
    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.config(image=img)
    video_label.image = img
    root.after(30, refresh_display)

def quit_app():
    if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
        try:
            generate_toxicity_report_graph()  # ✅ Generate graph before quitting
            messagebox.showinfo("Graph Saved", "Toxicity graph has been generated and saved.")
        except Exception as e:
            messagebox.showerror("Graph Error", f"Failed to generate graph:\n{e}")
        
        if shared_state.camera is not None:
            shared_state.camera.release()
        root.destroy()


# Add a title label
title_label = tk.Label(root, text="E-Waste Detection System", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Add buttons
start_button = tk.Button(root, text="Start Detection", font=("Arial", 12), command=run_yolo_detection, width=20)
start_button.pack(pady=10)



export_button = tk.Button(root, text="Export Summary to Excel", font=("Arial", 12), command=export_to_excel, width=20)
export_button.pack(pady=10)

unidentified_button = tk.Button(root, text="Export Unidentified to Excel", font=("Arial", 12), command=export_unidentified_to_excel, width=25)
unidentified_button.pack(pady=10)

quit_button = tk.Button(root, text="Quit", font=("Arial", 12), command=quit_app, width=20)
quit_button.pack(pady=10)

# Add bin status labels
bin_labels = {
    "non_toxic": tk.Label(root, text="Non-Toxic Bin: OK", fg="green", font=("Arial", 10)),
    "mildly_toxic": tk.Label(root, text="Mildly-Toxic Bin: OK", fg="green", font=("Arial", 10)),
    "highly_toxic": tk.Label(root, text="Highly-Toxic Bin: OK", fg="green", font=("Arial", 10)),
}
for label in bin_labels.values():
    label.pack()
shared_state.bin_labels = bin_labels

status_label = tk.Label(root, text="System ready.", font=("Arial", 10), fg="black")
status_label.pack(pady=10)
shared_state.status_label = status_label

# Start the video feed update loop
update_video_feed()
threading.Thread(target=update_camera_feed, daemon=True).start()
run_yolo_detection()
# Run the GUI
root.mainloop()
