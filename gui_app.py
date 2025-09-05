import tkinter as tk
import cv2
import threading
import time
import os
import json
import pandas as pd
import shared_state
#change line 10 from newArmControl to latestArmControl
import latestArmControl as latestArmControl
from shared_state import global_frame, frame_lock, last_annotated_frame, set_status_message, camera, camera2, latest_raw_frame_cam1, latest_raw_frame_cam2, last_annotated_frame_cam1, last_annotated_frame_cam2
from robotic_arm import RoboticArm
from tkinter import messagebox
from PIL import Image, ImageTk
from report_generator import generate_toxicity_report_graph
import numpy as np



# Create the main Tkinter window
root = tk.Tk()
root.title("E-Waste Detection System")
root.geometry("900x680")  # Increased width to accommodate two cameras
root.resizable(False, False)

# Initialize camera 1
print("\n🎥 Initializing camera 1...")
device_nodes_cam1 = [
    "/dev/video1",  # Main video device
    "/dev/video2",  # Secondary video device
]

for device in device_nodes_cam1:
    print(f"\nTrying device for camera 1: {device}")
    try:
        # First try to set the format using v4l2-ctl
        os.system(f'v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG -d {device}')
        
        # Then open with OpenCV
        shared_state.camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not shared_state.camera.isOpened():
            print(f"Failed to open device {device}")
            continue
        
        # Set camera properties
        shared_state.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        shared_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        shared_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        shared_state.camera.set(cv2.CAP_PROP_FPS, 30)
        shared_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test if we can actually read a frame
        ret, frame = shared_state.camera.read()
        if ret and frame is not None:
            print(f"✅ Camera 1 initialized successfully on {device}!")
            print(f"Frame size: {frame.shape}")
            # Try to read a few more frames to ensure stability
            for _ in range(5):
                ret, frame = shared_state.camera.read()
                if not ret or frame is None:
                    raise Exception("Failed to read stable frames")
            break
        else:
            print(f"Device opened but couldn't read frame")
            shared_state.camera.release()
    except Exception as e:
        print(f"Error trying device {device}: {str(e)}")
        if shared_state.camera is not None:
            shared_state.camera.release()

# Initialize camera 2
print("\n🎥 Initializing camera 2...")
device_nodes_cam2 = [
    "/dev/video3",  # Third video device
    "/dev/video4",  # Fourth video device
    "/dev/video0",  # Fallback device
]

for device in device_nodes_cam2:
    print(f"\nTrying device for camera 2: {device}")
    try:
        # First try to set the format using v4l2-ctl
        os.system(f'v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG -d {device}')
        
        # Then open with OpenCV
        shared_state.camera2 = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not shared_state.camera2.isOpened():
            print(f"Failed to open device {device}")
            continue
        
        # Set camera properties
        shared_state.camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        shared_state.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        shared_state.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        shared_state.camera2.set(cv2.CAP_PROP_FPS, 30)
        shared_state.camera2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test if we can actually read a frame
        ret, frame = shared_state.camera2.read()
        if ret and frame is not None:
            print(f"✅ Camera 2 initialized successfully on {device}!")
            print(f"Frame size: {frame.shape}")
            # Try to read a few more frames to ensure stability
            for _ in range(5):
                ret, frame = shared_state.camera2.read()
                if not ret or frame is None:
                    raise Exception("Failed to read stable frames")
            break
        else:
            print(f"Device opened but couldn't read frame")
            shared_state.camera2.release()
    except Exception as e:
        print(f"Error trying device {device}: {str(e)}")
        if shared_state.camera2 is not None:
            shared_state.camera2.release()

if not shared_state.camera or not shared_state.camera.isOpened():
    messagebox.showerror("Camera Error", "Could not initialize camera 1")
    exit()

if not shared_state.camera2 or not shared_state.camera2.isOpened():
    print("Warning: Could not initialize camera 2, continuing with single camera mode")
    shared_state.camera2 = None

# Create a frame to hold both video feeds
video_frame = tk.Frame(root)
video_frame.pack(pady=10)

# Add video feed labels for both cameras
video_label_cam1 = tk.Label(video_frame, text="Camera 1", font=("Arial", 10, "bold"))
video_label_cam1.grid(row=0, column=0, padx=5)

video_label_cam2 = tk.Label(video_frame, text="Camera 2", font=("Arial", 10, "bold"))
video_label_cam2.grid(row=0, column=1, padx=5)

video_feed_cam1 = tk.Label(video_frame)
video_feed_cam1.grid(row=1, column=0, padx=5)

video_feed_cam2 = tk.Label(video_frame)
video_feed_cam2.grid(row=1, column=1, padx=5)

# Add sorting time label under the video feed
sorting_time_label = tk.Label(root, text="Last Sorting Time: -- s", font=("Arial", 10), fg="blue")
sorting_time_label.pack(pady=5)
shared_state.sorting_time_label = sorting_time_label

def update_sorting_time_label():
    try:
        with open("summary.json", "r") as file:
            data = json.load(file)
        if isinstance(data, list) and len(data) > 0:
            last_entry = data[-1]
            sorting_time = last_entry.get("sorting_time_sec", "--")
            sorting_time_label.config(text=f"Last Sorting Time: {sorting_time} s")
        else:
            sorting_time_label.config(text="Last Sorting Time: -- s")
    except Exception:
        sorting_time_label.config(text="Last Sorting Time: -- s")
    root.after(1000, update_sorting_time_label)

def update_video_feed():
    # Update camera 1 feed
    with frame_lock:
        frame_to_show = last_annotated_frame_cam1 if last_annotated_frame_cam1 is not None else latest_raw_frame_cam1.copy()

    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (400, 300))
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_feed_cam1.config(image=img)
    video_feed_cam1.image = img

    # Update camera 2 feed
    if shared_state.camera2:
        with frame_lock:
            frame_to_show = last_annotated_frame_cam2 if last_annotated_frame_cam2 is not None else latest_raw_frame_cam2.copy()

        frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (400, 300))
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_feed_cam2.config(image=img)
        video_feed_cam2.image = img

    root.after(30, update_video_feed)

def run_yolo_detection():
    def detection_loop():
        while True:
            try:
                arm = RoboticArm()
                latestArmControl.detect_and_sort(arm)  # ✅ Use correct import
            except Exception as e:
                error_message = str(e)
                root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during detection:\n{error_message}"))
            time.sleep(3)

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
        # Update camera 1 feed
        ret, frame = shared_state.camera.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            with frame_lock:
                latest_raw_frame_cam1[:] = frame.copy()
        
        # Update camera 2 feed if available
        if shared_state.camera2:
            ret2, frame2 = shared_state.camera2.read()
            if ret2:
                frame2 = cv2.resize(frame2, (640, 480))
                with frame_lock:
                    latest_raw_frame_cam2[:] = frame2.copy()
        
        time.sleep(0.03)

def refresh_display():
    # Update camera 1 display
    with frame_lock:
        frame_to_show = last_annotated_frame_cam1 if last_annotated_frame_cam1 is not None else latest_raw_frame_cam1.copy()
    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (400, 300))
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_feed_cam1.config(image=img)
    video_feed_cam1.image = img

    # Update camera 2 display if available
    if shared_state.camera2:
        with frame_lock:
            frame_to_show = last_annotated_frame_cam2 if last_annotated_frame_cam2 is not None else latest_raw_frame_cam2.copy()

        frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (400, 300))
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_feed_cam2.config(image=img)
        video_feed_cam2.image = img

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
        if shared_state.camera2:
            shared_state.camera2.release()
        root.destroy()

# Create a horizontal frame to hold buttons and status/bin info side by side
main_content = tk.Frame(root)
main_content.pack(fill=tk.X, pady=10)

# Left frame for buttons
button_frame = tk.Frame(main_content)
button_frame.pack(side=tk.LEFT, padx=20, anchor="n")

# Right frame for status and bin info
status_frame = tk.Frame(main_content)
status_frame.pack(side=tk.LEFT, padx=40, anchor="n")

# Add a title label
title_label = tk.Label(root, text="E-Waste Detection System", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Add buttons to button_frame
start_button = tk.Button(button_frame, text="Start Detection", font=("Arial", 12), command=run_yolo_detection, width=20)
start_button.pack(pady=10)

export_button = tk.Button(button_frame, text="Export Summary to Excel", font=("Arial", 12), command=export_to_excel, width=20)
export_button.pack(pady=10)

unidentified_button = tk.Button(button_frame, text="Export Unidentified to Excel", font=("Arial", 12), command=export_unidentified_to_excel, width=25)
unidentified_button.pack(pady=10)

quit_button = tk.Button(button_frame, text="Quit", font=("Arial", 12), command=quit_app, width=20)
quit_button.pack(pady=10)

# Add bin status labels to status_frame
bin_labels = {
    "non_toxic": tk.Label(status_frame, text="Non-Toxic Bin: OK", fg="green", font=("Arial", 10)),
    "mildly_toxic": tk.Label(status_frame, text="Mildly-Toxic Bin: OK", fg="green", font=("Arial", 10)),
    "highly_toxic": tk.Label(status_frame, text="Highly-Toxic Bin: OK", fg="green", font=("Arial", 10)),
}
for label in bin_labels.values():
    label.pack(anchor="w", pady=2)
shared_state.bin_labels = bin_labels

# Add system status label to status_frame (below bin labels)
status_label = tk.Label(status_frame, text="System ready.", font=("Arial", 10), fg="black")
status_label.pack(pady=10, anchor="w")
shared_state.status_label = status_label

# Start the video feed update loop
update_video_feed()
threading.Thread(target=update_camera_feed, daemon=True).start()
update_sorting_time_label()
run_yolo_detection()
# Run the GUI
root.mainloop()


