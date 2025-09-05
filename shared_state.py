import threading
import numpy as np

# Shared frame for annotation
global_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Lock for thread-safe access
frame_lock = threading.Lock()

latest_raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
last_annotated_frame = None

# Status label holder and setter (initialized from gui_app)
status_label = None

# Camera instances for two cameras
camera = None  # Camera 1 (main camera)
camera2 = None  # Camera 2 (second camera)
camera_lock = threading.Lock()
camera2_lock = threading.Lock()

# Frames for each camera
latest_raw_frame_cam1 = np.zeros((480, 640, 3), dtype=np.uint8)
latest_raw_frame_cam2 = np.zeros((480, 640, 3), dtype=np.uint8)
last_annotated_frame_cam1 = None
last_annotated_frame_cam2 = None

# shared_state.py
current_display_frame = None  # 💡 this will hold the last frame with boxes

def set_status_message(msg, color="black"):
    """Update the GUI status message from any thread-safe context."""
    if status_label:
        def update():
            status_label.config(text=msg, fg=color)
        if threading.current_thread() is threading.main_thread():
            update()
        else:
            status_label.after(0, update)

confirm_trigger = False

# Bin status labels
bin_labels = {}

def mark_bin_full(bin_type):
    if bin_type in bin_labels:
        def update():
            bin_labels[bin_type].config(text=f"{bin_type.replace('_', ' ').title()} Bin: FULL", fg="red")
        if threading.current_thread() is threading.main_thread():
            update()
        else:
            bin_labels[bin_type].after(0, update)

def update_bin_status(bin_type, status):
    """Update the bin status in the GUI from any thread-safe context."""
    if bin_type in bin_labels:
        def update():
            color = "red" if status == "FULL" else "green"
            bin_labels[bin_type].config(text=f"{bin_type.replace('_', ' ').title()} Bin: {status}", fg=color)
        if threading.current_thread() is threading.main_thread():
            update()
        else:
            bin_labels[bin_type].after(0, update)
