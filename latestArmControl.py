import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import time
import json
from datetime import datetime
from robotic_arm import RoboticArm
import cv2
from ultralytics import YOLO
import threading
from queue import Queue, Empty
from threading import Event
from path_planning.pso import PSO
from path_planning.environment import Environment
from shared_state import global_frame, frame_lock, last_annotated_frame, set_status_message, update_bin_status
import shared_state
from report_generator import generate_toxicity_report_graph
import numpy as np
from claw_confirmation import confirm_claw_grip

# Initialize unidentified folder and logging
unidentified_folder = "unidentified"
os.makedirs(unidentified_folder, exist_ok=True)
unidentified_log_file = os.path.join(unidentified_folder, "unidentified.json")

try:
    with open(unidentified_log_file, "r") as file:
        unidentified_data = json.load(file)
        if not isinstance(unidentified_data, list):
            unidentified_data = []
except (FileNotFoundError, json.JSONDecodeError):
    unidentified_data = []

# Initialize summary logging
summary_file = "summary.json"
try:
    with open(summary_file, "r") as file:
        summary_data = json.load(file)
        if not isinstance(summary_data, list):
            summary_data = []
except (FileNotFoundError, json.JSONDecodeError):
    summary_data = []

# Add detection data structure for accurate logging workflow
class DetectionData:
    """
    Data structure to hold detection information until confirmation.
    
    This class ensures that detection data is not logged immediately upon YOLO detection,
    but rather held in memory until the entire pick-and-place operation is confirmed
    through grip confirmation and bin confirmation steps.
    
    Attributes:
        object: The detected object class name
        confidence: YOLO detection confidence score
        timestamp: Detection timestamp
        target_bin: Target bin number for sorting
        toxicity_level: Toxicity classification
        image_path: Path to saved image (for unidentified objects)
        status: Current status of the detection ("detected", "sorted", "grip_failed", etc.)
        sorting_time_sec: Total time taken for sorting operation
        grip_attempts: Number of grip attempts made
        max_grip_attempts: Maximum allowed grip attempts
    """
    def __init__(self, detected_class, confidence, toxicity_level, bin_number, timestamp, image_path=None):
        self.object = detected_class
        self.confidence = confidence
        self.timestamp = timestamp
        self.target_bin = bin_number
        self.toxicity_level = toxicity_level
        self.image_path = image_path
        self.status = "detected"  # Will be updated to "sorted", "grip_failed", or "drop_failed"
        self.sorting_time_sec = None
        self.grip_attempts = 0
        self.max_grip_attempts = 3

def log_to_unidentified(detection_data, reason):
    """Helper function to log detection data to unidentified.json"""
    global unidentified_data
    
    unidentified_entry = {
        "reason": reason,
        "label": detection_data.object,
        "timestamp": detection_data.timestamp,
        "status": detection_data.status
    }
    
    unidentified_data.append(unidentified_entry)
    with open(unidentified_log_file, "w") as file:
        json.dump(unidentified_data, file, indent=4)
    
    print(f"📝 Logged to unidentified.json: {detection_data.object} - {reason}")

def log_to_summary(detection_data):
    """Helper function to log successful detection data to summary.json"""
    global summary_data
    
    summary_entry = {
        "e_waste": detection_data.object,
        "classification": (
            "green" if detection_data.toxicity_level == "non_toxic"
            else "yellow" if detection_data.toxicity_level == "mildly_toxic"
            else "red" if detection_data.toxicity_level == "highly_toxic"
            else ""  # empty string for any unexpected cases
        ),
        "toxicity_level": detection_data.toxicity_level,
        "target_bin": detection_data.target_bin,
        "timestamp": detection_data.timestamp
    }
    
    summary_data.append(summary_entry)
    with open(summary_file, "w") as file:
        json.dump(summary_data, file, indent=4)
    
    print(f"📝 Logged to summary.json: {detection_data.object} - {detection_data.status}")

# Initialize PSO environment
env = Environment(arm_length=100, workspace_limits=(0, 200, 0, 200, 0, 200))

# Load initial path data
with open('path.json', 'r') as f:
    path_data = json.load(f)
    WAYPOINTS = path_data['waypoints']
    ANGLES = path_data['angles']

# Update toxicity map to include all e-waste categories
toxicity_map = {
    "USB flashdrive": "non_toxic",      # Bin 1 (30°)
    "Battery": "highly_toxic",           # Bin 3 (88°),     
    "USB cables": "non_toxic",                # Bin 1 (30°)
    "Sensor": "mildly_toxic",            # Bin 2 (50°)
    "PCB": "mildly_toxic",               # Bin 2 (50°)
    "Unidentified": "unidentified"               # Use 'unidentified' instead of 'N/A'
}

# Add toxicity percentage map
toxicity_percentage_map = {
    "Battery": 95,
    "PCB": 75,
    "Sensor": 55,
    "USB flashdrive": 45,
    "USB cables": 35,
    "Unidentified": 0
}

# Add bin mapping for clarity
bin_mapping = {
    "highly_toxic": 3,    # Bin 3 (88°)
    "mildly_toxic": 2,    # Bin 2 (50°)
    "non_toxic": 1,       # Bin 1 (30°)
    "unidentified": 4     # Bin 4 (110°)
}

# Verify PSO angles match our requirements
print("\n🔍 Verifying PSO angles match bin requirements:")
print(f"Bin 1 (non-toxic, 30°): {ANGLES[1][0]}°")
print(f"Bin 2 (mildly-toxic, 50°): {ANGLES[2][0]}°")
print(f"Bin 3 (highly-toxic, 88°): {ANGLES[3][0]}°")

# Initialize bin tracking
BIN_CAPACITY = 5  # Maximum items per bin
bin_counts = {
    "non_toxic": 0,
    "mildly_toxic": 0,
    "highly_toxic": 0,
    "unidentified": 0
}

def optimize_angles_for_position(target_position, is_pickup=False):
    """Use PSO to optimize angles for pickup positions only"""
    if not is_pickup:
        print("⚠️ PSO optimization skipped for non-pickup movement")
        return None
        
    print("\n🤖 PSO Optimization Starting for pickup...")
    print(f"🎯 Target Position: {target_position}")
    
    pso = PSO(num_particles=20, num_dimensions=4, 
             environment=env, 
             target_position=target_position,
             is_pickup=True)
    
    # Run optimization with fewer iterations for real-time performance
    best_angles, best_score = pso.optimize(max_iterations=20)
    
    # Only use the first 3 angles (base, left, right) from PSO
    optimized_angles = best_angles[:3]
    
    # Print detailed analysis of the best solution
    print("\n📊 Best Solution Analysis:")
    print(f"Base Angle: {optimized_angles[0]:.1f}°")
    print(f"Left Arm: {optimized_angles[1]:.1f}°")
    print(f"Right Arm: {optimized_angles[2]:.1f}°")
    
    # Calculate and show individual scores
    position_error = env.calculate_position_error(best_angles, target_position)
    stability_score = env.calculate_stability_score(best_angles)
    energy_score = env.calculate_energy_score(best_angles)
    smoothness_score = env.calculate_smoothness_score(best_angles)
    
    print("\n📈 Performance Metrics:")
    print(f"Position Error: {position_error:.2f} units")
    print(f"Stability Score: {stability_score} (lower is better)")
    print(f"Energy Score: {energy_score:.1f} (lower is better)")
    print(f"Smoothness Score: {smoothness_score} (lower is better)")
    print(f"Total Score: {best_score:.2f}")
    
    return optimized_angles

def quick_claw_movement(arm: RoboticArm, position: str):
    """Snappy claw movement for gripping and releasing"""
    if position == "open":
        print("👐 Opening claw...")
        arm.move_claw(180)  # Direct movement to open position
        # NO DELAY - needs to be instant for proper grip strength and impact
    elif position == "close":
        print("✊ Gripping...")
        arm.move_claw(0)  # Direct movement to closed position
        # NO DELAY - needs to be instant for proper grip strength and impact
    else:
        print(f"Moving claw to {position}°...")
        arm.move_claw(float(position))  # Direct movement to specified angle
        # NO DELAY - needs to be instant for proper grip strength and impact

def smooth_servo_movement(arm, servo_func, current_angle, target_angle, step_size=10, delay=0.05):
    """Smooth movement for base and arms only"""
    if current_angle == target_angle:
        return

    # Smooth movement for base and arms
    steps = int(abs(target_angle - current_angle) / step_size)
    step = step_size if target_angle > current_angle else -step_size

    for _ in range(steps):
        current_angle += step
        servo_func(current_angle)
        time.sleep(delay)
    
    # Final adjustment
    if current_angle != target_angle:
        servo_func(target_angle)
        time.sleep(delay)

def move_to_angles(arm: RoboticArm, angles, current_angles=[0,0,0,0], is_carrying=False):
    """Move to angles by sending target directly to each servo (no incremental stepping)."""
    print(f"\n📍 Movement Analysis:")
    print(f"Current Angles: {[f'{angle:.1f}°' for angle in current_angles]}")
    print(f"Target Angles: {[f'{angle:.1f}°' for angle in angles]}")
    
    # Calculate angle changes
    changes = [abs(target - current) for target, current in zip(angles, current_angles)]
    print(f"Angle Changes: {[f'{change:.1f}°' for change in changes]}")

    # Use angles directly from path.json without clamping
    target_left = angles[1]
    target_right = angles[2]
    target_base = angles[0]

    # IMPORTANT: Move base FIRST to prevent over-extension
    print("\n🔄 Moving base (direct)...")
    print(f"Base: {current_angles[0]:.1f}° → {target_base:.1f}°")
    arm.move_base(target_base)
    current_angles[0] = target_base
    time.sleep(0.5)  # Give base time to settle
    
    # Then move arms
    print("\n🔄 Moving right arm (direct)...")
    print(f"Right Arm: {current_angles[2]:.1f}° → {target_right:.1f}°")
    arm.move_right(target_right)
    
    print("\n🔄 Moving left arm (direct)...")
    print(f"Left Arm: {current_angles[1]:.1f}° → {target_left:.1f}°")
    arm.move_left(target_left)
    
    # Update current angles (only for base, left, right)
    current_angles[1] = target_left
    current_angles[2] = target_right
    
    print("\n✅ Movement Complete!")
    return current_angles

def move_to_bin(arm: RoboticArm, toxicity_level: str, current_angles):
    """Move to bin with fixed angles from path.json"""
    print(f"\n♻️ Moving to {toxicity_level.replace('_', ' ')} bin...")
    
    # Get bin number and use fixed angles from path.json
    bin_number = bin_mapping.get(toxicity_level, 4)  # Default to bin 4 if not found
    optimized_angles = ANGLES[bin_number][:3]  # Use angles directly from path.json
    
    print(f"Using angles for bin {bin_number}: {optimized_angles}")
    
    # Move arms smoothly while maintaining grip
    current_angles = move_to_angles(arm, optimized_angles, current_angles)
    
    # Quick release only after reaching the bin
    quick_claw_movement(arm, "open")
    
    return current_angles

def process_frame(frame, model):
    """Process a single frame with YOLO model"""
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    results = model(frame, verbose=False)
    detected_class = None
    highest_conf = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf > highest_conf:
                class_id = int(box.cls[0])
                detected_class = model.names[class_id]
                highest_conf = conf
                
                # Draw detection
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                toxicity = toxicity_map.get(detected_class, "unknown")
                color = (0, 255, 0)  # Green for non-toxic
                if toxicity == "highly_toxic":
                    color = (0, 0, 255)  # Red
                elif toxicity == "mildly_toxic":
                    color = (0, 165, 255)  # Orange
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{detected_class} ({conf:.2f}) - {toxicity}"
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, detected_class, highest_conf

def detection_thread_func(model, cap, detection_queue, stop_event, detection_pause_event):
    """
    Continuously process frames and enqueue detections, pausing when requested.
    
    This function now creates DetectionData objects instead of immediately logging.
    All logging is deferred until grip confirmation and bin confirmation succeed
    in the arm_thread_func.
    """
    CONFIDENCE_THRESHOLD = 0.4  # Set threshold for confirmed detections
    
    was_paused = False
    # Remove immediate unidentified logging - will be handled after grip confirmation
    last_save_time = time.time()
    SAVE_INTERVAL = 5  # Save unidentified data every 5 seconds
    
    while not stop_event.is_set():
        if detection_pause_event.is_set():
            time.sleep(0.05)
            was_paused = True
            continue
        
        if was_paused:
            print("[INFO] Detection just resumed. Flushing camera buffer...")
            # Clear the shared state frame
            with frame_lock:
                shared_state.current_display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Single buffer clear is sufficient
            cap.grab()
            was_paused = False
            continue
            
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Process frame at consistent size - do this once
        process_frame = cv2.resize(frame, (640, 480))
        results = model(process_frame, verbose=False)
        
        # Always make a copy of the frame for drawing
        display_frame = process_frame.copy()
        has_detections = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                detected_class = model.names[class_id]
                
                # Check if detected class is in toxicity map
                known_labels = set(toxicity_map.keys()) - {"default"}
                
                # Case 1: Low confidence OR not in toxicity map -> unidentified
                if conf < CONFIDENCE_THRESHOLD or detected_class not in known_labels:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    image_path = f"unidentified/no_detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(image_path, process_frame)
                    
                    detection_data = DetectionData(
                        detected_class=detected_class,
                        confidence=conf,
                        toxicity_level="unidentified",
                        bin_number=bin_mapping["unidentified"],
                        timestamp=timestamp,
                        image_path=image_path
                    )
                    
                    # Set appropriate status based on reason
                    if conf < CONFIDENCE_THRESHOLD:
                        detection_data.status = "low_confidence"
                        print(f"[DEBUG] Queuing low confidence detection: {detected_class}, conf: {conf}")
                    else:
                        detection_data.status = "unknown_object"
                        print(f"[DEBUG] Queuing unknown object detection: {detected_class}, conf: {conf}")
                    
                    detection_queue.put(detection_data)
                    continue
                
                # Case 2: Good confidence AND in toxicity map -> process normally
                has_detections = True
                print(f"[DEBUG] YOLO detected: {detected_class}")
                
                toxicity_level = toxicity_map.get(detected_class)
                bin_number = bin_mapping.get(toxicity_level, 4)  # Default to bin 4 if not found
                print(f"[DEBUG] Mapped toxicity_level: {toxicity_level} → bin {bin_number}")
                
                # Update status message
                set_status_message(f"Detected: {detected_class} ({conf:.2f})", "black")
                
                # Create DetectionData object for confirmed detection
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detection_data = DetectionData(
                    detected_class=detected_class,
                    confidence=conf,
                    toxicity_level=toxicity_level,
                    bin_number=bin_number,
                    timestamp=timestamp
                )
                
                # Add DetectionData to queue for processing
                print(f"[DEBUG] Queuing confirmed detection: {detected_class}, toxicity: {toxicity_level}, bin: {bin_number}, conf: {conf}")
                detection_queue.put(detection_data)
                
                # Draw detection for display with correct colors
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)  # Default green for non-toxic
                if toxicity_level == "highly_toxic":
                    color = (0, 0, 255)  # Red for highly toxic
                elif toxicity_level == "mildly_toxic":
                    color = (0, 165, 255)  # Orange for mildly toxic
                elif toxicity_level == "unidentified":
                    color = (0,0,0)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                toxicity_percent = toxicity_percentage_map.get(detected_class, 0)  # Use 0 as direct default
                label = f"{detected_class} ({toxicity_percent}%) - {toxicity_level} (Bin {bin_number})"
                cv2.putText(display_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update shared state frame - minimize lock time
        # Always show the frame, whether it has detections or not
        with frame_lock:
            shared_state.current_display_frame = display_frame
            shared_state.last_annotated_frame_cam1 = display_frame
        
        # Note: Unidentified data is now logged after grip confirmation in arm_thread_func

def arm_thread_func(arm, model, detection_queue, stop_event, detection_pause_event):
    """
    Wait for detections and execute pick-and-place with accurate logging workflow.
    
    This function implements the complete workflow:
    1. Receive DetectionData from detection thread
    2. Attempt pickup and grip confirmation
    3. Move to target bin and drop
    4. Confirm bin count increase
    5. Log to summary.json only after successful grip + bin confirmation
    6. Log failures to unidentified.json with detailed status
    """
    global bin_counts, unidentified_data, summary_data
    current_angles = [0, 0, 90, 0]  # Initialize with default position
    
    # Hardcoded angles
    PICKUP_ANGLES = [0, 0, 140, 0]  # [base, left, right, claw] - fixed pickup position 
    DEFAULT_ANGLES = [0, 0, 90, 0]   # [base, left, right, claw] - safe default position
    
    while not stop_event.is_set():
        try:
            # Get DetectionData object from queue
            detection_data = detection_queue.get(timeout=0.1)
            print(f"\n🎯 Processing detection: {detection_data.object} with confidence {detection_data.confidence:.2f}")
            print(f"📦 Toxicity level: {detection_data.toxicity_level}")
            print(f"🗑️  Assigned to Bin {detection_data.target_bin}")
            print(f"📊 Status: {detection_data.status}")
            print("🤖 Starting movement sequence...")
            
            # Start timing the sorting operation
            start_time = time.time()
            
            # Check if bin is full
            if bin_counts[detection_data.toxicity_level] >= BIN_CAPACITY:
                print(f"⚠️ {detection_data.toxicity_level.replace('_', ' ').title()} bin is full!")
                set_status_message(f"⚠️ {detection_data.toxicity_level.replace('_', ' ').title()} bin is full!", "red")
                update_bin_status(detection_data.toxicity_level, "FULL")
                
                # Log bin full failure to unidentified
                detection_data.status = "bin_full"
                log_to_unidentified(detection_data, "Bin full")
                continue
            
            # Update status message
            set_status_message(f"Sorting {detection_data.object} to {detection_data.toxicity_level} bin...", "black")

            # Pause detection
            detection_pause_event.set()
            while not detection_queue.empty():
                try:
                    detection_queue.get_nowait()
                except Empty:
                    break

            # === INITIALIZATION ===
            print("\n=== INITIALIZATION ===")
            # Start with claw open in default position (pickup location)
            print("[ACTION] Moving to pickup position...")
            quick_claw_movement(arm, "open")
            time.sleep(1.0)
            
            # Move to pickup position using hardcoded angles
            print("[ACTION] Moving to pickup position using fixed angles...")
            print(f"Pickup angles: Base={PICKUP_ANGLES[0]}°, Left={PICKUP_ANGLES[1]}°, Right={PICKUP_ANGLES[2]}°, Claw={PICKUP_ANGLES[3]}°")
            current_angles = move_to_angles(arm, PICKUP_ANGLES[:3], current_angles)
            time.sleep(1.0)
            
            # === PICKUP SEQUENCE ===
            print("\n=== PICKUP SEQUENCE ===")
            # 1. Verify pickup position
            print("\n[VERIFY] Checking pickup position...")
            print(f"Current angles: Base={arm.last_positions[arm.base_channel]:.1f}°, Left={arm.last_positions[arm.shoulder_channel]:.1f}°, Right={arm.last_positions[arm.right_channel]:.1f}°")
            
            # 2. Lower for pickup
            print("\n[ACTION] Lowering for pickup...")
            pickup_lower_angles = PICKUP_ANGLES.copy()
            pickup_lower_angles[2] = 150  # Lower by moving right arm down 10 degrees (from 140 to 150)
            print(f"Lower angles: Base={pickup_lower_angles[0]}°, Left={pickup_lower_angles[1]}°, Right={pickup_lower_angles[2]}°")
            current_angles = move_to_angles(arm, pickup_lower_angles[:3], current_angles)
            time.sleep(1.0)
            
            # 3. Grip object
            print("\n[ACTION] Gripping object...")
            quick_claw_movement(arm, "close")
            time.sleep(1.5)

            # 3.5. Lift right servo to 90° after grip, before confirmation
            print("\n[ACTION] Lifting right servo to 90° after grip, before confirmation...")
            arm.move_right(90)
            time.sleep(1.0)

            # === CLAW CONFIRMATION ===
            print("\n[CHECK] Confirming claw grip with camera...")
            grip_success = confirm_claw_grip(
                shared_state.camera, model,
                max_retries=3, confidence_threshold=0.3, target_classes=[detection_data.object], verbose=True
            )
            print(f"[DEBUG] Grip success: {grip_success} for class: {detection_data.object}")
            
            # Handle grip failure with proper logging
            if not grip_success:
                print("[FAIL] Claw failed to grip item after retries. Logging to unidentified.")
                detection_data.status = "grip_failed"
                detection_data.sorting_time_sec = round(time.time() - start_time, 2)
                
                # Log grip failure to unidentified
                log_to_unidentified(detection_data, "Grip confirmation failed")
                
                # Open claw and return to default position
                quick_claw_movement(arm, "open")
                current_angles = move_to_angles(arm, DEFAULT_ANGLES[:3], current_angles)
                detection_pause_event.clear()
                continue
            
            # 4. Small lift after grip to prevent dragging
            print("\n[ACTION] Lifting object slightly...")
            lift_angles = PICKUP_ANGLES.copy()
            lift_angles[2] = 130  # Lift by moving right arm up 10 degrees (from 140 to 130)
            print(f"Lift angles: Base={lift_angles[0]}°, Left={lift_angles[1]}°, Right={lift_angles[2]}°")
            current_angles = move_to_angles(arm, lift_angles[:3], current_angles, is_carrying=True)
            time.sleep(1.0)
            
            # DROP SEQUENCE
            print("\n=== DROP SEQUENCE ===")
            print(f"[DEBUG] Proceeding to sort {detection_data.object} to bin {detection_data.target_bin} (toxicity: {detection_data.toxicity_level})")
            # 4. Get optimized bin angles from path.json
            bin_angles = ANGLES[detection_data.target_bin][:3]  # Get first 3 angles (base, left, right) from PSO-optimized path
            print(f"\n[ACTION] Using PSO-optimized angles for bin {detection_data.target_bin}...")
            print(f"Bin angles from path.json: Base={bin_angles[0]}°, Left={bin_angles[1]}°, Right={bin_angles[2]}°")
            
            # Verify current angles before movement
            print("\n[DEBUG] Current angles before bin movement:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}°")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}°")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}°")
            
            # Move to optimized bin position while maintaining 90° height for carrying
            print("\n[ACTION] Moving to bin position while maintaining 90° height...")
            # Use bin angles for base and left, but keep right arm at 90° for carrying
            carry_angles = [bin_angles[0], bin_angles[1], 90]  # [base, left, right=90°]
            print(f"Carry angles: Base={carry_angles[0]}°, Left={carry_angles[1]}°, Right={carry_angles[2]}° (maintaining height)")
            current_angles = move_to_angles(arm, carry_angles, current_angles, is_carrying=True)
            time.sleep(1.5)
            
            # Verify angles after movement
            print("\n[DEBUG] Angles after reaching bin position:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}° (target was {carry_angles[0]}°)")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}° (target was {carry_angles[1]}°)")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}° (target was {carry_angles[2]}°)")
            
            # 5. Lower for drop using bin angles
            print("\n[ACTION] Lowering for drop...")
            # Use the original bin angles for drop (lower right arm to 115°)
            drop_angles = bin_angles.copy()  # Use exact same angles from path.json
            print(f"Drop angles: Base={drop_angles[0]}°, Left={drop_angles[1]}°, Right={drop_angles[2]}° (lowering to drop)")
            
            # Verify current angles before drop movement
            print("\n[DEBUG] Current angles before drop movement:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}°")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}°")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}°")
            
            # Update current_angles with actual servo positions before drop movement
            current_angles[0] = arm.last_positions[arm.base_channel]
            current_angles[1] = arm.last_positions[arm.shoulder_channel]
            current_angles[2] = arm.last_positions[arm.right_channel]
            
            current_angles = move_to_angles(arm, drop_angles, current_angles, is_carrying=True)
            
            # Verify final angles after drop movement
            print("\n[DEBUG] Final angles after drop movement:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}° (target was {drop_angles[0]}°)")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}° (target was {drop_angles[1]}°)")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}° (target was {drop_angles[2]}°)")
            
            # 6. Release object
            print("\n[ACTION] Releasing object...")
            quick_claw_movement(arm, "open")
            time.sleep(1.5)
            
            # Increment bin count
            bin_counts[detection_data.toxicity_level] += 1
            print(f"📊 Bin status - {detection_data.toxicity_level}: {bin_counts[detection_data.toxicity_level]}/{BIN_CAPACITY}")
            
            # Update bin status in GUI
            if bin_counts[detection_data.toxicity_level] >= BIN_CAPACITY:
                update_bin_status(detection_data.toxicity_level, "FULL")
            else:
                update_bin_status(detection_data.toxicity_level, "OK")
            
            # === RETURN SEQUENCE ===
            print("\n=== RETURN SEQUENCE ===")
            # 7. Move back up to bin position
            print("[ACTION] Moving back up from drop position...")
            current_angles = move_to_angles(arm, bin_angles, current_angles)
            time.sleep(1.0)
            
            # 8. Move directly to default position
            print("[ACTION] Moving to default position...")
            print(f"Default angles: Base={DEFAULT_ANGLES[0]}°, Left={DEFAULT_ANGLES[1]}°, Right={DEFAULT_ANGLES[2]}°")
            current_angles = move_to_angles(arm, DEFAULT_ANGLES[:3], current_angles)
            time.sleep(1.5)
            
            # === SUCCESSFUL SORT LOGGING ===
            # Log to summary after successful sorting
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update detection data with success status
            detection_data.status = "sorted"
            detection_data.sorting_time_sec = round(elapsed_time, 2)
            
            print(f"\n✅ SUCCESS: {detection_data.object} successfully sorted to {detection_data.toxicity_level} bin")
            print(f"📊 Final status: {detection_data.status}")
            print(f"⏱️  Total sorting time: {detection_data.sorting_time_sec} seconds")
            
            # Log successful sort to summary
            log_to_summary(detection_data)
            
            # Generate updated toxicity report
            generate_toxicity_report_graph()
            
            # Flush camera buffer after successful sorting (entire operation complete)
            print("[INFO] Flushing camera buffer after successful sorting...")
            if hasattr(shared_state, 'camera') and shared_state.camera is not None:
                shared_state.camera.grab()
            with frame_lock:
                shared_state.current_display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Resume detection
            detection_pause_event.clear()
            print("\n✅ Movement sequence completed. Ready for next detection.")
            
        except Exception as e:
            if isinstance(e, Empty):
                continue
            print(f"\n⚠️ Error during movement sequence: {str(e)}")
            
            # Log error to unidentified if we have detection data
            if 'detection_data' in locals():
                detection_data.status = "movement_error"
                detection_data.sorting_time_sec = round(time.time() - start_time, 2)
                log_to_unidentified(detection_data, f"Movement sequence error: {str(e)}")
            
            try:
                print("[ACTION] Error recovery: Moving to safe position...")
                quick_claw_movement(arm, "open")
                # Return to default position first
                current_angles = move_to_angles(arm, DEFAULT_ANGLES[:3], current_angles)
                time.sleep(1.5)
                # Then to pickup position
                current_angles = move_to_angles(arm, PICKUP_ANGLES[:3], current_angles)
                time.sleep(1.5)
            except:
                pass
            
            # Resume detection after error
            detection_pause_event.clear()

def detect_and_sort(arm: RoboticArm):
    """Main function with threaded detection and arm control, with default position logic."""
    global bin_counts
    stop_event = threading.Event()
    detection_queue = Queue()
    detection_pause_event = Event()
    detection_pause_event.set()  # Start with detection paused
    
    # Reset bin counts at start
    bin_counts = {
        "non_toxic": 0,
        "mildly_toxic": 0,
        "highly_toxic": 0,
        "unidentified": 0
    }
    
    # Initialize bin status in GUI
    for toxicity_level in bin_counts:
        update_bin_status(toxicity_level, "OK")
    
    try:
        print("\n🤖 Loading YOLOv8 model...")
        model = YOLO('train32/weights/best.pt')
        model.conf = 0.6
        print("✅ Model loaded successfully!")
        
        # Use the camera instance from shared_state
        if not shared_state.camera or not shared_state.camera.isOpened():
            print("Error: Camera not initialized")
            return

        print("\n📊 Using PSO-generated waypoints and angles:")
        print(f"Number of waypoints: {len(WAYPOINTS)}")
        print(f"Number of angle sets: {len(ANGLES)}")
        print("\nPSO Angles for each position:")
        for i, angles in enumerate(ANGLES):
            print(f"Position {i}: {angles}")
        # Move to default position before starting detection
        print("\n[ACTION] Moving arm to default position before starting detection...")
        arm.default_position()
        print("[INFO] Arm in default position. Starting detection loop.")
        detection_pause_event.clear()  # Allow detection
        # Start threads
        detection_thread = threading.Thread(target=detection_thread_func, args=(model, shared_state.camera, detection_queue, stop_event, detection_pause_event))
        arm_thread = threading.Thread(target=arm_thread_func, args=(arm, model, detection_queue, stop_event, detection_pause_event))
        detection_thread.start()
        arm_thread.start()
        detection_thread.join()
        stop_event.set()
        arm_thread.join()
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        stop_event.set()
    finally:
        print("🤖 Stopping robot...")
        try:
            quick_claw_movement(arm, "open")
            arm.move_base(0)
            arm.move_left(0)
            arm.move_right(0)
            arm.cleanup()
            print("✅ Robot stopped safely")
        except Exception as e:
            print(f"⚠️ Error stopping robot: {str(e)}")
        
        # Save final data and generate report
        try:
            with open(summary_file, "w") as file:
                json.dump(summary_data, file, indent=4)
            with open(unidentified_log_file, "w") as file:
                json.dump(unidentified_data, file, indent=4)
            generate_toxicity_report_graph()
            print("\n📝 Summary and unidentified data saved")
        except Exception as e:
            print(f"⚠️ Error saving final data: {str(e)}")
        
        print("\n👋 Program terminated")

if __name__ == "main":
    arm = RoboticArm()
    detect_and_sort(arm)