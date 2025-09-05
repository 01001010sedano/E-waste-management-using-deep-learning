import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import time
import json
from robotic_arm import RoboticArm
import cv2
from ultralytics import YOLO
import threading
from queue import Queue, Empty
from threading import Event
from path_planning.pso import PSO
from path_planning.environment import Environment

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
    "sensor": "mildly_toxic",            # Bin 2 (50°)
    "PCB": "mildly_toxic",               # Bin 2 (50°)
    "default": "non_toxic"               # Bin 1 (30°)
}

# Add bin mapping for clarity
bin_mapping = {
    "highly_toxic": 3,    # Bin 3 (88°)
    "mildly_toxic": 2,    # Bin 2 (50°)
    "non_toxic": 1        # Bin 1 (30°)
}

# Verify PSO angles match our requirements
print("\n🔍 Verifying PSO angles match bin requirements:")
print(f"Bin 1 (non-toxic, 30°): {ANGLES[1][0]}°")
print(f"Bin 2 (mildly-toxic, 50°): {ANGLES[2][0]}°")
print(f"Bin 3 (highly-toxic, 88°): {ANGLES[3][0]}°")

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
        time.sleep(0.1)  # Short delay after movement
    elif position == "close":
        print("✊ Gripping...")
        arm.move_claw(0)  # Direct movement to closed position
        time.sleep(0.1)  # Short delay after movement
    else:
        print(f"Moving claw to {position}°...")
        arm.move_claw(float(position))  # Direct movement to specified angle
        time.sleep(0.1)  # Short delay after movement

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
    bin_number = bin_mapping[toxicity_level]
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
    """Continuously process frames and enqueue detections, pausing when requested."""
    CONFIDENCE_THRESHOLD = 0.5  # Only process detections with confidence > 50%
    
    while not stop_event.is_set():
        if detection_pause_event.is_set():
            time.sleep(0.05)
            continue
            
        # Clear buffer
        for _ in range(2):
            cap.grab()
            
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
                
                # Add detection to queue
                detection_queue.put((detected_class, toxicity_level, bin_number, conf))
                
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
                
        # Display the processed frame
        cv2.imshow('E-Waste Detection', process_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def arm_thread_func(arm, detection_queue, stop_event, detection_pause_event):
    """Wait for detections and execute pick-and-place with PSO optimization for bin movements only"""
    current_angles = [0, 0, 90, 0]  # Initialize with default position
    
    # Hardcoded angles
    PICKUP_ANGLES = [0, 0, 140, 0]  # [base, left, right, claw] - fixed pickup position 
    DEFAULT_ANGLES = [0, 0, 90, 0]   # [base, left, right, claw] - safe default position
    
    while not stop_event.is_set():
        try:
            detected_class, toxicity_level, bin_number, conf = detection_queue.get(timeout=0.1)
            print(f"\n🎯 Detected {detected_class} with confidence {conf:.2f}")
            print(f"📦 Toxicity level: {toxicity_level}")
            print(f"🗑️  Assigned to Bin {bin_number}")
            print("🤖 Starting movement sequence...")
            
            # Pause detection
            detection_pause_event.set()
            
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
            
            # 2. Grip object
            print("\n[ACTION] Gripping object...")
            quick_claw_movement(arm, "close")
            time.sleep(1.5)
            
            # 3. Small lift after grip to prevent dragging
            print("\n[ACTION] Lifting object slightly...")
            lift_angles = PICKUP_ANGLES.copy()
            lift_angles[2] = 130  # Lift by moving right arm up 10 degrees (from 140 to 130)
            print(f"Lift angles: Base={lift_angles[0]}°, Left={lift_angles[1]}°, Right={lift_angles[2]}°")
            current_angles = move_to_angles(arm, lift_angles[:3], current_angles, is_carrying=True)
            time.sleep(1.0)
            
            # === DROP SEQUENCE ===
            print("\n=== DROP SEQUENCE ===")
            # 4. Get optimized bin angles from path.json
            bin_angles = ANGLES[bin_number][:3]  # Get first 3 angles (base, left, right) from PSO-optimized path
            print(f"\n[ACTION] Using PSO-optimized angles for bin {bin_number}...")
            print(f"Bin angles from path.json: Base={bin_angles[0]}°, Left={bin_angles[1]}°, Right={bin_angles[2]}°")
            
            # Verify current angles before movement
            print("\n[DEBUG] Current angles before bin movement:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}°")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}°")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}°")
            
            # Move to optimized bin position
            current_angles = move_to_angles(arm, bin_angles, current_angles, is_carrying=True)
            time.sleep(1.5)
            
            # Verify angles after movement
            print("\n[DEBUG] Angles after reaching bin position:")
            print(f"Base: {arm.last_positions[arm.base_channel]:.1f}° (target was {bin_angles[0]}°)")
            print(f"Left: {arm.last_positions[arm.shoulder_channel]:.1f}° (target was {bin_angles[1]}°)")
            print(f"Right: {arm.last_positions[arm.right_channel]:.1f}° (target was {bin_angles[2]}°)")
            
            # 5. Lower for drop using bin angles
            print("\n[ACTION] Lowering for drop...")
            # Use the same angles as bin position for drop
            drop_angles = bin_angles.copy()  # Use exact same angles from path.json
            print(f"Drop angles (using same as bin position): Base={drop_angles[0]}°, Left={drop_angles[1]}°, Right={drop_angles[2]}°")
            
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
            
            # Resume detection
            detection_pause_event.clear()
            print("\n✅ Movement sequence completed. Ready for next detection.")
            
        except Exception as e:
            if isinstance(e, Empty):
                continue
            print(f"\n⚠️ Error during movement sequence: {str(e)}")
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

def detect_and_sort(arm: RoboticArm):
    """Main function with threaded detection and arm control, with default position logic."""
    cap = None
    stop_event = threading.Event()
    detection_queue = Queue()
    detection_pause_event = Event()
    detection_pause_event.set()  # Start with detection paused
    try:
        print("\n🤖 Loading YOLOv8 model...")
        model = YOLO('train32/weights/best.pt')
        model.conf = 0.6
        print("✅ Model loaded successfully!")
        print("\n🎥 Initializing camera...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("⚠️ Error: Could not open camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.read()
        print("✅ Camera initialized successfully!")
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
        detection_thread = threading.Thread(target=detection_thread_func, args=(model, cap, detection_queue, stop_event, detection_pause_event))
        arm_thread = threading.Thread(target=arm_thread_func, args=(arm, detection_queue, stop_event, detection_pause_event))
        detection_thread.start()
        arm_thread.start()
        detection_thread.join()
        stop_event.set()
        arm_thread.join()
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        stop_event.set()
    finally:
        if cap is not None:
            print("🎥 Releasing camera...")
            cap.release()
            cv2.destroyAllWindows()
        print("🤖 Stopping robot...")
        try:
            quick_claw_movement(arm, "open")
            arm.move_base(0)
            arm.move_left(min(0, 80))
            arm.move_right(0)
            arm.cleanup()
            print("✅ Robot stopped safely")
        except Exception as e:
            print(f"⚠️ Error stopping robot: {str(e)}")
        print("\n👋 Program terminated")

if __name__ == "__main__":
    arm = RoboticArm()
    detect_and_sort(arm)