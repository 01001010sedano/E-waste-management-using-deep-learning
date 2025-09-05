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
from shared_state import global_frame, frame_lock, last_annotated_frame, set_status_message
import shared_state
from report_generator import generate_toxicity_report_graph
import numpy as np

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

# Initialize PSO environment
env = Environment(arm_length=100, workspace_limits=(0, 200, 0, 200, 0, 200))

# Load initial path data
with open('path.json', 'r') as f:
    path_data = json.load(f)
    WAYPOINTS = path_data['waypoints']
    ANGLES = path_data['angles']

# Update toxicity map to include all e-waste categories
toxicity_map = {
    "USB flashdrive": "non_toxic",       
    "Battery": "highly_toxic",              
    "USB cables": "non_toxic",           
    "Sensor": "mildly_toxic",            
    "PCB": "mildly_toxic",             
    "default": "non_toxic"      
}

# Add bin mapping for clarity
bin_mapping = {
    "highly_toxic": 3,    
    "mildly_toxic": 2,    
    "non_toxic": 1        
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
    """Quick claw movement for gripping and releasing"""
    if position == "open":
        print("👐 Opening claw...")
        arm.move_claw(180)  # Open claw wider
        time.sleep(0.5)
    elif position == "close":
        print("✊ Gripping...")
        arm.move_claw(0)    # Close claw to 0° for better grip
        time.sleep(0.5)
    else:
        print(f"Moving claw to {position}°...")
        arm.move_claw(position)
        time.sleep(0.5)

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
    """Move to angles with direct base movement"""
    print(f"\n📍 Movement Analysis:")
    print(f"Current Angles: {[f'{angle:.1f}°' for angle in current_angles]}")
    print(f"Target Angles: {[f'{angle:.1f}°' for angle in angles]}")
    
    # Calculate angle changes
    changes = [abs(target - current) for target, current in zip(angles, current_angles)]
    print(f"Angle Changes: {[f'{change:.1f}°' for change in changes]}")
    
    # 1. Move right arm directly
    print("\n🔄 Moving right arm...")
    print(f"Right Arm: {current_angles[2]:.1f}° → {angles[2]:.1f}°")
    arm.move_right(angles[2])
    time.sleep(0.5)
    
    # 2. Move left arm directly
    print("\n🔄 Moving left arm...")
    print(f"Left Arm: {current_angles[1]:.1f}° → {angles[1]:.1f}°")
    arm.move_left(angles[1])
    time.sleep(0.5)
    
    # 3. Move base directly
    print("\n🔄 Moving base...")
    print(f"Base: {current_angles[0]:.1f}° → {angles[0]:.1f}°")
    arm.move_base(angles[0])
    current_angles[0] = angles[0]
    time.sleep(0.5)
    
    # Update current angles (only for base, left, right)
    current_angles[1] = angles[1]
    current_angles[2] = angles[2]
    
    print("\n✅ Movement Complete!")
    return current_angles

def move_to_bin(arm: RoboticArm, toxicity_level: str, current_angles):
    """Move to bin with fixed angles from path.json"""
    print(f"\n♻️ Moving to {toxicity_level.replace('_', ' ')} bin...")
    
    # Get bin number and use fixed angles from path.json
    bin_number = bin_mapping[toxicity_level]
    optimized_angles = ANGLES[bin_number][:3]
    print(f"DEBUG → Using fixed angles for bin {bin_number}: {optimized_angles}")
    
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
    
    was_paused = False
    
    while not stop_event.is_set():
        if detection_pause_event.is_set():
            time.sleep(0.05)
            was_paused = True
            continue
        
        if was_paused:
            print("[INFO] Detection just resumed. Flushing camera buffer...")
            for _ in range(5):
                cap.grab()
            cap.read()
            time.sleep(0.3)  # give time for frame to refresh
            was_paused = False
        
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
                class_id = int(box.cls[0])
                detected_class = model.names[class_id]
                
                # Check for unidentified objects or low confidence
                known_labels = set(toxicity_map.keys()) - {"default"}
                if conf < 0.7 or detected_class not in known_labels:
                    reason = "Low confidence detection" if conf < 0.7 else "Unknown object"
                    image_path = f"unidentified/no_detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(image_path, process_frame)
                    unidentified_entry = {
                        "reason": reason,
                        "label": detected_class,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": image_path
                    }
                    unidentified_data.append(unidentified_entry)
                    with open(unidentified_log_file, "w") as file:
                        json.dump(unidentified_data, file, indent=4)
                    continue
                
                # Only process high confidence detections
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                    
                print(f"[DEBUG] YOLO detected: {detected_class}")
                
                toxicity_level = toxicity_map.get(detected_class, toxicity_map["default"])
                bin_number = bin_mapping[toxicity_level]
                print(f"[DEBUG] Mapped toxicity_level: {toxicity_level} → bin {bin_number}")
                
                # Update status message
                set_status_message(f"Detected: {detected_class} ({conf:.2f})", "black")
                
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
                
        # Update shared state frame
        with frame_lock:
            shared_state.current_display_frame = process_frame.copy()
                
        # Display the processed frame
        #cv2.imshow('E-Waste Detection', process_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    stop_event.set()
        #    break

def arm_thread_func(arm, detection_queue, stop_event, detection_pause_event):
    """Wait for detections and execute pick-and-place with optimized movements"""
    global summary_data 
    current_angles = [0, 88, 90, 0]  # Initialize with default position
    
    # Hardcoded pickup position angles
    PICKUP_ANGLES = [0, 88, 145, 0]  # [base, left, right, claw] - fixed pickup position
    LIFT_ANGLES = [0, 88, 175, 0]    # [base, left, right, claw] - lifted position
    
    while not stop_event.is_set():
        try:
            detected_class, toxicity_level, bin_number, conf = detection_queue.get(timeout=0.1)
            print(f"\n🎯 Detected {detected_class} with confidence {conf:.2f}")
            print(f"📦 Toxicity level: {toxicity_level}")
            print(f"🗑️  Assigned to Bin {bin_number}")
            print("🤖 Starting movement sequence...")
            
            # Update status message
            set_status_message(f"Sorting {detected_class} to {toxicity_level} bin...", "black")

        

            # Pause detection
            detection_pause_event.set()
            while not detection_queue.empty():
                try:
                    detection_queue.get_nowait()
                except Empty:
                    break
            
            # === INITIALIZATION ===
            print("\n=== INITIALIZATION ===")
            # Start with claw open in default position
            print("[ACTION] Moving to default position...")
            quick_claw_movement(arm, "open")  # Ensure claw is open
            time.sleep(1.0)
            arm.default_position()
            current_angles = [0, 88, 90, 0]
            time.sleep(1.0)
            
            # === PICKUP SEQUENCE ===
            print("\n=== PICKUP SEQUENCE ===")
            # 1. Move to hardcoded pickup position
            print("[ACTION] Moving to pickup position...")
            print(f"Pickup angles: Base={PICKUP_ANGLES[0]}°, Left={PICKUP_ANGLES[1]}°, Right={PICKUP_ANGLES[2]}°")
            
            # Move each servo directly to pickup position
            print("\n[ACTION] Moving right arm to pickup position...")
            arm.move_right(PICKUP_ANGLES[2])
            time.sleep(1.0)
            
            print("[ACTION] Moving left arm to pickup position...")
            arm.move_left(PICKUP_ANGLES[1])
            time.sleep(1.0)
            
            print("[ACTION] Moving base to pickup position...")
            arm.move_base(PICKUP_ANGLES[0])
            time.sleep(1.0)
            
            # 2. Verify we're at pickup position
            print("\n[VERIFY] Checking pickup position...")
            print(f"Current angles: Base={arm.last_positions[arm.base_channel]:.1f}°, Left={arm.last_positions[arm.shoulder_channel]:.1f}°, Right={arm.last_positions[arm.right_channel]:.1f}°")
            
            # 3. Ensure claw is open at pickup
            print("\n[VERIFY] Checking claw at pickup...")
            print(f"Current claw angle: {arm.last_positions[arm.claw_channel]:.1f}°")
            if arm.last_positions[arm.claw_channel] != 180:
                print("[ACTION] Opening claw for pickup...")
                quick_claw_movement(arm, "open")
                time.sleep(1.0)
            
            # 4. Grip object at pickup location
            print("\n[ACTION] Gripping object at pickup location...")
            print(f"Current claw angle before grip: {arm.last_positions[arm.claw_channel]:.1f}°")
            quick_claw_movement(arm, "close")
            time.sleep(1.5)  # Longer wait for grip to complete
            
            # Verify grip at pickup
            print("\n[VERIFY] Checking grip at pickup...")
            print(f"Current claw angle after grip: {arm.last_positions[arm.claw_channel]:.1f}°")
            if arm.last_positions[arm.claw_channel] != 0:
                print("⚠️ Grip not complete at pickup, retrying...")
                quick_claw_movement(arm, "close")
                time.sleep(1.5)
            
            # 5. Lift object to safe height
            print("\n[ACTION] Lifting object to safe height...")
            print(f"Lift angles: Base={LIFT_ANGLES[0]}°, Left={LIFT_ANGLES[1]}°, Right={LIFT_ANGLES[2]}°")
            
            # Move right arm to lift position
            print("[ACTION] Moving right arm to lift position...")
            arm.move_right(LIFT_ANGLES[2])
            time.sleep(1.5)  # Wait for lift to complete
            
            # === DROP SEQUENCE ===
            print("\n=== DROP SEQUENCE ===")
            # Only proceed to bin if grip and lift are complete
            print("\n[VERIFY] Checking if ready for drop sequence...")
            print(f"Claw angle: {arm.last_positions[arm.claw_channel]:.1f}° (should be 0°)")
            print(f"Right arm angle: {arm.last_positions[arm.right_channel]:.1f}° (should be near {LIFT_ANGLES[2]}°)")
            
            if (arm.last_positions[arm.claw_channel] == 0 and  # Claw is closed
                abs(arm.last_positions[arm.right_channel] - LIFT_ANGLES[2]) < 5):
                
                # 6. Move to bin position using PSO
                target_position = WAYPOINTS[bin_number]
                print(f"\n[ACTION] Moving to bin {bin_number}...")
                # Use fixed angles from path.json instead of PSO optimization
                bin_angles = ANGLES[bin_number][:3]  # Get first 3 angles (base, left, right)
                print(f"Using fixed angles for bin {bin_number}: {bin_angles}")
                current_angles = move_to_angles(arm, bin_angles, current_angles, is_carrying=True)
                time.sleep(1.5)  # Wait for movement to complete
                
                # 7. Lower arm for drop
                print("\n[ACTION] Lowering arm for drop...")
                drop_angles = bin_angles.copy()
                drop_angles[2] += 20  # Lower by 20 degrees
                arm.move_right(drop_angles[2])
                time.sleep(1.0)
                
                # 8. Release object
                print("\n[ACTION] Releasing object...")
                print(f"Current claw angle before release: {arm.last_positions[arm.claw_channel]:.1f}°")
                quick_claw_movement(arm, "open")
                time.sleep(1.5)  # Wait for object to drop
                print(f"Current claw angle after release: {arm.last_positions[arm.claw_channel]:.1f}°")
                
                # 9. Return to bin position
                print("\n[ACTION] Returning to bin position...")
                arm.move_right(bin_angles[2])
                time.sleep(1.0)
                
                # Log successful sort to summary
                summary_entry = {
                    "e_waste": detected_class,
                    "classification": (
                        "green" if toxicity_level == "non_toxic"
                        else "yellow" if toxicity_level == "mildly_toxic"
                        else "red"
                    ),
                    "toxicity_level": toxicity_level,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                summary_data.append(summary_entry)
                with open(summary_file, "w") as file:
                    json.dump(summary_data, file, indent=4)
                
                # Generate updated toxicity report
                generate_toxicity_report_graph()
            else:
                print("⚠️ Grip or lift not complete, aborting drop sequence...")
            
            # === RETURN SEQUENCE ===
            print("\n=== RETURN SEQUENCE ===")
            # 10. Return to default position
            print("[ACTION] Returning to default position...")
            arm.default_position()
            time.sleep(1.5)
            
            # 11. Verify final position
            print("\n[VERIFY] Checking final position...")
            print(f"Final angles: Base={arm.last_positions[arm.base_channel]:.1f}°, Shoulder={arm.last_positions[arm.shoulder_channel]:.1f}°, Right={arm.last_positions[arm.right_channel]:.1f}°, Left={arm.last_positions[arm.left_channel]:.1f}°")
            if (arm.last_positions[arm.base_channel] != 0 or
                arm.last_positions[arm.shoulder_channel] != 88 or
                arm.last_positions[arm.right_channel] != 90 or
                arm.last_positions[arm.left_channel] != 0):
                print("⚠️ Not in default position, correcting...")
                arm.default_position()
                time.sleep(1.5)
            
            current_angles = [0, 88, 90, 0]

            with frame_lock:
                shared_state.current_display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            set_status_message("✅ Ready for next item", "green")
            time.sleep(3.0)  # Give time for camera to update
            # Resume detection
            detection_pause_event.clear()
            print("\n✅ Movement sequence completed. Ready for next detection.")
            
            # Update status message
            set_status_message("✅ Ready for next item", "green")
            
        except Exception as e:
            if isinstance(e, Empty):
                continue
            print(f"\n⚠️ Error during movement sequence: {str(e)}")
            try:
                 
                print("[ACTION] Error recovery: Moving to safe position...")
                quick_claw_movement(arm, "open")  # Open claw in case of error

                # Safe fallback values if detection variables are missing
                summary_log = {
                    "e-waste": detected_class if 'detected_class' in locals() else "unknown",
                    "toxicity_level": toxicity_level if 'toxicity_level' in locals() else "unknown",
                    "bin_number": bin_number if 'bin_number' in locals() else "unknown",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                summary_data.append(summary_log)

                with open(summary_file, "w") as f:
                    json.dump(summary_data, f, indent=4)

                arm.default_position()
                time.sleep(1.5)
                current_angles = [0, 88, 90, 0]
                set_status_message("⚠️ Error occurred, system recovered", "red")

            except:
                pass


def detect_and_sort(arm: RoboticArm):
    """Main function with threaded detection and arm control, with default position logic."""
    stop_event = threading.Event()
    detection_queue = Queue()
    detection_pause_event = Event()
    detection_pause_event.set()  # Start with detection paused
    try:
        print("\n🤖 Loading YOLOv8 model...")
        model = YOLO('train32/weights/best.pt')
        model.conf = 0.3
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

if __name__ == "__main__":
    arm = RoboticArm()
    detect_and_sort(arm)
