import time
from robotic_arm import RoboticArm

# Adjustable values for servos
CLAW_OPEN = 150# Angle for open position
CLAW_CLOSE = 0  # Angle for closed position
BASE_ANGLE = 0    # Base servo angle
SHOULDER_ANGLE = 88 # Shoulder servo angle
RIGHT_ANGLE = 125    # Right servo angle
LEFT_ANGLE = 0    # Left servo angle
DELAY = 2  

def reset_arm_position():
    """Reset the robotic arm to home position."""
    arm = RoboticArm()
    print("\n🏠 Resetting arm to home position...")

    try:
        # Move all servos to specified angles
        print("⚙️ Moving all servos to reset position...")
        

        arm.move_claw(CLAW_OPEN)
        time.sleep(DELAY)
        arm.move_claw(CLAW_CLOSE)

        print("➡️ Resetting base...")
        arm.move_base(BASE_ANGLE)
        time.sleep(DELAY)
        
        print("➡️ Resetting shoulder...")
        arm.move_shoulder(SHOULDER_ANGLE)
        time.sleep(DELAY)
        
        print("➡️ Resetting right servo...")
        arm.move_right(RIGHT_ANGLE)
        time.sleep(DELAY)
        
        print("➡️ Resetting left servo...")
        arm.move_left(LEFT_ANGLE)
        time.sleep(DELAY)


        print("✅ Arm successfully reset to home position!")

    except KeyboardInterrupt:
        print("\n⚠️ Reset interrupted by user.")
    
    finally:
        arm.cleanup()
        print("\n🔌 All servos released.")

if __name__ == "__main__":
    reset_arm_position() 