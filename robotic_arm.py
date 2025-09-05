import board
import busio
import time
from digitalio import DigitalInOut
import adafruit_pca9685
from adafruit_pca9685 import PCA9685

class RoboticArm:
    def __init__(self):
        # Initialize I2C and PCA9685 (single instance)
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50  # 50Hz for servos

        # Define servo channels based on your setup
        self.claw_channel = 1    # Claw servo
        self.base_channel = 4    # Base servo
        self.shoulder_channel = 0 # Shoulder servo
        self.right_channel = 5   # Right servo
        self.left_channel = 6    # Left servo

        # Track last positions
        self.last_positions = {
            self.claw_channel: None,
            self.base_channel: None,
            self.shoulder_channel: None,
            self.right_channel: None,
            self.left_channel: None
        }

    def angle_to_duty_cycle(self, angle):
        """Convert angle (0-180) to duty cycle (0-0xFFFF)"""
        min_duty = 2.5  # Duty cycle for 0 degrees
        max_duty = 12.5  # Duty cycle for 180 degrees
        duty_cycle = min_duty + (angle / 180) * (max_duty - min_duty)
        return int(duty_cycle * 65535 / 100)

    def move_servo(self, channel, angle):
        """Move servo with position holding and verification"""
        # Check if movement is needed
        if self.last_positions[channel] == angle:
            return

        # Limit angle to 180 degrees
        angle = max(0, min(angle, 180))
        
        # Set new position (ONLY ONCE to prevent jittering)
        duty_cycle = self.angle_to_duty_cycle(angle)
        self.pca.channels[channel].duty_cycle = duty_cycle
        
        # Store position
        self.last_positions[channel] = angle
        
        print(f"[ACTION] Channel {channel}: Moving to {angle} degrees")

    def move_base(self, angle):
        """Move base servo with position verification"""
        print(f"[INFO] Moving base to {angle} degrees")
        self.move_servo(self.base_channel, angle)
        time.sleep(0.2)  # Additional stability delay for base

    def move_shoulder(self, angle):
        """Move shoulder servo"""
        print(f"[INFO] Moving shoulder to {angle} degrees")
        self.move_servo(self.shoulder_channel, angle)

    def move_right(self, angle):
        """Move right servo with position holding"""
        print(f"[INFO] Moving right servo to {angle} degrees")
        self.move_servo(self.right_channel, angle)
        time.sleep(0.2)  # Increased stability delay

    def move_left(self, angle):
        """Move left servo"""
        print(f"[INFO] Moving left servo to {angle} degrees")
        self.move_servo(self.left_channel, angle)
        time.sleep(0.2)  # Increased stability delay

    def move_claw(self, angle):
        """Move claw with stability control"""
        print(f"[INFO] Moving claw to {angle} degrees")
        self.move_servo(self.claw_channel, angle)
        # NO DELAY for claw - needs to be snappy for proper grip strength and impact

    def claw_open(self):
        """Open the claw wider"""
        print("[ACTION] Opening claw...")
        self.move_claw(180)

    def claw_close(self):
        """Close the claw with controlled grip"""
        print("[ACTION] Closing claw...")
        self.move_claw(0)

    def cleanup(self):
        """Release all servos safely"""
        print("[INFO] Cleaning up...")
        for channel in range(16):
            self.pca.channels[channel].duty_cycle = 0
            if channel in self.last_positions:
                self.last_positions[channel] = None
        print("[INFO] All servos released.")

    def default_position(self):
        """Move the arm to the default rest position: BASE=0, SHOULDER=88, RIGHT=90, LEFT=0"""
        print("[ACTION] Moving to default rest position (BASE=0, SHOULDER=88, RIGHT=90, LEFT=0)")
        self.move_base(0)
        time.sleep(0.3)  # Added delay between movements
        self.move_shoulder(88)
        time.sleep(0.3)
        self.move_right(90)
        time.sleep(0.3)
        self.move_left(0)
        time.sleep(0.3)
        print("[INFO] Arm is now in default rest position.")

