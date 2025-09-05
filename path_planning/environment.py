import numpy as np
import math

class Environment:
    def __init__(self, arm_length, workspace_limits):
        self.arm_length = arm_length
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = workspace_limits
        
        # Define specific angles for different positions
        self.PICKUP_RIGHT_ANGLE = 145  # Right servo angle for pickup
        self.DROP_RIGHT_ANGLE = 100    # Right servo angle for dropping
        self.LEFT_SERVO_ANGLE = 30     # Consistent left servo angle
        self.CLAW_GRIP_ANGLE = 0       # Grip angle for holding objects (changed from 90 to 0)
        
        # Define bin angles
        self.BIN1_ANGLE = 37   # Non-toxic (green)
        self.BIN2_ANGLE = 61   # Mildly-toxic (yellow)
        self.BIN3_ANGLE = 88   # Highly-toxic (red)
    
    def evaluate_path(self, angles, target_position):
        """Evaluate the effectiveness of a set of angles using multiple criteria"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # 1. Check if angles are within valid ranges
        if not (0 <= base_angle <= 180 and
                0 <= left_angle <= 180 and
                0 <= right_angle <= 180 and
                0 <= claw_angle <= 180):
            return float('inf')  # Invalid angles
        
        # 2. Calculate position error
        position_error = self.calculate_position_error(angles, target_position)
        
        # 3. Calculate stability score
        stability_score = self.calculate_stability_score(angles)
        
        # 4. Calculate energy efficiency
        energy_score = self.calculate_energy_score(angles)
        
        # 5. Calculate smoothness of movement
        smoothness_score = self.calculate_smoothness_score(angles)
        
        # Combine all scores with weights
        total_score = (
            position_error * 0.4 +      # Position accuracy is most important
            stability_score * 0.3 +     # Stability is second most important
            energy_score * 0.2 +        # Energy efficiency
            smoothness_score * 0.1      # Movement smoothness
        )
        
        return total_score
    
    def calculate_position_error(self, angles, target_position):
        """Calculate how well the angles achieve the target position"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # Calculate expected position based on angles
        expected_x = self.arm_length * math.cos(math.radians(base_angle))
        expected_y = self.arm_length * math.sin(math.radians(base_angle))
        expected_z = self.arm_length * math.sin(math.radians(right_angle))
        
        # Calculate error
        error = math.sqrt(
            (expected_x - target_position[0])**2 +
            (expected_y - target_position[1])**2 +
            (expected_z - target_position[2])**2
        )
        
        return error
    
    def calculate_stability_score(self, angles):
        """Calculate how stable the arm position is"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # Penalize extreme angles that might cause instability
        stability_penalty = 0
        
        # Base angle stability
        if base_angle < 20 or base_angle > 160:
            stability_penalty += 1
            
        # Right angle stability
        if right_angle < 30 or right_angle > 150:
            stability_penalty += 1
            
        # Left angle stability
        if left_angle < 20 or left_angle > 160:
            stability_penalty += 1
            
        return stability_penalty
    
    def calculate_energy_score(self, angles):
        """Calculate energy efficiency of the position"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # Calculate total angle movement from neutral position
        total_movement = (
            abs(base_angle - 90) +  # Base movement from center
            abs(left_angle - 90) +  # Left arm movement from center
            abs(right_angle - 90)   # Right arm movement from center
        )
        
        return total_movement
    
    def calculate_smoothness_score(self, angles):
        """Calculate how smooth the movement would be"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # Penalize sudden angle changes
        smoothness_penalty = 0
        
        # Check for large angle differences between servos
        if abs(base_angle - left_angle) > 90:
            smoothness_penalty += 1
        if abs(left_angle - right_angle) > 90:
            smoothness_penalty += 1
            
        return smoothness_penalty

class AngleValidator:
    def __init__(self):
        # Define specific angles for different positions
        self.PICKUP_RIGHT_ANGLE = 145  # Right servo angle for pickup
        self.DROP_RIGHT_ANGLE = 100    # Right servo angle for dropping
        self.LEFT_SERVO_ANGLE = 30     # Consistent left servo angle
        self.CLAW_GRIP_ANGLE = 0       # Grip angle for holding objects (changed from 90 to 0)
        
        # Define bin angles
        self.BIN1_ANGLE = 37   # Non-toxic (green)
        self.BIN2_ANGLE = 61   # Mildly-toxic (yellow)
        self.BIN3_ANGLE = 88   # Highly-toxic (red)
    
    def validate_angles(self, angles, is_pickup=False):
        """Validate if the angles are within acceptable ranges"""
        base_angle, left_angle, right_angle, claw_angle = angles
        
        # Check if angles are within valid ranges
        if not (0 <= base_angle <= 180 and
                0 <= left_angle <= 180 and
                0 <= right_angle <= 180 and
                0 <= claw_angle <= 180):
            return False
            
        # Check if right servo angle matches the position
        expected_right_angle = self.PICKUP_RIGHT_ANGLE if is_pickup else self.DROP_RIGHT_ANGLE
        if abs(right_angle - expected_right_angle) > 5:  # Allow 5 degrees of error
            return False
            
        return True