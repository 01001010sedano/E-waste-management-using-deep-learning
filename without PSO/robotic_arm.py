import time

class RoboticArm:
    def __init__(self):
        # Simulate servo positions
        self.last_positions = {
            0: 0,   # base_channel
            1: 0,   # left_channel
            2: 0,   # right_channel
            3: 180  # claw_channel (open by default)
        }
        self.base_channel = 0
        self.left_channel = 1
        self.right_channel = 2
        self.claw_channel = 3
        self.shoulder_channel = self.left_channel

    def move_base(self, angle):
        print(f"[MOCK] Move base to {angle}")
        self.last_positions[self.base_channel] = angle
        time.sleep(5.0)  # Add delay for non-PSO version

    def move_left(self, angle):
        print(f"[MOCK] Move left arm to {angle}")
        self.last_positions[self.left_channel] = angle
        time.sleep(5.0)  # Add delay for non-PSO version

    def move_right(self, angle):
        print(f"[MOCK] Move right arm to {angle}")
        self.last_positions[self.right_channel] = angle
        time.sleep(5.0)  # Add delay for non-PSO version

    def move_claw(self, angle):
        print(f"[MOCK] Move claw to {angle}")
        self.last_positions[self.claw_channel] = angle
        time.sleep(5.0)  # Add delay for non-PSO version

    def default_position(self):
        print("[MOCK] Moving to default position (0, 0, 0)")

    def cleanup(self):
        print("[MOCK] Cleanup called")
