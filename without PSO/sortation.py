from robotic_arm import RoboticArm
import time
import json
import shared_state

# Initialize shared state
state = shared_state

# Define the toxicity map
toxicity_map = {
    "USB flashdrive": "non_toxic",
    "Battery": "highly_toxic",
    "USB cables": "non_toxic",
    "Sensor": "mildly_toxic",
    "PCB": "mildly_toxic",
    "default": "non_toxic"
}

# Define items list that can be imported
items = list(toxicity_map.keys())

# Add bin mapping for clarity
bin_mapping = {
    "highly_toxic": 3,
    "mildly_toxic": 2,
    "non_toxic": 1
}

# Simulate picking up an item
def pick_up_item(arm):
    print("Picking up item...")
    arm.move_claw(180)  # Open claw
    time.sleep(0.5)

# Simulate dropping an item into a bin
def drop_to_bin(arm, bin_number):
    print(f"Dropping item into bin {bin_number}...")
    arm.move_claw(0)  # Close claw
    time.sleep(0.5)

# Move to pickup position
def move_to_pickup(arm):
    print("Moving to pickup position...")
    arm.move_base(0)
    arm.move_left(0)
    arm.move_right(140)
    time.sleep(1.0)

# Move to bin position
def move_to_bin(arm, bin_number):
    print(f"Moving to bin {bin_number}...")
    if bin_number == 1:
        arm.move_base(30)
    elif bin_number == 2:
        arm.move_base(50)
    elif bin_number == 3:
        arm.move_base(88)
    time.sleep(1.0)

# Main sorting function
def sort_items(arm, items):
    for item in items:
        print(f"\n🎯 Sorting {item}...")
        
        # Start timing the sorting operation
        start_time = time.time()
        
        pick_up_item(arm)
        toxicity = toxicity_map.get(item, "non_toxic")
        bin_number = bin_mapping[toxicity]
        move_to_bin(arm, bin_number)
        drop_to_bin(arm, bin_number)
        move_to_pickup(arm)
        
        # End timing the sorting operation
        end_time = time.time()
        elapsed_time = end_time - start_time
        result = {"item": item, "total_time_seconds": elapsed_time}
        print(json.dumps(result, indent=2))

# Example usage
if __name__ == "__main__":
    arm = RoboticArm()
    example_items = list(toxicity_map.keys())
    print(f"Items to sort: {example_items}")
    sort_items(arm, example_items) 