import json
from path_planning.environment import Environment
from path_planning.pso import PSO

def main():
    # Define waypoints (bins positions)
    waypoints = [
        [0, 0, 0],      # Home/pickup position
        [23, 13, 7],    # First bin (non-toxic, green)
        [32, 30, 7],    # Second bin (mildly-toxic, yellow)
        [6, 20, 7],     # Third bin (highly-toxic, red)
        [15, 35, 7]     # Fourth bin (unidentified)
    ]
    
    # Define specific angles for each position
    angles = [
        # [base, left, right, claw]
        [0, 0, 145, 180],     # Home/pickup: left=0, right=145, claw open (180°)
        [37, 0, 115, 0],      # Bin 1 (non-toxic): base=37°, left=0, right=125, claw closed (0°)
        [61, 0, 115, 0],      # Bin 2 (mildly-toxic): base=61°, left=0, right=125, claw closed (0°)
        [88, 0, 115, 0],      # Bin 3 (highly-toxic): base=88°, left=0, right=125, claw closed (0°)
        [110, 0, 115, 0]      # Bin 4 (unidentified): base=110°, left=0, right=125, claw closed (0°)
    ]
    
    # Create path data
    path_data = {
        "waypoints": waypoints,
        "angles": angles
    }
    
    # Save to path.json
    with open('path.json', 'w') as f:
        json.dump(path_data, f, indent=4)
    
    print("✅ Generated path.json with correct angles")
    print("\n📍 Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"Position {i}: {wp}")
    
    print("\n🔄 Angles:")
    for i, angle in enumerate(angles):
        print(f"Position {i}: {angle}")
    
    print("\n🔍 Bin Assignments:")
    print("Bin 1 (non-toxic, green): 37°")
    print("Bin 2 (mildly-toxic, yellow): 61°")
    print("Bin 3 (highly-toxic, red): 88°")
    print("Bin 4 (unidentified): 110°")

if __name__ == "__main__":
    main()