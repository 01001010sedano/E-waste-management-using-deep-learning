import numpy as np
from path_planning.environment import AngleValidator, Environment
import json

class AngleGenerator:
    def __init__(self):
        self.validator = AngleValidator()
        
    def generate_angles(self, is_pickup=False):
        """Generate valid angles for pickup or drop positions"""
        if is_pickup:
            return [
                0,  # base angle
                self.validator.LEFT_SERVO_ANGLE,
                self.validator.PICKUP_RIGHT_ANGLE,
                160  # claw open
            ]
        else:
            return [
                0,  # base angle
                self.validator.LEFT_SERVO_ANGLE,
                self.validator.DROP_RIGHT_ANGLE,
                0  # claw closed
            ]
    
    def get_bin_angles(self, bin_number):
        """Get angles for specific bin"""
        base_angles = {
            1: self.validator.BIN1_ANGLE,
            2: self.validator.BIN2_ANGLE,
            3: self.validator.BIN3_ANGLE
        }
        
        return [
            base_angles[bin_number],
            self.validator.LEFT_SERVO_ANGLE,
            self.validator.DROP_RIGHT_ANGLE,
            0  # claw closed
        ]

class PSO:
    def __init__(self, num_particles, num_dimensions, environment, target_position, is_pickup=False):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions  # 4 dimensions: base, left, right, claw
        self.environment = environment
        self.target_position = target_position
        self.is_pickup = is_pickup
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive weight
        self.c2 = 1.5  # Social weight
        
        # Initialize particles and velocities
        self.particles = np.random.uniform(0, 180, (num_particles, num_dimensions))
        self.velocities = np.random.uniform(-10, 10, (num_particles, num_dimensions))
        
        # Set fixed angles based on pickup/drop
        if is_pickup:
            self.particles[:, 2] = 145  # Right servo angle for pickup
            # Don't set claw angle, let direct movement control handle it
        else:
            self.particles[:, 2] = 100  # Right servo angle for drop
            # Don't set claw angle, let direct movement control handle it
        
        # Initialize personal bests
        self.pbest = self.particles.copy()
        self.pbest_scores = np.array([float('inf')] * num_particles)
        
        # Initialize global best
        self.gbest = None
        self.gbest_score = float('inf')
        
        # Initialize scores
        self.update_scores()
    
    def update_scores(self):
        """Update scores for all particles"""
        for i in range(self.num_particles):
            score = self.environment.evaluate_path(self.particles[i], self.target_position)
            
            # Update personal best if better
            if score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest[i] = self.particles[i].copy()
                
                # Update global best if better
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.particles[i].copy()
    
    def optimize(self, max_iterations=100):
        """Run PSO optimization"""
        for iteration in range(max_iterations):
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            
            # Update velocities
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.pbest - self.particles) +
                self.c2 * r2 * (self.gbest - self.particles)
            )
            
            # Update positions
            self.particles += self.velocities
            
            # Clamp positions to valid range [0, 180]
            self.particles = np.clip(self.particles, 0, 180)
            
            # Maintain fixed angles
            if self.is_pickup:
                self.particles[:, 2] = 145  # Right servo angle for pickup
                # Don't set claw angle, let direct movement control handle it
            else:
                self.particles[:, 2] = 100  # Right servo angle for drop
                # Don't set claw angle, let direct movement control handle it
            
            # Update scores
            self.update_scores()
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Score: {self.gbest_score:.2f}")
                print(f"Best Angles: {self.gbest}")
        
        return self.gbest, self.gbest_score
    
    def get_best_angles(self):
        """Get the best angles found by PSO"""
        return self.gbest if self.gbest is not None else None

def optimize_path_angles(path_file, environment):
    """Optimize angles for all waypoints in path.json"""
    # Load path.json
    with open(path_file, 'r') as f:
        path_data = json.load(f)
    
    waypoints = path_data['waypoints']
    optimized_angles = []
    
    # Optimize angles for each waypoint
    for i, waypoint in enumerate(waypoints):
        is_pickup = i == 0  # First waypoint is pickup
        pso = PSO(num_particles=20, num_dimensions=4, 
                 environment=environment, 
                 target_position=waypoint,
                 is_pickup=is_pickup)
        
        best_angles, _ = pso.optimize()
        optimized_angles.append(best_angles.tolist())
    
    # Update path.json with optimized angles
    path_data['angles'] = optimized_angles
    with open(path_file, 'w') as f:
        json.dump(path_data, f, indent=4)
    
    return optimized_angles