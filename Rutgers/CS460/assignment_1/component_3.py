import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

def interpolate_rigid_body(start_pose: np.array, goal_pose: np.array) -> np.ndarray:
    STEPS = 100
    
    # Linear interpolation for position (x, y)
    x_vals = np.linspace(start_pose[0], goal_pose[0], STEPS)
    y_vals = np.linspace(start_pose[1], goal_pose[1], STEPS)
    
    # Linear interpolation for orientation (theta)
    theta_vals = np.linspace(start_pose[2], goal_pose[2], STEPS)
    
    # Combine the interpolated x, y, and theta into poses
    path = np.vstack((x_vals, y_vals, theta_vals)).T
    
    return path

def forward_propagate_rigid_body(start_pose: np.array, plan: np.ndarray) -> np.ndarray:
    poses = [start_pose]  # Start with the initial pose
    
    # Loop through the plan and apply each velocity for the given duration
    for velocity, duration in plan:
        v_x, v_y, v_theta = velocity  # Unpack the velocity components
        
        # Get the current pose (last in the list)
        x, y, theta = poses[-1]
        
        # Calculate the new pose after applying the velocity over the given duration
        for _ in range(int(duration * 100)):  # Assuming 100 steps per second
            # Update position and orientation
            x += v_x * 0.1 # Assume small time steps (0.01 for 100 Hz)
            y += v_y * 0.1
            theta += v_theta * 0.1
            
            # Append the new pose to the path
            poses.append([x, y, theta])
    
    return np.array(poses)


def visualize_path(path: np.ndarray):
    fig, ax = plt.subplots()
    
    # Define the bounds of the 20 x 20 environment
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    
    # Plot the path as a line
    ax.plot(path[:, 0], path[:, 1], 'b-', label = 'Path')
    
    # Initialize the robot rectangle at the start pose
    robot = Rectangle((path[0, 0], path[0, 1]), 0.5, 0.3, angle = np.degrees(path[0, 2]), color = 'red')
    ax.add_patch(robot)
    
    def update(frame):
        # Update the robot's position and orientation
        robot.set_xy((path[frame, 0], path[frame, 1]))
        robot.angle = np.degrees(path[frame, 2])
        return robot
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames = len(path), interval = 200, blit = False)
    
    plt.legend()
    plt.show()