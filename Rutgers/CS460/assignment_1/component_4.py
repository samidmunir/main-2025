import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def interpolate_arm(start: np.array, goal: np.array) -> np.ndarray:
    start = np.array(start)
    goal = np.array(goal)
    
    delta_theta = np.linalg.norm(goal - start)
    
    steps = int(delta_theta / 0.1)
    
    interpolated_path = [start + (goal - start) * t for t in np.linspace(0, 1, steps)]
    
    return interpolated_path

def forward_propagate_arm(start_pose: np.array, plan: np.ndarray) -> np.ndarray:
    path = [start_pose]
    
    for velocity, duration in plan:
        last_pose = np.array(path[-1])
        delta_theta = np.array(velocity) * duration
        new_pose = last_pose + delta_theta
        
        path.append(new_pose)
    
    return path

def visualize_arm_path(path: np.ndarray):
    interval = 1500
    # Define the lengths of the two arm segments.
    l1 = 2
    l2 = 1.5
    
    # Function to compute the (x, y) coordinates of the arm's joints based on angles.
    def get_arm_position(theta_0, theta_1):
        # Joint 1 (base)
        x1, y1 = l1 * np.cos(theta_0), l1 * np.sin(theta_0)
        
        # Joint 2 (end effector)
        x2 = x1 + l2 * np.cos(theta_0 + theta_1)
        y2 = y1 + l2 * np.sin(theta_0 + theta_1)
        
        # Returns positions of the base, joint 1, and joint 2.
        return (0, 0), (x1, y1), (x2, y2)
    
    # Setup the plot.
    fig, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    
    # Initialize plot elements: base to joint 1 and joint 1 to joint 2.
    arm_line, = ax.plot([], [], 'o-', lw=4, color='blue', label='Arm')
    joint_markers, = ax.plot([], [], 'ro', label='Joints')  # Red circles for joints
    
    # Add labels for joints (Base, Joint 1, Joint 2)
    base_text = ax.text(0, 0, 'Base', fontsize=10, ha='right')
    joint1_text = ax.text(0, 0, 'Joint 1', fontsize=10, ha='right')
    joint2_text = ax.text(0, 0, 'End Effector', fontsize=10, ha='right')
    
    # Animation initialization function.
    def init():
        arm_line.set_data([], [])
        joint_markers.set_data([], [])
        return arm_line, joint_markers, base_text, joint1_text, joint2_text
    
    # Animation update function.
    def update(frame):
        theta_0, theta_1 = path[frame]
        base, joint1, joint2 = get_arm_position(theta_0, theta_1)
        
        # Update the arm's position (base to joint 1 and joint 1 to joint 2).
        arm_line.set_data([base[0], joint1[0], joint2[0]], [base[1], joint1[1], joint2[1]])
        
        # Update joint positions with markers.
        joint_markers.set_data([base[0], joint1[0], joint2[0]], [base[1], joint1[1], joint2[1]])
        
        # Update the text for joint labels.
        base_text.set_position((base[0], base[1]))
        joint1_text.set_position((joint1[0], joint1[1]))
        joint2_text.set_position((joint2[0], joint2[1]))
        
        return arm_line, joint_markers, base_text, joint1_text, joint2_text
    
    # Create the animation.
    anim = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, repeat=False, interval = interval)
    
    # Display the animation.
    plt.legend()
    plt.show()