import numpy as np
from component_1 import check_SOn
from utils import quaternion_to_rotation_matrix, visualize_rotation

def random_rotation_matrix(naive: bool) -> np.ndarray:
    if naive:
        yaw = np.random.uniform(0, 2 * np.pi) # rotation around z-axis
        pitch = np.random.uniform(0, 2 * np.pi) # rotation around y-axis
        roll = np.random.uniform(0, 2 * np.pi) # rotation around x-axis
        
        # Rotation matrix around z-axis (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Rotation matrix around y-axis (pitch)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation matrix around x-axis (roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        random_rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        
    else:
        # Generate a random quaternion
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 2 * np.pi)
        u3 = np.random.uniform(0, 2 * np.pi)
        
        # Convert to quaternion (q0, q1, q2, q3)
        q0 = np.sqrt(1 - u1) * np.cos(u2)
        q1 = np.sqrt(1 - u1) * np.sin(u2)
        q2 = np.sqrt(u1) * np.cos(u3)
        q3 = np.sqrt(u1) * np.sin(u3)
        
        # Convert quaternion to a rotation matrix
        random_rotation_matrix = quaternion_to_rotation_matrix(np.array([q0, q1, q2, q3]))
        
    # Confirm if random generated rotation matrix is in SO(n).
    if check_SOn(random_rotation_matrix):
        visualize_rotation(random_rotation_matrix)
        return random_rotation_matrix
    else:
        raise ValueError('Random generated rotation matrix does not belong to SO(n).')