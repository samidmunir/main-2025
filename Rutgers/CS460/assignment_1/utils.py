import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation_matrix(q: np.array) -> np.ndarray:
    # Qaternion elements
    q0, q1, q2, q3 = q
    
    # Rotation matrix from quaternion
    rotation_matrix = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])
    
    return rotation_matrix

def visualize_rotation(m: np.ndarray):
    # Define initial vectors (North pole and a nearby point)
    v0 = np.array([0, 0, 1]) # North pole
    epsilon = 1e-2 # small perturbation
    v1 = np.array([0, epsilon, 1]) # a point slightly displaced
    
    # Apply the rotation
    v0_prime = m @ v0 # rotate v0
    v1_prime = m @ v1 - v0 # rotate v1 and subtract v0 to get the direction
    
    # Create a sphere for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    # Create a sphere mesh
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere
    ax.plot_surface(x, y, z, color = 'lightblue', alpha = 0.3, rstride = 5, cstride = 5)
    
    # Plot the original vector v0 (North pole)
    ax.quiver(0, 0, 0, v0[0], v0[1], v0[2], color = 'blue', label = 'v0')
    
    # Plot the rotated vector v0_prime
    ax.quiver(0, 0, 0, v0_prime[0], v0_prime[1], v0_prime[2], color = 'red', label = 'v_0')
    
    # Plot the rotated vector v1_prime
    ax.quiver(v0_prime[0], v0_prime[1], v0_prime[2], v1_prime[0], v1_prime[1], v1_prime[2], color = 'green', label = 'v_1')
    
    # Add labels and legend
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    
    plt.show()