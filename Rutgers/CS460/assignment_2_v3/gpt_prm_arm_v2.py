# import matplotlib.pyplot as PLT
# import numpy as NP
# import random as RANDOM
# from scipy.spatial import KDTree
# import networkx as NX
# from matplotlib.animation import FuncAnimation

# # Constants
# ARM_ROBOT_LINK_1_LENGTH = 2.0
# ARM_ROBOT_LINK_2_LENGTH = 1.5
# ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX = -10, 10
# ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX = -10, 10

# def get_arm_robot_joint_positions(theta_1, theta_2):
#     """Get positions of the base, joint, and end-effector."""
#     BASE = (0, 0)
#     JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
#     JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
#     JOINT = (JOINT_X, JOINT_Y)
#     END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
#     END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
#     END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
#     return BASE, JOINT, END_EFFECTOR

# def scene_from_file(filename):
#     """Load obstacles from a file."""
#     obstacles = []
#     with open(filename, 'r') as f:
#         for line in f:
#             x, y, width, height, _, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
#             obstacles.append((x, y, width, height))
#     return obstacles

# def is_valid_configuration(theta_1, theta_2, obstacles):
#     """Check if the arm configuration is valid (no collisions)."""
#     base, joint, end_effector = get_arm_robot_joint_positions(theta_1, theta_2)

#     for obs_x, obs_y, obs_width, obs_height in obstacles:
#         if (obs_x - obs_width / 2 <= joint[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= joint[1] <= obs_y + obs_height / 2):
#             return False
#         if (obs_x - obs_width / 2 <= end_effector[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= end_effector[1] <= obs_y + obs_height / 2):
#             return False
#     return True

# def build_prm(nodes, k=5):
#     """Build the PRM graph using KDTree for nearest neighbors."""
#     graph = NX.Graph()
#     kdtree = KDTree(nodes)

#     for i, node in enumerate(nodes):
#         graph.add_node(i, pos=node)

#     for i, node in enumerate(nodes):
#         distances, indices = kdtree.query(node, k=k + 1)
#         for j in indices[1:]:
#             distance = NP.linalg.norm(NP.array(node) - NP.array(nodes[j]))
#             graph.add_edge(i, j, weight=distance)

#     return graph

def visualize_prm(nodes, edges, obstacles):
    """Visualize the PRM graph with obstacles."""
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

    # Plot obstacles
    for x, y, width, height in obstacles:
        rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
                             color='gray', alpha=0.5)
        ax.add_patch(rect)

    # Plot nodes and edges
    for node in nodes:
        ax.plot(node[0], node[1], 'bo', markersize=2)

    for edge in edges:
        n1, n2 = edge
        pos1, pos2 = nodes[n1], nodes[n2]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'g-', alpha=0.5)

    PLT.title('PRM Graph with Obstacles')
    PLT.show()

def visualize_path(nodes, path, obstacles):
    """Visualize the found path with obstacles."""
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

    # Plot obstacles
    for x, y, width, height in obstacles:
        rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
                             color='gray', alpha=0.5)
        ax.add_patch(rect)

    # Plot the path
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        pos1, pos2 = nodes[n1], nodes[n2]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r-', lw=2)

    PLT.title('Path Visualization with Obstacles')
    PLT.show()

def add_node_to_prm(graph, nodes, config_map, config, k=5):
    """Add a new node to the PRM and connect it to neighbors."""
    node_idx = len(nodes)
    nodes.append(config)
    config_map[node_idx] = config
    graph.add_node(node_idx, pos=config)

    kdtree = KDTree(nodes)
    distances, indices = kdtree.query(config, k=k + 1)
    for neighbor_idx in indices[1:]:
        distance = NP.linalg.norm(NP.array(config) - NP.array(nodes[neighbor_idx]))
        graph.add_edge(node_idx, neighbor_idx, weight=distance)

    return node_idx

def find_path(graph, start_idx, goal_idx):
    """Use A* search to find a path from start to goal."""
    try:
        return NX.astar_path(
            graph, start_idx, goal_idx,
            heuristic=lambda u, v: NP.linalg.norm(
                NP.array(graph.nodes[u]['pos']) - NP.array(graph.nodes[v]['pos'])
            )
        )
    except NX.NetworkXNoPath:
        print("No path found!")
        return []

# def animate_robot_movement(path, config_map, obstacles):
#     """Smoothly animate the robot along the path."""
#     fig, ax = PLT.subplots()
#     ax.set_aspect('equal')
#     ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
#     ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
#     for obstacle in obstacles:
#         x, y, width, height = obstacle
#         rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
#                              color='gray', alpha=0.5)
#         ax.add_patch(rect)

#     robot_line, = ax.plot([], [], 'b-', lw=2)
#     joint_marker, = ax.plot([], [], 'ro', markersize=6)
#     end_effector_marker, = ax.plot([], [], 'go', markersize=6)

#     def init():
#         robot_line.set_data([], [])
#         joint_marker.set_data([], [])
#         end_effector_marker.set_data([], [])
#         return robot_line, joint_marker, end_effector_marker

#     def update_frame(i):
#         config = config_map[path[i]]
#         base, joint, end_effector = get_arm_robot_joint_positions(*config)

#         robot_line.set_data([base[0], joint[0], end_effector[0]],
#                             [base[1], joint[1], end_effector[1]])

#         joint_marker.set_data([joint[0]], [joint[1]])
#         end_effector_marker.set_data([end_effector[0]], [end_effector[1]])

#         return robot_line, joint_marker, end_effector_marker

#     ani = FuncAnimation(fig, update_frame, frames=len(path),
#                         init_func=init, interval=1000, blit=True, repeat=False)

#     PLT.show()

# def main():
#     obstacles = scene_from_file('environment_1_10.txt')

#     valid_configs = [
#         (theta_1, theta_2) for theta_1 in NP.linspace(0, NP.pi, 10)
#         for theta_2 in NP.linspace(0, NP.pi, 10)
#         if is_valid_configuration(theta_1, theta_2, obstacles)
#     ]
#     end_effector_positions = [get_arm_robot_joint_positions(*config)[2] for config in valid_configs]
#     config_map = {i: config for i, config in enumerate(valid_configs)}

#     graph = build_prm(end_effector_positions)

#     start_config = (0.0, 0.0)
#     goal_config = (-2.0, 0.0)

#     _, _, start_pos = get_arm_robot_joint_positions(*start_config)
#     _, _, goal_pos = get_arm_robot_joint_positions(*goal_config)

#     start_idx = add_node_to_prm(graph, end_effector_positions, config_map, start_pos)
#     goal_idx = add_node_to_prm(graph, end_effector_positions, config_map, goal_pos)

#     path = find_path(graph, start_idx, goal_idx)

#     visualize_prm(end_effector_positions, graph.edges, obstacles)
#     visualize_path(end_effector_positions, path, obstacles)
#     animate_robot_movement(path, config_map, obstacles)

# if __name__ == '__main__':
#     main()









# import matplotlib.pyplot as PLT
# import numpy as NP
# import random as RANDOM
# from scipy.spatial import KDTree
# import networkx as NX
# from matplotlib.animation import FuncAnimation

# # Constants
# ARM_ROBOT_LINK_1_LENGTH = 2.0
# ARM_ROBOT_LINK_2_LENGTH = 1.5
# ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX = -10, 10
# ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX = -10, 10

# def get_arm_robot_joint_positions(theta_1, theta_2):
#     """Get positions of the base, joint, and end-effector."""
#     BASE = (0, 0)
#     JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
#     JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
#     JOINT = (JOINT_X, JOINT_Y)
#     END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
#     END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
#     END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
#     return BASE, JOINT, END_EFFECTOR

# def wrap_angle_difference(start, end):
#     """Wrap the angle difference to ensure the shortest path."""
#     delta = (end - start + NP.pi) % (2 * NP.pi) - NP.pi
#     return start + delta

# def interpolate_angles(theta_1_start, theta_1_end, theta_2_start, theta_2_end, steps=10):
#     """Generate interpolated angles with wrapping for smooth transitions."""
#     theta_1_end_wrapped = wrap_angle_difference(theta_1_start, theta_1_end)
#     theta_2_end_wrapped = wrap_angle_difference(theta_2_start, theta_2_end)

#     theta_1_vals = NP.linspace(theta_1_start, theta_1_end_wrapped, steps)
#     theta_2_vals = NP.linspace(theta_2_start, theta_2_end_wrapped, steps)

#     return [(theta_1, theta_2) for theta_1, theta_2 in zip(theta_1_vals, theta_2_vals)]

# def scene_from_file(filename):
#     """Load obstacles from a file."""
#     obstacles = []
#     with open(filename, 'r') as f:
#         for line in f:
#             x, y, width, height, _, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
#             obstacles.append((x, y, width, height))
#     return obstacles

# def is_valid_configuration(theta_1, theta_2, obstacles):
#     """Check if the arm configuration is valid (no collisions)."""
#     base, joint, end_effector = get_arm_robot_joint_positions(theta_1, theta_2)

#     for obs_x, obs_y, obs_width, obs_height in obstacles:
#         if (obs_x - obs_width / 2 <= joint[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= joint[1] <= obs_y + obs_height / 2):
#             return False
#         if (obs_x - obs_width / 2 <= end_effector[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= end_effector[1] <= obs_y + obs_height / 2):
#             return False
#     return True

# def build_prm(nodes, k=5):
#     """Build the PRM graph using KDTree for nearest neighbors."""
#     graph = NX.Graph()
#     kdtree = KDTree(nodes)

#     for i, node in enumerate(nodes):
#         graph.add_node(i, pos=node)

#     for i, node in enumerate(nodes):
#         distances, indices = kdtree.query(node, k=k + 1)
#         for j in indices[1:]:
#             distance = NP.linalg.norm(NP.array(node) - NP.array(nodes[j]))
#             graph.add_edge(i, j, weight=distance)

#     return graph

# def animate_robot_movement(path, config_map, obstacles):
#     """Smoothly animate the robot along the path with obstacles."""
#     fig, ax = PLT.subplots()
#     ax.set_aspect('equal')
#     ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
#     ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

#     for obstacle in obstacles:
#         x, y, width, height = obstacle
#         rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
#                              color='gray', alpha=0.5)
#         ax.add_patch(rect)

#     robot_line, = ax.plot([], [], 'b-', lw=2)
#     joint_marker, = ax.plot([], [], 'ro', markersize=6)
#     end_effector_marker, = ax.plot([], [], 'go', markersize=6)

#     # Generate a full interpolated path using wrapped angles
#     full_path = []
#     for i in range(len(path) - 1):
#         config1 = config_map[path[i]]
#         config2 = config_map[path[i + 1]]
#         full_path.extend(interpolate_angles(*config1, *config2, steps=20))

#     def init():
#         robot_line.set_data([], [])
#         joint_marker.set_data([], [])
#         end_effector_marker.set_data([], [])
#         return robot_line, joint_marker, end_effector_marker

#     def update_frame(i):
#         config = full_path[i]
#         base, joint, end_effector = get_arm_robot_joint_positions(*config)

#         robot_line.set_data([base[0], joint[0], end_effector[0]],
#                             [base[1], joint[1], end_effector[1]])

#         joint_marker.set_data([joint[0]], [joint[1]])
#         end_effector_marker.set_data([end_effector[0]], [end_effector[1]])

#         return robot_line, joint_marker, end_effector_marker

#     ani = FuncAnimation(fig, update_frame, frames=len(full_path),
#                         init_func=init, interval=100, blit=True, repeat=False)

#     PLT.show()

# def main():
#     obstacles = scene_from_file('environment_1_10.txt')

#     valid_configs = [
#         (theta_1, theta_2) for theta_1 in NP.linspace(0, NP.pi, 10)
#         for theta_2 in NP.linspace(0, NP.pi, 10)
#         if is_valid_configuration(theta_1, theta_2, obstacles)
#     ]
#     end_effector_positions = [get_arm_robot_joint_positions(*config)[2] for config in valid_configs]
#     config_map = {i: config for i, config in enumerate(valid_configs)}

#     graph = build_prm(end_effector_positions)

#     start_config = (0.0, 0.0)
#     goal_config = (-2.0, 0.0)

#     _, _, start_pos = get_arm_robot_joint_positions(*start_config)
#     _, _, goal_pos = get_arm_robot_joint_positions(*goal_config)

#     start_idx = add_node_to_prm(graph, end_effector_positions, config_map, start_pos)
#     goal_idx = add_node_to_prm(graph, end_effector_positions, config_map, goal_pos)

#     path = find_path(graph, start_idx, goal_idx)

#     visualize_prm(end_effector_positions, graph.edges, obstacles)
#     visualize_path(end_effector_positions, path, obstacles)
#     animate_robot_movement(path, config_map, obstacles)

# if __name__ == '__main__':
#     main()


# import matplotlib.pyplot as PLT
# import numpy as NP
# import random as RANDOM
# from scipy.spatial import KDTree
# import networkx as NX
# from matplotlib.animation import FuncAnimation

# # Constants
# ARM_ROBOT_LINK_1_LENGTH = 2.0
# ARM_ROBOT_LINK_2_LENGTH = 1.5
# ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX = -10, 10
# ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX = -10, 10

# def get_arm_robot_joint_positions(theta_1, theta_2):
#     """Get positions of the base, joint, and end-effector."""
#     BASE = (0, 0)
#     JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
#     JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
#     JOINT = (JOINT_X, JOINT_Y)
#     END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
#     END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
#     END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
#     return BASE, JOINT, END_EFFECTOR

# def wrap_angle_difference(start, end):
#     """Ensure consistent angle wrapping for shortest path rotation."""
#     delta = (end - start + NP.pi) % (2 * NP.pi) - NP.pi
#     return start + delta

# def interpolate_angles_consistent(start_config, end_config, steps=20):
#     """Generate interpolated angles with consistent rotational motion."""
#     theta_1_start, theta_2_start = start_config
#     theta_1_end, theta_2_end = end_config

#     # Wrap angles to ensure shortest rotational path
#     theta_1_end_wrapped = wrap_angle_difference(theta_1_start, theta_1_end)
#     theta_2_end_wrapped = wrap_angle_difference(theta_2_start, theta_2_end)

#     # Generate linear interpolation for both angles
#     theta_1_vals = NP.linspace(theta_1_start, theta_1_end_wrapped, steps)
#     theta_2_vals = NP.linspace(theta_2_start, theta_2_end_wrapped, steps)

#     return [(theta_1, theta_2) for theta_1, theta_2 in zip(theta_1_vals, theta_2_vals)]

# def scene_from_file(filename):
#     """Load obstacles from a file."""
#     obstacles = []
#     with open(filename, 'r') as f:
#         for line in f:
#             x, y, width, height, _, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
#             obstacles.append((x, y, width, height))
#     return obstacles

# def is_valid_configuration(theta_1, theta_2, obstacles):
#     """Check if the arm configuration is valid (no collisions)."""
#     base, joint, end_effector = get_arm_robot_joint_positions(theta_1, theta_2)

#     for obs_x, obs_y, obs_width, obs_height in obstacles:
#         if (obs_x - obs_width / 2 <= joint[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= joint[1] <= obs_y + obs_height / 2):
#             return False
#         if (obs_x - obs_width / 2 <= end_effector[0] <= obs_x + obs_width / 2 and
#                 obs_y - obs_height / 2 <= end_effector[1] <= obs_y + obs_height / 2):
#             return False
#     return True

# def build_prm(nodes, k=5):
#     """Build the PRM graph using KDTree for nearest neighbors."""
#     graph = NX.Graph()
#     kdtree = KDTree(nodes)

#     for i, node in enumerate(nodes):
#         graph.add_node(i, pos=node)

#     for i, node in enumerate(nodes):
#         distances, indices = kdtree.query(node, k=k + 1)
#         for j in indices[1:]:
#             distance = NP.linalg.norm(NP.array(node) - NP.array(nodes[j]))
#             graph.add_edge(i, j, weight=distance)

#     return graph

# def animate_robot_movement(path, config_map, obstacles):
#     """Smoothly animate the robot along the path with consistent motion."""
#     fig, ax = PLT.subplots()
#     ax.set_aspect('equal')
#     ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
#     ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

#     for obstacle in obstacles:
#         x, y, width, height = obstacle
#         rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
#                              color='gray', alpha=0.5)
#         ax.add_patch(rect)

#     robot_line, = ax.plot([], [], 'b-', lw=2)
#     joint_marker, = ax.plot([], [], 'ro', markersize=6)
#     end_effector_marker, = ax.plot([], [], 'go', markersize=6)

#     # Generate a full interpolated path using consistent angle wrapping
#     full_path = []
#     for i in range(len(path) - 1):
#         config1 = config_map[path[i]]
#         config2 = config_map[path[i + 1]]
#         full_path.extend(interpolate_angles_consistent(config1, config2, steps=20))

#     def init():
#         robot_line.set_data([], [])
#         joint_marker.set_data([], [])
#         end_effector_marker.set_data([], [])
#         return robot_line, joint_marker, end_effector_marker

#     def update_frame(i):
#         config = full_path[i]
#         base, joint, end_effector = get_arm_robot_joint_positions(*config)

#         robot_line.set_data([base[0], joint[0], end_effector[0]],
#                             [base[1], joint[1], end_effector[1]])

#         joint_marker.set_data([joint[0]], [joint[1]])
#         end_effector_marker.set_data([end_effector[0]], [end_effector[1]])

#         return robot_line, joint_marker, end_effector_marker

#     ani = FuncAnimation(fig, update_frame, frames=len(full_path),
#                         init_func=init, interval=100, blit=True, repeat=False)

#     PLT.show()

# def main():
#     obstacles = scene_from_file('environment_1_10.txt')

#     valid_configs = [
#         (theta_1, theta_2) for theta_1 in NP.linspace(0, NP.pi, 10)
#         for theta_2 in NP.linspace(0, NP.pi, 10)
#         if is_valid_configuration(theta_1, theta_2, obstacles)
#     ]
#     end_effector_positions = [get_arm_robot_joint_positions(*config)[2] for config in valid_configs]
#     config_map = {i: config for i, config in enumerate(valid_configs)}

#     graph = build_prm(end_effector_positions)

#     start_config = (0.0, 0.0)
#     goal_config = (1.5, 0.0)

#     _, _, start_pos = get_arm_robot_joint_positions(*start_config)
#     _, _, goal_pos = get_arm_robot_joint_positions(*goal_config)

#     start_idx = add_node_to_prm(graph, end_effector_positions, config_map, start_pos)
#     goal_idx = add_node_to_prm(graph, end_effector_positions, config_map, goal_pos)

#     path = find_path(graph, start_idx, goal_idx)

#     visualize_prm(end_effector_positions, graph.edges, obstacles)
#     visualize_path(end_effector_positions, path, obstacles)
#     animate_robot_movement(path, config_map, obstacles)

# if __name__ == '__main__':
#     main()

import matplotlib.pyplot as PLT
import numpy as NP
import random as RANDOM
from scipy.spatial import KDTree
import networkx as NX
from matplotlib.animation import FuncAnimation

# Constants
ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5
ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX = -10, 10
ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX = -10, 10

def get_arm_robot_joint_positions(theta_1, theta_2):
    """Get positions of the base, joint, and end-effector."""
    BASE = (0, 0)
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    return BASE, JOINT, END_EFFECTOR

def scene_from_file(filename):
    """Load obstacles from a file."""
    obstacles = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, width, height, _, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
            obstacles.append((x, y, width, height))
    return obstacles

def is_valid_configuration(theta_1, theta_2, obstacles):
    """Check if the arm configuration is valid (no collisions)."""
    base, joint, end_effector = get_arm_robot_joint_positions(theta_1, theta_2)

    for obs_x, obs_y, obs_width, obs_height in obstacles:
        if (obs_x - obs_width / 2 <= joint[0] <= obs_x + obs_width / 2 and
                obs_y - obs_height / 2 <= joint[1] <= obs_y + obs_height / 2):
            return False
        if (obs_x - obs_width / 2 <= end_effector[0] <= obs_x + obs_width / 2 and
                obs_y - obs_height / 2 <= end_effector[1] <= obs_y + obs_height / 2):
            return False
    return True

def is_valid_edge(config1, config2, obstacles, steps=10):
    """Check if the interpolated path between two configurations is collision-free."""
    for (theta_1, theta_2) in interpolate_angles_consistent(config1, config2, steps):
        if not is_valid_configuration(theta_1, theta_2, obstacles):
            return False
    return True

def interpolate_angles_consistent(start_config, end_config, steps=25):
    """Generate interpolated angles with consistent rotational motion."""
    theta_1_start, theta_2_start = start_config
    theta_1_end, theta_2_end = end_config

    # Wrap angles to ensure shortest rotational path
    theta_1_end_wrapped = wrap_angle_difference(theta_1_start, theta_1_end)
    theta_2_end_wrapped = wrap_angle_difference(theta_2_start, theta_2_end)

    # Generate linear interpolation for both angles
    theta_1_vals = NP.linspace(theta_1_start, theta_1_end_wrapped, steps)
    theta_2_vals = NP.linspace(theta_2_start, theta_2_end_wrapped, steps)

    return [(theta_1, theta_2) for theta_1, theta_2 in zip(theta_1_vals, theta_2_vals)]

def wrap_angle_difference(start, end):
    """Ensure consistent angle wrapping for shortest path rotation."""
    delta = (end - start + NP.pi) % (2 * NP.pi) - NP.pi
    return start + delta

def build_prm(nodes, obstacles, k=5):
    """Build the PRM graph with collision-aware edges."""
    graph = NX.Graph()
    kdtree = KDTree(nodes)

    for i, node in enumerate(nodes):
        graph.add_node(i, pos=node)

    for i, node in enumerate(nodes):
        distances, indices = kdtree.query(node, k=k + 1)
        for j in indices[1:]:
            if is_valid_edge(node, nodes[j], obstacles):  # Add only valid edges
                distance = NP.linalg.norm(NP.array(node) - NP.array(nodes[j]))
                graph.add_edge(i, j, weight=distance)

    return graph

def animate_robot_movement(path, config_map, obstacles):
    """Smoothly animate the robot along the path with consistent motion."""
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

    for obstacle in obstacles:
        x, y, width, height = obstacle
        rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height,
                             color='gray', alpha=0.5)
        ax.add_patch(rect)

    robot_line, = ax.plot([], [], 'b-', lw=2)
    joint_marker, = ax.plot([], [], 'ro', markersize=6)
    end_effector_marker, = ax.plot([], [], 'go', markersize=6)

    full_path = []
    for i in range(len(path) - 1):
        config1 = config_map[path[i]]
        config2 = config_map[path[i + 1]]
        full_path.extend(interpolate_angles_consistent(config1, config2, steps=20))

    def init():
        robot_line.set_data([], [])
        joint_marker.set_data([], [])
        end_effector_marker.set_data([], [])
        return robot_line, joint_marker, end_effector_marker

    def update_frame(i):
        config = full_path[i]
        base, joint, end_effector = get_arm_robot_joint_positions(*config)

        robot_line.set_data([base[0], joint[0], end_effector[0]],
                            [base[1], joint[1], end_effector[1]])

        joint_marker.set_data([joint[0]], [joint[1]])
        end_effector_marker.set_data([end_effector[0]], [end_effector[1]])

        return robot_line, joint_marker, end_effector_marker

    ani = FuncAnimation(fig, update_frame, frames=len(full_path),
                        init_func=init, interval=100, blit=True, repeat=False)

    PLT.show()

def main():
    obstacles = scene_from_file('environment_1_10.txt')

    valid_configs = [
        (theta_1, theta_2) for theta_1 in NP.linspace(0, NP.pi, 10)
        for theta_2 in NP.linspace(0, NP.pi, 10)
        if is_valid_configuration(theta_1, theta_2, obstacles)
    ]
    end_effector_positions = [get_arm_robot_joint_positions(*config)[2] for config in valid_configs]
    config_map = {i: config for i, config in enumerate(valid_configs)}

    graph = build_prm(end_effector_positions, obstacles)

    start_config = (0.0, 0.0)
    goal_config = (3.0, 3.0) 

    _, _, start_pos = get_arm_robot_joint_positions(*start_config)
    _, _, goal_pos = get_arm_robot_joint_positions(*goal_config)

    start_idx = add_node_to_prm(graph, end_effector_positions, config_map, start_pos)
    goal_idx = add_node_to_prm(graph, end_effector_positions, config_map, goal_pos)

    path = find_path(graph, start_idx, goal_idx)

    animate_robot_movement(path, config_map, obstacles)

if __name__ == '__main__':
    main()

