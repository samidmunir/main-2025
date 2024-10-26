# # # # import argparse
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from matplotlib.patches import Rectangle
# # # # import random
# # # # from collections import deque

# # # # def parse_arguments():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument('--robot', type=str, choices=['arm'], required=True)
# # # #     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
# # # #     return parser.parse_args()

# # # # def forward_kinematics(theta_1, theta_2):
# # # #     link_1 = 2.0
# # # #     link_2 = 1.5

# # # #     # Base is always at (0, 0)
# # # #     joint_x = link_1 * np.cos(theta_1)
# # # #     joint_y = link_1 * np.sin(theta_1)

# # # #     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
# # # #     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

# # # #     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # # def load_obstacles(map_file):
# # #     obstacles = []
# # #     with open(map_file, 'r') as f:
# # #         for line in f:
# # #             x, y, width, height, theta, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
# # #             obstacles.append((x, y, width, height, theta))
# # #     return obstacles

# # # # def is_collision(robot_points, obstacles):
# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
# # # #         for px, py in robot_points:
# # # #             if rect.contains_point((px, py)):
# # # #                 return True
# # # #     return False

# # # # def prm(start, goal, obstacles, num_nodes=5000, k=6):
# # # #     nodes = [start]
# # # #     edges = []

# # # #     while len(nodes) < num_nodes:
# # # #         theta_1 = random.uniform(0, 2 * np.pi)
# # # #         theta_2 = random.uniform(0, 2 * np.pi)
# # # #         config = (theta_1, theta_2)
        
# # # #         points = forward_kinematics(theta_1, theta_2)
# # # #         if not is_collision(points, obstacles):
# # # #             nodes.append(config)

# # # #     # Connect each node to its k-nearest neighbors
# # # #     for i, node in enumerate(nodes):
# # # #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# # # #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])
# # # #         for j, _ in distances[:k]:
# # # #             edges.append((node, nodes[j]))

# # # #     return nodes, edges

# # # # def a_star_search(start, goal, nodes, edges):
# # # #     frontier = deque([(0, start)])
# # # #     came_from = {start: None}
# # # #     cost_so_far = {start: 0}

# # # #     while frontier:
# # # #         _, current = frontier.popleft()

# # # #         if current == goal:
# # # #             break

# # # #         for next_node in [n for n1, n in edges if n1 == current]:
# # # #             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
# # # #             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
# # # #                 cost_so_far[next_node] = new_cost
# # # #                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
# # # #                 frontier.append((priority, next_node))
# # # #                 came_from[next_node] = current

# # # #     path = []
# # # #     node = goal
# # # #     while node:
# # # #         path.append(node)
# # # #         node = came_from.get(node)
# # # #     return path[::-1]

# # # # def visualize_prm(nodes, edges, obstacles):
# # # #     fig, ax = plt.subplots()
# # # #     ax.set_aspect('equal')

# # # #     # Draw obstacles
# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # # #         ax.add_patch(rect)

# # # #     # Draw nodes and edges
# # # #     for node in nodes:
# # # #         x, y = forward_kinematics(*node)[-1]
# # # #         ax.plot(x, y, 'bo', markersize=2)

# # # #     for (n1, n2) in edges:
# # # #         x1, y1 = forward_kinematics(*n1)[-1]
# # # #         x2, y2 = forward_kinematics(*n2)[-1]
# # # #         ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)

# # # #     plt.show()

# # # # def animate_solution(path, obstacles):
# # # #     fig, ax = plt.subplots()
# # # #     ax.set_aspect('equal')

# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # # #         ax.add_patch(rect)

# # # #     for config in path:
# # # #         points = forward_kinematics(*config)
# # # #         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
# # # #         plt.pause(0.1)

# # # #     plt.show()

# # # # if __name__ == "__main__":
# # # #     args = parse_arguments()

# # # #     start = tuple(args.start)
# # # #     goal = tuple(args.goal)
# # # #     obstacles = load_obstacles(args.map)

# # # #     nodes, edges = prm(start, goal, obstacles)
# # # #     path = a_star_search(start, goal, nodes, edges)

# # # #     visualize_prm(nodes, edges, obstacles)
# # # #     animate_solution(path, obstacles)

# # # # import argparse
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from matplotlib.patches import Rectangle
# # # # import random
# # # # from collections import deque

# # # # # Argument Parsing
# # # # def parse_arguments():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument('--robot', type=str, choices=['arm'], required=True)
# # # #     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
# # # #     return parser.parse_args()

# # # # # Forward Kinematics
# # # # def forward_kinematics(theta_1, theta_2):
# # # #     link_1 = 2.0
# # # #     link_2 = 1.5

# # # #     # Calculate joint and end-effector positions
# # # #     joint_x = link_1 * np.cos(theta_1)
# # # #     joint_y = link_1 * np.sin(theta_1)
# # # #     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
# # # #     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

# # # #     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # # # # Collision Detection
# # # # def is_collision(robot_points, obstacles):
# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
# # # #         for px, py in robot_points:
# # # #             if rect.contains_point((px, py)):
# # # #                 return True
# # # #     return False

# # # # # PRM Generation
# # # # def prm(start, goal, obstacles, num_nodes=5000, k=6):
# # # #     nodes = [start]
# # # #     edges = []

# # # #     while len(nodes) < num_nodes:
# # # #         theta_1 = random.uniform(0, 2 * np.pi)
# # # #         theta_2 = random.uniform(0, 2 * np.pi)
# # # #         config = (theta_1, theta_2)

# # # #         points = forward_kinematics(theta_1, theta_2)
# # # #         if not is_collision(points, obstacles):
# # # #             nodes.append(config)

# # # #     # Connect each node to its k-nearest neighbors
# # # #     for i, node in enumerate(nodes):
# # # #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# # # #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])
# # # #         for j, _ in distances[:k]:
# # # #             edges.append((node, nodes[j]))

# # # #     return nodes, edges

# # # # # A* Search Algorithm
# # # # def a_star_search(start, goal, nodes, edges):
# # # #     frontier = deque([(0, start)])
# # # #     came_from = {start: None}
# # # #     cost_so_far = {start: 0}

# # # #     while frontier:
# # # #         _, current = frontier.popleft()

# # # #         if current == goal:
# # # #             break

# # # #         for next_node in [n for n1, n in edges if n1 == current]:
# # # #             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
# # # #             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
# # # #                 cost_so_far[next_node] = new_cost
# # # #                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
# # # #                 frontier.append((priority, next_node))
# # # #                 came_from[next_node] = current

# # # #     path = []
# # # #     node = goal
# # # #     while node:
# # # #         path.append(node)
# # # #         node = came_from.get(node)
# # # #     return path[::-1]

# # # PRM Visualization with Path Overlay
# # # def visualize_prm_with_path(nodes, edges, obstacles, path):
# #     fig, ax = plt.subplots()
# #     ax.set_aspect('equal')

# #     # Draw obstacles
# #     for (x, y, width, height, theta) in obstacles:
# #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# #         ax.add_patch(rect)

# #     # Draw PRM nodes and edges
# #     for node in nodes:
# #         x, y = forward_kinematics(*node)[-1]
# #         ax.plot(x, y, 'bo', markersize=2)

# #     for (n1, n2) in edges:
# #         x1, y1 = forward_kinematics(*n1)[-1]
# #         x2, y2 = forward_kinematics(*n2)[-1]
# #         ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)

# #     # Overlay the path
# #     path_points = [forward_kinematics(*config)[-1] for config in path]
# #     ax.plot([p[0] for p in path_points], [p[1] for p in path_points], 'g-', linewidth=2)

# #     plt.show()

# # # # # Smooth Animation of the Arm Robot along the Path
# # # # def animate_solution(path, obstacles):
# # # #     fig, ax = plt.subplots()
# # # #     ax.set_aspect('equal')

# # # #     # Draw obstacles
# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # # #         ax.add_patch(rect)

# # # #     # Animate the arm robot along the path
# # # #     for config in path:
# # # #         points = forward_kinematics(*config)
# # # #         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
# # # #         plt.pause(0.1)  # Pause for smooth animation
# # # #         ax.cla()  # Clear the axis for the next frame

# # # #         # Redraw obstacles for each frame
# # # #         for (x, y, width, height, theta) in obstacles:
# # # #             rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # # #             ax.add_patch(rect)

# # # #     plt.show()

# # # # # Main Execution
# # # # if __name__ == "__main__":
# # # #     args = parse_arguments()

# # # #     start = tuple(args.start)
# # # #     goal = tuple(args.goal)
# # # #     obstacles = load_obstacles(args.map)

# # # #     nodes, edges = prm(start, goal, obstacles)
# # # #     path = a_star_search(start, goal, nodes, edges)

# # # #     visualize_prm_with_path(nodes, edges, obstacles, path)
# # # #     animate_solution(path, obstacles)

# # # # import argparse
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from matplotlib.patches import Rectangle
# # # # import random
# # # # from collections import deque
# # # # from matplotlib.animation import FuncAnimation

# # # # # Argument Parsing
# # # # def parse_arguments():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument('--robot', type=str, choices=['arm'], required=True)
# # # #     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
# # # #     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
# # # #     return parser.parse_args()

# # # # # Forward Kinematics for the Arm Robot
# # # # def forward_kinematics(theta_1, theta_2):
# # # #     link_1 = 2.0
# # # #     link_2 = 1.5

# # # #     # Calculate joint and end-effector positions
# # # #     joint_x = link_1 * np.cos(theta_1)
# # # #     joint_y = link_1 * np.sin(theta_1)
# # # #     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
# # # #     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

# # # #     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # # # # Collision Detection for the Arm Robot
# # # # def is_collision(robot_points, obstacles):
# # # #     for (x, y, width, height, theta) in obstacles:
# # # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
# # # #         for px, py in robot_points:
# # # #             if rect.contains_point((px, py)):
# # # #                 return True
# # # #     return False

# # # # # PRM Construction
# # # # def prm(start, goal, obstacles, num_nodes=5000, k=6):
# # # #     nodes = [start]
# # # #     edges = []

# # # #     while len(nodes) < num_nodes:
# # # #         theta_1 = random.uniform(0, 2 * np.pi)
# # # #         theta_2 = random.uniform(0, 2 * np.pi)
# # # #         config = (theta_1, theta_2)

# # # #         points = forward_kinematics(theta_1, theta_2)
# # # #         if not is_collision(points, obstacles):
# # # #             nodes.append(config)

# # # #     # Connect each node to its k-nearest neighbors
# # # #     for i, node in enumerate(nodes):
# # # #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# # # #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])
# # # #         for j, _ in distances[:k]:
# # # #             edges.append((node, nodes[j]))

# # # #     return nodes, edges

# # # # A* Search Algorithm to Find the Optimal Path
# # # def a_star_search(start, goal, nodes, edges):
# # #     frontier = deque([(0, start)])
# # #     came_from = {start: None}
# # #     cost_so_far = {start: 0}

# # #     while frontier:
# # #         _, current = frontier.popleft()

# # #         if current == goal:
# # #             break

# # #         for next_node in [n for n1, n in edges if n1 == current]:
# # #             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
# # #             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
# # #                 cost_so_far[next_node] = new_cost
# # #                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
# # #                 frontier.append((priority, next_node))
# # #                 came_from[next_node] = current

# # #     path = []
# # #     node = goal
# # #     while node:
# # #         path.append(node)
# # #         node = came_from.get(node)
# # #     return path[::-1]

# # # # Visualization of PRM with the Solution Path
# # # def visualize_prm_with_path(nodes, edges, obstacles, path):
# # #     fig, ax = plt.subplots()
# # #     ax.set_aspect('equal')

# # #     # Draw obstacles
# # #     for (x, y, width, height, theta) in obstacles:
# # #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # #         ax.add_patch(rect)

# # #     # Draw PRM nodes and edges
# # #     for node in nodes:
# # #         x, y = forward_kinematics(*node)[-1]
# # #         ax.plot(x, y, 'bo', markersize=2)

# # #     for (n1, n2) in edges:
# # #         x1, y1 = forward_kinematics(*n1)[-1]
# # #         x2, y2 = forward_kinematics(*n2)[-1]
# # #         ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)

# # #     # Overlay the path
# # #     path_points = [forward_kinematics(*config)[-1] for config in path]
# # #     ax.plot([p[0] for p in path_points], [p[1] for p in path_points], 'g-', linewidth=2)

# # #     plt.show()

# # # # # Animation of the Arm Robot Following the Solution Path
# # # # def animate_solution(path, obstacles):
# # # #     fig, ax = plt.subplots()
# # # #     ax.set_aspect('equal')

# # # #     # Draw Obstacles
# # # #     def draw_obstacles():
# # # #         for (x, y, width, height, theta) in obstacles:
# # # #             rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # # #             ax.add_patch(rect)

# # # #     def update(frame):
# # # #         ax.cla()  # Clear the axis for each frame
# # # #         draw_obstacles()  # Redraw obstacles

# # # #         config = path[frame]
# # # #         points = forward_kinematics(*config)

# # # #         # Draw the arm robot
# # # #         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
# # # #         ax.set_xlim(-5, 5)
# # # #         ax.set_ylim(-5, 5)

# # # #     # Create the animation
# # # #     ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=200)
# # # #     plt.show()

# # # # # Main Execution
# # # # if __name__ == "__main__":
# # # #     args = parse_arguments()

# # # #     start = tuple(args.start)
# # # #     goal = tuple(args.goal)
# # # #     obstacles = load_obstacles(args.map)

# # # #     nodes, edges = prm(start, goal, obstacles)
# # # #     path = a_star_search(start, goal, nodes, edges)
# # # #     print(len(path))

# # # #     visualize_prm_with_path(nodes, edges, obstacles, path)
# # # #     animate_solution(path, obstacles)


# # # import argparse
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from matplotlib.patches import Rectangle
# # # import random
# # # from collections import deque
# # # from matplotlib.animation import FuncAnimation

# # # def line_intersects_rect(p1, p2, rect):
# # #     """Check if a line segment (p1, p2) intersects with a rectangle."""
# # #     x, y, width, height, theta = rect

# # #     # Generate the 4 corners of the rotated rectangle
# # #     corners = np.array([
# # #         [x - width / 2, y - height / 2],
# # #         [x + width / 2, y - height / 2],
# # #         [x + width / 2, y + height / 2],
# # #         [x - width / 2, y + height / 2]
# # #     ])

# # #     # Rotate the corners around the center by the angle theta
# # #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
# # #                                 [np.sin(theta), np.cos(theta)]])
# # #     rotated_corners = np.dot(corners - np.array([x, y]), rotation_matrix.T) + np.array([x, y])

# # #     # Check if the line intersects any of the 4 edges of the rectangle
# # #     for i in range(4):
# # #         p3, p4 = rotated_corners[i], rotated_corners[(i + 1) % 4]
# # #         if line_segments_intersect(p1, p2, p3, p4):
# # #             return True
# # #     return False

# # # def line_segments_intersect(p1, p2, p3, p4):
# # #     """Check if two line segments (p1, p2) and (p3, p4) intersect."""
# # #     def orientation(a, b, c):
# # #         return np.sign((b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]))

# # #     o1 = orientation(p1, p2, p3)
# # #     o2 = orientation(p1, p2, p4)
# # #     o3 = orientation(p3, p4, p1)
# # #     o4 = orientation(p3, p4, p2)

# # #     return (o1 != o2) and (o3 != o4)

# # # # Argument Parsing
# # # def parse_arguments():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument('--robot', type=str, choices=['arm'], required=True)
# # #     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
# # #     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
# # #     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
# # #     return parser.parse_args()

# # # # Forward Kinematics for the Arm Robot
# # # def forward_kinematics(theta_1, theta_2):
# # #     link_1 = 2.0
# # #     link_2 = 1.5

# # #     joint_x = link_1 * np.cos(theta_1)
# # #     joint_y = link_1 * np.sin(theta_1)
# # #     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
# # #     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

# # #     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # # # Collision Detection
# # # def is_collision(robot_points, obstacles):
# # #     """Check if any part of the arm robot collides with obstacles."""
# # #     for (x, y, width, height, theta) in obstacles:
# # #         rect = (x, y, width, height, theta)

# # #         # Check for collisions with both links of the arm
# # #         if (line_intersects_rect(robot_points[0], robot_points[1], rect) or
# # #             line_intersects_rect(robot_points[1], robot_points[2], rect)):
# # #             return True
# # #     return False


# # # # # PRM Construction with Start and Goal Nodes
# # # # def prm(start, goal, obstacles, num_nodes=500, k=6):
# # # #     nodes = [start, goal]
# # # #     edges = []

# # # #     while len(nodes) < num_nodes:
# # # #         theta_1 = random.uniform(0, 2 * np.pi)
# # # #         theta_2 = random.uniform(0, 2 * np.pi)
# # # #         config = (theta_1, theta_2)

# # # #         if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
# # # #             nodes.append(config)

# # # #     # Connect nodes to their k-nearest neighbors
# # # #     for i, node in enumerate(nodes):
# # # #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# # # #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])
# # # #         for j, _ in distances[:k]:
# # # #             edges.append((node, nodes[j]))
# # # #             edges.append((nodes[j], node))

# # # #     return nodes, edges

# # # def prm(start, goal, obstacles, num_nodes=500, k=6):
# # #     nodes = [start, goal]
# # #     edges = []

# # #     # Generate random valid nodes
# # #     while len(nodes) < num_nodes:
# # #         theta_1 = random.uniform(0, 2 * np.pi)
# # #         theta_2 = random.uniform(0, 2 * np.pi)
# # #         config = (theta_1, theta_2)

# # #         if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
# # #             nodes.append(config)

# # #     # Connect nodes to their k-nearest neighbors with collision-free paths
# # #     for i, node in enumerate(nodes):
# # #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# # #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])

# # #         for j, _ in distances[:k]:
# # #             neighbor = nodes[j]
# # #             if not edge_collision(node, neighbor, obstacles):
# # #                 edges.append((node, neighbor))

# # #     return nodes, edges

# # # # def edge_collision(node1, node2, obstacles, steps=10):
# # # #     """Check if the edge between two nodes is collision-free using interpolation."""
# # # #     for t in np.linspace(0, 1, steps):
# # # #         intermediate_config = (1 - t) * np.array(node1) + t * np.array(node2)
# # # #         points = forward_kinematics(*intermediate_config)
# # # #         if is_collision(points, obstacles):
# # # #             return True  # Collision detected
# # # #     return False  # No collision

# # # def edge_collision(node1, node2, obstacles, max_step_size=0.1):
# # #     """Check if the edge between two nodes is collision-free using adaptive interpolation."""
# # #     # Calculate the Euclidean distance between the two nodes
# # #     distance = np.linalg.norm(np.array(node1) - np.array(node2))

# # #     # Determine the number of interpolation steps based on the distance
# # #     steps = max(int(distance / max_step_size), 10)

# # #     # Interpolate between the two nodes and check for collisions
# # #     for t in np.linspace(0, 1, steps):
# # #         intermediate_config = (1 - t) * np.array(node1) + t * np.array(node2)
# # #         points = forward_kinematics(*intermediate_config)

# # #         if is_collision(points, obstacles):
# # #             return True  # Collision detected

# # #     return False  # No collision


# # # # # def prm(start, goal, obstacles, num_nodes=500, radius=1.5):
# # # # #     """Builds a PRM graph with nodes connected within a certain radius."""
# # # # #     nodes = [start, goal]
# # # # #     edges = []

# # # # #     # Generate random valid nodes
# # # # #     while len(nodes) < num_nodes:
# # # # #         theta_1 = random.uniform(0, 2 * np.pi)
# # # # #         theta_2 = random.uniform(0, 2 * np.pi)
# # # # #         config = (theta_1, theta_2)

# # # # #         if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
# # # # #             nodes.append(config)

# # # # #     # Connect nodes only if they are within a certain radius and collision-free
# # # # #     for i, node in enumerate(nodes):
# # # # #         for j in range(i + 1, len(nodes)):
# # # # #             if np.linalg.norm(np.array(node) - np.array(nodes[j])) <= radius:
# # # # #                 if not edge_collision(node, nodes[j], obstacles):
# # # # #                     edges.append((node, nodes[j]))

# # # # #     return nodes, edges

# # # # # def edge_collision(node1, node2, obstacles, steps=10):
# # # # #     """Check if the edge between two nodes is collision-free using interpolation."""
# # # # #     for t in np.linspace(0, 1, steps):
# # # # #         intermediate_config = (1 - t) * np.array(node1) + t * np.array(node2)
# # # # #         points = forward_kinematics(*intermediate_config)
# # # # #         if is_collision(points, obstacles):
# # # # #             return True  # Collision detected
# # # # #     return False  # No collision

# # # # # A* Search Algorithm
# # # # def a_star_search(start, goal, nodes, edges):
# # # #     frontier = deque([(0, start)])
# # # #     came_from = {start: None}
# # # #     cost_so_far = {start: 0}

# # # #     while frontier:
# # # #         _, current = frontier.popleft()

# # # #         if current == goal:
# # # #             break

# # # #         for next_node in [n for n1, n in edges if n1 == current]:
# # # #             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
# # # #             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
# # # #                 cost_so_far[next_node] = new_cost
# # # #                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
# # # #                 frontier.append((priority, next_node))
# # # #                 came_from[next_node] = current

# # # #     # Reconstruct the path
# # # #     path = []
# # # #     node = goal
# # # #     while node:
# # # #         path.append(node)
# # # #         node = came_from.get(node)
# # # #     return path[::-1]

# # # # Linear Interpolation for Smooth Animation
# # # def interpolate_path(path, steps=50):
# # #     interpolated_path = []
# # #     for i in range(len(path) - 1):
# # #         start = np.array(path[i])
# # #         end = np.array(path[i + 1])
# # #         for t in np.linspace(0, 1, steps):
# # #             interpolated_config = (1 - t) * start + t * end
# # #             interpolated_path.append(tuple(interpolated_config))
# # #     return interpolated_path

# # # # Animation of the Arm Robot Following the Solution Path
# # # def animate_solution(path, obstacles):
# # #     fig, ax = plt.subplots()
# # #     ax.set_aspect('equal')

# # #     def draw_obstacles():
# # #         for (x, y, width, height, theta) in obstacles:
# # #             rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# # #             ax.add_patch(rect)

# # #     def update(frame):
# # #         ax.cla()  # Clear the axis for each frame
# # #         draw_obstacles()  # Redraw obstacles

# # #         config = path[frame]
# # #         points = forward_kinematics(*config)

# # #         # Draw the arm robot
# # #         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
# # #         ax.set_xlim(-5, 5)
# # #         ax.set_ylim(-5, 5)

# # #     ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=50)
# # #     plt.show()

# # # # Main Execution
# # # if __name__ == "__main__":
# # #     args = parse_arguments()

# # #     start = tuple(args.start)
# # #     goal = tuple(args.goal)
# # #     obstacles = load_obstacles(args.map)

# # #     nodes, edges = prm(start, goal, obstacles)
# # #     path = a_star_search(start, goal, nodes, edges)

# # #     visualize_prm_with_path(nodes, edges, obstacles, path)
# # #     if len(path) > 1:    
# # #         interpolated_path = interpolate_path(path)
# # #         animate_solution(interpolated_path, obstacles)
# # #     else:
# # #         print("No valid path found from start to goal.")


# # import argparse
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.patches import Rectangle
# # from collections import deque
# # from matplotlib.animation import FuncAnimation
# # import random

# # # Forward Kinematics for the Arm Robot
# # def forward_kinematics(theta_1, theta_2):
# #     link_1 = 2.0
# #     link_2 = 1.5

# #     joint_x = link_1 * np.cos(theta_1)
# #     joint_y = link_1 * np.sin(theta_1)
# #     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
# #     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

# #     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # # Collision Detection
# # def is_collision(robot_points, obstacles):
# #     for (x, y, width, height, theta) in obstacles:
# #         rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
# #         for p1, p2 in zip(robot_points[:-1], robot_points[1:]):
# #             if rect.contains_point(p1) or rect.contains_point(p2):
# #                 return True
# #     return False

# # # Load Obstacles from File
# # def load_obstacles(map_file):
# #     obstacles = []
# #     with open(map_file, 'r') as f:
# #         for line in f:
# #             x, y, width, height, theta, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
# #             obstacles.append((x, y, width, height, theta))
# #     return obstacles

# # # Edge Collision Check with Interpolation
# # def edge_collision(node1, node2, obstacles, steps=20):
# #     for t in np.linspace(0, 1, steps):
# #         config = (1 - t) * np.array(node1) + t * np.array(node2)
# #         points = forward_kinematics(*config)
# #         if is_collision(points, obstacles):
# #             return True
# #     return False

# # # Build PRM Graph
# # def prm(start, goal, obstacles, num_nodes=500, k=6):
# #     nodes = [start, goal]
# #     edges = []

# #     while len(nodes) < num_nodes:
# #         theta_1 = random.uniform(0, 2 * np.pi)
# #         theta_2 = random.uniform(0, 2 * np.pi)
# #         config = (theta_1, theta_2)

# #         if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
# #             nodes.append(config)

# #     # Connect nodes to k-nearest neighbors
# #     for i, node in enumerate(nodes):
# #         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
# #                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])

# #         for j, _ in distances[:k]:
# #             neighbor = nodes[j]
# #             if not edge_collision(node, neighbor, obstacles):
# #                 edges.append((node, neighbor))

# #     return nodes, edges

# # # A* Search Algorithm
# # def a_star_search(start, goal, nodes, edges):
# #     graph = build_graph(edges)
# #     frontier = deque([(0, start)])
# #     came_from = {start: None}
# #     cost_so_far = {start: 0}

# #     while frontier:
# #         _, current = frontier.popleft()

# #         if current == goal:
# #             break

# #         for next_node in graph[current]:
# #             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
# #             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
# #                 cost_so_far[next_node] = new_cost
# #                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
# #                 frontier.append((priority, next_node))
# #                 came_from[next_node] = current

# #     path = []
# #     node = goal
# #     while node:
# #         path.append(node)
# #         node = came_from.get(node)
# #     return path[::-1]

# # # Build Graph from Edges
# # def build_graph(edges):
# #     graph = {}
# #     for n1, n2 in edges:
# #         if n1 not in graph:
# #             graph[n1] = []
# #         if n2 not in graph:
# #             graph[n2] = []
# #         graph[n1].append(n2)
# #         graph[n2].append(n1)
# #     return graph

# # # Animate the Arm Robot Following the Path
# # def animate_solution(path, obstacles):
# #     fig, ax = plt.subplots()
# #     ax.set_aspect('equal')

# #     def draw_obstacles():
# #         for (x, y, width, height, theta) in obstacles:
# #             rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
# #             ax.add_patch(rect)

# #     def update(frame):
# #         ax.cla()
# #         draw_obstacles()
# #         config = path[frame]
# #         points = forward_kinematics(*config)
# #         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
# #         ax.set_xlim(-5, 5)
# #         ax.set_ylim(-5, 5)

# #     ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=100)
# #     plt.show()

# # # Parse Command Line Arguments
# # def parse_arguments():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
# #     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
# #     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
# #     return parser.parse_args()

# # # Main Function
# # if __name__ == "__main__":
# #     args = parse_arguments()

# #     start = tuple(args.start)
# #     goal = tuple(args.goal)
# #     obstacles = load_obstacles(args.map)

# #     nodes, edges = prm(start, goal, obstacles, num_nodes=500, k=6)

# #     print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")

# #     path = a_star_search(start, goal, nodes, edges)

# #     if len(path) > 1:
# #         print(f"Path found with {len(path)} configurations.")
# #         animate_solution(path, obstacles)
# #     else:
# #         print("No valid path found from start to goal.")


# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from collections import deque
# from matplotlib.animation import FuncAnimation
# import random

# # Forward Kinematics for the Arm Robot
# def forward_kinematics(theta_1, theta_2):
#     link_1 = 2.0
#     link_2 = 1.5

#     joint_x = link_1 * np.cos(theta_1)
#     joint_y = link_1 * np.sin(theta_1)
#     end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
#     end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

#     return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# # Collision Detection
# def is_collision(robot_points, obstacles):
#     for (x, y, width, height, theta) in obstacles:
#         rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
#         for p1, p2 in zip(robot_points[:-1], robot_points[1:]):
#             if rect.contains_point(p1) or rect.contains_point(p2):
#                 return True
#     return False

# # Load Obstacles from File
# def load_obstacles(map_file):
#     obstacles = []
#     with open(map_file, 'r') as f:
#         for line in f:
#             x, y, width, height, theta, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
#             obstacles.append((x, y, width, height, theta))
#     return obstacles

# # Edge Collision Check with Interpolation
# def edge_collision(node1, node2, obstacles, steps=20):
#     for t in np.linspace(0, 1, steps):
#         config = (1 - t) * np.array(node1) + t * np.array(node2)
#         points = forward_kinematics(*config)
#         if is_collision(points, obstacles):
#             return True
#     return False

# # PRM Construction
# def prm(start, goal, obstacles, num_nodes=500, k=6):
#     nodes = [start, goal]
#     edges = []

#     while len(nodes) < num_nodes:
#         theta_1 = random.uniform(0, 2 * np.pi)
#         theta_2 = random.uniform(0, 2 * np.pi)
#         config = (theta_1, theta_2)

#         if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
#             nodes.append(config)

#     # Connect nodes to k-nearest neighbors
#     for i, node in enumerate(nodes):
#         distances = sorted([(j, np.linalg.norm(np.array(node) - np.array(nodes[j]))) 
#                             for j in range(len(nodes)) if j != i], key=lambda x: x[1])

#         for j, _ in distances[:k]:
#             neighbor = nodes[j]
#             if not edge_collision(node, neighbor, obstacles):
#                 edges.append((node, neighbor))

#     # Ensure start and goal are connected
#     connect_to_nearest_neighbors(start, nodes, edges, obstacles, k)
#     connect_to_nearest_neighbors(goal, nodes, edges, obstacles, k)

#     return nodes, edges

# def connect_to_nearest_neighbors(node, nodes, edges, obstacles, k):
#     distances = sorted([(n, np.linalg.norm(np.array(node) - np.array(n))) 
#                         for n in nodes if n != node], key=lambda x: x[1])

#     for neighbor, _ in distances[:k]:
#         if not edge_collision(node, neighbor, obstacles):
#             edges.append((node, neighbor))

# # Build Graph from Edges
# def build_graph(edges, nodes):
#     graph = {node: [] for node in nodes}
#     for n1, n2 in edges:
#         graph[n1].append(n2)
#         graph[n2].append(n1)
#     return graph

# # A* Search Algorithm
# def a_star_search(start, goal, nodes, edges):
#     graph = build_graph(edges, nodes)

#     if start not in graph or goal not in graph:
#         print("Start or goal is not in the graph!")
#         return []

#     frontier = deque([(0, start)])
#     came_from = {start: None}
#     cost_so_far = {start: 0}

#     while frontier:
#         _, current = frontier.popleft()

#         if current == goal:
#             break

#         for next_node in graph[current]:
#             new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
#             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
#                 cost_so_far[next_node] = new_cost
#                 priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
#                 frontier.append((priority, next_node))
#                 came_from[next_node] = current

#     path = []
#     node = goal
#     while node:
#         path.append(node)
#         node = came_from.get(node)
#     return path[::-1]

# # Animate Solution
# def animate_solution(path, obstacles):
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')

#     def draw_obstacles():
#         for (x, y, width, height, theta) in obstacles:
#             rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
#             ax.add_patch(rect)

#     def update(frame):
#         ax.cla()
#         draw_obstacles()
#         config = path[frame]
#         points = forward_kinematics(*config)
#         ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
#         ax.set_xlim(-5, 5)
#         ax.set_ylim(-5, 5)

#     ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=100)
#     plt.show()

# # Parse Command Line Arguments
# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
#     parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
#     parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
#     return parser.parse_args()

# # Main Function
# if __name__ == "__main__":
#     args = parse_arguments()

#     start = tuple(args.start)
#     goal = tuple(args.goal)
#     obstacles = load_obstacles(args.map)

#     nodes, edges = prm(start, goal, obstacles, num_nodes=500, k=6)

#     print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")

#     path = a_star_search(start, goal, nodes, edges)

#     if len(path) > 1:
#         print(f"Path found with {len(path)} configurations.")
#         animate_solution(path, obstacles)
#     else:
#         print("No valid path found from start to goal.")


import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
from matplotlib.animation import FuncAnimation
import random

# Forward Kinematics for the Arm Robot
def forward_kinematics(theta_1, theta_2):
    link_1 = 2.0
    link_2 = 1.5

    joint_x = link_1 * np.cos(theta_1)
    joint_y = link_1 * np.sin(theta_1)
    end_effector_x = joint_x + link_2 * np.cos(theta_1 + theta_2)
    end_effector_y = joint_y + link_2 * np.sin(theta_1 + theta_2)

    return [(0, 0), (joint_x, joint_y), (end_effector_x, end_effector_y)]

# Collision Detection
def is_collision(robot_points, obstacles):
    for (x, y, width, height, theta) in obstacles:
        rect = Rectangle((x, y), width, height, angle=np.degrees(theta))
        for p1, p2 in zip(robot_points[:-1], robot_points[1:]):
            if rect.contains_point(p1) or rect.contains_point(p2):
                return True
    return False

# Load Obstacles from File
def load_obstacles(map_file):
    obstacles = []
    with open(map_file, 'r') as f:
        for line in f:
            x, y, width, height, theta, _, _, _, _, _, _, _, _ = map(float, line.strip().split(','))
            obstacles.append((x, y, width, height, theta))
    return obstacles

# Edge Collision Check with Interpolation
def edge_collision(node1, node2, obstacles, steps=10):
    """Check if the edge between two nodes is collision-free."""
    for t in np.linspace(0, 1, steps):
        config = (1 - t) * np.array(node1) + t * np.array(node2)
        points = forward_kinematics(*config)
        if is_collision(points, obstacles):
            return True
    return False

# Build PRM with Dynamic Radius Connection
def prm(start, goal, obstacles, num_nodes=600, radius=2.0):
    nodes = [start, goal]
    edges = []

    # Generate valid nodes
    while len(nodes) < num_nodes:
        theta_1 = random.uniform(0, 2 * np.pi)
        theta_2 = random.uniform(0, 2 * np.pi)
        config = (theta_1, theta_2)

        if not is_collision(forward_kinematics(theta_1, theta_2), obstacles):
            nodes.append(config)

    # Connect nodes within the given radius
    for i, node in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            if np.linalg.norm(np.array(node) - np.array(nodes[j])) <= radius:
                if not edge_collision(node, nodes[j], obstacles):
                    edges.append((node, nodes[j]))

    return nodes, edges

# Build Graph from Edges
def build_graph(edges, nodes):
    graph = {node: [] for node in nodes}
    for n1, n2 in edges:
        graph[n1].append(n2)
        graph[n2].append(n1)
    return graph

# A* Search Algorithm
def a_star_search(start, goal, nodes, edges):
    graph = build_graph(edges, nodes)

    if start not in graph or goal not in graph:
        print("Start or goal is not connected in the graph!")
        return []

    frontier = deque([(0, start)])
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = frontier.popleft()

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(next_node))
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next_node))
                frontier.append((priority, next_node))
                came_from[next_node] = current

    path = []
    node = goal
    while node:
        path.append(node)
        node = came_from.get(node)
    return path[::-1]

# Animate Solution
def animate_solution(path, obstacles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    def draw_obstacles():
        for (x, y, width, height, theta) in obstacles:
            rect = Rectangle((x, y), width, height, angle=np.degrees(theta), alpha=0.5, color='red')
            ax.add_patch(rect)

    def update(frame):
        ax.cla()
        draw_obstacles()
        config = path[frame]
        points = forward_kinematics(*config)
        ax.plot([p[0] for p in points], [p[1] for p in points], 'g-o')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=100)
    plt.show()

# Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=float, nargs=2, required=True, help="Start configuration (theta_1, theta_2)")
    parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal configuration (theta_1, theta_2)")
    parser.add_argument('--map', type=str, required=True, help="Obstacle map file")
    return parser.parse_args()

# Main Function
if __name__ == "__main__":
    args = parse_arguments()

    start = tuple(args.start)
    goal = tuple(args.goal)
    obstacles = load_obstacles(args.map)

    nodes, edges = prm(start, goal, obstacles, num_nodes=600, radius=2.0)

    print(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")

    path = a_star_search(start, goal, nodes, edges)

    if len(path) > 1:
        print(f"Path found with {len(path)} configurations.")
        animate_solution(path, obstacles)
    else:
        print("No valid path found from start to goal.")
