import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import random
import math
import argparse

class RRTBase:
    def __init__(self, start, goal, goal_radius, map_file, max_iterations=1000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.max_iterations = max_iterations
        self.tree = [self.start]
        self.parent = {tuple(self.start): None}
        self.obstacles = self.load_map(map_file)

    def load_map(self, filename):
        """Load obstacles from the given map file."""
        obstacles = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                x, y, width, height, theta = map(float, line.strip().split(','))
                corners = self.get_rotated_corners(x, y, width, height, theta)
                obstacles.append(Polygon(corners))
        return obstacles

    def get_rotated_corners(self, cx, cy, width, height, angle):
        """Get the corners of a rotated rectangle."""
        corners = np.array([
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2]
        ])
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        return rotated_corners

    def is_collision_free(self, p1, p2):
        """Check if the path between two points is collision-free."""
        line = np.linspace(p1, p2, num=100)
        for point in line:
            if any(obstacle.contains_point(point) for obstacle in self.obstacles):
                return False
        return True

    def reconstruct_path(self, node):
        """Reconstruct the path from goal to start."""
        path = [node]
        while self.parent[tuple(node)] is not None:
            node = np.array(self.parent[tuple(node)])
            path.append(node)
        path.reverse()
        return path

    def search(self):
        """Perform the RRT search."""
        for _ in range(self.max_iterations):
            sampled_point = self.sample_random_point()
            nearest = self.nearest_node(sampled_point)
            new_node = self.extend(nearest, sampled_point)

            if new_node is not None and np.linalg.norm(new_node - self.goal) < self.goal_radius:
                print("Path found!")
                return self.reconstruct_path(new_node)

        print("Failed to find a path.")
        return None

    def visualize(self, path=None):
        """Visualize the RRT tree and the path, if available."""
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        for obstacle in self.obstacles:
            ax.add_patch(obstacle)

        for node in self.tree:
            if self.parent[tuple(node)] is not None:
                parent = np.array(self.parent[tuple(node)])
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 'k-', linewidth=0.5)

        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)

        ax.add_patch(Circle(self.start, 0.05, color='blue'))
        ax.add_patch(Circle(self.goal, self.goal_radius, color='red', alpha=0.5))
        plt.show()


# class ArmRRT(RRTBase):
#     def sample_random_point(self):
#         """Sample a random point for the arm robot."""
#         if random.random() < 0.05:
#             return self.goal
#         return np.array([
#             random.uniform(-np.pi, np.pi),  # theta_1
#             random.uniform(-np.pi, np.pi)   # theta_2
#         ])

#     def nearest_node(self, point):
#         """Find the nearest node in the tree to the given point."""
#         return min(self.tree, key=lambda node: np.linalg.norm(node - point))

#     def extend(self, nearest, sampled):
#         """Extend the tree towards the sampled point."""
#         direction = sampled - nearest
#         direction = direction / np.linalg.norm(direction)
#         new_point = nearest + direction * 0.1
#         if self.is_collision_free(nearest, new_point):
#             self.tree.append(new_point)
#             self.parent[tuple(new_point)] = tuple(nearest)
#             return new_point
#         return None

class ArmRRT(RRTBase):
    def __init__(self, start, goal, goal_radius, map_file, max_iterations=1000):
        super().__init__(start, goal, goal_radius, map_file, max_iterations)

        # Convert start and goal configurations to end-effector positions
        self.start_pos = self.forward_kinematics(start)
        self.goal_pos = self.forward_kinematics(goal)

        # Reset the tree to store end-effector positions instead of angles
        self.tree = [self.start_pos]
        self.parent = {tuple(self.start_pos): None}

    def forward_kinematics(self, angles):
        """Convert joint angles (theta_1, theta_2) to end-effector position (x, y)."""
        theta_1, theta_2 = angles
        L1, L2 = 2.0, 1.5  # Length of the two links

        # Calculate the joint positions
        joint_x = L1 * math.cos(theta_1)
        joint_y = L1 * math.sin(theta_1)

        # Calculate the end-effector position
        end_x = joint_x + L2 * math.cos(theta_1 + theta_2)
        end_y = joint_y + L2 * math.sin(theta_1 + theta_2)

        return np.array([end_x, end_y])

    def sample_random_point(self):
        """Sample a random point for the arm robot."""
        if random.random() < 0.05:
            return self.goal_pos  # Occasionally bias towards the goal
        random_angles = np.array([
            random.uniform(-np.pi, np.pi),  # theta_1
            random.uniform(-np.pi, np.pi)   # theta_2
        ])
        return self.forward_kinematics(random_angles)

    def nearest_node(self, point):
        """Find the nearest node in the tree to the given point."""
        return min(self.tree, key=lambda node: np.linalg.norm(node - point))

    def extend(self, nearest, sampled):
        """Extend the tree towards the sampled point."""
        direction = sampled - nearest
        direction = direction / np.linalg.norm(direction)
        new_point = nearest + direction * 0.1  # Step size of 0.1

        if self.is_collision_free(nearest, new_point):
            self.tree.append(new_point)
            self.parent[tuple(new_point)] = tuple(nearest)
            return new_point
        return None

    def visualize(self, path=None):
        """Visualize the RRT tree using end-effector positions."""
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Draw obstacles
        for obstacle in self.obstacles:
            ax.add_patch(obstacle)

        # Draw the RRT tree using end-effector positions
        for node in self.tree:
            if self.parent[tuple(node)] is not None:
                parent = np.array(self.parent[tuple(node)])
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 'k-', linewidth=0.5)

        # Draw the path if available
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)

        # Draw start and goal positions
        ax.add_patch(Circle(self.start_pos, 0.05, color='blue'))
        ax.add_patch(Circle(self.goal_pos, self.goal_radius, color='red', alpha=0.5))

        plt.show()


class FreeBodyRRT(RRTBase):
    def sample_random_point(self):
        """Sample a random point for the freeBody robot."""
        if random.random() < 0.05:
            return self.goal
        return np.array([
            random.uniform(-10, 10),  # x
            random.uniform(-10, 10),  # y
            random.uniform(-np.pi, np.pi)  # theta
        ])

    def nearest_node(self, point):
        """Find the nearest node in the tree to the given point."""
        return min(self.tree, key=lambda node: np.linalg.norm(node - point))

    def extend(self, nearest, sampled):
        """Extend the tree towards the sampled point."""
        direction = sampled - nearest
        direction = direction / np.linalg.norm(direction)
        new_point = nearest + direction * 0.1
        if self.is_collision_free(nearest, new_point):
            self.tree.append(new_point)
            self.parent[tuple(new_point)] = tuple(nearest)
            return new_point
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RRT Path Planning')
    parser.add_argument('--robot', type=str, choices=['arm', 'freeBody'], required=True)
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    args = parser.parse_args()

    if args.robot == 'arm':
        rrt = ArmRRT(args.start, args.goal, args.goal_rad, args.map)
    elif args.robot == 'freeBody':
        rrt = FreeBodyRRT(args.start, args.goal, args.goal_rad, args.map)

    path = rrt.search()
    rrt.visualize(path)
