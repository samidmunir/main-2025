import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import random
import math
import argparse

class RRTStarBase:
    """Base class for RRT* with shared logic."""
    def __init__(self, start, goal, goal_radius, map_file, max_iterations=1000, radius=1.0):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.max_iterations = max_iterations
        self.radius = radius  # Radius for rewiring
        self.tree = [self.start]
        self.parent = {tuple(self.start): None}
        self.cost = {tuple(self.start): 0.0}  # Track cost from start to each node
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

    def nearest_node(self, point):
        """Find the nearest node in the tree to the given point."""
        return min(self.tree, key=lambda node: np.linalg.norm(node - point))

    def is_collision_free(self, p1, p2):
        """Check if the path between two points is collision-free."""
        line = np.linspace(p1, p2, num=20)  # Reduced number of checks for speed
        for point in line:
            if any(obstacle.contains_point(point) for obstacle in self.obstacles):
                return False
        return True

    def rewire(self, new_node):
        """Rewire the tree to ensure optimal paths."""
        for node in self.tree:
            if np.linalg.norm(node - new_node) < self.radius:
                new_cost = self.cost[tuple(new_node)] + np.linalg.norm(new_node - node)
                if new_cost < self.cost.get(tuple(node), float('inf')) and self.is_collision_free(new_node, node):
                    self.parent[tuple(node)] = tuple(new_node)
                    self.cost[tuple(node)] = new_cost

    def reconstruct_path(self, node):
        """Reconstruct the path from goal to start."""
        path = [node]
        while tuple(node) in self.parent:
            parent_node = self.parent[tuple(node)]
            if parent_node is None:
                break  # Reached the start node
            node = np.array(parent_node)
            if np.isnan(node).any():
                print(f"Encountered NaN node in path: {node}, skipping.")
                break  # Avoid adding NaN nodes to the path
            path.append(node)

        path.reverse()
        return path

    def search(self):
        """Perform the RRT* search."""
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
        """Visualize the RRT* tree and path."""
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Draw obstacles
        for obstacle in self.obstacles:
            ax.add_patch(obstacle)

        # Draw the RRT* tree
        for node in self.tree:
            if self.parent[tuple(node)] is not None:
                parent = np.array(self.parent[tuple(node)])
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 'k-', linewidth=0.5)

        # Draw the path if available
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)

        # Draw start and goal positions
        ax.add_patch(Circle(self.start, 0.05, color='blue'))
        ax.add_patch(Circle(self.goal, self.goal_radius, color='red', alpha=0.5))
        plt.show()


class ArmRRTStar(RRTStarBase):
    """RRT* implementation for the arm robot."""
    def __init__(self, start, goal, goal_radius, map_file, max_iterations=1000, radius=1.0):
        super().__init__(start, goal, goal_radius, map_file, max_iterations, radius)
        self.start_pos = self.forward_kinematics(start)
        self.goal_pos = self.forward_kinematics(goal)
        self.tree = [self.start_pos]
        self.parent = {tuple(self.start_pos): None}
        self.cost = {tuple(self.start_pos): 0.0}

    def forward_kinematics(self, angles):
        """Convert joint angles to end-effector position."""
        theta_1, theta_2 = angles
        L1, L2 = 2.0, 1.5  # Link lengths

        # Compute joint positions
        joint_x = L1 * math.cos(theta_1)
        joint_y = L1 * math.sin(theta_1)

        # Compute end-effector position
        end_x = joint_x + L2 * math.cos(theta_1 + theta_2)
        end_y = joint_y + L2 * math.sin(theta_1 + theta_2)

        return np.array([end_x, end_y])

    def sample_random_point(self):
        """Sample a random end-effector position."""
        if random.random() < 0.1:  # 10% chance to bias towards the goal
            return self.goal_pos

        random_angles = np.array([
            random.uniform(-np.pi, np.pi),
            random.uniform(-np.pi, np.pi)
        ])
        sampled_point = self.forward_kinematics(random_angles)

        if np.isnan(sampled_point).any():
            print(f"Skipping invalid sampled point: {sampled_point}")
            return self.sample_random_point()  # Resample if NaN

        return sampled_point

    def extend(self, nearest, sampled):
        """Extend the tree towards the sampled point."""
        direction = sampled - nearest
        distance = np.linalg.norm(direction)

        # Check for zero distance to avoid NaN issues
        if distance == 0:
            print("Skipping extension due to zero distance.")
            return None

        # Adaptive step size: move only part of the way towards the sampled point
        step_size = min(0.5, distance)  # Use 0.5 or the remaining distance, whichever is smaller
        direction = (direction / distance) * step_size  # Normalize and scale

        new_point = nearest + direction

        # Check if the new point is collision-free
        if self.is_collision_free(nearest, new_point):
            self.tree.append(new_point)
            self.parent[tuple(new_point)] = tuple(nearest)  # Track the parent
            self.cost[tuple(new_point)] = self.cost[tuple(nearest)] + np.linalg.norm(new_point - nearest)
            self.rewire(new_point)
            return new_point

        return None  # Return None if the extension fails



class FreeBodyRRTStar(RRTStarBase):
    """RRT* implementation for the freeBody robot."""
    def sample_random_point(self):
        """Sample a random pose (x, y, theta)."""
        if random.random() < 0.05:
            return self.goal
        return np.array([
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-np.pi, np.pi)
        ])

    def extend(self, nearest, sampled):
        """Extend the tree towards the sampled point."""
        direction = sampled - nearest
        direction = direction / np.linalg.norm(direction)
        new_point = nearest + direction * 0.1
        if self.is_collision_free(nearest, new_point):
            self.tree.append(new_point)
            self.parent[tuple(new_point)] = tuple(nearest)
            self.cost[tuple(new_point)] = self.cost[tuple(nearest)] + np.linalg.norm(new_point - nearest)
            self.rewire(new_point)
            return new_point
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RRT* Path Planning')
    parser.add_argument('--robot', type=str, choices=['arm', 'freeBody'], required=True)
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    args = parser.parse_args()

    if args.robot == 'arm':
        planner = ArmRRTStar(args.start, args.goal, args.goal_rad, args.map)
    elif args.robot == 'freeBody':
        planner = FreeBodyRRTStar(args.start, args.goal, args.goal_rad, args.map)

    path = planner.search()  # Now correctly inherited from the base class
    planner.visualize(path)
