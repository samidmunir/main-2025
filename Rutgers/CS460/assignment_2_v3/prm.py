import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from nearest_neighbors import find_k_nearest_neighbors
from collision_checking import is_collision_free
from component_1 import scene_from_file

class PRM:
    def __init__(self, map_file, robot_type, num_samples=5000, k=6):
        self.map = scene_from_file(map_file)
        self.robot_type = robot_type
        self.num_samples = num_samples
        self.k = k
        self.nodes = []  # List of free configurations
        self.edges = []  # List of edges (pairs of node indices)

    def sample_free_space(self):
        """Sample valid configurations for the robot."""
        while len(self.nodes) < self.num_samples:
            config = self.sample_random_configuration()
            if is_collision_free(config, self.map, self.robot_type):
                self.nodes.append(config)

    def sample_random_configuration(self):
        """Generate a random configuration."""
        x = random.uniform(0, 20)  # Example environment dimensions
        y = random.uniform(0, 20)
        theta = random.uniform(0, 2 * np.pi) if self.robot_type == 'freeBody' else None
        return np.array([x, y, theta]) if theta is not None else np.array([x, y])

    def build_roadmap(self):
        """Connect nodes with valid edges."""
        for i, node in enumerate(self.nodes):
            neighbors = find_k_nearest_neighbors(node, self.nodes, self.k)
            for neighbor in neighbors:
                if self.is_edge_valid(node, neighbor):
                    self.edges.append((i, self.nodes.index(neighbor)))

    def is_edge_valid(self, config1, config2):
        """Check if the edge between two configurations is collision-free."""
        return is_collision_free((config1, config2), self.map, self.robot_type)

    def a_star_search(self, start_idx, goal_idx):
        """A* search to find a path on the PRM."""
        open_set = {start_idx}
        came_from = {}
        g_score = {i: float('inf') for i in range(len(self.nodes))}
        g_score[start_idx] = 0
        f_score = {i: float('inf') for i in range(len(self.nodes))}
        f_score[start_idx] = self.heuristic(self.nodes[start_idx], self.nodes[goal_idx])

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            if current == goal_idx:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(self.nodes[current], self.nodes[neighbor])
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(self.nodes[neighbor], self.nodes[goal_idx])
                    open_set.add(neighbor)

        return []

    def get_neighbors(self, node_idx):
        """Get neighbors from the roadmap."""
        return [edge[1] for edge in self.edges if edge[0] == node_idx] + \
               [edge[0] for edge in self.edges if edge[1] == node_idx]

    def heuristic(self, config1, config2):
        """Heuristic for A* (Euclidean distance)."""
        return np.linalg.norm(config1[:2] - config2[:2])

    def distance(self, config1, config2):
        """Calculate Euclidean distance."""
        return np.linalg.norm(config1[:2] - config2[:2])

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    def visualize_roadmap(self):
        """Visualize the PRM roadmap."""
        plt.figure()
        for edge in self.edges:
            p1, p2 = self.nodes[edge[0]], self.nodes[edge[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=0.5)

        for node in self.nodes:
            plt.plot(node[0], node[1], 'bo', markersize=2)

        plt.title("PRM Roadmap")
        plt.show()

    def visualize_path(self, path):
        """Visualize the final path."""
        plt.figure()
        for idx in range(len(path) - 1):
            p1, p2 = self.nodes[path[idx]], self.nodes[path[idx + 1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)

        for node in self.nodes:
            plt.plot(node[0], node[1], 'bo', markersize=2)

        plt.title("Path Found")
        plt.show()

    def run(self, start, goal):
        """Run the PRM process."""
        self.sample_free_space()
        self.build_roadmap()

        # Insert start and goal
        self.nodes.insert(0, start)
        self.nodes.append(goal)

        # Find path with A*
        path = self.a_star_search(0, len(self.nodes) - 1)

        # Visualize
        self.visualize_roadmap()
        if path:
            print("Found path:", path)
            self.visualize_path(path)
        else:
            print("No path found.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='PRM Algorithm for Path Planning')
    parser.add_argument('--robot', type=str, choices=['arm', 'freeBody'], required=True, help='Type of robot')
    parser.add_argument('--start', type=float, nargs='+', required=True, help='Start configuration')
    parser.add_argument('--goal', type=float, nargs='+', required=True, help='Goal configuration')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    prm = PRM(map_file=args.map, robot_type=args.robot)

    start = np.array(args.start)
    goal = np.array(args.goal)
    prm.run(start, goal)
