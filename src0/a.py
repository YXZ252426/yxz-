import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class QueueFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier.pop(0)
            return node


class Maze:
    def __init__(self, filename):
        # Initialize an empty maze and set start and goal
        self.maze = []
        self.start = None
        self.goal = None

        # Read the file and set the maze, start, and goal
        with open(filename) as f:
            contents = f.read()
        self.load(contents)

    def load(self, contents):
        contents = contents.splitlines()
        for row, line in enumerate(contents):
            row_contents = []
            for col, char in enumerate(line):
                if char == "A":
                    self.start = (row, col)
                elif char == "B":
                    self.goal = (row, col)
                if char != "#":
                    row_contents.append(True)  # Path
                else:
                    row_contents.append(False)  # Wall
            self.maze.append(row_contents)

    def to_numpy_array(self):
        maze_array = np.zeros((len(self.maze), len(self.maze[0])), dtype=int)
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                maze_array[i, j] = 1 if cell else 0
        return maze_array

    def solve(self):
        # Initialize frontier to the starting position
        start_node = Node(state=self.start, parent=None, action=None)
        frontier = QueueFrontier()
        frontier.add(start_node)

        # Initialize an empty explored set
        self.explored = set()

        # Loop until the frontier is empty
        while not frontier.empty():
            # Remove a node from the frontier
            current_node = frontier.remove()

            # Check if we have reached the goal
            if current_node.state == self.goal:
                actions = []
                cells = []
                while current_node.parent is not None:
                    actions.append(current_node.action)
                    cells.append(current_node.state)
                    current_node = current_node.parent
                actions.reverse()
                cells.reverse()
                return actions, cells

            # Mark node as explored
            self.explored.add(current_node.state)

            # Add neighbors to frontier
            for action, (dr, dc) in [("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))]:
                row, col = current_node.state
                next_state = (row + dr, col + dc)
                if 0 <= next_state[0] < len(self.maze) and 0 <= next_state[1] < len(self.maze[0]) and \
                   self.maze[next_state[0]][next_state[1]] and next_state not in self.explored:
                    child_node = Node(state=next_state,
                                      parent=current_node, action=action)
                    if not frontier.contains_state(next_state):
                        frontier.add(child_node)

        return None


def visualize_maze_animation(maze_array, start, goal, solution_steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(maze_array, cmap='Greys')
    ax.text(start[1], start[0], 'A', ha='center', va='center', color='green')
    ax.text(goal[1], goal[0], 'B', ha='center', va='center', color='red')
    ax.axis('off')

    # 用于动画的矩形
    rect = plt.Rectangle((start[1]-0.5, start[0]-0.5),
                         1, 1, fill=False, color='blue', lw=2)
    ax.add_patch(rect)

    def animate(step):
        row, col = step
        rect.set_xy((col-0.5, row-0.5))

    anim = FuncAnimation(fig, animate, frames=solution_steps,
                         interval=300, repeat=False)
    plt.show()


maze = Maze("C:/Users/叶xz/Desktop/src0/maze2.txt")  # 用实际的文件路径替换

# 求解迷宫
solution = maze.solve()

# 转换迷宫为 NumPy 数组
maze_array = maze.to_numpy_array()

# 可视化迷宫
if solution:
    actions, solution_steps = solution
    visualize_maze_animation(maze_array, maze.start, maze.goal, solution_steps)
else:
    print("No solution found")
