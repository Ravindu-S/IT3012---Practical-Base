# visual_grid_game.py
import random
import tkinter as tk
from agent import SearchAgent


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None, num_traps=5):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls and pos_tuple not in self.food_positions:
                self.toxic_traps.add(pos_tuple)

        self.score = 0
        self.steps = 0
        self.collision = False
        self.facing = 'Up'

    def get_percept(self) -> dict:
        # determine the coordinates of the adjacent cell ahead based on current facing direction
        x, y = self.agent_pos
        ahead_x, ahead_y = x, y

        if self.facing == 'Up':
            ahead_y = y + 1
        elif self.facing == 'Down':
            ahead_y = y - 1
        elif self.facing == 'Left':
            ahead_x = x - 1
        elif self.facing == 'Right':
            ahead_x = x + 1

        ahead_pos = (ahead_x, ahead_y)
        out_of_bounds = not (0 <= ahead_x < self.width and 0 <= ahead_y < self.height)

        return {
            'facing': self.facing,
            'wall_ahead': out_of_bounds or ahead_pos in self.walls,
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'food_ahead': ahead_pos in self.food_positions,
            'opponent_ahead': any(tuple(op) == ahead_pos for op in self.opponents),
            'toxin_ahead': ahead_pos in self.toxic_traps,
            'hit_wall': tuple(self.agent_pos) in self.walls,
            'collision': self.collision,
            'smells_toxin': tuple(self.agent_pos) in self.toxic_traps,
            'grid_size': (self.width, self.height),
            'walls': list(self.walls),
            'all_food': list(self.food_positions),
            'remaining_food': list(self.food_positions)
        }

    def execute_action(self, action: str):
        self.steps += 1
        new_pos = list(self.agent_pos)
        if action in ('Up', 'Down', 'Left', 'Right') and action != self.facing:
            self.facing = action  # a direction different from the current facing is a turn in place
            action = None         # no movement this step

        if action == 'Up':
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 'Down':
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 'Left':
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 'Right':
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        if tuple(new_pos) in self.walls:
            self.score -= 5
        else:
            self.agent_pos = new_pos

            if tuple(self.agent_pos) in self.toxic_traps:
                self.score -= 15

        tuple_pos = tuple(self.agent_pos)
        if action == 'Suck' and tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20

        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


class SimpleReflexAgent:
    """Step 1.2: strictly IF-THEN rules on the current percept. No __init__, no stored history."""

    def sense_and_act(self, percept):
        directions = ['Up', 'Right', 'Down', 'Left']
        facing = percept.get('facing', 'Up')

        if percept['food_here']:
            return 'Suck'
        if percept['wall_ahead']:
            return directions[(directions.index(facing) - 1) % 4]  # turn left
        return facing  # move forward


class ModelBasedAgent:
    """Step 1.3: keeps internal state (heading, relative position, visited cells, known walls)."""

    def __init__(self):
        self.rel_pos = (0, 0)  # position relative to the start cell
        self.visited_cells = {(0, 0)}
        self.known_walls = set()
        self.facing = 'Up'
        self.last_action = None
        self.last_percept = None

    def sense_and_act(self, percept):
        directions = ['Up', 'Right', 'Down', 'Left']

        # 1. Update state from the last action and the percepts
        if self.last_action in directions:
            if self.last_action != self.facing:
                self.facing = self.last_action  # last action was a turn
            elif not self.last_percept['wall_ahead']:
                self.rel_pos = self._peek(self.facing)  # last action was a successful step
                self.visited_cells.add(self.rel_pos)
        if percept['wall_ahead']:
            self.known_walls.add(self._peek(self.facing))

        # 2. IF-THEN rules that query the memory
        idx = directions.index(self.facing)
        left_dir = directions[(idx - 1) % 4]
        right_dir = directions[(idx + 1) % 4]

        if percept['food_here']:
            action = 'Suck'
        elif percept['wall_ahead']:
            left_cell = self._peek(left_dir)
            left_is_visited = left_cell in self.visited_cells or left_cell in self.known_walls
            action = right_dir if left_is_visited else left_dir
        else:
            action = self.facing  # move forward
            if self._peek(self.facing) in self.visited_cells:  # ahead already visited: probably looping
                for d in (left_dir, right_dir):
                    cell = self._peek(d)
                    if cell not in self.visited_cells and cell not in self.known_walls:
                        action = d  # turn towards an unexplored side instead
                        break

        self.last_action = action
        self.last_percept = percept
        return action

    def _peek(self, direction):
        x, y = self.rel_pos
        if direction == 'Up':
            y += 1
        elif direction == 'Down':
            y -= 1
        elif direction == 'Left':
            x -= 1
        elif direction == 'Right':
            x += 1
        return (x, y)


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None,
                 agent_class=ModelBasedAgent):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents,
                                      custom_walls=walls)
        self.agent = agent_class()

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")
        
        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.2
            x1 = tx * self.cell_size + offset
            y1 = (self.env.height - 1 - ty) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#800080", outline="#4B0082")


        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    # Try a larger grid size like 12x12 with 15 food and 3 opponents!
    # SearchAgent plans offline. Set SearchAgent.active_algo to 'BFS', 'DFS', 'UCS' or 'AStar' in agent.py.
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0, agent_class=SearchAgent)
    root.mainloop()