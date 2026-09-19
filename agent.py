# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


from collections import deque
import heapq
import math


class SearchAgent:
    """Practical 3: offline planning with uninformed search (BFS / DFS / UCS)."""

    MOVES = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'   # 'BFS', 'DFS', 'UCS' or 'AStar'
        self.pos = (0, 0)          # agent tracks its own position from the actions it takes
        self.facing = 'Up'

    # ---------- shared helpers ----------
    def _neighbors(self, state, walls, grid_size):
        width, height = grid_size
        for action, (dx, dy) in self.MOVES.items():
            nxt = (state[0] + dx, state[1] + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in walls:
                yield action, nxt

    def _build_path(self, came_from, goal):
        path = []
        node = goal
        while came_from[node] is not None:
            prev, action = came_from[node]
            path.append(action)
            node = prev
        path.reverse()
        return path

    # ---------- BFS: FIFO queue ----------
    def bfs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = deque([start])
        reached = {start}
        came_from = {start: None}
        while frontier:
            state = frontier.popleft()
            if state == goal:
                return self._build_path(came_from, state)
            for action, nxt in self._neighbors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    came_from[nxt] = (state, action)
                    frontier.append(nxt)
        return None

    # ---------- DFS: LIFO stack ----------
    def dfs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = [start]
        reached = {start}
        came_from = {start: None}
        while frontier:
            state = frontier.pop()
            if state == goal:
                return self._build_path(came_from, state)
            for action, nxt in self._neighbors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    came_from[nxt] = (state, action)
                    frontier.append(nxt)
        return None

    # ---------- UCS: priority queue ordered by path cost g(n) ----------
    def ucs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = [(0, start)]
        best_cost = {start: 0}
        came_from = {start: None}
        reached = set()
        while frontier:
            cost, state = heapq.heappop(frontier)
            if state in reached:
                continue
            reached.add(state)
            if state == goal:
                return self._build_path(came_from, state)
            for action, nxt in self._neighbors(state, walls, grid_size):
                new_cost = cost + 1
                if nxt not in reached and new_cost < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_cost
                    came_from[nxt] = (state, action)
                    heapq.heappush(frontier, (new_cost, nxt))
        return None

    # ---------- heuristics ----------
    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ---------- A*: priority queue ordered by f(n) = g(n) + h(n) ----------
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        walls = set(walls)
        heuristic = self.euclidean_distance if heuristic_type == 'euclidean' else self.manhattan_distance

        frontier = []
        reached_states = set()
        g = 0
        h = heuristic(start_pos, goal_pos)
        heapq.heappush(frontier, (g + h, g, start_pos, []))

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for action, nxt in self._neighbors(current_pos, walls, grid_size):
                if nxt not in reached_states:
                    g_new = g_cost + 1
                    h_new = heuristic(nxt, goal_pos)
                    f_new = g_new + h_new
                    heapq.heappush(frontier, (f_new, g_new, nxt, path_taken + [action]))
        return None

    # ---------- plan execution ----------
    def sense_and_act(self, percept):
        if percept['food_here']:
            return 'Suck'

        if not self.plan:
            food = percept['remaining_food']
            if not food:
                return 'Stay'
            goal_pos = tuple(min(food, key=lambda f: self.manhattan_distance(self.pos, f)))
            walls, grid_size = percept['walls'], percept['grid_size']

            if self.active_algo == 'BFS':
                path = self.bfs_search(self.pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(self.pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(self.pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'AStar':
                path = self.astar_search(self.pos, goal_pos, walls, grid_size)
            else:
                path = None
            self.plan = path if path else []
            if not self.plan:
                return 'Stay'

        action = self.plan.pop(0)
        # the environment turns in place when the direction differs from facing,
        # so re-queue the step and only turn this tick
        if action != self.facing:
            self.plan.insert(0, action)
            self.facing = action
            return action
        dx, dy = self.MOVES[action]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return action


if __name__ == "__main__":
    # Testing checkpoint for Step 1.1
    agent = SearchAgent()
    print("Manhattan (0,0)->(3,4):", agent.manhattan_distance((0, 0), (3, 4)))
    print("Euclidean (0,0)->(3,4):", agent.euclidean_distance((0, 0), (3, 4)))
