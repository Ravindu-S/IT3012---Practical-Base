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


class SearchAgent:
    """Practical 3: offline planning with uninformed search (BFS / DFS / UCS)."""

    MOVES = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.plan = []
        self.active_algo = 'UCS'   # change to 'DFS' or 'UCS' and observe the paths
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

    # ---------- plan execution ----------
    def sense_and_act(self, percept):
        if percept['food_here']:
            return 'Suck'

        if not self.plan:
            food = percept['all_food']
            if not food:
                return 'Stay'
            goal = min(food, key=lambda f: abs(f[0] - self.pos[0]) + abs(f[1] - self.pos[1]))
            search = {'BFS': self.bfs_search, 'DFS': self.dfs_search, 'UCS': self.ucs_search}[self.active_algo]
            path = search(self.pos, tuple(goal), percept['walls'], percept['grid_size'])
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
