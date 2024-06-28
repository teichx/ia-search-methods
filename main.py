import collections
from datetime import timedelta
import heapq
import json
import time


class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def find_blank_square(state):
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val != 0:
                continue
            return i, j


def actions(state):
    moves = []
    row, col = find_blank_square(state)
    if row > 0:
        moves.append(("up", (row - 1, col)))
    if row < 2:
        moves.append(("down", (row + 1, col)))
    if col > 0:
        moves.append(("left", (row, col - 1)))
    if col < 2:
        moves.append(("right", (row, col + 1)))
    return moves


def apply_action(state, action):
    row, col = find_blank_square(state)
    dr, dc = action[1][0] - row, action[1][1] - col
    new_state = [row[:] for row in state]
    new_state[row][col], new_state[row + dr][col + dc] = (
        new_state[row + dr][col + dc],
        new_state[row][col],
    )

    return new_state


def heuristic_hamming_distance(state):
    hamming_distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == goal_state[i][j]:
                continue
            hamming_distance += 1

    return hamming_distance


def heuristic_manhattan(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            current = state[i][j]
            if current == 0:
                continue
            goal_i, goal_j = divmod(linear_goal_state.index(current), 3)
            distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def a_star_search(
    initial_state: list[list[int]],
    heuristic: "type.Callable[list[list[int]], int]",
):
    initial_node = Node(
        state=initial_state,
        g=0,
        h=heuristic(initial_state),
    )
    heap = [initial_node]
    visited = set()

    current_level = -1

    while heap:
        current_node = heapq.heappop(heap)

        if current_node.g > current_level:
            current_level = current_node.g

        if current_node.state == goal_state:
            return current_node, len(visited)
        visited.add(str(current_node.state))
        for action in actions(current_node.state):
            new_state = apply_action(current_node.state, action)
            if str(new_state) not in visited:
                new_node = Node(
                    state=new_state,
                    parent=current_node,
                    action=action,
                    g=current_node.g + 1,
                    h=heuristic(new_state),
                )
                heapq.heappush(heap, new_node)

    return None, len(visited)


def greedy_search(
    initial_state: list[list[int]],
    heuristic: "type.Callable[list[list[int]], int]",
):
    initial_node = Node(state=initial_state, h=heuristic(initial_state))
    heap = [initial_node]
    visited = set()

    while heap:
        current_node = heapq.heappop(heap)

        if current_node.state == goal_state:
            return current_node, len(visited)
        visited.add(str(current_node.state))
        for action in actions(current_node.state):
            new_state = apply_action(current_node.state, action)
            if str(new_state) not in visited:
                new_node = Node(
                    state=new_state,
                    parent=current_node,
                    action=action,
                    h=heuristic(new_state),
                )
                heapq.heappush(heap, new_node)

    return None, len(visited)


def bfs_search(initial_state: list[list[int]]):
    initial_node = Node(state=initial_state, g=0)
    queue = collections.deque([initial_node])
    visited = set()

    while queue:
        current_node = queue.popleft()

        if current_node.state == goal_state:
            return current_node, len(visited)
        visited.add(str(current_node.state))
        for action in actions(current_node.state):
            new_state = apply_action(current_node.state, action)
            if str(new_state) not in visited:
                new_node = Node(
                    state=new_state,
                    parent=current_node,
                    action=action,
                    g=current_node.g + 1,
                )
                queue.append(new_node)

    return None, len(visited)


def dfs_search(initial_state: list[list[int]]):
    initial_node = Node(state=initial_state, g=0)
    stack = [initial_node]
    visited = set()

    while stack:
        current_node = stack.pop()

        if current_node.state == goal_state:
            return current_node, len(visited)
        visited.add(str(current_node.state))
        for action in actions(current_node.state):
            new_state = apply_action(current_node.state, action)
            if str(new_state) not in visited:
                new_node = Node(
                    state=new_state,
                    parent=current_node,
                    action=action,
                    g=current_node.g + 1,
                )
                stack.append(new_node)

    return None, len(visited)


goal_state = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5],
]

linear_goal_state = [num for row in goal_state for num in row]


def run_once(run_algorithm):
    start_at = time.time()
    result, visited_count = run_algorithm()
    end_at = time.time()
    time_lapsed = str(timedelta(seconds=end_at - start_at))
    if not result:
        return {"time": time_lapsed, "moves": None, "visited": visited_count}

    moves = []
    node = result
    while node.parent is not None:
        moves.append(node.action[0])
        node = node.parent
    return {"time": time_lapsed, "moves": len(moves), "visited": visited_count}


def chunks(linear_list):
    return [linear_list[i: i + 3] for i in range(0, len(linear_list), 3)]


def run():
    with open("./input.txt", "r") as f:
        options = f.readlines()

    state = list(
        map(
            chunks,
            map(
                lambda x: list(map(int, x.split(","))),
                options,
            ),
        )
    )

    for current in state:
        yield {
            "input": current,
            "a_star_hamming": run_once(
                lambda: a_star_search(
                    current,
                    heuristic_hamming_distance,
                )
            ),
            "a_star_manhattan": run_once(
                lambda: a_star_search(
                    current,
                    heuristic_manhattan,
                )
            ),
            "greedy_hamming": run_once(
                lambda: greedy_search(
                    current,
                    heuristic_hamming_distance,
                )
            ),
            "greedy_manhattan": run_once(
                lambda: greedy_search(
                    current,
                    heuristic_manhattan,
                )
            ),
            "bfs": run_once(lambda: bfs_search(current)),
            "dfs": run_once(lambda: dfs_search(current)),
        }


with open("output.txt", "w") as f:
    for result in run():
        f.write(f"{json.dumps(result)}\n")
