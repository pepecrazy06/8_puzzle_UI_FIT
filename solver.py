import time
import heapq
import random
import math
from collections import deque
from models import Node
from itertools import permutations

class PuzzleSolver:
    def __init__(self):
        self.goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.actions = ["Left", "Right", "Up", "Down"]

    def _get_node_letter(self, index):
        """Tạo tên Node tự động dạng chuỗi chữ cái tăng dần: A, B... Z, AA, AB..."""
        res = ""
        while index >= 0:
            res = chr(65 + (index % 26)) + res
            index = index // 26 - 1
        return res

    def heuristic_manhattan(self, state):
        """h2(n): Tính tổng khoảng cách Manhattan của các ô từ 1 đến 8 tới vị trí đích"""
        total = 0
        for value in range(1, 9):
            current_index = state.index(value)
            goal_index = self.goal_state.index(value)
            current_row, current_col = current_index // 3, current_index % 3
            goal_row, goal_col = goal_index // 3, goal_index % 3
            total += abs(current_row - goal_row) + abs(current_col - goal_col)
        return total

    def heuristic_misplaced(self, state):
        """h1(n): Đếm số lượng ô sai vị trí (Không tính ô trống số 0)"""
        count = 0
        for i in range(9):
            if state[i] != 0 and state[i] != self.goal_state[i]:
                count += 1
        return count

    def _generate_solvable_state(self, base_state=None, steps=20):
        """Tạo ra một trạng thái ngẫu nhiên chắc chắn giải được để phục vụ Restart/Beam Search"""
        current = base_state if base_state else self.goal_state
        for _ in range(steps):
            neighbors = self.get_neighbors(current)
            if neighbors:
                current = random.choice(neighbors)[0]
        return current

    def _move_state(self, state, action):
        """Hàm dịch chuyển trạng thái thủ công theo từng Action (Phục vụ BFS Child/Expand)"""
        zero_index = state.index(0)
        row, col = zero_index // 3, zero_index % 3
        
        if action == "Up": new_row, new_col = row - 1, col
        elif action == "Down": new_row, new_col = row + 1, col
        elif action == "Left": new_row, new_col = row, col - 1
        elif action == "Right": new_row, new_col = row, col + 1
        else: return None
        
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = new_row * 3 + new_col
            new_state = list(state)
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            return tuple(new_state)
        return None

    def get_neighbors(self, state):
        """Sinh tất cả các trạng thái kế cận cùng lúc"""
        neighbors = []
        zero_index = state.index(0)
        row, col = zero_index // 3, zero_index % 3

        moves = {
            "Left": (0, -1, -1),
            "Right": (0, 1, 1),
            "Up": (-1, 0, -3),
            "Down": (1, 0, 3)
        }

        for action in self.actions:
            dr, dc, delta = moves[action]
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = list(state)
                new_index = zero_index + delta
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                neighbors.append((tuple(new_state), action))
        return neighbors

    def solve(self, start_state, goal_state, algorithm, k=2, t=100.0, alpha=0.95):
        self.goal_state = tuple(goal_state)
        start_tuple = tuple(start_state)

        if algorithm == "BFS_CHILD":
            return self._bfs_child_search(start_tuple)
        elif algorithm == "BFS_EXPAND":
            return self._bfs_expand_search(start_tuple)
        elif algorithm == "DFS":
            return self._dfs_search(start_tuple)
        elif algorithm == "IDS":
            return self._ids_search(start_tuple)
        elif algorithm in ["UCS", "GREEDY_MANHATTAN", "GREEDY_MISPLACED", "ASTAR_MANHATTAN", "ASTAR_MISPLACED"]:
            return self._informed_search(start_tuple, algorithm)
        elif algorithm == "LOCAL_BEAM":
            return self._hill_climbing_search(start_tuple, algorithm, k)
        elif algorithm in ["HILL_SIMPLE", "HILL_STEEPEST", "HILL_SIMPLE_MISPLACED", "HILL_STEEPEST_MISPLACED", "HILL_FIRST_CHOICE", "HILL_STOCHASTIC", "HILL_RESTART"]:
            return self._hill_climbing_search(start_tuple, algorithm)
        elif algorithm == "SIMULATED_ANNEALING":
            return self._simulated_annealing_search(start_tuple, t, alpha)
        else:
            return {"success": False, "history": [], "time": 0, "error": f"Unknown algorithm: {algorithm}"}

    # --- 1. BFS CÁCH 1: CHILD-NODE APPROACH ---
    def _bfs_child_search(self, start_state):
        start_time = time.time()
        node_counter = 0
        
        start_node = Node(start_state, node_id=self._get_node_letter(node_counter))
        node_counter += 1
        
        if start_node.state == self.goal_state:
            log = {"node": {"id": start_node.node_id, "state": start_node.state, "action": start_node.action},
                   "frontier_added": [{"is_goal": True, "state": start_node.state, "id": start_node.node_id, "parent_id": "", "action": "", "cost": start_node.cost}],
                   "reached": [start_state]}
            return {"success": True, "history": [log], "time": round((time.time() - start_time) * 1000, 2)}
            
        frontier = deque([start_node])
        reached_set = {start_state}
        reached_list = [start_state]
        history_logs = []
        bfs_actions = ["Up", "Down", "Left", "Right"]
        
        while frontier:
            current_node = frontier.popleft()
            frontier_snapshot = []
            
            for action in bfs_actions:
                child_tuple = self._move_state(current_node.state, action)
                if child_tuple is not None:
                    if child_tuple not in reached_set:
                        child_node = Node(child_tuple, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter))
                        node_counter += 1
                        
                        is_current_goal = (child_tuple == self.goal_state)
                        frontier_snapshot.append({
                            "state": child_tuple, "parent_id": current_node.node_id,
                            "action": action, "cost": child_node.cost, "id": child_node.node_id, "is_goal": is_current_goal
                        })
                        
                        reached_set.add(child_tuple)
                        reached_list.append(child_tuple)
                        frontier.append(child_node)
                        
                        if is_current_goal:
                            history_logs.append({
                                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                                "frontier_added": frontier_snapshot, "reached": list(reached_list)
                            })
                            return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}
            
            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": frontier_snapshot, "reached": list(reached_list)
            })
        return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 2. BFS CÁCH 2: EXPAND APPROACH ---
    def _bfs_expand_search(self, start_state):
        start_time = time.time()
        node_counter = 0
        start_node = Node(start_state, node_id=self._get_node_letter(node_counter))
        node_counter += 1
        
        if start_node.state == self.goal_state:
            log = {"node": {"id": start_node.node_id, "state": start_node.state, "action": start_node.action},
                   "frontier_added": [{"is_goal": True, "state": start_node.state, "id": start_node.node_id, "parent_id": "", "action": "", "cost": start_node.cost}],
                   "reached": [start_state]}
            return {"success": True, "history": [log], "time": round((time.time() - start_time) * 1000, 2)}
            
        frontier = deque([start_node])
        reached_set = {start_state}
        reached_list = [start_state]
        history_logs = []
        bfs_actions = ["Up", "Down", "Left", "Right"]
        
        while frontier:
            current_node = frontier.popleft()
            frontier_snapshot = []
            children_to_process = []
            
            for action in bfs_actions:
                child_tuple = self._move_state(current_node.state, action)
                if child_tuple is not None:
                    child_node = Node(child_tuple, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter))
                    node_counter += 1
                    children_to_process.append(child_node)
            
            goal_found_node = None
            for child in children_to_process:
                is_current_goal = (child.state == self.goal_state)
                if is_current_goal:
                    goal_found_node = child
                    frontier_snapshot.append({
                        "state": child.state, "parent_id": current_node.node_id,
                        "action": child.action, "cost": child.cost, "id": child.node_id, "is_goal": True
                    })
                    if child.state not in reached_set:
                        reached_set.add(child.state)
                        reached_list.append(child.state)
                    break
                
                if child.state not in reached_set:
                    reached_set.add(child.state)
                    reached_list.append(child.state)
                    frontier.append(child)
                    
                    frontier_snapshot.append({
                        "state": child.state, "parent_id": current_node.node_id,
                        "action": child.action, "cost": child.cost, "id": child.node_id, "is_goal": False
                    })
            
            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": frontier_snapshot, "reached": list(reached_list)
            })
            if goal_found_node:
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}
        return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 3. THUẬT TOÁN DFS ---
    def _dfs_search(self, start_state):
        start_time = time.time()
        node_counter = 0
        start_node = Node(start_state, node_id=self._get_node_letter(node_counter))
        node_counter += 1
        frontier = [start_node]
        reached_set = {start_state}
        reached_list = [start_state]
        history_logs = []

        while frontier:
            current_node = frontier.pop()
            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost}],
                    "reached": list(reached_list)
                })
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

            frontier_snapshot = []
            for next_state, action in self.get_neighbors(current_node.state):
                if next_state not in reached_set:
                    reached_set.add(next_state)
                    reached_list.append(next_state)
                    child_node = Node(next_state, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter))
                    node_counter += 1
                    frontier.append(child_node)
                    frontier_snapshot.append({
                        "state": next_state, "parent_id": current_node.node_id,
                        "action": action, "cost": child_node.cost, "id": child_node.node_id, "is_goal": (next_state == self.goal_state)
                    })
            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": frontier_snapshot, "reached": list(reached_list)
            })
        return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 4. THUẬT TOÁN IDS ---
    def _ids_search(self, start_state, max_depth=30):
        start_time = time.time()
        history_logs = []
        node_counter = 0
        for limit in range(max_depth + 1):
            reached_list = [start_state]
            path_states = {start_state}
            start_node = Node(start_state, node_id=self._get_node_letter(node_counter))
            node_counter += 1
            result, node_counter = self._dls_recursive(start_node, limit, path_states, reached_list, history_logs, node_counter)
            if result is not None:
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}
        return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    def _dls_recursive(self, node, limit, path_states, reached_list, history_logs, node_counter):
        if node.state == self.goal_state:
            history_logs.append({
                "node": {"id": node.node_id, "state": node.state, "action": node.action},
                "frontier_added": [{"is_goal": True, "state": node.state, "id": node.node_id, "parent_id": node.parent.node_id if node.parent else "", "action": "", "cost": node.cost}],
                "reached": list(reached_list)
            })
            return node, node_counter
        if node.depth == limit:
            history_logs.append({
                "node": {"id": node.node_id, "state": node.state, "action": node.action},
                "frontier_added": [], "reached": list(reached_list)
            })
            return None, node_counter

        children_snapshot = []
        children_nodes = []
        for next_state, action in self.get_neighbors(node.state):
            if next_state not in path_states:
                child = Node(next_state, node, action, node.cost + 1, node.depth + 1, self._get_node_letter(node_counter))
                node_counter += 1
                children_nodes.append(child)
                children_snapshot.append({
                    "state": next_state, "parent_id": node.node_id,
                    "action": action, "cost": child.cost, "id": child.node_id, "is_goal": (next_state == self.goal_state)
                })
        history_logs.append({
            "node": {"id": node.node_id, "state": node.state, "action": node.action},
            "frontier_added": children_snapshot, "reached": list(reached_list)
        })
        for child in children_nodes:
            path_states.add(child.state)
            reached_list.append(child.state)
            res, node_counter = self._dls_recursive(child, limit, path_states, reached_list, history_logs, node_counter)
            if res is not None: return res, node_counter
            path_states.remove(child.state)
        return None, node_counter

    # --- 5. HÀM CHUNG CHO UCS, GREEDY, A* ---
    def _informed_search(self, start_state, algorithm):
        start_time = time.time()
        node_counter = 0
        start_node = Node(start_state, node_id=self._get_node_letter(node_counter))
        node_counter += 1
        
        def get_priority(n):
            if algorithm == "UCS": return n.cost
            elif algorithm == "GREEDY_MANHATTAN": return self.heuristic_manhattan(n.state)
            elif algorithm == "GREEDY_MISPLACED": return self.heuristic_misplaced(n.state)
            elif algorithm == "ASTAR_MANHATTAN": return n.cost + self.heuristic_manhattan(n.state)
            elif algorithm == "ASTAR_MISPLACED": return n.cost + self.heuristic_misplaced(n.state)

        frontier = []
        order = 0
        heapq.heappush(frontier, (get_priority(start_node), order, start_node))
        reached_dict = {start_state: start_node.cost}
        reached_list = [start_state]
        history_logs = []

        while frontier:
            _, _, current_node = heapq.heappop(frontier)
            if current_node.cost > reached_dict.get(current_node.state, float('inf')): continue
            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost}],
                    "reached": list(reached_list)
                })
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

            frontier_snapshot = []
            for next_state, action in self.get_neighbors(current_node.state):
                new_cost = current_node.cost + 1
                if next_state not in reached_dict or new_cost < reached_dict[next_state]:
                    reached_dict[next_state] = new_cost
                    if next_state not in reached_list: reached_list.append(next_state)
                    child_node = Node(next_state, current_node, action, new_cost, current_node.depth + 1, self._get_node_letter(node_counter))
                    node_counter += 1
                    order += 1
                    heapq.heappush(frontier, (get_priority(child_node), order, child_node))
                    frontier_snapshot.append({
                        "state": next_state, "parent_id": current_node.node_id,
                        "action": action, "cost": child_node.cost, "id": child_node.node_id, "is_goal": (next_state == self.goal_state)
                    })
            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": frontier_snapshot, "reached": list(reached_list)
            })
        return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 6. HÀM CHUNG CHO CÁC THUẬT TOÁN LEO ĐỒI & LOCAL BEAM ---
    def _hill_climbing_search(self, start_state, algorithm, k=2):
        start_time = time.time()
        node_counter = 0
        reached_list = [start_state]
        history_logs = []

        # --- BIẾN THỂ LOCAL BEAM SEARCH (k=2) ---
        if algorithm == "LOCAL_BEAM":
            current_states = [start_state]
            for _ in range(k - 1):
                current_states.append(self._generate_solvable_state(base_state=start_state, steps=3))

            current_nodes = []
            for s in current_states:
                current_nodes.append(Node(s, node_id=self._get_node_letter(node_counter)))
                node_counter += 1
                if s not in reached_list: reached_list.append(s)

            step_count = 0
            while step_count < 100:
                best_current_node = min(current_nodes, key=lambda n: self.heuristic_manhattan(n.state))
                
                if best_current_node.state == self.goal_state:
                    history_logs.append({
                        "node": {"id": best_current_node.node_id, "state": best_current_node.state, "action": "Goal Found"},
                        "frontier_added": [], "reached": list(reached_list)
                    })
                    return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

                all_successors = []
                frontier_snapshot = []

                for c_node in current_nodes:
                    neighbors = self.get_neighbors(c_node.state)
                    for next_state, action in neighbors:
                        child_node = Node(next_state, c_node, action, c_node.cost + 1, c_node.depth + 1, self._get_node_letter(node_counter))
                        node_counter += 1
                        if next_state not in reached_list: reached_list.append(next_state)
                        all_successors.append(child_node)
                        
                        frontier_snapshot.append({
                            "state": next_state, "parent_id": c_node.node_id,
                            "action": f"{c_node.node_id}➔{action}", "cost": child_node.cost, "id": child_node.node_id, "is_goal": (next_state == self.goal_state)
                        })

                history_logs.append({
                    "node": {"id": best_current_node.node_id, "state": best_current_node.state, "action": f"Beam Step {step_count+1}"},
                    "frontier_added": frontier_snapshot, "reached": list(reached_list)
                })

                goal_nodes = [n for n in all_successors if n.state == self.goal_state]
                if goal_nodes:
                    current_nodes = [goal_nodes[0]]
                    continue

                all_successors.sort(key=lambda n: self.heuristic_manhattan(n.state))
                unique_nodes = all_successors[:k]
                seen_states = set()
                for n in all_successors:
                    if n.state not in seen_states:
                        seen_states.add(n.state)
                        unique_nodes.append(n)
                    if len(unique_nodes) == k: break

                if not unique_nodes or self.heuristic_manhattan(unique_nodes[0].state) >= self.heuristic_manhattan(best_current_node.state):
                    return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

                current_nodes = unique_nodes
                step_count += 1
            return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

        # --- CÁC BIẾN THỂ LEO ĐỒI CÒN LẠI ---
        current_node = Node(start_state, node_id=self._get_node_letter(node_counter))
        node_counter += 1
        restart_count = 0
        max_restarts = 5

        while True:
            is_misplaced = "MISPLACED" in algorithm
            current_h = self.heuristic_misplaced(current_node.state) if is_misplaced else self.heuristic_manhattan(current_node.state)
            
            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost}],
                    "reached": list(reached_list)
                })
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

            neighbors = self.get_neighbors(current_node.state)
            frontier_snapshot = []
            next_chosen_node = None
            best_h = current_h
            better_neighbors = []

            for next_state, action in neighbors:
                child_h = self.heuristic_misplaced(next_state) if is_misplaced else self.heuristic_manhattan(next_state)
                child_node = Node(next_state, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter))
                node_counter += 1
                
                if next_state not in reached_list: reached_list.append(next_state)
                    
                frontier_snapshot.append({
                    "state": next_state, "parent_id": current_node.node_id,
                    "action": action, "cost": child_node.cost, "id": child_node.node_id, "is_goal": (next_state == self.goal_state)
                })
                
                if child_h < current_h: better_neighbors.append(child_node)

                if "HILL_SIMPLE" in algorithm or algorithm == "HILL_FIRST_CHOICE":
                    if child_h < current_h and next_chosen_node is None:
                        next_chosen_node = child_node
                        if algorithm == "HILL_FIRST_CHOICE": break

                elif "HILL_STEEPEST" in algorithm or algorithm == "HILL_RESTART":
                    if child_h < best_h:
                        best_h = child_h
                        next_chosen_node = child_node

            if algorithm == "HILL_STOCHASTIC" and better_neighbors:
                next_chosen_node = random.choice(better_neighbors)

            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": frontier_snapshot, "reached": list(reached_list)
            })

            if next_chosen_node is None:
                if algorithm == "HILL_RESTART" and restart_count < max_restarts:
                    restart_count += 1
                    random_state = self._generate_solvable_state(steps=25)
                    current_node = Node(random_state, node_id=f"RST-{restart_count}")
                    if random_state not in reached_list: reached_list.append(random_state)
                    continue
                else:
                    return {"success": False, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}
                
            current_node = next_chosen_node
            
    def _simulated_annealing_search(self, start_state, T=100.0, alpha=0.95):
        start_time = time.time()
        current_node = Node(start_state, node_id="Start")
        current_h = self.heuristic_manhattan(current_node.state)
        
        history_logs = []
        T_min = 0.01 
        step_count = 0
        
        # Lưu trạng thái ban đầu
        history_logs.append({
            "node": {"id": "Start", "state": current_node.state, "action": "Start"},
            "temp": round(T, 2),
            "delta": 0,
            "accepted": True,
            "frontier_added": [], 
            "reached": [list(current_node.state)]
        })
        
        while T > T_min:
            if current_node.state == self.goal_state:
                break
                
            # Lấy các trạng thái láng giềng (đúng tên hàm là get_neighbors)
            neighbors = self.get_neighbors(current_node.state)
            if not neighbors: break
                
            # Chọn ngẫu nhiên 1 láng giềng
            next_state, action = random.choice(neighbors)
            next_h = self.heuristic_manhattan(next_state)
            
            delta = next_h - current_h
            accepted = (delta < 0) or (random.random() < math.exp(-delta / T))
            
            if accepted:
                current_node = Node(next_state, current_node, action, 
                                   cost=current_node.cost + 1, node_id=self._get_node_letter(step_count))
                current_h = next_h
            
            T *= alpha
            step_count += 1
            
            history_logs.append({
                "node": {"id": str(step_count), "state": current_node.state, "action": action},
                "temp": round(T, 2),
                "delta": round(delta, 2),
                "accepted": accepted,
                "frontier_added": [],
                "reached": [list(current_node.state)]
            })
            
        return {"success": (current_node.state == self.goal_state), "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 8. BELIEF STATE SEARCH - COMMON ACTION ---
    def _normalize_belief_pattern(self, pattern):
        """Chuyển pattern về list 9 phần tử. Ô chưa biết dùng "?"."""
        if pattern is None:
            return ["?"] * 9

        result = []
        for v in pattern:
            if v in [None, "", "?", "x", "X", -1, "-1"]:
                result.append("?")
            else:
                result.append(int(v))
        return result

    def _generate_candidates_from_belief(self, pattern, limit=2):
        """
        Từ một pattern có dấu ?, sinh ra các trạng thái cụ thể.
        Ví dụ: [1,2,3,4,5,6,7,"?",8]
        -> có thể sinh [1,2,3,4,5,6,7,0,8].
        """
        pattern = self._normalize_belief_pattern(pattern)

        if len(pattern) != 9:
            return []

        fixed_values = [v for v in pattern if v != "?"]

        for v in fixed_values:
            if v < 0 or v > 8:
                return []

        if len(set(fixed_values)) != len(fixed_values):
            return []

        missing_values = [v for v in range(9) if v not in fixed_values]
        unknown_indexes = [i for i, v in enumerate(pattern) if v == "?"]

        candidates = []
        seen = set()

        # Nếu quá nhiều ô chưa biết thì sinh ngẫu nhiên để không bị nổ hoán vị.
        if len(unknown_indexes) >= 7:
            attempts = 0
            while len(candidates) < limit and attempts < limit * 200:
                attempts += 1
                values = missing_values[:]
                random.shuffle(values)
                new_state = pattern[:]

                for idx, value in zip(unknown_indexes, values):
                    new_state[idx] = value

                state_tuple = tuple(new_state)
                if state_tuple not in seen:
                    seen.add(state_tuple)
                    candidates.append(state_tuple)
            return candidates

        # Nếu ít ô chưa biết thì sinh theo thứ tự hoán vị.
        for perm in permutations(missing_values):
            new_state = pattern[:]
            for idx, value in zip(unknown_indexes, perm):
                new_state[idx] = value

            state_tuple = tuple(new_state)
            if state_tuple not in seen:
                seen.add(state_tuple)
                candidates.append(state_tuple)

            if len(candidates) >= limit:
                break

        return candidates

    def _move_state_belief(self, state, action):
        """
        Trong Belief State, một action chung được áp dụng cho mọi trạng thái.
        Nếu action không hợp lệ với một trạng thái thì trạng thái đó đứng yên.
        """
        moved = self._move_state(tuple(state), action)
        if moved is None:
            return tuple(state)
        return moved

    def _belief_goal_test(self, belief_states):
        """Goal test: tất cả trạng thái trong belief node đều phải bằng goal_state."""
        for state in belief_states:
            if tuple(state) != self.goal_state:
                return False
        return True

    def _belief_state_key(self, belief_states):
        """Chuẩn hóa belief node để kiểm tra visited, không phụ thuộc thứ tự state."""
        return tuple(sorted(tuple(s) for s in belief_states))

    def solve_belief_common_action(self, start_belief, goal_state, max_depth=30):
        """
        Belief State Search đúng nghĩa:
        - 1 node gồm nhiều trạng thái có thể xảy ra.
        - Mỗi bước chọn 1 action chung.
        - Action đó áp dụng lên toàn bộ trạng thái trong node.
        - Thành công khi tất cả trạng thái trong node cùng về goal.
        """
        start_time = time.time()
        self.goal_state = tuple(goal_state)

        start_belief = [tuple(s) for s in start_belief]
        start_key = self._belief_state_key(start_belief)

        node_counter = 0
        start_node = {
            "id": self._get_node_letter(node_counter),
            "belief_states": start_belief,
            "parent_id": "",
            "action": "Start",
            "cost": 0
        }
        node_counter += 1

        frontier = deque([start_node])
        reached_set = {start_key}
        reached_list = [[list(s) for s in start_belief]]
        history_logs = []
        actions = ["Up", "Down", "Left", "Right"]

        while frontier:
            current_node = frontier.popleft()
            current_belief = current_node["belief_states"]

            if self._belief_goal_test(current_belief):
                history_logs.append({
                    "belief": True,
                    "belief_mode": "COMMON_ACTION",
                    "node": {
                        "id": current_node["id"],
                        "states": [list(s) for s in current_belief],
                        "action": current_node["action"],
                        "cost": current_node["cost"]
                    },
                    "frontier_added": [],
                    "reached": reached_list,
                    "status": "GOAL"
                })
                return {
                    "success": True,
                    "belief": True,
                    "belief_mode": "COMMON_ACTION",
                    "history": history_logs,
                    "time": round((time.time() - start_time) * 1000, 2)
                }

            if current_node["cost"] >= max_depth:
                continue

            frontier_snapshot = []

            for action in actions:
                next_belief = []

                for state in current_belief:
                    next_state = self._move_state_belief(state, action)
                    next_belief.append(next_state)

                # Bỏ trùng trạng thái trong cùng belief node.
                unique_next_belief = []
                seen_states = set()
                for s in next_belief:
                    if s not in seen_states:
                        seen_states.add(s)
                        unique_next_belief.append(s)

                next_key = self._belief_state_key(unique_next_belief)

                if next_key not in reached_set:
                    child_node = {
                        "id": self._get_node_letter(node_counter),
                        "belief_states": unique_next_belief,
                        "parent_id": current_node["id"],
                        "action": action,
                        "cost": current_node["cost"] + 1
                    }
                    node_counter += 1

                    reached_set.add(next_key)
                    reached_list.append([[x for x in s] for s in unique_next_belief])
                    frontier.append(child_node)

                    frontier_snapshot.append({
                        "id": child_node["id"],
                        "parent_id": current_node["id"],
                        "action": action,
                        "cost": child_node["cost"],
                        "states": [list(s) for s in unique_next_belief],
                        "is_goal": self._belief_goal_test(unique_next_belief)
                    })

            history_logs.append({
                "belief": True,
                "belief_mode": "COMMON_ACTION",
                "node": {
                    "id": current_node["id"],
                    "states": [list(s) for s in current_belief],
                    "action": current_node["action"],
                    "cost": current_node["cost"]
                },
                "frontier_added": frontier_snapshot,
                "reached": reached_list
            })

        return {
            "success": False,
            "belief": True,
            "belief_mode": "COMMON_ACTION",
            "history": history_logs,
            "time": round((time.time() - start_time) * 1000, 2)
        }

    def solve_belief_common_from_pattern(self, start_pattern, goal_pattern, max_belief_states=2, max_depth=30):
        """
        Nhận Start Belief / Goal Belief dạng pattern có dấu ? từ giao diện,
        tự sinh ra nhiều trạng thái start, rồi chạy BFS trên belief node.
        """
        start_candidates = self._generate_candidates_from_belief(start_pattern, limit=max_belief_states)
        goal_candidates = self._generate_candidates_from_belief(goal_pattern, limit=1)

        if not start_candidates:
            return {
                "success": False,
                "belief": True,
                "belief_mode": "COMMON_ACTION",
                "history": [],
                "error": "Không sinh được Start Belief từ pattern."
            }

        if not goal_candidates:
            return {
                "success": False,
                "belief": True,
                "belief_mode": "COMMON_ACTION",
                "history": [],
                "error": "Không sinh được Goal từ pattern."
            }

        goal_state = goal_candidates[0]
        result = self.solve_belief_common_action(
            start_belief=start_candidates,
            goal_state=goal_state,
            max_depth=max_depth
        )

        result["start_belief"] = [list(s) for s in start_candidates]
        result["goal"] = list(goal_state)
        result["start_candidates"] = len(start_candidates)
        result["goal_candidates"] = len(goal_candidates)
        result["tried_pairs"] = 1
        return result

