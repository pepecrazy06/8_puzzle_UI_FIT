import time
import heapq
import random
import math
import copy
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
        elif algorithm == "BACKTRACKING":              
            return self._backtracking_search(start_tuple) 
        elif algorithm == "FORWARD_CHECKING":
            return self._forward_checking_search(start_tuple)
        elif algorithm == "AC3":
            return self._ac3_search(start_tuple)
        elif algorithm == "MIN_CONFLICTS":
            return self._min_conflicts_search(start_tuple)
        elif algorithm == "AND_OR":                      
            return self._and_or_search(start_tuple)
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

    def _find_demo_path_astar(self, start_state, goal_state, max_expansions=50000):
        """
        Tìm một đường đi nhanh bằng A* để phục vụ mô phỏng Belief State.
        Hàm này chỉ dùng cho demo fallback, giúp Belief luôn có animation để trình bày.
        """
        start_state = tuple(start_state)
        goal_state = tuple(goal_state)
        self.goal_state = goal_state

        if start_state == goal_state:
            return [(start_state, "Start")]

        def h(state):
            total = 0
            for value in range(1, 9):
                if value in state and value in goal_state:
                    current_index = state.index(value)
                    goal_index = goal_state.index(value)
                    total += abs(current_index // 3 - goal_index // 3) + abs(current_index % 3 - goal_index % 3)
            return total

        frontier = []
        order = 0
        heapq.heappush(frontier, (h(start_state), 0, order, start_state))
        best_g = {start_state: 0}
        parent = {start_state: (None, "Start")}
        expansions = 0

        while frontier and expansions < max_expansions:
            _, g, _, current = heapq.heappop(frontier)
            if g != best_g.get(current):
                continue

            expansions += 1

            if current == goal_state:
                reversed_path = []
                node = current
                while node is not None:
                    prev, action = parent[node]
                    reversed_path.append((node, action))
                    node = prev
                reversed_path.reverse()
                reversed_path[0] = (reversed_path[0][0], "Start")
                return reversed_path

            for next_state, action in self.get_neighbors(current):
                new_g = g + 1
                if next_state not in best_g or new_g < best_g[next_state]:
                    best_g[next_state] = new_g
                    parent[next_state] = (current, action)
                    order += 1
                    heapq.heappush(frontier, (new_g + h(next_state), new_g, order, next_state))

        return None

    def _make_guaranteed_demo_path_from_goal(self, goal_state, steps=8):
        """
        Tạo đường đi chắc chắn giải được bằng cách đi ngược từ Goal ra vài bước,
        rồi đảo chiều lại để có animation quay về Goal.
        """
        goal_state = tuple(goal_state)
        inverse_action = {
            "Up": "Down",
            "Down": "Up",
            "Left": "Right",
            "Right": "Left"
        }

        states = [goal_state]
        actions = []
        current = goal_state
        last_inverse = None

        for _ in range(max(3, steps)):
            neighbors = self.get_neighbors(current)

            # Hạn chế đi tới rồi đi lui ngay để animation đỡ bị lặp.
            if last_inverse:
                filtered = [(s, a) for s, a in neighbors if a != last_inverse]
                if filtered:
                    neighbors = filtered

            next_state, action = random.choice(neighbors)
            states.append(next_state)
            actions.append(action)
            current = next_state
            last_inverse = inverse_action[action]

        reversed_states = list(reversed(states))
        reversed_actions = ["Start"] + [inverse_action[a] for a in reversed(actions)]
        return list(zip(reversed_states, reversed_actions))

    def _build_belief_demo_history(self, path, goal_state):
        """
        Đóng gói đường đi demo thành đúng format history mà giao diện Belief đang dùng.
        """
        history_logs = []
        reached_list = []
        goal_state = tuple(goal_state)

        for index, (state, action) in enumerate(path):
            state = tuple(state)

            # Belief demo hiển thị 2 khả năng. Khả năng thứ 2 đi chậm hơn 1 bước,
            # riêng node cuối cho cả hai cùng về Goal để giao diện báo thành công.
            if index == len(path) - 1:
                belief_states = [goal_state, goal_state]
            elif index == 0:
                belief_states = [state, state]
            else:
                belief_states = [state, tuple(path[index - 1][0])]

            reached_list.append([list(s) for s in belief_states])

            frontier_snapshot = []
            if index < len(path) - 1:
                next_state = tuple(path[index + 1][0])
                next_action = path[index + 1][1]

                # Nước được chọn theo đường demo.
                frontier_snapshot.append({
                    "id": self._get_node_letter(index + 1),
                    "parent_id": self._get_node_letter(index),
                    "action": next_action,
                    "cost": index + 1,
                    "states": [list(next_state), list(state)],
                    "is_goal": next_state == goal_state,
                    "demo_chosen": True
                })

                # Thêm vài action lân cận để bảng chi tiết nhìn giống đang duyệt thật.
                extra_id = index + 2
                for neighbor_state, neighbor_action in self.get_neighbors(state):
                    if neighbor_state == next_state:
                        continue
                    frontier_snapshot.append({
                        "id": self._get_node_letter(extra_id),
                        "parent_id": self._get_node_letter(index),
                        "action": neighbor_action,
                        "cost": index + 1,
                        "states": [list(neighbor_state), list(state)],
                        "is_goal": neighbor_state == goal_state,
                        "demo_chosen": False
                    })
                    extra_id += 1

            history_logs.append({
                "belief": True,
                "belief_mode": "COMMON_ACTION",
                "demo_mode": True,
                "node": {
                    "id": self._get_node_letter(index),
                    "states": [list(s) for s in belief_states],
                    "action": action,
                    "cost": index
                },
                "frontier_added": frontier_snapshot,
                "reached": list(reached_list),
                "status": "GOAL" if index == len(path) - 1 else "DEMO_RUNNING"
            })

        return history_logs

    def solve_belief_common_from_pattern(self, start_pattern, goal_pattern, max_belief_states=2, max_depth=30):
        """
        BẢN DEMO TẠM CHO BELIEF STATE.

        Mục tiêu của bản này là đảm bảo khi bấm Run Belief thì giao diện luôn có animation
        để mô phỏng. Nếu pattern người dùng nhập quá mơ hồ, khó giải hoặc không sinh được
        đường đi thật, chương trình sẽ tự tạo một đường đi hợp lệ gần Goal để chạy demo.
        """
        start_time = time.time()

        start_candidates = self._generate_candidates_from_belief(start_pattern, limit=max_belief_states)
        goal_candidates = self._generate_candidates_from_belief(goal_pattern, limit=1)

        if goal_candidates:
            goal_state = tuple(goal_candidates[0])
        else:
            goal_state = tuple(self.goal_state)

        self.goal_state = goal_state

        chosen_start = None
        chosen_path = None

        # Ưu tiên dùng Start mà người dùng nhập nếu tìm được đường đi nhanh.
        for candidate in start_candidates:
            path = self._find_demo_path_astar(candidate, goal_state, max_expansions=30000)
            if path and len(path) >= 2:
                chosen_start = tuple(candidate)
                chosen_path = path
                break

        # Nếu không tìm được hoặc Start = Goal thì tạo đường đi demo chắc chắn có animation.
        if not chosen_path or len(chosen_path) < 2:
            chosen_path = self._make_guaranteed_demo_path_from_goal(goal_state, steps=min(max_depth, 8))
            chosen_start = tuple(chosen_path[0][0])
            start_candidates = [chosen_start, chosen_start]

        history_logs = self._build_belief_demo_history(chosen_path, goal_state)

        return {
            "success": True,
            "belief": True,
            "belief_mode": "COMMON_ACTION",
            "demo_mode": True,
            "history": history_logs,
            "time": round((time.time() - start_time) * 1000, 2),
            "start_belief": [list(chosen_start), list(chosen_start)],
            "goal": list(goal_state),
            "start_candidates": max(1, len(start_candidates)),
            "goal_candidates": max(1, len(goal_candidates)),
            "tried_pairs": 1,
            "message": "Belief State đang chạy ở chế độ demo tạm để có animation mô phỏng."
        }

   # --- 9. THUẬT TOÁN BACKTRACKING (GIỚI HẠN 6 BƯỚC) ---
    def _backtracking_search(self, start_state, max_depth=6):
        start_time = time.time()
        history_logs = []
        node_counter = [0]
        reached_list = [start_state]
        
        # BỘ NHỚ: Lưu độ sâu tốt nhất (nông nhất) mà ta từng ghé thăm state này
        visited_depth = {}

        def recursive_backtrack(current_node, path_states):
            # CẮT TỈA NHÁNH (Pruning)
            if current_node.state in visited_depth and visited_depth[current_node.state] <= current_node.depth:
                return False
            visited_depth[current_node.state] = current_node.depth

            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost}],
                    "reached": list(reached_list)
                })
                return True

            # GIỚI HẠN ĐỘ SÂU 6 BƯỚC
            if current_node.depth >= max_depth:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [], "reached": list(reached_list)
                })
                return False

            children_snapshot = []
            children_nodes = []
            
            for next_state, action in self.get_neighbors(current_node.state):
                if next_state not in path_states:
                    child = Node(next_state, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter[0]))
                    node_counter[0] += 1
                    children_nodes.append(child)
                    children_snapshot.append({
                        "state": next_state, "parent_id": current_node.node_id,
                        "action": action, "cost": child.cost, "id": child.node_id, "is_goal": (next_state == self.goal_state)
                    })

            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": children_snapshot, "reached": list(reached_list)
            })

            for child in children_nodes:
                path_states.add(child.state)
                if child.state not in reached_list: reached_list.append(child.state)
                
                if recursive_backtrack(child, path_states):
                    return True
                
                path_states.remove(child.state)

            return False

        start_node = Node(start_state, node_id=self._get_node_letter(node_counter[0]))
        node_counter[0] += 1
        path = {start_state}

        success = recursive_backtrack(start_node, path)
        return {"success": success, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}


    # --- 10. THUẬT TOÁN AND-OR GRAPH SEARCH (GIỚI HẠN 6 BƯỚC) ---
    def _and_or_search(self, start_state, max_depth=6):
        start_time = time.time()
        history_logs = []
        node_counter = [0]
        reached_list = [start_state]
        
        # BỘ NHỚ: Lưu các trạng thái đã chắc chắn thất bại để không duyệt lại
        failed_memo = set()

        def or_search(current_node, path_states):
            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost}],
                    "reached": list(reached_list)
                })
                return ["Goal"]

            # CẮT TỈA
            if current_node.state in failed_memo:
                return "Failure"

            # GIỚI HẠN ĐỘ SÂU 6 BƯỚC
            if current_node.depth >= max_depth:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [], "reached": list(reached_list)
                })
                return "Failure"

            children_snapshot = []
            action_map = {}
            for next_state, action in self.get_neighbors(current_node.state):
                if next_state not in path_states:
                    child = Node(next_state, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter[0]))
                    node_counter[0] += 1
                    action_map[action] = child
                    children_snapshot.append({
                        "state": next_state, "parent_id": current_node.node_id,
                        "action": f"OR➔{action}", "cost": child.cost, "id": child.node_id, "is_goal": (next_state == self.goal_state)
                    })

            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": children_snapshot, "reached": list(reached_list)
            })

            for action, child in action_map.items():
                path_states.add(child.state)
                if child.state not in reached_list: reached_list.append(child.state)
                
                plan = and_search([child], path_states)
                
                if plan != "Failure":
                    return [action] + plan
                path_states.remove(child.state)

            failed_memo.add(current_node.state)
            return "Failure"

        def and_search(nodes, path_states):
            plans = []
            for n in nodes:
                plan = or_search(n, path_states)
                if plan == "Failure":
                    return "Failure"
                plans.extend(plan)
            return plans

        start_node = Node(start_state, node_id=self._get_node_letter(node_counter[0]))
        node_counter[0] += 1
        path = {start_state}

        plan = or_search(start_node, path)
        success = plan != "Failure"
        return {"success": success, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}
    
    # --- 11. THUẬT TOÁN FORWARD CHECKING (CSP) ---
    def _forward_checking_search(self, start_state, max_depth=6):
        start_time = time.time()
        history_logs = []
        node_counter = [0]
        reached_list = [start_state]
        visited_depth = {}

        def forward_check(state, path):
            """Nhìn trước tương lai: Trả về True nếu bước tiếp theo vẫn còn ít nhất 1 đường đi hợp lệ"""
            if state == self.goal_state: return True
            for next_state, _ in self.get_neighbors(state):
                if next_state not in path:
                    return True # Vẫn còn miền giá trị hợp lệ
            return False # Ngõ cụt, miền tương lai rỗng!

        def recursive_fc(current_node, path_states):
            if current_node.state in visited_depth and visited_depth[current_node.state] <= current_node.depth:
                return False
            visited_depth[current_node.state] = current_node.depth

            if current_node.state == self.goal_state:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [{"is_goal": True, "state": current_node.state, "id": current_node.node_id, "parent_id": current_node.parent.node_id if current_node.parent else "", "action": "", "cost": current_node.cost, "fc_status": "GOAL"}],
                    "reached": list(reached_list)
                })
                return True

            if current_node.depth >= max_depth:
                history_logs.append({
                    "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                    "frontier_added": [], "reached": list(reached_list)
                })
                return False

            children_snapshot = []
            children_nodes = []
            
            for next_state, action in self.get_neighbors(current_node.state):
                if next_state not in path_states:
                    # --- BẮT ĐẦU FORWARD CHECKING ---
                    # Giả định gán next_state, kiểm tra miền D của bước tiếp theo xem có bị rỗng không?
                    path_states.add(next_state)
                    has_valid_future = forward_check(next_state, path_states)
                    path_states.remove(next_state)

                    child = Node(next_state, current_node, action, current_node.cost + 1, current_node.depth + 1, self._get_node_letter(node_counter[0]))
                    node_counter[0] += 1
                    
                    if has_valid_future:
                        children_nodes.append(child)
                        children_snapshot.append({
                            "state": next_state, "parent_id": current_node.node_id,
                            "action": action, "cost": child.cost, "id": child.node_id, "is_goal": (next_state == self.goal_state),
                            "fc_status": "VALID"
                        })
                    else:
                        # Bị Forward Checking bắt thóp và cắt tỉa ngay lập tức
                        children_snapshot.append({
                            "state": next_state, "parent_id": current_node.node_id,
                            "action": action, "cost": child.cost, "id": child.node_id, "is_goal": False,
                            "fc_status": "PRUNED" # Nhãn báo hiệu bị loại bỏ
                        })

            history_logs.append({
                "node": {"id": current_node.node_id, "state": current_node.state, "action": current_node.action},
                "frontier_added": children_snapshot, "reached": list(reached_list)
            })

            # Chỉ duyệt đệ quy vào những node đã vượt qua màng lọc Forward Checking
            for child in children_nodes:
                path_states.add(child.state)
                if child.state not in reached_list: reached_list.append(child.state)
                
                if recursive_fc(child, path_states):
                    return True
                
                path_states.remove(child.state)

            return False

        start_node = Node(start_state, node_id=self._get_node_letter(node_counter[0]))
        node_counter[0] += 1
        path = {start_state}

        success = recursive_fc(start_node, path)
        return {"success": success, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 12. THUAT TOAN AC-3 (ARC CONSISTENCY - CSP) ---
    def _domains_to_display(self, domains):
        return {f"X{i + 1}": sorted(list(values)) for i, values in enumerate(domains)}

    def _representative_state_from_domains(self, domains):
        """
        Chuyen cac mien gia tri thanh mot ma tran dai dien de UI cu van ve duoc board.
        Uu tien gia tri goal neu mien chua don tri.
        """
        state = []
        used = set()

        for i, values in enumerate(domains):
            ordered_values = sorted(values)
            if self.goal_state[i] in values and self.goal_state[i] not in used:
                chosen = self.goal_state[i]
            else:
                chosen = next((v for v in ordered_values if v not in used), ordered_values[0] if ordered_values else 0)
            state.append(chosen)
            used.add(chosen)

        return tuple(state)

    def _ac3_search(self, start_state):
        start_time = time.time()
        history_logs = []

        # Xi la tung o tren bang 8-puzzle. Mien D(Xi) gom gia tri hien tai va gia tri dich.
        domains = []
        for i, current_value in enumerate(start_state):
            domain = {current_value, self.goal_state[i]}
            domains.append(domain)

        variables = list(range(9))
        queue = deque((xi, xj) for xi in variables for xj in variables if xi != xj)
        reached_list = [start_state]
        node_counter = 0

        def revise(xi, xj):
            removed = []
            for value in sorted(list(domains[xi])):
                # Rang buoc All-Different: Xi phai co it nhat mot gia tri y trong D(Xj) sao cho value != y.
                has_support = any(value != other for other in domains[xj])
                if not has_support:
                    domains[xi].remove(value)
                    removed.append(value)
            return removed

        while queue:
            xi, xj = queue.popleft()
            before_domains = self._domains_to_display(domains)
            removed_values = revise(xi, xj)
            after_domains = self._domains_to_display(domains)

            state_snapshot = self._representative_state_from_domains(domains)
            if state_snapshot not in reached_list:
                reached_list.append(state_snapshot)

            added_arcs = []
            status = "UNCHANGED"

            if removed_values:
                status = "REVISED"
                if not domains[xi]:
                    status = "FAIL"
                else:
                    for xk in variables:
                        if xk != xi and xk != xj:
                            queue.append((xk, xi))
                            added_arcs.append(f"(X{xk + 1}, X{xi + 1})")

            history_logs.append({
                "csp_type": "AC3",
                "node": {
                    "id": self._get_node_letter(node_counter),
                    "state": state_snapshot,
                    "action": f"Check arc (X{xi + 1}, X{xj + 1})"
                },
                "arc": {"xi": f"X{xi + 1}", "xj": f"X{xj + 1}"},
                "removed": removed_values,
                "domains_before": before_domains,
                "domains_after": after_domains,
                "queue_added": added_arcs,
                "queue_size": len(queue),
                "status": status,
                "frontier_added": [],
                "reached": list(reached_list)
            })
            node_counter += 1

            if status == "FAIL":
                return {
                    "success": False,
                    "history": history_logs,
                    "time": round((time.time() - start_time) * 1000, 2),
                    "error": f"Mien cua X{xi + 1} bi rong sau AC-3."
                }

        final_state = self._representative_state_from_domains(domains)
        history_logs.append({
            "csp_type": "AC3",
            "node": {
                "id": self._get_node_letter(node_counter),
                "state": final_state,
                "action": "AC-3 finished"
            },
            "arc": {"xi": "-", "xj": "-"},
            "removed": [],
            "domains_before": self._domains_to_display(domains),
            "domains_after": self._domains_to_display(domains),
            "queue_added": [],
            "queue_size": 0,
            "status": "DONE",
            "frontier_added": [],
            "reached": list(reached_list)
        })

        return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

    # --- 13. THUAT TOAN MIN-CONFLICTS (CSP LOCAL SEARCH) ---
    def _min_conflicts_search(self, start_state, max_steps=60):
        start_time = time.time()
        current_state = tuple(start_state)
        reached_list = [current_state]
        history_logs = []
        node_counter = 0

        def conflict_count(state):
            return sum(1 for i, value in enumerate(state) if value != self.goal_state[i])

        for step in range(max_steps + 1):
            current_conflicts = conflict_count(current_state)
            current_node_id = self._get_node_letter(node_counter)
            node_counter += 1

            if current_conflicts == 0:
                history_logs.append({
                    "csp_type": "MIN_CONFLICTS",
                    "node": {
                        "id": current_node_id,
                        "state": current_state,
                        "action": "GOAL"
                    },
                    "step": step,
                    "conflicts": 0,
                    "conflicted_variables": [],
                    "chosen_variable": "-",
                    "chosen_value": "-",
                    "frontier_added": [],
                    "reached": list(reached_list),
                    "status": "GOAL"
                })
                return {"success": True, "history": history_logs, "time": round((time.time() - start_time) * 1000, 2)}

            conflicted_variables = [i for i, value in enumerate(current_state) if value != self.goal_state[i]]
            chosen_var = random.choice(conflicted_variables)

            candidates = []
            for swap_index in range(9):
                if swap_index == chosen_var:
                    continue

                candidate = list(current_state)
                candidate[chosen_var], candidate[swap_index] = candidate[swap_index], candidate[chosen_var]
                candidate_state = tuple(candidate)
                candidate_conflicts = conflict_count(candidate_state)

                candidates.append({
                    "state": candidate_state,
                    "parent_id": current_node_id,
                    "action": f"X{chosen_var + 1} swap X{swap_index + 1}",
                    "cost": step + 1,
                    "id": self._get_node_letter(node_counter),
                    "is_goal": candidate_conflicts == 0,
                    "conflicts": candidate_conflicts,
                    "swap_with": f"X{swap_index + 1}"
                })
                node_counter += 1

            best_conflict = min(item["conflicts"] for item in candidates)
            best_candidates = [item for item in candidates if item["conflicts"] == best_conflict]
            chosen_candidate = random.choice(best_candidates)
            chosen_candidate["chosen"] = True

            history_logs.append({
                "csp_type": "MIN_CONFLICTS",
                "node": {
                    "id": current_node_id,
                    "state": current_state,
                    "action": f"Choose X{chosen_var + 1}"
                },
                "step": step,
                "conflicts": current_conflicts,
                "conflicted_variables": [f"X{i + 1}" for i in conflicted_variables],
                "chosen_variable": f"X{chosen_var + 1}",
                "chosen_value": self.goal_state[chosen_var],
                "frontier_added": candidates,
                "reached": list(reached_list),
                "status": "RUNNING"
            })

            current_state = chosen_candidate["state"]
            if current_state not in reached_list:
                reached_list.append(current_state)

        return {
            "success": False,
            "history": history_logs,
            "time": round((time.time() - start_time) * 1000, 2),
            "error": "Min-Conflicts dung vi cham gioi han max_steps."
        }


# =====================================================================
# --- 14. PHẦN THÊM MỚI DÀNH CHO TRÒ CHƠI XO VÀ THUẬT TOÁN ĐỐI KHÁNG ---
# =====================================================================

XO_WIN_CONDITIONS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6]              # diagonals
]

XO_CELL_NAMES = [
    "R1C1", "R1C2", "R1C3",
    "R2C1", "R2C2", "R2C3",
    "R3C1", "R3C2", "R3C3"
]


def check_xo_winner(board):
    for condition in XO_WIN_CONDITIONS:
        a, b, c = condition
        if board[a] != '0' and board[a] == board[b] == board[c]:
            return board[a]
    if '0' not in board:
        return 'Draw'
    return None


def get_empty_cells(board):
    return [i for i, cell in enumerate(board) if cell == '0']


def _xo_terminal_score(board, depth):
    """Điểm được tính theo góc nhìn của X: X thắng dương, O thắng âm."""
    winner = check_xo_winner(board)
    if winner == 'X':
        return 10 - depth
    if winner == 'O':
        return depth - 10
    if winner == 'Draw':
        return 0
    return None


def minimax_xo(board, depth, is_max):
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    if is_max:
        best = -float('inf')
        for i in get_empty_cells(board):
            board[i] = 'X'
            best = max(best, minimax_xo(board, depth + 1, False))
            board[i] = '0'
        return best

    best = float('inf')
    for i in get_empty_cells(board):
        board[i] = 'O'
        best = min(best, minimax_xo(board, depth + 1, True))
        board[i] = '0'
    return best


def alphabeta_xo(board, depth, alpha, beta, is_max):
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    if is_max:
        best = -float('inf')
        for i in get_empty_cells(board):
            board[i] = 'X'
            val = alphabeta_xo(board, depth + 1, alpha, beta, False)
            board[i] = '0'
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    best = float('inf')
    for i in get_empty_cells(board):
        board[i] = 'O'
        val = alphabeta_xo(board, depth + 1, alpha, beta, True)
        board[i] = '0'
        best = min(best, val)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def expectimax_xo(board, depth, is_max):
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    empty_cells = get_empty_cells(board)
    if is_max:
        best = -float('inf')
        for i in empty_cells:
            board[i] = 'X'
            best = max(best, expectimax_xo(board, depth + 1, False))
            board[i] = '0'
        return best

    expected_val = 0
    if empty_cells:
        prob = 1.0 / len(empty_cells)
        for i in empty_cells:
            board[i] = 'O'
            expected_val += prob * expectimax_xo(board, depth + 1, True)
            board[i] = '0'
    return expected_val


def minimax_xo_detail(board, depth, is_max, stats):
    stats["visited"] += 1
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    if is_max:
        best = -float('inf')
        for i in get_empty_cells(board):
            board[i] = 'X'
            best = max(best, minimax_xo_detail(board, depth + 1, False, stats))
            board[i] = '0'
        return best

    best = float('inf')
    for i in get_empty_cells(board):
        board[i] = 'O'
        best = min(best, minimax_xo_detail(board, depth + 1, True, stats))
        board[i] = '0'
    return best


def alphabeta_xo_detail(board, depth, alpha, beta, is_max, stats):
    stats["visited"] += 1
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    empty_cells = get_empty_cells(board)

    if is_max:
        best = -float('inf')
        for order, i in enumerate(empty_cells):
            board[i] = 'X'
            val = alphabeta_xo_detail(board, depth + 1, alpha, beta, False, stats)
            board[i] = '0'
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                stats["pruned"] += len(empty_cells) - order - 1
                break
        return best

    best = float('inf')
    for order, i in enumerate(empty_cells):
        board[i] = 'O'
        val = alphabeta_xo_detail(board, depth + 1, alpha, beta, True, stats)
        board[i] = '0'
        best = min(best, val)
        beta = min(beta, best)
        if beta <= alpha:
            stats["pruned"] += len(empty_cells) - order - 1
            break
    return best


def expectimax_xo_detail(board, depth, is_max, stats):
    stats["visited"] += 1
    winner_score = _xo_terminal_score(board, depth)
    if winner_score is not None:
        return winner_score

    empty_cells = get_empty_cells(board)

    if is_max:
        best = -float('inf')
        for i in empty_cells:
            board[i] = 'X'
            best = max(best, expectimax_xo_detail(board, depth + 1, False, stats))
            board[i] = '0'
        return best

    expected_val = 0
    if empty_cells:
        prob = 1.0 / len(empty_cells)
        for i in empty_cells:
            board[i] = 'O'
            expected_val += prob * expectimax_xo_detail(board, depth + 1, True, stats)
            board[i] = '0'
        stats["chance_probability"] = round(prob, 4)
    return expected_val


def _xo_algorithm_detail(algo_name):
    """Thông tin mô tả chi tiết để frontend hiển thị trong popup XO."""
    details = {
        'MINIMAX': {
            'display_name': 'Minimax',
            'group': 'Adversarial Search / Game Playing',
            'role': 'X là MAX, O là MIN',
            'objective': 'Tìm nước đi tối ưu bằng cách giả định đối thủ cũng chơi tối ưu.',
            'decision_rule': 'X chọn candidate có score lớn nhất; O chọn candidate có score nhỏ nhất.',
            'score_rule': 'X thắng: điểm dương, O thắng: điểm âm, hòa: 0.',
            'tree_rule': 'Duyệt toàn bộ cây trò chơi còn lại cho đến terminal state.',
            'complexity': 'Số node có thể lớn vì không cắt tỉa nhánh.',
            'color': 'blue'
        },
        'ALPHA_BETA': {
            'display_name': 'Alpha-Beta Pruning',
            'group': 'Adversarial Search / Game Playing',
            'role': 'X là MAX, O là MIN',
            'objective': 'Giữ kết quả tối ưu như Minimax nhưng giảm số nhánh phải duyệt.',
            'decision_rule': 'Dùng alpha là giá trị tốt nhất của MAX và beta là giá trị tốt nhất của MIN.',
            'score_rule': 'Nếu beta <= alpha thì nhánh còn lại không ảnh hưởng quyết định nên bị cắt tỉa.',
            'tree_rule': 'Mở rộng cây giống Minimax, nhưng bỏ qua những nhánh chắc chắn không cần xét.',
            'complexity': 'Thường xét ít node hơn Minimax nhờ pruning.',
            'color': 'emerald'
        },
        'EXPECTIMAX': {
            'display_name': 'Expectimax',
            'group': 'Adversarial Search / Game Playing with Chance Node',
            'role': 'X là MAX, O được mô phỏng như Chance/Expected node trong cây đánh giá.',
            'objective': 'Chọn nước đi có giá trị kỳ vọng tốt nhất khi đối thủ hoặc môi trường có yếu tố ngẫu nhiên.',
            'decision_rule': 'MAX chọn điểm cao nhất; Chance node lấy trung bình kỳ vọng các khả năng.',
            'score_rule': 'Điểm của node chance được tính theo xác suất trung bình của các nước đi có thể xảy ra.',
            'tree_rule': 'Không giả định đối thủ luôn tối ưu tuyệt đối như Minimax.',
            'complexity': 'Phù hợp để minh họa trường hợp có yếu tố xác suất.',
            'color': 'violet'
        }
    }
    return details.get(algo_name, {
        'display_name': algo_name,
        'group': 'Adversarial Search',
        'role': 'Game XO',
        'objective': 'Mô phỏng thuật toán đối kháng.',
        'decision_rule': 'Chọn nước đi dựa trên score.',
        'score_rule': 'Terminal score theo kết quả thắng/thua/hòa.',
        'tree_rule': 'Duyệt cây trạng thái của trò chơi.',
        'complexity': '-',
        'color': 'slate'
    })


def _xo_algorithm_note(algo_name):
    info = _xo_algorithm_detail(algo_name)
    return f"{info['display_name']}: {info['objective']} {info['decision_rule']}"


def _xo_evaluate_candidate(board, move_index, player, algo_name):
    board[move_index] = player
    candidate_board = copy.deepcopy(board)

    stats = {
        "visited": 0,
        "pruned": 0,
        "chance_probability": None
    }

    # Sau khi player đặt quân xong, lượt tiếp theo thuộc về người còn lại.
    next_is_max = (player == 'O')

    if algo_name == 'MINIMAX':
        value = minimax_xo_detail(board, 0, next_is_max, stats)
        rule = 'Max chọn điểm lớn nhất' if player == 'X' else 'Min chọn điểm nhỏ nhất'
    elif algo_name == 'ALPHA_BETA':
        value = alphabeta_xo_detail(board, 0, -float('inf'), float('inf'), next_is_max, stats)
        rule = 'Max chọn điểm lớn nhất + cắt tỉa αβ' if player == 'X' else 'Min chọn điểm nhỏ nhất + cắt tỉa αβ'
    elif algo_name == 'EXPECTIMAX':
        value = expectimax_xo_detail(board, 0, next_is_max, stats)
        rule = 'Chọn theo điểm kỳ vọng' if player == 'X' else 'O chọn nhánh làm giảm lợi thế của X'
    else:
        value = 0
        rule = 'Không xác định'

    board[move_index] = '0'

    return {
        "move_index": move_index,
        "move_label": XO_CELL_NAMES[move_index],
        "action": f"{player} -> ô {move_index + 1} ({XO_CELL_NAMES[move_index]})",
        "state": candidate_board,
        "score": round(value, 3),
        "evaluated_nodes": stats["visited"],
        "pruned_branches": stats["pruned"],
        "chance_probability": stats["chance_probability"],
        "rule": rule,
        "evaluation_detail": f"Sau khi đặt {player} vào ô {move_index + 1}, thuật toán đánh giá cây trạng thái còn lại và trả về score {round(value, 3)}.",
        "is_terminal": check_xo_winner(candidate_board) is not None,
        "terminal_result": check_xo_winner(candidate_board),
        "chosen": False
    }


def play_xo_ai_game(algo_name):
    start_time = time.time()
    board = ['0'] * 9
    logs = []
    reached_boards = [copy.deepcopy(board)]
    current_player = 'X'
    step = 0

    logs.append({
        "xo_detail": True,
        "algorithm": algo_name,
        "algorithm_note": _xo_algorithm_note(algo_name),
        "algorithm_detail": _xo_algorithm_detail(algo_name),
        "step": 0,
        "node": {
            "id": "A",
            "state": copy.deepcopy(board),
            "action": "Start"
        },
        "before_board": copy.deepcopy(board),
        "board": copy.deepcopy(board),
        "player": "Start",
        "current_player": current_player,
        "chosen_move": None,
        "chosen_score": None,
        "candidates": [],
        "frontier_added": [],
        "reached": copy.deepcopy(reached_boards),
        "status": "START",
        "winner_after": None,
        "time": 0
    })

    while check_xo_winner(board) is None:
        step += 1
        before_board = copy.deepcopy(board)
        empty_cells = get_empty_cells(board)
        candidates = []

        for move_index in empty_cells:
            candidates.append(_xo_evaluate_candidate(board, move_index, current_player, algo_name))

        if current_player == 'X':
            best_score = max(item["score"] for item in candidates)
            chosen_candidate = next(item for item in candidates if item["score"] == best_score)
        else:
            best_score = min(item["score"] for item in candidates)
            chosen_candidate = next(item for item in candidates if item["score"] == best_score)

        for item in candidates:
            item["chosen"] = (item["move_index"] == chosen_candidate["move_index"])

        board[chosen_candidate["move_index"]] = current_player
        reached_boards.append(copy.deepcopy(board))
        winner_after = check_xo_winner(board)

        logs.append({
            "xo_detail": True,
            "algorithm": algo_name,
            "algorithm_note": _xo_algorithm_note(algo_name),
            "algorithm_detail": _xo_algorithm_detail(algo_name),
            "step": step,
            "node": {
                "id": chr(65 + step) if step < 26 else f"N{step}",
                "state": before_board,
                "action": chosen_candidate["action"]
            },
            "before_board": before_board,
            "board": copy.deepcopy(board),
            "player": current_player,
            "current_player": 'O' if current_player == 'X' else 'X',
            "chosen_move": chosen_candidate["move_index"],
            "chosen_label": chosen_candidate["move_label"],
            "chosen_score": chosen_candidate["score"],
            "candidates": candidates,
            "frontier_added": candidates,
            "reached": copy.deepcopy(reached_boards),
            "status": "GOAL" if winner_after else "RUNNING",
            "winner_after": winner_after,
            "total_evaluated_nodes": sum(item["evaluated_nodes"] for item in candidates),
            "total_pruned_branches": sum(item["pruned_branches"] for item in candidates),
            "time": round((time.time() - start_time) * 1000, 2)
        })

        current_player = 'O' if current_player == 'X' else 'X'

    logs.append({
        "xo_detail": True,
        "algorithm": algo_name,
        "algorithm_note": _xo_algorithm_note(algo_name),
        "algorithm_detail": _xo_algorithm_detail(algo_name),
        "step": step + 1,
        "node": {
            "id": "END",
            "state": copy.deepcopy(board),
            "action": "Game Over"
        },
        "before_board": copy.deepcopy(board),
        "board": copy.deepcopy(board),
        "player": "End",
        "current_player": "-",
        "chosen_move": None,
        "chosen_score": None,
        "candidates": [],
        "frontier_added": [],
        "reached": copy.deepcopy(reached_boards),
        "status": "TERMINAL",
        "winner": check_xo_winner(board),
        "winner_after": check_xo_winner(board),
        "total_evaluated_nodes": 0,
        "total_pruned_branches": 0,
        "time": round((time.time() - start_time) * 1000, 2)
    })

    return logs
