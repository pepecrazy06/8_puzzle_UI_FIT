class Node:
    def __init__(self, state, parent=None, action="", cost=0, depth=0, node_id=""):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth
        self.node_id = node_id
