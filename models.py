class Node:
    def __init__(self, state, parent=None, action="", cost=0, depth=0, node_id=""):
        self.state = state        # Trạng thái ma trận dạng Tuple 9 phần tử
        self.parent = parent      # Trỏ tới đối tượng Node cha
        self.action = action      # Hành động di chuyển từ cha dịch tới ('Up', 'Down', 'Left', 'Right')
        self.cost = cost          # Chi phí g(n) từ Start đến hiện tại
        self.depth = depth        # Độ sâu của node trong cây tìm kiếm
        self.node_id = node_id    # Tên Node định danh tự động (A, B, C...)