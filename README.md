# Thông tin
- Đại Học Công Nghệ Kỹ Thuật TP.HCM
- Khoa Công Nghệ Thông Tin
- Học phần: Trí Tuệ Nhân Tạo
- Mã LHP: ARIN330585_08
- GVHD: TS. Phan Thị Huyền Trang 
- SVTH: Thái Nhựt Huy
- Đề Tài: Giao diện trò chơi 8 PUZZLE ứng dụng các thuật toán
- Cập nhật lần cuối: 06/04/2026
- Link github: https://github.com/pepecrazy06/8_puzzle_UI_FIT
# 8-Puzzle Multi-Algorithm AI Search Visualizer

Dự án mô phỏng và trực quan hóa quá trình giải bài toán **8-Puzzle** bằng nhiều thuật toán tìm kiếm AI. Người dùng có thể nhập trạng thái bắt đầu, trạng thái đích, chọn thuật toán và quan sát từng bước duyệt trạng thái trên giao diện web.

## 1. Chức năng chính

- Hiển thị bàn cờ 8-Puzzle dạng ma trận 3x3.
- Cho phép nhập **Start State** và **Goal State**.
- Chọn và chạy nhiều thuật toán tìm kiếm khác nhau.
- Gửi dữ liệu từ giao diện web về Flask backend thông qua API `/api/solve`.
- Hiển thị kết quả chạy thuật toán: thành công, thất bại, thời gian xử lý và số bước duyệt.
- Xem chi tiết quá trình tìm kiếm gồm:
  - Node hiện tại.
  - Frontier / hàng đợi / heap.
  - Reached / visited states.
  - Neighbors và giá trị heuristic đối với nhóm thuật toán leo đồi.

## 2. Công nghệ sử dụng

- **Python 3**
- **Flask**: xây dựng backend và API.
- **HTML, CSS, JavaScript**: xây dựng giao diện web.
- **Tailwind CSS CDN**: tạo giao diện nhanh, hiện đại.
- **OOP Python**: tổ chức Node và PuzzleSolver.

## 3. Cấu trúc thư mục

```text
8-puzzle-ai-visualizer/
│
├── app.py              # Flask server, định nghĩa route giao diện và API solve
├── solver.py           # Xử lý thuật toán tìm kiếm 8-Puzzle
├── models.py           # Định nghĩa class Node
├── templates/
│   └── index.html      # Giao diện web chính
└── README.md
```

> Lưu ý: Trong `app.py`, chương trình đang render file `templates/index.html`, vì vậy nên đặt file giao diện vào thư mục `templates/index.html` để chạy đúng.

## 4. Cài đặt

### Bước 1: Tải project về máy

```bash
git clone <link-repository>
cd 8-puzzle-ai-visualizer
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Bước 3: Cài thư viện cần thiết

```bash
pip install flask
```

## 5. Cách chạy chương trình

Chạy Flask server:

```bash
python app.py
```

Sau đó mở trình duyệt và truy cập:

```text
http://127.0.0.1:5000
```

## 6. Cách sử dụng

1. Mở giao diện web.
2. Chọn thuật toán trong ô **Chọn thuật toán**.
3. Chọn loại trạng thái muốn chỉnh:
   - **Mảng bắt đầu**: trạng thái ban đầu.
   - **Mảng đích**: trạng thái cần đạt.
4. Nhập các số từ `0` đến `8`, trong đó `0` là ô trống.
5. Nhấn **Cập nhật** để lưu trạng thái.
6. Nhấn **Run AI Engine** để chạy thuật toán.
7. Dùng các nút:
   - `←`: lùi một bước.
   - `→`: tiến một bước.
   - `Tự động`: tự động chạy animation.
   - `Reset Trạng Thái`: làm mới trạng thái.
8. Nhấn **Xem chi tiết quá trình** để xem bảng duyệt trạng thái.

## 7. Danh sách thuật toán hỗ trợ

### 7.1. Uninformed Search

- **BFS - Child-Node Approach**
- **BFS - Expand Approach**
- **Depth-First Search (DFS)**
- **Iterative Deepening Search (IDS)**
- **Uniform Cost Search (UCS)**

### 7.2. Informed Search

- **Greedy Best-First Search - Manhattan Distance**
- **Greedy Best-First Search - Misplaced Tiles**
- **A\* Search - Manhattan Distance**
- **A\* Search - Misplaced Tiles**

### 7.3. Local Search

- **Simple Hill Climbing - Manhattan**
- **Steepest Ascent Hill Climbing - Manhattan**
- **Simple Hill Climbing - Misplaced Tiles**
- **Steepest Ascent Hill Climbing - Misplaced Tiles**
- **First-Choice Hill Climbing**
- **Stochastic Hill Climbing**
- **Random Restart Hill Climbing**
- **Local Beam Search**, với `k = 2`

## 8. Heuristic sử dụng

### 8.1. Misplaced Tiles

Đếm số ô đang sai vị trí so với trạng thái đích, không tính ô trống `0`.

### 8.2. Manhattan Distance

Tính tổng khoảng cách hàng và cột của từng ô từ vị trí hiện tại đến vị trí đích.

Công thức cho từng ô:

```text
|current_row - goal_row| + |current_col - goal_col|
```

Tổng Manhattan là tổng khoảng cách của các ô từ `1` đến `8`, không tính ô trống `0`.

## 9. API backend

### Endpoint

```http
POST /api/solve
```

### Request body

```json
{
  "start": [1, 2, 3, 4, 0, 6, 7, 5, 8],
  "goal": [1, 2, 3, 4, 5, 6, 7, 8, 0],
  "algo": "ASTAR_MANHATTAN"
}
```

### Response mẫu

```json
{
  "success": true,
  "history": [],
  "time": 12.35
}
```

Trong đó:

- `success`: cho biết thuật toán có tìm thấy trạng thái đích hay không.
- `history`: danh sách các bước duyệt trạng thái.
- `time`: thời gian xử lý tính bằng mili giây.

## 10. Ý nghĩa các file chính

### `models.py`

Chứa class `Node`, dùng để lưu thông tin một trạng thái trong cây tìm kiếm:

- `state`: trạng thái ma trận 8-Puzzle.
- `parent`: node cha.
- `action`: hành động di chuyển.
- `cost`: chi phí đường đi `g(n)`.
- `depth`: độ sâu node.
- `node_id`: mã định danh node.

### `solver.py`

Chứa class `PuzzleSolver`, chịu trách nhiệm:

- Sinh trạng thái kề.
- Tính heuristic.
- Chạy các thuật toán tìm kiếm.
- Ghi lại lịch sử duyệt trạng thái.
- Trả kết quả về backend.

### `app.py`

Chứa Flask server:

- Route `/` để hiển thị giao diện.
- Route `/api/solve` để nhận trạng thái từ frontend và gọi bộ giải.
- Chạy server ở port `5000`.

### `index.html`

Chứa giao diện chính:

- Bàn cờ 8-Puzzle.
- Bảng trạng thái.
- Nhật ký chạy thuật toán.
- Form chọn thuật toán.
- Modal xem chi tiết quá trình duyệt.

## 11. Hướng phát triển

- Thêm kiểm tra trạng thái có giải được hay không trước khi chạy.
- Hiển thị đường đi lời giải cuối cùng riêng biệt.
- Thêm so sánh số node mở rộng giữa các thuật toán.
- Thêm biểu đồ hiệu năng theo thời gian chạy.
- Cho phép lưu lịch sử chạy ra file.
- Tối ưu giao diện trên thiết bị di động.


