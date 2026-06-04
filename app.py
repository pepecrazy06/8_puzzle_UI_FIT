from flask import Flask, request, jsonify, render_template
from solver import PuzzleSolver

app = Flask(__name__, template_folder='.')
solver_ai = PuzzleSolver()

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    start = data.get('start')
    goal = data.get('goal')
    algo = data.get('algo')
    
    # Kích hoạt bộ giải đa thuật toán OOP
    result = solver_ai.solve(start, goal, algo)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)