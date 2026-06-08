from tracemalloc import start

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
    k = data.get('k', 2) 
    t = data.get('t', 100.0)
    alpha = data.get('alpha', 0.95)
    
    result = solver_ai.solve(start, goal, algo, k=k, t=t, alpha=alpha)
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True, port=5000)