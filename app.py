from flask import Flask, request, jsonify, render_template
from solver import PuzzleSolver

app = Flask(__name__, template_folder='templates')
solver_ai = PuzzleSolver()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json(force=True)

        start = data.get('start')
        goal = data.get('goal')
        algo = data.get('algo')

        k = int(data.get('k', 2))
        t = float(data.get('t', 100.0))
        alpha = float(data.get('alpha', 0.95))

        result = solver_ai.solve(start, goal, algo, k=k, t=t, alpha=alpha)
        return jsonify(result)

    except Exception as e:
        app.logger.exception("Solve error")
        return jsonify({
            "success": False,
            "history": [],
            "error": str(e)
        }), 500


@app.route('/api/solve-belief-common', methods=['POST'])
def solve_belief_common():
    try:
        data = request.get_json(force=True)

        result = solver_ai.solve_belief_common_from_pattern(
            start_pattern=data.get('start_pattern'),
            goal_pattern=data.get('goal_pattern'),
            max_belief_states=int(data.get('max_belief_states', 2)),
            max_depth=int(data.get('max_depth', 30))
        )

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Belief common action solve error")
        return jsonify({
            "success": False,
            "belief": True,
            "belief_mode": "COMMON_ACTION",
            "history": [],
            "error": str(e)
        }), 500


@app.route('/api/solve-belief', methods=['POST'])
def solve_belief_alias():
    try:
        data = request.get_json(force=True)

        result = solver_ai.solve_belief_common_from_pattern(
            start_pattern=data.get('start_pattern'),
            goal_pattern=data.get('goal_pattern'),
            max_belief_states=int(data.get('max_belief_states', 2)),
            max_depth=int(data.get('max_depth', 30))
        )

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Belief alias solve error")
        return jsonify({
            "success": False,
            "belief": True,
            "belief_mode": "COMMON_ACTION",
            "history": [],
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
