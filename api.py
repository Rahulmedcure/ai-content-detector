from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route('/')
def home():
    return "<p>✅ AI Content Detector API is running.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Dummy logic for testing
    ai_score = 0.68  # example
    result = "AI-Generated" if ai_score > 0.5 else "Human-Written"

    return jsonify({
        "ai_score": ai_score,
        "result": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
