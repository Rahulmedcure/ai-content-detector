from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '✅ AI Content Detector API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Simulated detection logic (replace with real model later)
        import random
        ai_score = random.uniform(0.2, 0.95)

        highlights = []
        words = text.split()
        for word in words:
            highlights.append({
                "text": word + " ",
                "is_ai": random.random() < ai_score
            })

        result = "Likely AI" if ai_score > 0.6 else "Likely Human"

        return jsonify({
            "result": result,
            "ai_score": ai_score,
            "highlights": highlights
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong."}), 500

if __name__ == '__main__':
    app.run()
