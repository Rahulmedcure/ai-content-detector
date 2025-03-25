from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://theaicontentdetector.com"}})

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/")
def home():
    return "AI Content Detector API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_text = data.get("text", "")

        if not input_text.strip():
            return jsonify({"error": "Empty input text"}), 400

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI content detector. Analyze the following text and determine whether it appears to be written by an AI or human. Reply only with 'AI-Generated' or 'Human-Written' and an AI confidence score between 0 and 1."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )

        reply = response["choices"][0]["message"]["content"]
        result_line = reply.split("\n")[0]
        score_line = next((line for line in reply.split("\n") if "score" in line.lower()), None)

        result = "AI-Generated" if "AI" in result_line else "Human-Written"
        ai_score = float("".join(filter(lambda c: c.isdigit() or c == ".", score_line))) if score_line else 0.5

        return jsonify({"result": result, "ai_score": ai_score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
