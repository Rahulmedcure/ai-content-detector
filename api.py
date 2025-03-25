from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
# Allow only your frontend domain, or use "*" for testing
CORS(app, resources={r"/predict": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def home():
    return "✅ AI Content Detector API is running."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI content detector. Check if the following text is written by AI."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )

        ai_reply = response.choices[0].message.content.strip()
        ai_score = 0.5  # Optional: placeholder AI score logic

        return jsonify({"result": ai_reply, "ai_score": ai_score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
