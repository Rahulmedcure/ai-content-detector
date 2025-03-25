import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)

# ✅ Allow only your frontend domain to access the API
CORS(app, resources={r"/*": {"origins": "https://theaicontentdetector.com"}})

# ✅ Load OpenAI API Key securely from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def home():
    return "✅ AI Content Detector API is running."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    input_text = data["text"]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI content detection assistant. Check how likely the given text is AI-generated. Respond with a likelihood score from 0 to 1."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )

        result_text = response.choices[0].message.content.strip()

        # Try to extract score from the response
        import re
        match = re.search(r"([0]\.\d+|1\.0+)", result_text)
        score = float(match.group(1)) if match else 0.5

        return jsonify({
            "result": "AI-Generated" if score > 0.5 else "Human-Written",
            "ai_score": score
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong while processing the request."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
