from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    return 'AI Content Detector API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Is the following text AI-generated or human-written? Just answer 'AI-generated' or 'Human-written' with a score out of 1.\n\n{text}"}
            ],
            temperature=0.2
        )

        result_text = response['choices'][0]['message']['content']
        if "AI-generated" in result_text:
            label = "AI-Generated"
            score = 0.85
        else:
            label = "Human-Written"
            score = 0.15

        return jsonify({
            "result": label,
            "ai_score": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
