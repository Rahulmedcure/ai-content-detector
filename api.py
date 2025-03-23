from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# Root route (for health check or info)
@app.route('/')
def home():
    return "✅ AI Content Detector API is running."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400  # Return error if no text is provided
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate the AI score
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    ai_score = float(scores[0][1])  # Class 1 corresponds to AI-generated
    
    # Return the result as JSON
    result = "AI-Generated" if ai_score > 0.5 else "Human-Written"
    return jsonify({
        "ai_score": ai_score,
        "result": result
    })

# Ensure the app runs with Gunicorn or another WSGI server in production
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)  # Make sure your app is accessible
