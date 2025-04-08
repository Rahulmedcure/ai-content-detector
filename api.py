@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Token limit check (500 tokens max)
        tokenized = tokenizer(text, return_tensors="pt", truncation=False)
        if tokenized['input_ids'].shape[1] > 500:
            return jsonify({"error": "Text too long. Please submit up to 500 tokens."}), 400

        # Split text into sentences
        sentences = split_into_sentences(text)
        highlights = []
        total_score = 0

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                ai_score = probs[0][1].item()
                total_score += ai_score
                highlights.append({
                    "text": sentence,
                    "is_ai": ai_score > 0.6,
                    "ai_score": round(ai_score, 3)
                })

        avg_score = total_score / len(highlights)
        result = "Likely AI" if avg_score > 0.6 else "Likely Human"

        return jsonify({
            "result": result,
            "ai_score": round(avg_score, 3),
            "highlights": highlights
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong."}), 500
