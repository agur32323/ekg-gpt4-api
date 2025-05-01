import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # API key geçerli mi?
        if not CLAUDE_API_KEY:
            return jsonify({"comment": "API anahtarı eksik!"}), 401

        data = request.get_json()
        voltages = data.get("voltages", [])
        heart_rate = data.get("heartRate", 0)

        if not voltages or not isinstance(voltages, list):
            return jsonify({"comment": "Geçerli voltaj verisi yok."}), 400

        prompt = (
            f"Nabız: {heart_rate} bpm\n"
            f"Voltajlar: {voltages[:20]}\n\n"
            "Bu EKG verisini analiz et.\n"
            "- P, QRS, T dalgalarını açıkla\n"
            "- Ritim türünü belirle\n"
            "- Tıbbi olarak yorumla\n"
            "- Kısa ve net bir açıklama yap (maks. 3-4 cümle).\n"
        )

        anthropic = Anthropic(api_key=CLAUDE_API_KEY)
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Claude'dan gelen cevabı doğru al
        comment = ""
        for block in response.content:
            if hasattr(block, "text"):
                comment += block.text

        return jsonify({"comment": comment.strip()})

    except Exception as e:
        print("❌ Claude sunucu hatası:", str(e))
        return jsonify({"comment": f"Sunucu hatası: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)