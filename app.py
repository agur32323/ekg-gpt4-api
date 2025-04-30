import os
import anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Anthropic API anahtarını ortam değişkeninden al
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Anthropic istemcisini oluştur
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def yorum_claude_ile(voltages, heart_rate):
    try:
        prompt = (
            f"EKG verisi geldi. Nabız: {heart_rate} bpm.\n"
            f"Volt: {voltages[:10]}...\n"
            "P, QRS, T dalgalarını ve ritmi yorumla:"
        )

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text
    except Exception as e:
        print("Claude yorum hatası:", e)
        return "Yorum alınamadı."

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        voltages = data["voltages"]
        heart_rate = data["heartRate"]
        yorum = yorum_claude_ile(voltages, heart_rate)
        return jsonify({"comment": yorum})
    except Exception as e:
        print("Sunucu hatası:", e)
        return jsonify({"comment": "Sunucu hatası oluştu."}), 500

if __name__ == "__main__":
    app.run()