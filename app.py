import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic

# Ortam değişkenlerini yükle (.env dosyasından CLAUDE_API_KEY için)
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
anthropic = Anthropic(api_key=CLAUDE_API_KEY)

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        voltages = data.get("voltages", [])
        heart_rate = data.get("heartRate", 0)

        prompt = (
            f"📈 Nabız: {heart_rate} bpm\n"
            f"🔌 Voltajlar: {voltages[:30]}\n\n"
            "Bu EKG verisini tıbbi olarak analiz et.\n"
            "- P, QRS ve T dalgalarını açıkla\n"
            "- Ritim tipi belirt\n"
            "- Varsa anormallikleri yorumla\n"
            "- Açıklaman kısa ve net olsun (maksimum 4 cümle).\n"
        )

        completion = anthropic.messages.create(
            model="claude-3-opus-20240229",  # İsteğe göre haiku vs. yapılabilir
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        yorum = completion.content[0].text.strip()
        return jsonify({"comment": yorum})

    except Exception as e:
        print("❌ Claude yorum hatası:", e)
        return jsonify({"comment": "Yorum alınamadı."}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)