import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

app = Flask(__name__)
CORS(app)

anthropic = Anthropic(api_key=CLAUDE_API_KEY)

# ----------------- EKG ANALİZİ ----------------- #
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
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

        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        comment = "".join(block.text for block in response.content if hasattr(block, "text"))
        return jsonify({"comment": comment.strip()})

    except Exception as e:
        print("❌ Claude sunucu hatası:", str(e))
        return jsonify({"comment": f"Sunucu hatası: {str(e)}"}), 500

# ----------------- GLUKOZ ANALİZİ ----------------- #
@app.route("/analyze_glucose", methods=["POST"])
def analyze_glucose():
    try:
        if not CLAUDE_API_KEY:
            return jsonify({"interpretation": "API anahtarı eksik!"}), 401

        data = request.get_json()
        glucose_data = data.get("glucose_data", [])

        if not glucose_data or not isinstance(glucose_data, list):
            return jsonify({"interpretation": "Glukoz verisi alınamadı."}), 400

        # GlukozEntry formatı: { "glucoseValue": 110, "timestamp": "2025-05-04T14:00:00" }
        values = [entry["glucoseValue"] for entry in glucose_data]
        timestamps = [entry["timestamp"] for entry in glucose_data]

        avg = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)

        prompt = (
            f"Glukoz ölçüm verileri (mg/dL): {values}\n"
            f"Zamanlar: {timestamps}\n"
            f"Ortalama: {avg:.1f} mg/dL, En yüksek: {max_val}, En düşük: {min_val}\n\n"
            "Bu glukoz verilerini tıbbi olarak yorumla.\n"
            "- Hipoglisemi ya da hiperglisemi var mı?\n"
            "- Glukoz trendi nasıl?\n"
            "- 3-4 cümleyle kısa ve açıklayıcı bir yorum yap.\n"
        )

        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        interpretation = "".join(block.text for block in response.content if hasattr(block, "text"))
        return jsonify({"interpretation": interpretation.strip()})

    except Exception as e:
        print("❌ Glukoz sunucu hatası:", str(e))
        return jsonify({"interpretation": f"Sunucu hatası: {str(e)}"}), 500

# ----------------- SUNUCU BAŞLAT ----------------- #
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
