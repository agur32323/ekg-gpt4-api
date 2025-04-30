import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# .env içeriğini yükle
load_dotenv()

# Flask başlat
app = Flask(__name__)
CORS(app)

# Hugging Face API anahtarı
HF_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face yorumlama fonksiyonu
def yorum_huggingface_ile(voltages, heart_rate):
    try:
        # Prompt: Daha iyi yorum için açık ve İngilizce
        prompt = (
            f"You are an AI cardiologist.\n\n"
            f"Heart rate: {heart_rate} bpm\n"
            f"Voltages (mV): {voltages[:10]}...\n\n"
            "Please analyze:\n"
            "- Is the P wave visible?\n"
            "- Is the QRS complex distinct?\n"
            "- Is the T wave present?\n"
            "- Is the rhythm regular?\n"
            "- Is this consistent with sinus rhythm?\n"
            "- Provide a brief clinical interpretation.\n\n"
            "Answer:"
        )

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-small",
            headers=headers,
            json={"inputs": prompt},
            timeout=20
        )

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            print("⛔ Beklenmeyen cevap:", result)
            return "Modelden geçerli bir yorum alınamadı."

    except Exception as e:
        print("❌ HuggingFace yorum hatası:", e)
        return "Yorum alınamadı (HuggingFace bağlantısı başarısız)."

# API uç noktası
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        voltages = data.get("voltages", [])
        heart_rate = data.get("heartRate", 0)
        comment = yorum_huggingface_ile(voltages, heart_rate)
        return jsonify({"comment": comment})
    except Exception as e:
        print("🛑 Sunucu hatası:", e)
        return jsonify({"comment": "Sunucu hatası oluştu."}), 500

# Uygulamayı başlat
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)