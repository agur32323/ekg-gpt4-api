import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY")

def yorum_huggingface_ile(voltages, heart_rate):
    try:
        prompt = (
            f"EKG verisi geldi. Nabız: {heart_rate} bpm.\n"
            f"Volt: {voltages[:10]}...\n"
            "P, QRS, T dalgalarını ve ritmi yorumla:"
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
        return result[0].get("generated_text", "Yorum alınamadı.")
    except Exception as e:
        print("HuggingFace yorum hatası:", e)
        return "Yorum alınamadı."

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        voltages = data["voltages"]
        heart_rate = data["heartRate"]
        yorum = yorum_huggingface_ile(voltages, heart_rate)
        return jsonify({"comment": yorum})
    except Exception as e:
        print("Sunucu hatası:", e)
        return jsonify({"comment": "Sunucu hatası oluştu."}), 500

if __name__ == "__main__":
    app.run()