import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        voltages = data.get("voltages", [])
        heart_rate = data.get("heartRate", 0)

        prompt = (
            f"EKG verisi geldi. Nabız: {heart_rate} bpm.\n"
            f"Voltaj örnekleri: {voltages[:10]}...\n"
            "Lütfen şu başlıklarla tıbbi yorum yap:\n"
            "- P dalgası görünüyor mu?\n"
            "- QRS kompleksi belirgin mi?\n"
            "- T dalgası var mı?\n"
            "- Ritim düzenli mi?\n"
            "- Klinik yorum:\n\n"
            "Yorum:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # veya "gpt-4" planına bağlı olarak
            messages=[
                {"role": "system", "content": "Sen bir kardiyolog gibi davranan tıbbi yapay zekasın."},
                {"role": "user", "content": prompt}
            ]
        )

        comment = response['choices'][0]['message']['content']
        return jsonify({"comment": comment})

    except Exception as e:
        print("❌ GPT API Hatası:", e)
        return jsonify({"comment": "Yorum alınamadı."}), 500

if __name__ == "__main__":
    app.run(debug=True)