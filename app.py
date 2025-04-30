import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic

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

        # 👇 Claude'a gönderilecek istem
        prompt = (
            f"Nabız: {heart_rate} bpm\n"
            f"Voltajlar: {voltages[:10]}...\n"
            f"EKG verisini tıbbi olarak yorumla. P, QRS, T dalgalarını açıklayıp ritim analizi yap.\n"
        )

        completion = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        yorum = completion.content[0].text.strip()
        return jsonify({"comment": yorum})

    except Exception as e:
        print("Claude yorum hatası:", e)
        return jsonify({"comment": "Yorum alınamadı."}), 500

if __name__ == "__main__":
    app.run()