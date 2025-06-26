import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic
import numpy as np # RR aralıklarından HRV metrikleri hesaplamak için eklendi

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

app = Flask(__name__)
CORS(app)

anthropic = Anthropic(api_key=CLAUDE_API_KEY)
# ----------------- KALP ATIŞ ANALİZİ ----------------- #
@app.route("/analyze_heart", methods=["POST"])
def analyze_heart():
    """
    İstek örneği (Flutter tarafı):
    {
      "bpm_values": [90, 92, 91, ...],
      "min": 90,
      "max": 93,
      "average": 91.8
    }
    Dönen JSON:
    { "heart_interpretation": "..." }
    """
    try:
        if not CLAUDE_API_KEY:
            return jsonify({"heart_interpretation": "API anahtarı eksik!"}), 401

        data = request.get_json() or {}
        bpm_values = data.get("bpm_values", [])
        avg = data.get("average")
        max_val = data.get("max")
        min_val = data.get("min")

        # Temel kontroller
        if not bpm_values or not isinstance(bpm_values, list):
            return jsonify({"heart_interpretation": "Geçerli BPM verisi yok."}), 400

        # İstemci göndermediyse sunucu tarafında hesapla
        avg = avg or (sum(bpm_values) / len(bpm_values))
        max_val = max_val or max(bpm_values)
        min_val = min_val or min(bpm_values)

        # --- Claude'a gönderilecek prompt ---
        prompt = (
            f"Son 20 kalp atışı değerleri (BPM): {bpm_values}\n"
            f"Maksimum: {max_val} bpm, Minimum: {min_val} bpm, Ortalama: {avg:.1f} bpm\n\n"
            "Bu kalp atış hızı verisini klinik olarak yorumla:\n"
            "- Ritim düzenli mi, düzensiz mi?\n"
            "- Değerler istirahat için normal mi (<60 bradikardi, >100 taşikardi kabul edilir)?\n"
            "- Kardiyovasküler sağlık veya olası riskler hakkında kısaca bilgi ver.\n"
            "- 3-4 cümlelik net ve eyleme geçirilebilir tavsiyeler ekle.\n"
        )

        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=256,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )

        interpretation = "".join(
            blk.text for blk in response.content if hasattr(blk, "text")
        )
        return jsonify({"heart_interpretation": interpretation.strip()})

    except Exception as e:
        print("❌ Heart analysis error:", str(e))
        return jsonify({"heart_interpretation": f"Sunucu hatası: {str(e)}"}), 500
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
            f"Voltajlar: {voltages[:20]}\n\n" # İlk 20 voltajı gönderiyoruz, çok uzun olmaması için
            "Bu EKG verisini analiz et.\n"
            "- P, QRS, T dalgalarını açıkla\n"
            "- Ritim türünü belirle\n"
            "- Tıbbi olarak yorumla\n"
            "- Kısa ve net bir açıklama yap (maks. 3-4 cümle).\n"
        )

        response = anthropic.messages.create(
            model="claude-3-opus-20240229", # Daha gelişmiş bir model kullanıldı
            max_tokens=512,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        comment = "".join(block.text for block in response.content if hasattr(block, "text"))
        return jsonify({"comment": comment.strip()})

    except Exception as e:
        print("❌ Claude sunucu hatası (EKG):", str(e))
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
        values = [
            entry.get("glucoseValue") or
            entry.get("blood_glucose") or
            entry.get("value") or
            0.0
            for entry in glucose_data
        ]
        timestamps = [
            entry.get("timestamp") or
            entry.get("recorded_at") or
            entry.get("dateTime") or
            "unknown"
            for entry in glucose_data
        ]

        # Sadece geçerli glukoz değerlerini (0'dan büyük) dikkate al
        valid_values = [v for v in values if v > 0]
        if not valid_values:
            return jsonify({"interpretation": "Yorumlanacak geçerli glukoz verisi yok."}), 400

        avg = sum(valid_values) / len(valid_values)
        max_val = max(valid_values)
        min_val = min(valid_values)

        prompt = (
            f"Glukoz ölçüm verileri (mg/dL): {valid_values}\n"
            f"Zamanlar: {timestamps}\n"
            f"Ortalama: {avg:.1f} mg/dL, En yüksek: {max_val}, En düşük: {min_val}\n\n"
            "Bu glukoz verilerini tıbbi olarak yorumla.\n"
            "- Hipoglisemi (düşük kan şekeri) ya da hiperglisemi (yüksek kan şekeri) var mı? Normal aralık 70-140 mg/dL'dir.\n"
            "- Glukoz trendi nasıl (yükseliyor, düşüyor, stabil)?\n"
            "- Veri toplama zamanları göz önüne alındığında, yemek sonrası mı yoksa açlık durumu mu daha olası?\n"
            "- 3-4 cümleyle kısa ve açıklayıcı bir yorum yap ve olası tavsiyelerde bulun (örn: doktora danışın, beslenmeye dikkat edin).\n"
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


# ----------------- HRV & RR Intervals ANALİZİ ----------------- #
@app.route("/analyze_hrv_rr", methods=["POST"])
def analyze_hrv_rr():
    try:
        if not CLAUDE_API_KEY:
            return jsonify({"hrv_interpretation": "API anahtarı eksik!"}), 401

        data = request.get_json()
        rr_intervals = data.get("rr_intervals", []) # RR aralıkları milisaniye cinsinden
        sdnn = data.get("sdnn", None) # Opsiyonel: Hesaplanan SDNN değeri
        rmssd = data.get("rmssd", None) # Opsiyonel: Hesaplanan RMSSD değeri

        if not rr_intervals or not isinstance(rr_intervals, list):
            return jsonify({"hrv_interpretation": "Geçerli RR aralığı verisi yok."}), 400

        # Eğer SDNN ve RMSSD istemciden gelmezse, burada temel hesaplamaları yapabiliriz.
        # Bu kısım, isterseniz istemci tarafında da yapılabilir.
        if sdnn is None and len(rr_intervals) > 1:
            rr_array = np.array(rr_intervals)
            sdnn = np.std(rr_array) # Standart sapma
            diff_rr = np.diff(rr_array)
            rmssd = np.sqrt(np.mean(diff_rr**2)) # Karekök ortalama kare farkı

        prompt = (
            f"HRV (Kalp Atış Hızı Değişkenliği) ve RR Aralığı verilerini yorumla:\n"
            f"RR Aralıkları (ms): {rr_intervals[:50]}{'...' if len(rr_intervals) > 50 else ''}\n" # İlk 50 değeri göster
        )

        if sdnn is not None:
            prompt += f"SDNN: {sdnn:.2f} ms\n"
        if rmssd is not None:
            prompt += f"RMSSD: {rmssd:.2f} ms\n"

        prompt += (
            "\nBu HRV ve RR aralıkları verilerini tıbbi ve genel sağlık açısından analiz et.\n"
            "- Otonom sinir sistemi (sempatik ve parasempatik) aktivitesi hakkında ne söylenebilir?\n"
            "- Stres düzeyi, iyileşme durumu veya genel kardiyovasküler sağlık hakkında yorum yap.\n"
            "- Düşük/yüksek SDNN ve RMSSD değerlerinin potansiyel anlamlarını açıkla.\n"
            "- 3-5 cümleyle kısa, anlaşılır ve eyleme geçirilebilir bir yorum yap ve gerekirse bir uzmana danışma öner.\n"
            "Örnek yorum: 'RR aralıklarınızdaki varyasyonlar normal sınırlar içinde görünüyor. SDNN ve RMSSD değerleriniz, otonom sinir sisteminizin iyi dengelendiğini ve stres yönetimi kapasitenizin iyi olduğunu düşündürmektedir. Bu, genel kardiyovasküler sağlığınızın iyi olduğuna işaret eder.'\n"
            "Olumsuz örnek: 'RR aralıklarınız düşük varyasyon gösteriyor. Düşük SDNN ve RMSSD değerleri, artmış stres veya azalmış parasempatik aktivite ile ilişkili olabilir. Dinlenmeye ve stres yönetimine daha fazla odaklanmanız faydalı olabilir, gerekirse bir uzmana danışın.'\n"
        )


        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.7, # HRV yorumu için biraz daha yaratıcı olabilir
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        hrv_interpretation = "".join(block.text for block in response.content if hasattr(block, "text"))
        return jsonify({"hrv_interpretation": hrv_interpretation.strip()})

    except Exception as e:
        print("❌ HRV & RR Sunucu hatası:", str(e))
        return jsonify({"hrv_interpretation": f"Sunucu hatası: {str(e)}"}), 500


# ----------------- SUNUCU BAŞLAT ----------------- #
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)