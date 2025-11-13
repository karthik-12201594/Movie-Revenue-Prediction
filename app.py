# app.py
from flask import Flask, request, jsonify, render_template
import os, joblib, pandas as pd

app = Flask(__name__)

# ====== Paths (adjust if your structure differs) ======
BASE = os.path.abspath(".")
MODEL_PATH = os.path.join(BASE, "artifacts", "models", "random_forest.joblib")
PREPROCESSOR_PATH = os.path.join(BASE, "artifacts", "transformer", "preprocessor.joblib")

# ====== Load artifacts ======
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ====== Language mapping (code -> full name) ======
LANGUAGE_FULL = {
    "en": "English", "hi": "Hindi", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
    "da": "Danish", "el": "Greek", "nl": "Dutch", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam"
}
LANGUAGES = list(LANGUAGE_FULL.keys())

# Example genre list (keep or replace with your real values)
GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Horror",
    "Romance", "Thriller", "Science Fiction", "Fantasy",
]

# ====== Helpers ======
def safe_float(x):
    try: return float(x)
    except: return 0.0

def safe_int(x):
    try: return int(x)
    except: return 0

def build_raw_from_payload(payload):
    """Construct the raw row expected by preprocessor from partial payload dict."""
    raw = {
        "id": 0,
        "title": payload.get("title", ""),
        "vote_average": safe_float(payload.get("vote_average", 0)),
        "vote_count": safe_int(payload.get("vote_count", 0)),
        "status": "Released",
        "release_date": "",
        "runtime": safe_float(payload.get("runtime", 0)),
        "budget": safe_float(payload.get("budget", 0)),
        "original_language": payload.get("original_language", ""),
        "original_title": payload.get("title", ""),
        "overview": "",
        "genres": payload.get("genres", ""),
        "production_companies": "",
        "production_countries": ""
    }
    # compose release_date if parts provided
    y = payload.get("release_year", "")
    m = payload.get("release_month", "")
    d = payload.get("release_day", "")
    if y and m and d:
        try:
            raw["release_date"] = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        except:
            raw["release_date"] = ""
    return raw

# ====== Routes ======
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        revenue_millions=None,
        roi=None,
        category=None,
        category_color=None,
        sample=None,
        genres=GENRES,
        languages=LANGUAGES,
        LANGUAGE_FULL=LANGUAGE_FULL
    )

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
      - JSON AJAX requests (Content-Type: application/json)
      - Form submissions (application/x-www-form-urlencoded or multipart/form-data)
      - A form field named 'input_json' that contains JSON text
    Returns:
      - JSON when the request was JSON (for AJAX)
      - rendered HTML when the request was a form submit
    """
    try:
        is_json_req = request.is_json

        # 1) Prefer proper JSON (AJAX)
        payload = None
        if request.is_json:
            # get_json(silent=True) returns None instead of raising if body isn't valid JSON
            payload = request.get_json(silent=True)

        # 2) If no JSON, look for an 'input_json' field in form (some forms use this)
        if payload is None:
            raw_input_json = request.form.get("input_json")
            if raw_input_json:
                try:
                    import json
                    payload = json.loads(raw_input_json)
                except Exception:
                    # ignore parse error and fallback to form fields
                    payload = None

        # 3) Fall back to regular form fields (common when browser does a normal POST)
        if payload is None:
            # Collect expected keys from form
            form = request.form
            # If the form is empty, also check request.data as a last resort
            if not form or len(form) == 0:
                # try to parse body as JSON silently (some clients set wrong header)
                payload = request.get_json(silent=True)
                if payload is None:
                    # last resort: parse urlencoded body manually
                    # but usually form will have data
                    payload = {}
            else:
                payload = {
                    "title": form.get("title", ""),
                    "budget": form.get("budget", ""),
                    "runtime": form.get("runtime", ""),
                    "vote_average": form.get("vote_average", ""),
                    "vote_count": form.get("vote_count", ""),
                    "original_language": form.get("original_language", ""),
                    "genres": form.get("genres", ""),
                    "release_year": form.get("release_year", ""),
                    "release_month": form.get("release_month", ""),
                    "release_day": form.get("release_day", "")
                }

        # Now payload should be a dict or list
        if isinstance(payload, list):
            results = []
            for item in payload:
                raw = build_raw_from_payload(item)
                # ensure DataFrame columns match preprocessor
                expected = list(preprocessor.feature_names_in_)
                ordered = {c: raw.get(c, "") for c in expected}
                df = pd.DataFrame([ordered], columns=expected)
                X = preprocessor.transform(df)
                pred = float(model.predict(X)[0])
                results.append(pred)
            if is_json_req:
                return jsonify({"predictions": results})
            # form submit: show first
            first = results[0] if results else None
            return render_template("index.html",
                                   result=f"${first:,.2f}",
                                   revenue_millions=(round(first/1e6,2) if first else None),
                                   roi=None,
                                   category=None,
                                   category_color=None,
                                   sample=None,
                                   genres=GENRES,
                                   languages=LANGUAGES,
                                   LANGUAGE_FULL=LANGUAGE_FULL)

        if isinstance(payload, dict):
            raw = build_raw_from_payload(payload)
            expected = list(preprocessor.feature_names_in_)
            ordered = {c: raw.get(c, "") for c in expected}
            df = pd.DataFrame([ordered], columns=expected)
            X = preprocessor.transform(df)
            pred = float(model.predict(X)[0])

            if is_json_req:
                return jsonify({"prediction": pred})

            # form submit -> render html page
            result_str = f"${pred:,.2f}"
            revenue_millions = round(pred/1e6,2)
            budget_v = safe_float(payload.get("budget", 0))
            roi = round((pred - budget_v) / budget_v * 100, 2) if budget_v > 0 else None
            category = "Blockbuster" if pred >= 150_000_000 else ("Hit" if pred >= 50_000_000 else ("Moderate" if pred >= 10_000_000 else "Low"))
            sample = {
                "title": payload.get("title", ""),
                "budget": payload.get("budget", ""),
                "runtime": payload.get("runtime", ""),
                "vote_average": payload.get("vote_average", ""),
                "vote_count": payload.get("vote_count", ""),
                "release_date": payload.get("release_date", ""),
                "language": payload.get("original_language", ""),
                "genre": payload.get("genres", "")
            }
            return render_template("index.html",
                                   result=result_str,
                                   revenue_millions=revenue_millions,
                                   roi=roi,
                                   category=category,
                                   category_color=None,
                                   sample=sample,
                                   genres=GENRES,
                                   languages=LANGUAGES,
                                   LANGUAGE_FULL=LANGUAGE_FULL)

        # Unknown payload type
        if is_json_req:
            return jsonify({"error":"Invalid payload type; must be JSON object or list"}), 400
        return render_template("index.html", error="Invalid payload type"), 400

    except Exception as e:
        # Log traceback to console (helpful in dev)
        import traceback; traceback.print_exc()
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template("index.html", error=str(e), genres=GENRES, languages=LANGUAGES, LANGUAGE_FULL=LANGUAGE_FULL), 500


# ====== Run ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
