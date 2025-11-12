# backend_app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory, send_file, abort

# App
app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
RF_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_rf.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_COLS = ["Close", "Volume", "SMA_10", "SMA_30", "Return"]
LAGS = 30

# Helper: compute features for uploaded CSV (client-side CSV should already be normalized as Date,Close,Volume)
def compute_features(df):
    df = df.copy()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.sort_values(by=date_col).reset_index(drop=True)
    # normalize columns
    if "Close" not in df.columns:
        close_col = next((c for c in df.columns if "close" in c.lower()), None)
        if close_col:
            df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        else:
            raise ValueError("No Close column found")
    else:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    else:
        df["Volume"] = 0
    # rolling features
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA_30"] = df["Close"].rolling(window=30, min_periods=1).mean()
    df["Return"] = df["Close"].pct_change().fillna(0)
    df = df.dropna().reset_index(drop=True)
    return df

# Serve the frontend single-page app (frontend_build/index.html)
@app.route("/")
def serve_frontend():
    index_path = os.path.join(BASE_DIR, "frontend_build", "index.html")
    if os.path.exists(index_path):
        return send_from_directory(os.path.join(BASE_DIR, "frontend_build"), "index.html")
    return "Flask backend running. Frontend not built.", 200

# Endpoint to download last saved predictions file (optional)
@app.route("/download_predictions", methods=["GET"])
def download_predictions():
    path = os.path.join(MODEL_DIR, "predictions_next7_rf_combined.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Predictions file not found"}), 404

# Predict endpoint: requires uploaded CSV (form field 'file') and optional 'days' form value
@app.route("/predict", methods=["POST"])
def predict():
    # require a file
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": 'Please upload a CSV file under the "file" form field.'}), 400
    # read CSV
    f = request.files["file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": "Invalid CSV: " + str(e)}), 400

    # compute features
    try:
        df = compute_features(df)
    except Exception as e:
        return jsonify({"error": "Preprocessing error: " + str(e)}), 400

    # load RF model & scaler
    if not (os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH)):
        return jsonify({"error": "Model files not found on server. Please ensure models/ contains model_rf.pkl and scaler_rf.pkl"}), 500

    try:
        model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(RF_SCALER_PATH)
    except Exception as e:
        return jsonify({"error": "Failed to load model or scaler: " + str(e)}), 500

    # optional days parameter
    try:
        days = int(request.form.get("days", 1))
    except Exception:
        days = 1
    if days < 1:
        days = 1
    if days > 30:
        days = 30  # limit to reasonable

    # prepare seed (need at least LAGS rows)
    if len(df) < LAGS:
        return jsonify({"error": f"Not enough rows for prediction (need at least {LAGS} rows after preprocessing)."}), 400

    seed_window = df[FEATURE_COLS].iloc[-LAGS:].values.flatten().reshape(1, -1)
    try:
        current_seed = scaler.transform(seed_window)
    except Exception:
        # scaler may expect different shape; try fallback (no scaling)
        current_seed = seed_window

    # load residual std for fluctuation range if available
    resid_std = 0.0
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as fh:
                m = json.load(fh)
                resid_std = float(m.get("resid_std", 0.0))
        except Exception:
            resid_std = 0.0

    preds = []
    last_date = None
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        last_date = df[date_col].iloc[-1]
    else:
        last_date = pd.Timestamp.today()

    # iterative prediction
    for i in range(days):
        try:
            pred = float(model.predict(current_seed)[0])
        except Exception as e:
            return jsonify({"error": "Model prediction failed: " + str(e)}), 500

        low = pred - resid_std
        high = pred + resid_std
        preds.append({"Date": (last_date + timedelta(days=i+1)).strftime("%d/%m/%Y"),
                      "Predicted_Close": pred, "Fluctuation_low": low, "Fluctuation_high": high})
        # update seed (inverse transform -> shift -> append)
        try:
            inv = scaler.inverse_transform(current_seed)[0]
            inv_matrix = inv.reshape(LAGS, len(FEATURE_COLS))
        except Exception:
            # fall back: reshape current_seed as-is
            inv_matrix = current_seed.reshape(LAGS, len(FEATURE_COLS))
        recent_closes = list(inv_matrix[:, 0])
        recent_closes = recent_closes[1:] + [pred]
        sma10 = float(pd.Series(recent_closes[-10:]).mean())
        sma30 = float(pd.Series(recent_closes[-30:]).mean()) if len(recent_closes) >= 30 else float(pd.Series(recent_closes).mean())
        ret = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2] if len(recent_closes) > 1 else 0.0
        new_row = [pred, float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0, sma10, sma30, ret]
        inv_matrix = np.vstack([inv_matrix[1:], np.array(new_row).reshape(1, -1)])
        new_seed = inv_matrix.flatten().reshape(1, -1)
        try:
            current_seed = scaler.transform(new_seed)
        except Exception:
            current_seed = new_seed

    # prepare history output (last up to 200 rows)
    history = []
    if date_col:
        hist_dates = df[date_col].dt.strftime("%d/%m/%Y").tolist()
        hist_vals = df["Close"].tolist()
        start_idx = max(0, len(hist_dates) - 200)
        history = [{"Date": hist_dates[i], "Close": hist_vals[i]} for i in range(start_idx, len(hist_dates))]

    return jsonify({"history": history, "predictions": preds}), 200


# Run (use PORT env var on Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use debug=False in production
    app.run(host="0.0.0.0", port=port, debug=False)
