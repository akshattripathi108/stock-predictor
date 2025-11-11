from flask import Flask, request, jsonify, send_file, send_from_directory
import joblib, os, pandas as pd, io, json
from datetime import timedelta
import numpy as np

    import os
    from flask import send_from_directory

    def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), 'frontend_build', 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(os.path.join(os.path.dirname(__file__), 'frontend_build'), 'index.html')
    return "Flask backend running. Frontend not built."

@app.route('/')
def _root():
    return serve_frontend()

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'model_rf.pkl')
RF_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_rf.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'model_lstm.h5')
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_lstm.pkl')

FEATURE_COLS = ['Close','Volume','SMA_10','SMA_30','Return']
LAGS = 30

def compute_features(df):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    else:
        df['Volume'] = 0
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_30'] = df['Close'].rolling(window=30, min_periods=1).mean()
    df['Return'] = df['Close'].pct_change().fillna(0)
    df = df.dropna().reset_index(drop=True)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    days = int(request.form.get('days', 1))
    # Accept uploaded CSV or fallback to server CSV at ./data/data.csv
    if 'file' in request.files and request.files['file'].filename != '':
        try:
            df = pd.read_csv(request.files['file'])
        except Exception as e:
            return jsonify({'error': 'Invalid CSV: ' + str(e)}), 400
    else:
        fallback = os.path.join(BASE_DIR, 'data', 'data.csv')
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
        else:
            return jsonify({'error': 'No file uploaded and no server CSV found.'}), 400

    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(by=date_col).reset_index(drop=True)
    df = compute_features(df)

    # Prefer LSTM if available and model file exists; otherwise use RF if available
    use_lstm = False
    try:
        import tensorflow as tf  # type: ignore
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
            use_lstm = True
    except Exception:
        use_lstm = False

    if use_lstm:
        # LSTM path
        from tensorflow.keras.models import load_model  # type: ignore
        scaler = joblib.load(LSTM_SCALER_PATH)
        model = load_model(LSTM_MODEL_PATH)
        # prepare sequences (simple sliding window)
        data = df[FEATURE_COLS].values
        scaled = scaler.transform(data)
        if len(scaled) < LAGS:
            return jsonify({'error': f'Not enough rows for LSTM prediction (need at least {LAGS})'}), 400
        seed = scaled[-LAGS:]
        current_seed = seed.copy()
        preds = []
        last_date = df[date_col].iloc[-1] if date_col else pd.Timestamp.today()
        for i in range(days):
            X = current_seed.reshape(1, current_seed.shape[0], current_seed.shape[1])
            pred_scaled = float(model.predict(X)[0,0])
            # inverse transform predicted close: build dummy row copying last row and replace close
            dummy = current_seed[-1].copy()
            dummy[0] = pred_scaled
            inv = scaler.inverse_transform(dummy.reshape(1,-1))[0]
            pred_close = float(inv[0])
            low = pred_close  # no resid estimate for LSTM here
            high = pred_close
            preds.append({'Date': (last_date + timedelta(days=i+1)).strftime('%d/%m/%Y'),
                          'Predicted_Close': pred_close, 'Fluctuation_low': low, 'Fluctuation_high': high})
            # update seed (append scaled predicted close and shift)
            next_row = current_seed[-1].copy()
            next_row[0] = pred_scaled
            current_seed = np.vstack([current_seed[1:], next_row])
    else:
        # RF path
        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(RF_SCALER_PATH):
            return jsonify({'error': 'No RF or LSTM model found in models/. Train first.'}), 400
        model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(RF_SCALER_PATH)
        metrics = {}
        if os.path.exists(METRICS_PATH):
            try:
                metrics = json.load(open(METRICS_PATH))
            except:
                metrics = {}
        resid_std = float(metrics.get('resid_std', 0.0))
        # prepare seed from last LAGS rows
        if len(df) < LAGS:
            return jsonify({'error': f'Not enough rows for RF prediction (need at least {LAGS})'}), 400
        seed_window = df[FEATURE_COLS].iloc[-LAGS:].values.flatten().reshape(1,-1)
        current_seed = scaler.transform(seed_window)
        preds = []
        last_date = df[date_col].iloc[-1] if date_col else pd.Timestamp.today()
        for i in range(days):
            pred = float(model.predict(current_seed)[0])
            low = pred - resid_std
            high = pred + resid_std
            preds.append({'Date': (last_date + timedelta(days=i+1)).strftime('%d/%m/%Y'),
                          'Predicted_Close': pred, 'Fluctuation_low': low, 'Fluctuation_high': high})
            # update seed (inverse transform, shift, append approx new row)
            inv = scaler.inverse_transform(current_seed)[0]
            inv_matrix = inv.reshape(LAGS, len(FEATURE_COLS))
            recent_closes = list(inv_matrix[:,0])
            recent_closes = recent_closes[1:] + [pred]
            sma10 = float(pd.Series(recent_closes[-10:]).mean())
            sma30 = float(pd.Series(recent_closes[-30:]).mean()) if len(recent_closes)>=30 else float(pd.Series(recent_closes).mean())
            ret = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2] if len(recent_closes)>1 else 0.0
            new_row = [pred, float(df['Volume'].iloc[-1]), sma10, sma30, ret]
            inv_matrix = np.vstack([inv_matrix[1:], np.array(new_row).reshape(1,-1)])
            new_seed = inv_matrix.flatten().reshape(1,-1)
            current_seed = scaler.transform(new_seed)

    # prepare history
    history = []
    if date_col:
        history = df[[c for c in df.columns if 'date' in c.lower()][0]].dt.strftime('%d/%m/%Y').tolist()
        history_vals = df['Close'].tolist()
        hist_out = [{'Date': history[i], 'Close': history_vals[i]} for i in range(max(0,len(history)-200), len(history))]
    else:
        hist_out = []

    return jsonify({'history': hist_out, 'predictions': preds})

@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    preds_path = os.path.join(MODEL_DIR, 'predictions_next7.csv')
    if os.path.exists(preds_path):
        return send_file(preds_path, as_attachment=True)
    return jsonify({'error':'Predictions file not found.'}), 404


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
