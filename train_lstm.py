\"\"\"Train an LSTM model on a CSV file.
Usage: python train_lstm.py --file path/to/data.csv --epochs 20 --window 60
Saves model to models/model_lstm.h5 and scaler to models/scaler_lstm.pkl
\"\"\"
import argparse, os, joblib
import pandas as pd, numpy as np

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

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

def main(args):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as e:
        print('TensorFlow not available in this environment. Install tensorflow to train LSTM. Error:', e)
        return

    df = pd.read_csv(args.file)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(by=date_col).reset_index(drop=True)

    df = compute_features(df)
    FEATURE_COLS = ['Close','Volume','SMA_10','SMA_30','Return']
    data = df[FEATURE_COLS].values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, window=args.window)
    split = int(0.85*len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=32, callbacks=[es])

    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models','model_lstm.h5'))
    joblib.dump(scaler, os.path.join('models','scaler_lstm.pkl'))
    print('Saved LSTM model to models/model_lstm.h5 and scaler to models/scaler_lstm.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window', type=int, default=60)
    args = parser.parse_args()
    main(args)
