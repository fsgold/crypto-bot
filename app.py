# app.py - ربات ترید هوشمند تحت وب
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

# تنظیمات صفحه
st.set_page_config(page_title="🤖 ربات ترید هوشمند", layout="wide")
st.title("🚀 ربات ترید هوشمند ارز دیجیتال")

# --- دریافت داده ---
@st.cache_data(ttl=3600)
def get_data(symbol='BTCUSDT', interval='1h', limit=1000):
    client = Client("", "")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    df['close'] = pd.to_numeric(df['close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- اندیکاتورها ---
def add_indicators(df):
    df = df.copy()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

# --- آموزش مدل ---
@st.cache_resource
def train_model(df):
    features = ['rsi', 'macd_line', 'macd_signal', 'ema20', 'ema50']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, features, acc

# --- سیگنال ---
def generate_signal(row):
    if row['rsi'] < 30 and row['macd_line'] > row['macd_signal'] and row['ema20'] > row['ema50']:
        return '🟢 خرید'
    elif row['rsi'] > 70 and row['macd_line'] < row['macd_signal'] and row['ema20'] < row['ema50']:
        return '🔴 فروش'
    else:
        return '⚪ هولْد'

# --- بک تست ---
def backtest(df):
    df = df.copy()
    df['signal'] = df.apply(generate_signal, axis=1)
    df['return'] = df['close'].pct_change().shift(-1)
    df['strategy_return'] = df['signal'].map({'🟢 خرید': 1, '🔴 فروش': -1, '⚪ هولْد': 0}) * df['return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

# -----------------------------
#       UI
# -----------------------------
symbol = st.selectbox("انتخاب جفت ارز:", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
if st.button("🔄 بروزرسانی") or 'df' not in st.session_state:
    with st.spinner(f"در حال دریافت داده از {symbol}..."):
        df_raw = get_data(symbol=symbol)
        df_ind = add_indicators(df_raw)
        model, features, acc = train_model(df_ind)
        st.session_state.df = df_ind
        st.session_state.model = model
        st.session_state.features = features
        st.session_state.acc = acc
        st.session_state.symbol = symbol

if 'df' in st.session_state:
    df = st.session_state.df
    last_row = df.iloc[-1]
    signal = generate_signal(last_row)

    col1, col2, col3 = st.columns(3)
    col1.metric("آخرین قیمت", f"${last_row['close']:,.2f}")
    col2.metric("RSI", f"{last_row['rsi']:.1f}")
    col3.metric("سیگنال", signal)

    # نمودار
    df_backtest = backtest(df.tail(500))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_backtest.index, df_backtest['cumulative_return'], color='blue')
    ax.set_title(f'بک تست ۶ ماه گذشته - {symbol}')
    ax.grid(True)
    st.pyplot(fig)

    st.success(f"دقت مدل هوش مصنوعی: {st.session_state.acc:.2f}")
