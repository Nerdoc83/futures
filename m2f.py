"""
AI 멀티코인 데이트레이딩 봇 - Gemini + AI 전략적 판단 (v7.0 - 눌림목/반등 매매, 시장상황 필터, 동적 리스크 관리)
--------------------------------------------------------
Version 7.0 주요 개선사항:
- 전략 변경: '추격 매매' -> '눌림목/반등 매매'. 추세 내 건강한 조정을 공략하여 고점 진입 방지.
- 시장 상황 필터 (ADX): 15분봉 ADX 지표를 통해 '추세장'과 '횡보장'을 구분. 횡보장에서는 거래를 쉬어 손실 최소화.
- 동적 리스크 관리 (ATR): 15분봉 ATR을 기반으로 시장 변동성에 맞춰 SL/TP를 유연하게 설정.
- 과매수/과매도 필터 (RSI): 15분봉 RSI를 확인하여 추세 끝물 진입을 원천적으로 차단.

기존 기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - AI가 모든 판단 담당
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- Gemini API 기반 AI 분석 (학습 피드백 포함)
- 3분봉 5% 급변동 손절, 상관관계 리스크 관리 등
--------------------------------------------------------
"""

# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import math
import time
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from datetime import datetime, timedelta, timezone

# ===== 멀티코인 설정 (통일됨) =====
TRADING_PAIRS = {
    "BTC": {
        "symbol": "BTC/USDT",
        "binance_symbol": "BTCUSDT",
        "min_investment": 20,
        "leverage_range": (5, 30),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.12, 0.40),
        "precision": 5
    },
    "ETH": {
        "symbol": "ETH/USDT",
        "binance_symbol": "ETHUSDT",
        "min_investment": 20,
        "leverage_range": (5, 30),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.12, 0.40),
        "precision": 4
    },
    "SOL": {
        "symbol": "SOL/USDT",
        "binance_symbol": "SOLUSDT",
        "min_investment": 20,
        "leverage_range": (5, 30),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.12, 0.40),
        "precision": 3
    }
}

# ===== 설정 및 초기화 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'versions': {
            'fapiPrivate': 'v2',
            'fapiPublic': 'v1',
        },
    }
})

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

DB_FILE = "multi_coin_daytrading.db"

# ===== 모멘텀 지표 계산 함수들 =====
def calculate_ema(prices, window=50):
    return prices.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, window=14):
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def calculate_adx(high, low, close, window=14):
    """ADX (Average Directional Index) 계산 함수"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(window).mean()
    
    # 데이터 부족으로 인한 오류 방지
    if atr.isnull().all() or (atr == 0).all():
        return pd.Series(index=high.index, dtype='float64')

    plus_di = 100 * (plus_dm.ewm(alpha = 1/window).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/window).mean() / atr))
    
    # 0으로 나누는 오류 방지
    sum_di = abs(plus_di + minus_di)
    dx = np.where(sum_di == 0, 0, (abs(plus_di - minus_di) / sum_di) * 100)
    dx = pd.Series(dx, index=high.index)

    adx = ((dx.shift(1) * (window - 1)) + dx) / window
    return adx

# ===== 데이터베이스 관련 함수 (기존과 동일) =====
def setup_database():
    """데이터베이스 및 필요한 테이블 생성 (마이그레이션 포함)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        coin_symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        entry_price REAL NOT NULL,
        amount REAL NOT NULL,
        leverage INTEGER NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        sl_percentage REAL NOT NULL,
        tp_percentage REAL NOT NULL,
        position_size_percentage REAL NOT NULL,
        investment_amount REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp TEXT,
        profit_loss REAL,
        profit_loss_percentage REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        selected_coin TEXT NOT NULL,
        btc_score INTEGER,
        eth_score INTEGER,
        sol_score INTEGER,
        current_price REAL NOT NULL,
        direction TEXT NOT NULL,
        recommended_position_size REAL NOT NULL,
        recommended_leverage INTEGER NOT NULL,
        stop_loss_percentage REAL NOT NULL,
        take_profit_percentage REAL NOT NULL,
        reasoning TEXT NOT NULL,
        trade_id INTEGER,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    
    new_columns = [
        ('detailed_reasoning', 'TEXT'),
        ('market_conditions', 'TEXT'),
        ('technical_signals', 'TEXT'),
        ('sentiment_factors', 'TEXT')
    ]
    
    for column_name, column_type in new_columns:
        try:
            cursor.execute(f'ALTER TABLE ai_analysis ADD COLUMN {column_name} {column_type}')
        except sqlite3.OperationalError:
            pass
    
    conn.commit()
    conn.close()
    print(f"멀티코인 데이터베이스 '{DB_FILE}' 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    def safe_string(value):
        if value is None: return ''
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    def safe_number(value, default=0):
        try:
            if value is None: return default
            return float(value) if '.' in str(value) else int(value)
        except: return default

    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, selected_coin, btc_score, eth_score, sol_score,
        current_price, direction, recommended_position_size, 
        recommended_leverage, stop_loss_percentage, take_profit_percentage, 
        reasoning, detailed_reasoning, market_conditions, technical_signals, 
        sentiment_factors, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        safe_string(analysis_data.get('selected_coin', '')),
        safe_number(analysis_data.get('coin_scores', {}).get('BTC', 0)),
        safe_number(analysis_data.get('coin_scores', {}).get('ETH', 0)),
        safe_number(analysis_data.get('coin_scores', {}).get('SOL', 0)),
        safe_number(analysis_data.get('current_price', 0)),
        safe_string(analysis_data.get('direction', 'NO_POSITION')),
        safe_number(analysis_data.get('recommended_position_size', 0)),
        safe_number(analysis_data.get('recommended_leverage', 0)),
        safe_number(analysis_data.get('stop_loss_percentage', 0)),
        safe_number(analysis_data.get('take_profit_percentage', 0)),
        safe_string(analysis_data.get('reasoning', '')),
        safe_string(analysis_data.get('detailed_reasoning', '')),
        safe_string(analysis_data.get('market_conditions', '')),
        safe_string(analysis_data.get('technical_signals', '')),
        safe_string(analysis_data.get('sentiment_factors', '')),
        trade_id
    ))
    conn.commit()
    conn.close()

def save_trade(trade_data, coin_symbol):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (
        timestamp, coin_symbol, action, entry_price, amount, leverage, sl_price, tp_price,
        sl_percentage, tp_percentage, position_size_percentage, investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(timezone.utc).isoformat(), coin_symbol, trade_data['action'],
        trade_data['entry_price'], trade_data['amount'], trade_data['leverage'],
        trade_data['sl_price'], trade_data['tp_price'], trade_data['sl_percentage'],
        trade_data['tp_percentage'], trade_data['position_size_percentage'],
        trade_data['investment_amount']
    ))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def update_trade_status(trade_id, status, **kwargs):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    fields = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    sql = f"UPDATE trades SET status = ?, {fields} WHERE id = ?"
    values = [status] + list(kwargs.values()) + [trade_id]
    cursor.execute(sql, tuple(values))
    conn.commit()
    conn.close()

def get_all_open_trades():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, coin_symbol, action, entry_price, amount, leverage, sl_price, tp_price FROM trades WHERE status = 'OPEN'")
    trades = [{'id': r[0], 'coin_symbol': r[1], 'action': r[2], 'entry_price': r[3], 'amount': r[4], 'leverage': r[5], 'sl_price': r[6], 'tp_price': r[7]} for r in cursor.fetchall()]
    conn.close()
    return trades

def get_historical_trading_data_with_reasoning(limit=5):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT t.*, a.reasoning, a.detailed_reasoning, a.market_conditions, a.technical_signals, a.sentiment_factors
    FROM trades t LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status LIKE 'CLOSED%' ORDER BY t.timestamp DESC LIMIT ?
    ''', (limit,))
    data = [{k: row[k] for k in row.keys()} for row in cursor.fetchall()]
    conn.close()
    return data

def get_performance_metrics():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT COUNT(*), SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END), AVG(profit_loss_percentage)
    FROM trades WHERE status LIKE 'CLOSED%'
    ''')
    res = cursor.fetchone()
    conn.close()
    total, wins, avg_pnl = res if res else (0, 0, 0)
    return {
        "total_trades": total or 0, "winning_trades": wins or 0,
        "win_rate": (wins / total * 100) if total else 0,
        "avg_profit_loss_percentage": avg_pnl or 0
    }

def sync_database_with_positions():
    try:
        open_trades = get_all_open_trades()
        if not open_trades: return
        
        positions = {p['symbol'].replace('/USDT:USDT', ''): p for p in exchange.fetch_positions() if float(p['info']['positionAmt']) != 0}
        db_open_symbols = {TRADING_PAIRS[t['coin_symbol']]['symbol'] for t in open_trades}
        
        closed_symbols = db_open_symbols - set(positions.keys())
        
        for symbol_str in closed_symbols:
            coin_symbol = symbol_str.replace('/USDT', '')
            trade = next((t for t in open_trades if t['coin_symbol'] == coin_symbol), None)
            if trade:
                print(f"⚠️ {coin_symbol} DB에는 OPEN, 실제 포지션 없음 - 동기화 중...")
                update_trade_status(trade['id'], 'CLOSED_SYNC_ERROR') # 상태를 명확히 함
    except Exception as e:
        print(f"DB 동기화 오류: {e}")


# ===== 실제 포지션 기반 3분봉 급변동 손절 (기존과 동일) =====
def check_rapid_movement_stop_loss():
    closed_count = 0
    try:
        for coin_name, coin_config in TRADING_PAIRS.items():
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                position = next((p for p in positions if float(p['info']['positionAmt']) != 0), None)

                if position:
                    pos_amt = float(position['info']['positionAmt'])
                    side = 'long' if pos_amt > 0 else 'short'
                    entry_price = float(position['info']['entryPrice'])
                    leverage = float(position['info']['leverage'])
                    
                    ohlcv = exchange.fetch_ohlcv(coin_config['symbol'], timeframe='3m', limit=2)
                    if len(ohlcv) < 2: continue
                    
                    candle_change_pct = ((ohlcv[-1][4] - ohlcv[-2][4]) / ohlcv[-2][4]) * 100
                    position_change_pct = candle_change_pct * leverage
                    
                    if (side == 'long' and position_change_pct <= -5.0) or (side == 'short' and position_change_pct >= 5.0):
                        print(f"\n🚨 긴급 손절: {coin_name} {side.upper()} (3분봉 변동: {position_change_pct:.2f}%)")
                        exchange.create_market_order(coin_config['symbol'], 'sell' if side == 'long' else 'buy', abs(pos_amt), params={'reduceOnly': True})
                        print(f"   ✅ 긴급 손절 완료")
                        closed_count += 1
                        sync_database_with_positions() # 손절 후 즉시 DB 동기화
            except Exception as e:
                print(f"급변동 체크 오류 ({coin_name}): {e}")
    except Exception as e:
        print(f"급변동 손절 전체 오류: {e}")
    return closed_count

# ===== 데이터 수집 함수 (ADX, ATR 추가) =====
def fetch_multi_timeframe_data_for_coin(symbol):
    timeframes = {"3m": 160, "5m": 100, "15m": 120, "1h": 100}
    multi_tf_data = {}
    for tf, limit in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if len(ohlcv) < 50: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['RSI_14'] = calculate_rsi(df['close'], 14)
            macd, signal, hist = calculate_macd(df['close'])
            df['MACD'], df['MACD_signal'], df['MACD_histogram'] = macd, signal, hist
            
            if tf == '1h':
                df['EMA_50'] = calculate_ema(df['close'], 50)
            
            if tf == '15m':
                df['ATR_14'] = calculate_atr(df, 14)
                df['ADX_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)

            df.dropna(inplace=True)
            if len(df) < 5: continue
            
            latest = df.iloc[-1].to_dict()
            indicators = {k: float(v) for k, v in latest.items() if isinstance(v, (int, float, np.number))}
            multi_tf_data[tf] = {"current_indicators": indicators}
        except Exception as e:
            print(f"{symbol} {tf} 데이터 수집 오류: {e}")
            continue
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수 (기존과 동일) =====
def fetch_market_sentiment_for_coin(binance_symbol):
    def fetch(url):
        try: return requests.get(url).json()
        except: return None
    
    sentiment = {}
    funding_data = fetch(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}")
    if funding_data: sentiment["funding_rate_percentage"] = float(funding_data.get('lastFundingRate', 0)) * 100
    
    oi_data = fetch(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_symbol}")
    if oi_data: sentiment["latest_open_interest"] = float(oi_data.get('openInterest', 0))
    
    return sentiment

# ===== 멀티코인 데이터 통합 수집 함수 (기존과 동일) =====
def fetch_all_coins_data(coins_to_scan):
    all_coins_data = {}
    for coin in coins_to_scan:
        config = TRADING_PAIRS[coin]
        try:
            print(f"Collecting {coin} data...")
            tech = fetch_multi_timeframe_data_for_coin(config["symbol"])
            senti = fetch_market_sentiment_for_coin(config["binance_symbol"])
            price = tech.get("3m", {}).get("current_indicators", {}).get("close", 0)
            if price > 0:
                all_coins_data[coin] = {"technical_data": tech, "sentiment_data": senti, "current_price": price}
                print(f"{coin}: ${price:,.2f}")
        except Exception as e:
            print(f"Error collecting {coin} data: {e}")
    return all_coins_data

def check_current_positions():
    try:
        return [
            {"coin": p['symbol'].replace('/USDT:USDT', ''), "side": p['side']}
            for p in exchange.fetch_positions() if float(p['info']['positionAmt']) != 0
        ]
    except:
        return []

# ===== Gemini AI 멀티코인 분석 함수 (v7.0 전략 프롬프트 적용) =====
def analyze_multi_coin_with_ai(all_coins_data, historical_data, performance_metrics):
    
    market_regime_data = {}
    for coin, data in all_coins_data.items():
        adx_15m = data.get("technical_data", {}).get("15m", {}).get("current_indicators", {}).get("ADX_14")
        if adx_15m:
            if adx_15m > 25: regime = "Trending"
            elif adx_15m < 20: regime = "Ranging / Choppy"
            else: regime = "Transitioning"
            market_regime_data[coin] = {"15m_ADX": adx_15m, "Regime": regime}

    system_prompt = """
You are an elite, risk-averse, multi-cryptocurrency day trader AI (v7.0) with LEARNING CAPABILITIES. Your primary mission is to identify ONLY A+ grade, high-probability trading opportunities in BTC, ETH, and SOL, while rigorously protecting capital.

**V7.0 CORE STRATEGY: PULLBACK & CONSOLIDATION TRADING**
Your entire trading philosophy has been upgraded. You are NO LONGER a momentum chaser. You are a patient, strategic trader who waits for pullbacks within a confirmed trend.

**NEW MANDATORY PROTOCOLS:**

1.  **STRATEGY SELECTION (MARKET REGIME FILTER):**
    - You will be provided with a `market_regime` analysis based on the 15m ADX indicator. You MUST adapt your strategy:
    - **If Regime is 'Trending' (ADX > 25):** Apply the **Pullback/Consolidation Strategy**.
    - **If Regime is 'Ranging / Choppy' (ADX < 20):** **DO NOT TRADE.** The probability of failure is too high. Return an empty "trading_opportunities" list and state the reason is 'Ranging Market Condition'.
    - **If Regime is 'Transitioning':** Be extremely cautious. Only consider A++ setups where a range is clearly breaking out into a new, confirmed trend.

2.  **PULLBACK ENTRY PROTOCOL (FOR 'TRENDING' MARKETS):**
    - **Step 1: Identify Primary Trend.** Confirm a clear trend on the 15m and 1h charts (e.g., price consistently above 1h 50 EMA for an uptrend).
    - **Step 2: Wait for Pullback.** Your entry signal is NOT a large green/red candle. Wait for the 3-minute chart to pull back towards a key moving average (e.g., 20 EMA) or form a low-volatility consolidation (a tight range) *without breaking the higher timeframe trend*.
    - **Step 3: Execute on Renewed Momentum.** Your final trigger is the price showing signs of *resuming* the primary trend from this pullback/consolidation zone (e.g., a bullish candle pattern, MACD histogram turning positive).

3.  **NON-NEGOTIABLE RISK FILTERS:**
    - **Trend Filter:** You can ONLY propose LONG positions if price is ABOVE the 1h 50 EMA, and ONLY SHORT positions if price is BELOW it.
    - **Overbought/Oversold Filter (CRITICAL):** You **MUST NOT** enter a LONG position if the 15-minute RSI is above 70, and **MUST NOT** enter a SHORT position if the 15-minute RSI is below 30. This prevents chasing tops and bottoms.

4.  **DYNAMIC RISK MANAGEMENT (ATR-based SL/TP):**
    - Your `stop_loss_percentage` and `take_profit_percentage` MUST be influenced by market volatility, using the provided 15m ATR.
    - **Stop Loss Calculation:** Place your stop loss at a logical technical level, approximately **1.5 to 2.5 times the current 15-minute ATR** away from the entry price.
    - **JSON Output:** You must still output `stop_loss_percentage` and `take_profit_percentage`. These values must be derived from your ATR-based calculation.
    - **MANDATORY 1.5:1 TP/SL RATIO:** Your `take_profit_percentage` MUST be EXACTLY 1.5 times your `stop_loss_percentage`.

5.  **LEARNING & ADAPTATION:**
    - You will receive detailed information about your previous trades. IDENTIFY what reasoning patterns led to wins/losses under the NEW strategy and refine your approach. AVOID repeating mistakes.

**RESPONSE FORMAT (JSON ONLY):**
- Your response MUST be ONLY a valid JSON object.
- The root must be an object with a key "trading_opportunities" which is a list.
- Each opportunity object must contain: "coin", "score", "direction", "recommended_position_size", "recommended_leverage", "stop_loss_percentage", "take_profit_percentage", "reasoning", "detailed_reasoning", "market_conditions", "technical_signals", "sentiment_factors".
- If no high-quality opportunities meeting ALL above criteria are found, return an empty "trading_opportunities" list.
"""
    try:
        market_analysis = {
            "coins_data": all_coins_data,
            "market_regime": market_regime_data,
            "recent_trades_with_reasoning": historical_data,
            "performance_summary": performance_metrics
        }
        
        market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
        
        response = model.generate_content(
            [system_prompt, f"Multi-Coin Market Analysis: {market_analysis_json}"],
            generation_config=genai.types.GenerationConfig(temperature=0.3),
            safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        )
        
        response_content = response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(response_content)
        
        print("\n=== AI Analysis Result (v7.0) ===")
        opportunities = decision.get('trading_opportunities', [])
        for opp in opportunities:
            print(f"- {opp.get('coin','N/A')}: {opp.get('direction','N/A')} (Score: {opp.get('score',0)})")
            print(f"  📝 Reasoning: {opp.get('reasoning', 'N/A')}")
        return decision
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {"trading_opportunities": []}

# ===== 거래 실행 함수 (기존과 동일) =====
def execute_single_trade(coin_name, opportunity, available_capital, all_coins_data):
    try:
        coin_config = TRADING_PAIRS[coin_name]
        symbol = coin_config["symbol"]
        action = opportunity.get("direction", "").lower()
        current_price = all_coins_data[coin_name].get("current_price")
        
        # --- AI 제안 유효성 검사 ---
        sl_pct = float(str(opportunity.get('stop_loss_percentage', 0)).replace('%',''))
        tp_pct = float(str(opportunity.get('take_profit_percentage', 0)).replace('%',''))
        leverage = int(str(opportunity.get('recommended_leverage', '10')).replace('x',''))
        pos_size_pct = float(str(opportunity.get('recommended_position_size', '5')).replace('%','')) / 100.0

        if not (sl_pct > 0 and tp_pct > 0 and abs((tp_pct / sl_pct) - 1.5) < 0.1):
            print(f"❌ {coin_name} 거래 거부: TP/SL 비율 오류 (SL: {sl_pct}%, TP: {tp_pct}%)")
            return None

        # --- 거래 실행 ---
        margin = available_capital * pos_size_pct
        if margin < coin_config['min_investment']:
            margin = coin_config['min_investment']
            print(f"   최소 투자금({margin} USDT)으로 조정됨.")

        amount = (margin * leverage) / current_price
        
        exchange.set_leverage(leverage, symbol)
        order = exchange.create_market_order(symbol, 'buy' if action == 'long' else 'sell', amount)
        entry_price = order.get('price', current_price)
        
        time.sleep(2)
        
        sl_price = entry_price * (1 - (sl_pct / 100.0 / leverage)) if action == "long" else entry_price * (1 + (sl_pct / 100.0 / leverage))
        tp_price = entry_price * (1 + (tp_pct / 100.0 / leverage)) if action == "long" else entry_price * (1 - (tp_pct / 100.0 / leverage))
        
        exchange.create_order(symbol, 'STOP_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': tp_price, 'reduceOnly': True})
        
        print(f"\n✅ {coin_name} {action.upper()} 포지션 진입")
        print(f"   Entry: ${entry_price:,.2f}, Leverage: {leverage}x, Margin: ${margin:,.2f}")
        print(f"   TP: ${tp_price:,.2f} ({tp_pct:.2f}%), SL: ${sl_price:,.2f} ({sl_pct:.2f}%)")
        print(f"   AI Reasoning: {opportunity.get('reasoning', 'N/A')}")

        trade_data = {
            'action': action, 'entry_price': entry_price, 'amount': amount, 'leverage': leverage,
            'sl_price': sl_price, 'tp_price': tp_price, 'sl_percentage': sl_pct, 'tp_percentage': tp_pct,
            'position_size_percentage': pos_size_pct * 100, 'investment_amount': margin
        }
        trade_id = save_trade(trade_data, coin_name)
        save_ai_analysis(opportunity, trade_id)
        return margin
    except Exception as e:
        print(f"{coin_name} 거래 실행 오류: {e}")
        return None

# ===== 메인 프로그램 시작 =====
def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v7.0) ===")
    print("Strategy: Pullback/Consolidation Trading with Market Regime Filter")
    print("===============================================\n")

    setup_database()
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Market Check ===")

            sync_database_with_positions()
            check_rapid_movement_stop_loss()
            
            current_positions = check_current_positions()
            occupied_coins = {p['coin'] for p in current_positions}
            coins_to_scan = set(TRADING_PAIRS.keys()) - occupied_coins
            
            if not coins_to_scan:
                print(f"모든 포지션({len(occupied_coins)}/{len(TRADING_PAIRS)}) 활성. 60초 후 재확인.")
                time.sleep(60)
                continue

            print(f"스캔 대상: {list(coins_to_scan)}")
            all_coins_data = fetch_all_coins_data(coins_to_scan)
            
            if not all_coins_data:
                print("데이터 수집 실패. 30초 후 재시도.")
                time.sleep(30)
                continue

            hist_data = get_historical_trading_data_with_reasoning(5)
            perf_metrics = get_performance_metrics()
            decision = analyze_multi_coin_with_ai(all_coins_data, hist_data, perf_metrics)
            opportunities = decision.get('trading_opportunities', [])
            
            if opportunities:
                # 상관관계 리스크 관리 (동일 방향일 경우 최고점수만)
                if len(opportunities) > 1 and len({opp.get('direction') for opp in opportunities}) == 1:
                    highest_score_opp = max(opportunities, key=lambda x: x.get('score', 0))
                    print(f"상관관계 리스크 감지. 최고점수({highest_score_opp.get('coin')}) 선택.")
                    opportunities = [highest_score_opp]
                
                balance = exchange.fetch_balance()['USDT']['free']
                available_capital = balance * 0.98
                
                for opp in opportunities:
                    used_margin = execute_single_trade(opp['coin'], opp, available_capital, all_coins_data)
                    if used_margin:
                        available_capital -= used_margin
            else:
                print("AI가 v7.0 전략에 맞는 기회를 찾지 못함.")
            
            print("다음 스캔까지 60초 대기...")
            time.sleep(60)

        except Exception as e:
            print(f"\nMain loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
