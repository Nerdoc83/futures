"""
AI 멀티코인 데이트레이딩 봇 - Gemini + 고급 지표 (v4.0 - Multi-Position + 1M + 4H Timeout)
--------------------------------------------------------
기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - 최고 확률 기회 선택 또는 동시 매수
- 데이트레이딩 최적화 (1분, 5분, 15분, 1시간 차트)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini API 기반 AI 분석
- 동적 레버리지 및 포지션 사이징 (3-50배)
- 개선된 SL/TP 설정 (원금 대비 8-70% 범위)
- 4시간 타임아웃 (수익 포지션 자동 정리)
- 24시간 무제한 거래
- 최소 투자금액: 50-100 USDT
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
from datetime import datetime

# ===== 멀티코인 설정 =====
TRADING_PAIRS = {
    "BTC": {
        "symbol": "BTC/USDT",
        "binance_symbol": "BTCUSDT",
        "min_investment": 100,
        "leverage_range": (5, 50),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.15, 0.50),
        "precision": 5
    },
    "ETH": {
        "symbol": "ETH/USDT", 
        "binance_symbol": "ETHUSDT",
        "min_investment": 100,
        "leverage_range": (5, 35),
        "volatility_factor": 1.3,
        "sl_range": (0.10, 0.30),
        "tp_range": (0.20, 0.60),
        "precision": 4
    },
    "SOL": {
        "symbol": "SOL/USDT",
        "binance_symbol": "SOLUSDT", 
        "min_investment": 50,
        "leverage_range": (3, 25),
        "volatility_factor": 1.8,
        "sl_range": (0.12, 0.35),
        "tp_range": (0.25, 0.70),
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
        'adjustForTimeDifference': True
    }
})

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

DB_FILE = "multi_coin_daytrading.db"

# ===== 모멘텀 지표 계산 함수들 =====
def calculate_rsi(prices, window=14):
    """RSI 계산 함수"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산 함수"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator 계산"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, window=14):
    """Williams %R 계산"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_atr(df, window=14):
    """Average True Range 계산"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# ===== 데이터베이스 관련 함수 =====
def setup_database():
    """데이터베이스 및 필요한 테이블 생성"""
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
    
    conn.commit()
    conn.close()
    print("멀티코인 데이터베이스 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, selected_coin, btc_score, eth_score, sol_score,
        current_price, direction, recommended_position_size, 
        recommended_leverage, stop_loss_percentage, take_profit_percentage, 
        reasoning, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        analysis_data.get('selected_coin', ''),
        analysis_data.get('coin_scores', {}).get('BTC', 0),
        analysis_data.get('coin_scores', {}).get('ETH', 0),
        analysis_data.get('coin_scores', {}).get('SOL', 0),
        analysis_data.get('current_price', 0),
        analysis_data.get('direction', 'NO_POSITION'),
        analysis_data.get('recommended_position_size', 0),
        analysis_data.get('recommended_leverage', 0),
        analysis_data.get('stop_loss_percentage', 0),
        analysis_data.get('take_profit_percentage', 0),
        analysis_data.get('reasoning', ''),
        trade_id
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def save_trade(trade_data, coin_symbol):
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp, coin_symbol, action, entry_price, amount, leverage, sl_price, tp_price,
        sl_percentage, tp_percentage, position_size_percentage, investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        coin_symbol,
        trade_data.get('action', ''),
        trade_data.get('entry_price', 0),
        trade_data.get('amount', 0),
        trade_data.get('leverage', 0),
        trade_data.get('sl_price', 0),
        trade_data.get('tp_price', 0),
        trade_data.get('sl_percentage', 0),
        trade_data.get('tp_percentage', 0),
        trade_data.get('position_size_percentage', 0),
        trade_data.get('investment_amount', 0)
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def update_trade_status(trade_id, status, exit_price=None, exit_timestamp=None, profit_loss=None, profit_loss_percentage=None):
    """거래 상태를 업데이트합니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    update_fields = ["status = ?"]
    update_values = [status]
    
    if exit_price is not None:
        update_fields.append("exit_price = ?")
        update_values.append(exit_price)
    
    if exit_timestamp is not None:
        update_fields.append("exit_timestamp = ?")
        update_values.append(exit_timestamp)
    
    if profit_loss is not None:
        update_fields.append("profit_loss = ?")
        update_values.append(profit_loss)
    
    if profit_loss_percentage is not None:
        update_fields.append("profit_loss_percentage = ?")
        update_values.append(profit_loss_percentage)
    
    update_sql = f"UPDATE trades SET {', '.join(update_fields)} WHERE id = ?"
    update_values.append(trade_id)
    
    cursor.execute(update_sql, update_values)
    conn.commit()
    conn.close()

def get_all_open_trades():
    """모든 열린 거래 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, coin_symbol, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    open_trades = []
    for result in results:
        open_trades.append({
            'id': result[0],
            'coin_symbol': result[1],
            'action': result[2],
            'entry_price': result[3],
            'amount': result[4],
            'leverage': result[5],
            'sl_price': result[6],
            'tp_price': result[7]
        })
    
    return open_trades

def get_trade_summary(days=7):
    """지정된 일수 동안의 거래 요약 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage
    FROM trades
    WHERE exit_timestamp IS NOT NULL
    AND timestamp >= datetime('now', ?)
    ''', (f'-{days} days',))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'total_trades': result[0] or 0,
            'winning_trades': result[1] or 0,
            'losing_trades': result[2] or 0,
            'total_profit_loss': result[3] or 0,
            'avg_profit_loss_percentage': result[4] or 0
        }
    return None

def get_historical_trading_data(limit=5):
    """과거 거래 내역 가져오기"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.coin_symbol, t.action, t.entry_price, t.exit_price, t.leverage,
        t.profit_loss_percentage, a.reasoning
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status = 'CLOSED'
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        historical_data.append({k: row[k] for k in row.keys()})
    
    conn.close()
    return historical_data

def get_performance_metrics():
    """거래 성과 메트릭스를 계산합니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage,
        MAX(profit_loss_percentage) as max_profit_percentage,
        MIN(profit_loss_percentage) as max_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    ''')
    
    overall_metrics = cursor.fetchone()
    conn.close()
    
    metrics = {
        "total_trades": overall_metrics[0] or 0,
        "winning_trades": overall_metrics[1] or 0,
        "avg_profit_loss_percentage": overall_metrics[2] or 0,
        "max_profit_percentage": overall_metrics[3] or 0,
        "max_loss_percentage": overall_metrics[4] or 0
    }
    
    if metrics["total_trades"] > 0:
        metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
    else:
        metrics["win_rate"] = 0
    
    return metrics

def check_trade_timeout():
    """4시간이 지난 수익 거래를 확인하고 타임아웃 처리"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, coin_symbol, action, entry_price, amount, leverage
    FROM trades
    WHERE status = 'OPEN' 
    AND datetime(timestamp) <= datetime('now', '-4 hours')
    ''')
    
    timeout_trades = cursor.fetchall()
    conn.close()
    
    timeout_count = 0
    
    for trade in timeout_trades:
        trade_id, coin_symbol, action, entry_price, amount, leverage = trade
        
        try:
            coin_config = TRADING_PAIRS.get(coin_symbol)
            if not coin_config:
                continue
                
            current_price = exchange.fetch_ticker(coin_config['symbol'])['last']
            
            is_profitable = False
            if action == 'long' and current_price > entry_price:
                is_profitable = True
            elif action == 'short' and current_price < entry_price:
                is_profitable = True
            
            if is_profitable:
                print(f"\n=== 4시간 타임아웃: {coin_symbol} {action.upper()} 수익 포지션 정리 ===")
                print(f"Entry: ${entry_price:,.2f}, Current: ${current_price:,.2f}")
                
                try:
                    open_orders = exchange.fetch_open_orders(coin_config['symbol'])
                    for order in open_orders:
                        exchange.cancel_order(order['id'], coin_config['symbol'])
                    print(f"기존 SL/TP 주문 취소됨")
                except Exception as e:
                    print(f"주문 취소 오류: {e}")
                
                try:
                    if action == 'long':
                        exchange.create_market_sell_order(coin_config['symbol'], amount)
                    else:
                        exchange.create_market_buy_order(coin_config['symbol'], amount)
                    
                    if action == 'long':
                        profit_loss = (current_price - entry_price) * amount
                        profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
                    else:
                        profit_loss = (entry_price - current_price) * amount
                        profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                    
                    update_trade_status(
                        trade_id,
                        'CLOSED_TIMEOUT',
                        exit_price=current_price,
                        exit_timestamp=datetime.now().isoformat(),
                        profit_loss=profit_loss,
                        profit_loss_percentage=profit_loss_percentage
                    )
                    
                    print(f"타임아웃 정리 완료: P/L ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                    timeout_count += 1
                    
                except Exception as e:
                    print(f"포지션 정리 오류: {e}")
            else:
                print(f"{coin_symbol} {action} 포지션이 손실 상태이므로 타임아웃 처리하지 않음")
                
        except Exception as e:
            print(f"타임아웃 체크 오류 ({coin_symbol}): {e}")
    
    if timeout_count > 0:
        print("=== 타임아웃 정리 완료, 새로운 스캔을 시작합니다 ===")
    
    return timeout_count

# ===== 데이터 수집 함수 =====
def fetch_multi_timeframe_data_for_coin(symbol):
    """단일 코인의 멀티 타임프레임 데이터 수집"""
    timeframes = {
        "1m": {"timeframe": "1m", "limit": 120},
        "5m": {"timeframe": "5m", "limit": 100},
        "15m": {"timeframe": "15m", "limit": 96},
        "1h": {"timeframe": "1h", "limit": 48}
    }
    
    multi_tf_data = {}
    coin_name = symbol.split('/')[0]
    
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf_params["timeframe"], 
                limit=tf_params["limit"]
            )
            
            if not ohlcv or len(ohlcv) < 20:
                print(f"Insufficient {tf_name} data for {symbol}")
                continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 모멘텀 지표 계산
            df['RSI_14'] = calculate_rsi(df['close'], window=14)
            df['RSI_21'] = calculate_rsi(df['close'], window=21)
            
            macd, signal, histogram = calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_histogram'] = histogram
            
            stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
            df['ATR'] = calculate_atr(df)
            
            df.dropna(inplace=True)
            
            if len(df) < 5:
                print(f"Insufficient valid {tf_name} data for {symbol} after indicator calculation")
                continue
            
            latest_row = df.iloc[-1]
            current_indicators = {
                "current_price": float(latest_row['close']),
                "rsi_14": float(latest_row['RSI_14']),
                "rsi_21": float(latest_row['RSI_21']),
                "macd": float(latest_row['MACD']),
                "macd_signal": float(latest_row['MACD_signal']),
                "macd_histogram": float(latest_row['MACD_histogram']),
                "stoch_k": float(latest_row['Stoch_K']),
                "stoch_d": float(latest_row['Stoch_D']),
                "williams_r": float(latest_row['Williams_R']),
                "atr": float(latest_row['ATR']),
                "volume": float(latest_row['volume'])
            }
            
            recent_candles = df.tail(5).copy()
            recent_candles['timestamp'] = recent_candles['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_candles_dict = recent_candles.to_dict('records')
            
            for candle in recent_candles_dict:
                for key, value in candle.items():
                    if isinstance(value, (np.integer, np.floating)):
                        candle[key] = float(value)
                    elif pd.isna(value):
                        candle[key] = 0.0
            
            multi_tf_data[tf_name] = {
                "current_indicators": current_indicators,
                "recent_candles": recent_candles_dict
            }
            
            print(f"Collected {coin_name} {tf_name} data with momentum indicators: {len(df)} candles")
            
        except Exception as e:
            print(f"Error fetching {symbol} {tf_name} data: {e}")
            continue
    
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수 =====
def fetch_funding_rate_for_symbol(binance_symbol):
    """특정 코인의 펀딩비 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {'symbol': binance_symbol}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            funding_rate = float(data.get('lastFundingRate', 0))
            next_funding_time = datetime.fromtimestamp(int(data.get('nextFundingTime', 0)) / 1000).isoformat()
            
            return {
                "current_funding_rate": funding_rate,
                "funding_rate_percentage": funding_rate * 100,
                "next_funding_time": next_funding_time,
                "funding_sentiment": "HIGH_LONG_LEVERAGE" if funding_rate > 0.01 else 
                                  "HIGH_SHORT_LEVERAGE" if funding_rate < -0.01 else "NEUTRAL"
            }
    except Exception as e:
        print(f"Error fetching funding rate for {binance_symbol}: {e}")
        return {"current_funding_rate": 0, "funding_rate_percentage": 0, "funding_sentiment": "UNKNOWN"}

def fetch_open_interest_for_symbol(binance_symbol):
    """특정 코인의 미결제약정 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {'symbol': binance_symbol}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            current_data = response.json()
            current_oi = float(current_data.get('openInterest', 0))
            
            url_historical = "https://fapi.binance.com/futures/data/openInterestHist"
            params_hist = {
                'symbol': binance_symbol,
                'period': '1h',
                'limit': 24
            }
            hist_response = requests.get(url_historical, params=params_hist)
            
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if hist_data and len(hist_data) > 1:
                    previous_oi = float(hist_data[-2].get('sumOpenInterest', current_oi))
                    first_oi = float(hist_data[0].get('sumOpenInterest', current_oi))
                    
                    oi_change_percentage = ((current_oi - previous_oi) / previous_oi) * 100 if previous_oi > 0 else 0
                    oi_change_24h_percentage = ((current_oi - first_oi) / first_oi) * 100 if first_oi > 0 else 0
                    
                    return {
                        "latest_open_interest": current_oi,
                        "oi_change_1h_percentage": oi_change_percentage,
                        "oi_change_24h_percentage": oi_change_24h_percentage,
                        "oi_trend": "INCREASING" if oi_change_percentage > 2 else "DECREASING" if oi_change_percentage < -2 else "STABLE"
                    }
            
            return {
                "latest_open_interest": current_oi,
                "oi_change_1h_percentage": 0,
                "oi_change_24h_percentage": 0,
                "oi_trend": "STABLE"
            }
        
    except Exception as e:
        print(f"Error fetching open interest for {binance_symbol}: {e}")
        return {"latest_open_interest": 0, "oi_change_1h_percentage": 0, "oi_trend": "UNKNOWN"}

def fetch_long_short_ratio_for_symbol(binance_symbol):
    """특정 코인의 롱숏 비율 데이터 수집"""
    try:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {
            'symbol': binance_symbol,
            'period': '5m',
            'limit': 50
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            
            latest_ratio = df['longShortRatio'].iloc[-1]
            avg_ratio = df['longShortRatio'].mean()
            
            sentiment = "EXTREME_LONG_BIAS" if latest_ratio > 2.5 else \
                       "LONG_BIAS" if latest_ratio > 1.5 else \
                       "SHORT_BIAS" if latest_ratio < 0.8 else \
                       "EXTREME_SHORT_BIAS" if latest_ratio < 0.5 else "BALANCED"
            
            return {
                "latest_long_short_ratio": latest_ratio,
                "average_long_short_ratio": avg_ratio,
                "ls_sentiment": sentiment,
                "contrarian_signal": "SHORT_OPPORTUNITY" if latest_ratio > 2.0 else 
                                   "LONG_OPPORTUNITY" if latest_ratio < 0.8 else "NO_CLEAR_SIGNAL"
            }
    except Exception as e:
        print(f"Error fetching long/short ratio for {binance_symbol}: {e}")
        return {"latest_long_short_ratio": 1.0, "ls_sentiment": "UNKNOWN", "contrarian_signal": "NO_DATA"}

def fetch_liquidation_data_for_symbol(binance_symbol):
    """특정 코인의 청산 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/forceOrders"
        params = {
            'symbol': binance_symbol,
            'limit': 100
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            long_liquidations = [order for order in data if order['side'] == 'SELL']
            short_liquidations = [order for order in data if order['side'] == 'BUY']
            
            total_long_liq = sum([float(order['executedQty']) for order in long_liquidations])
            total_short_liq = sum([float(order['executedQty']) for order in short_liquidations])
            
            liq_ratio = total_long_liq / (total_short_liq + 0.0001)
            
            return {
                "long_liquidations": total_long_liq,
                "short_liquidations": total_short_liq,
                "liquidation_ratio": liq_ratio,
                "liquidation_pressure": "LONG_SQUEEZE" if liq_ratio > 2 else 
                                      "SHORT_SQUEEZE" if liq_ratio < 0.5 else "BALANCED"
            }
    except Exception as e:
        print(f"Error fetching liquidation data for {binance_symbol}: {e}")
        return {"liquidation_pressure": "UNKNOWN", "liquidation_ratio": 1.0}

def fetch_market_sentiment_for_coin(binance_symbol):
    """특정 코인의 모든 시장 심리 지표를 통합 수집"""
    funding_data = fetch_funding_rate_for_symbol(binance_symbol)
    oi_data = fetch_open_interest_for_symbol(binance_symbol)
    ls_ratio_data = fetch_long_short_ratio_for_symbol(binance_symbol)
    liquidation_data = fetch_liquidation_data_for_symbol(binance_symbol)
    
    return {
        "funding_rate": funding_data,
        "open_interest": oi_data,
        "long_short_ratio": ls_ratio_data,
        "liquidations": liquidation_data
    }

# ===== 객관적 스코어링 함수 추가 =====
def calculate_objective_score(coin_data, coin_name):
    """객관적 지표 기반 스코어 계산 (0-100점)"""
    score = 0
    
    try:
        # 5분봉 데이터를 기준으로 사용 (가장 신뢰성 높음)
        if '5m' not in coin_data['technical_data'] or 'current_indicators' not in coin_data['technical_data']['5m']:
            return 0
            
        indicators = coin_data['technical_data']['5m']['current_indicators']
        
        # RSI 오버솔드/오버봇 (40점 만점)
        rsi_14 = indicators.get('rsi_14', 50)
        if rsi_14 < 30:  # 오버솔드 (매수 신호)
            score += 20
            print(f"{coin_name} RSI 오버솔드: {rsi_14:.1f} (+20점)")
        elif rsi_14 > 70:  # 오버봇 (매도 신호) 
            score += 20
            print(f"{coin_name} RSI 오버봇: {rsi_14:.1f} (+20점)")
        
        # MACD 크로스오버 (30점 만점)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:  # 골든 크로스 (매수 신호)
            score += 15
            print(f"{coin_name} MACD 골든크로스 (+15점)")
        
        # 추가 지표들 (향후 확장 가능)
        # Williams %R 극값
        williams_r = indicators.get('williams_r', -50)
        if williams_r < -80:  # 과매도
            score += 10
            print(f"{coin_name} Williams %R 과매도: {williams_r:.1f} (+10점)")
        elif williams_r > -20:  # 과매수
            score += 10
            print(f"{coin_name} Williams %R 과매수: {williams_r:.1f} (+10점)")
            
        # 볼륨 확인 (거래량이 평소보다 높으면 가점)
        volume = indicators.get('volume', 0)
        if volume > 0:  # 볼륨이 있으면 신뢰성 가점
            score += 5
            
        print(f"{coin_name} 객관적 스코어: {score}/100")
        return min(score, 100)
        
    except Exception as e:
        print(f"Error calculating objective score for {coin_name}: {e}")
        return 0

# ===== 멀티코인 데이터 통합 수집 함수 (스코어링 추가) =====
def fetch_all_coins_data():
    """모든 코인(BTC, ETH, SOL)의 데이터를 수집하고 객관적 스코어 계산"""
    all_coins_data = {}
    
    for coin_name, coin_config in TRADING_PAIRS.items():
        try:
            print(f"Collecting {coin_name} data...")
            
            technical_data = fetch_multi_timeframe_data_for_coin(coin_config["symbol"])
            sentiment_data = fetch_market_sentiment_for_coin(coin_config["binance_symbol"])
            
            current_price = 0
            if technical_data and any(tf_data for tf_data in technical_data.values()):
                for tf in ["1m", "5m", "15m", "1h"]:
                    if tf in technical_data and "current_indicators" in technical_data[tf]:
                        current_price = technical_data[tf]["current_indicators"]["current_price"]
                        break
            
            if current_price == 0:
                print(f"No valid price data for {coin_name}, skipping...")
                continue
            
            # 임시 코인 데이터 생성 (객관적 스코어 계산용)
            temp_coin_data = {
                "config": coin_config,
                "technical_data": technical_data,
                "sentiment_data": sentiment_data,
                "current_price": current_price
            }
            
            # 객관적 스코어 계산 추가
            objective_score = calculate_objective_score(temp_coin_data, coin_name)
            
            all_coins_data[coin_name] = {
                "config": coin_config,
                "technical_data": technical_data,
                "sentiment_data": sentiment_data,
                "current_price": current_price,
                "objective_score": objective_score  # 객관적 스코어 추가
            }
            
            print(f"{coin_name}: ${current_price:,.2f} (객관적 스코어: {objective_score}/100)")
            
        except Exception as e:
            print(f"Error collecting {coin_name} data: {e}")
            continue
    
    return all_coins_data

# ===== 포지션 관리 함수 =====
def check_current_positions():
    """모든 코인의 현재 포지션 상태 확인"""
    current_positions = []
    
    for coin_name, coin_config in TRADING_PAIRS.items():
        try:
            positions = exchange.fetch_positions([coin_config["symbol"]])
            for position in positions:
                if position['symbol'] == f"{coin_config['symbol']}:USDT":
                    amt = float(position['info']['positionAmt'])
                    if amt != 0:
                        position_info = {
                            "coin": coin_name,
                            "symbol": coin_config["symbol"],
                            "side": "long" if amt > 0 else "short",
                            "amount": abs(amt),
                            "config": coin_config
                        }
                        current_positions.append(position_info)
                        print(f"Active Position: {coin_name} {position_info['side'].upper()} {position_info['amount']}")
        except Exception as e:
            print(f"Error checking {coin_name} position: {e}")
            continue
    
    return current_positions

def handle_position_closure(current_price, side, amount, coin_symbol, current_trade_id=None):
    """포지션 종료 시 데이터베이스를 업데이트하고 결과를 표시합니다"""
    if current_trade_id is None:
        open_trades = get_all_open_trades()
        for trade in open_trades:
            if trade['coin_symbol'] == coin_symbol:
                current_trade_id = trade['id']
                break
    
    if current_trade_id:
        open_trades = get_all_open_trades()
        latest_trade = None
        for trade in open_trades:
            if trade['id'] == current_trade_id:
                latest_trade = trade
                break
        
        if latest_trade:
            entry_price = latest_trade['entry_price']
            action = latest_trade['action']
            leverage = latest_trade['leverage']
            
            if action == 'long':
                profit_loss = (current_price - entry_price) * amount
                profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
            else:
                profit_loss = (entry_price - current_price) * amount
                profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                
            update_trade_status(
                current_trade_id,
                'CLOSED',
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            
            print(f"\n=== {coin_symbol} Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("============================")
            
            summary = get_trade_summary(days=7)
            if summary:
                print("\n=== 7-Day Trading Summary ===")
                print(f"Total Trades: {summary['total_trades']}")
                print(f"Win/Loss: {summary['winning_trades']}/{summary['losing_trades']}")
                if summary['total_trades'] > 0:
                    win_rate = (summary['winning_trades'] / summary['total_trades']) * 100
                    print(f"Win Rate: {win_rate:.2f}%")
                print(f"Total P/L: ${summary['total_profit_loss']:,.2f}")
                print(f"Avg P/L %: {summary['avg_profit_loss_percentage']:.2f}%")
                print("=============================")

# ===== Gemini AI 멀티코인 분석 함수 =====
def analyze_multi_coin_with_ai(all_coins_data, historical_data, performance_metrics):
    """3개 코인을 AI로 분석하고 최고 확률의 거래 기회를 선택"""
    
    system_prompt = """
You are an expert multi-cryptocurrency day trader who analyzes BTC, ETH, and SOL simultaneously to find high probability trading opportunities.

COIN CHARACTERISTICS:
- BTC: Most stable, strong trend following, lower volatility. Best for breakout/trend following strategies.
- ETH: Medium volatility, DeFi correlation, good for momentum trades. More reactive to news.
- SOL: High volatility, meme/retail driven, explosive moves but higher risk. Can have sudden pumps/dumps.

ANALYSIS PROCESS:
1. Analyze each coin's technical indicators (RSI, MACD, Stochastic, Williams %R) across 1m/5m/15m/1h timeframes
2. Pay special attention to 1m timeframe for precise entry timing and immediate momentum confirmation
3. Evaluate market sentiment (funding rate, open interest, long/short ratio, liquidations) for each coin
4. Score each coin from 0-100 based on trading opportunity quality
5. MULTI-POSITION STRATEGY: If multiple coins score above 75, recommend trading them simultaneously
6. Capital allocation: Split available capital between high-scoring opportunities

TIMEFRAME PRIORITY:
- 1H: Overall trend direction and major support/resistance levels
- 15M: Medium-term momentum confirmation and trend strength
- 5M: Entry timing and short-term price action
- 1M: Precise entry point, immediate momentum, and scalping opportunities

SCORING CRITERIA (0-100):
- Technical confluence (40%): Multiple timeframe alignment, especially 1m+5m+15m momentum convergence
- Market sentiment (30%): Funding rate extremes, OI changes, Long/Short ratio contrarian signals
- Risk/Reward potential (20%): Clear support/resistance levels, good R/R setup
- Volatility consideration (10%): Match strategy to coin's typical behavior

MULTI-POSITION DECISION RULES:
- Score 85+: HIGH conviction - Include in multi-position with larger allocation
- Score 75-84: MEDIUM conviction - Include in multi-position with medium allocation
- Score 70-74: LOW conviction - Only trade if no better opportunities available
- Below 70: NO_POSITION

CAPITAL ALLOCATION STRATEGY:
- Single high-conviction trade (only one coin 85+): Use 40-60% of capital
- Two high-conviction trades: Split 30-40% each
- Three high-conviction trades: Split 20-30% each
- Mixed conviction: Allocate based on score differences

LEVERAGE & RISK ADJUSTMENT:
- BTC: Can handle higher leverage (up to 50x) due to stability
- ETH: Medium leverage (up to 35x) 
- SOL: Lower leverage (up to 25x) due to high volatility
- Multi-position: Reduce leverage by 20-30% to manage overall risk

POSITION SIZING & SL/TP (Margin-based):
- Stop Loss: 8-35% of invested margin per trade (wider for more volatile coins)
- Take Profit: 15-70% of invested margin per trade
- Always aim for minimum 1:1.5 Risk/Reward ratio per trade
- Multi-position: Consider correlation - avoid same direction on highly correlated coins

TIMEOUT CONSIDERATION:
- The bot automatically closes profitable positions after 4 hours if SL/TP not hit
- Factor this into your analysis - prefer setups likely to reach targets within 2-4 hours
- 1m timeframe helps identify momentum that could reach targets faster
- Multi-position: Stagger entry times if possible for better risk management

IMPORTANT: Return ALL coins that score 75+ for multi-position trading.

Return ONLY valid JSON (no markdown):
{
  "trading_opportunities": [
    {
      "coin": "BTC/ETH/SOL",
      "score": score_0_100,
      "direction": "LONG/SHORT",
      "recommended_position_size": 0.1-0.6,
      "recommended_leverage": range_based_on_coin,
      "stop_loss_percentage": 0.08-0.35,
      "take_profit_percentage": 0.15-0.70,
      "priority": "HIGH/MEDIUM/LOW",
      "reasoning": "Why this specific coin scored high and deserves allocation"
    }
  ],
  "total_opportunities": number_of_opportunities,
  "overall_strategy": "SINGLE_POSITION/DUAL_POSITION/TRIPLE_POSITION/NO_POSITION",
  "capital_allocation_notes": "How to split capital between opportunities",
  "market_correlation_warning": "Any warnings about correlation risk between selected coins"
}
"""
    
    try:
        market_analysis = {
            "coins_data": all_coins_data,
            "recent_trades": historical_data,
            "performance_summary": performance_metrics
        }
        
        market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
        
        response = model.generate_content([
            system_prompt,
            f"Multi-Coin Market Analysis: {market_analysis_json}"
        ])
        
        response_content = response.text.strip()
        if response_content.startswith("```"):
            content_parts = response_content.split("\n", 1)
            if len(content_parts) > 1:
                response_content = content_parts[1]
            if "```" in response_content:
                response_content = response_content.rsplit("```", 1)[0]
            response_content = response_content.strip()
        
        trading_decision = json.loads(response_content)
        
        print(f"\n=== AI Multi-Position Analysis Results ===")
        opportunities = trading_decision.get('trading_opportunities', [])
        strategy = trading_decision.get('overall_strategy', 'NO_POSITION')
        
        print(f"Strategy: {strategy}")
        print(f"Total Opportunities: {len(opportunities)}")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['coin']} (Score: {opp['score']}) - {opp['direction']} - Priority: {opp['priority']}")
            print(f"   Position Size: {opp['recommended_position_size']*100:.1f}%, Leverage: {opp['recommended_leverage']}x")
        
        if opportunities:
            print(f"Capital Allocation: {trading_decision.get('capital_allocation_notes', '')}")
            if trading_decision.get('market_correlation_warning'):
                print(f"⚠️ Correlation Warning: {trading_decision.get('market_correlation_warning')}")
        
        print("=====================================")
        
        return trading_decision
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {"trading_opportunities": [], "overall_strategy": "NO_POSITION"}

# ===== 거래 실행 함수 =====
def execute_multi_position_trades(trading_opportunities, all_coins_data):
    """여러 코인 동시 거래 실행"""
    if not trading_opportunities:
        return []
    
    executed_trades = []
    balance = exchange.fetch_balance()
    available_capital = balance['USDT']['free']
    
    print(f"\n=== Multi-Position Trading Execution ===")
    print(f"Available Capital: ${available_capital:,.2f}")
    print(f"Number of Opportunities: {len(trading_opportunities)}")
    
    for i, opportunity in enumerate(trading_opportunities, 1):
        try:
            coin_name = opportunity['coin']
            coin_config = TRADING_PAIRS[coin_name]
            current_price = all_coins_data[coin_name]['current_price']
            
            print(f"\n--- Executing Trade {i}/{len(trading_opportunities)}: {coin_name} ---")
            
            trade_id = execute_single_trade(
                coin_name=coin_name,
                coin_config=coin_config,
                opportunity=opportunity,
                current_price=current_price,
                available_capital=available_capital
            )
            
            if trade_id:
                executed_trades.append({
                    'coin': coin_name,
                    'trade_id': trade_id,
                    'direction': opportunity['direction'],
                    'score': opportunity['score']
                })
                print(f"✅ {coin_name} trade executed successfully")
                
                used_margin = available_capital * opportunity['recommended_position_size']
                available_capital -= used_margin
            else:
                print(f"❌ {coin_name} trade execution failed")
                
        except Exception as e:
            print(f"Error executing {coin_name} trade: {e}")
            continue
    
    print(f"\n=== Multi-Position Summary ===")
    print(f"Successfully executed: {len(executed_trades)}/{len(trading_opportunities)} trades")
    for trade in executed_trades:
        print(f"✅ {trade['coin']}: {trade['direction']} (Score: {trade['score']})")
    print("==============================")
    
    return executed_trades

def execute_single_trade(coin_name, coin_config, opportunity, current_price, available_capital):
    """단일 코인 거래 실행"""
    try:
        symbol = coin_config["symbol"]
        action = opportunity["direction"].lower()
        
        position_size_percentage = opportunity['recommended_position_size']
        recommended_leverage = opportunity['recommended_leverage']
        sl_percentage = opportunity['stop_loss_percentage']
        tp_percentage = opportunity['take_profit_percentage']
        
        investment_amount = available_capital * position_size_percentage
        
        min_investment = coin_config.get('min_investment', 100)
        if investment_amount < min_investment:
            investment_amount = min_investment
            print(f"최소 투입 마진({min_investment} USDT)으로 조정됨")

        print(f"투입 마진: {investment_amount:.2f} USDT")
        
        total_position_value = investment_amount * recommended_leverage
        print(f"총 포지션 가치: {total_position_value:.2f} USDT")

        precision = coin_config.get('precision', 4)
        multiplier = 10 ** precision
        amount = math.ceil((total_position_value / current_price) * multiplier) / multiplier
        
        if amount <= 0:
            print("계산된 주문 수량이 0보다 작거나 같아 주문을 진행하지 않습니다.")
            return None
        
        print(f"주문 수량: {amount} {coin_name}")

        exchange.set_leverage(recommended_leverage, symbol)
        print(f"레버리지 설정: {recommended_leverage}x")

        sl_price_change_ratio = sl_percentage / recommended_leverage
        tp_price_change_ratio = tp_percentage / recommended_leverage

        if action == "long":
            order = exchange.create_market_buy_order(symbol, amount)
            entry_price = current_price
            
            sl_price = round(entry_price * (1 - sl_price_change_ratio), 2)
            tp_price = round(entry_price * (1 + tp_price_change_ratio), 2)
            
            exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
            exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
            
            print(f"{coin_name} LONG Entry: ${entry_price:,.2f}")
            print(f"SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f}")
            
        elif action == "short":
            order = exchange.create_market_sell_order(symbol, amount)
            entry_price = current_price
            
            sl_price = round(entry_price * (1 + sl_price_change_ratio), 2)
            tp_price = round(entry_price * (1 - tp_price_change_ratio), 2)
            
            exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
            exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
            
            print(f"{coin_name} SHORT Entry: ${entry_price:,.2f}")
            print(f"SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f}")
        
        trade_data = {
            'action': action,
            'entry_price': entry_price,
            'amount': amount,
            'leverage': recommended_leverage,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_percentage': sl_percentage,
            'tp_percentage': tp_percentage,
            'position_size_percentage': position_size_percentage,
            'investment_amount': investment_amount
        }
        trade_id = save_trade(trade_data, coin_name)
        
        return trade_id
        
    except Exception as e:
        print(f"Single trade execution error for {coin_name}: {e}")
        return None

# ===== 메인 프로그램 시작 =====
def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v4.0 - Multi-Position + 1M + 4H Timeout) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Trading Pairs: BTC/USDT, ETH/USDT, SOL/USDT")
    print("Strategy: Multi-Position Simultaneous Trading (최대 3개 코인 동시)")
    print("AI Engine: Google Gemini 2.5 Flash")
    print("Timeframes: 1m, 5m, 15m, 1h (1분봉 스캘핑 지원)")
    print("Leverage Range: 3-50x (Coin-specific)")
    print("SL/TP Range: 8-70% on margin (Dynamic)")
    print("Multi-Position: 75점+ 코인들 동시 거래 지원")
    print("Timeout: 4시간 후 수익 포지션 자동 정리")
    print("Capital Allocation: 스코어 기반 자동 분배")
    print("Market Sentiment: Funding Rate, OI, L/S Ratio, Liquidations")
    print("Momentum Indicators: RSI, MACD, Stochastic, Williams %R")
    print("Execution Frequency: Every 2 minutes")
    print("Trading Hours: 24/7 Unlimited")
    print("Min Margin: 50-100 USDT (Coin-specific)")
    print("===============================================\n")

    setup_database()

    while True:
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{current_time}] === Multi-Coin Market Scan ===")

            current_positions = check_current_positions()
            
            timeout_trades = check_trade_timeout()
            if timeout_trades > 0:
                print(f"타임아웃으로 {timeout_trades}개 거래 정리됨. 새로운 스캔 시작...")
                time.sleep(10)
                continue
            
            if current_positions:
                print(f"Monitoring {len(current_positions)} active positions:")
                for pos in current_positions:
                    try:
                        current_price = exchange.fetch_ticker(pos['symbol'])['last']
                        print(f"  {pos['coin']}: {pos['side'].upper()} ${current_price:,.2f}")
                    except Exception as e:
                        print(f"  Error monitoring {pos['coin']}: {e}")
                
                time.sleep(60)
                continue

            open_trades = get_all_open_trades()
            if open_trades:
                for trade in open_trades:
                    coin_name = trade['coin_symbol']
                    coin_config = TRADING_PAIRS.get(coin_name)
                    if coin_config:
                        try:
                            current_price = exchange.fetch_ticker(coin_config['symbol'])['last']
                            handle_position_closure(current_price, trade['action'], 
                                                  trade['amount'], coin_name, trade['id'])
                        except Exception as e:
                            print(f"Error handling closure for {coin_name}: {e}")
            
            for coin_name, coin_config in TRADING_PAIRS.items():
                try:
                    open_orders = exchange.fetch_open_orders(coin_config['symbol'])
                    if open_orders:
                        for order in open_orders:
                            exchange.cancel_order(order['id'], coin_config['symbol'])
                        print(f"Cancelled remaining {coin_name} orders")
                except Exception as e:
                    print(f"Error cancelling {coin_name} orders: {e}")
            
            print("No positions. Scanning all coins for multi-position opportunities...")
            time.sleep(5)

            all_coins_data = fetch_all_coins_data()
            
            if len(all_coins_data) < 1:
                print("Insufficient coin data collected. Waiting...")
                time.sleep(120)
                continue

            historical_trading_data = get_historical_trading_data(limit=3)
            performance_metrics = get_performance_metrics()

            trading_decision = analyze_multi_coin_with_ai(all_coins_data, historical_trading_data, performance_metrics)

            # 객관적 스코어 필터링 추가 (AI 결과와 결합)
            opportunities = trading_decision.get('trading_opportunities', [])
            filtered_opportunities = []
            
            for opportunity in opportunities:
                coin_name = opportunity['coin']
                ai_score = opportunity['score']
                objective_score = all_coins_data.get(coin_name, {}).get('objective_score', 0)
                
                # 객관적 스코어가 30점 이상인 것만 거래 (추가 안전장치)
                if objective_score >= 30:
                    # AI 스코어와 객관적 스코어 결합 (가중평균)
                    combined_score = (ai_score * 0.7) + (objective_score * 0.3)
                    opportunity['combined_score'] = combined_score
                    opportunity['objective_score'] = objective_score
                    filtered_opportunities.append(opportunity)
                    print(f"{coin_name} 통과: AI({ai_score}) + 객관적({objective_score}) = 결합({combined_score:.1f})")
                else:
                    print(f"{coin_name} 필터링됨: 객관적 스코어 부족 ({objective_score}/100)")
            
            # 필터링된 기회들로 trading_decision 업데이트
            if filtered_opportunities:
                trading_decision['trading_opportunities'] = filtered_opportunities
                # 가장 높은 결합 스코어 순으로 정렬
                trading_decision['trading_opportunities'].sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
            else:
                trading_decision['trading_opportunities'] = []
                trading_decision['overall_strategy'] = 'NO_POSITION'
                print("모든 코인이 객관적 스코어 필터링에서 제외됨")

            opportunities = trading_decision.get('trading_opportunities', [])
            
            if opportunities:
                main_opportunity = opportunities[0]
                analysis_data = {
                    'selected_coin': main_opportunity['coin'],
                    'coin_scores': {
                        'BTC': next((opp['score'] for opp in opportunities if opp['coin'] == 'BTC'), 0),
                        'ETH': next((opp['score'] for opp in opportunities if opp['coin'] == 'ETH'), 0),
                        'SOL': next((opp['score'] for opp in opportunities if opp['coin'] == 'SOL'), 0)
                    },
                    'current_price': all_coins_data.get(main_opportunity['coin'], {}).get('current_price', 0),
                    'direction': main_opportunity['direction'],
                    'recommended_position_size': main_opportunity['recommended_position_size'],
                    'recommended_leverage': main_opportunity['recommended_leverage'],
                    'stop_loss_percentage': main_opportunity['stop_loss_percentage'],
                    'take_profit_percentage': main_opportunity['take_profit_percentage'],
                    'reasoning': f"Multi-position strategy: {len(opportunities)} opportunities. " + main_opportunity['reasoning']
                }
            else:
                analysis_data = {
                    'selected_coin': 'NO_POSITION',
                    'coin_scores': {'BTC': 0, 'ETH': 0, 'SOL': 0},
                    'current_price': 0,
                    'direction': 'NO_POSITION',
                    'recommended_position_size': 0,
                    'recommended_leverage': 0,
                    'stop_loss_percentage': 0,
                    'take_profit_percentage': 0,
                    'reasoning': trading_decision.get('capital_allocation_notes', 'No opportunities found')
                }
            
            analysis_id = save_ai_analysis(analysis_data)

            strategy = trading_decision.get('overall_strategy', 'NO_POSITION')
            
            if strategy == "NO_POSITION" or not opportunities:
                print("No high-probability setup found across BTC/ETH/SOL.")
                time.sleep(120)
                continue

            executed_trades = execute_multi_position_trades(opportunities, all_coins_data)
            
            if executed_trades:
                main_trade_id = executed_trades[0]['trade_id']
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (main_trade_id, analysis_id))
                conn.commit()
                conn.close()
                
                print(f"\n🚀 Multi-Position Strategy Executed Successfully!")
                print(f"Strategy: {strategy}")
                print(f"Active Trades: {len(executed_trades)}")
                for trade in executed_trades:
                    print(f"  - {trade['coin']}: {trade['direction']} (Score: {trade['score']})")
            else:
                print("No trades were executed successfully.")

            time.sleep(120)

        except Exception as e:
            print(f"\n Main Loop Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()