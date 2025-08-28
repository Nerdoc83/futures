"""
AI 멀티코인 데이트레이딩 봇 - Gemini + AI 전담 판단 (v6.3 - 최종 안정화)
--------------------------------------------------------
기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - AI가 모든 판단 담당
- 데이트레이딩 최적화 (3분봉 메인)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini API 기반 AI 분석 (100% AI 의존)
- 동적 레버리지 및 포지션 사이징 (AI 자율 결정)
- AI 기반 동적 SL/TP 설정
- 90분 타임컷: 90분 경과 시 수익/손실 무관 포지션 자동 정리 (시간 기반 손절)
- 10분 부분 스캔: 포지션 보유 시 10분마다 비어있는 코인 재스캔 (시작 시 즉시 실행)
- 상관관계 리스크 관리: 모든 코인 동시 진입 신호 시 최고점수 포지션만 진입
- 합리성 필터: AI의 비현실적인 TP/SL 제안 자동 거부 (안전장치)
- 레버리지 필터: AI의 레버리지 제안이 설정 범위를 벗어날 경우 자동 조정
- DB-포지션 동기화 기능
- 24시간 무제한 거래
- 최소 투자금액: 40-10 USDT
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
from datetime import datetime, timedelta

# ===== 멀티코인 설정 =====
TRADING_PAIRS = {
    "BTC": {
        "symbol": "BTC/USDT",
        "binance_symbol": "BTCUSDT",
        "min_investment": 40,
        "leverage_range": (5, 50),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.15, 0.50),
        "precision": 5
    },
    "ETH": {
        "symbol": "ETH/USDT", 
        "binance_symbol": "ETHUSDT",
        "min_investment": 20,
        "leverage_range": (5, 35),
        "volatility_factor": 1.3,
        "sl_range": (0.10, 0.30),
        "tp_range": (0.20, 0.60),
        "precision": 4
    },
    "SOL": {
        "symbol": "SOL/USDT",
        "binance_symbol": "SOLUSDT", 
        "min_investment": 10,
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
        'adjustForTimeDifference': True,
        # <<<< FIX: API 버전을 명시적으로 지정하여 통신 오류 방지 >>>>
        'versions': {
            'fapiPrivate': 'v2',
            'fapiPublic': 'v1',
        },
    }
})

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

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
    WHERE t.status IN ('CLOSED', 'CLOSED_TIMEOUT')
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
    WHERE status IN ('CLOSED', 'CLOSED_TIMEOUT')
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

# ===== 새로운 동기화 함수 =====
def sync_database_with_positions():
    """데이터베이스의 OPEN 거래와 실제 포지션 상태를 동기화"""
    try:
        open_trades = get_all_open_trades()
        if not open_trades:
            return
        
        all_positions = {}
        for coin_name, coin_config in TRADING_PAIRS.items():
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                for position in positions:
                    if position['symbol'] == f"{coin_config['symbol']}:USDT":
                        position_amt = float(position['info']['positionAmt'])
                        if position_amt != 0:
                            all_positions[coin_name] = {
                                'amount': abs(position_amt),
                                'side': 'long' if position_amt > 0 else 'short',
                                'unrealized_pnl': float(position['info']['unRealizedProfit']),
                                'entry_price': float(position['info']['entryPrice'])
                            }
            except Exception as e:
                # <<<< FIX: 오류 로깅 강화 >>>>
                print(f"동기화 중 {coin_name} 포지션 조회 오류: {e}")
        
        for trade in open_trades:
            coin_symbol = trade['coin_symbol']
            trade_id = trade['id']
            
            if coin_symbol not in all_positions:
                print(f"⚠️ {coin_symbol} DB에는 OPEN, 실제로는 포지션 없음 - 동기화 중...")
                try:
                    coin_config = TRADING_PAIRS.get(coin_symbol)
                    if coin_config:
                        current_price = exchange.fetch_ticker(coin_config['symbol'])['last']
                        entry_price, action, leverage, amount = trade['entry_price'], trade['action'], trade['leverage'], trade['amount']
                        
                        if action == 'long':
                            profit_loss = (current_price - entry_price) * amount
                            profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
                        else:
                            profit_loss = (entry_price - current_price) * amount
                            profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                        
                        update_trade_status(
                            trade_id, 'CLOSED', exit_price=current_price, exit_timestamp=datetime.now().isoformat(),
                            profit_loss=profit_loss, profit_loss_percentage=profit_loss_percentage
                        )
                        print(f"✅ {coin_symbol} DB 동기화 완료: P/L ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                except Exception as e:
                    print(f"❌ {coin_symbol} 동기화 오류: {e}")
        
    except Exception as e:
        print(f"동기화 프로세스 오류: {e}")

# ===== 타임아웃 함수 (정확한 수익률 계산으로 수정) =====
def check_trade_timeout():
    """90분 이상 경과된 모든 포지션을 수익/손실 관계없이 정리 (시간 기반 손절)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, coin_symbol, action, entry_price, amount, leverage
    FROM trades
    WHERE status = 'OPEN' 
    AND datetime(timestamp) <= datetime('now', '-90 minutes')
    ''')
    
    timeout_candidates = cursor.fetchall()
    conn.close()
    
    if not timeout_candidates:
        return 0

    timeout_count = 0
    
    for trade in timeout_candidates:
        trade_id, coin_symbol, action, entry_price, amount, leverage = trade
        
        try:
            coin_config = TRADING_PAIRS.get(coin_symbol)
            if not coin_config:
                continue
            
            positions = exchange.fetch_positions([coin_config["symbol"]])
            current_pnl_percentage = 0
            actual_entry_price = entry_price 
            position_found = False

            for position in positions:
                if position['symbol'] == f"{coin_config['symbol']}:USDT":
                    pos_amt = float(position['info']['positionAmt'])
                    if (action == 'long' and pos_amt > 0) or (action == 'short' and pos_amt < 0):
                        unrealized_pnl = float(position['info']['unRealizedProfit'])
                        initial_margin = float(position['info']['initialMargin'])
                        actual_entry_price = float(position['info']['entryPrice']) 
                        if initial_margin > 0:
                            current_pnl_percentage = (unrealized_pnl / initial_margin) * 100
                        position_found = True
                        break
            
            if not position_found:
                print(f"타임아웃 후보 {coin_symbol} 포지션을 거래소에서 찾을 수 없어 건너뜁니다.")
                continue

            current_price = exchange.fetch_ticker(coin_config['symbol'])['last']

            print(f"\n=== 90분 타임컷 실행: {coin_symbol} {action.upper()} 포지션 정리 시작 ===")
            print(f" (90분 이상 경과, 현재 실제 수익률: {current_pnl_percentage:.2f}%)")
            print(f"Entry (Actual): ${actual_entry_price:,.2f}, Current: ${current_price:,.2f}")
            
            try:
                open_orders = exchange.fetch_open_orders(coin_config['symbol'])
                for order in open_orders:
                    exchange.cancel_order(order['id'], coin_config['symbol'])
                print(f"기존 SL/TP 주문 취소됨")
            except Exception as e:
                print(f"주문 취소 오류: {e}")
            
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                actual_amount = 0
                for position in positions:
                    if position['symbol'] == f"{coin_config['symbol']}:USDT":
                        position_amt = float(position['info']['positionAmt'])
                        if position_amt != 0:
                            actual_amount = abs(position_amt)
                            break
                
                if actual_amount == 0:
                    print(f"포지션이 이미 정리되었거나 찾을 수 없음")
                    update_trade_status(trade_id, 'CLOSED_TIMEOUT', exit_price=current_price, exit_timestamp=datetime.now().isoformat())
                    continue
                
                params = {'reduceOnly': True}
                if action == 'long':
                    exchange.create_order(coin_config['symbol'], 'market', 'sell', actual_amount, params=params)
                else:
                    exchange.create_order(coin_config['symbol'], 'market', 'buy', actual_amount, params=params)
                
                profit_loss = 0
                if action == 'long':
                    profit_loss = (current_price - actual_entry_price) * actual_amount
                else:
                    profit_loss = (actual_entry_price - current_price) * actual_amount
                
                update_trade_status(
                    trade_id, 'CLOSED_TIMEOUT', exit_price=current_price,
                    exit_timestamp=datetime.now().isoformat(), profit_loss=profit_loss,
                    profit_loss_percentage=current_pnl_percentage
                )
                
                print(f"타임컷 정리 완료: P/L ${profit_loss:,.2f} ({current_pnl_percentage:.2f}%)")
                timeout_count += 1
                
            except Exception as e:
                print(f"포지션 정리 오류: {e}")
                
        except Exception as e:
            print(f"타임컷 체크 오류 ({coin_symbol}): {e}")
    
    if timeout_count > 0:
        print("=== 타임컷 정리 완료, 새로운 스캔을 시작합니다 ===")
    
    return timeout_count

# ===== 데이터 수집 함수 (JSON 직렬화 오류 수정) =====
def fetch_multi_timeframe_data_for_coin(symbol):
    """단일 코인의 멀티 타임프레임 데이터 수집 (3분봉 메인)"""
    timeframes = {
        "3m": {"timeframe": "3m", "limit": 160},
        "1m": {"timeframe": "1m", "limit": 120},
        "5m": {"timeframe": "5m", "limit": 100},
        "15m": {"timeframe": "15m", "limit": 96},
        "1h": {"timeframe": "1h", "limit": 48}
    }
    multi_tf_data = {}
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_params["timeframe"], limit=tf_params["limit"])
            if not ohlcv or len(ohlcv) < 20: continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['RSI_14'] = calculate_rsi(df['close'], 14)
            df['RSI_21'] = calculate_rsi(df['close'], 21)
            macd, signal, histogram = calculate_macd(df['close'])
            df['MACD'], df['MACD_signal'], df['MACD_histogram'] = macd, signal, histogram
            stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'], df['Stoch_D'] = stoch_k, stoch_d
            df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
            df['ATR'] = calculate_atr(df)
            df.dropna(inplace=True)
            if len(df) < 5: continue
            
            latest = df.iloc[-1].to_dict()
            current_indicators = {k: float(v) for k, v in latest.items() if isinstance(v, (int, float, np.number))}
            
            recent_candles_df = df.tail(5).copy()
            recent_candles_df['timestamp'] = recent_candles_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_candles = recent_candles_df.to_dict('records')

            multi_tf_data[tf_name] = {"current_indicators": current_indicators, "recent_candles": recent_candles}
        except Exception:
            continue
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수 =====
def fetch_market_sentiment_for_coin(binance_symbol):
    """특정 코인의 모든 시장 심리 지표를 통합 수집"""
    return {
        "funding_rate": fetch_funding_rate_for_symbol(binance_symbol),
        "open_interest": fetch_open_interest_for_symbol(binance_symbol),
        "long_short_ratio": fetch_long_short_ratio_for_symbol(binance_symbol),
        "liquidations": fetch_liquidation_data_for_symbol(binance_symbol)
    }

def fetch_funding_rate_for_symbol(binance_symbol):
    try:
        data = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_symbol}").json()
        rate = float(data.get('lastFundingRate', 0))
        return {"funding_rate_percentage": rate * 100}
    except: return {}

def fetch_open_interest_for_symbol(binance_symbol):
    try:
        data = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_symbol}").json()
        return {"latest_open_interest": float(data.get('openInterest', 0))}
    except: return {}

def fetch_long_short_ratio_for_symbol(binance_symbol):
    try:
        data = requests.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={binance_symbol}&period=5m&limit=1").json()
        return {"latest_long_short_ratio": float(data[0].get('longShortRatio', 1))}
    except: return {}

def fetch_liquidation_data_for_symbol(binance_symbol):
    try:
        data = requests.get(f"https://fapi.binance.com/fapi/v1/forceOrders?symbol={binance_symbol}&limit=100").json()
        longs = sum(float(o['executedQty']) for o in data if o['side'] == 'SELL')
        shorts = sum(float(o['executedQty']) for o in data if o['side'] == 'BUY')
        return {"long_liquidations": longs, "short_liquidations": shorts}
    except: return {}


# ===== 멀티코인 데이터 통합 수집 함수 (AI 전담) =====
def fetch_all_coins_data():
    """모든 코인(BTC, ETH, SOL)의 데이터를 수집"""
    all_coins_data = {}
    for coin, config in TRADING_PAIRS.items():
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

# ===== 포지션 관리 함수 =====
def check_current_positions():
    """모든 코인의 현재 포지션 상태 확인"""
    current_positions = []
    try:
        positions = exchange.fetch_positions()
        active_positions = [p for p in positions if float(p['info']['positionAmt']) != 0]
        for pos in active_positions:
            symbol = pos['symbol'].replace(":USDT", "")
            coin_name = symbol.replace("/USDT", "")
            if coin_name in TRADING_PAIRS:
                current_positions.append({
                    "coin": coin_name, "symbol": TRADING_PAIRS[coin_name]['symbol'],
                    "side": pos['side'], "amount": pos['contracts']
                })
    except Exception as e:
        print(f"Error checking positions: {e}")
    return current_positions

# ===== Gemini AI 멀티코인 분석 함수 (상관관계 리스크 인지) =====
def analyze_multi_coin_with_ai(all_coins_data, historical_data, performance_metrics):
    """3개 코인을 AI로 분석하고 최고 확률의 거래 기회를 선택 (AI가 100% 판단)"""
    
    system_prompt = """
You are an expert multi-cryptocurrency day trader AI. Your SOLE mission is to analyze real-time market data for BTC, ETH, and SOL to identify the highest probability long or short trading opportunities. You have COMPLETE AUTONOMY.

**CRITICAL INSTRUCTION: Your trading decisions (LONG/SHORT) MUST be based ONLY on the provided real-time technical and sentiment data for each coin. Do NOT let the general coin characteristics below create a preconceived bias. A high-volatility coin can have a strong uptrend, and a low-volatility coin can have a strong downtrend.**

ANALYSIS PROCESS:
1.  **Data-First Analysis**: Objectively analyze each coin's technical indicators (RSI, MACD, Stoch, etc.) and market sentiment data across all timeframes.
2.  **Primary Timeframe**: The **3M (3-minute) chart is your PRIMARY** tool for determining the current momentum and precise entry timing.
3.  **Objective Scoring**: Score each coin from 0-100 based purely on the quality of the immediate trading setup.

**RISK MANAGEMENT OVERLAY (VERY IMPORTANT):**
- **Correlation Risk**: Be aware that the system applies a correlation risk filter. If you provide strong opportunities for all available coins in the same direction (all LONG or all SHORT), the system will automatically select ONLY the one with the highest score to execute. Therefore, your scoring must be precise to reflect the true conviction level for each setup.
- **Time-Cut Risk**: This strategy uses a strict **90-minute time-cut**. Any open position will be automatically closed after 90 minutes. Therefore, you MUST set **tighter, more realistic TP and SL targets** that are likely to be hit within this 90-minute window.

**RESPONSE JSON FORMAT:**
- YOUR RESPONSE must be ONLY a valid JSON object, with no markdown.
- The root of the JSON must be an object with a key named "trading_opportunities" which is a list of opportunity objects.
- Each opportunity object MUST contain: "coin", "score", "direction", "recommended_position_size", "recommended_leverage", "stop_loss_percentage", "take_profit_percentage", "priority", "reasoning".
- The value for "coin" MUST be one of ["BTC", "ETH", "SOL"].
- If no opportunities are found, return an empty "trading_opportunities" list.
"""

    try:
        market_analysis = {
            "coins_data": all_coins_data,
            "recent_trades": historical_data,
            "performance_summary": performance_metrics
        }
        
        market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content([
            system_prompt,
            f"Multi-Coin Market Analysis: {market_analysis_json}"
        ], safety_settings=safety_settings)
        
        response_content = response.text.strip().replace("```json", "").replace("```", "")
        trading_decision = json.loads(response_content)
        
        print("\n=== AI Analysis Result ===")
        opportunities = trading_decision.get('trading_opportunities', [])
        for opp in opportunities:
            print(f"- {opp.get('coin','N/A')}: {opp.get('direction','N/A')} (Score: {opp.get('score',0)})")
        
        return trading_decision
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {"trading_opportunities": []}

# ===== 거래 실행 함수 (레버리지 필터 및 마진 계산 수정) =====
def execute_single_trade(coin_name, opportunity, available_capital):
    """단일 코인 거래 실행"""
    try:
        coin_config = TRADING_PAIRS[coin_name]
        symbol = coin_config["symbol"]
        action = opportunity.get("direction", "").lower()
        
        def parse_percentage(value):
            """AI가 반환하는 퍼센트 값을 소수점으로 변환"""
            if isinstance(value, list):
                value = value[0] if value else "0"
            if isinstance(value, str):
                value = value.strip().replace('%', '')
            
            num_value = float(value)
            if num_value > 1:
                return num_value / 100.0
            return num_value

        def parse_leverage(value):
            """AI가 반환하는 레버리지 값을 정수로 변환"""
            if isinstance(value, list):
                value = value[0] if value else "1"
            if isinstance(value, str):
                value = value.strip().lower().replace('x', '')
            return int(float(value))

        pos_size_pct = parse_percentage(opportunity.get('recommended_position_size', 0))
        leverage = parse_leverage(opportunity.get('recommended_leverage', 1))
        sl_pct = parse_percentage(opportunity.get('stop_loss_percentage', 0))
        tp_pct = parse_percentage(opportunity.get('take_profit_percentage', 0))
        
        min_lev, max_lev = coin_config['leverage_range']
        original_leverage = leverage
        leverage = max(min_lev, min(leverage, max_lev)) 
        if original_leverage != leverage:
            print(f"   Leverage Adjusted: AI recommended {original_leverage}x, but it was adjusted to {leverage}x to fit the range ({min_lev}x-{max_lev}x).")

        current_price = exchange.fetch_ticker(symbol)['last']
        
        sl_price_check = current_price * (1 - (sl_pct / leverage)) if action == "long" else current_price * (1 + (sl_pct / leverage))
        tp_price_check = current_price * (1 + (tp_pct / leverage)) if action == "long" else current_price * (1 - (tp_pct / leverage))

        tp_change_pct = abs((tp_price_check / current_price) - 1) * 100
        sl_change_pct = abs((sl_price_check / current_price) - 1) * 100

        REALISTIC_CHANGE_LIMIT = 20.0 
        
        if tp_change_pct > REALISTIC_CHANGE_LIMIT or sl_change_pct > REALISTIC_CHANGE_LIMIT:
            print(f"\n❌ Trade Rejected (Unrealistic Target): {coin_name} {action.upper()}")
            print(f"   AI proposed TP change: {tp_change_pct:.2f}%, SL change: {sl_change_pct:.2f}%")
            print(f"   This exceeds the safety limit of {REALISTIC_CHANGE_LIMIT}%.")
            return None

        try:
            exchange.cancel_all_orders(symbol)
            print(f"{coin_name} 기존 미체결 주문 취소됨")
        except Exception: pass
        
        margin = available_capital * pos_size_pct
        min_margin = coin_config.get('min_investment', 10)
        if margin < min_margin:
            margin = min_margin
            print(f"최소 투자 마진({min_margin} USDT)으로 조정됨")

        if margin > available_capital:
            print(f"❌ Trade Rejected (Insufficient Margin): Required margin ${margin:,.2f} > Available capital ${available_capital:,.2f}")
            return None

        amount = (margin * leverage) / current_price
        
        exchange.set_leverage(leverage, symbol)
        
        order = None
        if action == "long":
            order = exchange.create_market_buy_order(symbol, amount)
        elif action == "short":
            order = exchange.create_market_sell_order(symbol, amount)
        
        if not order: return None
        
        entry_price = order.get('price', current_price)
        
        sl_price = entry_price * (1 - (sl_pct / leverage)) if action == "long" else entry_price * (1 + (sl_pct / leverage))
        tp_price = entry_price * (1 + (tp_pct / leverage)) if action == "long" else entry_price * (1 - (tp_pct / leverage))
        
        exchange.create_order(symbol, 'STOP_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': sl_price})
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': tp_price})
        
        print(f"{coin_name} {action.upper()} Entry: ${entry_price:,.2f} | SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f}")
        
        trade_data = {
            'action': action, 'entry_price': entry_price, 'amount': amount, 'leverage': leverage,
            'sl_price': sl_price, 'tp_price': tp_price, 'sl_percentage': sl_pct, 'tp_percentage': tp_pct,
            'position_size_percentage': pos_size_pct, 'investment_amount': margin
        }
        save_trade(trade_data, coin_name)
        return margin # Return the used margin
        
    except Exception as e:
        print(f"Single trade execution error for {coin_name}: {e}")
        return None

# ===== 메인 프로그램 시작 =====
def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v6.2 - 최종 안정화) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Strategy: AI 완전 자율 판단, 3분봉 메인")
    print("Risk Management: 90분 타임컷, 상관관계 필터, 합리성 필터, 레버리지 필터")
    print("Partial Scan: 10분마다 빈 코인슬롯 스캔 (시작 시 즉시 실행)")
    print("===============================================\n")

    setup_database()
    last_partial_scan_time = datetime.now() - timedelta(minutes=10)

    while True:
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{current_time}] === Market Check ===")

            sync_database_with_positions()
            
            if check_trade_timeout() > 0:
                time.sleep(10)
                continue
            
            current_positions = check_current_positions()
            
            if not current_positions:
                print("No open positions. Starting full scan...")
                all_data = fetch_all_coins_data()
                if not all_data:
                    time.sleep(60)
                    continue
                
                hist_data = get_historical_trading_data(3)
                perf_metrics = get_performance_metrics()
                decision = analyze_multi_coin_with_ai(all_data, hist_data, perf_metrics)
                opportunities = decision.get('trading_opportunities', [])
                
                if len(opportunities) == 3:
                    directions = {opp.get('direction') for opp in opportunities}
                    if len(directions) == 1:
                        print("\n⚠️ Correlation Risk Detected: All 3 coins have the same direction.")
                        highest_score_opp = max(opportunities, key=lambda x: x.get('score', 0))
                        print(f"   Filtering to the highest score: {highest_score_opp.get('coin')} (Score: {highest_score_opp.get('score', 0)})")
                        opportunities = [highest_score_opp]
                
                if opportunities:
                    balance = exchange.fetch_balance()['USDT']['free']
                    available_capital_for_loop = balance * 0.98
                    
                    for opp in opportunities:
                        used_margin = execute_single_trade(opp['coin'], opp, available_capital_for_loop)
                        if used_margin and used_margin > 0:
                            available_capital_for_loop -= used_margin
                        time.sleep(5)
                    last_partial_scan_time = datetime.now()
                else:
                    print("AI found no high-probability setups.")
                
                time.sleep(120)

            else: # 포지션이 있는 경우
                print(f"{len(current_positions)} active position(s). Monitoring...")
                
                time_since_scan = (datetime.now() - last_partial_scan_time).total_seconds() / 60
                if time_since_scan >= 10:
                    print("\n--- 10 min passed, scanning for additional opportunities ---")
                    available_coins = set(TRADING_PAIRS.keys()) - {p['coin'] for p in current_positions}
                    
                    if available_coins:
                        available_data = {c: d for c, d in fetch_all_coins_data().items() if c in available_coins}
                        if available_data:
                            hist_data = get_historical_trading_data(3)
                            perf_metrics = get_performance_metrics()
                            decision = analyze_multi_coin_with_ai(available_data, hist_data, perf_metrics)
                            opportunities = decision.get('trading_opportunities', [])
                            
                            num_scanned = len(available_data)
                            if num_scanned > 1 and len(opportunities) == num_scanned:
                                directions = {opp.get('direction') for opp in opportunities}
                                if len(directions) == 1:
                                    print(f"\n⚠️ Correlation Risk Detected: All {num_scanned} scanned coins have the same direction.")
                                    highest_score_opp = max(opportunities, key=lambda x: x.get('score', 0))
                                    print(f"   Filtering to the highest score: {highest_score_opp.get('coin')} (Score: {highest_score_opp.get('score', 0)})")
                                    opportunities = [highest_score_opp]

                            if opportunities:
                                balance = exchange.fetch_balance()['USDT']['free']
                                available_capital_for_loop = balance * 0.98
                                
                                for opp in opportunities:
                                    used_margin = execute_single_trade(opp['coin'], opp, available_capital_for_loop)
                                    if used_margin and used_margin > 0:
                                        available_capital_for_loop -= used_margin
                                    time.sleep(5)
                            else:
                                print("AI found no additional setups.")
                    else:
                        print("All coins have open positions.")
                    last_partial_scan_time = datetime.now()
                else:
                    remaining_minutes = 10 - int(time_since_scan)
                    print(f"다음 부분 스캔까지 {remaining_minutes}분 남음")

                time.sleep(60)

        except Exception as e:
            print(f"\nMain loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
