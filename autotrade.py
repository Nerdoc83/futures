"""
AI 비트코인 데이트레이딩 봇 - Gemini Pro + 고급 지표
--------------------------------------------------------
기능:
- 데이트레이딩 최적화 (5분, 15분, 1시간 차트)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini Pro API 기반 AI 분석
- 동적 레버리지 및 포지션 사이징 (5-50배)
- 개선된 SL/TP 설정 (1-6% 범위)
- 24시간 무제한 거래
--------------------------------------------------------
"""
# ===== 필요한 라이브러리 임포트 =====
import ccxt  # 암호화폐 거래소 API 라이브러리
import os  # 환경 변수 및 파일 시스템 접근
import math  # 수학 연산
import time  # 시간 지연 및 타임스탬프
import pandas as pd  # 데이터 분석 및 조작
import numpy as np  # 수치 계산
import requests  # HTTP 요청
import json  # JSON 데이터 처리
import sqlite3  # 로컬 데이터베이스
from dotenv import load_dotenv  # 환경 변수 로드
load_dotenv()  # .env 파일에서 환경 변수 로드
import google.generativeai as genai  # Gemini API
from datetime import datetime  # 날짜 및 시간 처리

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

# ===== 설정 및 초기화 =====
# 바이낸스 API 설정
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
symbol = "BTC/USDT"

# Gemini 2.5 Flash API 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')  # 최신 Gemini 2.5 Flash 모델

# SQLite 데이터베이스 설정
DB_FILE = "bitcoin_daytrading.db"

# ===== 데이터베이스 관련 함수 (기존과 동일) =====
def setup_database():
    """데이터베이스 및 필요한 테이블 생성"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 거래 기록 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
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
    
    # AI 분석 결과 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
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
    print("데이터베이스 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, current_price, direction, recommended_position_size, 
        recommended_leverage, stop_loss_percentage, take_profit_percentage, 
        reasoning, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
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

def save_trade(trade_data):
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp, action, entry_price, amount, leverage, sl_price, tp_price,
        sl_percentage, tp_percentage, position_size_percentage, investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
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

def get_latest_open_trade():
    """가장 최근의 열린 거래 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC
    LIMIT 1
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'action': result[1],
            'entry_price': result[2],
            'amount': result[3],
            'leverage': result[4],
            'sl_price': result[5],
            'tp_price': result[6]
        }
    return None

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
    """과거 거래 내역 가져오기 (토큰 절약을 위해 제한)"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.action, t.entry_price, t.exit_price, t.leverage,
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

# ===== 데이터 수집 함수 (데이트레이딩 최적화) =====
def fetch_multi_timeframe_data():
    """
    데이트레이딩 최적화된 멀티 타임프레임 데이터 수집
    5분, 15분, 1시간 차트로 변경
    """
    timeframes = {
        "5m": {"timeframe": "5m", "limit": 100},   # 8시간 20분 데이터
        "15m": {"timeframe": "15m", "limit": 96},  # 24시간 데이터
        "1h": {"timeframe": "1h", "limit": 48}     # 48시간 데이터
    }
    
    multi_tf_data = {}
    
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf_params["timeframe"], 
                limit=tf_params["limit"]
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # ===== 모멘텀 지표 계산 =====
            # RSI (14, 21)
            df['RSI_14'] = calculate_rsi(df['close'], window=14)
            df['RSI_21'] = calculate_rsi(df['close'], window=21)
            
            # MACD
            macd, signal, histogram = calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_histogram'] = histogram
            
            # Stochastic Oscillator
            stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Williams %R
            df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
            
            # ATR (변동성 측정)
            df['ATR'] = calculate_atr(df)
            
            # NaN 값 제거
            df.dropna(inplace=True)
            
            # 토큰 절약을 위해 최근 데이터와 핵심 지표만 저장
            current_indicators = {
                "current_price": float(df['close'].iloc[-1]),
                "rsi_14": float(df['RSI_14'].iloc[-1]),
                "rsi_21": float(df['RSI_21'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "macd_signal": float(df['MACD_signal'].iloc[-1]),
                "macd_histogram": float(df['MACD_histogram'].iloc[-1]),
                "stoch_k": float(df['Stoch_K'].iloc[-1]),
                "stoch_d": float(df['Stoch_D'].iloc[-1]),
                "williams_r": float(df['Williams_R'].iloc[-1]),
                "atr": float(df['ATR'].iloc[-1]),
                "volume": float(df['volume'].iloc[-1])
            }
            
            # JSON 직렬화 가능한 형태로 최근 캔들 데이터 변환
            recent_candles = df.tail(5).copy()
            recent_candles['timestamp'] = recent_candles['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_candles_dict = recent_candles.to_dict('records')
            
            # float64를 일반 float로 변환
            for candle in recent_candles_dict:
                for key, value in candle.items():
                    if isinstance(value, (np.integer, np.floating)):
                        candle[key] = float(value)
            
            multi_tf_data[tf_name] = {
                "current_indicators": current_indicators,
                "recent_candles": recent_candles_dict
            }
            
            print(f"Collected {tf_name} data with momentum indicators: {len(df)} candles")
            
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수 =====
def fetch_funding_rate():
    """펀딩비 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {'symbol': 'BTCUSDT'}
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
        print(f"Error fetching funding rate: {e}")
        return {"current_funding_rate": 0, "funding_rate_percentage": 0, "funding_sentiment": "UNKNOWN"}

def fetch_open_interest():
    """미결제약정 데이터 수집"""
    try:
        # 바이낸스 직접 API 호출로 변경
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {'symbol': 'BTCUSDT'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            current_data = response.json()
            current_oi = float(current_data.get('openInterest', 0))
            
            # 과거 데이터도 가져오기 (24시간 전 비교용)
            url_historical = "https://fapi.binance.com/futures/data/openInterestHist"
            params_hist = {
                'symbol': 'BTCUSDT',
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
            
            # 과거 데이터를 가져올 수 없는 경우 현재 데이터만 반환
            return {
                "latest_open_interest": current_oi,
                "oi_change_1h_percentage": 0,
                "oi_change_24h_percentage": 0,
                "oi_trend": "STABLE"
            }
        
    except Exception as e:
        print(f"Error fetching open interest: {e}")
        return {"latest_open_interest": 0, "oi_change_1h_percentage": 0, "oi_trend": "UNKNOWN"}

def fetch_long_short_ratio():
    """롱숏 비율 데이터 수집"""
    try:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {
            'symbol': 'BTCUSDT',
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
        print(f"Error fetching long/short ratio: {e}")
        return {"latest_long_short_ratio": 1.0, "ls_sentiment": "UNKNOWN", "contrarian_signal": "NO_DATA"}

def fetch_liquidation_data():
    """청산 데이터 수집 (24시간)"""
    try:
        url = "https://fapi.binance.com/fapi/v1/forceOrders"
        params = {
            'symbol': 'BTCUSDT',
            'limit': 100
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            long_liquidations = [order for order in data if order['side'] == 'SELL']
            short_liquidations = [order for order in data if order['side'] == 'BUY']
            
            total_long_liq = sum([float(order['executedQty']) for order in long_liquidations])
            total_short_liq = sum([float(order['executedQty']) for order in short_liquidations])
            
            liq_ratio = total_long_liq / (total_short_liq + 0.0001)  # 0으로 나누기 방지
            
            return {
                "long_liquidations_btc": total_long_liq,
                "short_liquidations_btc": total_short_liq,
                "liquidation_ratio": liq_ratio,
                "liquidation_pressure": "LONG_SQUEEZE" if liq_ratio > 2 else 
                                      "SHORT_SQUEEZE" if liq_ratio < 0.5 else "BALANCED"
            }
    except Exception as e:
        print(f"Error fetching liquidation data: {e}")
        return {"liquidation_pressure": "UNKNOWN", "liquidation_ratio": 1.0}

def fetch_market_sentiment():
    """모든 시장 심리 지표를 통합 수집"""
    print("Collecting market sentiment data...")
    
    funding_data = fetch_funding_rate()
    oi_data = fetch_open_interest()
    ls_ratio_data = fetch_long_short_ratio()
    liquidation_data = fetch_liquidation_data()
    
    market_sentiment = {
        "funding_rate": funding_data,
        "open_interest": oi_data,
        "long_short_ratio": ls_ratio_data,
        "liquidations": liquidation_data
    }
    
    print(f"Market Sentiment - Funding: {funding_data['funding_rate_percentage']:.4f}%, "
          f"LS Ratio: {ls_ratio_data['latest_long_short_ratio']:.2f}, "
          f"OI Trend: {oi_data['oi_trend']}")
    
    return market_sentiment

# ===== 포지션 관리 함수 =====
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    """포지션 종료 시 데이터베이스를 업데이트하고 결과를 표시합니다"""
    if current_trade_id is None:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            current_trade_id = latest_trade['id']
    
    if current_trade_id:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            entry_price = latest_trade['entry_price']
            action = latest_trade['action']
            
            if action == 'long':
                profit_loss = (current_price - entry_price) * amount
                profit_loss_percentage = (current_price / entry_price - 1) * 100
            else:
                profit_loss = (entry_price - current_price) * amount
                profit_loss_percentage = (1 - current_price / entry_price) * 100
                
            update_trade_status(
                current_trade_id,
                'CLOSED',
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            
            print(f"\n=== Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("=======================")
            
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

# ===== 메인 프로그램 시작 =====
print("\n=== Bitcoin Day Trading Bot Started (Gemini Pro) ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Strategy: Day Trading (5m/15m/1h analysis)")
print("AI Engine: Google Gemini Pro")
print("Leverage Range: 5-50x (Dynamic)")
print("SL/TP Range: 1-6% (ATR-based)")
print("Market Sentiment: Funding Rate, OI, L/S Ratio, Liquidations")
print("Momentum Indicators: RSI, MACD, Stochastic, Williams %R")
print("Execution Frequency: Every 2 minutes")
print("Trading Hours: 24/7 Unlimited")
print("===============================================\n")

# 데이터베이스 설정
setup_database()

# ===== 메인 트레이딩 루프 =====
while True:
    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

        # ===== 1. 현재 포지션 확인 =====
        current_side = None
        amount = 0
        
        positions = exchange.fetch_positions([symbol])
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if amt > 0:
                    current_side = 'long'
                    amount = amt
                elif amt < 0:
                    current_side = 'short'
                    amount = abs(amt)
        
        current_trade = get_latest_open_trade()
        current_trade_id = current_trade['id'] if current_trade else None
        
        # ===== 2. 포지션이 있는 경우 처리 =====
        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")
            
            if not current_trade:
                temp_trade_data = {
                    'action': current_side,
                    'entry_price': current_price,
                    'amount': amount,
                    'leverage': 1,
                    'sl_price': 0,
                    'tp_price': 0,
                    'sl_percentage': 0,
                    'tp_percentage': 0,
                    'position_size_percentage': 0,
                    'investment_amount': 0
                }
                current_trade_id = save_trade(temp_trade_data)
                print("새로운 거래 기록 생성 (기존 포지션)")
        
        # ===== 3. 포지션이 없는 경우 처리 =====
        else:
            if current_trade:
                handle_position_closure(current_price, current_trade['action'], current_trade['amount'], current_trade_id)
            
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)
                
            time.sleep(5)
            print("No position. Analyzing market for day trading opportunities...")

            # ===== 4. 데이트레이딩 데이터 수집 =====
            multi_tf_data = fetch_multi_timeframe_data()
            market_sentiment = fetch_market_sentiment()
            historical_trading_data = get_historical_trading_data(limit=3)
            performance_metrics = get_performance_metrics()
            
            # ===== 5. Gemini Pro 분석을 위한 데이터 준비 =====
            market_analysis = {
                "current_price": current_price,
                "timeframes": multi_tf_data,
                "market_sentiment": market_sentiment,
                "recent_trades": historical_trading_data,
                "performance_summary": {
                    "total_trades": performance_metrics["total_trades"],
                    "win_rate": performance_metrics["win_rate"],
                    "avg_profit_loss_pct": performance_metrics["avg_profit_loss_percentage"]
                }
            }
            
            # ===== 6. Gemini Pro AI 트레이딩 결정 요청 =====
            system_prompt = """
You are an expert Bitcoin day trader specializing in momentum-based strategies with advanced market sentiment analysis.

TRADING PHILOSOPHY:
- Day trading focus: Hold positions for 30 minutes to 8 hours
- Target 1-4% profit per trade, 2-5 trades per day
- Use momentum indicators + market sentiment for high-conviction entries
- Risk management through dynamic leverage and tight SL/TP

ANALYSIS PROCESS:

1. MOMENTUM ANALYSIS (Multi-timeframe):
   - 1H: Overall trend direction and key levels
   - 15M: Entry timing and momentum confirmation  
   - 5M: Precise entry point and immediate momentum

   Key Momentum Signals:
   - RSI: Look for 40-60 range (not overbought/oversold), divergences
   - MACD: Crossovers, histogram strength, signal line momentum
   - Stochastic: %K/%D crossovers, oversold/overbought bounces
   - Williams %R: Quick reversal signals, momentum confirmation

2. MARKET SENTIMENT ANALYSIS (Critical for Day Trading):
   
   Funding Rate Signals:
   - High positive (>0.01%): Longs overleveraged → SHORT opportunity
   - High negative (<-0.01%): Shorts overleveraged → LONG opportunity
   - Neutral: Use technical analysis primarily
   
   Open Interest Analysis:
   - Price UP + OI UP: Strong trend continuation
   - Price UP + OI DOWN: Weak rally (short covering)
   - Price DOWN + OI UP: Strong selling pressure
   - Price DOWN + OI DOWN: Weak selling (long liquidation)
   
   Long/Short Ratio:
   - >2.0: Extreme long bias → Contrarian SHORT signal
   - <0.8: Extreme short bias → Contrarian LONG signal
   - 1.2-1.8: Normal range, follow technicals
   
   Liquidation Data:
   - Long squeeze in progress: Wait for reversal LONG entry
   - Short squeeze in progress: Wait for reversal SHORT entry

3. HISTORICAL PERFORMANCE REVIEW:
   - Learn from recent trades: What worked? What didn't?
   - Adjust leverage based on recent win rate
   - If win rate <60%, be more selective
   - If on winning streak, maintain strategy

4. DECISION MAKING:
   - Only trade with >65% conviction
   - Combine momentum + sentiment for optimal entry
   - Perfect setup: Technical momentum + Contrarian sentiment signal

5. POSITION SIZING & LEVERAGE:
   - High conviction (80%+): 0.5-0.8 position size, 20-50x leverage
   - Medium conviction (65-80%): 0.2-0.5 position size, 10-25x leverage
   - Low conviction (<65%): NO_POSITION

6. STOP LOSS / TAKE PROFIT:
   - Use ATR for dynamic SL/TP calculation
   - SL: 1-3% (tighter in low volatility, wider in high volatility)
   - TP: 2-6% (minimum 1:2 risk/reward ratio)
   - Adjust based on timeframe alignment

Return ONLY valid JSON (no markdown, no code blocks):
{
  "direction": "LONG/SHORT/NO_POSITION",
  "recommended_position_size": 0.1-0.8,
  "recommended_leverage": 5-50,
  "stop_loss_percentage": 0.01-0.03,
  "take_profit_percentage": 0.02-0.06,
  "reasoning": "Detailed analysis covering momentum indicators, market sentiment signals, and why this setup has high probability of success"
}
"""
            
            try:
                # Gemini Pro API 호출 (JSON 직렬화 안전하게)
                market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
                
                response = model.generate_content([
                    system_prompt,
                    f"Market Analysis Data: {market_analysis_json}"
                ])
                
                response_content = response.text.strip()
                print(f"Raw Gemini response: {response_content}")
                
                # JSON 형식 정리
                if response_content.startswith("```"):
                    content_parts = response_content.split("\n", 1)
                    if len(content_parts) > 1:
                        response_content = content_parts[1]
                    if "```" in response_content:
                        response_content = response_content.rsplit("```", 1)[0]
                    response_content = response_content.strip()
                
                # JSON 파싱
                trading_decision = json.loads(response_content)
                
                print(f"AI 거래 결정 (Gemini 2.5 Flash):")
                print(f"방향: {trading_decision['direction']}")
                print(f"추천 포지션 크기: {trading_decision['recommended_position_size']*100:.1f}%")
                print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                print(f"스탑로스 레벨: {trading_decision['stop_loss_percentage']*100:.2f}%")
                print(f"테이크프로핏 레벨: {trading_decision['take_profit_percentage']*100:.2f}%")
                print(f"분석 근거: {trading_decision['reasoning']}")
                
                # AI 분석 결과를 데이터베이스에 저장
                analysis_data = {
                    'current_price': current_price,
                    'direction': trading_decision['direction'],
                    'recommended_position_size': trading_decision['recommended_position_size'],
                    'recommended_leverage': trading_decision['recommended_leverage'],
                    'stop_loss_percentage': trading_decision['stop_loss_percentage'],
                    'take_profit_percentage': trading_decision['take_profit_percentage'],
                    'reasoning': trading_decision['reasoning']
                }
                analysis_id = save_ai_analysis(analysis_data)
                
                action = trading_decision['direction'].lower()
                
                # ===== 7. 트레이딩 결정에 따른 액션 실행 =====
                if action == "no_position":
                    print("현재 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    print(f"이유: {trading_decision['reasoning']}")
                    time.sleep(120)  # 2분 대기
                    continue
                    
                # ===== 8. 투자 금액 및 레버리지 계산 =====
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']
                
                position_size_percentage = trading_decision['recommended_position_size']
                investment_amount = available_capital * position_size_percentage
                
                # 최소 주문 금액 확인
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"최소 주문 금액(100 USDT)으로 조정됨")
                
                print(f"투자 금액: {investment_amount:.2f} USDT")
                
                # BTC 수량 계산
                amount = math.ceil((investment_amount / current_price) * 1000) / 1000
                print(f"주문 수량: {amount} BTC")

                # 레버리지 설정
                recommended_leverage = trading_decision['recommended_leverage']
                exchange.set_leverage(recommended_leverage, symbol)
                print(f"레버리지 설정: {recommended_leverage}x")

                # SL/TP 비율
                sl_percentage = trading_decision['stop_loss_percentage']
                tp_percentage = trading_decision['take_profit_percentage']

                # ===== 9. 포지션 진입 및 SL/TP 주문 실행 =====
                if action == "long":
                    # 롱 포지션 진입
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    
                    sl_price = round(entry_price * (1 - sl_percentage), 2)
                    tp_price = round(entry_price * (1 + tp_percentage), 2)
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'long',
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
                    trade_id = save_trade(trade_data)
                    
                    # AI 분석과 거래 연결
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    print(f"\n=== LONG Position Opened (Day Trading) ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (-{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (+{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"Expected Hold Time: 30min - 8hours")
                    print("=======================================")

                elif action == "short":
                    # 숏 포지션 진입
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    
                    sl_price = round(entry_price * (1 + sl_percentage), 2)
                    tp_price = round(entry_price * (1 - tp_percentage), 2)
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'short',
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
                    trade_id = save_trade(trade_data)
                    
                    # AI 분석과 거래 연결
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    print(f"\n=== SHORT Position Opened (Day Trading) ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (+{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (-{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"Expected Hold Time: 30min - 8hours")
                    print("========================================")
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"Gemini 응답: {response.text}")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Gemini API 오류: {e}")
                time.sleep(30)
                continue

        # ===== 10. 대기 시간 (데이트레이딩 최적화) =====
        if current_side:
            time.sleep(60)  # 포지션 있을 때 1분마다 모니터링
        else:
            time.sleep(120)  # 포지션 없을 때 2분마다 분석

    except Exception as e:
        print(f"\n Main Loop Error: {e}")
        time.sleep(10)