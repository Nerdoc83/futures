"""
AI 멀티코인 데이트레이딩 봇 - Gemini + AI 전담 판단 (v6.8 - 타임컷 제거, TP/SL 비율 통일, 학습 피드백)
--------------------------------------------------------
기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - AI가 모든 판단 담당
- 데이트레이딩 최적화 (3분봉 메인)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini API 기반 AI 분석 (100% AI 의존)
- 동적 레버리지 및 포지션 사이징 (AI 자율 결정)
- AI 기반 동적 SL/TP 설정 (1.5:1 비율 고정)
- 3분봉 5% 급변동 손절: 레버리지 효과 고려한 급격한 변동 시 즉시 손절
- 포지션 종료 시 재탐색: 포지션 종료될 때마다 빈 코인 재스캔
- 상관관계 리스크 관리: 모든 코인 동시 진입 신호 시 최고점수 포지션만 진입
- 합리성 필터: AI의 비현실적인 TP/SL 제안 자동 거부
- 레버리지 필터: AI의 레버리지 제안이 설정 범위를 벗어날 경우 자동 조정
- 실제 포지션 기반 추적: 수동 거래 포함 모든 포지션 추적
- 24시간 무제한 거래
- 통일된 설정: 모든 코인 동일한 최소투자금, 레버리지, TP/SL 범위
- AI 학습 피드백: 판단 근거 로깅 및 성공/실패 패턴 학습
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
        "tp_range": (0.12, 0.40),  # 1.5:1 비율 고려
        "precision": 5
    },
    "ETH": {
        "symbol": "ETH/USDT",
        "binance_symbol": "ETHUSDT",
        "min_investment": 20,
        "leverage_range": (5, 30),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.12, 0.40),  # 1.5:1 비율 고려
        "precision": 4
    },
    "SOL": {
        "symbol": "SOL/USDT",
        "binance_symbol": "SOLUSDT",
        "min_investment": 20,
        "leverage_range": (5, 30),
        "volatility_factor": 1.0,
        "sl_range": (0.08, 0.25),
        "tp_range": (0.12, 0.40),  # 1.5:1 비율 고려
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
    """Exponential Moving Average 계산"""
    return prices.ewm(span=window, adjust=False).mean()

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
    
    # 기존 테이블에 새로운 컬럼들 추가 (마이그레이션)
    new_columns = [
        ('detailed_reasoning', 'TEXT'),
        ('market_conditions', 'TEXT'),
        ('technical_signals', 'TEXT'),
        ('sentiment_factors', 'TEXT')
    ]
    
    for column_name, column_type in new_columns:
        try:
            cursor.execute(f'ALTER TABLE ai_analysis ADD COLUMN {column_name} {column_type}')
            print(f"Added new column: {column_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                # 컬럼이 이미 존재하는 경우 (정상)
                pass
            else:
                print(f"Error adding column {column_name}: {e}")
    
    conn.commit()
    conn.close()
    print("멀티코인 데이터베이스 설정 완료 (마이그레이션 포함)")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장 (상세 근거 포함, 타입 안전성 강화)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 데이터 타입 안전성을 위한 헬퍼 함수
    def safe_string(value):
        if value is None:
            return ''
        elif isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        else:
            return str(value)
    
    def safe_number(value, default=0):
        try:
            if value is None:
                return default
            elif isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                return float(value) if '.' in value else int(value)
            else:
                return default
        except (ValueError, TypeError):
            return default
    
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
        datetime.now(timezone.utc).isoformat(),
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
    
    cursor.execute(update_sql, tuple(update_values))
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
    """과거 거래 내역 가져오기 (기본 버전)"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.coin_symbol, t.action, t.profit_loss
    FROM trades t
    WHERE t.status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        is_win = row["profit_loss"] is not None and row["profit_loss"] > 0
        trade_result = {
            "coin": row["coin_symbol"],
            "direction": row["action"].upper(),
            "result": "WIN" if is_win else "LOSS"
        }
        historical_data.append(trade_result)
    
    conn.close()
    return historical_data

def get_historical_trading_data_with_reasoning(limit=5):
    """과거 거래 내역과 AI 판단 근거를 함께 가져오기 (학습용 확장 버전)"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.coin_symbol, t.action, t.profit_loss, t.leverage, t.sl_percentage, t.tp_percentage,
        a.reasoning, a.detailed_reasoning, a.market_conditions, a.technical_signals, 
        a.sentiment_factors, a.recommended_leverage
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        is_win = row["profit_loss"] is not None and row["profit_loss"] > 0
        trade_result = {
            "coin": row["coin_symbol"],
            "direction": row["action"].upper(),
            "result": "WIN" if is_win else "LOSS",
            "profit_loss": row["profit_loss"],
            "leverage": row["leverage"],
            "sl_percentage": row["sl_percentage"],
            "tp_percentage": row["tp_percentage"],
            "ai_reasoning": row["reasoning"],
            "detailed_reasoning": row["detailed_reasoning"],
            "market_conditions": row["market_conditions"],
            "technical_signals": row["technical_signals"],
            "sentiment_factors": row["sentiment_factors"],
            "recommended_leverage": row["recommended_leverage"]
        }
        historical_data.append(trade_result)
    
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
    WHERE status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    ''')
    
    overall_metrics = cursor.fetchone()
    conn.close()
    
    if overall_metrics is None or overall_metrics[0] == 0:
        return {
            "total_trades": 0, "winning_trades": 0, "win_rate": 0,
            "avg_profit_loss_percentage": 0, "max_profit_percentage": 0,
            "max_loss_percentage": 0
        }
    
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
                            trade_id, 'CLOSED', exit_price=current_price, exit_timestamp=datetime.now(timezone.utc).isoformat(),
                            profit_loss=profit_loss, profit_loss_percentage=profit_loss_percentage
                        )
                        print(f"✅ {coin_symbol} DB 동기화 완료: P/L ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                except Exception as e:
                    print(f"❌ {coin_symbol} 동기화 오류: {e}")
        
    except Exception as e:
        print(f"동기화 프로세스 오류: {e}")

# ===== 실제 포지션 기반 3분봉 급변동 손절 =====
def check_rapid_movement_stop_loss():
    """실제 포지션 기준 3분봉 급격한 변동 감지 시 즉시 손절"""
    closed_count = 0
    
    try:
        for coin_name, coin_config in TRADING_PAIRS.items():
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                
                for position in positions:
                    if position['symbol'] == f"{coin_config['symbol']}:USDT":
                        pos_amt = float(position['info']['positionAmt'])
                        
                        if pos_amt != 0:
                            actual_amount = abs(pos_amt)
                            side = 'long' if pos_amt > 0 else 'short'
                            entry_price = float(position['info']['entryPrice'])
                            
                            notional = actual_amount * entry_price
                            initial_margin = float(position['info']['initialMargin'])
                            leverage = int(notional / initial_margin) if initial_margin > 0 else 10
                            
                            max_leverage = coin_config['leverage_range'][1]
                            leverage = min(leverage, max_leverage)
                            
                            ohlcv = exchange.fetch_ohlcv(coin_config['symbol'], timeframe='3m', limit=2)
                            if len(ohlcv) < 2:
                                continue
                            
                            prev_candle = ohlcv[-2]
                            current_candle = ohlcv[-1]
                            
                            prev_close = prev_candle[4]
                            current_price = current_candle[4]
                            
                            candle_change_pct = ((current_price - prev_close) / prev_close) * 100
                            position_change_pct = candle_change_pct * leverage
                            
                            should_stop_loss = False
                            loss_reason = ""
                            
                            if side == 'long' and position_change_pct <= -5.0:
                                should_stop_loss = True
                                loss_reason = f"급락 (3분봉 {candle_change_pct:.3f}% 하락 → {leverage}x = {position_change_pct:.2f}% 손실)"
                            elif side == 'short' and position_change_pct >= 5.0:
                                should_stop_loss = True
                                loss_reason = f"급등 (3분봉 {candle_change_pct:.3f}% 상승 → {leverage}x = {abs(position_change_pct):.2f}% 손실)"
                            
                            if should_stop_loss:
                                print(f"\n{'='*60}")
                                print(f"🚨 긴급 손절 신호: {coin_name} {side.upper()}")
                                print(f"   사유: {loss_reason}")
                                print(f"   Entry: ${entry_price:,.2f}, Current: ${current_price:,.2f}")
                                print(f"   추정 레버리지: {leverage}x")
                                
                                try:
                                    exchange.cancel_all_orders(coin_config['symbol'])
                                    print(f"   기존 주문 취소 완료")
                                except Exception as e:
                                    print(f"   주문 취소 오류: {e}")
                                
                                try:
                                    params = {'reduceOnly': True}
                                    if side == 'long':
                                        exchange.create_order(coin_config['symbol'], 'market', 'sell', 
                                                            actual_amount, params=params)
                                    else:
                                        exchange.create_order(coin_config['symbol'], 'market', 'buy', 
                                                            actual_amount, params=params)
                                    
                                    if side == 'long':
                                        profit_loss = (current_price - entry_price) * actual_amount
                                        profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
                                    else:
                                        profit_loss = (entry_price - current_price) * actual_amount
                                        profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                                    
                                    print(f"   ✅ 긴급 손절 완료: P/L ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                                    
                                    try:
                                        conn = sqlite3.connect(DB_FILE)
                                        cursor = conn.cursor()
                                        
                                        cursor.execute('''
                                            SELECT id FROM trades 
                                            WHERE coin_symbol = ? 
                                            AND action = ?
                                            AND status = 'OPEN'
                                            ORDER BY timestamp DESC
                                            LIMIT 1
                                        ''', (coin_name, side))
                                        
                                        result = cursor.fetchone()
                                        if result:
                                            trade_id = result[0]
                                            
                                            cursor.execute('''
                                                UPDATE trades 
                                                SET status = 'CLOSED_RAPID_STOP', 
                                                    exit_price = ?, 
                                                    exit_timestamp = ?,
                                                    profit_loss = ?,
                                                    profit_loss_percentage = ?
                                                WHERE id = ?
                                            ''', (current_price, datetime.now().isoformat(), 
                                                  profit_loss, profit_loss_percentage, trade_id))
                                            
                                            conn.commit()
                                            print(f"   DB 업데이트 완료 (Trade ID: {trade_id})")
                                        else:
                                            print(f"   DB에 해당 거래 기록 없음 (수동 거래일 가능성)")
                                        
                                        conn.close()
                                        
                                    except Exception as db_error:
                                        print(f"   DB 업데이트 오류: {db_error}")
                                    
                                    closed_count += 1
                                    
                                except Exception as e:
                                    print(f"   ❌ 긴급 손절 실행 오류: {e}")
                                
                                print(f"{'='*60}")
                                
            except Exception as e:
                print(f"급변동 체크 오류 ({coin_name}): {e}")
                continue
        
        return closed_count
        
    except Exception as e:
        print(f"급변동 손절 체크 전체 오류: {e}")
        return 0

# ===== 데이터 수집 함수 =====
def fetch_multi_timeframe_data_for_coin(symbol):
    """단일 코인의 멀티 타임프레임 데이터 수집"""
    timeframes = {
        "3m": {"timeframe": "3m", "limit": 160},
        "1m": {"timeframe": "1m", "limit": 120},
        "5m": {"timeframe": "5m", "limit": 100},
        "15m": {"timeframe": "15m", "limit": 96},
        "1h": {"timeframe": "1h", "limit": 100}
    }
    multi_tf_data = {}
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_params["timeframe"], limit=tf_params["limit"])
            if not ohlcv or len(ohlcv) < 50: continue
            
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
            
            if tf_name == '1h':
                df['EMA_50'] = calculate_ema(df['close'], 50)

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

# ===== 멀티코인 데이터 통합 수집 함수 =====
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

def analyze_multi_coin_with_ai(all_coins_data, historical_data, performance_metrics):
    """3개 코인을 AI로 분석하고 최고 확률의 거래 기회를 선택 (ULTRA-CONSERVATIVE 모드)"""
    
    system_prompt = """
You are an EXTREMELY CONSERVATIVE cryptocurrency day trader AI. Your past performance shows 33.3% win rate, which is UNACCEPTABLE. You must make RADICAL changes.

**CRITICAL PERFORMANCE ANALYSIS:**
Your "High-Momentum Bullish Breakout" pattern FAILED 6 out of 7 times (14.3% win rate). This pattern is NOW BANNED.

**MANDATORY ULTRA-CONSERVATIVE RULES:**

1. **BANNED PATTERNS (NEVER TRADE THESE):**
   - HIGH-MOMENTUM BULLISH BREAKOUT (proven failure - 14.3% win rate)
   - Any "breakout continuation" setups
   - Any "momentum chasing" trades
   - Trading during high volatility spikes

2. **EXTREME SELECTIVITY REQUIREMENTS:**
   - Conviction score MUST be >95 (not 85, not 90, but 95+)
   - RSI EXTREME CONDITIONS ONLY:
     * LONG trades: Only when RSI14 < 35 AND price near strong support
     * SHORT trades: Only when RSI14 > 65 AND price near strong resistance
   - ALL three timeframes (1h, 15m, 3m) must align perfectly
   - Volume must be >150% of 20-period average

3. **FOCUS ON MEAN REVERSION:**
   - Look for OVERSOLD bounces, not breakouts
   - Look for OVERBOUGHT reversals, not momentum continuation
   - Wait for clear rejection at support/resistance levels
   - Prefer counter-trend trades over trend-following

4. **DYNAMIC LEVERAGE BASED ON VOLATILITY:**
   You MUST vary leverage based on market volatility (ATR/price ratio):
   - LOW volatility (ATR/price < 1.5%): Use 18-25x leverage
   - MEDIUM volatility (ATR/price 1.5-3%): Use 12-18x leverage
   - HIGH volatility (ATR/price 3-5%): Use 8-12x leverage
   - VERY HIGH volatility (ATR/price > 5%): Use 5-8x leverage only
   
   NEVER use 10x repeatedly. Calculate ATR/price ratio and choose accordingly.

5. **MANDATORY FILTERS:**
   - Price must be at a significant support (for longs) or resistance (for shorts)
   - No trades if any timeframe shows indecision (spinning tops, doji)
   - Must have clear invalidation level for stop loss
   - Risk/reward must be minimum 1.5:1 (TP = SL × 1.5)

**CRITICAL: MOST OF THE TIME, RETURN EMPTY LIST**
Only trade when ALL conditions are met. It's better to miss trades than lose money.

**JSON RESPONSE:**
- "coin": BTC/ETH/SOL
- "score": Must be >95
- "direction": long/short
- "recommended_leverage": Calculate based on ATR/price ratio above
- "stop_loss_percentage": 5-12%
- "take_profit_percentage": SL × 1.5 exactly
- "market_conditions": Explain why this is NOT a banned pattern
- "technical_signals": RSI extreme + support/resistance confirmation
- "detailed_reasoning": Why conviction is >95

If no setup meets ALL criteria, return empty "trading_opportunities" list.
"""

    try:
        market_analysis = {
            "coins_data": all_coins_data,
            "recent_trades_with_reasoning": historical_data,
            "performance_summary": performance_metrics
        }
        
        market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content(
            [system_prompt, f"Multi-Coin Market Analysis with Learning History: {market_analysis_json}"],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3  # 더 보수적으로 0.4 → 0.3
            ),
            safety_settings=safety_settings
        )
        
        response_content = response.text.strip().replace("```json", "").replace("```", "")
        trading_decision = json.loads(response_content)
        
        print("\n=== AI Analysis Result (ULTRA-CONSERVATIVE MODE) ===")
        
        if isinstance(trading_decision, dict):
            opportunities = trading_decision.get('trading_opportunities', [])
        elif isinstance(trading_decision, list):
            opportunities = trading_decision
        else:
            opportunities = []

        for opp in opportunities:
            coin = opp.get('coin','N/A')
            direction = opp.get('direction','N/A')
            score = opp.get('score',0)
            leverage = opp.get('recommended_leverage', 10)
            print(f"- {coin}: {direction} (Score: {score}, Leverage: {leverage}x)")
            
            print(f"  📊 Market Conditions: {opp.get('market_conditions', 'N/A')}")
            print(f"  🔧 Technical Signals: {opp.get('technical_signals', 'N/A')}")
            print(f"  💭 Sentiment Factors: {opp.get('sentiment_factors', 'N/A')}")
            print(f"  📝 Detailed Reasoning: {opp.get('detailed_reasoning', 'N/A')}")
        
        if len(opportunities) == 0:
            print("✅ AI correctly rejected all setups - waiting for high-conviction opportunities")
        
        if isinstance(trading_decision, list):
            return {"trading_opportunities": trading_decision}
        return trading_decision
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {"trading_opportunities": []}

# ===== 거래 실행 함수 (1.5:1 비율 강제 + 상세 로깅) =====
def execute_single_trade(coin_name, opportunity, available_capital, all_coins_data):
    """단일 코인 거래 실행"""
    try:
        coin_config = TRADING_PAIRS[coin_name]
        symbol = coin_config["symbol"]
        action = opportunity.get("direction", "").lower()
        
        ema_1h = all_coins_data[coin_name].get("technical_data", {}).get("1h", {}).get("current_indicators", {}).get("EMA_50")
        current_price = all_coins_data[coin_name].get("current_price")

        if ema_1h and current_price:
            if action == 'long' and current_price < ema_1h:
                print(f"❌ Trade Rejected (Trend Filter): {coin_name} LONG attempt below 1H EMA 50 (${ema_1h:,.2f}).")
                return None
            if action == 'short' and current_price > ema_1h:
                print(f"❌ Trade Rejected (Trend Filter): {coin_name} SHORT attempt above 1H EMA 50 (${ema_1h:,.2f}).")
                return None
        else:
            print(f"   Warning: 1H EMA data not available for {coin_name}, trend filter skipped.")

        def parse_percentage(value):
            if isinstance(value, list): value = value[0] if value else "0"
            if isinstance(value, str): value = value.strip().replace('%', '')
            try:
                num_value = float(value)
                return num_value
            except (ValueError, TypeError):
                return 0.0

        def parse_leverage(value):
            if isinstance(value, list): value = value[0] if value else "1"
            if isinstance(value, str): value = value.strip().lower().replace('x', '')
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return 1

        pos_size_pct = parse_percentage(opportunity.get('recommended_position_size', 0)) / 100.0
        leverage = parse_leverage(opportunity.get('recommended_leverage', 1))
        sl_pct = parse_percentage(opportunity.get('stop_loss_percentage', 0))
        tp_pct = parse_percentage(opportunity.get('take_profit_percentage', 0))

        # 1.5:1 비율 강제 검증
        expected_tp = sl_pct * 1.5
        if abs(tp_pct - expected_tp) > 0.5:
            print(f"\n❌ Trade Rejected (Invalid TP/SL Ratio): {coin_name}")
            print(f"   AI proposed SL: {sl_pct:.2f}%, TP: {tp_pct:.2f}%")
            print(f"   Required 1.5:1 ratio TP: {expected_tp:.2f}%")
            return None

        MAX_ACCEPTABLE_SL_PERCENTAGE = 15.0
        MIN_ACCEPTABLE_TP_PERCENTAGE = 4.2
        
        if sl_pct > MAX_ACCEPTABLE_SL_PERCENTAGE:
            print(f"\n❌ Trade Rejected (Excessive Risk): AI proposed SL of {sl_pct:.2f}%, which exceeds the hard limit of {MAX_ACCEPTABLE_SL_PERCENTAGE}%.")
            return None
            
        if tp_pct < MIN_ACCEPTABLE_TP_PERCENTAGE:
            print(f"\n❌ Trade Rejected (Insufficient Profit Target): AI proposed TP of {tp_pct:.2f}%, below the minimum required target of {MIN_ACCEPTABLE_TP_PERCENTAGE}%.")
            return None
        
        min_lev, max_lev = coin_config['leverage_range']
        original_leverage = leverage
        leverage = max(min_lev, min(leverage, max_lev)) 
        if original_leverage != leverage:
            print(f"   Leverage Adjusted: AI recommended {original_leverage}x, adjusted to {leverage}x (Range: {min_lev}x-{max_lev}x).")
        
        sl_price_check = current_price * (1 - (sl_pct / 100.0 / leverage)) if action == "long" else current_price * (1 + (sl_pct / 100.0 / leverage))
        tp_price_check = current_price * (1 + (tp_pct / 100.0 / leverage)) if action == "long" else current_price * (1 - (tp_pct / 100.0 / leverage))

        tp_change_pct = abs((tp_price_check / current_price) - 1) * 100
        sl_change_pct = abs((sl_price_check / current_price) - 1) * 100

        REALISTIC_CHANGE_LIMIT = 20.0 
        
        if tp_change_pct > REALISTIC_CHANGE_LIMIT or sl_change_pct > REALISTIC_CHANGE_LIMIT:
            print(f"\n❌ Trade Rejected (Unrealistic Target): {coin_name} {action.upper()}")
            print(f"   AI proposed TP change: {tp_change_pct:.2f}%, SL change: {sl_change_pct:.2f}%")
            return None

        try:
            exchange.cancel_all_orders(symbol)
            print(f"{coin_name} 기존 미체결 주문 취소됨")
        except Exception: pass
        
        margin = available_capital * pos_size_pct
        min_margin = coin_config.get('min_investment', 20)
        if margin < min_margin:
            margin = min_margin
            print(f"최소 투자 마진({min_margin} USDT)으로 조정됨")

        if margin > available_capital:
            print(f"❌ Trade Rejected (Insufficient Margin): Required ${margin:,.2f} > Available ${available_capital:,.2f}")
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
        
        time.sleep(2) 
        
        sl_price = entry_price * (1 - (sl_pct / 100.0 / leverage)) if action == "long" else entry_price * (1 + (sl_pct / 100.0 / leverage))
        tp_price = entry_price * (1 + (tp_pct / 100.0 / leverage)) if action == "long" else entry_price * (1 - (tp_pct / 100.0 / leverage))
        
        sl_placed = False
        tp_placed = False
        
        for i in range(3): 
            try:
                if not sl_placed:
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
                    sl_placed = True
                if not tp_placed:
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell' if action == "long" else 'buy', amount, None, {'stopPrice': tp_price, 'reduceOnly': True})
                    tp_placed = True
                
                if sl_placed and tp_placed:
                    print(f"\n{'='*50}")
                    print(f"✅ {coin_name} {action.upper()} POSITION OPENED")
                    print(f"{'='*50}")
                    print(f"📊 Entry: ${entry_price:,.2f}")
                    print(f"🎯 Take Profit: ${tp_price:,.2f} ({tp_pct:.1f}%)")
                    print(f"🛡️  Stop Loss: ${sl_price:,.2f} ({sl_pct:.1f}%)")
                    print(f"⚖️  TP/SL Ratio: 1.5:1 ✓")
                    print(f"🔢 Leverage: {leverage}x")
                    print(f"💰 Investment: ${margin:,.2f}")
                    print(f"📝 AI Decision Reasoning:")
                    print(f"    💭 {opportunity.get('detailed_reasoning', 'N/A')}")
                    print(f"    📈 Market: {opportunity.get('market_conditions', 'N/A')}")
                    print(f"    🔧 Technical: {opportunity.get('technical_signals', 'N/A')}")
                    print(f"    😌 Sentiment: {opportunity.get('sentiment_factors', 'N/A')}")
                    print(f"{'='*50}")
                    break
            except Exception as e:
                print(f"   Attempt {i+1} to set TP/SL failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)

        if not (sl_placed and tp_placed):
            print(f"   ❌ CRITICAL: Failed to set TP/SL for {coin_name} after 3 attempts. Closing position for safety.")
            if action == "long":
                exchange.create_market_sell_order(symbol, amount, {'reduceOnly': True})
            else:
                exchange.create_market_buy_order(symbol, amount, {'reduceOnly': True})
            return None
        
        trade_data = {
            'action': action, 'entry_price': entry_price, 'amount': amount, 'leverage': leverage,
            'sl_price': sl_price, 'tp_price': tp_price, 'sl_percentage': sl_pct, 'tp_percentage': tp_pct,
            'position_size_percentage': pos_size_pct, 'investment_amount': margin
        }
        trade_id = save_trade(trade_data, coin_name)
        
        # AI 분석 결과를 상세 근거와 함께 저장
        analysis_data = {
            'selected_coin': coin_name,
            'current_price': current_price,
            'direction': action,
            'recommended_position_size': pos_size_pct * 100,
            'recommended_leverage': leverage,
            'stop_loss_percentage': sl_pct,
            'take_profit_percentage': tp_pct,
            'reasoning': opportunity.get('reasoning', ''),
            'detailed_reasoning': opportunity.get('detailed_reasoning', ''),
            'market_conditions': opportunity.get('market_conditions', ''),
            'technical_signals': opportunity.get('technical_signals', ''),
            'sentiment_factors': opportunity.get('sentiment_factors', '')
        }
        save_ai_analysis(analysis_data, trade_id)
        
        return margin
        
    except Exception as e:
        print(f"Single trade execution error for {coin_name}: {e}")
        return None

# ===== 포지션 종료 감지 함수 =====
def detect_position_closure():
    """포지션 종료를 감지하여 True/False 반환"""
    try:
        current_positions = check_current_positions()
        return len(current_positions) == 0
    except Exception as e:
        print(f"포지션 종료 감지 오류: {e}")
        return False

# ===== 메인 프로그램 시작 =====
def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v6.8 - 포지션 최적화) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Strategy: AI 완전 자율 판단, 3분봉 메인, 1.5:1 TP/SL 비율 고정")
    print("Learning: AI 판단 근거 로깅 및 성공/실패 패턴 학습")
    print("Risk Management: 3분봉 5% 급변동 손절, 상관관계 필터")
    print("Position Strategy: 3개 포지션 유지 최적화 - 빈 슬롯 있으면 10분마다 스캔")
    print("Unified Settings: 모든 코인 동일한 설정 (최소 20 USDT, 5-30x 레버리지)")
    print("===============================================\n")

    setup_database()
    
    last_rapid_check_time = datetime.now(timezone.utc)
    last_partial_scan_time = datetime.now(timezone.utc) - timedelta(minutes=10)  # 초기 스캔을 위해 10분 전으로 설정

    while True:
        try:
            current_utc = datetime.now(timezone.utc)
            current_local = current_utc.astimezone()
            print(f"\n[{current_local.strftime('%H:%M:%S')}] === Market Check ===")

            sync_database_with_positions()
            
            # 1. 급변동 손절 체크 (30초마다)
            if (datetime.now(timezone.utc) - last_rapid_check_time).seconds >= 30:
                if check_rapid_movement_stop_loss() > 0:
                    last_rapid_check_time = datetime.now(timezone.utc)
                    time.sleep(10)
                    continue
                last_rapid_check_time = datetime.now(timezone.utc)
            
            current_positions = check_current_positions()
            total_coins = len(TRADING_PAIRS)  # 3개 코인
            current_position_count = len(current_positions)
            
            print(f"현재 포지션: {current_position_count}/{total_coins}")
            
            # 2. 포지션 전략: 빈 슬롯이 있으면 적극적으로 채우기
            if current_position_count < total_coins:
                print(f"빈 포지션 슬롯 감지: {total_coins - current_position_count}개")
                
                # 10분마다 또는 포지션이 없는 경우 즉시 스캔
                time_since_last_scan = (current_utc - last_partial_scan_time).total_seconds() / 60
                should_scan = False
                
                if current_position_count == 0:
                    print("포지션 없음 - 즉시 전체 스캔")
                    should_scan = True
                elif time_since_last_scan >= 10:
                    print(f"10분 경과 ({time_since_last_scan:.1f}분) - 빈 슬롯 스캔")
                    should_scan = True
                else:
                    remaining_time = 10 - time_since_last_scan
                    print(f"다음 스캔까지 {remaining_time:.1f}분 남음")
                
                if should_scan:
                    # 빈 코인 식별
                    occupied_coins = {p['coin'] for p in current_positions}
                    available_coins = set(TRADING_PAIRS.keys()) - occupied_coins
                    
                    print(f"점유된 코인: {occupied_coins if occupied_coins else '없음'}")
                    print(f"스캔 대상 코인: {available_coins}")
                    
                    if available_coins:
                        # 빈 코인들만 데이터 수집
                        available_data = {}
                        for coin in available_coins:
                            try:
                                config = TRADING_PAIRS[coin]
                                print(f"Collecting {coin} data...")
                                tech = fetch_multi_timeframe_data_for_coin(config["symbol"])
                                senti = fetch_market_sentiment_for_coin(config["binance_symbol"])
                                price = tech.get("3m", {}).get("current_indicators", {}).get("close", 0)
                                if price > 0:
                                    available_data[coin] = {"technical_data": tech, "sentiment_data": senti, "current_price": price}
                                    print(f"{coin}: ${price:,.2f}")
                            except Exception as e:
                                print(f"Error collecting {coin} data: {e}")
                        
                        if available_data:
                            hist_data = get_historical_trading_data_with_reasoning(5)
                            perf_metrics = get_performance_metrics()
                            decision = analyze_multi_coin_with_ai(available_data, hist_data, perf_metrics)
                            opportunities = decision.get('trading_opportunities', [])
                            
                            # 상관관계 리스크 관리 (2개 이상 기회가 있고 모두 같은 방향인 경우)
                            if len(opportunities) >= 2:
                                directions = {opp.get('direction') for opp in opportunities}
                                if len(directions) == 1:
                                    print(f"\n상관관계 리스크 감지: {len(opportunities)}개 기회 모두 {list(directions)[0]} 방향")
                                    highest_score_opp = max(opportunities, key=lambda x: x.get('score', 0))
                                    print(f"최고점수 선택: {highest_score_opp.get('coin')} (점수: {highest_score_opp.get('score', 0)})")
                                    opportunities = [highest_score_opp]
                            
                            if opportunities:
                                balance = exchange.fetch_balance()['USDT']['free']
                                available_capital = balance * 0.98
                                
                                print(f"가용 자본: ${available_capital:,.2f}")
                                
                                for opp in opportunities:
                                    used_margin = execute_single_trade(opp['coin'], opp, available_capital, available_data)
                                    if used_margin is not None and used_margin > 0:
                                        available_capital -= used_margin
                                        current_position_count += 1
                                        print(f"포지션 추가됨: {current_position_count}/{total_coins}")
                                    time.sleep(5)
                            else:
                                print("AI가 적합한 기회를 찾지 못함")
                        
                    last_partial_scan_time = current_utc
                    time.sleep(60)  # 스캔 후 1분 대기
                else:
                    time.sleep(60)  # 스캔하지 않는 경우 1분 대기
            
            else:  # 모든 포지션이 차있는 경우 (3/3)
                print("모든 포지션 활성 - 종료 대기 모드")
                
                # 포지션 종료 감지
                prev_position_count = current_position_count
                time.sleep(60)  # 1분 대기
                
                # 포지션 재확인
                new_positions = check_current_positions()
                new_position_count = len(new_positions)
                
                if new_position_count < prev_position_count:
                    closed_count = prev_position_count - new_position_count
                    print(f"\n=== 포지션 종료 감지: {prev_position_count} → {new_position_count} ({closed_count}개 종료) ===")
                    
                    # 즉시 다음 사이클에서 빈 슬롯 스캔하도록 설정
                    last_partial_scan_time = current_utc - timedelta(minutes=10)
                    
                    print("다음 사이클에서 빈 슬롯 즉시 스캔 예정")

        except Exception as e:
            print(f"\nMain loop error: {e.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    main()