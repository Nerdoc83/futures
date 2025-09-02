"""
AI 멀티코인 데이트레이딩 봇 - Gemini + AI 전담 판단 (v6.7 - 거래 비용 및 시장 상황 필터 강화)
--------------------------------------------------------
기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - AI가 모든 판단 담당
- 데이트레이딩 최적화 (3분봉 메인)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini API 기반 AI 분석 (100% AI 의존)
- 동적 레버리지 및 포지션 사이징 (AI 자율 결정)
- AI 기반 동적 SL/TP 설정
- 60분 타임컷: 60분 경과 시 수익/손실 무관 포지션 자동 정리 (실제 포지션 기반)
- 3분봉 5% 급변동 손절: 레버리지 효과 고려한 급격한 변동 시 즉시 손절
- 10분 부분 스캔: 포지션 보유 시 10분마다 비어있는 코인 재스캔
- 상관관계 리스크 관리: 모든 코인 동시 진입 신호 시 최고점수 포지션만 진입
- 합리성 필터: AI의 비현실적인 TP/SL 제안 자동 거부
- 레버리지 필터: AI의 레버리지 제안이 설정 범위를 벗어날 경우 자동 조정
- 실제 포지션 기반 추적: 수동 거래 포함 모든 포지션 추적
- 24시간 무제한 거래
- 최소 투자금액: 40-10 USDT
- 수정사항(v6.7): 거래비용(수수료, 슬리피지) 고려 및 횡보장 회피를 위한 AI 프롬프트 강화, 코드 레벨 최소이익률 필터 추가
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
# 'timezone' 추가
from datetime import datetime, timedelta, timezone

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
        'versions': {
            'fapiPrivate': 'v2',
            'fapiPublic': 'v1',
        },
    }
})

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')

DB_FILE = "multi_coin_daytrading.db"

# ===== 포지션 추적 딕셔너리 (프로그램 실행 중 메모리에 유지) =====
# 각 포지션의 첫 발견 시간을 추적
POSITION_TRACKER = {}

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
    """과거 거래 내역 가져오기 (None 값 처리 강화)"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.coin_symbol, t.action, t.profit_loss
    FROM trades t
    WHERE t.status IN ('CLOSED', 'CLOSED_TIMEOUT', 'CLOSED_RAPID_STOP')
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        # profit_loss가 None이 아닌 경우에만 > 0 비교를 수행하여 TypeError 방지
        is_win = row["profit_loss"] is not None and row["profit_loss"] > 0
        trade_result = {
            "coin": row["coin_symbol"],
            "direction": row["action"].upper(),
            "result": "WIN" if is_win else "LOSS"
        }
        historical_data.append(trade_result)
    
    conn.close()
    return historical_data

def get_performance_metrics():
    """거래 성과 메트릭스를 계산합니다 (안정성 강화)"""
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
    WHERE status IN ('CLOSED', 'CLOSED_TIMEOUT', 'CLOSED_RAPID_STOP')
    ''')
    
    overall_metrics = cursor.fetchone()
    conn.close()
    
    # 거래 내역이 전혀 없을 경우를 처리하여 초기 실행 오류 방지
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

# ===== 포지션 진입 시간 찾기 함수 =====
def get_position_entry_time(symbol, position_side, position_amount):
    """거래 내역에서 포지션 진입 시간을 찾습니다"""
    try:
        # 최근 500개 거래 내역 조회 (더 많이 필요하면 증가)
        trades = exchange.fetch_my_trades(symbol, limit=500)
        trades.sort(key=lambda x: x['timestamp'], reverse=True)  # 최신순 정렬
        
        cumulative_amount = 0
        entry_time = None
        
        for trade in trades:
            # 포지션 방향과 일치하는 거래만 확인
            if (position_side == 'long' and trade['side'] == 'buy') or \
               (position_side == 'short' and trade['side'] == 'sell'):
                cumulative_amount += trade['amount']
                entry_time = trade['datetime']
                
                # 누적 수량이 현재 포지션 수량과 일치하면 진입 시점 찾음
                if cumulative_amount >= position_amount * 0.99:  # 0.99는 부동소수점 오차 고려
                    return datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        
        # 찾지 못한 경우 None 반환
        return None
    except Exception as e:
        print(f"포지션 진입 시간 조회 오류: {e}")
        return None

# ===== 봇 시작 시 포지션 추적 초기화 함수 =====
def initialize_position_tracker_on_startup():
    """봇 시작 시 기존 포지션들을 POSITION_TRACKER에 등록"""
    global POSITION_TRACKER
    current_time = datetime.now(timezone.utc)
    
    print("기존 포지션 추적 초기화 중...")
    for coin_name, coin_config in TRADING_PAIRS.items():
        try:
            positions = exchange.fetch_positions([coin_config["symbol"]])
            for position in positions:
                if position['symbol'] == f"{coin_config['symbol']}:USDT":
                    pos_amt = float(position['info']['positionAmt'])
                    if pos_amt != 0:
                        position_key = f"{coin_name}_{position['side']}"
                        # 실제 진입 시간을 조회해서 등록
                        entry_time = get_position_entry_time(
                            coin_config["symbol"], 
                            position['side'], 
                            abs(pos_amt)
                        )
                        if entry_time:
                            POSITION_TRACKER[position_key] = entry_time
                            elapsed = (current_time - entry_time).total_seconds() / 60
                            print(f"기존 포지션 등록: {coin_name} {position['side'].upper()} ({elapsed:.1f}분 경과)")
                        else:
                            # 진입 시간을 찾지 못한 경우 현재 시간으로 대체
                            POSITION_TRACKER[position_key] = current_time
                            print(f"기존 포지션 등록: {coin_name} {position['side'].upper()} (진입 시간 불명, 현재부터 추적)")
        except Exception as e:
            print(f"초기화 중 {coin_name} 오류: {e}")
    
    if POSITION_TRACKER:
        print(f"총 {len(POSITION_TRACKER)}개의 기존 포지션이 추적에 등록되었습니다.")
    else:
        print("기존 포지션이 없습니다.")

# ===== 실제 포지션 기반 60분 타임아웃 체크 (수정됨) =====
def check_trade_timeout():
    """실제 거래소 포지션을 기준으로 60분 타임아웃 체크 (생성 시간 기반) - SQL 오류 수정"""
    global POSITION_TRACKER
    timeout_count = 0
    current_time = datetime.now(timezone.utc)
    
    try:
        # 모든 코인의 실제 포지션 확인
        for coin_name, coin_config in TRADING_PAIRS.items():
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                
                for position in positions:
                    if position['symbol'] == f"{coin_config['symbol']}:USDT":
                        pos_amt = float(position['info']['positionAmt'])
                        
                        if pos_amt != 0:  # 포지션이 있는 경우
                            position_key = f"{coin_name}_{position['side']}"
                            actual_amount = abs(pos_amt)
                            
                            # 처음 발견한 포지션인 경우 실제 진입 시간 조회
                            if position_key not in POSITION_TRACKER:
                                entry_time = get_position_entry_time(
                                    coin_config["symbol"], 
                                    position['side'], 
                                    actual_amount
                                )
                                
                                if entry_time:
                                    POSITION_TRACKER[position_key] = entry_time
                                    time_since_entry = (current_time - entry_time).total_seconds() / 60
                                    print(f"포지션 발견: {coin_name} {position['side'].upper()}")
                                    print(f"  진입 시간: {entry_time.strftime('%H:%M:%S')}")
                                    print(f"  경과 시간: {time_since_entry:.1f}분")
                                else:
                                    # 진입 시간을 찾지 못한 경우 현재 시간으로 대체
                                    POSITION_TRACKER[position_key] = current_time
                                    print(f"포지션 감지: {coin_name} {position['side'].upper()} (진입 시간 불명, 현재부터 추적)")
                                continue
                            
                            # 60분 경과 체크
                            entry_time = POSITION_TRACKER[position_key]
                            time_elapsed = (current_time - entry_time).total_seconds() / 60
                            
                            if time_elapsed >= 60:
                                unrealized_pnl = float(position['info']['unRealizedProfit'])
                                entry_price = float(position['info']['entryPrice'])
                                current_price = exchange.fetch_ticker(coin_config['symbol'])['last']
                                
                                print(f"\n{'='*60}")
                                print(f"⏰ 60분 타임컷 실행: {coin_name} {position['side'].upper()}")
                                print(f"   포지션 진입: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"   현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"   경과 시간: {time_elapsed:.1f}분")
                                print(f"   Entry: ${entry_price:,.2f}, Current: ${current_price:,.2f}")
                                print(f"   미실현 손익: ${unrealized_pnl:,.2f}")
                                
                                # 기존 주문 취소
                                try:
                                    exchange.cancel_all_orders(coin_config['symbol'])
                                    print(f"   기존 주문 취소 완료")
                                except Exception as e:
                                    print(f"   주문 취소 오류: {e}")
                                
                                # 포지션 청산
                                try:
                                    params = {'reduceOnly': True}
                                    if pos_amt > 0:  # Long position
                                        exchange.create_order(coin_config['symbol'], 'market', 'sell', 
                                                            actual_amount, params=params)
                                    else:  # Short position
                                        exchange.create_order(coin_config['symbol'], 'market', 'buy', 
                                                            actual_amount, params=params)
                                    
                                    print(f"   ✅ 타임컷 청산 완료!")
                                    
                                    # 추적 딕셔너리에서 제거
                                    del POSITION_TRACKER[position_key]
                                    timeout_count += 1
                                    
                                    # DB에 기록이 있다면 업데이트 (봇이 연 포지션인 경우) - SQL 수정
                                    try:
                                        conn = sqlite3.connect(DB_FILE)
                                        cursor = conn.cursor()
                                        
                                        # 먼저 해당 거래의 ID를 찾기
                                        cursor.execute('''
                                            SELECT id FROM trades 
                                            WHERE coin_symbol = ? 
                                            AND status = 'OPEN'
                                            ORDER BY timestamp DESC
                                            LIMIT 1
                                        ''', (coin_name,))
                                        
                                        result = cursor.fetchone()
                                        if result:
                                            trade_id = result[0]
                                            profit_loss_percentage = (unrealized_pnl/entry_price)*100 if entry_price > 0 else 0
                                            
                                            # ID를 이용해 업데이트
                                            cursor.execute('''
                                                UPDATE trades 
                                                SET status = 'CLOSED_TIMEOUT', 
                                                    exit_price = ?, 
                                                    exit_timestamp = ?,
                                                    profit_loss = ?,
                                                    profit_loss_percentage = ?
                                                WHERE id = ?
                                            ''', (current_price, current_time.isoformat(), 
                                                  unrealized_pnl, profit_loss_percentage, trade_id))
                                            
                                            conn.commit()
                                            print(f"   DB 업데이트 완료 (Trade ID: {trade_id})")
                                        else:
                                            print(f"   DB에 해당 거래 기록 없음 (수동 거래일 가능성)")
                                        
                                        conn.close()
                                        
                                    except Exception as db_error:
                                        print(f"   DB 업데이트 오류: {db_error}")
                                    
                                except Exception as e:
                                    print(f"   ❌ 포지션 청산 오류: {e}")
                                
                                print(f"{'='*60}")
                            else:
                                remaining = 60 - time_elapsed
                                if remaining <= 10:  # 10분 이하 남았을 때 경고
                                    print(f"⚠️ {coin_name} {position['side'].upper()} - 타임컷까지 {remaining:.1f}분 남음")
                                    
                        else:  # 포지션이 없는 경우
                            # 추적 중이던 포지션이 사라진 경우 (수동 또는 자동 청산)
                            for side in ['long', 'short']:
                                position_key = f"{coin_name}_{side}"
                                if position_key in POSITION_TRACKER:
                                    del POSITION_TRACKER[position_key]
                                    print(f"포지션 종료 감지: {coin_name} {side.upper()} (추적 중단)")
                                    
            except Exception as e:
                print(f"{coin_name} 포지션 체크 오류: {e}")
                continue
                
    except Exception as e:
        print(f"타임아웃 체크 전체 오류: {e}")
    
    # 추적 중인 모든 포지션 상태 출력 (디버깅용)
    if POSITION_TRACKER:
        print(f"현재 추적 중인 포지션: {len(POSITION_TRACKER)}개")
        for key, time_val in POSITION_TRACKER.items():
            elapsed = (current_time - time_val).total_seconds() / 60
            time_val_local = time_val.astimezone()  # UTC를 로컬로 변환
            print(f"  - {key}: {elapsed:.1f}분 경과 (진입: {time_val_local.strftime('%H:%M:%S')} 로컬)")
    
    return timeout_count

# ===== 실제 포지션 기반 3분봉 급변동 손절 (SQL 수정) =====
def check_rapid_movement_stop_loss():
    """실제 포지션 기준 3분봉 급격한 변동 감지 시 즉시 손절 (수동 거래 포함) - SQL 오류 수정"""
    closed_count = 0
    
    try:
        # 모든 코인의 실제 포지션 확인
        for coin_name, coin_config in TRADING_PAIRS.items():
            try:
                positions = exchange.fetch_positions([coin_config["symbol"]])
                
                for position in positions:
                    if position['symbol'] == f"{coin_config['symbol']}:USDT":
                        pos_amt = float(position['info']['positionAmt'])
                        
                        if pos_amt != 0:  # 포지션이 있는 경우
                            # 포지션 정보
                            actual_amount = abs(pos_amt)
                            side = 'long' if pos_amt > 0 else 'short'
                            entry_price = float(position['info']['entryPrice'])
                            
                            # 레버리지 추정 (포지션의 명목가치 / 증거금)
                            notional = actual_amount * entry_price
                            initial_margin = float(position['info']['initialMargin'])
                            leverage = int(notional / initial_margin) if initial_margin > 0 else 10
                            
                            # 코인별 최대 레버리지로 제한
                            max_leverage = coin_config['leverage_range'][1]
                            leverage = min(leverage, max_leverage)
                            
                            # 3분봉 데이터 확인
                            ohlcv = exchange.fetch_ohlcv(coin_config['symbol'], timeframe='3m', limit=2)
                            if len(ohlcv) < 2:
                                continue
                            
                            prev_candle = ohlcv[-2]
                            current_candle = ohlcv[-1]
                            
                            prev_close = prev_candle[4]
                            current_price = current_candle[4]
                            
                            # 봉의 변화율 계산
                            candle_change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            # 레버리지를 고려한 포지션 손익률
                            position_change_pct = candle_change_pct * leverage
                            
                            # 손절 조건: 포지션 반대방향으로 5% 이상 변동
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
                                
                                # 기존 주문 취소
                                try:
                                    exchange.cancel_all_orders(coin_config['symbol'])
                                    print(f"   기존 주문 취소 완료")
                                except Exception as e:
                                    print(f"   주문 취소 오류: {e}")
                                
                                # 긴급 시장가 청산
                                try:
                                    params = {'reduceOnly': True}
                                    if side == 'long':
                                        exchange.create_order(coin_config['symbol'], 'market', 'sell', 
                                                            actual_amount, params=params)
                                    else:
                                        exchange.create_order(coin_config['symbol'], 'market', 'buy', 
                                                            actual_amount, params=params)
                                    
                                    # 손익 계산
                                    if side == 'long':
                                        profit_loss = (current_price - entry_price) * actual_amount
                                        profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
                                    else:
                                        profit_loss = (entry_price - current_price) * actual_amount
                                        profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                                    
                                    print(f"   ✅ 긴급 손절 완료: P/L ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                                    
                                    # 추적 딕셔너리에서 제거
                                    position_key = f"{coin_name}_{side}"
                                    if position_key in POSITION_TRACKER:
                                        del POSITION_TRACKER[position_key]
                                    
                                    # DB에 기록이 있다면 업데이트 (봇이 연 포지션인 경우) - SQL 수정
                                    try:
                                        conn = sqlite3.connect(DB_FILE)
                                        cursor = conn.cursor()
                                        
                                        # 먼저 해당 거래의 ID를 찾기
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
                                            
                                            # ID를 이용해 업데이트
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

# ===== 데이터 수집 함수 (JSON 직렬화 오류 수정) =====
def fetch_multi_timeframe_data_for_coin(symbol):
    """단일 코인의 멀티 타임프레임 데이터 수집 (3분봉 메인)"""
    timeframes = {
        "3m": {"timeframe": "3m", "limit": 160},
        "1m": {"timeframe": "1m", "limit": 120},
        "5m": {"timeframe": "5m", "limit": 100},
        "15m": {"timeframe": "15m", "limit": 96},
        "1h": {"timeframe": "1h", "limit": 100} # EMA50 계산을 위해 100개로 늘림
    }
    multi_tf_data = {}
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_params["timeframe"], limit=tf_params["limit"])
            if not ohlcv or len(ohlcv) < 50: continue # EMA50 계산을 위해 최소 50개 캔들 필요
            
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
    
    # ===== [STRATEGY UPGRADE] ===== AI 시스템 프롬프트 대폭 강화
    system_prompt = """
You are an elite, risk-averse, multi-cryptocurrency day trader AI. Your primary mission is to identify ONLY A+ grade, high-probability trading opportunities in BTC, ETH, and SOL, while rigorously protecting capital.

**ANALYSIS & EXECUTION PROTOCOL:**

1.  **Market Condition Filter (CRITICAL):**
    - You MUST AVOID trading in low-volatility, sideways, or 'choppy' market conditions where price action is unclear.
    - Only propose trades when there is a clear, high-momentum directional bias on the 3-minute chart, supported by higher timeframes.
    - **If no such high-quality condition exists across any of the coins, you MUST return an empty "trading_opportunities" list. Waiting for a clear opportunity is a critical part of this strategy.**

2.  **Data-Driven Decision:**
    - Your trading decisions (LONG/SHORT) MUST be based ONLY on the provided real-time technical and sentiment data for each coin.
    - The **3M (3-minute) chart is your PRIMARY** tool for determining precise entry timing. Higher timeframes (15m, 1h) provide context for the overall trend.

3.  **Strict Risk Management Overlay (NON-NEGOTIABLE):**
    
    - **Time-Cut Mandate (NEW & CRITICAL):** All positions are subject to a mandatory 90-minute 'Time-Cut' rule, meaning they are automatically closed after 90 minutes. Therefore, your **primary objective is to have trades close via TP or SL, NOT the time-cut.** You MUST propose TP and SL targets that are realistically achievable well within this 90-minute window.

    - **Trend Filter:** You are ONLY allowed to propose LONG positions if the current price is ABOVE the 1-hour 50 EMA, and ONLY SHORT positions if the price is BELOW the 1-hour 50 EMA.
    
    - **Maximum Stop Loss:** The **MAXIMUM acceptable `stop_loss_percentage`** on any single trade is **15%** of the invested margin. Your typical SL should be in the **5% to 10%** range.
    
    - **Risk/Reward Ratio:** The proposed `take_profit_percentage` MUST be at least **1.5 times greater** than the `stop_loss_percentage`.
    
    - **Self-Correction:** If you see 2 or more consecutive losses in `recent_trades`, enter a "conservative mode": only enter trades with an extremely high conviction score (>90) and reduce the recommended position size by half.

4.  **Profitability Mandate:**
    - **Acknowledge Costs:** All trades incur approximately **1.2%** in round-trip fees (Taker) and potential slippage at 10x leverage.
    - **Net Profit Target:** Your proposed `take_profit_percentage` MUST be high enough to generate a **net profit of at least 3%** after covering these costs.
    - **This means your gross `take_profit_percentage` must be greater than 4.2% (3% net profit + 1.2% costs).**

**RESPONSE JSON FORMAT:**
- YOUR RESPONSE must be ONLY a valid JSON object, with no markdown.
- If an opportunity is found, the root of the JSON must be an object with a key named "trading_opportunities" which is a list of opportunity objects.
- Each opportunity object MUST contain: "coin", "score", "direction", "recommended_position_size", "recommended_leverage", "stop_loss_percentage", "take_profit_percentage", "priority", "reasoning".
- The value for "coin" MUST be one of ["BTC", "ETH", "SOL"].
- If no high-quality opportunities meeting ALL above criteria are found, return an empty "trading_opportunities" list.
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
        
        response = model.generate_content(
            [system_prompt, f"Multi-Coin Market Analysis: {market_analysis_json}"],
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Is already default
                # stop_sequences=['x'],
                # max_output_tokens=2048,
                temperature=0.4 # Slightly lower temperature for more deterministic and rule-following behavior
            ),
            safety_settings=safety_settings
        )
        
        response_content = response.text.strip().replace("```json", "").replace("```", "")
        trading_decision = json.loads(response_content)
        
        print("\n=== AI Analysis Result ===")
        
        # AI 응답 형식에 따른 유연한 처리 로직
        if isinstance(trading_decision, dict):
            opportunities = trading_decision.get('trading_opportunities', [])
        elif isinstance(trading_decision, list):
            opportunities = trading_decision
        else:
            opportunities = []

        for opp in opportunities:
            print(f"- {opp.get('coin','N/A')}: {opp.get('direction','N/A')} (Score: {opp.get('score',0)})")
        
        # 'trading_decision'이 리스트인 경우 딕셔너리로 감싸서 반환 통일
        if isinstance(trading_decision, list):
            return {"trading_opportunities": trading_decision}
        return trading_decision
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {"trading_opportunities": []}

# ===== 거래 실행 함수 (안정성 강화) =====
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

        # ===== [STRATEGY UPGRADE] ===== 리스크 및 수익성 하드 리밋 필터 추가
        MAX_ACCEPTABLE_SL_PERCENTAGE = 15.0
        MIN_ACCEPTABLE_TP_PERCENTAGE = 4.2 # Net 3% + Cost 1.2%
        
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
        min_margin = coin_config.get('min_investment', 10)
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
                    print(f"{coin_name} {action.upper()} Entry: ${entry_price:,.2f} | SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f}")
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
        save_trade(trade_data, coin_name)
        return margin
        
    except Exception as e:
        print(f"Single trade execution error for {coin_name}: {e}")
        return None

# ===== 메인 프로그램 시작 =====
def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v6.7 - 거래 비용 및 시장 상황 필터 강화) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Strategy: AI 완전 자율 판단, 3분봉 메인, 거래 비용 및 횡보장 필터 적용")
    print("Risk Management: 60분 타임컷, 3분봉 5% 급변동 손절, 상관관계 필터")
    print("Partial Scan: 10분마다 빈 코인슬롯 스캔")
    print("Position Tracking: 실제 포지션 기반 (수동 거래 포함)")
    print("===============================================\n")

    setup_database()
    
    # 봇 시작 시 기존 포지션 추적 초기화
    initialize_position_tracker_on_startup()
    
    last_partial_scan_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    last_rapid_check_time = datetime.now(timezone.utc)

    while True:
        try:
            current_utc = datetime.now(timezone.utc)
            current_local = current_utc.astimezone()
            print(f"\n[{current_local.strftime('%H:%M:%S')}] === Market Check === (경과시간도 로컬시간 기준)")

            sync_database_with_positions()
            
            # 1. 60분 타임아웃 체크 (최우선)
            if check_trade_timeout() > 0:
                time.sleep(10)
                continue
            
            # 2. 급변동 손절 체크 (30초마다)
            if (datetime.now(timezone.utc) - last_rapid_check_time).seconds >= 30:
                if check_rapid_movement_stop_loss() > 0:
                    last_rapid_check_time = datetime.now(timezone.utc)
                    time.sleep(10)
                    continue
                last_rapid_check_time = datetime.now(timezone.utc)
            
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
                
                if len(opportunities) >= 2:
                    directions = {opp.get('direction') for opp in opportunities}
                    if len(directions) == 1:
                        print(f"\n⚠️ Correlation Risk Detected: All {len(opportunities)} opportunities have the same direction.")
                        highest_score_opp = max(opportunities, key=lambda x: x.get('score', 0))
                        print(f"   Filtering to the highest score: {highest_score_opp.get('coin')} (Score: {highest_score_opp.get('score', 0)})")
                        opportunities = [highest_score_opp]
                
                if opportunities:
                    balance = exchange.fetch_balance()['USDT']['free']
                    available_capital_for_loop = balance * 0.98
                    
                    for opp in opportunities:
                        used_margin = execute_single_trade(opp['coin'], opp, available_capital_for_loop, all_data)
                        if used_margin is not None and used_margin > 0:
                            available_capital_for_loop -= used_margin
                        time.sleep(5)
                    last_partial_scan_time = datetime.now(timezone.utc)
                else:
                    print("AI found no high-probability setups.")
                
                time.sleep(120)

            else: # 포지션이 있는 경우
                print(f"{len(current_positions)} active position(s). Monitoring...")
                
                time_since_scan = (datetime.now(timezone.utc) - last_partial_scan_time).total_seconds() / 60
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
                                    used_margin = execute_single_trade(opp['coin'], opp, available_capital_for_loop, available_data)
                                    if used_margin is not None and used_margin > 0:
                                        available_capital_for_loop -= used_margin
                                    time.sleep(5)
                            else:
                                print("AI found no additional setups.")
                    else:
                        print("All coins have open positions.")
                    last_partial_scan_time = datetime.now(timezone.utc)
                else:
                    remaining_minutes = 10 - int(time_since_scan)
                    print(f"다음 부분 스캔까지 {remaining_minutes}분 남음")

                time.sleep(60)

        except Exception as e:
            print(f"\nMain loop error: {e.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    main()