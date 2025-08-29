"""
AI 멀티코인 데이트레이딩 봇 - Gemini + AI 전담 판단 (v6.8 - API 연결 안정화)
--------------------------------------------------------
기능:
- 멀티코인 스캔 (BTC, ETH, SOL) - AI가 모든 판단 담당
- 데이트레이딩 최적화 (3분봉 메인)
- 고급 모멘텀 지표 (RSI, MACD)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- Gemini API 기반 AI 분석 (Gemini 2.5 Pro 모델 사용)
- 동적 레버리지 및 포지션 사이징 (AI 자율 결정)
- AI 기반 동적 SL/TP 설정
- 90분 타임컷: 90분 경과 시 수익/손실 무관 포지션 자동 정리
- 10분 부분 스캔: 포지션 보유 시 10분마다 비어있는 코인 재스캔
- 상관관계 리스크 관리: 모든 코인 동시 진입 신호 시 최고점수 포지션만 진입
- 합리성 필터: AI의 비현실적인 TP/SL 제안 자동 거부
- 레버리지 필터: AI의 레버리지 제안이 설정 범위를 벗어날 경우 자동 조정
- 추세 필터: 1시간봉 EMA를 기준으로 추세를 거스르는 거래 방지
- AI 자가 학습: 연속 손실 발생 시 자동으로 보수적 모드로 전환
- 외부 포지션 자동 인수: 봇 외부에서 생성된 포지션을 감지하여 관리 시스템에 편입
- DB-포지션 동기화 기능
- 24시간 무제한 거래
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
    },
    "ETH": {
        "symbol": "ETH/USDT", 
        "binance_symbol": "ETHUSDT",
        "min_investment": 20,
        "leverage_range": (5, 35),
    },
    "SOL": {
        "symbol": "SOL/USDT",
        "binance_symbol": "SOLUSDT", 
        "min_investment": 10,
        "leverage_range": (3, 25),
    }
}

# ===== 설정 및 초기화 (API 버전 지정 삭제) =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    }
})

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')

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
        sl_price REAL,
        tp_price REAL,
        sl_percentage REAL,
        tp_percentage REAL,
        position_size_percentage REAL,
        investment_amount REAL,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp TEXT,
        profit_loss REAL,
        profit_loss_percentage REAL,
        notes TEXT
    )
    ''')
    
    try:
        cursor.execute("SELECT notes FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE trades ADD COLUMN notes TEXT")
        print("DB Schema Updated: 'notes' column added to trades table.")

    conn.commit()
    conn.close()
    print("멀티코인 데이터베이스 설정 완료")

def save_trade(trade_data, coin_symbol):
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp, coin_symbol, action, entry_price, amount, leverage, sl_price, tp_price,
        sl_percentage, tp_percentage, position_size_percentage, investment_amount, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data.get('timestamp', datetime.now().isoformat()),
        coin_symbol,
        trade_data.get('action', ''),
        trade_data.get('entry_price', 0),
        trade_data.get('amount', 0),
        trade_data.get('leverage', 0),
        trade_data.get('sl_price'),
        trade_data.get('tp_price'),
        trade_data.get('sl_percentage'),
        trade_data.get('tp_percentage'),
        trade_data.get('position_size_percentage'),
        trade_data.get('investment_amount'),
        trade_data.get('notes', '')
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


def get_all_open_trades_from_db():
    """데이터베이스에서 모든 열린 거래 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY timestamp DESC")
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

def get_historical_trading_data(limit=5):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT t.coin_symbol, t.action, t.profit_loss
    FROM trades t
    WHERE t.status IN ('CLOSED', 'CLOSED_TIMEOUT', 'CLOSED_UNKNOWN')
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        trade_result = {
            "coin": row["coin_symbol"],
            "direction": row["action"].upper(),
            "result": "WIN" if row["profit_loss"] is not None and row["profit_loss"] > 0 else "LOSS"
        }
        historical_data.append(trade_result)
    conn.close()
    return historical_data

# ===== 포지션 인수 기능이 포함된 동기화 함수 =====
def get_all_binance_positions():
    """거래소의 모든 활성 포지션 정보를 가져옵니다"""
    all_positions = {}
    try:
        positions = exchange.fetch_positions()
        active_positions = [p for p in positions if float(p['info']['positionAmt']) != 0]
        for position in active_positions:
            symbol = position['symbol'].replace(":USDT", "")
            coin_name = symbol.replace("/USDT", "")
            if coin_name in TRADING_PAIRS:
                all_positions[coin_name] = {
                    'amount': float(position['contracts']),
                    'side': position['side'],
                    'entry_price': float(position['entryPrice']),
                    'leverage': int(position['leverage']),
                    'investment_amount': float(position['initialMargin'])
                }
    except Exception as e:
        print(f"거래소 포지션 조회 오류: {e}")
    return all_positions

def sync_and_adopt_positions():
    """DB와 실제 포지션을 동기화하고, 외부 포지션을 시스템에 인수합니다."""
    print("--- 포지션 동기화 및 인수 시작 ---")
    binance_positions = get_all_binance_positions()
    db_trades = get_all_open_trades_from_db()
    
    db_symbols = {trade['coin_symbol'] for trade in db_trades}
    binance_symbols = set(binance_positions.keys())

    # 1단계: DB에는 있지만 실제로는 없는 포지션 정리 (Sync)
    for trade in db_trades:
        if trade['coin_symbol'] not in binance_symbols:
            print(f"⚠️ 동기화: {trade['coin_symbol']} 포지션이 거래소에 없어 DB를 업데이트합니다.")
            try:
                current_price = exchange.fetch_ticker(TRADING_PAIRS[trade['coin_symbol']]['symbol'])['last']
                update_trade_status(trade['id'], 'CLOSED', exit_price=current_price, exit_timestamp=datetime.now().isoformat())
            except Exception as e:
                print(f"   - 동기화 중 가격 조회 오류: {e}")
                update_trade_status(trade['id'], 'CLOSED_UNKNOWN')

    # 2단계: 실제로는 있지만 DB에는 없는 포지션 인수 (Adopt)
    for symbol, position in binance_positions.items():
        if symbol not in db_symbols:
            print(f"⚠️ 외부 포지션 감지: {symbol} {position['side'].upper()} 포지션을 시스템에 인수합니다.")
            trade_data = {
                'action': position['side'],
                'entry_price': position['entry_price'],
                'amount': position['amount'],
                'leverage': position['leverage'],
                'investment_amount': position['investment_amount'],
                'notes': 'Adopted External Position',
                'timestamp': datetime.now().isoformat(),
                'sl_price': None, 'tp_price': None,
                'sl_percentage': None, 'tp_percentage': None,
                'position_size_percentage': None
            }
            save_trade(trade_data, symbol)
            print(f"   - {symbol} 포지션이 이제 90분 타임컷 등 관리 대상에 포함됩니다.")
    
    print("--- 포지션 동기화 및 인수 완료 ---")

def check_trade_timeout():
    """90분 이상 경과된 모든 포지션을 정리"""
    timeout_count = 0
    open_trades = get_all_open_trades_from_db()
    
    for trade in open_trades:
        try:
            trade_time = datetime.fromisoformat(trade['timestamp'])
            if datetime.now() - trade_time <= timedelta(minutes=90):
                continue

            coin_symbol = trade['coin_symbol']
            action = trade['action']
            trade_id = trade['id']
            
            timeout_count += 1
            print(f"\n=== 90분 타임컷 실행: {coin_symbol} {action.upper()} 포지션 정리 시작 ===")
            
            # ... (포지션 정리 로직은 이전과 동일) ...

        except Exception as e:
            print(f"타임컷 체크 오류 ({trade.get('coin_symbol', 'N/A')}): {e}")

    if timeout_count > 0:
        print("=== 타임컷 정리 완료, 새로운 스캔을 시작합니다 ===")
    return timeout_count

# ... (데이터 수집 및 AI 분석 함수들은 이전과 동일) ...

def main():
    print("\n=== Multi-Coin Day Trading Bot Started (v6.8 - API 연결 안정화) ===")
    # ... (로그 출력)

    setup_database()
    last_partial_scan_time = datetime.now() - timedelta(minutes=10)

    while True:
        try:
            sync_and_adopt_positions()
            
            if check_trade_timeout() > 0:
                time.sleep(10)
                continue
            
            current_positions = get_all_binance_positions()
            
            if not current_positions:
                # ... (전체 스캔 로직)
                pass
            else:
                # ... (부분 스캔 로직)
                pass

        except Exception as e:
            print(f"\nMain loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()

