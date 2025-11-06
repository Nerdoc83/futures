"""
AI Live Trading Bot v1.0 (실거래 버전)
----------------------------------------------------------------------
⚠️ 실제 바이낸스 선물 거래 - 실제 자금 사용
- 실시간 바이낸스 선물 데이터 사용
- 실제 잔고로 거래 (Available Balance 기준)
- Isolated Margin 모드
- 실제 주문 체결 및 청산
- AI 보수적 리스크 관리
- 🆕 처음부터 트레일링 스탑 설정 (20% 체크 제거)

🛡️ 수동거래 보호 기능
----------------------------------------------------------------------
수동거래를 등록하면 AI가 자동으로 청산하지 않습니다.

사용법:
1. 수동거래 등록:
   - register_manual_trade("BTC", "LONG", 50000, 0.1, 10)
   - add_manual_long("BTC", 50000, 0.1, 10)  # 롱 포지션
   - add_manual_short("ETH", 3000, 1.0, 5)   # 숏 포지션

2. 기존 거래를 수동거래로 보호:
   - protect_trade(12)  # 거래 ID 12를 AI 청산에서 보호

3. 수동거래 보호 해제:
   - unprotect_trade(12)  # 거래 ID 12의 보호 해제 (AI 청산 허용)

4. 수동거래 현황 확인:
   - display_manual_trades_status()
   - list_manual_trades()
   - check_trade_protection(거래_ID)  # 특정 거래 보호 상태 확인
   - show_all_trades_protection()      # 모든 거래 보호 상태 확인

⚠️ 중요:
- AI 포지션 관리가 10분마다 실행됩니다 (실시간 대응)
- 최소 보유시간 제약이 완전히 제거되었습니다
- 진입 직후라도 시장 급변 시 즉시 청산 가능
- 변동성에 따라 청산 임계값이 자동 조정됩니다
- 수동거래는 여전히 AI 청산에서 보호됩니다
- 수동거래는 DB 동기화에서도 보호됩니다
- 수동으로 청산한 후에는 unprotect_trade(ID) 호출하여 DB 정리 필요
----------------------------------------------------------------------
"""

import ccxt
import os
import time
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import pandas_ta as ta
import google.generativeai as genai
import random
import json
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import sys

load_dotenv()

# ===== 시간 설정 (UTC 통일) =====
def get_utc_now():
    """UTC 현재 시간 반환 (timezone-aware)"""
    return datetime.now(timezone.utc)

def get_utc_timestamp_ms():
    """UTC 타임스탬프 (밀리초) 반환"""
    return int(get_utc_now().timestamp() * 1000)

def parse_db_timestamp(timestamp_str):
    """DB timestamp를 UTC datetime으로 변환 (timezone-aware)"""
    if not timestamp_str:
        return None
    # 공백을 T로 변환 (ISO 형식)
    iso_str = timestamp_str.replace(' ', 'T')
    dt = datetime.fromisoformat(iso_str)
    # timezone 정보가 없으면 UTC로 간주
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

load_dotenv()

# ===== 로깅 설정 =====
LOG_FILE = "trading_bot.log"

class Logger:
    """파일과 콘솔에 동시 로깅 (즉시 flush)"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        # buffering=1 = 라인 버퍼링 (줄 단위 즉시 flush)
        self.log = open(filename, 'a', encoding='utf-8', buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 🆕 터미널도 즉시 flush
        self.log.write(message)
        self.log.flush()  # 🆕 파일도 즉시 flush
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 로거 초기화
sys.stdout = Logger(LOG_FILE)
sys.stderr = Logger(LOG_FILE)

# 🆕 Python unbuffered mode 강제 (환경변수)
os.environ['PYTHONUNBUFFERED'] = '1'

# ===== AI 모델 선택 =====
AI_MODEL_CONFIG = {
    "provider": "deepseek",  # 🔧 여기서 변경: "gemini" 또는 "deepseek" (유료, 정확)
    "models": {
        "gemini": "gemini-2.5-flash-latest",  # 무료, 빠름
        "deepseek": "deepseek-chat"  # DeepSeek V3 (최신, 추천)
    },
    "rate_limit": {
        "gemini": {
            "requests_per_minute": 15,
            "delay_between_requests": 4.5
        },
        "deepseek": {
            "requests_per_minute": 60,
            "delay_between_requests": 1.0
        }
    }
}

last_api_call_time = 0

# ===== AI 초기화 =====
ai_client = None

def initialize_ai():
    global ai_client
    provider = AI_MODEL_CONFIG["provider"].lower()
    
    try:
        if provider == "gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            print(f"✅ Gemini AI ({AI_MODEL_CONFIG['models']['gemini']}) 설정 완료")
            return True
        elif provider == "deepseek":
            from openai import OpenAI
            ai_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            print(f"✅ DeepSeek AI ({AI_MODEL_CONFIG['models']['deepseek']}) 설정 완료")
            return True
        else:
            print(f"❌ 지원하지 않는 AI 제공자: {provider}")
            return False
    except Exception as e:
        print(f"❌ AI 설정 오류: {e}")
        return False

if not initialize_ai():
    print("\n사용 가능한 AI 제공자: gemini, deepseek")
    exit(1)

# ===== 실거래 설정 =====
LIVE_TRADING_CONFIG = {
    "MAX_CONCURRENT_POSITIONS": 5,  # 최대 동시 포지션 (실제 진입 제한)
    "MIN_CAPITAL_THRESHOLD": 100.0,  # 최소 잔고 (USDT)
    "AI_ANALYSIS_INTERVAL": 60,  # 신규 진입 분석 (1분마다)
    "PERFORMANCE_REVIEW_INTERVAL": 600,  # AI 성과 리뷰 (10분)
    "POSITION_CHECK_INTERVAL": 600,  # 🔧 10분마다 AI 포지션 평가 (실시간 시장 대응)
    
    # 🆕 트레일링 스탑 설정 (처음부터 활성화)
    "TRAILING_STOP_ENABLED": True,              # 트레일링 스탑 활성화
    "TRAILING_STOP_MIN_PROFIT_PCT": 7.0,        # 최소 확보 수익률 (%) - 레버리지 고려하여 자동 계산됨
                                                  # 예: 7% 수익 확보 + 10배 레버리지 = 0.7% 콜백
    
    # 🔧 자금 관리 설정 (동적 균등 분할)
    "MAX_POSITION_SIZE_PCT": 50,  # 안전장치: 가용 자금의 최대 50% (동적 균등 분할 활용)
    "MIN_POSITION_SIZE_PCT": 3,   # 최소 3% (너무 작은 포지션 방지)
    "DYNAMIC_EQUAL_SPLIT": True,  # 동적 균등 분할 활성화
    "VOLATILITY_BASED_SIZING": True,  # 변동성 기반 포지션 크기 조절
    "HIGH_VOLATILITY_THRESHOLD": 5.0,  # 5% 이상이면 고변동성
    "LOW_VOLATILITY_MULTIPLIER": 1.2,  # 저변동성 = 1.2배 투자
    "HIGH_VOLATILITY_MULTIPLIER": 0.8,  # 고변동성 = 0.8배 투자
    
    # 🔧 거래 수수료 (바이낸스 선물 일반회원)
    "MAKER_FEE": 0.02,  # 0.02%
    "TAKER_FEE": 0.05,  # 0.05%
    
    # 🔧 레버리지 설정
    "MAX_LEVERAGE": 15,
    "CONSERVATIVE_LEVERAGE": 10,
    
    # 🔧 마진 모드
    "MARGIN_MODE": "isolated",  # isolated 또는 cross
}

# ===== API 설정 (실거래 모드) =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")

if not api_key or not secret:
    print("❌ BINANCE_API_KEY 또는 BINANCE_SECRET_KEY가 설정되지 않았습니다.")
    print("   .env 파일을 확인하세요.")
    exit(1)

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # 선물 거래
        'adjustForTimeDifference': True,
        'test': False  # ⚠️ 실거래 모드
    }
})

DB_FILE = "live_trading.db"

# ===== 바이낸스 선물 거래 함수 =====

def get_available_balance() -> float:
    """실제 바이낸스 선물 계정의 Available Balance 조회"""
    try:
        balance = exchange.fetch_balance({'type': 'future'})
        usdt_balance = balance.get('USDT', {})
        available = usdt_balance.get('free', 0)
        print(f"   💰 Available Balance: ${available:,.2f}")
        return available
    except Exception as e:
        print(f"   ❌ 잔고 조회 실패: {e}")
        return 0.0

def get_coin_precision(symbol: str) -> dict:
    """코인별 가격/수량 정밀도 조회"""
    try:
        market = exchange.market(symbol)
        return {
            'price': market['precision']['price'],
            'amount': market['precision']['amount'],
            'min_amount': market['limits']['amount']['min']
        }
    except Exception as e:
        print(f"   ⚠️ 정밀도 조회 실패: {e}, 기본값 사용")
        return {'price': 2, 'amount': 3, 'min_amount': 0.001}

def set_leverage(symbol: str, leverage: int) -> bool:
    """레버리지 설정"""
    try:
        exchange.set_leverage(leverage, symbol)
        print(f"   ✅ 레버리지 {leverage}x 설정 완료")
        return True
    except Exception as e:
        print(f"   ⚠️ 레버리지 설정 오류: {e}")
        return False

def set_margin_mode(symbol: str, mode: str = "isolated") -> bool:
    """마진 모드 설정"""
    try:
        exchange.set_margin_mode(mode, symbol)
        print(f"   ✅ 마진 모드: {mode.upper()}")
        return True
    except ccxt.BadRequest as e:
        if "No need to change margin type" in str(e):
            print(f"   ✅ 마진 모드 이미 {mode.upper()}로 설정됨")
            return True
        print(f"   ⚠️ 마진 모드 설정 오류: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ 마진 모드 설정 오류: {e}")
        return False

def create_market_order(symbol: str, side: str, amount: float) -> Optional[dict]:
    """시장가 주문"""
    try:
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=amount
        )
        print(f"   ✅ 시장가 주문 체결: {side.upper()} {amount}")
        return order
    except Exception as e:
        print(f"   ❌ 주문 실패: {e}")
        return None

def set_stop_loss_order(symbol: str, side: str, amount: float, stop_price: float) -> Optional[dict]:
    """
    스탑로스 주문 설정
    - LONG 포지션: 가격이 stop_price 아래로 떨어지면 SELL
    - SHORT 포지션: 가격이 stop_price 위로 올라가면 BUY
    """
    try:
        # 포지션과 반대 방향으로 주문
        order_side = 'sell' if side.upper() == 'LONG' else 'buy'
        
        params = {
            'stopPrice': stop_price,
            'closePosition': True  # 🆕 전체 포지션 청산 (수량 자동 계산)
        }
        
        order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',  # STOP_MARKET = 스탑 가격 도달 시 시장가 체결
            side=order_side,
            amount=None,  # closePosition=True이면 수량 불필요
            params=params
        )
        
        print(f"   ✅ 스탑로스 설정: ${stop_price:,.2f} (포지션 전체 청산)")
        return order
    except Exception as e:
        print(f"   ❌ 스탑로스 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def set_trailing_stop_order(symbol: str, side: str, leverage: int) -> Optional[dict]:
    """
    🆕 트레일링 스탑 주문 설정 (바이낸스 선물) - 레버리지 고려
    
    Parameters:
        symbol: 거래 심볼 (예: "BTC/USDT:USDT")
        side: 포지션 방향 ("LONG" 또는 "SHORT")
        leverage: 레버리지 배수 (예: 10)
    
    Returns:
        order: 주문 정보 (실패 시 None)
    
    로직:
        - 최소 확보 수익률: TRAILING_STOP_MIN_PROFIT_PCT (예: 20%)
        - 실제 콜백 비율 = 최소 수익률 / 레버리지
        - 예: 20% 수익 확보 + 10배 레버리지 = 2% 콜백
    """
    try:
        # 레버리지 고려한 콜백 비율 계산
        min_profit_pct = LIVE_TRADING_CONFIG.get("TRAILING_STOP_MIN_PROFIT_PCT", 20.0)
        callback_rate = round(min_profit_pct / leverage, 2)
        
        # 바이낸스 최소값 체크 (0.1% 이상)
        if callback_rate < 0.1:
            callback_rate = 0.1
            print(f"   ⚠️ 콜백 비율 최소값 적용: 0.1%")
        
        # 바이낸스 최대값 체크 (5% 이하 권장)
        if callback_rate > 5.0:
            callback_rate = 5.0
            print(f"   ⚠️ 콜백 비율 최대값 적용: 5.0%")
        
        # 포지션과 반대 방향으로 주문
        order_side = 'sell' if side.upper() == 'LONG' else 'buy'
        
        # 바이낸스 선물 트레일링 스탑 파라미터
        params = {
            'callbackRate': callback_rate,  # 레버리지 고려한 콜백 비율
            'closePosition': True,  # 전체 포지션 청산
            'workingType': 'CONTRACT_PRICE',  # 계약 가격 기준 (마크 가격 아님)
            'priceProtect': True  # 가격 보호 활성화
        }
        
        # TRAILING_STOP_MARKET 주문 생성
        order = exchange.create_order(
            symbol=symbol,
            type='TRAILING_STOP_MARKET',  # 트레일링 스탑 마켓 주문
            side=order_side,
            amount=None,  # closePosition=True이면 수량 불필요
            params=params
        )
        
        # 예상 수익률 계산
        expected_profit = callback_rate * leverage
        
        print(f"   ✅ 트레일링 스탑 설정:")
        print(f"      - 레버리지: {leverage}x")
        print(f"      - 콜백 비율: {callback_rate}% (가격 기준)")
        print(f"      - 최소 확보 수익률: {expected_profit:.1f}%")
        print(f"      - 주문 ID: {order.get('id', 'N/A')}")
        return order
    except Exception as e:
        print(f"   ❌ 트레일링 스탑 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def cancel_all_orders(symbol: str) -> bool:
    """특정 심볼의 모든 열린 주문 취소"""
    try:
        exchange.cancel_all_orders(symbol)
        print(f"   ✅ 모든 주문 취소 완료")
        return True
    except Exception as e:
        print(f"   ⚠️ 주문 취소 오류: {e}")
        return False

def get_open_positions() -> List[dict]:
    """바이낸스에서 실제 오픈 포지션 조회"""
    try:
        positions = exchange.fetch_positions()
        open_positions = [
            pos for pos in positions 
            if float(pos.get('contracts', 0)) > 0
        ]
        return open_positions
    except Exception as e:
        print(f"   ❌ 포지션 조회 실패: {e}")
        return []

def get_accurate_pnl_from_history(symbol: str, start_time: int = None) -> Optional[dict]:
    """
    🆕 Position History에서 정확한 PnL 조회 (수수료 포함, 가장 정확)
    
    Parameters:
        symbol: 거래 심볼 (예: "BTC/USDT:USDT")
        start_time: 시작 시간 (밀리초, None이면 최근 7일)
    
    Returns:
        {
            'realized_pnl': float,  # 실현 손익 (수수료 포함)
            'unrealized_pnl': float,  # 미실현 손익
            'total_pnl': float  # 총 손익
        }
    """
    try:
        if start_time is None:
            # 기본값: 7일 전
            start_time = get_utc_timestamp_ms() - (7 * 24 * 60 * 60 * 1000)
        
        # Position History 조회 (가장 정확한 데이터)
        history = exchange.fetch_positions_history(
            symbols=[symbol],
            since=start_time,
            limit=100
        )
        
        if not history:
            return None
        
        # 최근 포지션 (종료된 것)
        realized_pnl = 0.0
        for pos in history:
            if pos['symbol'] == symbol:
                realized_pnl += float(pos.get('realizedPnl', 0))
        
        # 현재 오픈 포지션의 미실현 손익
        current_positions = get_open_positions()
        unrealized_pnl = 0.0
        for pos in current_positions:
            if pos['symbol'] == symbol:
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                break
        
        return {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl
        }
    
    except Exception as e:
        print(f"   ⚠️ Position History 조회 실패: {e}")
        return None

def close_position(symbol: str, side: str) -> bool:
    """포지션 청산"""
    try:
        # 현재 포지션 조회
        positions = get_open_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if not position:
            print(f"   ⚠️ 포지션 없음: {symbol}")
            return False
        
        amount = abs(float(position['contracts']))
        close_side = 'sell' if side.upper() == 'LONG' else 'buy'
        
        # 모든 미체결 주문 취소
        cancel_all_orders(symbol)
        
        # 시장가로 청산
        order = create_market_order(symbol, close_side, amount)
        
        if order:
            print(f"   ✅ 포지션 청산 완료: {side.upper()} {amount}")
            
            # 🆕 청산 후 잠시 대기 후 Position History에서 정확한 PnL 조회
            time.sleep(2)
            
            return True
        return False
    except Exception as e:
        print(f"   ❌ 포지션 청산 실패: {e}")
        return False

# ===== DB 초기화 =====

def init_db():
    """데이터베이스 초기화 (trades 및 performance 테이블)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # trades 테이블 생성 (manual_trade 컬럼 추가)
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            coin_symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL NOT NULL,
            leverage INTEGER NOT NULL,
            position_size REAL NOT NULL,
            pnl REAL,
            pnl_percent REAL,
            reason TEXT,
            status TEXT DEFAULT 'OPEN',
            ai_confidence REAL,
            ai_reasoning TEXT,
            stop_loss_price REAL,
            take_profit_price REAL,
            manual_trade INTEGER DEFAULT 0,
            closed_at TEXT
        )
    ''')
    
    # 🆕 기존 테이블에 누락된 컬럼 추가 (마이그레이션)
    try:
        # 현재 컬럼 목록 확인
        c.execute("PRAGMA table_info(trades)")
        columns = [col[1] for col in c.fetchall()]
        
        # 필수 컬럼 목록과 기본값
        required_columns = {
            'side': ('TEXT', 'LONG'),
            'quantity': ('REAL', 0),
            'leverage': ('INTEGER', 10),
            'position_size': ('REAL', 0),
            'entry_price': ('REAL', 0),
            'exit_price': ('REAL', None),
            'pnl': ('REAL', None),
            'pnl_percent': ('REAL', None),
            'reason': ('TEXT', None),
            'status': ('TEXT', 'OPEN'),
            'manual_trade': ('INTEGER', 0),
            'closed_at': ('TEXT', None),
            'stop_loss_price': ('REAL', None),
            'take_profit_price': ('REAL', None),
            'ai_confidence': ('REAL', None),
            'ai_reasoning': ('TEXT', None),
            'timestamp': ('TEXT', None),
            'coin_symbol': ('TEXT', None)
        }
        
        # 없는 컬럼 추가
        for col_name, (col_type, default_val) in required_columns.items():
            if col_name not in columns:
                print(f"   🔄 DB 마이그레이션: '{col_name}' 컬럼 추가 중...")
                if default_val is None:
                    c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                elif isinstance(default_val, str) and col_type == 'TEXT':
                    c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type} DEFAULT '{default_val}'")
                else:
                    c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type} DEFAULT {default_val}")
                print(f"   ✅ '{col_name}' 컬럼 추가 완료")
        
        conn.commit()
        
    except Exception as e:
        print(f"   ⚠️ DB 마이그레이션 오류: {e}")
        import traceback
        traceback.print_exc()
    
    # performance 테이블 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate REAL,
            total_pnl REAL,
            avg_win REAL,
            avg_loss REAL,
            profit_factor REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            notes TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 초기화 완료")

# ===== 거래 기록 함수 =====

def record_trade(coin_symbol: str, side: str, entry_price: float, quantity: float, 
                leverage: int, position_size: float, ai_confidence: float, 
                ai_reasoning: str, stop_loss_price: float, take_profit_price: float,
                manual_trade: bool = False) -> int:
    """거래 기록 저장"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    timestamp = get_utc_now().strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute('''
        INSERT INTO trades (timestamp, coin_symbol, side, entry_price, quantity, 
                           leverage, position_size, ai_confidence, ai_reasoning,
                           stop_loss_price, take_profit_price, status, manual_trade)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    ''', (timestamp, coin_symbol, side, entry_price, quantity, leverage, 
          position_size, ai_confidence, ai_reasoning, stop_loss_price, 
          take_profit_price, 1 if manual_trade else 0))
    
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id

def update_trade_exit(trade_id: int, exit_price: float, pnl: float, pnl_percent: float, reason: str):
    """거래 청산 정보 업데이트"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    closed_at = get_utc_now().strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute('''
        UPDATE trades 
        SET exit_price = ?, pnl = ?, pnl_percent = ?, reason = ?, status = 'CLOSED', closed_at = ?
        WHERE id = ?
    ''', (exit_price, pnl, pnl_percent, reason, closed_at, trade_id))
    
    conn.commit()
    conn.close()

def get_all_open_trades() -> List[dict]:
    """모든 오픈 거래 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, coin_symbol, side, entry_price, quantity, leverage, 
               position_size, timestamp, ai_confidence, stop_loss_price, 
               take_profit_price, manual_trade
        FROM trades 
        WHERE status = 'OPEN'
        ORDER BY timestamp DESC
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    trades = []
    for row in rows:
        trades.append({
            'id': row[0],
            'coin_symbol': row[1],
            'side': row[2],
            'entry_price': row[3],
            'quantity': row[4],
            'leverage': row[5],
            'position_size': row[6],
            'timestamp': row[7],
            'ai_confidence': row[8],
            'stop_loss_price': row[9],
            'take_profit_price': row[10],
            'manual_trade': bool(row[11])
        })
    
    return trades

def get_recent_performance(days: int = 7) -> dict:
    """최근 성과 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    cutoff_date = (get_utc_now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
               AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
               AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
               SUM(pnl) as total_pnl
        FROM trades
        WHERE status = 'CLOSED' AND timestamp >= ?
    ''', (cutoff_date,))
    
    row = c.fetchone()
    conn.close()
    
    total = row[0] or 0
    wins = row[1] or 0
    losses = row[2] or 0
    avg_win = row[3] or 0
    avg_loss = row[4] or 0
    total_pnl = row[5] or 0
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': abs(avg_loss) if avg_loss else 0,
        'total_pnl': total_pnl,
        'days': days
    }

# ===== 시장 데이터 수집 =====

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """OHLCV 데이터 조회"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        print(f"   ❌ OHLCV 조회 실패 ({timeframe}): {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    if df.empty:
        return df
    
    try:
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 볼린저 밴드 (컬럼명 안전 처리)
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            # pandas_ta 버전별 컬럼명 처리
            bb_cols = bb.columns.tolist()
            
            # 상단 밴드 찾기
            upper_col = next((col for col in bb_cols if 'BBU' in col or 'upper' in col.lower()), None)
            # 중간 밴드 찾기
            middle_col = next((col for col in bb_cols if 'BBM' in col or 'middle' in col.lower()), None)
            # 하단 밴드 찾기
            lower_col = next((col for col in bb_cols if 'BBL' in col or 'lower' in col.lower()), None)
            
            if upper_col:
                df['bb_upper'] = bb[upper_col]
            if middle_col:
                df['bb_middle'] = bb[middle_col]
            if lower_col:
                df['bb_lower'] = bb[lower_col]
        
        # MACD (컬럼명 안전 처리)
        macd = ta.macd(df['close'])
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            
            # MACD 라인
            macd_col = next((col for col in macd_cols if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col), None)
            # 시그널 라인
            signal_col = next((col for col in macd_cols if 'MACDs' in col or 'signal' in col.lower()), None)
            # 히스토그램
            hist_col = next((col for col in macd_cols if 'MACDh' in col or 'histogram' in col.lower()), None)
            
            if macd_col:
                df['macd'] = macd[macd_col]
            if signal_col:
                df['macd_signal'] = macd[signal_col]
            if hist_col:
                df['macd_hist'] = macd[hist_col]
        
        # 이동평균
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # ATR (변동성)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 거래량 이동평균
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        
        return df
    except Exception as e:
        print(f"   ⚠️ 지표 계산 오류: {e}")
        return df

def fetch_funding_rate(symbol: str) -> Optional[float]:
    """펀딩비 조회 (바이낸스 선물)"""
    try:
        funding = exchange.fetch_funding_rate(symbol)
        
        # 여러 가능한 키 시도
        if isinstance(funding, dict):
            # 방법 1: 'fundingRate'
            if 'fundingRate' in funding and funding['fundingRate'] is not None:
                return float(funding['fundingRate']) * 100  # 퍼센트로 변환
            
            # 방법 2: 'info' 안에 있는 경우
            if 'info' in funding and isinstance(funding['info'], dict):
                info = funding['info']
                if 'fundingRate' in info and info['fundingRate']:
                    return float(info['fundingRate']) * 100
                if 'lastFundingRate' in info and info['lastFundingRate']:
                    return float(info['lastFundingRate']) * 100
        
        # 숫자로 직접 반환되는 경우
        if isinstance(funding, (int, float)):
            return float(funding) * 100
        
        print(f"   ⚠️ 펀딩비 데이터 형식 불명: {type(funding)}")
        return None
        
    except Exception as e:
        print(f"   ⚠️ 펀딩비 조회 실패: {e}")
        return None

def fetch_open_interest(symbol: str) -> Optional[float]:
    """미결제약정 조회 (바이낸스 선물)"""
    try:
        # ccxt의 fetch_open_interest 응답 구조
        oi_data = exchange.fetch_open_interest(symbol)
        
        # 여러 가능한 키 시도
        if isinstance(oi_data, dict):
            # 방법 1: 'openInterestAmount' (계약 수량)
            if 'openInterestAmount' in oi_data and oi_data['openInterestAmount']:
                return float(oi_data['openInterestAmount'])
            
            # 방법 2: 'openInterestValue' (USDT 가치)
            if 'openInterestValue' in oi_data and oi_data['openInterestValue']:
                return float(oi_data['openInterestValue'])
            
            # 방법 3: 'openInterest' (레거시)
            if 'openInterest' in oi_data and oi_data['openInterest']:
                return float(oi_data['openInterest'])
            
            # 방법 4: 'info' 안에 있는 경우
            if 'info' in oi_data:
                info = oi_data['info']
                if isinstance(info, dict):
                    if 'openInterest' in info:
                        return float(info['openInterest'])
                    if 'sumOpenInterest' in info:
                        return float(info['sumOpenInterest'])
        
        # 숫자로 직접 반환되는 경우
        if isinstance(oi_data, (int, float)):
            return float(oi_data)
        
        print(f"   ⚠️ 미결제약정 데이터 형식 불명: {type(oi_data)}")
        return None
        
    except Exception as e:
        print(f"   ⚠️ 미결제약정 조회 실패: {e}")
        return None

def fetch_comprehensive_market_data(symbol: str) -> dict:
    """종합 시장 데이터 수집 (multi-timeframe + 펀딩비 + 미결제약정)"""
    print(f"   📊 시장 데이터 수집: {symbol}")
    
    market_data = {}
    
    # 멀티 타임프레임 데이터
    timeframes = {
        '1m': 100,
        '5m': 100,
        '15m': 100,
        '1h': 200,
        '4h': 200
    }
    
    for tf, limit in timeframes.items():
        df = fetch_ohlcv(symbol, tf, limit)
        if not df.empty:
            df = calculate_technical_indicators(df)
            market_data[tf] = df
            time.sleep(0.2)  # API 레이트 리밋 방지
    
    # 펀딩비
    funding_rate = fetch_funding_rate(symbol)
    if funding_rate is not None:
        market_data['funding_rate'] = funding_rate
        print(f"   ✅ 펀딩비: {funding_rate:.4f}%")
    else:
        print(f"   ⚠️ 펀딩비 데이터 없음")
    
    # 미결제약정
    open_interest = fetch_open_interest(symbol)
    if open_interest is not None:
        market_data['open_interest'] = open_interest
        print(f"   ✅ 미결제약정: ${open_interest/1_000_000:.2f}M")
    else:
        print(f"   ⚠️ 미결제약정 데이터 없음")
    
    return market_data

def get_top_volume_coins(top_n: int = 10) -> List[dict]:
    """거래량 상위 코인 조회"""
    try:
        tickers = exchange.fetch_tickers()
        
        # USDT 선물만 필터링
        usdt_futures = {
            symbol: ticker for symbol, ticker in tickers.items()
            if ':USDT' in symbol and ticker.get('quoteVolume', 0) > 0
        }
        
        # 거래량 기준 정렬
        sorted_coins = sorted(
            usdt_futures.items(),
            key=lambda x: x[1].get('quoteVolume', 0),
            reverse=True
        )
        
        top_coins = []
        for symbol, ticker in sorted_coins[:top_n]:
            coin = symbol.split('/')[0]
            top_coins.append({
                'coin': coin,
                'symbol': symbol,
                'price': ticker.get('last', 0),
                'volume_24h': ticker.get('quoteVolume', 0),
                'change_24h': ticker.get('percentage', 0)
            })
        
        return top_coins
    except Exception as e:
        print(f"   ❌ 코인 조회 실패: {e}")
        return []

# ===== AI 분석 =====

def ai_comprehensive_analysis(coin_data: dict, market_data: dict, performance_history: dict) -> dict:
    """AI 종합 분석"""
    global last_api_call_time
    
    provider = AI_MODEL_CONFIG["provider"]
    delay = AI_MODEL_CONFIG["rate_limit"][provider]["delay_between_requests"]
    
    # API 레이트 리밋
    elapsed = time.time() - last_api_call_time
    if elapsed < delay:
        time.sleep(delay - elapsed)
    
    # 시장 데이터 요약
    def safe_float(val, default=0.0):
        """None이나 NaN을 안전하게 float로 변환"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    summary = {
        'coin': coin_data['coin'],
        'price': safe_float(coin_data.get('price'), 0.0),
        'volume_24h': safe_float(coin_data.get('volume_24h'), 0.0) / 1_000_000,  # 백만 단위
        'change_24h': safe_float(coin_data.get('change_24h'), 0.0),
        'funding_rate': safe_float(market_data.get('funding_rate'), 0.0),
        'open_interest': safe_float(market_data.get('open_interest'), 0.0) / 1_000_000,  # 백만 단위
        'timeframes': {}
    }
    
    # 각 타임프레임별 최신 지표
    for tf in ['1m', '5m', '15m', '1h', '4h']:
        if tf in market_data:
            df = market_data[tf]
            if not df.empty:
                latest = df.iloc[-1]
                
                summary['timeframes'][tf] = {
                    'rsi': safe_float(latest.get('rsi'), 50.0),  # RSI 기본값 50 (중립)
                    'macd': safe_float(latest.get('macd'), 0.0),
                    'macd_signal': safe_float(latest.get('macd_signal'), 0.0),
                    'close': safe_float(latest.get('close'), 0.0),
                    'sma_20': safe_float(latest.get('sma_20'), 0.0),
                    'sma_50': safe_float(latest.get('sma_50'), 0.0),
                    'atr': safe_float(latest.get('atr'), 0.0),
                    'volume': safe_float(latest.get('volume'), 0.0),
                    'volume_sma': safe_float(latest.get('volume_sma'), 0.0)
                }
    
    # AI 프롬프트
    prompt = f"""
당신은 암호화폐 선물 거래 전문가입니다. 다음 시장 데이터를 분석하여 거래 결정을 내려주세요.

## 시장 데이터
- 코인: {summary['coin']}
- 현재가: ${summary['price']:,.2f}
- 24h 변동: {summary['change_24h']:+.2f}%
- 24h 거래량: ${summary['volume_24h']:.1f}M
- 펀딩비: {summary['funding_rate']:.4f}%
- 미결제약정: ${summary['open_interest']:.1f}M

## 타임프레임별 기술적 지표
{json.dumps(summary['timeframes'], indent=2)}

## 최근 성과
- 총 거래: {performance_history['total_trades']}회
- 승률: {performance_history['win_rate']:.1f}%
- 총 손익: ${performance_history['total_pnl']:,.2f}
- 평균 수익: ${performance_history['avg_win']:,.2f}
- 평균 손실: ${performance_history['avg_loss']:,.2f}

## 분석 지침
1. 멀티 타임프레임 분석: 1m부터 4h까지 트렌드 일치 여부 확인
2. 펀딩비: 극단적인 펀딩비는 추세 반전 신호
3. 미결제약정: 급격한 증가는 추세 강화 신호
4. RSI: 과매수(>70) 또는 과매도(<30) 확인
5. MACD: 크로스오버 및 다이버전스 확인
6. 거래량: 평균 대비 급증 여부 확인
7. 최근 성과: 승률과 손익을 고려하여 신중하게 판단

## 출력 형식 (JSON)
{{
    "trade": true/false,
    "side": "LONG" 또는 "SHORT" (trade가 true일 때만),
    "confidence": 0-100 (정수),
    "leverage": 8-15 (정수, trade가 true일 때만),
    "stop_loss_pct": 3-7 (정수, trade가 true일 때만),
    "take_profit_pct": 10-20 (정수, trade가 true일 때만),
    "reasoning": "간결한 판단 근거 (2-3문장)"
}}

⚠️ 주의사항:
- confidence 70% 미만이면 trade: false
- 리스크/보상 비율 최소 1:2 유지
- 펀딩비가 ±0.1% 이상이면 매우 신중하게 판단
- 타임프레임 간 신호 불일치 시 trade: false
- JSON 형식만 출력 (다른 텍스트 없음)
"""
    
    try:
        if provider == "gemini":
            model = genai.GenerativeModel(AI_MODEL_CONFIG["models"]["gemini"])
            response = model.generate_content(prompt)
            response_text = response.text
        else:  # deepseek
            response = ai_client.chat.completions.create(
                model=AI_MODEL_CONFIG["models"]["deepseek"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            response_text = response.choices[0].message.content
        
        last_api_call_time = time.time()
        
        # JSON 파싱
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        decision = json.loads(response_text)
        
        # 기본값 설정
        if 'confidence' not in decision:
            decision['confidence'] = 0
        if 'trade' not in decision:
            decision['trade'] = False
        
        return decision
        
    except Exception as e:
        print(f"   ❌ AI 분석 오류: {e}")
        return {
            'trade': False,
            'confidence': 0,
            'reasoning': f'AI 분석 실패: {str(e)}'
        }

# ===== 실거래 실행 =====

def calculate_dynamic_position_size(available_balance: float, confidence: float, 
                                   volatility: float) -> float:
    """
    동적 포지션 크기 계산
    
    전략:
    1. 동적 균등 분할: 가용 자금 ÷ (MAX_POSITIONS - 현재 포지션 수)
    2. AI 신뢰도 반영: confidence에 따라 50-100% 투자
    3. 변동성 조정: 고변동성이면 줄이고, 저변동성이면 늘림
    """
    config = LIVE_TRADING_CONFIG
    
    # 현재 오픈 포지션 수
    open_trades = get_all_open_trades()
    current_positions = len(open_trades)
    
    # 동적 균등 분할 기본 비율
    if config.get("DYNAMIC_EQUAL_SPLIT", True):
        remaining_slots = max(1, config['MAX_CONCURRENT_POSITIONS'] - current_positions)
        base_pct = 100 / remaining_slots  # 남은 슬롯에 균등 분할
    else:
        base_pct = 20  # 고정 20%
    
    # AI 신뢰도 기반 조정 (70-95% -> 50-100% 투자)
    confidence_multiplier = 0.5 + (confidence - 70) / 50  # 70% = 0.5x, 95% = 1.0x
    
    # 변동성 기반 조정
    volatility_multiplier = 1.0
    if config.get("VOLATILITY_BASED_SIZING", True):
        if volatility > config['HIGH_VOLATILITY_THRESHOLD']:
            volatility_multiplier = config['HIGH_VOLATILITY_MULTIPLIER']
        else:
            volatility_multiplier = config['LOW_VOLATILITY_MULTIPLIER']
    
    # 최종 비율 계산
    final_pct = base_pct * confidence_multiplier * volatility_multiplier
    
    # 최소/최대 제한
    final_pct = max(config['MIN_POSITION_SIZE_PCT'], 
                   min(config['MAX_POSITION_SIZE_PCT'], final_pct))
    
    position_size = available_balance * (final_pct / 100)
    
    print(f"   💰 포지션 크기 계산:")
    print(f"      - 기본 비율: {base_pct:.1f}% (남은 슬롯: {remaining_slots}개)")
    print(f"      - AI 신뢰도 조정: {confidence_multiplier:.2f}x (confidence {confidence}%)")
    print(f"      - 변동성 조정: {volatility_multiplier:.2f}x (volatility {volatility:.2f}%)")
    print(f"      - 최종 비율: {final_pct:.1f}%")
    print(f"      - 포지션 크기: ${position_size:,.2f}")
    
    return position_size

def execute_live_trade(coin_data: dict, decision: dict, available_balance: float) -> bool:
    """
    실거래 실행 (🆕 처음부터 트레일링 스탑 설정)
    
    변경사항:
    - TP 주문 제거
    - SL + 트레일링 스탑만 설정
    - 20% 체크 로직 제거
    """
    symbol = coin_data['symbol']
    coin = coin_data['coin']
    current_price = coin_data['price']
    side = decision['side']
    leverage = decision.get('leverage', LIVE_TRADING_CONFIG['CONSERVATIVE_LEVERAGE'])
    stop_loss_pct = decision.get('stop_loss_pct', 5)
    
    print(f"\n{'='*80}")
    print(f"🚀 실거래 진입: {coin} ({side.upper()})")
    print(f"{'='*80}")
    
    try:
        # 1. 마진 모드 설정
        margin_mode = LIVE_TRADING_CONFIG['MARGIN_MODE']
        if not set_margin_mode(symbol, margin_mode):
            print(f"   ⚠️ 마진 모드 설정 실패, 계속 진행")
        
        # 2. 레버리지 설정
        if not set_leverage(symbol, leverage):
            print(f"   ⚠️ 레버리지 설정 실패, 기본값 사용")
        
        # 3. 포지션 크기 계산
        volatility = coin_data.get('change_24h', 0)
        confidence = decision.get('confidence', 70)
        position_size = calculate_dynamic_position_size(available_balance, confidence, abs(volatility))
        
        # 4. 수량 계산
        precision = get_coin_precision(symbol)
        quantity = position_size * leverage / current_price
        # 수정: precision['amount']를 int()로 감싸서 정수형으로 변환합니다.
        quantity = round(quantity, int(precision['amount']))
        
        # 최소 수량 체크
        if quantity < precision['min_amount']:
            print(f"   ❌ 수량이 최소값보다 작음: {quantity} < {precision['min_amount']}")
            return False
        
        print(f"   📊 거래 상세:")
        print(f"      - 진입가: ${current_price:,.2f}")
        print(f"      - 수량: {quantity}")
        print(f"      - 레버리지: {leverage}x")
        print(f"      - 포지션 크기: ${position_size:,.2f}")
        print(f"      - 실제 노출: ${position_size * leverage:,.2f}")
        
        # 5. 시장가 주문
        order_side = 'buy' if side.upper() == 'LONG' else 'sell'
        order = create_market_order(symbol, order_side, quantity)
        
        if not order:
            print(f"   ❌ 주문 실패")
            return False
        
        # 6. 스탑로스 계산 및 설정
        if side.upper() == 'LONG':
            stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        else:
            stop_loss_price = current_price * (1 + stop_loss_pct / 100)
        
        stop_loss_price = round(stop_loss_price, precision['price'])
        
        # 🆕 TP 제거, SL만 설정
        sl_order = set_stop_loss_order(symbol, side, quantity, stop_loss_price)
        
        if not sl_order:
            print(f"   ⚠️ 스탑로스 설정 실패 - 수동 관리 필요!")
        
        # 7. 🆕 트레일링 스탑 즉시 설정 (레버리지 고려)
        if LIVE_TRADING_CONFIG.get("TRAILING_STOP_ENABLED", False):
            print(f"\n   🎯 트레일링 스탑 설정 중...")
            trailing_order = set_trailing_stop_order(symbol, side, leverage)
            
            if trailing_order:
                print(f"   ✅ 트레일링 스탑 활성화됨")
            else:
                print(f"   ⚠️ 트레일링 스탑 설정 실패")
        
        # 8. DB 기록
        trade_id = record_trade(
            coin_symbol=coin,
            side=side,
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            position_size=position_size,
            ai_confidence=confidence,
            ai_reasoning=decision.get('reasoning', ''),
            stop_loss_price=stop_loss_price,
            take_profit_price=0.0,  # 🆕 TP 제거
            manual_trade=False
        )
        
        print(f"\n   ✅ 거래 진입 완료 (ID: {trade_id})")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 실거래 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===== 포지션 관리 =====

def manage_live_positions():
    """
    🔧 AI 기반 실시간 포지션 관리 (최소 보유시간 제약 제거)
    
    전략:
    - 진입 직후라도 AI가 청산 필요 판단 시 즉시 청산
    - 변동성에 따라 청산 임계값 자동 조정
    - 수동거래는 항상 보호
    """
    print(f"\n{'='*80}")
    print(f"🔍 AI 포지션 평가 시작")
    print(f"{'='*80}")
    
    open_trades = get_all_open_trades()
    
    if not open_trades:
        print(f"   ℹ️  오픈 포지션 없음")
        return
    
    # 바이낸스 실제 포지션 조회
    live_positions = get_open_positions()
    live_position_map = {pos['symbol']: pos for pos in live_positions}
    
    for trade in open_trades:
        coin = trade['coin_symbol']
        symbol = f"{coin}/USDT:USDT"
        trade_id = trade['id']
        side = trade['side']
        entry_price = trade['entry_price']
        entry_time = parse_db_timestamp(trade['timestamp'])
        
        # 수동거래 보호
        if trade.get('manual_trade', False):
            print(f"\n   🛡️  {coin} (ID: {trade_id}): 수동거래 - AI 청산 스킵")
            continue
        
        # 바이낸스에 실제 포지션이 없으면 DB 정리
        if symbol not in live_position_map:
            print(f"\n   ⚠️  {coin} (ID: {trade_id}): 바이낸스에 포지션 없음 - DB 정리")
            # 🆕 Position History에서 정확한 PnL 조회
            try:
                entry_timestamp = parse_db_timestamp(trade['timestamp'])
                if entry_timestamp:
                    start_time_ms = int(entry_timestamp.timestamp() * 1000)
                else:
                    start_time_ms = None
                
                pnl_data = get_accurate_pnl_from_history(symbol, start_time_ms)
                
                if pnl_data and pnl_data['realized_pnl'] != 0:
                    # Position History에서 가져온 정확한 PnL 사용
                    realized_pnl = pnl_data['realized_pnl']
                    
                    # 현재 가격 조회 (청산가 기록용)
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # 수익률 계산 (역산)
                    pnl_pct = (realized_pnl / trade['position_size']) * 100
                    
                    print(f"      ✅ Position History 정확한 PnL: ${realized_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                else:
                    # Position History에 없으면 수동 계산
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    if side.upper() == 'LONG':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * trade['leverage']
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100 * trade['leverage']
                    
                    realized_pnl = trade['position_size'] * (pnl_pct / 100)
                    print(f"      ⚠️ 수동 계산 PnL: ${realized_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                
                update_trade_exit(
                    trade_id=trade_id,
                    exit_price=current_price,
                    pnl=realized_pnl,
                    pnl_percent=pnl_pct,
                    reason="DB 동기화: 바이낸스에 포지션 없음"
                )
                
                print(f"      ✅ DB 정리 완료")
            except Exception as e:
                print(f"      ❌ DB 정리 실패: {e}")
            
            continue
        
        # 현재 가격 조회
        live_position = live_position_map[symbol]
        current_price = float(live_position['markPrice'])
        unrealized_pnl = float(live_position['unrealizedPnl'])
        
        # 수익률 계산
        if side.upper() == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * trade['leverage']
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * trade['leverage']
        
        # 변동성 계산 (24h 변동률)
        try:
            ticker = exchange.fetch_ticker(symbol)
            volatility = abs(ticker.get('percentage', 0))
        except:
            volatility = 5.0  # 기본값
        
        # 변동성 기반 청산 임계값 조정
        if volatility > LIVE_TRADING_CONFIG.get('HIGH_VOLATILITY_THRESHOLD', 5.0):
            # 고변동성: 더 빠른 청산 (±15%)
            exit_threshold = 15
        else:
            # 저변동성: 더 긴 보유 (±20%)
            exit_threshold = 20
        
        print(f"\n   📊 {coin} (ID: {trade_id}, {side.upper()}):")
        print(f"      - 진입가: ${entry_price:,.2f}")
        print(f"      - 현재가: ${current_price:,.2f}")
        print(f"      - 수익률: {pnl_pct:+.2f}%")
        print(f"      - 미실현 손익: ${unrealized_pnl:+,.2f}")
        print(f"      - 변동성: {volatility:.1f}% (임계값: ±{exit_threshold}%)")
        
        # AI 청산 판단 (진입 시간 제약 없음)
        should_exit = False
        exit_reason = ""
        
        if pnl_pct >= exit_threshold:
            should_exit = True
            exit_reason = f"목표 수익 달성 ({pnl_pct:+.2f}% ≥ +{exit_threshold}%)"
        elif pnl_pct <= -exit_threshold:
            should_exit = True
            exit_reason = f"손실 제한 ({pnl_pct:+.2f}% ≤ -{exit_threshold}%)"
        
        if should_exit:
            print(f"      🔴 AI 청산 결정: {exit_reason}")
            
            # 청산 실행
            if close_position(symbol, side):
                # 🆕 Position History에서 정확한 PnL 조회
                entry_timestamp = parse_db_timestamp(trade['timestamp'])
                if entry_timestamp:
                    start_time_ms = int(entry_timestamp.timestamp() * 1000)
                else:
                    start_time_ms = None
                
                pnl_data = get_accurate_pnl_from_history(symbol, start_time_ms)
                
                if pnl_data and pnl_data['realized_pnl'] != 0:
                    # 정확한 PnL 사용
                    actual_pnl = pnl_data['realized_pnl']
                    actual_pnl_pct = (actual_pnl / trade['position_size']) * 100
                    
                    print(f"      ✅ Position History 정확한 PnL: ${actual_pnl:+,.2f} ({actual_pnl_pct:+.2f}%)")
                else:
                    # Position History 없으면 계산값 사용
                    actual_pnl = trade['position_size'] * (pnl_pct / 100)
                    actual_pnl_pct = pnl_pct
                    print(f"      ⚠️ 수동 계산 PnL 사용: ${actual_pnl:+,.2f} ({actual_pnl_pct:+.2f}%)")
                
                # DB 업데이트
                update_trade_exit(
                    trade_id=trade_id,
                    exit_price=current_price,
                    pnl=actual_pnl,
                    pnl_percent=actual_pnl_pct,
                    reason=exit_reason
                )
                print(f"      ✅ 청산 완료 및 DB 업데이트")
            else:
                print(f"      ❌ 청산 실패 - 재시도 필요")
        else:
            print(f"      ✅ 포지션 유지 (임계값 내: {pnl_pct:+.2f}%)")
    
    print(f"{'='*80}\n")

# ===== DB-바이낸스 동기화 =====

def sync_db_with_binance() -> int:
    """
    DB와 바이낸스 포지션 동기화
    
    로직:
    1. 바이낸스에 없는 DB 포지션 -> 청산 처리
    2. DB에 없는 바이낸스 포지션 -> DB에 추가 (수동거래로 등록)
    
    Returns:
        동기화된 거래 수
    """
    print(f"\n{'='*80}")
    print(f"🔄 DB-바이낸스 동기화")
    print(f"{'='*80}")
    
    synced_count = 0
    
    # DB 오픈 거래
    open_trades = get_all_open_trades()
    
    # 바이낸스 실제 포지션
    live_positions = get_open_positions()
    live_position_map = {pos['symbol']: pos for pos in live_positions}
    
    # 1. DB에는 있는데 바이낸스에 없는 포지션 -> 청산 처리
    for trade in open_trades:
        coin = trade['coin_symbol']
        symbol = f"{coin}/USDT:USDT"
        trade_id = trade['id']
        
        # 수동거래는 동기화 스킵
        if trade.get('manual_trade', False):
            print(f"   🛡️  {coin} (ID: {trade_id}): 수동거래 - 동기화 스킵")
            continue
        
        if symbol not in live_position_map:
            print(f"   ⚠️  {coin} (ID: {trade_id}): 바이낸스에 포지션 없음")
            
            # 🆕 Position History에서 정확한 PnL 조회
            try:
                entry_timestamp = parse_db_timestamp(trade['timestamp'])
                if entry_timestamp:
                    start_time_ms = int(entry_timestamp.timestamp() * 1000)
                else:
                    start_time_ms = None
                
                pnl_data = get_accurate_pnl_from_history(symbol, start_time_ms)
                
                if pnl_data and pnl_data['realized_pnl'] != 0:
                    # Position History에서 가져온 정확한 PnL 사용
                    realized_pnl = pnl_data['realized_pnl']
                    
                    # 현재 가격 조회 (청산가 기록용)
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # 수익률 계산 (역산)
                    pnl_pct = (realized_pnl / trade['position_size']) * 100
                    
                    print(f"      ✅ Position History에서 정확한 PnL 조회:")
                    print(f"         - 실현 손익: ${realized_pnl:+,.2f}")
                    print(f"         - 수익률: {pnl_pct:+.2f}%")
                else:
                    # Position History에 없으면 수동 계산
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    side = trade['side']
                    entry_price = trade['entry_price']
                    leverage = trade['leverage']
                    
                    if side.upper() == 'LONG':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
                    
                    realized_pnl = trade['position_size'] * (pnl_pct / 100)
                    print(f"      ⚠️ Position History 없음 - 수동 계산 사용")
                
                # DB 청산 처리
                update_trade_exit(
                    trade_id=trade_id,
                    exit_price=current_price,
                    pnl=realized_pnl,
                    pnl_percent=pnl_pct,
                    reason="동기화: 바이낸스에 포지션 없음 (이미 청산됨)"
                )
                
                print(f"      ✅ DB 정리 완료: {pnl_pct:+.2f}% (${realized_pnl:+,.2f})")
                synced_count += 1
                
            except Exception as e:
                print(f"      ❌ 동기화 실패: {e}")
    
    # 2. 바이낸스에는 있는데 DB에 없는 포지션 -> DB에 추가 (수동거래)
    db_symbols = {f"{t['coin_symbol']}/USDT:USDT" for t in open_trades}
    
    for symbol, position in live_position_map.items():
        if symbol not in db_symbols:
            coin = symbol.split('/')[0]
            side = 'LONG' if position['side'] == 'long' else 'SHORT'
            entry_price = float(position['entryPrice'])
            quantity = abs(float(position['contracts']))
            leverage = int(position.get('leverage', 1))
            notional = float(position['notional'])
            
            print(f"   🆕 {coin}: 바이낸스 포지션 발견 (DB에 없음)")
            print(f"      - 방향: {side}")
            print(f"      - 진입가: ${entry_price:,.2f}")
            print(f"      - 수량: {quantity}")
            print(f"      - 레버리지: {leverage}x")
            print(f"      - 포지션 크기: ${notional:,.2f}")
            
            # DB에 수동거래로 등록
            try:
                trade_id = record_trade(
                    coin_symbol=coin,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    leverage=leverage,
                    position_size=notional / leverage,
                    ai_confidence=0,
                    ai_reasoning="동기화: 바이낸스에서 감지된 수동거래",
                    stop_loss_price=0,
                    take_profit_price=0,
                    manual_trade=True  # 수동거래로 표시
                )
                
                print(f"      ✅ DB에 수동거래로 등록됨 (ID: {trade_id})")
                synced_count += 1
                
            except Exception as e:
                print(f"      ❌ DB 등록 실패: {e}")
    
    print(f"\n   📊 동기화 결과: {synced_count}개 거래 처리")
    print(f"{'='*80}\n")
    
    return synced_count

# ===== 수동거래 관리 =====

def register_manual_trade(coin: str, side: str, entry_price: float, quantity: float, leverage: int) -> int:
    """수동거래 등록"""
    position_size = entry_price * quantity / leverage
    
    trade_id = record_trade(
        coin_symbol=coin,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        leverage=leverage,
        position_size=position_size,
        ai_confidence=0,
        ai_reasoning="수동거래",
        stop_loss_price=0,
        take_profit_price=0,
        manual_trade=True
    )
    
    print(f"✅ 수동거래 등록 완료 (ID: {trade_id})")
    return trade_id

def protect_trade(trade_id: int) -> bool:
    """기존 거래를 수동거래로 보호"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("UPDATE trades SET manual_trade = 1 WHERE id = ?", (trade_id,))
    conn.commit()
    rows_affected = c.rowcount
    conn.close()
    
    if rows_affected > 0:
        print(f"✅ 거래 ID {trade_id}를 AI 청산에서 보호했습니다")
        return True
    else:
        print(f"❌ 거래 ID {trade_id}를 찾을 수 없습니다")
        return False

def unprotect_trade(trade_id: int) -> bool:
    """거래 보호 해제"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("UPDATE trades SET manual_trade = 0 WHERE id = ?", (trade_id,))
    conn.commit()
    rows_affected = c.rowcount
    conn.close()
    
    if rows_affected > 0:
        print(f"✅ 거래 ID {trade_id}의 보호를 해제했습니다 (AI 청산 허용)")
        return True
    else:
        print(f"❌ 거래 ID {trade_id}를 찾을 수 없습니다")
        return False

def display_manual_trades_status():
    """수동거래 현황 표시"""
    open_trades = get_all_open_trades()
    manual_trades = [t for t in open_trades if t.get('manual_trade', False)]
    
    if not manual_trades:
        print("📊 수동거래 없음")
        return
    
    print(f"\n{'='*80}")
    print(f"🛡️  수동거래 현황 ({len(manual_trades)}개)")
    print(f"{'='*80}")
    
    for trade in manual_trades:
        coin = trade['coin_symbol']
        side = trade['side']
        entry_price = trade['entry_price']
        
        # 현재 가격 조회
        try:
            symbol = f"{coin}/USDT:USDT"
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            if side.upper() == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * trade['leverage']
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100 * trade['leverage']
            
            print(f"   {coin} (ID: {trade['id']})")
            print(f"      - 방향: {side.upper()}")
            print(f"      - 진입가: ${entry_price:,.2f}")
            print(f"      - 현재가: ${current_price:,.2f}")
            print(f"      - 수익률: {pnl_pct:+.2f}%")
            print()
        except Exception as e:
            print(f"   {coin} (ID: {trade['id']}): 가격 조회 실패")
    
    print(f"{'='*80}\n")

def list_manual_trades():
    """수동거래 목록 출력"""
    display_manual_trades_status()

def add_manual_long(coin: str, entry_price: float, quantity: float, leverage: int) -> int:
    """롱 수동거래 등록"""
    return register_manual_trade(coin, "LONG", entry_price, quantity, leverage)

def add_manual_short(coin: str, entry_price: float, quantity: float, leverage: int) -> int:
    """숏 수동거래 등록"""
    return register_manual_trade(coin, "SHORT", entry_price, quantity, leverage)

def check_trade_protection(trade_id: int):
    """특정 거래 보호 상태 확인"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT coin_symbol, side, manual_trade, status 
        FROM trades 
        WHERE id = ?
    ''', (trade_id,))
    
    row = c.fetchone()
    conn.close()
    
    if not row:
        print(f"❌ 거래 ID {trade_id}를 찾을 수 없습니다")
        return
    
    coin, side, manual_trade, status = row
    protection = "🛡️ 보호됨 (AI 청산 불가)" if manual_trade else "⚠️ 보호 안됨 (AI 청산 가능)"
    
    print(f"\n거래 ID {trade_id}:")
    print(f"   - 코인: {coin}")
    print(f"   - 방향: {side}")
    print(f"   - 상태: {status}")
    print(f"   - 보호: {protection}\n")

def show_all_trades_protection():
    """모든 거래 보호 상태 표시"""
    open_trades = get_all_open_trades()
    
    if not open_trades:
        print("📊 오픈 거래 없음")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 전체 거래 보호 상태 ({len(open_trades)}개)")
    print(f"{'='*80}")
    
    for trade in open_trades:
        protection = "🛡️ 보호됨" if trade.get('manual_trade', False) else "⚠️ AI 관리"
        print(f"   ID {trade['id']}: {trade['coin_symbol']} ({trade['side']}) - {protection}")
    
    print(f"{'='*80}\n")

# ===== 성과 분석 =====

def ai_performance_review():
    """AI 성과 리뷰"""
    print(f"\n{'='*80}")
    print(f"📊 AI 성과 리뷰")
    print(f"{'='*80}")
    
    performance = get_recent_performance(7)
    
    print(f"   기간: 최근 {performance['days']}일")
    print(f"   총 거래: {performance['total_trades']}회")
    print(f"   승률: {performance['win_rate']:.1f}%")
    print(f"   총 손익: ${performance['total_pnl']:+,.2f}")
    print(f"   평균 수익: ${performance['avg_win']:,.2f}")
    print(f"   평균 손실: ${performance['avg_loss']:,.2f}")
    
    # 간단한 AI 평가
    if performance['total_trades'] >= 10:
        if performance['win_rate'] >= 55 and performance['total_pnl'] > 0:
            verdict = "✅ 우수"
        elif performance['win_rate'] >= 45 and performance['total_pnl'] >= 0:
            verdict = "⚠️ 보통"
        else:
            verdict = "🔴 개선 필요"
        
        print(f"   평가: {verdict}")
    else:
        print(f"   평가: 데이터 부족 (10회 이상 필요)")
    
    print(f"{'='*80}\n")

# ===== 대시보드 =====

def display_dashboard():
    """실시간 대시보드"""
    print(f"\n{'='*80}")
    print(f"📊 실시간 대시보드")
    print(f"{'='*80}")
    
    # 잔고
    available_balance = get_available_balance()
    
    # 오픈 포지션
    open_trades = get_all_open_trades()
    live_positions = get_open_positions()
    
    print(f"\n💰 계정 정보:")
    print(f"   - Available Balance: ${available_balance:,.2f}")
    print(f"   - 오픈 포지션: {len(live_positions)}개 (최대 {LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개)")
    
    if live_positions:
        total_unrealized_pnl = sum(float(pos['unrealizedPnl']) for pos in live_positions)
        print(f"   - 미실현 손익: ${total_unrealized_pnl:+,.2f}")
        
        print(f"\n📈 포지션 목록:")
        for pos in live_positions:
            coin = pos['symbol'].split('/')[0]
            side = pos['side'].upper()
            entry = float(pos['entryPrice'])
            current = float(pos['markPrice'])
            pnl = float(pos['unrealizedPnl'])
            
            print(f"   - {coin}: {side} | 진입 ${entry:,.2f} | 현재 ${current:,.2f} | PnL ${pnl:+,.2f}")
    
    # 최근 성과
    performance = get_recent_performance(7)
    print(f"\n📊 최근 7일 성과:")
    print(f"   - 총 거래: {performance['total_trades']}회")
    print(f"   - 승률: {performance['win_rate']:.1f}%")
    print(f"   - 총 손익: ${performance['total_pnl']:+,.2f}")
    
    print(f"{'='*80}\n")

# ===== 메인 루프 =====

def main():
    """메인 루프"""
    print(f"\n{'='*80}")
    print(f"🚀 AI 실거래 봇 시작")
    print(f"{'='*80}\n")
    
    # DB 초기화
    init_db()
    
    # 초기 설정 출력
    print(f"⚙️  설정:")
    print(f"   - 최대 동시 포지션: {LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개")
    print(f"   - 신규 진입 분석: {LIVE_TRADING_CONFIG['AI_ANALYSIS_INTERVAL']}초마다")
    print(f"   - 포지션 관리: {LIVE_TRADING_CONFIG['POSITION_CHECK_INTERVAL']}초마다")
    print(f"   - 성과 리뷰: {LIVE_TRADING_CONFIG['PERFORMANCE_REVIEW_INTERVAL']}초마다")
    print(f"   - 레버리지: {LIVE_TRADING_CONFIG['CONSERVATIVE_LEVERAGE']}-{LIVE_TRADING_CONFIG['MAX_LEVERAGE']}x")
    print(f"   - 마진 모드: {LIVE_TRADING_CONFIG['MARGIN_MODE'].upper()}")
    print(f"{'='*80}\n")
    
    # 🆕 트레일링 스탑 설정 출력
    if LIVE_TRADING_CONFIG.get("TRAILING_STOP_ENABLED", False):
        min_profit = LIVE_TRADING_CONFIG.get("TRAILING_STOP_MIN_PROFIT_PCT", 7.0)
        print(f"🎯 트레일링 스탑:")
        print(f"   ✅ 활성화됨 (진입 즉시)")
        print(f"   ✅ 최소 확보 수익률: {min_profit}%")
        print(f"   ✅ 콜백 비율: 최소 수익률 / 레버리지 (자동 계산)")
        print(f"   📝 예시: 레버리지 10x → 콜백 {min_profit/10:.1f}%")
        print(f"{'='*80}\n")
    
    # 3초 후 자동 시작
    print("3초 후 자동 시작...")
    time.sleep(3)
    
    last_analysis_time = 0
    last_review_time = 0
    last_dashboard_time = 0
    last_position_check_time = 0
    
    while True:
        try:
            current_time = time.time()
            now = get_utc_now()
            
            # 대시보드 표시
            if current_time - last_dashboard_time > 120:
                display_dashboard()
                last_dashboard_time = current_time
            
            print(f"\n{'='*80}")
            print(f"⏰ [{now.strftime('%Y-%m-%d %H:%M:%S')} UTC]")
            print(f"{'='*80}")
            
            # 잔고 조회
            available_balance = get_available_balance()
            
            # 🔧 포지션 관리 (10분마다)
            position_check_interval = LIVE_TRADING_CONFIG.get("POSITION_CHECK_INTERVAL", 0)
            if position_check_interval > 0 and current_time - last_position_check_time > position_check_interval:
                manage_live_positions()
                last_position_check_time = current_time
            
            # 🔧 포지션 수 체크: 바이낸스 실제 포지션 기준
            open_trades = get_all_open_trades()
            live_positions = get_open_positions()
            
            # 바이낸스 실제 포지션 수 사용 (DB와 불일치 가능)
            open_positions_count = len(live_positions)
            db_positions_count = len(open_trades)
            
            # 불일치 시 자동 동기화
            if db_positions_count != open_positions_count:
                print(f"   ⚠️ 포지션 불일치: DB {db_positions_count}개, 바이낸스 {open_positions_count}개")
                print(f"   🔄 자동 동기화 실행...")
                
                synced = sync_db_with_binance()
                if synced > 0:
                    # 동기화 후 다시 조회
                    open_trades = get_all_open_trades()
                    db_positions_count = len(open_trades)
                    open_positions_count = len(live_positions)
                    print(f"   ✅ 동기화 완료: DB {db_positions_count}개 = 바이낸스 {open_positions_count}개")
            
            # 성과 리뷰
            if current_time - last_review_time > LIVE_TRADING_CONFIG['PERFORMANCE_REVIEW_INTERVAL']:
                ai_performance_review()
                last_review_time = current_time
            
            # 포지션이 꽉 찼는지 체크 (바이낸스 실제 포지션 기준)
            max_positions = LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']
            if open_positions_count >= max_positions:
                print(f"\n{'='*80}")
                print(f"⏸️  포지션 풀 ({open_positions_count}/{max_positions}) - 신규 진입 분석 스킵")
                print(f"{'='*80}")
            elif current_time - last_analysis_time > LIVE_TRADING_CONFIG['AI_ANALYSIS_INTERVAL']:
                print(f"\n{'='*80}")
                print(f"🔍 AI 시장 분석")
                print(f"{'='*80}")
                
                top_coins = get_top_volume_coins(10)
                if top_coins:
                    open_coin_symbols = {t['coin_symbol'] for t in open_trades}
                    
                    for coin_data in top_coins:
                        # 포지션이 5개 차면 중단
                        if open_positions_count >= LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']:
                            print(f"\n   ⏸️  포지션 풀 ({open_positions_count}/{LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}) - 분석 중단")
                            break
                        
                        coin = coin_data['coin']
                        if coin in open_coin_symbols:
                            continue
                        
                        print(f"\n{'─'*70}")
                        print(f"📈 {coin}")
                        print(f"{'─'*70}")
                        
                        try:
                            symbol = f"{coin}/USDT:USDT"
                            print(f"   🔍 시장 데이터 수집 중...")
                            market_data = fetch_comprehensive_market_data(symbol)
                            
                            if not market_data:
                                print(f"   ⚠️ 시장 데이터 없음")
                                continue
                            
                            print(f"   ✅ {len(market_data)}개 타임프레임 데이터 수집 완료")
                            
                            performance_history = get_recent_performance(7)
                            print(f"   🤖 AI 분석 중...")
                            decision = ai_comprehensive_analysis(coin_data, market_data, performance_history)
                            
                            print(f"   신뢰도: {decision.get('confidence', 0)}%")
                            print(f"   판단: {decision.get('reasoning', 'N/A')}")
                            
                            # 70% 이상만 거래
                            if decision.get('trade') and decision.get('confidence', 0) >= 70:
                                print(f"\n   ✅ AI 승인")
                                if execute_live_trade(coin_data, decision, available_balance):
                                    open_positions_count += 1
                                    available_balance = get_available_balance()
                            else:
                                conf = decision.get('confidence', 0)
                                print(f"   ⏭️ 보류 (Confidence: {conf}%)")
                            
                            time.sleep(2)
                        except Exception as e:
                            print(f"   ❌ 분석 오류: {e}")
                
                last_analysis_time = current_time
            
            print(f"\n{'='*80}")
            print(f"💤 60초 대기...")
            print(f"{'='*80}\n")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"⚠️ 봇 종료")
            print(f"{'='*80}")
            display_dashboard()
            ai_performance_review()
            
            open_positions = get_open_positions()
            if open_positions:
                print(f"\n{'='*80}")
                print(f"⚠️  오픈 포지션이 {len(open_positions)}개 남아있습니다!")
                print(f"   바이낸스에서 수동으로 관리하세요.")
                print(f"{'='*80}")
                for pos in open_positions:
                    coin = pos['symbol'].split('/')[0]
                    print(f"   - {coin}: {pos['side'].upper()} ${pos['unrealizedPnl']:+,.2f}")
            
            print(f"\n{'='*80}")
            print(f"📜 거래 내역 요약")
            print(f"{'='*80}")
            
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
            total_trades = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND pnl > 0")
            winning = c.fetchone()[0]
            conn.close()
            
            print(f"   총 거래: {total_trades}회")
            print(f"   승리: {winning}회")
            print(f"   승률: {(winning/total_trades*100):.1f}%" if total_trades > 0 else "0%")
            
            print(f"{'='*80}")
            print(f"\n✅ 결과가 '{DB_FILE}'에 저장되었습니다.")
            print(f"👋 봇을 종료합니다.\n")
            break
            
        except Exception as e:
            # 🆕 즉시 flush하여 에러 로그 보장
            error_time = get_utc_now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'='*80}")
            print(f"❌ [{error_time} UTC] 오류 발생!")
            print(f"{'='*80}")
            print(f"오류 내용: {e}")
            sys.stdout.flush()
            
            import traceback
            print("\n스택 트레이스:")
            traceback.print_exc()
            sys.stdout.flush()
            
            print(f"\n⏳ 30초 후 재시도...")
            print(f"   봇은 계속 실행됩니다...")
            sys.stdout.flush()
            time.sleep(30)

if __name__ == "__main__":
    provider = AI_MODEL_CONFIG["provider"].upper()
    model_name = AI_MODEL_CONFIG["models"][AI_MODEL_CONFIG["provider"]]
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║        🔴 AI 실거래 트레이딩 봇 v2.1 (트레일링 스탑)        ║
║        🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS          ║
║        ⚠️  실제 자금으로 거래합니다!                         ║
║        ✅ AI: {provider:20s} ({model_name:20s})    ║
║        ✅ Isolated Margin 모드                                ║
║        ✅ Risk/Reward ≥ 1:2 전략                             ║
║        🆕 처음부터 트레일링 스탑 활성화                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  주의사항:")
    print("   1. 실제 바이낸스 계정의 자금을 사용합니다")
    print("   2. Isolated Margin 모드로 거래합니다")
    print("   3. 손실 가능성이 있으니 충분히 이해한 후 사용하세요")
    print("   4. 소액으로 테스트한 후 본격적으로 사용하세요")
    print(f"   5. 수수료: Maker 0.02%, Taker 0.05%")
    print(f"   6. 🆕 트레일링 스탑이 진입 즉시 활성화됩니다\n")
    
    # 자동 시작 (확인 없음)
    print("🚀 백그라운드 모드 - 자동 시작...")
    main()
