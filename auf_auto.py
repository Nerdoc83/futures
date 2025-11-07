"""
AI Live Trading Bot v2.5 (추세 반전 감지)
----------------------------------------------------------------------
⚠️ 실제 바이낸스 선물 거래 - 실제 자금 사용
- 실시간 바이낸스 선물 데이터 사용
- 실제 잔고로 거래 (Available Balance 기준)
- Isolated Margin 모드
- 실제 주문 체결 및 청산
- AI 보수적 리스크 관리
- 🆕 처음부터 트레일링 스탑 설정
- 🆕 명확한 추세 반전시 AI 개입 청산

🔧 v2.5 신규 기능:
  1. ✅ 명확한 추세 반전 감지 (+10% 이상 수익시)
  2. ✅ 다중 기술적 지표 분석 (RSI, MACD, EMA, 캔들 패턴)
  3. ✅ 신호 강도 점수화 (7/10 이상시만 청산)
  4. ✅ 최소 3개 이상의 동시 신호 요구
  5. ✅ 수익 보호 조기 청산 (트레일링 스탑 보완)

v2.4 기능:
  - 공격적 포지션 크기 (목표 80% 자금 사용)
  - 동적 균등 분할 및 변동성 조정

v2.3 기능:
  - 바이낸스 실제 realized PnL 조회
  - 수수료 반영 손익 계산
  - 정확한 손익 동기화

⚠️ 중요:
- AI는 10분마다 포지션 모니터링
- 추세 반전은 수익 중(+10% 이상)일 때만 체크
- 명확한 신호(강도 7/10 이상)에만 개입
- 트레일링 스탑은 계속 작동 (기본 보호)
- 추세 유지시 AI는 개입하지 않음
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
    "TRAILING_STOP_MIN_PROFIT_PCT": 12.0,       # 최소 확보 수익률 (%) - 레버리지 고려하여 자동 계산됨
                                                  # 예: 12% 수익 확보 + 10배 레버리지 = 1.2% 콜백
    
    # 🆕 추세 반전 감지 설정
    "TREND_REVERSAL_DETECTION": True,           # 추세 반전 감지 활성화
    "TREND_REVERSAL_MIN_PROFIT_PCT": 10,        # 최소 수익률 (10% 이상일 때만 체크)
    "TREND_REVERSAL_MIN_STRENGTH": 7,           # 최소 신호 강도 (7/10 이상)
    "TREND_REVERSAL_MIN_SIGNALS": 3,            # 최소 신호 개수 (3개 이상)
    
    # 🔧 자금 관리 설정 (공격적 균등 분할)
    "TARGET_TOTAL_USAGE_PCT": 100,  # 🆕 목표: 전체 자금의 100% 사용
    "MAX_POSITION_SIZE_PCT": 40,   # 🔧 단일 포지션 최대 40% (안전장치)
    "MIN_POSITION_SIZE_PCT": 15,   # 🔧 최소 15% (너무 작은 포지션 방지)
    "DYNAMIC_EQUAL_SPLIT": True,   # 동적 균등 분할 활성화
    "VOLATILITY_BASED_SIZING": True,  # 변동성 기반 포지션 크기 조절 (완화)
    "HIGH_VOLATILITY_THRESHOLD": 5.0,  # 5% 이상이면 고변동성
    "LOW_VOLATILITY_MULTIPLIER": 1.1,   # 🔧 저변동성 = 1.1배 투자 (완화)
    "HIGH_VOLATILITY_MULTIPLIER": 0.9,  # 🔧 고변동성 = 0.9배 투자 (완화)
    
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
        available = usdt_balance.get('free', 0.0)
        return float(available)
    except Exception as e:
        print(f"❌ 잔고 조회 실패: {e}")
        return 0.0

def cancel_all_pending_orders(symbol: str) -> bool:
    """특정 심볼의 모든 미체결 주문 취소"""
    try:
        # 오픈 주문 조회
        open_orders = exchange.fetch_open_orders(symbol)
        
        if not open_orders:
            return True
        
        print(f"   🔄 {symbol} 미체결 주문 {len(open_orders)}개 정리 중...")
        
        canceled_count = 0
        for order in open_orders:
            try:
                exchange.cancel_order(order['id'], symbol)
                canceled_count += 1
                print(f"      ✅ 주문 취소: {order['type']} (ID: {order['id']})")
            except Exception as e:
                print(f"      ❌ 주문 취소 실패 (ID: {order['id']}): {e}")
        
        print(f"   ✅ {canceled_count}개 주문 정리 완료")
        return True
        
    except Exception as e:
        print(f"   ❌ 주문 정리 실패: {e}")
        return False

def calculate_dynamic_callback_rate(symbol: str, leverage: int, base_profit_pct: float = 12.0) -> Tuple[float, str]:
    """
    변동성 기반 동적 콜백 비율 계산
    
    Args:
        symbol: 거래 심볼
        leverage: 레버리지
        base_profit_pct: 기본 목표 수익률 (%)
    
    Returns:
        (최적 콜백 비율, 변동성 수준)
    """
    try:
        # 1시간봉 데이터로 ATR 계산 (최근 변동성)
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # ATR 계산
        atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
        if atr_result is not None and len(atr_result) > 0:
            atr = float(atr_result.iloc[-1])
            current_price = float(df['close'].iloc[-1])
            atr_pct = (atr / current_price) * 100
        else:
            atr_pct = 3.0  # 기본값
        
        # 기본 콜백 비율 계산
        base_callback = base_profit_pct / leverage
        
        # 변동성에 따른 멀티플라이어
        if atr_pct < 1.5:
            # 초저변동성 (BTC, ETH 횡보)
            multiplier = 0.6
            volatility_level = "초저변동성"
        elif atr_pct < 3.0:
            # 저변동성 (BTC, ETH 일반)
            multiplier = 0.7
            volatility_level = "저변동성"
        elif atr_pct < 5.0:
            # 중변동성 (대부분의 알트코인)
            multiplier = 0.8
            volatility_level = "중변동성"
        elif atr_pct < 8.0:
            # 고변동성 (변동성 큰 알트코인)
            multiplier = 1.0
            volatility_level = "고변동성"
        else:
            # 초고변동성 (밈코인, 신규코인)
            multiplier = 1.3
            volatility_level = "초고변동성"
        
        # 최종 콜백 비율
        callback_rate = base_callback * multiplier
        
        # 안전 범위 제한
        callback_rate = max(0.5, min(5.0, callback_rate))
        
        print(f"      📊 변동성 분석:")
        print(f"         - ATR: {atr_pct:.2f}% ({volatility_level})")
        print(f"         - 기본 콜백: {base_callback:.2f}%")
        print(f"         - 변동성 조정: ×{multiplier}")
        print(f"         - 최종 콜백: {callback_rate:.2f}%")
        
        return callback_rate, volatility_level
        
    except Exception as e:
        print(f"      ⚠️ 변동성 계산 실패: {e}")
        # 기본값 반환
        return base_profit_pct / leverage, "중변동성"

def set_trailing_stop_order(symbol: str, side: str, leverage: int, position_size: float) -> Optional[dict]:
    """
    🔧 트레일링 스탑 설정 (재시도 로직 포함)
    
    Args:
        symbol: 거래 심볼 (예: BTC/USDT:USDT)
        side: 포지션 방향 ('LONG' 또는 'SHORT')
        leverage: 레버리지
        position_size: 포지션 크기 (USDT)
    """
    max_retries = 3
    base_wait_time = 2
    
    # 최소 확보 수익률
    min_profit_pct = LIVE_TRADING_CONFIG.get("TRAILING_STOP_MIN_PROFIT_PCT", 12.0)
    
    print(f"   🎯 트레일링 스탑 설정 시작...")
    print(f"      - 목표 수익률: {min_profit_pct}%")
    
    # 🆕 변동성 기반 동적 콜백 계산
    dynamic_callback, volatility_level = calculate_dynamic_callback_rate(symbol, leverage, min_profit_pct)
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   🔄 시도 {attempt}/{max_retries}...")
            
            # 포지션 확립 대기 (첫 번째 시도가 아닐 때)
            if attempt > 1:
                wait_time = base_wait_time * attempt
                print(f"      ⏱️ {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
            
            # 콜백 비율 조정 (재시도마다 약간씩 증가)
            if attempt == 1:
                callback_rate = dynamic_callback
            elif attempt == 2:
                callback_rate = dynamic_callback + 0.2
            else:
                callback_rate = dynamic_callback + 0.4
            
            # 최대 5% 제한
            callback_rate = min(5.0, callback_rate)
            
            coin_name = symbol.split('/')[0]
            print(f"      📏 콜백 비율: {callback_rate:.2f}% ({coin_name} {volatility_level})")
            
            # 포지션 정보 조회
            positions = exchange.fetch_positions([symbol])
            current_position = None
            
            for pos in positions:
                if abs(float(pos['contracts'])) > 0:
                    current_position = pos
                    break
            
            if not current_position:
                print(f"      ❌ 활성 포지션 없음 - 재시도 필요")
                if attempt < max_retries:
                    continue
                else:
                    print(f"   ❌ 최종 실패: 포지션 없음")
                    return None
            
            # 포지션 수량 확인
            position_contracts = abs(float(current_position['contracts']))
            
            print(f"      🔍 포지션 확인:")
            print(f"         - 수량: {position_contracts:.2f}")
            print(f"         - 방향: {side.upper()}")
            
            # 트레일링 스탑 파라미터 (바이낸스 선물 표준)
            # activationPrice 없이 설정하면 현재가에서 즉시 활성화
            params = {
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'workingType': 'CONTRACT_PRICE'
            }
            
            print(f"      🔧 트레일링 스탑 주문:")
            print(f"         - 콜백 비율: {callback_rate}%")
            print(f"         - 활성화: 즉시 (현재가)")
            print(f"         - 수량: {position_contracts:.2f}")
            
            # 트레일링 스탑 마켓 주문 생성
            order = exchange.create_order(
                symbol=symbol,
                type='TRAILING_STOP_MARKET',  # 트레일링 스탑 주문
                side='sell' if side.upper() == 'LONG' else 'buy',  # 청산 방향
                amount=position_contracts,  # 실제 포지션 수량
                price=None,
                params=params
            )
            
            # 성공!
            actual_profit = callback_rate * leverage
            print(f"   ✅ 트레일링 스탑 설정 성공! (시도 {attempt}/{max_retries})")
            print(f"      - 콜백 비율: {callback_rate:.2f}%")
            print(f"      - 확보 수익률: {actual_profit:.1f}%")
            print(f"      - 주문 ID: {order.get('id', 'N/A')}")
            
            return order
            
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ 시도 {attempt} 실패: {error_msg}")
            
            # 특정 오류에 대한 대응
            if 'leverage' in error_msg.lower() and attempt == 1:
                print(f"      🔧 레버리지 오류 감지 - 레버리지 재설정 시도")
                try:
                    reduced_leverage = max(1, leverage // 2)
                    exchange.set_leverage(reduced_leverage, symbol)
                    print(f"      ✅ 레버리지 {reduced_leverage}x로 조정")
                    time.sleep(1)
                except Exception as lev_e:
                    print(f"      ❌ 레버리지 조정 실패: {lev_e}")
            
            # 마지막 시도가 아니면 계속
            if attempt < max_retries:
                print(f"      🔄 재시도 준비 중...")
            else:
                print(f"   🚨 트레일링 스탑 설정 최종 실패!")
                print(f"   ⚠️ 포지션이 보호되지 않음 - 수동 설정 필요")
                print(f"   💡 권장 설정: {symbol} 트레일링 스탑 {dynamic_callback:.2f}% 콜백 ({volatility_level})")
                break
    
    return None

def get_open_positions() -> List[dict]:
    """바이낸스에서 실제 오픈 포지션 조회"""
    try:
        positions = exchange.fetch_positions()
        open_positions = []
        
        for position in positions:
            if abs(float(position['contracts'])) > 0:
                open_positions.append(position)
        
        return open_positions
    except Exception as e:
        print(f"❌ 포지션 조회 실패: {e}")
        return []

def close_position(symbol: str, side: str) -> bool:
    """
    포지션 청산 (시장가 주문) - 단순 청산만
    
    주의: 이 함수는 PnL을 반환하지 않습니다.
    PnL이 필요한 경우 close_position_and_get_pnl()을 사용하세요.
    """
    try:
        print(f"   🔄 {symbol} {side} 포지션 청산 중...")
        
        # 미체결 주문 먼저 정리
        cancel_all_pending_orders(symbol)
        
        # 포지션 정보 조회
        positions = exchange.fetch_positions([symbol])
        target_position = None
        
        for pos in positions:
            if abs(float(pos['contracts'])) > 0:
                target_position = pos
                break
        
        if not target_position:
            print(f"      ⚠️ 청산할 포지션 없음")
            return True  # 이미 청산됨
        
        # 청산 주문 생성 (시장가)
        position_side = target_position['side']
        contracts = abs(float(target_position['contracts']))
        
        if position_side == 'long':
            order_side = 'sell'
        else:
            order_side = 'buy'
        
        order = exchange.create_market_order(
            symbol=symbol,
            side=order_side,
            amount=contracts,
            params={'reduceOnly': True}
        )
        
        print(f"      ✅ 청산 주문 체결 (ID: {order.get('id', 'N/A')})")
        
        # 청산 확인 (최대 3초 대기)
        for i in range(3):
            time.sleep(1)
            positions = exchange.fetch_positions([symbol])
            has_position = any(abs(float(pos['contracts'])) > 0 for pos in positions)
            
            if not has_position:
                print(f"      ✅ 포지션 청산 확인")
                return True
        
        print(f"      ⚠️ 청산 확인 실패 (시간 초과)")
        return False
        
    except Exception as e:
        print(f"      ❌ 청산 실패: {e}")
        return False

def close_position_and_get_pnl(symbol: str, side: str, trade_info: dict) -> Tuple[bool, float, float, float]:
    """
    🆕 포지션 청산 + 즉시 실제 PnL 조회
    
    Args:
        symbol: 거래 심볼 (예: "ZEC/USDT:USDT")
        side: 'LONG' 또는 'SHORT'
        trade_info: DB 거래 정보 (position_size, leverage, entry_price 포함)
    
    Returns:
        (success, exit_price, realized_pnl, pnl_percentage)
    """
    try:
        print(f"   🔄 {symbol} {side} 포지션 청산 중...")
        
        # 1. 미체결 주문 정리
        cancel_all_pending_orders(symbol)
        
        # 2. 포지션 정보 조회
        positions = exchange.fetch_positions([symbol])
        target_position = None
        
        for pos in positions:
            if abs(float(pos['contracts'])) > 0:
                target_position = pos
                break
        
        if not target_position:
            print(f"      ⚠️ 청산할 포지션 없음")
            return True, 0.0, 0.0, 0.0  # 이미 청산됨
        
        # 3. 청산 전 현재가 기록 (fallback용)
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
        except:
            current_price = float(target_position['markPrice'])
        
        # 4. 청산 주문 생성
        position_side = target_position['side']
        contracts = abs(float(target_position['contracts']))
        
        if position_side == 'long':
            order_side = 'sell'
        else:
            order_side = 'buy'
        
        order = exchange.create_market_order(
            symbol=symbol,
            side=order_side,
            amount=contracts,
            params={'reduceOnly': True}
        )
        
        print(f"      ✅ 청산 주문 체결 (ID: {order.get('id', 'N/A')})")
        
        # 5. 청산 확인 (최대 3초 대기)
        close_confirmed = False
        for i in range(3):
            time.sleep(1)
            positions = exchange.fetch_positions([symbol])
            has_position = any(abs(float(pos['contracts'])) > 0 for pos in positions)
            
            if not has_position:
                print(f"      ✅ 포지션 청산 확인")
                close_confirmed = True
                break
        
        if not close_confirmed:
            print(f"      ⚠️ 청산 확인 실패 (시간 초과)")
            # 그래도 PnL 조회는 시도
        
        # 6. 🆕🆕🆕 즉시 바이낸스 실제 PnL 조회 (가장 중요!)
        print(f"      🔍 바이낸스 실제 PnL 조회 중...")
        time.sleep(0.5)  # API 동기화 대기
        
        pnl_data = get_realized_pnl_from_binance(symbol, start_time=None)
        
        if pnl_data and pnl_data['realized_pnl'] != 0:
            # ✅ 실제 PnL 조회 성공!
            realized_pnl = pnl_data['realized_pnl']
            commission = pnl_data['commission']
            
            # 실제 청산가 역산 (근사)
            entry_price = trade_info['entry_price']
            leverage = trade_info['leverage']
            position_size = trade_info['position_size']
            
            if side.upper() == 'LONG':
                pnl_pct = (realized_pnl / position_size) * 100
                price_move_pct = pnl_pct / leverage
                exit_price = entry_price * (1 + price_move_pct / 100)
            else:
                pnl_pct = (realized_pnl / position_size) * 100
                price_move_pct = pnl_pct / leverage
                exit_price = entry_price * (1 - price_move_pct / 100)
            
            print(f"      ✅ 바이낸스 실제 PnL: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"      💰 수수료: ${commission:.4f}")
            print(f"      📍 실제 청산가: ${exit_price:,.4f}")
            
            return True, exit_price, realized_pnl, pnl_pct
        
        else:
            # ⚠️ 조회 실패 -> fallback: 추정 계산
            print(f"      ⚠️ 바이낸스 PnL 조회 실패 - 추정 계산 사용")
            
            realized_pnl, pnl_pct = calculate_pnl_from_price(
                trade_info['entry_price'], 
                current_price,
                trade_info['position_size'], 
                side, 
                trade_info['leverage']
            )
            
            print(f"      📊 추정 PnL: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"      ⚠️ 실제 값과 다를 수 있음")
            
            return True, current_price, realized_pnl, pnl_pct
        
    except Exception as e:
        print(f"      ❌ 청산 실패: {e}")
        return False, 0.0, 0.0, 0.0

def calculate_pnl_from_price(entry_price: float, exit_price: float, position_size: float, side: str, leverage: int) -> Tuple[float, float]:
    """
    🆕 수수료를 반영한 PnL 계산 (개선 버전)
    
    Returns:
        (realized_pnl, pnl_percentage)
    """
    try:
        # 실제 거래 금액 (레버리지 적용)
        notional_value = position_size * leverage
        
        # 손익 계산 (레버리지 반영)
        if side.upper() == 'LONG':
            price_diff_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            price_diff_pct = ((entry_price - exit_price) / entry_price) * 100
        
        pnl_pct = price_diff_pct * leverage
        pnl_before_fees = position_size * (pnl_pct / 100)
        
        # 수수료 계산 (진입 + 청산)
        maker_fee = LIVE_TRADING_CONFIG.get('MAKER_FEE', 0.02)
        taker_fee = LIVE_TRADING_CONFIG.get('TAKER_FEE', 0.05)
        
        entry_fee = notional_value * (maker_fee / 100)
        exit_fee = notional_value * (taker_fee / 100)
        total_fees = entry_fee + exit_fee
        
        # 최종 실현 손익 (수수료 차감)
        realized_pnl = pnl_before_fees - total_fees
        
        # 수수료 반영 수익률
        final_pnl_pct = (realized_pnl / position_size) * 100
        
        return realized_pnl, final_pnl_pct
        
    except Exception as e:
        print(f"   ❌ PnL 계산 오류: {e}")
        return 0.0, 0.0

def detect_trend_reversal(symbol: str, side: str, current_price: float, entry_price: float) -> Tuple[bool, str, int]:
    """
    🆕 명확한 추세 반전 감지
    
    Args:
        symbol: 거래 심볼
        side: 'LONG' 또는 'SHORT'
        current_price: 현재가
        entry_price: 진입가
    
    Returns:
        (is_reversed, reason, strength)
        - is_reversed: 추세 반전 여부
        - reason: 반전 이유 설명
        - strength: 신호 강도 (0-10)
    """
    try:
        # 5분봉 기술적 지표 조회 (단기 추세 파악)
        ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 기술적 지표 계산
        close_series = df['close']
        high_series = df['high']
        low_series = df['low']
        
        # RSI
        try:
            rsi = ta.rsi(close_series, length=14)
            current_rsi = float(rsi.iloc[-1]) if rsi is not None else 50.0
        except:
            current_rsi = 50.0
        
        # MACD
        try:
            macd_result = ta.macd(close_series, fast=12, slow=26, signal=9)
            if macd_result is not None and isinstance(macd_result, pd.DataFrame):
                macd_line = float(macd_result.iloc[-1, 0])
                signal_line = float(macd_result.iloc[-1, 1])
                prev_macd = float(macd_result.iloc[-2, 0])
                prev_signal = float(macd_result.iloc[-2, 1])
            else:
                macd_line = signal_line = prev_macd = prev_signal = 0.0
        except:
            macd_line = signal_line = prev_macd = prev_signal = 0.0
        
        # EMA
        try:
            ema_20 = ta.ema(close_series, length=20)
            current_ema = float(ema_20.iloc[-1]) if ema_20 is not None else current_price
        except:
            current_ema = current_price
        
        # 최근 캔들 패턴 (마지막 3개 캔들)
        recent_candles = df.tail(3)
        bullish_candles = sum(1 for _, c in recent_candles.iterrows() if c['close'] > c['open'])
        bearish_candles = 3 - bullish_candles
        
        # 신호 점수 초기화
        reversal_signals = []
        signal_strength = 0
        
        # ===== SHORT 포지션 추세 반전 감지 =====
        if side.upper() == 'SHORT':
            # 신호 1: RSI 과매수 (70 이상)
            if current_rsi > 70:
                reversal_signals.append(f"RSI 과매수 ({current_rsi:.1f})")
                signal_strength += 3
            elif current_rsi > 65:
                reversal_signals.append(f"RSI 높음 ({current_rsi:.1f})")
                signal_strength += 2
            
            # 신호 2: MACD 상승 크로스
            if macd_line > signal_line and prev_macd <= prev_signal:
                reversal_signals.append("MACD 상승 크로스")
                signal_strength += 3
            elif macd_line > signal_line:
                reversal_signals.append("MACD 상승세")
                signal_strength += 1
            
            # 신호 3: 가격이 EMA20 돌파
            if current_price > current_ema * 1.01:  # 1% 이상 돌파
                reversal_signals.append(f"EMA20 돌파 ({((current_price/current_ema-1)*100):+.1f}%)")
                signal_strength += 3
            elif current_price > current_ema:
                reversal_signals.append("EMA20 상회")
                signal_strength += 1
            
            # 신호 4: 연속 상승 캔들
            if bullish_candles >= 3:
                reversal_signals.append("3연속 상승 캔들")
                signal_strength += 2
            elif bullish_candles == 2:
                signal_strength += 1
            
            # 신호 5: 진입가 대비 불리한 움직임
            price_move_pct = ((current_price - entry_price) / entry_price) * 100
            if price_move_pct > 2:  # 2% 이상 불리
                reversal_signals.append(f"진입가 대비 +{price_move_pct:.1f}% 불리")
                signal_strength += 2
        
        # ===== LONG 포지션 추세 반전 감지 =====
        else:  # LONG
            # 신호 1: RSI 과매도 (30 이하)
            if current_rsi < 30:
                reversal_signals.append(f"RSI 과매도 ({current_rsi:.1f})")
                signal_strength += 3
            elif current_rsi < 35:
                reversal_signals.append(f"RSI 낮음 ({current_rsi:.1f})")
                signal_strength += 2
            
            # 신호 2: MACD 하락 크로스
            if macd_line < signal_line and prev_macd >= prev_signal:
                reversal_signals.append("MACD 하락 크로스")
                signal_strength += 3
            elif macd_line < signal_line:
                reversal_signals.append("MACD 하락세")
                signal_strength += 1
            
            # 신호 3: 가격이 EMA20 이탈
            if current_price < current_ema * 0.99:  # 1% 이상 이탈
                reversal_signals.append(f"EMA20 이탈 ({((current_price/current_ema-1)*100):+.1f}%)")
                signal_strength += 3
            elif current_price < current_ema:
                reversal_signals.append("EMA20 하회")
                signal_strength += 1
            
            # 신호 4: 연속 하락 캔들
            if bearish_candles >= 3:
                reversal_signals.append("3연속 하락 캔들")
                signal_strength += 2
            elif bearish_candles == 2:
                signal_strength += 1
            
            # 신호 5: 진입가 대비 불리한 움직임
            price_move_pct = ((entry_price - current_price) / entry_price) * 100
            if price_move_pct > 2:  # 2% 이상 불리
                reversal_signals.append(f"진입가 대비 +{price_move_pct:.1f}% 불리")
                signal_strength += 2
        
        # ===== 명확한 반전 판단 =====
        # 조건: 신호 강도 7 이상 AND 3개 이상의 신호
        is_clear_reversal = signal_strength >= 7 and len(reversal_signals) >= 3
        
        reason = " | ".join(reversal_signals) if reversal_signals else "추세 유지"
        
        return is_clear_reversal, reason, signal_strength
        
    except Exception as e:
        print(f"      ⚠️ 추세 분석 오류: {e}")
        return False, "분석 실패", 0

def get_realized_pnl_from_binance(symbol: str, start_time: int = None) -> Optional[Dict]:
    """
    🆕 바이낸스 Income History에서 실제 realized PnL 조회
    
    Args:
        symbol: 심볼 (예: "ZEC/USDT:USDT")
        start_time: 조회 시작 시간 (밀리초), None이면 최근 내역만
    
    Returns:
        {
            'realized_pnl': float,
            'commission': float,
            'trade_time': int
        } 또는 None
    """
    try:
        # Income History API 호출
        binance_symbol = symbol.replace('/USDT:USDT', 'USDT')
        
        params = {
            'symbol': binance_symbol,
            'incomeType': 'REALIZED_PNL',
            'limit': 10
        }
        
        if start_time:
            params['startTime'] = start_time
        
        # CCXT를 통한 호출
        income_history = exchange.fapiprivate_get_income(params)
        
        if not income_history:
            return None
        
        # 가장 최근 realized PnL
        latest_pnl = income_history[0]
        
        return {
            'realized_pnl': float(latest_pnl['income']),
            'commission': abs(float(latest_pnl.get('commission', 0))),
            'trade_time': int(latest_pnl['time'])
        }
        
    except Exception as e:
        # API 호출 실패시 조용히 None 반환
        return None

def get_position_history_pnl(symbol: str, side: str) -> Optional[float]:
    """
    🆕 바이낸스 거래 내역에서 실제 PnL 조회 (대체 방법)
    
    Args:
        symbol: 심볼 (예: "ZEC/USDT:USDT")
        side: 'LONG' 또는 'SHORT'
    
    Returns:
        실제 realized PnL (float) 또는 None
    """
    try:
        # 최근 거래 내역 조회
        trades = exchange.fetch_my_trades(symbol, limit=50)
        
        if not trades:
            return None
        
        # 최근 청산 거래 찾기 (reduceOnly=True)
        for trade in reversed(trades):
            trade_info = trade.get('info', {})
            
            # reduceOnly 거래 = 포지션 청산 거래
            if trade_info.get('reduceOnly') or trade_info.get('positionSide') == 'BOTH':
                # realizedPnl 필드 확인
                realized_pnl = float(trade_info.get('realizedPnl', 0))
                if realized_pnl != 0:
                    return realized_pnl
        
        return None
        
    except Exception as e:
        return None

def get_top_volume_coins(limit: int = 10) -> List[dict]:
    """거래량 기준 상위 코인 조회"""
    try:
        tickers = exchange.fetch_tickers()
        
        # USDT 선물 필터링
        futures_tickers = {}
        for symbol, ticker in tickers.items():
            if ':USDT' in symbol and '/USDT' in symbol:
                base_coin = symbol.split('/')[0]
                # 메이저 코인만 (3글자 이상)
                if len(base_coin) >= 3 and base_coin not in ['1000', 'USDC']:
                    futures_tickers[symbol] = ticker
        
        # 거래량 기준 정렬
        sorted_tickers = sorted(
            futures_tickers.items(),
            key=lambda x: float(x[1].get('quoteVolume', 0)),
            reverse=True
        )
        
        result = []
        for symbol, ticker in sorted_tickers[:limit]:
            coin = symbol.split('/')[0]
            result.append({
                'coin': coin,
                'symbol': symbol,
                'price': float(ticker['last']),
                'volume': float(ticker['quoteVolume']),
                'change': float(ticker['percentage'] or 0)
            })
        
        return result
        
    except Exception as e:
        print(f"❌ 코인 목록 조회 실패: {e}")
        return []

def fetch_futures_market_indicators(symbol: str) -> Dict:
    """선물 특화 지표: 펀딩비 + OI"""
    indicators = {
        'funding_rate': None,
        'funding_rate_status': 'neutral',
        'open_interest': None,
        'oi_status': 'neutral',
    }
    
    try:
        # 1. 펀딩 비율 (Funding Rate)
        funding_info = exchange.fetch_funding_rate(symbol)
        if funding_info and 'fundingRate' in funding_info:
            fr = float(funding_info['fundingRate']) * 100  # 퍼센트로 변환
            indicators['funding_rate'] = fr
            
            # 펀딩비 상태 분류
            if fr > 0.05:
                indicators['funding_rate_status'] = 'long_overheated'  # 롱 과열
            elif fr > 0.01:
                indicators['funding_rate_status'] = 'bullish'  # 롱 우세
            elif fr < -0.05:
                indicators['funding_rate_status'] = 'short_overheated'  # 숏 과열
            elif fr < -0.01:
                indicators['funding_rate_status'] = 'bearish'  # 숏 우세
            else:
                indicators['funding_rate_status'] = 'neutral'  # 중립
    except Exception as e:
        print(f"   ⚠️ 펀딩비 조회 실패: {e}")
    
    try:
        # 2. 미결제 약정 (Open Interest)
        oi_data = exchange.fetch_open_interest(symbol)
        if oi_data and 'openInterestAmount' in oi_data:
            oi = float(oi_data['openInterestAmount'])
            indicators['open_interest'] = oi
            
            # OI 상태 분류 (간단한 기준)
            if oi > 100000000:  # 1억 이상
                indicators['oi_status'] = 'very_high'
            elif oi > 50000000:  # 5천만 이상
                indicators['oi_status'] = 'high'
            elif oi > 10000000:  # 1천만 이상
                indicators['oi_status'] = 'medium'
            else:
                indicators['oi_status'] = 'low'
    except Exception as e:
        print(f"   ⚠️ OI 조회 실패: {e}")
    
    return indicators

def fetch_comprehensive_market_data(symbol: str) -> Dict:
    """포괄적 시장 데이터 수집 (기술적 지표 + 선물 특화 지표)"""
    try:
        # 멀티 타임프레임 OHLCV (7개)
        timeframes = ['1m', '3m', '5m', '15m', '1h', '1d', '1w']
        ohlcv_data = {}
        
        for tf in timeframes:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 기술적 지표 추가 (안전한 방식)
                close_series = df['close']
                high_series = df['high']
                low_series = df['low']
                
                # 1. RSI
                try:
                    rsi_result = ta.rsi(close_series, length=14)
                    df['rsi'] = rsi_result if rsi_result is not None else 50.0
                except:
                    df['rsi'] = 50.0
                
                # 2. 볼린저밴드
                try:
                    bb_result = ta.bbands(close_series, length=20, std=2)
                    if bb_result is not None and isinstance(bb_result, pd.DataFrame):
                        # pandas_ta bbands 반환: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
                        cols = bb_result.columns.tolist()
                        if len(cols) >= 3:
                            df['bb_lower'] = bb_result.iloc[:, 0]  # BBL
                            df['bb_middle'] = bb_result.iloc[:, 1]  # BBM
                            df['bb_upper'] = bb_result.iloc[:, 2]  # BBU
                        else:
                            df['bb_lower'] = close_series
                            df['bb_middle'] = close_series
                            df['bb_upper'] = close_series
                    else:
                        df['bb_lower'] = close_series
                        df['bb_middle'] = close_series
                        df['bb_upper'] = close_series
                except Exception as e:
                    df['bb_lower'] = close_series
                    df['bb_middle'] = close_series
                    df['bb_upper'] = close_series
                
                # 3. MACD
                try:
                    macd_result = ta.macd(close_series, fast=12, slow=26, signal=9)
                    if macd_result is not None and isinstance(macd_result, pd.DataFrame):
                        cols = macd_result.columns.tolist()
                        if len(cols) >= 3:
                            df['macd'] = macd_result.iloc[:, 0]
                            df['macd_signal'] = macd_result.iloc[:, 1]
                            df['macd_hist'] = macd_result.iloc[:, 2]
                        else:
                            df['macd'] = 0.0
                            df['macd_signal'] = 0.0
                            df['macd_hist'] = 0.0
                    else:
                        df['macd'] = 0.0
                        df['macd_signal'] = 0.0
                        df['macd_hist'] = 0.0
                except:
                    df['macd'] = 0.0
                    df['macd_signal'] = 0.0
                    df['macd_hist'] = 0.0
                
                # 4. EMA
                try:
                    ema_20 = ta.ema(close_series, length=20)
                    df['ema_20'] = ema_20 if ema_20 is not None else close_series
                except:
                    df['ema_20'] = close_series
                
                try:
                    ema_50 = ta.ema(close_series, length=50)
                    df['ema_50'] = ema_50 if ema_50 is not None else close_series
                except:
                    df['ema_50'] = close_series
                
                # 5. ATR
                try:
                    atr_result = ta.atr(high_series, low_series, close_series, length=14)
                    df['atr'] = atr_result if atr_result is not None else (high_series - low_series).mean()
                except:
                    df['atr'] = (high_series - low_series).mean()
                
                ohlcv_data[tf] = df
                
            except Exception as e:
                print(f"   ⚠️ {tf} 타임프레임 수집 실패: {e}")
                continue
        
        # 현재 가격
        ticker = exchange.fetch_ticker(symbol)
        
        # 24h 통계
        stats_24h = {
            'price': float(ticker['last']),
            'change_24h': float(ticker['percentage'] or 0),
            'volume_24h': float(ticker['quoteVolume']),
            'high_24h': float(ticker['high']),
            'low_24h': float(ticker['low'])
        }
        
        # 🆕 선물 특화 지표 추가
        futures_indicators = fetch_futures_market_indicators(symbol)
        
        return {
            'ohlcv': ohlcv_data,
            'current': stats_24h,
            'futures_indicators': futures_indicators,
            'symbol': symbol
        }
        
    except Exception as e:
        print(f"❌ 시장 데이터 수집 실패: {e}")
        return {}

def ai_comprehensive_analysis(coin_data: dict, market_data: dict, performance_history: dict) -> dict:
    """AI 종합 분석 (모든 타임프레임 + 선물 특화 지표)"""
    global last_api_call_time
    
    try:
        provider = AI_MODEL_CONFIG["provider"].lower()
        delay = AI_MODEL_CONFIG["rate_limit"][provider]["delay_between_requests"]
        
        # Rate limiting
        time_since_last = time.time() - last_api_call_time
        if time_since_last < delay:
            time.sleep(delay - time_since_last)
        
        # 시장 데이터 요약
        current = market_data.get('current', {})
        ohlcv = market_data.get('ohlcv', {})
        futures_indicators = market_data.get('futures_indicators', {})
        
        # 🆕 모든 타임프레임의 기술적 지표 정리 (7개)
        timeframe_analysis = []
        for tf in ['1m', '3m', '5m', '15m', '1h', '1d', '1w']:
            tf_data = ohlcv.get(tf)
            if tf_data is not None and len(tf_data) > 0:
                latest = tf_data.iloc[-1]
                
                # RSI (안전하게 접근)
                rsi = float(latest.get('rsi', 50)) if 'rsi' in latest and not pd.isna(latest['rsi']) else 50.0
                
                # 볼린저밴드 위치 (안전하게 접근)
                bb_position = 50.0
                if 'bb_upper' in latest and 'bb_lower' in latest and 'close' in latest:
                    if not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']):
                        bb_range = latest['bb_upper'] - latest['bb_lower']
                        if bb_range > 0:
                            bb_position = (latest['close'] - latest['bb_lower']) / bb_range * 100
                
                # EMA 추세 (안전하게 접근)
                ema_trend = "중립"
                if 'ema_20' in latest and 'close' in latest and not pd.isna(latest['ema_20']):
                    ema_trend = "상승" if latest['close'] > latest['ema_20'] else "하락"
                
                # MACD (안전하게 접근)
                macd_signal = "중립"
                if 'macd' in latest and 'macd_signal' in latest:
                    if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                        macd_signal = "매수" if latest['macd'] > latest['macd_signal'] else "매도"
                
                # ATR (변동성) (안전하게 접근)
                atr = 0.0
                atr_pct = 0.0
                if 'atr' in latest and 'close' in latest and not pd.isna(latest['atr']):
                    atr = float(latest['atr'])
                    if latest['close'] > 0:
                        atr_pct = (atr / latest['close']) * 100
                
                timeframe_analysis.append(f"""
  [{tf}] RSI: {rsi:.1f} | BB위치: {bb_position:.0f}% | EMA추세: {ema_trend} | MACD: {macd_signal} | 변동성: {atr_pct:.2f}%""")
        
        # 🆕 선물 특화 지표 텍스트
        futures_text = "\n【선물 특화 지표 - 매우 중요】\n"
        futures_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 펀딩 비율
        if futures_indicators.get('funding_rate') is not None:
            fr = futures_indicators['funding_rate']
            fr_status = futures_indicators['funding_rate_status']
            
            fr_emoji = "🔥" if abs(fr) > 0.05 else "⚠️" if abs(fr) > 0.01 else "✅"
            futures_text += f"{fr_emoji} 펀딩 비율 (Funding Rate): {fr:+.4f}% ({fr_status})\n"
            
            if fr_status == 'long_overheated':
                futures_text += "   💡 롱 과열! 펀딩비 매우 높음 → SHORT 기회 or 롱 진입 회피\n"
            elif fr_status == 'short_overheated':
                futures_text += "   💡 숏 과열! 펀딩비 매우 낮음 → LONG 기회 or 숏 진입 회피\n"
            elif fr_status == 'bullish':
                futures_text += "   💡 롱 우세 → 상승 추세 가능성, 숏 신중\n"
            elif fr_status == 'bearish':
                futures_text += "   💡 숏 우세 → 하락 추세 가능성, 롱 신중\n"
            else:
                futures_text += "   💡 중립적 펀딩비 → 방향성 판단 어려움\n"
        
        # 미결제 약정
        if futures_indicators.get('open_interest') is not None:
            oi = futures_indicators['open_interest']
            oi_status = futures_indicators['oi_status']
            
            futures_text += f"\n📊 미결제 약정 (Open Interest): {oi:,.0f} ({oi_status})\n"
            futures_text += "   💡 OI 증가 + 가격 상승 = 강한 상승 추세\n"
            futures_text += "   💡 OI 증가 + 가격 하락 = 강한 하락 추세\n"
            futures_text += "   💡 OI 감소 = 포지션 청산 중, 추세 약화\n"
        
        futures_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 프롬프트 작성
        prompt = f"""
다음 암호화폐의 선물 거래를 양방향(LONG/SHORT) 분석해주세요:

**코인 정보:**
- 심볼: {coin_data['coin']}
- 현재가: ${current.get('price', 0):,.4f}
- 24h 변동: {current.get('change_24h', 0):+.2f}%
- 24h 거래량: ${current.get('volume_24h', 0):,.0f}

**멀티 타임프레임 기술적 분석:**
{''.join(timeframe_analysis)}

{futures_text}

**AI 과거 성과:**
- 최근 {performance_history.get('days', 0)}일 거래: {performance_history.get('total_trades', 0)}회
- 승률: {performance_history.get('win_rate', 0):.1f}%
- 총 수익: ${performance_history.get('total_pnl', 0):+,.2f}

**거래 조건:**
- 레버리지: 10-15x
- 리스크/리워드: 최소 1:2
- 트레일링 스탑 활용 (7% 최소 확보)

**분석 방향:**
1. 추세 순응 거래 (With the Trend):
   - 단기 시그널(1m~15m)이 장기 추세(1d, 1w)와 일치
   - 높은 확신도, 더 큰 익절 목표 (ATR x 3.0+)
   - 예: 주봉 상승추세 + 3분봉 매수신호 = 강한 LONG

2. 역추세 거래 (Counter-Trend):
   - 단기 시그널이 장기 추세와 반대
   - 단기 조정 노림, 빠른 익절 (ATR x 1.5~2.0)
   - 예: 주봉 상승추세 + 3분봉 매도신호 = 짧은 SHORT

3. LONG 기회:
   - 과매도 반등 (RSI < 30)
   - 지지선 테스트 후 반등
   - 볼린저밴드 하단 근처 + 매수 시그널
   - 🔥 펀딩비 숏 과열 (<-0.05%)
   - 주봉/일봉 상승추세 중 단기 되돌림

4. SHORT 기회:
   - 급등(+50% 이상) 후 모멘텀 약화
   - RSI 70 이상 과매수 + 볼린저밴드 상단 근처
   - 대량 거래량 후 매수세 소진 징후
   - 저항선에서 거부 패턴
   - 🔥 펀딩비 롱 과열 (>0.05%)
   - 주봉/일봉 하락추세 중 단기 반등

다음 형식으로 JSON 응답해주세요:
{{
    "trade": true/false,
    "direction": "LONG" 또는 "SHORT",
    "confidence": 0-100,
    "leverage": 10-15,
    "reasoning": "멀티 타임프레임과 선물 특화 지표(펀딩비, OI)를 모두 고려한 상세 분석"
}}

주의사항:
1. 70% 이상 확신할 때만 거래 추천
2. LONG과 SHORT 양방향 모두 검토 (편향 금지)
3. 🔥 펀딩비 과열 상황 반드시 고려
4. OI 증가/감소 추세를 가격 변동과 함께 분석
5. ⭐ 주봉/일봉 장기 추세와 단기 시그널(1m~15m) 방향성 확인
6. 추세 순응 거래 > 역추세 거래 (확신도 차이)
7. 여러 타임프레임의 시그널이 일치할수록 신뢰도 높음
8. 급등/급락 후 역추세 기회 적극 평가
9. 불분명한 상황에서는 관망 선택
10. 리스크 관리 우선, 확실한 기회만 공략
"""
        
        # AI 호출
        if provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel(AI_MODEL_CONFIG["models"]["gemini"])
            response = model.generate_content(prompt)
            ai_response = response.text
        elif provider == "deepseek":
            response = ai_client.chat.completions.create(
                model=AI_MODEL_CONFIG["models"]["deepseek"],
                messages=[{"role": "user", "content": prompt}]
            )
            ai_response = response.choices[0].message.content
        
        last_api_call_time = time.time()
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{[^{}]*\}', ai_response)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {"trade": False, "confidence": 0, "reasoning": "AI 응답 파싱 실패"}
            
    except Exception as e:
        print(f"❌ AI 분석 오류: {e}")
        return {"trade": False, "confidence": 0, "reasoning": f"AI 오류: {e}"}

# ===== 데이터베이스 함수 =====

def init_database():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            coin_symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            leverage INTEGER NOT NULL,
            position_size REAL NOT NULL,
            ai_confidence INTEGER NOT NULL,
            ai_reasoning TEXT NOT NULL,
            stop_loss_price REAL,
            take_profit_price REAL,
            exit_price REAL,
            pnl REAL,
            pnl_percent REAL,
            exit_reason TEXT,
            status TEXT DEFAULT 'OPEN'
        )
    ''')
    
    # 🆕 기존 테이블에 exit_reason 컬럼이 없으면 추가
    try:
        c.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
        print("✅ exit_reason 컬럼 추가됨")
    except sqlite3.OperationalError:
        # 이미 존재하면 무시
        pass
    
    # 기존 manual_trade 컬럼이 있으면 모두 FALSE로 설정
    try:
        c.execute("UPDATE trades SET manual_trade = FALSE WHERE manual_trade = TRUE")
        if c.rowcount > 0:
            print(f"✅ {c.rowcount}개 거래의 수동거래 보호 해제됨")
    except sqlite3.OperationalError:
        # manual_trade 컬럼이 없으면 무시
        pass
    
    conn.commit()
    conn.close()

def record_trade(coin_symbol: str, side: str, entry_price: float, quantity: float, 
                leverage: int, position_size: float, ai_confidence: int, ai_reasoning: str,
                stop_loss_price: float = 0, take_profit_price: float = 0) -> int:
    """거래 기록"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    timestamp = get_utc_now().isoformat()
    
    c.execute('''
        INSERT INTO trades (
            timestamp, coin_symbol, side, entry_price, quantity, leverage,
            position_size, ai_confidence, ai_reasoning, stop_loss_price,
            take_profit_price, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (timestamp, coin_symbol, side, entry_price, quantity, leverage,
          position_size, ai_confidence, ai_reasoning, stop_loss_price,
          take_profit_price))
    
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id

def update_trade_exit(trade_id: int, exit_price: float, pnl: float, 
                     pnl_percent: float, reason: str):
    """거래 청산 업데이트"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        UPDATE trades SET 
            exit_price = ?, pnl = ?, pnl_percent = ?, 
            exit_reason = ?, status = 'CLOSED'
        WHERE id = ?
    ''', (exit_price, pnl, pnl_percent, reason, trade_id))
    
    conn.commit()
    conn.close()

def get_all_open_trades() -> List[dict]:
    """모든 오픈 거래 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, timestamp, coin_symbol, side, entry_price, quantity, 
               leverage, position_size, ai_confidence, ai_reasoning,
               stop_loss_price, take_profit_price
        FROM trades WHERE status = 'OPEN'
        ORDER BY timestamp DESC
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    trades = []
    for row in rows:
        trades.append({
            'id': row[0],
            'timestamp': row[1],
            'coin_symbol': row[2],
            'side': row[3],
            'entry_price': row[4],
            'quantity': row[5],
            'leverage': row[6],
            'position_size': row[7],
            'ai_confidence': row[8],
            'ai_reasoning': row[9],
            'stop_loss_price': row[10],
            'take_profit_price': row[11]
        })
    
    return trades

def update_closed_trades_pnl_from_history():
    """
    기존 청산 거래들의 PnL을 Position History에서 가져와 업데이트
    """
    print(f"\n{'='*80}")
    print(f"🔧 청산 거래 PnL 정정 (Position History 기반)")
    print(f"{'='*80}")
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 최근 7일간 청산된 거래 조회
    since_date = (get_utc_now() - timedelta(days=7)).isoformat()
    c.execute('''
        SELECT id, coin_symbol, side, entry_price, position_size, leverage, timestamp
        FROM trades 
        WHERE status = 'CLOSED' AND timestamp >= ?
        ORDER BY timestamp DESC
    ''', (since_date,))
    
    closed_trades = c.fetchall()
    conn.close()
    
    if not closed_trades:
        print("   📊 업데이트할 청산 거래가 없습니다")
        print(f"{'='*80}\n")
        return
    
    print(f"   📊 {len(closed_trades)}개 청산 거래 PnL 정정 시작...")
    
    updated_count = 0
    
    for trade in closed_trades:
        trade_id, coin, side, entry_price, position_size, leverage, timestamp = trade
        symbol = f"{coin}USDT"
        
        try:
            print(f"\n   🔍 {coin} (ID: {trade_id}) PnL 정정 중...")
            
            # 거래 시간 기준으로 Position History 조회 범위 설정
            trade_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            start_time = int((trade_time - timedelta(hours=1)).timestamp() * 1000)
            end_time = int((trade_time + timedelta(hours=6)).timestamp() * 1000)
            
            # 바이낸스 계정 거래 내역에서 해당 심볼의 최근 거래 조회
            try:
                # fetchMyTrades로 해당 심볼의 최근 거래 내역 조회
                trades = exchange.fetchMyTrades(f"{coin}/USDT:USDT", since=start_time, limit=50)
                
                # 해당 거래 시간대와 매칭되는 거래들 찾기
                realized_pnl = None
                total_pnl = 0
                trade_count = 0
                
                for trade_record in trades:
                    trade_timestamp = trade_record['timestamp']
                    trade_amount = trade_record['amount']
                    trade_price = trade_record['price']
                    trade_side = trade_record['side']
                    trade_fee = trade_record['fee']['cost'] if trade_record['fee'] else 0
                    
                    # 거래 시간이 우리 거래 시간 근처인지 확인 (6시간 이내)
                    time_diff = abs(trade_timestamp - int(trade_time.timestamp() * 1000))
                    if time_diff < 6 * 60 * 60 * 1000:  # 6시간 이내
                        # 간단한 PnL 추정 (정확하지 않지만 근사값)
                        if trade_side == 'sell' and side.upper() == 'LONG':
                            pnl_estimate = (trade_price - entry_price) * trade_amount - trade_fee
                        elif trade_side == 'buy' and side.upper() == 'SHORT':
                            pnl_estimate = (entry_price - trade_price) * trade_amount - trade_fee
                        else:
                            pnl_estimate = 0
                        
                        total_pnl += pnl_estimate * leverage
                        trade_count += 1
                
                if trade_count > 0:
                    realized_pnl = total_pnl
                    pnl_pct = (realized_pnl / position_size) * 100 if position_size > 0 else 0
                    print(f"      ✅ 거래 내역 기반 PnL: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                else:
                    print(f"      ⚠️ 매칭되는 거래 내역 없음 - 건너뜀")
                    continue
                    
            except Exception as trades_error:
                print(f"      ❌ 거래 내역 조회 실패: {trades_error}")
                print(f"      ⚠️ 해당 거래 건너뜀")
                continue
            if realized_pnl is not None:
                # 수익률 계산
                pnl_pct = (realized_pnl / position_size) * 100 if position_size > 0 else 0
                
                # DB 업데이트
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                
                c.execute('''
                    UPDATE trades 
                    SET pnl = ?, pnl_percent = ?
                    WHERE id = ?
                ''', (realized_pnl, pnl_pct, trade_id))
                
                conn.commit()
                conn.close()
                
                print(f"      ✅ PnL 정정 완료: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                updated_count += 1
            else:
                print(f"      ⚠️ PnL 계산 불가")
                
        except Exception as e:
            print(f"      ❌ PnL 정정 실패: {e}")
    
    print(f"\n   📊 정정 완료: {updated_count}/{len(closed_trades)}개 거래")
    print(f"   ✅ AI 피드백이 이제 정확한 손익 기반으로 동작합니다")
    print(f"{'='*80}\n")

def get_recent_performance(days: int = 7) -> dict:
    """최근 성과 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    since_date = (get_utc_now() - timedelta(days=days)).isoformat()
    
    c.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as total_pnl,
               AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
               AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
        FROM trades 
        WHERE status = 'CLOSED' AND timestamp >= ?
    ''', (since_date,))
    
    row = c.fetchone()
    conn.close()
    
    total_trades = row[0] or 0
    wins = row[1] or 0
    total_pnl = row[2] or 0
    avg_win = row[3] or 0
    avg_loss = row[4] or 0
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'days': days,
        'total_trades': total_trades,
        'wins': wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

# ===== 실거래 실행 =====

def execute_live_trade(coin_data: dict, ai_decision: dict, available_balance: float) -> bool:
    """실제 거래 실행"""
    try:
        coin = coin_data['coin']
        symbol = f"{coin}/USDT:USDT"
        side = ai_decision['direction']
        confidence = ai_decision['confidence']
        leverage = ai_decision.get('leverage', 10)
        
        # 🔧 개선된 동적 균등 분할 (목표: 전체 80% 사용)
        max_positions = LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']
        target_total_usage = 80  # 전체 자금의 80% 목표
        base_position_pct = target_total_usage / max_positions  # 16%씩 분할 (5개 = 80%)
        
        # 변동성 기반 조정 (완화)
        volatility = abs(coin_data.get('change', 0))
        if volatility > LIVE_TRADING_CONFIG.get('HIGH_VOLATILITY_THRESHOLD', 5.0):
            volatility_multiplier = 0.9  # 고변동성: 10% 감소
        else:
            volatility_multiplier = 1.1  # 저변동성: 10% 증가
        
        # 신뢰도 기반 조정 (완화: 70-95% → 85-115%)
        # 70% confidence → 0.85배
        # 95% confidence → 1.15배
        confidence_multiplier = 0.85 + (confidence - 70) / 25 * 0.3
        confidence_multiplier = max(0.85, min(1.15, confidence_multiplier))
        
        # 최종 포지션 크기
        position_pct = base_position_pct * volatility_multiplier * confidence_multiplier
        
        # 안전 범위 체크
        min_pct = LIVE_TRADING_CONFIG.get('MIN_POSITION_SIZE_PCT', 10)  # 최소 10%
        max_pct = LIVE_TRADING_CONFIG.get('MAX_POSITION_SIZE_PCT', 25)  # 최대 25%
        position_pct = max(min_pct, min(max_pct, position_pct))
        
        position_size = available_balance * (position_pct / 100)
        
        print(f"   💰 포지션 크기 계산:")
        print(f"      - 목표 총 사용률: {target_total_usage}%")
        print(f"      - 기본: {base_position_pct:.1f}%")
        print(f"      - 변동성 조정: ×{volatility_multiplier:.2f} ({volatility:.1f}%)")
        print(f"      - 신뢰도 조정: ×{confidence_multiplier:.2f} ({confidence}%)")
        print(f"      - 최종: {position_pct:.1f}% = ${position_size:,.2f}")
        
        # 최소 포지션 크기 체크
        if position_size < 50:
            print(f"   ❌ 포지션 크기 부족: ${position_size:.2f} < $50")
            return False
        
        # 레버리지 설정
        exchange.set_leverage(leverage, symbol)
        print(f"   ⚙️ 레버리지 설정: {leverage}x")
        
        # 마진 모드 설정
        margin_mode = LIVE_TRADING_CONFIG['MARGIN_MODE']
        if margin_mode == 'isolated':
            exchange.set_margin_mode('isolated', symbol)
            print(f"   ⚙️ 마진 모드: Isolated")
        
        # 주문 실행
        current_price = coin_data['price']
        quantity = position_size * leverage / current_price
        
        if side.upper() == 'LONG':
            order = exchange.create_market_buy_order(symbol, quantity)
        else:
            order = exchange.create_market_sell_order(symbol, quantity)
        
        print(f"   ✅ 주문 체결:")
        print(f"      - 심볼: {symbol}")
        print(f"      - 방향: {side.upper()}")
        print(f"      - 수량: {quantity:.6f}")
        print(f"      - 진입가: ${current_price:,.4f}")
        print(f"      - 주문 ID: {order.get('id', 'N/A')}")
        
        # DB 기록
        trade_id = record_trade(
            coin_symbol=coin,
            side=side.upper(),
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            position_size=position_size,
            ai_confidence=confidence,
            ai_reasoning=ai_decision.get('reasoning', 'AI 추천')
        )
        
        print(f"   📝 DB 기록: 거래 ID {trade_id}")
        
        # 🆕 트레일링 스탑 설정 (진입 즉시)
        if LIVE_TRADING_CONFIG.get("TRAILING_STOP_ENABLED", False):
            print(f"   🎯 트레일링 스탑 설정 중...")
            time.sleep(2)  # 포지션 확립 대기
            
            trailing_order = set_trailing_stop_order(symbol, side.upper(), leverage, position_size)
            if trailing_order:
                print(f"   ✅트레일링 스탑 활성화")
            else:
                print(f"   ⚠️ 트레일링 스탑 설정 실패")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 거래 실행 실패: {e}")
        return False

# ===== AI 포지션 관리 =====

def manage_live_positions():
    """
    🔧 AI 포지션 모니터링 (실제 PnL 조회 추가)
    
    변경사항:
    - 청산 시 즉시 바이낸스 실제 PnL 조회
    - close_position_and_get_pnl() 사용
    - 트레일링 스탑 활용 및 추세 반전 감지
    """
    print(f"\n{'='*80}")
    print(f"📊 AI 포지션 모니터링 (트레일링 스탑 관리)")
    print(f"{'='*80}")
    
    # 오픈 거래 조회
    open_trades = get_all_open_trades()
    if not open_trades:
        print(f"   📊 관리할 포지션 없음")
        print(f"{'='*80}\n")
        return
    
    # 바이낸스 실제 포지션
    live_positions = get_open_positions()
    live_position_map = {pos['symbol']: pos for pos in live_positions}
    
    for trade in open_trades:
        trade_id = trade['id']
        coin = trade['coin_symbol']
        side = trade['side']
        entry_price = trade['entry_price']
        symbol = f"{coin}/USDT:USDT"
        
        # 바이낸스 포지션 확인
        if symbol not in live_position_map:
            print(f"   ⚠️ {coin} (ID: {trade_id}): 바이낸스에 포지션 없음 - 이미 청산됨")
            continue
        
        # 현재가 조회
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            unrealized_pnl = float(live_position_map[symbol]['unrealizedPnl'])
        except Exception as e:
            print(f"   ❌ {coin} 가격 조회 실패: {e}")
            continue
        
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
        
        # 트레일링 스탑 설정 확인
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            has_trailing_stop = any(
                'trailing' in str(order.get('type', '')).lower() or 
                'trailingPercent' in str(order.get('info', {})) 
                for order in open_orders
            )
        except:
            has_trailing_stop = False
        
        print(f"\n   📊 {coin} (ID: {trade_id}, {side.upper()}):")
        print(f"      - 진입가: ${entry_price:,.4f}")
        print(f"      - 현재가: ${current_price:,.4f}")
        print(f"      - 수익률: {pnl_pct:+.2f}%")
        print(f"      - 미실현 손익: ${unrealized_pnl:+,.2f}")
        print(f"      - 변동성: {volatility:.1f}%")
        print(f"      - 트레일링 스탑: {'✅ 설정됨' if has_trailing_stop else '❌ 없음'}")
        
        # 🆕 상태별 메시지 표시
        if pnl_pct >= 15:
            print(f"      🎯 익절 영역 (+{pnl_pct:.2f}%) - 트레일링 스탑이 수익 보호 중")
        elif pnl_pct >= 5:
            print(f"      📈 수익 구간 (+{pnl_pct:.2f}%) - 트레일링 스탑이 추적 중")
        elif pnl_pct >= -5:
            print(f"      ⚖️ 균형 구간 ({pnl_pct:+.2f}%) - 정상 범위")
        else:
            print(f"      ⚠️ 손실 구간 ({pnl_pct:+.2f}%) - 트레일링 스탑이 손절 대기 중")
        
        # 🆕 추세 반전 감지 (수익 중일 때만, +10% 이상)
        should_check_reversal = pnl_pct >= 10 and has_trailing_stop
        
        if should_check_reversal:
            print(f"      🔍 추세 반전 감지 중...")
            is_reversed, reversal_reason, signal_strength = detect_trend_reversal(
                symbol, side, current_price, entry_price
            )
            
            if signal_strength > 0:
                print(f"      📊 추세 분석: {reversal_reason}")
                print(f"      💪 신호 강도: {signal_strength}/10")
            
            # 명확한 추세 반전 감지시 청산
            if is_reversed:
                print(f"      🚨 명확한 추세 반전 감지! (강도: {signal_strength}/10)")
                print(f"      🔄 수익 보호 청산 시도...")
                print(f"      📝 반전 신호: {reversal_reason}")
                
                # 🆕 즉시 청산 + 실제 PnL 조회
                success, exit_price, realized_pnl, pnl_percentage = close_position_and_get_pnl(
                    symbol, side, trade
                )
                
                if success:
                    print(f"      ✅ 추세 반전 청산 완료")
                    print(f"      📊 최종 PnL: ${realized_pnl:+.2f} ({pnl_percentage:+.2f}%)")
                    
                    # DB 업데이트
                    update_trade_exit(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl=realized_pnl,
                        pnl_percent=pnl_percentage,
                        reason=f"AI 추세 반전 감지 청산 (강도 {signal_strength}/10): {reversal_reason}"
                    )
                    continue  # 다음 포지션으로
                else:
                    print(f"      ❌ 추세 반전 청산 실패 - 트레일링 스탑에 맡김")
            else:
                if signal_strength > 3:
                    print(f"      ✅ 추세 유지 (약한 반전 신호: {signal_strength}/10)")
                else:
                    print(f"      ✅ 추세 유지")
        
        # 🚨 응급상황 체크 (극단적인 경우에만 AI 개입)
        emergency_threshold = -30  # 30% 이상 손실시 응급상황
        if pnl_pct <= emergency_threshold and not has_trailing_stop:
            print(f"      🚨 응급상황: 트레일링 스탑 없이 {pnl_pct:+.2f}% 손실!")
            print(f"      🔄 응급 청산 시도...")
            
            # 🆕 즉시 청산 + 실제 PnL 조회
            success, exit_price, realized_pnl, pnl_percentage = close_position_and_get_pnl(
                symbol, side, trade
            )
            
            if success:
                print(f"      📊 응급청산 PnL: ${realized_pnl:+.2f} ({pnl_percentage:+.2f}%)")
                
                # DB 업데이트
                update_trade_exit(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    pnl_percent=pnl_percentage,
                    reason=f"AI 응급청산: 트레일링 스탑 없이 {pnl_pct:+.2f}% 손실"
                )
                print(f"      ✅ 응급 청산 완료")
            else:
                print(f"      ❌ 응급 청산 실패")
        elif not should_check_reversal:
            print(f"      ✅ 트레일링 스탑이 관리 중 - AI 개입 불필요")
    
    print(f"{'='*80}\n")

# ===== DB-바이낸스 동기화 =====

def sync_db_with_binance() -> int:
    """
    🔧 개선된 DB-바이낸스 동기화 (실제 realized PnL 사용)
    
    변경사항:
    1. 바이낸스 Income History에서 실제 realized PnL 조회
    2. 조회 실패시에만 가격 기반 계산 (수수료 반영)
    3. Position History를 통한 대체 조회 방법 추가
    
    로직:
    1. 바이낸스에 없는 DB 포지션 -> 청산 처리 (이미 청산됨)
    2. DB에 없는 바이낸스 포지션 -> DB에 추가 (수동거래로 등록)
    
    Returns:
        동기화된 거래 수
    """
    print(f"\n{'='*80}")
    print(f"🔄 DB-바이낸스 동기화 (실제 PnL 우선)")
    print(f"{'='*80}")
    
    synced_count = 0
    
    # DB 오픈 거래
    open_trades = get_all_open_trades()
    
    # 바이낸스 실제 포지션
    live_positions = get_open_positions()
    live_position_map = {pos['symbol']: pos for pos in live_positions}
    
    print(f"   📊 DB 오픈 거래: {len(open_trades)}개")
    print(f"   📊 바이낸스 포지션: {len(live_positions)}개")
    
    # 1. 바이낸스에 없는 DB 포지션 -> 청산 처리 (이미 청산됨)
    for trade in open_trades:
        symbol = f"{trade['coin_symbol']}/USDT:USDT"
        
        if symbol not in live_position_map:
            trade_id = trade['id']
            coin = trade['coin_symbol']
            side = trade['side']
            entry_price = trade['entry_price']
            entry_time = trade.get('entry_time', None)
            
            print(f"   ⚠️ {coin} (ID: {trade_id}): 바이낸스에 포지션 없음 - 청산 확인")
            
            try:
                # 🆕 방법 1: Income History에서 실제 realized PnL 조회
                start_time = None
                if entry_time:
                    entry_dt = parse_db_timestamp(entry_time)
                    if entry_dt:
                        start_time = int(entry_dt.timestamp() * 1000)
                
                pnl_data = get_realized_pnl_from_binance(symbol, start_time)
                
                if pnl_data and pnl_data['realized_pnl'] != 0:
                    # ✅ 실제 realized PnL 발견!
                    realized_pnl = pnl_data['realized_pnl']
                    commission = pnl_data['commission']
                    
                    # 수익률 계산
                    pnl_pct = (realized_pnl / trade['position_size']) * 100
                    
                    # 청산가 역산 (근사값)
                    if side.upper() == 'LONG':
                        price_move_pct = pnl_pct / trade['leverage']
                        exit_price_estimated = entry_price * (1 + price_move_pct / 100)
                    else:
                        price_move_pct = pnl_pct / trade['leverage']
                        exit_price_estimated = entry_price * (1 - price_move_pct / 100)
                    
                    print(f"      ✅ 바이낸스 실제 PnL: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                    print(f"      💰 수수료: ${commission:.4f}")
                    print(f"      📍 추정 청산가: ${exit_price_estimated:,.4f}")
                    
                else:
                    # 🆕 방법 2: Position History/My Trades에서 조회 시도
                    print(f"      🔍 거래 내역에서 조회 시도...")
                    history_pnl = get_position_history_pnl(symbol, side)
                    
                    if history_pnl is not None and history_pnl != 0:
                        realized_pnl = history_pnl
                        pnl_pct = (realized_pnl / trade['position_size']) * 100
                        
                        # 청산가 역산
                        if side.upper() == 'LONG':
                            price_move_pct = pnl_pct / trade['leverage']
                            exit_price_estimated = entry_price * (1 + price_move_pct / 100)
                        else:
                            price_move_pct = pnl_pct / trade['leverage']
                            exit_price_estimated = entry_price * (1 - price_move_pct / 100)
                        
                        print(f"      ✅ 거래 내역 PnL: ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                        print(f"      📍 추정 청산가: ${exit_price_estimated:,.4f}")
                    
                    else:
                        # 방법 3: 현재가 기반 계산 (수수료 반영)
                        print(f"      ⚠️ 실제 PnL 조회 실패 - 추정 계산 사용")
                        ticker = exchange.fetch_ticker(symbol)
                        exit_price_estimated = float(ticker['last'])
                        
                        # 🆕 수수료 반영 계산
                        realized_pnl, pnl_pct = calculate_pnl_from_price(
                            entry_price, exit_price_estimated, 
                            trade['position_size'], side, trade['leverage']
                        )
                        
                        print(f"      📊 추정 PnL (수수료 포함): ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                        print(f"      ⚠️ 주의: 실제 체결가와 다를 수 있음")
                
                # 🔧 안전한 DB 업데이트 (exit_reason 컬럼 확인)
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                
                # exit_reason 컬럼 존재 확인
                c.execute("PRAGMA table_info(trades)")
                columns = [column[1] for column in c.fetchall()]
                
                if 'exit_reason' in columns:
                    c.execute('''
                        UPDATE trades SET 
                            exit_price = ?, pnl = ?, pnl_percent = ?, 
                            exit_reason = ?, status = 'CLOSED'
                        WHERE id = ?
                    ''', (exit_price_estimated, realized_pnl, pnl_pct, 
                         "동기화: 트레일링 스탑 청산 (바이낸스 PnL)", trade_id))
                else:
                    c.execute('''
                        UPDATE trades SET 
                            exit_price = ?, pnl = ?, pnl_percent = ?, 
                            status = 'CLOSED'
                        WHERE id = ?
                    ''', (exit_price_estimated, realized_pnl, pnl_pct, trade_id))
                
                conn.commit()
                conn.close()
                
                print(f"      ✅ DB 업데이트 완료")
                synced_count += 1
                
            except Exception as e:
                print(f"      ❌ 동기화 실패: {e}")
                import traceback
                traceback.print_exc()
    
    # 2. 바이낸스에는 있는데 DB에 없는 포지션 -> DB에 추가 (수동거래)
    db_symbols = {f"{t['coin_symbol']}/USDT:USDT" for t in open_trades}
    
    for symbol, position in live_position_map.items():
        if symbol not in db_symbols:
            coin = symbol.split('/')[0]
            side = 'LONG' if position['side'] == 'long' else 'SHORT'
            entry_price = float(position['entryPrice'])
            quantity = abs(float(position['contracts']))
            
            # 레버리지 안전 처리
            raw_leverage = position.get('leverage')
            if raw_leverage is None:
                leverage = 1 
            else:
                leverage = int(raw_leverage)
                
            notional = float(position['notional'])
            
            print(f"   🆕 {coin}: 바이낸스 포지션 발견 (DB에 없음)")
            print(f"      - 방향: {side}")
            print(f"      - 진입가: ${entry_price:,.4f}")
            print(f"      - 수량: {quantity}")
            print(f"      - 레버리지: {leverage}x")
            print(f"      - 포지션 크기: ${notional:,.2f}")
            
            # DB에 거래로 등록
            try:
                trade_id = record_trade(
                    coin_symbol=coin,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    leverage=leverage,
                    position_size=notional / leverage,
                    ai_confidence=0,
                    ai_reasoning="동기화: 바이낸스에서 감지된 거래",
                    stop_loss_price=0,
                    take_profit_price=0
                )
                
                print(f"      ✅ DB에 거래로 등록됨 (ID: {trade_id})")
                synced_count += 1
                
            except Exception as e:
                print(f"      ❌ DB 등록 실패: {e}")
    
    print(f"\n   📊 동기화 결과: {synced_count}개 거래 처리")
    if synced_count > 0:
        print(f"   ✅ 바이낸스 포지션과 DB가 동기화되었습니다")
    else:
        print(f"   ℹ️ 동기화할 항목이 없습니다")
    print(f"{'='*80}\n")
    
    return synced_count

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
    performance = get_recent_performance(3)
    print(f"\n📊 최근 3일 성과:")
    print(f"   - 총 거래: {performance['total_trades']}회")
    print(f"   - 승률: {performance['win_rate']:.1f}%")
    print(f"   - 총 손익: ${performance['total_pnl']:+,.2f}")
    
    print(f"{'='*80}\n")

# ===== 메인 함수 =====

def main():
    """메인 거래 루프"""
    init_database()
    
    print(f"{'='*80}")
    print(f"🚀 AI 실거래 봇 시작")
    print(f"{'='*80}")
    
    # 잔고 확인
    balance = get_available_balance()
    min_balance = LIVE_TRADING_CONFIG['MIN_CAPITAL_THRESHOLD']
    
    if balance < min_balance:
        print(f"❌ 잔고 부족: ${balance:.2f} < ${min_balance:.2f}")
        print(f"   최소 ${min_balance:.2f} USDT가 필요합니다.")
        return
    
    print(f"   💰 Available Balance: ${balance:,.2f}")
    print(f"   🎯 최대 포지션: {LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개")
    print(f"   🤖 AI 신규 분석: {LIVE_TRADING_CONFIG['AI_ANALYSIS_INTERVAL']}초마다")
    print(f"   🔄 포지션 관리: {LIVE_TRADING_CONFIG['POSITION_CHECK_INTERVAL']}초마다")
    print(f"   📊 성과 리뷰: {LIVE_TRADING_CONFIG['PERFORMANCE_REVIEW_INTERVAL']}초마다")
    print(f"   📈 레버리지: {LIVE_TRADING_CONFIG['CONSERVATIVE_LEVERAGE']}-{LIVE_TRADING_CONFIG['MAX_LEVERAGE']}x")
    print(f"   🛡️ 마진 모드: {LIVE_TRADING_CONFIG['MARGIN_MODE'].upper()}")
    print(f"{'='*80}\n")
    
    # 🆕 트레일링 스탑 설정 출력
    if LIVE_TRADING_CONFIG.get("TRAILING_STOP_ENABLED", False):
        min_profit = LIVE_TRADING_CONFIG.get("TRAILING_STOP_MIN_PROFIT_PCT", 12.0)
        print(f"🎯 트레일링 스탑 (전체 관리):")
        print(f"   ✅ 활성화됨 (진입 즉시)")
        print(f"   ✅ 최소 확보 수익률: {min_profit}%")
        print(f"   ✅ 콜백 비율: 변동성에 따라 자동 조절")
        print(f"   📝 예시:")
        print(f"      - 초저변동성 (ATR<1.5%): 콜백 {min_profit/10*0.7:.1f}% (레버리지 10x)")
        print(f"      - 저변동성 (ATR<3%): 콜백 {min_profit/10*0.85:.1f}%")
        print(f"      - 중변동성 (ATR<5%): 콜백 {min_profit/10*1.0:.1f}%")
        print(f"      - 고변동성 (ATR<8%): 콜백 {min_profit/10*1.3:.1f}%")
        print(f"      - 초고변동성 (ATR>8%): 콜백 {min_profit/10*1.6:.1f}%")
        print(f"   🎯 익절과 손절을 모두 자동 처리")
        print(f"   🤖 AI는 모니터링만 수행 (응급상황시에만 개입)")
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
                            
                            # 수집된 타임프레임 개수
                            ohlcv_count = len(market_data.get('ohlcv', {}))
                            futures_ind = market_data.get('futures_indicators', {})
                            
                            print(f"   ✅ {ohlcv_count}개 타임프레임 데이터 수집 완료")
                            
                            # 🆕 선물 지표 간단히 표시
                            if futures_ind.get('funding_rate') is not None:
                                fr = futures_ind['funding_rate']
                                fr_status = futures_ind['funding_rate_status']
                                print(f"   🔥 펀딩비: {fr:+.4f}% ({fr_status})")
                            
                            if futures_ind.get('open_interest') is not None:
                                oi = futures_ind['open_interest']
                                oi_status = futures_ind['oi_status']
                                print(f"   📊 OI: {oi:,.0f} ({oi_status})")
                            
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
║        🔴 AI 실거래 트레이딩 봇 v2.3 (PnL 수정본)            ║
║        🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS          ║
║        ⚠️  실제 자금으로 거래합니다!                         ║
║        ✅ AI: {provider:20s} ({model_name:20s})    ║
║        ✅ Isolated Margin 모드                                ║
║        ✅ Risk/Reward ≥ 1:2 전략                             ║
║        🆕 트레일링 스탑 + 바이낸스 실제 PnL 조회              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  주의사항:")
    print("   1. 실제 바이낸스 계정의 자금을 사용합니다")
    print("   2. Isolated Margin 모드로 거래합니다")
    print("   3. 손실 가능성이 있으니 충분히 이해한 후 사용하세요")
    print("   4. 소액으로 테스트한 후 본격적으로 사용하세요")
    print(f"   5. 수수료: Maker 0.02%, Taker 0.05%")
    print(f"   6. 🆕 트레일링 스탑이 진입 즉시 활성화됩니다")
    print(f"   7. 🔧 AI는 손절에서만 개입, 익절은 트레일링 스탑에 맡김\n")
    
    print("\n🔧 v2.3 수정사항:")
    print("   1. ✅ 바이낸스 Income History API로 실제 realized PnL 조회")
    print("   2. ✅ 거래 내역(My Trades)에서 PnL 대체 조회")
    print("   3. ✅ 수수료 반영 PnL 계산 (Maker 0.02%, Taker 0.05%)")
    print("   4. ✅ 트레일링 스탑 청산시 정확한 손익 계산")
    print("   5. ✅ 손익 +인데 -로 표시되는 문제 해결\n")
    
    # 자동 시작 (확인 없음)
    print("🚀 백그라운드 모드 - 자동 시작...")
    main()
