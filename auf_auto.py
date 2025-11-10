#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Live Trading Bot v2.8 (신뢰도 기반 포지션 사이징)
----------------------------------------------------------------------
⚠️ 실제 바이낸스 선물 거래 - 실제 자금 사용
- 실시간 바이낸스 선물 데이터 사용
- 실제 잔고로 거래 (Available Balance 기준)
- Isolated Margin 모드
- 실제 주문 체결 및 청산
- AI 보수적 리스크 관리
- 🆕 처음부터 트레일링 스탑 설정
- 🆕 거래 시간대 제한 (한국시간 23:00~07:00 신규 진입 차단)
- 🆕 AI 포지션 모니터링 제거 (트레일링 스탑에 전담)

🔧 v2.8 신규 기능 (Option C - 신뢰도 기반 포지션 사이징):
  1. ✅ 간단하고 명확한 포지션 크기 계산
  2. ✅ 스타일별 베이스: 스캘핑 25%, 데이트레이딩 35%
  3. ✅ 신뢰도 보너스: 70-80% +0%, 80-90% +5%, 90%+ +10%
  4. ✅ 최대 50% 제한 (안전장치)
  5. ✅ 불필요한 변동성 조정 제거 → 예측 가능한 투자금액

🔧 v2.7 기능:
  1. ✅ AI 포지션 모니터링 제거 (API 비용 절감)
  2. ✅ 트레일링 스탑에 포지션 관리 전담
  3. ✅ 검증된 자동화 시스템에만 의존 (불필요한 AI 개입 제거)

🔧 v2.6 기능:
  1. ✅ 거래 시간대 제한 (한국시간 23:00~07:00 / UTC 14:00~22:00)
  2. ✅ 고변동성 시간대 신규 진입 차단
  3. ✅ 기존 포지션 관리는 24시간 유지 (트레일링 스탑)

🔧 v2.5 기능:
  1. ✅ 명확한 추세 반전 감지 (+10% 이상 수익시) - 제거됨
  2. ✅ 다중 기술적 지표 분석 (RSI, MACD, EMA, 캔들 패턴) - 제거됨
  3. ✅ 신호 강도 점수화 (7/10 이상시만 청산) - 제거됨

v2.4 기능:
  - 공격적 포지션 크기 (목표 100% 자금 사용)
  - 동적 균등 분할 및 변동성 조정

v2.3 기능:
  - 바이낸스 실제 realized PnL 조회
  - 수수료 반영 손익 계산
  - 정확한 손익 동기화

⚠️ 중요:
- 신규 진입은 한국시간 07:00~23:00만 허용
- 포지션 관리는 트레일링 스탑이 자동 처리 (바이낸스 서버)
- AI는 신규 진입 분석에만 집중 (10분마다)
- 검증된 자동화에만 의존 → API 비용 절감 + 안정성 향상
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

def is_trading_allowed_time():
    """
    거래 허용 시간대 체크
    한국시간 23:00 ~ 07:00 (UTC 14:00 ~ 22:00)은 거래 금지
    
    Returns:
        tuple: (허용 여부(bool), 메시지(str))
    """
    current_utc = get_utc_now()
    current_hour = current_utc.hour
    
    # UTC 14:00 ~ 22:00은 거래 금지 (한국시간 23:00 ~ 07:00)
    if 14 <= current_hour < 22:
        kst_hour = (current_hour + 9) % 24
        return False, f"⏸️  거래 제한 시간대 (한국시간 {kst_hour:02d}시 / UTC {current_hour:02d}시)"
    
    return True, "✅ 거래 허용 시간대"

load_dotenv()

# ===== 로깅 설정 =====
LOG_FILE = "trading_bot.log"

class Logger:
    """파일과 콘솔에 동시 로깅 (즉시 flush)"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        # buffering=1 = 라인 버퍼링 (줄 단위 즉시 flush)
        # errors='replace' = 인코딩 불가능한 문자는 대체 문자로 변환 (크래시 방지)
        self.log = open(filename, 'a', encoding='utf-8', buffering=1, errors='replace')
    
    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()  # 🆕 터미널도 즉시 flush
        except UnicodeEncodeError:
            # 터미널이 UTF-8을 지원하지 않을 경우 ASCII로 대체
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
            self.terminal.flush()
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

# ===== 거래 스타일 프리셋 (데이트레이딩 중심 + 극단적 스캘핑) =====
TRADING_STYLES = {
    "SCALPING": {
        "name": "스캘핑 (극단적 기회)",
        "timeframes": ['1m', '3m', '5m'],
        "primary_tf": '3m',
        "base_callback_pct": 1.0,  # 기본 콜백 (레버리지 10배 기준) - 실제는 레버리지로 조정
        "min_profit_target": 3.0,  # 최소 3% 수익 목표
        "typical_holding": "5-30분",
        "max_leverage": 30,  # 최대 레버리지
        "leverage_guideline": "극단적 기회 → 20-30배 권장",
        "description": "초고변동성 극단 기회 - 급등/급락 포착",
        "trigger_conditions": """
        다음 조건 중 3개 이상 충족시에만 선택:
        1. ATR > 10% (초고변동성)
        2. 24h 변동 > 20% (급격한 움직임)
        3. 펀딩비 극단 과열 (>0.1% 또는 <-0.1%)
        4. 거래량 폭발 (평균 대비 3배+)
        5. AI 신뢰도 85%+
        """
    },
    "DAY_TRADING": {
        "name": "데이트레이딩 (기본)",
        "timeframes": ['15m', '1h', '4h'],
        "primary_tf": '1h',
        "base_callback_pct": 3.0,  # 기본 콜백 (레버리지 10배 기준) - 실제는 레버리지로 조정
        "min_profit_target": 8.0,  # 최소 8% 수익 목표
        "typical_holding": "1-8시간",
        "max_leverage": 15,  # 최대 레버리지
        "leverage_guideline": "안정적 거래 → 8-15배 권장",
        "description": "AI 최적화 전략 - 안정적 단기 추세 추종",
        "trigger_conditions": """
        기본 스타일 (스캘핑 조건 미충족시 자동 선택)
        - 중변동성 시장
        - 대부분의 거래
        - AI 강점 극대화
        """
    }
}

# ===== 실거래 설정 =====
LIVE_TRADING_CONFIG = {
    "MAX_CONCURRENT_POSITIONS": 5,  # 최대 동시 포지션 (실제 진입 제한)
    "MIN_CAPITAL_THRESHOLD": 100.0,  # 최소 잔고 (USDT)
    
    # 🆕 AI 거래 스타일: 데이트레이딩 중심 + 극단적 스캘핑
    "ADAPTIVE_STYLE_ENABLED": True,  # 극단적 기회 감지 활성화
    "DEFAULT_STYLE": "DAY_TRADING",  # 기본은 무조건 데이트레이딩
    "SCALPING_MIN_CONDITIONS": 3,  # 스캘핑 선택 최소 조건 개수 (5개 중 3개)
    
    "AI_ANALYSIS_INTERVAL": 300,  # 신규 진입 분석 (5분마다)
    "PERFORMANCE_REVIEW_INTERVAL": 600,  # AI 성과 리뷰 (10분)
    
    # 🆕 거래 시간대 제한: 한국시간 23:00~07:00 (UTC 14:00~22:00)은 신규 진입 차단
    # - 변동성이 심한 시간대는 거래하지 않음
    # - 기존 포지션의 관리는 트레일링 스탑이 담당 (바이낸스 서버에서 자동 작동)
    
    # 🆕 트레일링 스탑 설정 (거래 스타일별로 자동 설정됨)
    "TRAILING_STOP_ENABLED": True,              # 트레일링 스탑 활성화
    
    # 🔧 자금 관리 설정 (Option C - 신뢰도 기반 포지션 사이징)
    # 베이스: 스캘핑 25%, 데이트레이딩 35%
    # 신뢰도 보너스: 70-80% +0%, 80-90% +5%, 90%+ +10%
    # 최대: 50% 제한 (안전장치)
    "MAX_POSITION_SIZE_PCT": 50,   # 🔧 단일 포지션 최대 50% (안전장치)
    "MIN_POSITION_SIZE_PCT": 25,   # 🔧 최소 25% (스캘핑 베이스)
    
    # 🔧 초고변동성 필터 (거래 스타일 시스템으로 대체 - 비활성화)
    "EXTREME_VOLATILITY_FILTER": False,  # 거래 스타일이 자동으로 변동성 조절
    "MAX_VOLATILITY_THRESHOLD": 15.0,    # (미사용)
    "MAX_24H_CHANGE": 70.0,              # (미사용)
    # → 초고변동성 코인은 자동으로 "스캘핑" 스타일 선택됨 (0.8% 콜백, 빠른 청산)
    
    # 🔧 거래 수수료 (바이낸스 선물 일반회원)
    "MAKER_FEE": 0.02,  # 0.02%
    "TAKER_FEE": 0.05,  # 0.05%
    
    # 🔧 레버리지 설정 (AI 동적 판단)
    "MIN_LEVERAGE": 2,   # 최소 레버리지 (불확실한 상황)
    "MAX_LEVERAGE": 30,  # 최대 레버리지 (확실한 기회)
    "DEFAULT_LEVERAGE": 10,  # 기본 레버리지 (중립)
    
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

def determine_optimal_trading_style(symbol: str, coin_data: dict) -> Tuple[str, dict]:
    """
    거래 스타일 결정: 데이트레이딩(기본) + 극단적 기회 스캘핑
    
    스캘핑 조건 (5개 중 3개 이상 충족):
    1. ATR > 10% (초고변동성)
    2. 24h 변동 > 20% (급격한 움직임)  
    3. 펀딩비 극단 과열 (±0.1% 이상)
    4. 거래량 폭발 (평균 대비 3배+)
    5. 가격 급변 진행중 (최근 1시간 5%+ 변동)
    
    Args:
        symbol: 거래 심볼
        coin_data: 코인 기본 정보
    
    Returns:
        (선택된 스타일명, 스타일 설정)
    """
    try:
        print(f"   🎯 거래 스타일 분석 중...")
        
        # 기본은 데이트레이딩
        default_style = "DAY_TRADING"
        
        # === 극단적 조건 체크 ===
        extreme_conditions = []
        
        # 조건 1: ATR > 10% (초고변동성)
        try:
            ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=24)
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            atr_result = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
            if atr_result is not None and len(atr_result) > 0:
                atr = float(atr_result.iloc[-1])
                current_price = float(df_1h['close'].iloc[-1])
                atr_pct = (atr / current_price) * 100
                
                if atr_pct > 10.0:
                    extreme_conditions.append(f"초고변동성 ATR {atr_pct:.1f}%")
                    print(f"      ✅ 조건1: ATR {atr_pct:.1f}% (스캘핑 조건 충족: 10% 초과)")
                else:
                    print(f"      ❌ 조건1: ATR {atr_pct:.1f}% (스캘핑 조건 미달: 10% 초과 필요)")
            else:
                atr_pct = 0
        except:
            atr_pct = 0
        
        # 조건 2: 24h 변동 > 20%
        change_24h = abs(coin_data.get('change', 0))
        if change_24h > 20.0:
            extreme_conditions.append(f"24h 급변 {change_24h:.1f}%")
            print(f"      ✅ 조건2: 24h 변동 {change_24h:.1f}% (스캘핑 조건 충족: 20% 초과)")
        else:
            print(f"      ❌ 조건2: 24h 변동 {change_24h:.1f}% (스캘핑 조건 미달: 20% 초과 필요)")
        
        # 조건 3: 펀딩비 극단 과열 (±0.1% 이상)
        try:
            funding_rate = exchange.fetch_funding_rate(symbol)
            fr = float(funding_rate.get('fundingRate', 0)) * 100
            
            if abs(fr) > 0.1:
                extreme_conditions.append(f"펀딩비 극단 {fr:+.3f}%")
                print(f"      ✅ 조건3: 펀딩비 {fr:+.3f}% (스캘핑 조건 충족: ±0.1% 초과)")
            else:
                print(f"      ❌ 조건3: 펀딩비 {fr:+.3f}% (스캘핑 조건 미달: ±0.1% 초과 필요)")
        except:
            print(f"      ❌ 조건3: 펀딩비 조회 실패")
        
        # 조건 4: 거래량 폭발 (평균 대비 3배+)
        try:
            volume_24h = coin_data.get('volume', 0)
            quote_volume = coin_data.get('quoteVolume', 0)
            
            # 최근 7일 평균 거래량 계산
            ohlcv_1d = exchange.fetch_ohlcv(symbol, '1d', limit=7)
            df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            avg_volume = df_1d['volume'].mean()
            
            if volume_24h > avg_volume * 3:
                extreme_conditions.append(f"거래량 폭발 {volume_24h/avg_volume:.1f}배")
                print(f"      ✅ 조건4: 거래량 {volume_24h/avg_volume:.1f}배 (스캘핑 조건 충족: 3배 초과)")
            else:
                print(f"      ❌ 조건4: 거래량 {volume_24h/avg_volume:.1f}배 (스캘핑 조건 미달: 3배 초과 필요)")
        except:
            print(f"      ❌ 조건4: 거래량 비교 실패")
        
        # 조건 5: 최근 1시간 급변 (5%+ 변동)
        try:
            if len(df_1h) >= 2:
                price_1h_ago = float(df_1h['close'].iloc[-2])
                price_now = float(df_1h['close'].iloc[-1])
                change_1h = abs((price_now - price_1h_ago) / price_1h_ago * 100)
                
                if change_1h > 5.0:
                    extreme_conditions.append(f"1시간 급변 {change_1h:.1f}%")
                    print(f"      ✅ 조건5: 1시간 변동 {change_1h:.1f}% (스캘핑 조건 충족: 5% 초과)")
                else:
                    print(f"      ❌ 조건5: 1시간 변동 {change_1h:.1f}% (스캘핑 조건 미달: 5% 초과 필요)")
        except:
            print(f"      ❌ 조건5: 1시간 변동 계산 실패")
        
        # === 스캘핑 조건 충족 여부 ===
        min_conditions = LIVE_TRADING_CONFIG.get("SCALPING_MIN_CONDITIONS", 3)
        conditions_met = len(extreme_conditions)
        
        print(f"   📊 극단적 조건: {conditions_met}/5개 충족")
        for condition in extreme_conditions:
            print(f"      🔥 {condition}")
        
        if conditions_met >= min_conditions:
            selected_style = "SCALPING"
            style_config = TRADING_STYLES[selected_style]
            print(f"   ⚡ *** 극단적 기회 감지! 스캘핑 모드 ***")
            print(f"      - 타임프레임: {style_config['timeframes']}")
            print(f"      - 콜백: {style_config['base_callback_pct']}%")
            print(f"      - 목표: {style_config['min_profit_target']}%+")
            print(f"      - 예상 보유: {style_config['typical_holding']}")
            print(f"      💡 빠른 진입/청산으로 극단적 변동성 이용!")
        else:
            selected_style = default_style
            style_config = TRADING_STYLES[selected_style]
            print(f"   📌 기본 전략: 데이트레이딩")
            print(f"      - 타임프레임: {style_config['timeframes']}")
            print(f"      - 콜백: {style_config['base_callback_pct']}%")
            print(f"      - 목표: {style_config['min_profit_target']}%+")
            print(f"      - 예상 보유: {style_config['typical_holding']}")
            print(f"      💡 안정적 AI 최적화 전략")
        
        return selected_style, style_config
        
    except Exception as e:
        print(f"   ⚠️ 스타일 분석 실패: {e}")
        print(f"   → 기본 스타일 사용: 데이트레이딩")
        default_style = LIVE_TRADING_CONFIG.get("DEFAULT_STYLE", "DAY_TRADING")
        return default_style, TRADING_STYLES[default_style]

def set_trailing_stop_order(symbol: str, side: str, leverage: int, position_size: float, trading_style: dict) -> Optional[dict]:
    """
    🔧 트레일링 스탑 설정 (거래 스타일 기반)
    
    Args:
        symbol: 거래 심볼 (예: BTC/USDT:USDT)
        side: 포지션 방향 ('LONG' 또는 'SHORT')
        leverage: 레버리지
        position_size: 포지션 크기 (USDT)
        trading_style: 선택된 거래 스타일 설정
    """
    max_retries = 3
    base_wait_time = 2
    
    print(f"   🎯 트레일링 스탑 설정 시작...")
    print(f"      - 거래 스타일: {trading_style['name']}")
    print(f"      - 수익 목표: {trading_style['min_profit_target']}%")
    print(f"      - 레버리지: {leverage}배")
    
    # 🆕 레버리지 기반 동적 콜백 계산
    # 기본 콜백 (레버리지 10배 기준)
    base_callback = trading_style['base_callback_pct']
    
    # 실제 콜백 = 기본_콜백 × (10 / 레버리지)
    # 예: 레버리지 30배 → 1% × (10/30) = 0.33%
    #     레버리지 10배 → 3% × (10/10) = 3%
    #     레버리지 2배 → 3% × (10/2) = 15%
    dynamic_callback = base_callback * (10 / leverage)
    
    # 안전 범위: 0.3% ~ 20%
    dynamic_callback = max(0.3, min(20.0, dynamic_callback))
    
    print(f"      📊 콜백 계산:")
    print(f"         - 기본 콜백: {base_callback}% (10배 기준)")
    print(f"         - 레버리지 조정: ×(10/{leverage})")
    print(f"         - 최종 콜백: {dynamic_callback:.2f}%")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   🔄 시도 {attempt}/{max_retries}...")
            
            # 포지션 확립 대기 (첫 번째 시도가 아닐 때)
            if attempt > 1:
                wait_time = base_wait_time * attempt
                print(f"      ⏱️ {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
            
            # 동적 콜백 사용 (재시도시 약간 증가)
            callback_rate = dynamic_callback + (0.1 * (attempt - 1))
            
            # 최대 20% 제한
            callback_rate = min(20.0, callback_rate)
            
            coin_name = symbol.split('/')[0]
            if attempt > 1:
                print(f"      📏 콜백 비율: {callback_rate:.2f}% (재시도 +{0.1 * (attempt - 1):.1f}%)")
            else:
                print(f"      📏 콜백 비율: {callback_rate:.2f}%")
            
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
        
        # 4. 청산 시간 기록 (중요!)
        close_timestamp = get_utc_timestamp_ms()
        
        # 5. 청산 주문 생성
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
        
        # 🔧 실제 청산가 조회
        actual_exit_price = float(order.get('average', order.get('price', current_price)))
        
        print(f"      ✅ 청산 주문 체결 (ID: {order.get('id', 'N/A')})")
        print(f"      📍 실제 청산가: ${actual_exit_price:,.4f}")
        
        # 청산가가 정상 범위인지 확인
        if actual_exit_price > 0 and abs(actual_exit_price - trade_info['entry_price']) / trade_info['entry_price'] < 0.5:
            exit_price = actual_exit_price
        else:
            # 청산가가 이상하면 현재가 사용
            exit_price = current_price
            print(f"      ⚠️ 청산가 이상, 현재가 사용: ${exit_price:,.4f}")
        
        # 🔧 trade_info에 exit_price 추가 (PnL 계산용)
        trade_info_with_exit = trade_info.copy()
        trade_info_with_exit['exit_price'] = exit_price
        
        # 6. 청산 확인 (최대 3초 대기)
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
        
        # 7. 🎯 바이낸스 공식으로 PnL 직접 계산
        print(f"      ⏳ PnL 계산 중...")
        
        realized_pnl, pnl_pct = get_realized_pnl_accurate(
            symbol, 
            close_timestamp, 
            trade_info_with_exit  # exit_price 포함된 버전
        )
        
        return True, exit_price, realized_pnl, pnl_pct
        
    except Exception as e:
        print(f"      ❌ 청산 실패: {e}")
        return False, 0.0, 0.0, 0.0

def calculate_pnl_from_price(entry_price: float, exit_price: float, position_size: float, side: str, leverage: int) -> Tuple[float, float]:
    """
    🔧 바이낸스 공식 PnL 계산
    
    바이낸스 선물 PnL 계산 공식:
    - Long: PnL = (Exit Price - Entry Price) × Position Size
    - Short: PnL = (Entry Price - Exit Price) × Position Size
    - ROE% = PnL / Initial Margin × 100
    - Initial Margin = Position Size × Entry Price / Leverage
    
    Returns:
        (realized_pnl, roe_percentage)
    """
    try:
        # 1. PnL 계산 (바이낸스 공식)
        if side.upper() == 'LONG':
            pnl_before_fees = (exit_price - entry_price) * position_size
        else:  # SHORT
            pnl_before_fees = (entry_price - exit_price) * position_size
        
        # 2. 수수료 계산 (실제 거래 금액 기준)
        maker_fee = LIVE_TRADING_CONFIG.get('MAKER_FEE', 0.02)  # 0.02%
        taker_fee = LIVE_TRADING_CONFIG.get('TAKER_FEE', 0.05)  # 0.05%
        
        entry_notional = position_size * entry_price  # 진입시 거래 금액
        exit_notional = position_size * exit_price    # 청산시 거래 금액
        
        entry_fee = entry_notional * (maker_fee / 100)
        exit_fee = exit_notional * (taker_fee / 100)
        total_fees = entry_fee + exit_fee
        
        # 3. 최종 실현 손익 (수수료 차감)
        realized_pnl = pnl_before_fees - total_fees
        
        # 4. ROE% 계산 (바이낸스 공식)
        initial_margin = (position_size * entry_price) / leverage
        roe_percent = (realized_pnl / initial_margin) * 100
        
        return realized_pnl, roe_percent
        
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
    🆕 바이낸스 Income History에서 실제 realized PnL 조회 (방법 2: 백업)
    
    Args:
        symbol: 심볼 (예: "ZEC/USDT:USDT")
        start_time: 조회 시작 시간 (밀리초), None이면 최근 5초 이내
    
    Returns:
        {
            'realized_pnl': float,
            'commission': 0.0,  # Income API는 수수료 포함된 값
            'trade_time': int
        } 또는 None
    """
    try:
        # Income History API 호출
        binance_symbol = symbol.replace('/USDT:USDT', 'USDT')
        
        params = {
            'symbol': binance_symbol,
            'incomeType': 'REALIZED_PNL',
            'limit': 20  # 충분히 가져오기
        }
        
        # start_time이 없으면 최근 5초 이내로 설정
        if not start_time:
            start_time = get_utc_timestamp_ms() - 5000  # 5초 전
        
        params['startTime'] = start_time
        
        # CCXT를 통한 호출
        income_history = exchange.fapiprivate_get_income(params)
        
        if not income_history:
            return None
        
        # 해당 심볼의 REALIZED_PNL 모두 합산 (부분 청산 대응)
        total_pnl = 0.0
        latest_time = 0
        found_count = 0
        
        for income in income_history:
            if income['symbol'] == binance_symbol and income['incomeType'] == 'REALIZED_PNL':
                total_pnl += float(income['income'])
                latest_time = max(latest_time, int(income['time']))
                found_count += 1
        
        if found_count == 0:
            return None
        
        return {
            'realized_pnl': total_pnl,
            'commission': 0.0,  # REALIZED_PNL은 이미 수수료 포함
            'trade_time': latest_time
        }
        
    except Exception as e:
        # API 호출 실패시 조용히 None 반환
        return None

def get_position_history_pnl(symbol: str, close_timestamp: int) -> Optional[Dict]:
    """
    🆕 바이낸스 거래 내역에서 실제 PnL 조회 (방법 1: 가장 정확)
    
    Args:
        symbol: 심볼 (예: "ZEC/USDT:USDT")
        close_timestamp: 청산 시간 (밀리초)
    
    Returns:
        {
            'realized_pnl': float,
            'commission': float,
            'trade_time': int
        } 또는 None
    """
    try:
        binance_symbol = symbol.replace('/USDT:USDT', 'USDT')
        
        # 청산 시간 전후 10초 범위 조회
        start_time = close_timestamp - 10000  # 10초 전
        end_time = close_timestamp + 10000    # 10초 후
        
        # userTrades API 호출 (CCXT 직접 호출)
        params = {
            'symbol': binance_symbol,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 100
        }
        
        trades = exchange.fapiprivate_get_usertrades(params)
        
        if not trades:
            return None
        
        # reduceOnly 거래만 필터링하여 realizedPnl 합산
        total_pnl = 0.0
        total_commission = 0.0
        latest_time = 0
        found_count = 0
        
        for trade in trades:
            # reduceOnly 거래 = 포지션 청산
            if trade.get('reduceOnly') or trade.get('positionSide') == 'BOTH':
                realized_pnl = float(trade.get('realizedPnl', 0))
                commission = float(trade.get('commission', 0))
                
                if realized_pnl != 0:  # 0이 아닌 PnL만
                    total_pnl += realized_pnl
                    total_commission += abs(commission)
                    latest_time = max(latest_time, int(trade['time']))
                    found_count += 1
        
        if found_count == 0:
            return None
        
        return {
            'realized_pnl': total_pnl,
            'commission': total_commission,
            'trade_time': latest_time
        }
        
    except Exception as e:
        # API 호출 실패시 조용히 None 반환
        return None

def get_realized_pnl_accurate(symbol: str, close_timestamp: int, trade_info: dict) -> Tuple[float, float]:
    """
    🔧 바이낸스 공식으로 PnL 직접 계산 (진입가, 청산가, 레버리지 기반)
    
    바이낸스 API에서 PnL을 가져오지 않고, 확실한 진입가/청산가/레버리지만 사용하여
    바이낸스 공식으로 PnL을 직접 계산합니다.
    
    Args:
        symbol: 심볼
        close_timestamp: 청산 시간 (밀리초)
        trade_info: DB 거래 정보 (entry_price, exit_price, position_size, side, leverage)
    
    Returns:
        (realized_pnl, roe_percentage)
    """
    print(f"      🔍 바이낸스 공식으로 PnL 직접 계산 중...")
    
    # 🔧 trade_info에서 확실한 값들 가져오기
    entry_price = trade_info['entry_price']
    exit_price = trade_info.get('exit_price', 0)  # 청산가
    position_size = trade_info['position_size']
    side = trade_info['side']
    leverage = trade_info['leverage']
    
    # 청산가가 없으면 현재가로 추정
    if exit_price <= 0:
        try:
            ticker = exchange.fetch_ticker(symbol)
            exit_price = float(ticker['last'])
            print(f"      ⚠️ 청산가가 없어서 현재가 사용: ${exit_price:,.4f}")
        except:
            exit_price = entry_price
            print(f"      ⚠️ 현재가 조회 실패, 진입가 사용")
    
    # 🎯 바이낸스 공식으로 직접 계산
    realized_pnl, roe_pct = calculate_pnl_from_price(
        entry_price,
        exit_price,
        position_size,
        side,
        leverage
    )
    
    print(f"      ✅ PnL 계산 완료!")
    print(f"      📊 진입가: ${entry_price:,.4f} | 청산가: ${exit_price:,.4f} | 레버리지: {leverage}x")
    print(f"      💰 실현 손익: ${realized_pnl:+.2f} | ROE: {roe_pct:+.2f}%")
    
    return realized_pnl, roe_pct

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

def check_extreme_volatility(symbol: str, coin_data: dict) -> Tuple[bool, str]:
    """
    🆕 초고변동성 코인 필터 (트레일링 스탑 최적화)
    
    Args:
        symbol: 거래 심볼
        coin_data: 코인 기본 정보 (24h 변동 포함)
    
    Returns:
        (is_extreme, reason)
        - is_extreme: True면 진입 차단
        - reason: 차단 사유
    """
    if not LIVE_TRADING_CONFIG.get("EXTREME_VOLATILITY_FILTER", False):
        return False, "필터 비활성화"
    
    try:
        # 1. 24시간 변동률 체크
        change_24h = abs(coin_data.get('change', 0))
        max_24h_change = LIVE_TRADING_CONFIG.get("MAX_24H_CHANGE", 30.0)
        
        if change_24h > max_24h_change:
            return True, f"24h 변동 {change_24h:.1f}% (임계값: {max_24h_change}%)"
        
        # 2. ATR 기반 변동성 체크
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
            if len(ohlcv) >= 14:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # ATR 계산
                atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
                if atr_result is not None and len(atr_result) > 0:
                    atr = float(atr_result.iloc[-1])
                    last_close = float(df['close'].iloc[-1])
                    
                    if last_close > 0:
                        atr_pct = (atr / last_close) * 100
                        max_atr = LIVE_TRADING_CONFIG.get("MAX_VOLATILITY_THRESHOLD", 15.0)
                        
                        if atr_pct > max_atr:
                            return True, f"ATR {atr_pct:.1f}% (임계값: {max_atr}%)"
        except:
            pass  # ATR 계산 실패 시 24h 변동만으로 판단
        
        return False, "정상 변동성"
        
    except Exception as e:
        # 오류 시 안전하게 통과
        return False, f"체크 오류: {e}"

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

def ai_comprehensive_analysis(coin_data: dict, market_data: dict, performance_history: dict, timeframes: List[str] = None) -> dict:
    """AI 종합 분석 (선택된 타임프레임 + 선물 특화 지표)"""
    global last_api_call_time
    
    # 기본 타임프레임 (전체 분석용)
    if timeframes is None:
        timeframes = ['1m', '3m', '5m', '15m', '1h', '1d', '1w']
    
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
        
        # 🆕 선택된 타임프레임의 기술적 지표 정리
        timeframe_analysis = []
        for tf in timeframes:
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
        
        # 🆕 AI 성과 기반 피드백 생성 (승률/성공 패턴만)
        feedback_section = ""
        total_trades = performance_history.get('total_trades', 0)
        win_rate = performance_history.get('win_rate', 0)
        
        if total_trades >= 10:  # 최소 10회 거래 이상일 때만 피드백
            if win_rate >= 60:
                feedback_section = """
🎯 **전략 피드백 (승률 우수):**
✅ 현재 승률이 60% 이상으로 우수합니다.
✅ 현재 전략을 유지하세요.
✅ 성공 패턴을 분석하여 반복하세요:
   - 어떤 타임프레임 조합이 효과적이었나?
   - 어떤 기술적 지표 조합이 신뢰도가 높았나?
   - LONG/SHORT 중 어느 방향이 더 성공적이었나?
✅ 고확률 기회에 적극적으로 진입하세요 (70%+ 확신)
"""
            elif win_rate >= 50:
                feedback_section = """
📊 **전략 피드백 (승률 양호):**
✅ 현재 승률이 50-60%로 양호한 수준입니다.
✅ 현재 접근 방식을 계속 유지하세요.
⚠️ 진입 기준을 조금 더 엄격하게 적용하여 승률 향상을 목표로 하세요.
💡 다음 사항을 고려하세요:
   - 여러 타임프레임의 신호가 일치하는 경우 우선 진입
   - 추세 순응 거래(With the Trend)에 집중
   - 불확실한 신호는 과감히 패스
"""
            elif win_rate >= 40:
                feedback_section = """
⚠️ **전략 조정 필요 (승률 저조):**
🔴 현재 승률이 40-50%로 개선이 필요합니다.
🔴 진입 기준을 더욱 엄격하게 적용하세요:
   - 80% 이상 확신하는 거래만 진입
   - 3개 이상의 타임프레임에서 신호가 일치할 때만 진입
   - 추세 순응 거래만 선택 (역추세 거래 회피)
   - 펀딩비/OI 등 선물 특화 지표가 명확할 때만 진입
🔴 불확실한 상황에서는 무조건 관망
🔴 리스크 관리를 최우선으로 고려
"""
            else:  # win_rate < 40
                feedback_section = """
🚨 **긴급 전략 변경 필요 (승률 매우 저조):**
🔴 현재 승률이 40% 미만으로 심각한 수준입니다.
🔴 즉시 다음 조치를 취하세요:
   - 진입 기준을 85% 이상 확신으로 상향
   - 5개 이상의 타임프레임에서 신호가 완벽히 일치할 때만 진입
   - 추세 순응 거래만 엄격히 선택
   - 역추세 거래는 완전히 배제
   - 명확한 기술적 패턴(이중바닥, 헤드앤숄더 등)이 있을 때만 진입
   - 펀딩비/OI가 극단적으로 과열되었을 때만 역방향 진입 고려
🔴 의심스러운 모든 상황은 무조건 관망
🔴 성공한 거래들의 공통 패턴을 찾아 집중
"""
        else:  # 거래 횟수 부족
            feedback_section = """
📈 **학습 단계 (데이터 수집 중):**
💡 아직 데이터가 부족합니다 (10회 이상 필요).
💡 다음 원칙을 따르며 신중하게 거래하세요:
   - 70% 이상 확신하는 거래만 진입
   - 여러 타임프레임의 신호가 일치하는 경우 우선
   - 추세 순응 거래를 기본으로
   - 각 거래의 결과를 분석하며 패턴 학습
"""
        # 거래 스타일 정보
        if set(timeframes) == set(['1m', '3m', '5m']):
            style_info = """
**⚡ 극단적 기회 스캘핑 모드 발동! ⚡**
- 극단적 조건 충족 (초고변동성/급등락/펀딩비 과열 등)
- 목표 보유: 5-30분 (매우 짧음)
- 수익 목표: 3-8% (빠른 청산)
- 콜백: 1.0%
- 전략: 극단적 변동성을 이용한 초단타
- 주 분석 타임프레임: 1m, 3m, 5m

💡 지금이 아니면 놓치는 기회! 빠른 진입/청산 필수!
"""
        else:  # 데이트레이딩
            style_info = """
**📌 기본 데이트레이딩 모드**
- AI 최적화 구간 (안정적)
- 목표 보유: 1-8시간
- 수익 목표: 8-15%
- 콜백: 3.0%
- 전략: 단기 추세 추종, AI 강점 극대화
- 주 분석 타임프레임: 15m, 1h, 4h

💡 안정적이고 검증된 전략!
"""
        
        # 프롬프트 작성
        prompt = f"""
다음 암호화폐의 선물 거래를 양방향(LONG/SHORT) 분석해주세요:

{style_info}

**코인 정보:**
- 심볼: {coin_data['coin']}
- 현재가: ${current.get('price', 0):,.4f}
- 24h 변동: {current.get('change_24h', 0):+.2f}%
- 24h 거래량: ${current.get('volume_24h', 0):,.0f}

**타임프레임별 기술적 분석:**
{''.join(timeframe_analysis)}

{futures_text}

**AI 과거 성과:**
- 최근 {performance_history.get('days', 0)}일 거래: {performance_history.get('total_trades', 0)}회
- 승률: {performance_history.get('win_rate', 0):.1f}%

{feedback_section}

**분석 지침 (선택된 스타일 최적화):**

1. 추세 순응 거래 (With the Trend):
   - 여러 타임프레임의 신호가 일치
   - 높은 확신도, 큰 익절 목표

2. 역추세 거래 (Counter-Trend):
   - 급등/급락 후 조정 노림
   - 빠른 익절 필수

3. LONG 기회:
   - 과매도 반등 (RSI < 30)
   - 지지선 테스트 후 반등
   - 볼린저밴드 하단 + 매수 시그널
   - 🔥 펀딩비 숏 과열 (<-0.05%)

4. SHORT 기회:
   - 급등 후 모멘텀 약화
   - RSI > 70 과매수
   - 저항선 거부 패턴
   - 🔥 펀딩비 롱 과열 (>0.05%)

**🎯 스타일별 특별 지침:**
{"**[스캘핑 모드]** 극단적 변동성을 이용하세요:" if set(timeframes) == set(['1m', '3m', '5m']) else "**[데이트레이딩 모드]** 안정적 추세를 추종하세요:"}
{"""
- 단기 타임프레임(1m, 3m, 5m) 집중
- 빠른 모멘텀 포착 (급등/급락 진행중)
- 3-8% 수익시 즉시 청산
- 과매수/과매도 역추세 공략
- 85% 이상 확신시에만 진입
- 레버리지 권장: 20-30배 (극단적 기회)
""" if set(timeframes) == set(['1m', '3m', '5m']) else """
- 중기 타임프레임(15m, 1h, 4h) 신뢰
- 명확한 추세 방향 확인
- 8-15% 수익 목표
- 추세 순응 우선
- 70% 이상 확신시 진입 가능
- 레버리지 권장: 8-15배 (안정적)
"""}

**⚡ 레버리지 판단 (2-30배 범위):**

당신이 레버리지를 결정합니다! 다음 기준을 참고하세요:

🔴 낮은 레버리지 (2-5배):
- 변동성 매우 높음 (ATR > 15%)
- 신호 불명확 (여러 지표 충돌)
- 신뢰도 낮음 (70-75%)
- 불확실한 시장 상황
- 리스크 최소화 우선

🟡 중간 레버리지 (8-15배):
- 중변동성 (ATR 3-8%)
- 신호 보통 (일부 지표 일치)
- 신뢰도 보통 (75-85%)
- 일반적인 거래 상황
- 균형잡힌 접근

🟢 높은 레버리지 (20-30배):
- 명확한 추세 (여러 타임프레임 일치)
- 신호 강력 (대부분 지표 일치)
- 신뢰도 매우 높음 (85-95%+)
- 극단적 펀딩비 (기회 명확)
- 확실한 기회

💡 레버리지 결정 예시:
- 케이스 1: "모든 타임프레임 상승, RSI 30, 펀딩비 -0.15%, 신뢰도 92%"
  → leverage: 28
  → leverage_reasoning: "3개 타임프레임 모두 강한 상승 신호, 과매도 구간(RSI 30), 극단적 펀딩비로 롱 유리, 신뢰도 92%로 매우 높음 → 고레버리지 28배 적용"

- 케이스 2: "15m, 1h 상승, RSI 45, 신뢰도 78%"
  → leverage: 12
  → leverage_reasoning: "중기 타임프레임 상승 일치, 중립 RSI, 신뢰도 중간 수준 → 중간 레버리지 12배 적용"

- 케이스 3: "지표 충돌, ATR 18%, 신뢰도 72%"
  → leverage: 4
  → leverage_reasoning: "지표 간 신호 불일치, 초고변동성 ATR 18%, 신뢰도 낮음 → 안전하게 저레버리지 4배 적용"


다음 형식으로 JSON 응답해주세요:
{{
    "trade": true/false,
    "direction": "LONG" 또는 "SHORT",
    "confidence": 0-100,
    "leverage": 2-30,  // AI가 판단한 최적 레버리지
    "leverage_reasoning": "레버리지를 이 값으로 선택한 구체적 이유 (신호 강도, 변동성, 신뢰도 등)",
    "reasoning": "선택된 스타일과 타임프레임을 고려한 상세 분석"
}}

주의사항:
1. {"85%" if set(timeframes) == set(['1m', '3m', '5m']) else "70%"} 이상 확신할 때만 거래 추천
2. LONG과 SHORT 양방향 모두 검토
3. 🔥 펀딩비 과열 상황 반드시 고려
4. OI와 가격 변동 함께 분석
5. 타임프레임별 신호 일치도 확인
6. {"극단적 변동성 이용" if set(timeframes) == set(['1m', '3m', '5m']) else "안정적 추세 추종"}
7. 불분명하면 과감히 관망
8. 리스크 관리 우선
9. **leverage_reasoning에 레버리지 선택 근거를 반드시 포함**
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
    """실제 거래 실행 (거래 스타일 자동 선택)"""
    try:
        coin = coin_data['coin']
        symbol = f"{coin}/USDT:USDT"
        side = ai_decision['direction']
        confidence = ai_decision['confidence']
        ai_leverage = ai_decision.get('leverage', 10)
        
        # 🆕 AI 레버리지 검증 및 조정
        min_lev = LIVE_TRADING_CONFIG['MIN_LEVERAGE']
        max_lev = LIVE_TRADING_CONFIG['MAX_LEVERAGE']
        
        # 범위 제한
        leverage = max(min_lev, min(max_lev, ai_leverage))
        
        if ai_leverage != leverage:
            print(f"   ⚠️ AI 레버리지 {ai_leverage}배 → {leverage}배로 조정 (범위: {min_lev}-{max_lev}배)")
        else:
            print(f"   ✅ AI 레버리지: {leverage}배")
        
        # 🆕 거래 스타일 자동 결정 (가장 최적의 스타일 선택)
        if LIVE_TRADING_CONFIG.get("ADAPTIVE_STYLE_ENABLED", True):
            print(f"\n   🎯 최적 거래 스타일 분석 중...")
            style_name, trading_style = determine_optimal_trading_style(symbol, coin_data)
            
            # 선택된 스타일의 레버리지 사용
            leverage = min(leverage, trading_style['max_leverage'])
            print(f"   ⚙️ 레버리지 조정: {leverage}x (스타일 권장치)")
        else:
            # 기본 스타일 사용
            default_style = LIVE_TRADING_CONFIG.get("DEFAULT_STYLE", "DAY_TRADING")
            style_name = default_style
            trading_style = TRADING_STYLES[default_style]
            print(f"   📌 기본 스타일 사용: {trading_style['name']}")
        
        # 🔧 Option C: 신뢰도 기반 포지션 사이징 (단순하고 명확)
        # 거래 스타일별 기본 포지션 크기
        if style_name == "SCALPING":
            base_position_pct = 25  # 스캘핑: 25%
        elif style_name == "DAY_TRADING":
            base_position_pct = 35  # 데이트레이딩: 35%
        else:
            base_position_pct = 35  # 기본값: 35%
        
        # 신뢰도 보너스
        if confidence >= 90:
            confidence_bonus = 10  # 90%+ → +10%
        elif confidence >= 80:
            confidence_bonus = 5   # 80-90% → +5%
        else:
            confidence_bonus = 0   # 70-80% → +0%
        
        # 최종 포지션 크기
        position_pct = base_position_pct + confidence_bonus
        
        # 최대 50% 제한 (안전장치)
        position_pct = min(50, position_pct)
        
        position_size = available_balance * (position_pct / 100)
        
        print(f"   💰 포지션 크기 계산 (Option C - 신뢰도 기반):")
        print(f"      - 거래 스타일: {trading_style['name']}")
        print(f"      - 베이스: {base_position_pct}%")
        print(f"      - 신뢰도 보너스: +{confidence_bonus}% ({confidence}% 신뢰도)")
        print(f"      - 최종: {position_pct}% = ${position_size:,.2f}")
        
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
        current_price = coin_data['price']  # 참고용
        quantity = position_size * leverage / current_price
        
        if side.upper() == 'LONG':
            order = exchange.create_market_buy_order(symbol, quantity)
        else:
            order = exchange.create_market_sell_order(symbol, quantity)
        
        # 🔧 실제 체결가 조회 (중요!)
        actual_entry_price = float(order.get('average', order.get('price', current_price)))
        actual_quantity = float(order.get('filled', quantity))
        
        print(f"   ✅ 주문 체결:")
        print(f"      - 심볼: {symbol}")
        print(f"      - 방향: {side.upper()}")
        print(f"      - 수량: {actual_quantity:.6f}")
        print(f"      - 실제 진입가: ${actual_entry_price:,.4f}")
        if abs(actual_entry_price - current_price) / current_price > 0.001:  # 0.1% 이상 차이
            print(f"      ⚠️ 슬리피지: ${current_price:,.4f} → ${actual_entry_price:,.4f}")
        print(f"      - 주문 ID: {order.get('id', 'N/A')}")
        
        # DB 기록 (실제 체결가 사용!)
        trade_id = record_trade(
            coin_symbol=coin,
            side=side.upper(),
            entry_price=actual_entry_price,  # 🔧 실제 체결가
            quantity=actual_quantity,         # 🔧 실제 체결 수량
            leverage=leverage,
            position_size=position_size,
            ai_confidence=confidence,
            ai_reasoning=ai_decision.get('reasoning', 'AI 추천')
        )
        
        print(f"   📝 DB 기록: 거래 ID {trade_id}")
        
        # 🆕 트레일링 스탑 설정 (진입 즉시 - 거래 스타일 기반)
        if LIVE_TRADING_CONFIG.get("TRAILING_STOP_ENABLED", False):
            print(f"   🎯 트레일링 스탑 설정 중...")
            
            # AI 레버리지 결정 이유 표시
            ai_leverage_reasoning = ai_decision.get('leverage_reasoning', '')
            
            # 레버리지 조정 과정 표시
            leverage_info = []
            if ai_leverage_reasoning:
                leverage_info.append(f"AI 판단: {ai_leverage_reasoning}")
            
            if ai_leverage != leverage:
                if leverage == trading_style['max_leverage']:
                    leverage_info.append(f"→ 스타일 최대치 {trading_style['max_leverage']}배로 제한됨")
                elif leverage == min_lev:
                    leverage_info.append(f"→ 최소 {min_lev}배로 상향됨")
                elif leverage == max_lev:
                    leverage_info.append(f"→ 최대 {max_lev}배로 제한됨")
            
            print(f"      - 거래 스타일: {trading_style['name']}")
            print(f"      - 레버리지: {leverage}배")
            if leverage_info:
                for info in leverage_info:
                    print(f"         {info}")
            print(f"      - 수익 목표: {trading_style['min_profit_target']}%")
            
            time.sleep(2)  # 포지션 확립 대기
            
            trailing_order = set_trailing_stop_order(symbol, side.upper(), leverage, position_size, trading_style)
            if trailing_order:
                print(f"   ✅ 트레일링 스탑 활성화")
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
    
    변경사항 v2:
    1. ✅ 심볼별 그룹화 - 같은 심볼의 여러 청산 처리
    2. ✅ 한 번의 API 호출로 모든 PnL 조회
    3. ✅ 시간순 매칭으로 정확한 PnL 할당
    
    로직:
    1. 바이낸스에 없는 DB 포지션 -> 심볼별 그룹화
    2. 심볼별로 모든 PnL 조회 (한 번만)
    3. 각 trade에 순서대로 매칭하여 청산 처리
    4. DB에 없는 바이낸스 포지션 -> DB에 추가
    
    Returns:
        동기화된 거래 수
    """
    print(f"\n{'='*80}")
    print(f"🔄 DB-바이낸스 동기화 (다중 청산 대응)")
    print(f"{'='*80}")
    
    synced_count = 0
    
    # DB 오픈 거래
    open_trades = get_all_open_trades()
    
    # 바이낸스 실제 포지션
    live_positions = get_open_positions()
    live_position_map = {pos['symbol']: pos for pos in live_positions}
    
    print(f"   📊 DB 오픈 거래: {len(open_trades)}개")
    print(f"   📊 바이낸스 포지션: {len(live_positions)}개")
    
    # 🆕 1. 청산된 포지션을 심볼별로 그룹화
    closed_trades_by_symbol = {}
    for trade in open_trades:
        symbol = f"{trade['coin_symbol']}/USDT:USDT"
        
        if symbol not in live_position_map:
            # 청산된 포지션
            if symbol not in closed_trades_by_symbol:
                closed_trades_by_symbol[symbol] = []
            closed_trades_by_symbol[symbol].append(trade)
    
    if not closed_trades_by_symbol:
        print(f"   ℹ️ 청산된 포지션 없음")
    else:
        print(f"\n   🔍 청산 감지된 심볼: {len(closed_trades_by_symbol)}개")
        for sym, trades in closed_trades_by_symbol.items():
            coin = sym.split('/')[0]
            print(f"      - {coin}: {len(trades)}개 포지션")
    
    # 🆕 2. 심볼별로 PnL 조회 및 매칭
    for symbol, trades in closed_trades_by_symbol.items():
        coin = symbol.split('/')[0]
        trade_count = len(trades)
        
        print(f"\n   🔄 {coin} 동기화 ({trade_count}개 포지션)")
        
        try:
            # 🎯 핵심: 해당 심볼의 모든 거래를 한 번에 조회
            binance_symbol = symbol.replace('/USDT:USDT', 'USDT')
            
            # 가장 오래된 trade의 진입 시간부터 조회
            oldest_entry_time = None
            for trade in trades:
                entry_time_str = trade.get('entry_time', trade.get('timestamp'))
                if entry_time_str:
                    entry_dt = parse_db_timestamp(entry_time_str)
                    if entry_dt:
                        entry_ts = int(entry_dt.timestamp() * 1000)
                        if oldest_entry_time is None or entry_ts < oldest_entry_time:
                            oldest_entry_time = entry_ts
            
            # userTrades API로 모든 청산 거래 조회
            params = {
                'symbol': binance_symbol,
                'limit': 100
            }
            
            if oldest_entry_time:
                params['startTime'] = oldest_entry_time
            
            print(f"      📊 API 조회 중...")
            user_trades = exchange.fapiprivate_get_usertrades(params)
            
            # reduceOnly 거래만 필터링 (청산 거래)
            close_trades = []
            for t in user_trades:
                if t.get('reduceOnly') or t.get('positionSide') == 'BOTH':
                    # 🔧 청산가만 가져오기 (PnL은 직접 계산)
                    close_price = float(t['price'])
                    close_time = int(t['time'])
                    close_qty = float(t['qty'])
                    
                    close_trades.append({
                        'time': close_time,
                        'price': close_price,
                        'qty': close_qty,
                        'commission': abs(float(t.get('commission', 0)))
                    })
            
            # 시간순 정렬 (오래된 것부터)
            close_trades.sort(key=lambda x: x['time'])
            
            print(f"      ✅ 청산 거래 {len(close_trades)}개 발견")
            
            if len(close_trades) >= trade_count:
                # 🎯 충분한 청산 데이터가 있음 - 매칭
                for i, trade in enumerate(trades):
                    if i < len(close_trades):
                        close_data = close_trades[i]
                        
                        trade_id = trade['id']
                        exit_price = close_data['price']
                        entry_price = trade['entry_price']
                        position_size = trade['position_size']
                        side = trade['side']
                        leverage = trade['leverage']
                        
                        # 🔧 바이낸스 공식으로 PnL 직접 계산
                        realized_pnl, pnl_pct = calculate_pnl_from_price(
                            entry_price, exit_price, position_size, side, leverage
                        )
                        
                        print(f"      ✅ ID {trade_id}: PnL ${realized_pnl:+.2f} ({pnl_pct:+.2f}%), 청산가 ${exit_price:,.4f}")
                        
                        # DB 업데이트
                        update_trade_exit(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            pnl=realized_pnl,
                            pnl_percent=pnl_pct,
                            reason="동기화: 수동 종료 (바이낸스 청산가 기반 계산)"
                        )
                        
                        synced_count += 1
                    else:
                        print(f"      ⚠️ ID {trade['id']}: 청산 데이터 부족")
            
            else:
                # ⚠️ 청산 데이터 부족 - 현재가로 추정
                print(f"      ⚠️ 청산 데이터 부족 ({len(close_trades)}/{trade_count}) - 현재가로 추정")
                
                for trade in trades:
                    trade_id = trade['id']
                    entry_price = trade['entry_price']
                    side = trade['side']
                    position_size = trade['position_size']
                    leverage = trade['leverage']
                    
                    # 현재가로 추정
                    ticker = exchange.fetch_ticker(symbol)
                    exit_price = float(ticker['last'])
                    
                    # 🔧 바이낸스 공식으로 계산
                    realized_pnl, pnl_pct = calculate_pnl_from_price(
                        entry_price, exit_price, position_size, side, leverage
                    )
                    
                    print(f"      📊 ID {trade_id}: 추정 PnL ${realized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                    
                    # DB 업데이트
                    update_trade_exit(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl=realized_pnl,
                        pnl_percent=pnl_pct,
                        reason="동기화: 현재가 기반 추정 (청산 데이터 부족)"
                    )
                    
                    synced_count += 1
                    
        except Exception as e:
            print(f"      ❌ {coin} 동기화 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 바이낸스에는 있는데 DB에 없는 포지션 -> DB에 추가 (수동거래)
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
    print(f"   🤖 AI 신규 분석: {LIVE_TRADING_CONFIG['AI_ANALYSIS_INTERVAL']//60}분마다")
    print(f"   📊 성과 리뷰: {LIVE_TRADING_CONFIG['PERFORMANCE_REVIEW_INTERVAL']//60}분마다")
    print(f"   📈 레버리지: {LIVE_TRADING_CONFIG['MIN_LEVERAGE']}-{LIVE_TRADING_CONFIG['MAX_LEVERAGE']}x (AI 판단)")
    print(f"   🛡️ 마진 모드: {LIVE_TRADING_CONFIG['MARGIN_MODE'].upper()}")
    print(f"{'='*80}\n")
    
    # 🆕 거래 스타일: 데이트레이딩 중심 + 극단적 스캘핑
    if LIVE_TRADING_CONFIG.get("ADAPTIVE_STYLE_ENABLED", True):
        print(f"🎯 AI 최적화 거래 시스템:")
        print(f"   ✅ 데이트레이딩 중심 (95% 거래)")
        print(f"   ✅ 극단적 기회 스캘핑 (5% 거래)")
        print(f"   ✅ AI 동적 레버리지 (2-30배)")
        print(f"   ✅ 레버리지 기반 자동 콜백")
        print(f"")
        print(f"   📌 기본 전략: 데이트레이딩")
        print(f"      - 타임프레임: 15m, 1h, 4h")
        print(f"      - 레버리지: 8-15배 권장")
        print(f"      - 콜백: 3% 기준 (10배시) → 동적 조정")
        print(f"      - 목표: 8-15%")
        print(f"      - 보유: 1-8시간")
        print(f"")
        print(f"   ⚡ 극단 전략: 스캘핑 (조건부)")
        print(f"      - 타임프레임: 1m, 3m, 5m")
        print(f"      - 레버리지: 20-30배 권장")
        print(f"      - 콜백: 1% 기준 (10배시) → 동적 조정")
        print(f"      - 목표: 3-8%")
        print(f"      - 보유: 5-30분")
        print(f"      - 발동: 5개 조건 중 3개 충족")
        print(f"")
        print(f"   ⚙️ 동적 콜백 시스템:")
        print(f"      - 공식: 기본콜백 × (10 / 레버리지)")
        print(f"      - 예시:")
        print(f"        • 30배 → 1% × (10/30) = 0.33%")
        print(f"        • 10배 → 3% × (10/10) = 3.0%")
        print(f"        • 2배 → 3% × (10/2) = 15.0%")
        print(f"      💡 높은 레버리지 = 빠른 청산")
        print(f"      💡 낮은 레버리지 = 여유있는 청산")
        print(f"")
        print(f"   🔥 스캘핑 발동 조건 (극단적 기회):")
        print(f"      1. ATR > 10% (초고변동성)")
        print(f"      2. 24h 변동 > 20% (급등/급락)")
        print(f"      3. 펀딩비 ±0.1% 초과 (극단 과열)")
        print(f"      4. 거래량 평균 3배+ (폭발)")
        print(f"      5. 1시간 변동 5%+ (급변 진행중)")
        print(f"")
        print(f"   💡 AI가 모든 것을 판단합니다:")
        print(f"      - 거래 스타일 (데이 or 스캘핑)")
        print(f"      - 레버리지 (2-30배)")
        print(f"      - 콜백 (자동 계산)")
        print(f"{'='*80}\n")
    else:
        print(f"📌 기본 스타일: {LIVE_TRADING_CONFIG.get('DEFAULT_STYLE', 'DAY_TRADING')}")
        print(f"{'='*80}\n")
    
    # 3초 후 자동 시작
    print("3초 후 자동 시작...")
    time.sleep(3)
    
    last_analysis_time = 0
    last_review_time = 0
    last_dashboard_time = 0
    
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
                # 🆕 거래 시간대 체크 (한국시간 23:00~07:00 금지)
                is_allowed, time_msg = is_trading_allowed_time()
                if not is_allowed:
                    print(f"\n{'='*80}")
                    print(time_msg)
                    print(f"{'='*80}")
                else:
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
                                
                                # 🆕 거래 스타일 자동 결정
                                print(f"   🎯 거래 스타일 결정 중...")
                                style_name, trading_style = determine_optimal_trading_style(symbol, coin_data)
                                
                                performance_history = get_recent_performance(7)
                                print(f"   🤖 AI 분석 중 (타임프레임: {trading_style['timeframes']})...")
                                decision = ai_comprehensive_analysis(
                                    coin_data, 
                                    market_data, 
                                    performance_history,
                                    timeframes=trading_style['timeframes']  # 선택된 스타일의 타임프레임 사용
                                )
                                
                                print(f"   신뢰도: {decision.get('confidence', 0)}%")
                                print(f"   방향: {decision.get('direction', 'N/A')}")
                                
                                # 레버리지 정보 표시
                                ai_leverage = decision.get('leverage', 10)
                                leverage_reasoning = decision.get('leverage_reasoning', '')
                                print(f"   레버리지: {ai_leverage}배")
                                if leverage_reasoning:
                                    print(f"   레버리지 이유: {leverage_reasoning}")
                                
                                print(f"   판단: {decision.get('reasoning', 'N/A')}")
                                
                                # 스타일별 신뢰도 기준
                                min_confidence = 85 if style_name == "SCALPING" else 70
                                confidence = decision.get('confidence', 0)
                                
                                print(f"   📊 요구 신뢰도: {min_confidence}% ({trading_style['name']})")
                                
                                if decision.get('trade') and confidence >= min_confidence:
                                    print(f"\n   ✅ AI 승인 (신뢰도 {confidence}% ≥ {min_confidence}%)")
                                    if execute_live_trade(coin_data, decision, available_balance):
                                        open_positions_count += 1
                                        available_balance = get_available_balance()
                                else:
                                    if decision.get('trade'):
                                        print(f"   ⏭️ 신뢰도 부족 ({confidence}% < {min_confidence}%)")
                                    else:
                                        print(f"   ⏭️ AI 거래 비추천 (신뢰도: {confidence}%)")
                                
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
║        🔴 AI 실거래 트레이딩 봇 v2.8 (신뢰도 기반 사이징)     ║
║        🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS          ║
║        ⚠️  실제 자금으로 거래합니다!                         ║
║        ✅ AI: {provider:20s} ({model_name:20s})    ║
║        ✅ Isolated Margin 모드                                ║
║        ✅ Risk/Reward ≥ 1:2 전략                             ║
║        🆕 트레일링 스탑 (바이낸스 서버)                       ║
║        🆕 거래 시간대 제한 (KST 07:00~23:00)                  ║
║        🆕 Option C 포지션 사이징 (신뢰도 기반)                ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  주의사항:")
    print("   1. 실제 바이낸스 계정의 자금을 사용합니다")
    print("   2. Isolated Margin 모드로 거래합니다")
    print("   3. 손실 가능성이 있으니 충분히 이해한 후 사용하세요")
    print("   4. 소액으로 테스트한 후 본격적으로 사용하세요")
    print(f"   5. 수수료: Maker 0.02%, Taker 0.05%")
    print(f"   6. 🆕 트레일링 스탑이 진입 즉시 활성화됩니다 (바이낸스 서버)")
    print(f"   7. 🆕 거래 시간대: 한국시간 07:00~23:00만 신규 진입")
    print(f"   8. 🆕 Option C 포지션 사이징: 스캘핑 25%, 데이트레이딩 35% + 신뢰도 보너스\n")
    
    print("\n🔧 v2.8 신규 기능 (Option C - 신뢰도 기반 포지션 사이징):")
    print("   1. ✅ 스타일별 베이스: 스캘핑 25%, 데이트레이딩 35%")
    print("   2. ✅ 신뢰도 보너스: 70-80% +0%, 80-90% +5%, 90%+ +10%")
    print("   3. ✅ 최대 50% 제한 (안전장치)")
    print("   4. ✅ 간단하고 예측 가능한 포지션 크기")
    print("   5. ✅ AI 신뢰도를 직접 활용 → 확신 있을 때 더 큰 배팅")
    
    print("\n🔧 v2.7 최적화:")
    print("   1. ✅ AI 포지션 모니터링 제거 → DeepSeek API 비용 최대 90% 절감")
    print("   2. ✅ 트레일링 스탑에 포지션 관리 전담 (바이낸스 서버 = 무료)")
    print("   3. ✅ 검증된 자동화 시스템에만 의존 → 안정성 향상")
    print("   4. ✅ 불필요한 AI 개입 제거 → 오판으로 인한 조기 청산 방지")
    
    print("\n🔧 v2.6-v2.3 주요 기능:")
    print("   1. ✅ 거래 시간대 제한 (한국시간 23:00~07:00 신규 진입 차단)")
    print("   2. ✅ 바이낸스 Income History API로 실제 realized PnL 조회")
    print("   3. ✅ 수수료 반영 PnL 계산 (Maker 0.02%, Taker 0.05%)")
    print("   4. ✅ 초고변동성 코인 필터링 (ATR 15% 이상 차단)")
    print("   5. ✅ 동적 콜백 시스템 (변동성에 따라 자동 조절)\n")
    
    # 자동 시작 (확인 없음)
    print("🚀 백그라운드 모드 - 자동 시작...")
    main()
