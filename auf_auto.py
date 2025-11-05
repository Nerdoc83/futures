"""
AI Live Trading Bot v1.0 (실거래 버전)
----------------------------------------------------------------------
⚠️ 실제 바이낸스 선물 거래 - 실제 자금 사용
- 실시간 바이낸스 선물 데이터 사용
- 실제 잔고로 거래 (Available Balance 기준)
- Isolated Margin 모드
- 실제 주문 체결 및 청산
- AI 보수적 리스크 관리

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
import threading  # 🆕 트레일링 스탑용

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
    
    # 🆕 바이낸스 트레일링 스탑 설정
    "BINANCE_TRAILING_STOP_ENABLED": True,      # 바이낸스 네이티브 트레일링 스탑 사용
    "TRAILING_STOP_ACTIVATION": 20,             # 20% 이익부터 트레일링 스탑 시작
    "TRAILING_STOP_CALLBACK_RATE": 7,           # 콜백 7% (최고가 대비 -7% 하락 시 청산)
    "TRAILING_STOP_CHECK_INTERVAL": 300,        # 5분마다 체크
    
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
        print(f"   💰 Available Balance: ${available:,.2f} USDT")
        return available
    except Exception as e:
        print(f"❌ 잔고 조회 오류: {e}")
        return 0

def set_leverage_and_margin_mode(symbol: str, leverage: int):
    """레버리지 및 마진 모드 설정 (Isolated)"""
    try:
        # 마진 모드 설정 (Isolated)
        margin_mode = LIVE_TRADING_CONFIG["MARGIN_MODE"]
        try:
            exchange.set_margin_mode(margin_mode, symbol)
            print(f"   ✅ 마진 모드: {margin_mode.upper()}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'no need to change' in error_msg or 'already' in error_msg:
                print(f"   ✅ 마진 모드: {margin_mode.upper()} (이미 설정됨)")
            else:
                print(f"   ⚠️ 마진 모드 설정 오류: {e}")
        
        # 레버리지 설정
        exchange.set_leverage(leverage, symbol)
        print(f"   ✅ 레버리지: {leverage}x")
        
    except Exception as e:
        print(f"❌ 레버리지/마진 설정 오류: {e}")
        raise

def fetch_futures_market_indicators(symbol: str) -> Dict:
    """
    바이낸스 선물 특화 지표 수집
    1. 펀딩 비율 (Funding Rate)
    2. 미결제 약정 (Open Interest)
    """
    indicators = {
        'funding_rate': None,
        'funding_rate_status': 'neutral',
        'open_interest': None,
        'oi_status': 'neutral',
    }
    
    try:
        # 1. 펀딩 비율 (Funding Rate) - CCXT 표준 메서드 사용
        try:
            # fetch_funding_rate 사용 (CCXT 표준)
            funding_info = exchange.fetch_funding_rate(symbol)
            if funding_info and 'fundingRate' in funding_info:
                funding_rate = float(funding_info['fundingRate']) * 100  # 퍼센트로 변환
                indicators['funding_rate'] = round(funding_rate, 4)
                
                # 펀딩비 상태 분류
                if funding_rate > 0.05:
                    indicators['funding_rate_status'] = 'long_overheated'  # 롱 과열
                elif funding_rate < -0.05:
                    indicators['funding_rate_status'] = 'short_overheated'  # 숏 과열
                elif funding_rate > 0.01:
                    indicators['funding_rate_status'] = 'bullish'
                elif funding_rate < -0.01:
                    indicators['funding_rate_status'] = 'bearish'
            
        except Exception as e:
            print(f"      ⚠️ 펀딩비 조회 실패: {e}")
        
        # 2. 미결제 약정 (Open Interest) - CCXT 표준 메서드 사용
        try:
            # fetch_open_interest 사용 (CCXT 표준)
            oi_data = exchange.fetch_open_interest(symbol)
            if oi_data and 'openInterestAmount' in oi_data:
                oi_value = float(oi_data['openInterestAmount'])
                indicators['open_interest'] = oi_value
                indicators['oi_status'] = 'high' if oi_value > 0 else 'neutral'
            
        except Exception as e:
            print(f"      ⚠️ OI 조회 실패: {e}")
        
        return indicators
        
    except Exception as e:
        print(f"   ❌ 선물 지표 조회 오류: {e}")
        return indicators

def create_market_order(symbol: str, side: str, amount: float, leverage: int) -> Dict:
    """실제 시장가 주문 생성"""
    try:
        print(f"   📤 주문 전송 중: {side.upper()} {amount:.8f} {symbol}")
        
        # 레버리지 및 마진 모드 설정
        set_leverage_and_margin_mode(symbol, leverage)
        
        # 시장가 주문
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=amount,
            params={'reduceOnly': False}
        )
        
        print(f"   ✅ 주문 체결: {order['id']}")
        print(f"   ├─ 체결가: ${order.get('average', order.get('price', 0)):,.4f}")
        print(f"   ├─ 수량: {order['filled']}")
        print(f"   └─ 상태: {order['status']}")
        
        return order
    except Exception as e:
        print(f"❌ 주문 생성 오류: {e}")
        raise

def create_stop_loss_order(symbol: str, side: str, amount: float, stop_price: float) -> Dict:
    """실제 손절 주문 생성"""
    try:
        order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=side,
            amount=amount,
            params={
                'stopPrice': stop_price,
                'reduceOnly': True
            }
        )
        print(f"   ✅ 손절 주문: ${stop_price:,.4f}")
        return order
    except Exception as e:
        print(f"❌ 손절 주문 오류: {e}")
        raise

def create_take_profit_order(symbol: str, side: str, amount: float, stop_price: float) -> Dict:
    """실제 익절 주문 생성"""
    try:
        order = exchange.create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=side,
            amount=amount,
            params={
                'stopPrice': stop_price,
                'reduceOnly': True
            }
        )
        print(f"   ✅ 익절 주문: ${stop_price:,.4f}")
        return order
    except Exception as e:
        print(f"❌ 익절 주문 오류: {e}")
        raise

def cancel_all_tpsl_orders(symbol: str) -> int:
    """특정 코인의 모든 TP/SL 주문 취소 (청산 감지 시 즉시 호출)"""
    try:
        print(f"   🗑️ {symbol} TP/SL 주문 취소 중...")
        open_orders = exchange.fetch_open_orders(symbol)
        
        cancelled_count = 0
        for order in open_orders:
            order_type = order.get('type', '').upper()
            # STOP_MARKET (손절) 또는 TAKE_PROFIT_MARKET (익절) 주문만 취소
            if 'STOP' in order_type or 'TAKE_PROFIT' in order_type:
                try:
                    exchange.cancel_order(order['id'], symbol)
                    cancelled_count += 1
                    print(f"      ├─ 취소: {order_type} 주문 ID {order['id']}")
                except Exception as e:
                    print(f"      ├─ 취소 실패: {order['id']} ({e})")
        
        if cancelled_count > 0:
            print(f"   ✅ {cancelled_count}개 TP/SL 주문 취소 완료")
        else:
            print(f"   ℹ️ 취소할 TP/SL 주문 없음")
        
        return cancelled_count
    except Exception as e:
        print(f"   ⚠️ TP/SL 주문 취소 중 오류: {e}")
        return 0

def close_position(symbol: str, side: str, amount: float) -> Dict:
    """포지션 청산 (실제 청산 결과 반환) - 안전 버전"""
    try:
        # 기존 주문 모두 취소
        cancel_all_orders(symbol)
        time.sleep(0.5)
        
        # 🆕 실제 포지션 재확인 (청산되었을 수 있음)
        positions = exchange.fetch_positions([symbol])
        actual_position = None
        
        for pos in positions:
            if pos['symbol'] == symbol:
                contracts = pos.get('contracts', 0)
                if contracts is None:
                    contracts = 0
                contracts = float(contracts)
                
                if contracts != 0:
                    actual_position = pos
                    break
        
        # 포지션이 이미 청산됨 (TP/SL에 걸렸을 가능성)
        if not actual_position:
            print(f"   ⚠️ 포지션이 이미 청산되었습니다 (TP/SL 체결?)")
            
            # 🆕 이미 청산된 경우에도 TP/SL 주문 취소
            cancel_all_tpsl_orders(symbol)
            
            # 실현 PnL 조회
            time.sleep(1)
            pnl_info = get_position_pnl(symbol)
            
            return {
                'order': None,
                'pnl': pnl_info.get('realizedPnl', 0),
                'close_price': 0
            }
        
        # 🆕 실제 포지션 수량 사용 (정확한 수량)
        actual_amount = abs(float(actual_position.get('contracts', 0)))
        actual_side = actual_position.get('side', side)
        
        print(f"   📊 실제 포지션: {actual_side.upper()} {actual_amount:.8f}")
        
        # 반대 방향으로 청산
        close_side = 'sell' if actual_side == 'long' else 'buy'
        
        print(f"   🔄 포지션 청산 중: {symbol} ({close_side.upper()} {actual_amount:.8f})")
        
        # 🆕 reduceOnly 대신 positionSide 사용 (Hedge 모드 대응)
        try:
            # 시도 1: reduceOnly 사용
            order = exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=actual_amount,
                params={'reduceOnly': True}
            )
        except Exception as e1:
            error_msg = str(e1).lower()
            if 'reduconly' in error_msg or '-2022' in error_msg:
                print(f"   ⚠️ reduceOnly 실패, 일반 주문 시도...")
                # 시도 2: 일반 시장가 주문 (reduceOnly 없이)
                try:
                    order = exchange.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=actual_amount
                    )
                except Exception as e2:
                    print(f"   ⚠️ 일반 주문도 실패, closePosition API 시도...")
                    # 시도 3: Binance closePosition API
                    order = exchange.fapiPrivatePostOrder({
                        'symbol': symbol.replace('/', '').replace(':USDT', ''),
                        'side': close_side.upper(),
                        'type': 'MARKET',
                        'quantity': actual_amount,
                        'closePosition': 'true'  # 전체 포지션 청산
                    })
            else:
                raise e1
        
        print(f"   ✅ 청산 완료: {order.get('id', 'N/A')}")
        print(f"   ├─ 청산가: ${order.get('average', order.get('price', 0)):,.4f}")
        print(f"   └─ 수량: {order.get('filled', actual_amount)}")
        
        # 🆕 청산 성공 시 TP/SL 주문 취소
        cancel_all_tpsl_orders(symbol)
        
        # 실제 청산 후 PnL 조회
        time.sleep(1)
        pnl_info = get_position_pnl(symbol)
        
        return {
            'order': order,
            'pnl': pnl_info.get('realizedPnl', 0),
            'close_price': order.get('average', order.get('price', 0))
        }
        
    except Exception as e:
        print(f"❌ 청산 오류: {e}")
        
        # 🆕 오류 발생 시에도 PnL은 조회 시도
        try:
            time.sleep(1)
            pnl_info = get_position_pnl(symbol)
            if pnl_info.get('realizedPnl', 0) != 0:
                print(f"   ℹ️ 실현 PnL 확인됨: ${pnl_info['realizedPnl']:,.2f}")
                return {
                    'order': None,
                    'pnl': pnl_info['realizedPnl'],
                    'close_price': 0
                }
        except:
            pass
        
        raise

def get_binance_income_history(days: int = 7) -> List[Dict]:
    """바이낸스 선물 수익 내역 조회 (봇 시작 이후만)"""
    try:
        # 🆕 봇 시작 시점 확인 (DB 첫 거래 시점)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT MIN(timestamp) FROM trades")
        first_trade = c.fetchone()[0]
        conn.close()
        
        if first_trade:
            # 첫 거래 시점부터 조회 (DB의 timestamp는 UTC)
            start_time = int(datetime.strptime(first_trade, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
            print(f"   🔍 봇 시작 이후 수익 내역 조회 중... (첫 거래: {first_trade} UTC)")
        else:
            # DB에 거래가 없으면 최근 N일
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            print(f"   🔍 바이낸스 수익 내역 조회 중... (최근 {days}일)")
        
        end_time = int(time.time() * 1000)  # 현재 시간 (밀리초)
        
        # 🆕 여러 방법으로 바이낸스 API 호출 시도
        income_data = None
        
        try:
            # 방법 1: fapiPrivate_get_income 시도
            print(f"   📊 방법 1: fapiPrivate_get_income 시도...")
            income_data = exchange.fapiPrivate_get_income({
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            })
        except (AttributeError, Exception) as e1:
            print(f"   ⚠️ 방법 1 실패: {e1}")
            
            try:
                # 방법 2: fapiPrivateGetIncome 시도
                print(f"   📊 방법 2: fapiPrivateGetIncome 시도...")
                income_data = exchange.fapiPrivateGetIncome({
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                })
            except (AttributeError, Exception) as e2:
                print(f"   ⚠️ 방법 2 실패: {e2}")
                
                try:
                    # 방법 3: private_post 시도
                    print(f"   📊 방법 3: private_post 시도...")
                    income_data = exchange.private_post('fapi/v1/income', {
                        'startTime': start_time,
                        'endTime': end_time,
                        'limit': 1000
                    })
                except (AttributeError, Exception) as e3:
                    print(f"   ⚠️ 방법 3 실패: {e3}")
                    print(f"   ❌ 바이낸스 API 모든 방법 실패 - DB 데이터만 사용")
                    return []
        
        # income_data가 성공적으로 조회된 경우
        if income_data is not None:
            # 실현 손익만 필터링 (REALIZED_PNL)
            realized_pnl_data = []
            for item in income_data:
                if item.get('incomeType') == 'REALIZED_PNL':
                    realized_pnl_data.append({
                        'symbol': item.get('symbol', ''),
                        'income': float(item.get('income', 0)),
                        'asset': item.get('asset', 'USDT'),
                        'time': int(item.get('time', 0)),
                        'timestamp': datetime.fromtimestamp(int(item.get('time', 0)) / 1000, tz=timezone.utc),
                        'tranId': item.get('tranId', ''),
                        'tradeId': item.get('tradeId', '')
                    })
            
            print(f"   ✅ 실현 손익 내역: {len(realized_pnl_data)}건 (봇 시작 이후)")
            return realized_pnl_data
        
        return []
        
    except Exception as e:
        print(f"   ❌ 바이낸스 수익 내역 조회 오류: {e}")
        # 디버깅: 사용 가능한 메서드 확인
        try:
            available_methods = [method for method in dir(exchange) if 'income' in method.lower() or 'fapi' in method.lower()]
            print(f"   🔍 사용 가능한 메서드: {available_methods[:5]}...")
        except:
            pass
        return []

def get_bot_trades_with_binance_pnl(days: int = 7) -> List[Dict]:
    """DB 봇 거래를 기반으로 바이낸스에서 실제 PnL 매칭"""
    try:
        # 1. DB에서 완료된 봇 거래 조회
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        cutoff_date = (get_utc_now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            SELECT coin_symbol, action, pnl, binance_pnl, entry_price, exit_price,
                   close_timestamp, confidence_score, ai_reasoning, reasoning, id
            FROM trades 
            WHERE status = 'CLOSED' 
            AND close_timestamp >= ?
            ORDER BY close_timestamp DESC
        ''', (cutoff_date,))
        
        db_trades = c.fetchall()
        conn.close()
        
        if not db_trades:
            print(f"   ⚠️ DB에 완료된 봇 거래 없음 (최근 {days}일)")
            return []
        
        print(f"   📊 DB에서 완료된 봇 거래 {len(db_trades)}건 조회")
        
        # 2. 바이낸스에서 실제 PnL 데이터 조회 (시간 범위)
        try:
            # 첫 거래와 마지막 거래 시간 찾기
            first_trade_time = min(trade[6] for trade in db_trades if trade[6])  # close_timestamp
            last_trade_time = max(trade[6] for trade in db_trades if trade[6])
            
            start_time = int(datetime.strptime(first_trade_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
            end_time = int(datetime.strptime(last_trade_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000  # +1일
            
            print(f"   🔍 바이낸스 PnL 조회 중... ({first_trade_time} ~ {last_trade_time} UTC)")
            
            # 바이낸스 API 호출
            binance_pnl_data = None
            try:
                # 방법 2가 성공했으므로 바로 시도
                binance_pnl_data = exchange.fapiPrivateGetIncome({
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                })
                print(f"   ✅ 바이낸스 PnL 데이터 조회 성공")
            except Exception as e:
                print(f"   ⚠️ 바이낸스 PnL 조회 실패: {e}")
                binance_pnl_data = []
            
            # REALIZED_PNL만 필터링
            realized_pnl_map = {}
            if binance_pnl_data:
                for item in binance_pnl_data:
                    if item.get('incomeType') == 'REALIZED_PNL':
                        symbol = item.get('symbol', '')
                        income = float(item.get('income', 0))
                        time_key = int(item.get('time', 0))
                        
                        # 심볼별로 그룹화
                        if symbol not in realized_pnl_map:
                            realized_pnl_map[symbol] = []
                        realized_pnl_map[symbol].append({
                            'income': income,
                            'time': time_key,
                            'timestamp': datetime.fromtimestamp(time_key / 1000, tz=timezone.utc)
                        })
                
                print(f"   📊 바이낸스 실현 손익: {len(realized_pnl_map)}개 심볼")
        
        except Exception as e:
            print(f"   ⚠️ 바이낸스 데이터 조회 실패: {e}")
            realized_pnl_map = {}
        
        # 3. DB 거래와 바이낸스 PnL 매칭
        matched_trades = []
        
        for trade in db_trades:
            coin, action, db_pnl, binance_pnl, entry_price, exit_price, close_time, confidence, ai_reasoning, reasoning, trade_id = trade
            
            # 바이낸스 PnL이 이미 있으면 사용
            if binance_pnl is not None and binance_pnl != 0:
                actual_pnl = binance_pnl
                data_source = "DB저장"
            else:
                # 바이낸스에서 매칭 시도
                symbol_binance = f"{coin}USDT"
                actual_pnl = db_pnl if db_pnl else 0  # 기본값
                data_source = "DB계산"
                
                if symbol_binance in realized_pnl_map and close_time:
                    try:
                        close_dt = datetime.strptime(close_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        
                        # 청산 시간 ±30분 내의 바이낸스 PnL 찾기
                        for binance_item in realized_pnl_map[symbol_binance]:
                            time_diff = abs((binance_item['timestamp'] - close_dt).total_seconds())
                            if time_diff <= 1800:  # 30분 이내
                                actual_pnl = binance_item['income']
                                data_source = "바이낸스매칭"
                                break
                    except:
                        pass
            
            if close_time:
                try:
                    timestamp_dt = datetime.strptime(close_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    
                    matched_trades.append({
                        'symbol': f"{coin}USDT",
                        'income': float(actual_pnl),
                        'asset': 'USDT',
                        'time': int(timestamp_dt.timestamp() * 1000),
                        'timestamp': timestamp_dt,
                        'tranId': f"bot_trade_{trade_id}",
                        'tradeId': f"bot_trade_{trade_id}",
                        'action': action,
                        'confidence': confidence if confidence else 50,
                        'ai_reasoning': ai_reasoning or 'N/A',
                        'reasoning': reasoning or 'N/A',
                        'data_source': data_source
                    })
                except:
                    continue
        
        # 데이터 소스 요약 출력
        source_count = {}
        for trade in matched_trades:
            source = trade['data_source']
            source_count[source] = source_count.get(source, 0) + 1
        
        print(f"   ✅ PnL 매칭 완료: {len(matched_trades)}건")
        for source, count in source_count.items():
            print(f"   ├─ {source}: {count}건")
        
        return matched_trades
        
    except Exception as e:
        print(f"   ❌ DB-바이낸스 PnL 매칭 오류: {e}")
        return []

def calculate_binance_performance(days: int = 7) -> Dict:
    """봇 거래 성과 계산 (DB-바이낸스 PnL 매칭)"""
    
    # 🆕 DB 거래 기반 바이낸스 PnL 매칭
    try:
        income_data = get_bot_trades_with_binance_pnl(days)
        if income_data:
            data_source = "DB-바이낸스 매칭"
        else:
            # 매칭 실패 시 기존 DB 데이터만 사용
            print(f"   ⚠️ 바이낸스 매칭 실패, DB 데이터만 사용")
            income_data = []
            # DB에서 직접 조회
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            cutoff_date = (get_utc_now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            c.execute('''
                SELECT coin_symbol, pnl, close_timestamp FROM trades 
                WHERE status = 'CLOSED' AND close_timestamp >= ?
            ''', (cutoff_date,))
            trades = c.fetchall()
            conn.close()
            
            for coin, pnl, close_time in trades:
                if pnl and close_time:
                    try:
                        timestamp_dt = datetime.strptime(close_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        income_data.append({
                            'symbol': f"{coin}USDT",
                            'income': float(pnl),
                            'timestamp': timestamp_dt
                        })
                    except:
                        continue
            data_source = "DB 전용"
    except:
        income_data = []
        data_source = "N/A"
    
    if not income_data:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'income_data': [],
            'data_source': data_source
        }
    
    # 손익 분석
    total_pnl = sum(item['income'] for item in income_data)
    winning_trades = [item for item in income_data if item['income'] > 0]
    losing_trades = [item for item in income_data if item['income'] < 0]
    
    total_trades = len(income_data)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = sum(item['income'] for item in winning_trades) / win_count if win_count > 0 else 0
    avg_loss = sum(item['income'] for item in losing_trades) / loss_count if loss_count > 0 else 0
    
    print(f"   📊 {data_source} 성과 분석:")
    print(f"   ├─ 총 거래: {total_trades}건")
    print(f"   ├─ 수익 거래: {win_count}건 (평균: ${avg_win:,.2f})")
    print(f"   ├─ 손실 거래: {loss_count}건 (평균: ${avg_loss:,.2f})")
    print(f"   ├─ 승률: {win_rate:.1f}%")
    print(f"   └─ 총 손익: ${total_pnl:,.2f}")
    
    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': abs(avg_loss),  # 절댓값으로 표시
        'income_data': income_data,
        'data_source': data_source
    }
    """포지션의 실현/미실현 손익 조회"""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = pos.get('contracts', 0)
            if contracts is None:
                contracts = 0
            contracts = float(contracts)
            
            if pos['symbol'] == symbol and contracts != 0:
                unrealized_pnl = pos.get('unrealizedPnl', 0)
                realized_pnl = pos.get('realizedPnl', 0)
                percentage = pos.get('percentage', 0)
                
                return {
                    'unrealizedPnl': float(unrealized_pnl) if unrealized_pnl is not None else 0,
                    'realizedPnl': float(realized_pnl) if realized_pnl is not None else 0,
                    'percentage': float(percentage) if percentage is not None else 0
                }
        
        # 포지션이 없으면 거래 히스토리에서 조회
        trades = exchange.fetch_my_trades(symbol, limit=10)
        if trades:
            latest_trade = trades[-1]
            realized_pnl = latest_trade.get('realizedPnl', 0)
            
            return {
                'unrealizedPnl': 0,
                'realizedPnl': float(realized_pnl) if realized_pnl is not None else 0,
                'percentage': 0
            }
        
        return {'unrealizedPnl': 0, 'realizedPnl': 0, 'percentage': 0}
    except Exception as e:
        print(f"   ⚠️ PnL 조회 오류: {e}")
        return {'unrealizedPnl': 0, 'realizedPnl': 0, 'percentage': 0}

def get_position_pnl(symbol: str) -> Dict:
    """포지션의 실현/미실현 손익 조회"""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = pos.get('contracts', 0)
            if contracts is None:
                contracts = 0
            contracts = float(contracts)
            
            if pos['symbol'] == symbol and contracts != 0:
                unrealized_pnl = pos.get('unrealizedPnl', 0)
                realized_pnl = pos.get('realizedPnl', 0)
                percentage = pos.get('percentage', 0)
                
                return {
                    'unrealizedPnl': float(unrealized_pnl) if unrealized_pnl is not None else 0,
                    'realizedPnl': float(realized_pnl) if realized_pnl is not None else 0,
                    'percentage': float(percentage) if percentage is not None else 0
                }
        
        # 포지션이 없으면 거래 히스토리에서 조회
        trades = exchange.fetch_my_trades(symbol, limit=10)
        if trades:
            latest_trade = trades[-1]
            realized_pnl = latest_trade.get('realizedPnl', 0)
            
            return {
                'unrealizedPnl': 0,
                'realizedPnl': float(realized_pnl) if realized_pnl is not None else 0,
                'percentage': 0
            }
        
        return {'unrealizedPnl': 0, 'realizedPnl': 0, 'percentage': 0}
    except Exception as e:
        print(f"   ⚠️ PnL 조회 오류: {e}")
        return {'unrealizedPnl': 0, 'realizedPnl': 0, 'percentage': 0}

def get_open_positions() -> List[Dict]:
    """현재 오픈된 모든 포지션 조회"""
    try:
        positions = exchange.fetch_positions()
        open_positions = []
        
        for pos in positions:
            contracts = pos.get('contracts', 0)
            if contracts is None:
                contracts = 0
            contracts = float(contracts)
            
            if contracts != 0:
                # None 값 체크 후 float 변환
                entry_price = pos.get('entryPrice', 0)
                mark_price = pos.get('markPrice', 0)
                unrealized_pnl = pos.get('unrealizedPnl', 0)
                percentage = pos.get('percentage', 0)
                leverage = pos.get('leverage', 1)
                notional = pos.get('notional', 0)
                initial_margin = pos.get('initialMargin', 0)
                collateral = pos.get('collateral', 0)
                
                # 실제 레버리지 계산 (notional / initialMargin)
                actual_leverage = leverage  # 기본값
                if initial_margin and float(initial_margin) != 0:
                    actual_leverage = abs(float(notional) / float(initial_margin))
                
                open_positions.append({
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'contracts': contracts,
                    'entryPrice': float(entry_price) if entry_price is not None else 0,
                    'markPrice': float(mark_price) if mark_price is not None else 0,
                    'unrealizedPnl': float(unrealized_pnl) if unrealized_pnl is not None else 0,
                    'percentage': float(percentage) if percentage is not None else 0,
                    'leverage': float(leverage) if leverage is not None else 1,
                    'actualLeverage': float(actual_leverage),  # 🆕 실제 계산된 레버리지
                    'notional': float(notional) if notional is not None else 0,
                    'initialMargin': float(initial_margin) if initial_margin is not None else 0,
                    'collateral': float(collateral) if collateral is not None else 0,
                    'marginMode': pos.get('marginMode', 'isolated')
                })
        
        return open_positions
    except Exception as e:
        print(f"❌ 포지션 조회 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def get_open_orders(symbol: str = None) -> List[Dict]:
    """미체결 주문 조회"""
    try:
        # 경고 메시지 억제
        if not symbol:
            exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
        
        if symbol:
            orders = exchange.fetch_open_orders(symbol)
        else:
            orders = exchange.fetch_open_orders()
        
        open_orders = []
        for order in orders:
            open_orders.append({
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'price': order.get('price', 0),
                'amount': order.get('amount', 0),
                'status': order['status'],
                'info': order.get('info', {})
            })
        
        return open_orders
    except Exception as e:
        # 경고는 무시하고 에러만 표시
        error_msg = str(e)
        if 'WARNING' not in error_msg:
            print(f"⚠️ 미체결 주문 조회 오류: {e}")
        return []

def cancel_all_orders(symbol: str) -> int:
    """특정 심볼의 모든 미체결 주문 취소"""
    try:
        result = exchange.cancel_all_orders(symbol)
        cancelled_count = len(result) if isinstance(result, list) else 0
        return cancelled_count
    except Exception as e:
        print(f"⚠️ 주문 취소 오류: {e}")
        return 0

def sync_db_with_binance():
    """🆕 DB 포지션을 바이낸스 실제 포지션과 동기화 (양방향) - 수동거래 보호"""
    try:
        # 1. DB 오픈 포지션 조회 (manual_trade 컬럼 포함)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            SELECT id, coin_symbol, entry_price, amount, leverage, investment_amount, timestamp, action, sl_price, tp_price, manual_trade
            FROM trades
            WHERE status = 'OPEN'
        ''')
        db_positions = c.fetchall()
        db_coins = {pos[1] for pos in db_positions}  # coin_symbol만 추출
        
        # 2. 바이낸스 실제 포지션 조회
        live_positions = get_open_positions()
        binance_coins = {pos['symbol'].split('/')[0] for pos in live_positions}
        
        synced_count = 0
        
        # 3. DB에만 있고 바이낸스에 없는 포지션 찾기 (청산 처리)
        if db_positions:
            for db_pos in db_positions:
                trade_id, coin, entry_price, amount, leverage, investment, timestamp, action, sl_price, tp_price, manual_trade = db_pos
                
                # 🛡️ 수동거래 보호 - 동기화에서 제외
                if manual_trade == 1:
                    print(f"   🛡️ {coin} 수동거래 보호 - DB 동기화에서 제외")
                    continue
                
                if coin not in binance_coins:
                    # 바이낸스에 없음 → DB에서 청산 처리 (AI 거래만)
                    print(f"   🔄 동기화: {coin} AI거래가 바이낸스에 없음 → DB 청산 처리")
                    
                    # 🆕 바이낸스에서 실제 청산 정보 조회
                    symbol_binance = f"{coin}USDT"
                    actual_pnl = 0
                    binance_pnl = None
                    exit_price = None
                    
                    try:
                        # 진입 시간 이후의 거래 내역 조회
                        if timestamp:
                            start_time = int(parse_db_timestamp(timestamp).timestamp() * 1000)
                            end_time = get_utc_timestamp_ms()
                            
                            # 1. Income History에서 실현 손익 조회
                            income_data = exchange.fapiPrivateGetIncome({
                                'symbol': symbol_binance,
                                'incomeType': 'REALIZED_PNL',
                                'startTime': start_time,
                                'endTime': end_time,
                                'limit': 10
                            })
                            
                            # 가장 최근 실현 손익 찾기
                            if income_data:
                                for item in reversed(income_data):  # 최신 순서
                                    pnl_value = float(item.get('income', 0))
                                    if pnl_value != 0:
                                        binance_pnl = pnl_value
                                        # 🔧 actual_pnl에 바이낸스 값 사용 안 함!
                                        # 바이낸스 realizedPnl은 수수료/펀딩비가 차감된 값
                                        # actual_pnl은 0으로 유지 → 아래에서 정확히 재계산
                                        print(f"     📊 바이낸스 실제 PnL: ${binance_pnl:+,.2f} (수수료/펀딩비 포함)")
                                        break
                            
                            # 2. 🆕 User Trades에서 청산가 조회 (최근 체결 내역)
                            try:
                                trades = exchange.fapiPrivateGetUserTrades({
                                    'symbol': symbol_binance,
                                    'startTime': start_time,
                                    'endTime': end_time,
                                    'limit': 50
                                })
                                
                                # 청산 거래 찾기 (reduceOnly=true 또는 positionSide가 반대)
                                if trades:
                                    for trade in reversed(trades):  # 최신 순서
                                        # realizedPnl이 있고 0이 아닌 거래 = 청산 거래
                                        realized_pnl = float(trade.get('realizedPnl', 0))
                                        if realized_pnl != 0:
                                            exit_price = float(trade.get('price', 0))
                                            print(f"     💰 청산가 발견: ${exit_price:,.4f}")
                                            break
                            except Exception as trade_err:
                                print(f"     ⚠️ 거래 내역 조회 실패: {trade_err}")
                            
                    except Exception as e:
                        print(f"     ⚠️ 바이낸스 데이터 조회 실패: {e}")
                    
                    # 청산가가 없으면 현재가 사용
                    if exit_price is None:
                        try:
                            symbol = f"{coin}/USDT:USDT"
                            ticker = exchange.fetch_ticker(symbol)
                            exit_price = ticker['last']
                            print(f"     📊 현재가 사용: ${exit_price:,.4f}")
                        except:
                            exit_price = entry_price  # 조회 실패 시 진입가 사용
                    
                    # PnL 계산 (바이낸스 조회 실패 시 또는 정확한 계산 필요 시)
                    # 🔧 actual_pnl이 0이면 정확히 재계산 (레버리지 포함)
                    if actual_pnl == 0 and exit_price:
                        if action == 'long':
                            price_change = (exit_price - entry_price) / entry_price
                        else:  # short
                            price_change = (entry_price - exit_price) / entry_price
                        
                        actual_pnl = investment * price_change * leverage
                        print(f"     📊 계산된 PnL: ${actual_pnl:+,.2f} (레버리지 {leverage}x 적용)")
                    
                    pnl_pct = (actual_pnl / investment * 100) if investment > 0 else 0
                    
                    # 🆕 청산 이유 판단 (sl_price, tp_price는 이미 언팩됨)
                    close_reason = "자동 청산"
                    
                    if exit_price and entry_price:
                        price_change_pct = abs((exit_price - entry_price) / entry_price * 100)
                        
                        # TP 도달 확인 (수익 청산 + 가격이 TP 근처)
                        if actual_pnl > 0:
                            if tp_price and tp_price > 0 and abs(exit_price - tp_price) / tp_price < 0.02:  # 2% 이내
                                close_reason = "익절 목표가 도달 (TP)"
                            else:
                                close_reason = "익절 청산"
                        # SL 도달 확인 (손실 청산 + 가격이 SL 근처)
                        elif actual_pnl < 0:
                            if sl_price and sl_price > 0 and abs(exit_price - sl_price) / sl_price < 0.02:  # 2% 이내
                                close_reason = "손절 라인 도달 (SL)"
                            else:
                                close_reason = "손절 청산"
                        else:
                            close_reason = "무손익 청산"
                    
                    # 🆕 청산 감지 즉시 TP/SL 주문 취소
                    symbol = f"{coin}/USDT:USDT"
                    cancel_all_tpsl_orders(symbol)
                    
                    # DB 업데이트
                    c.execute('''
                        UPDATE trades
                        SET status = 'CLOSED',
                            exit_price = ?,
                            pnl = ?,
                            pnl_percentage = ?,
                            binance_pnl = ?,
                            close_timestamp = CURRENT_TIMESTAMP,
                            ai_reasoning = COALESCE(ai_reasoning, '') || ?
                        WHERE id = ?
                    ''', (exit_price, actual_pnl, pnl_pct, binance_pnl, f'\n[청산 이유] {close_reason}', trade_id))
                    
                    synced_count += 1
                    print(f"     ✅ {coin} DB 청산 완료 (청산가: ${exit_price:,.2f}, PnL: ${actual_pnl:+,.2f})")
        
        # 4. 바이낸스에만 있고 DB에 없는 포지션 찾기 (DB에 추가)
        binance_only_coins = binance_coins - db_coins
        if binance_only_coins:
            print(f"\n   🆕 바이낸스 전용 포지션 발견: {len(binance_only_coins)}개")
            for coin in binance_only_coins:
                # 해당 코인의 바이낸스 포지션 정보 찾기
                pos_info = None
                for pos in live_positions:
                    if pos['symbol'].split('/')[0] == coin:
                        pos_info = pos
                        break
                
                if pos_info:
                    print(f"   📥 {coin} 포지션을 DB에 추가 중...")
                    
                    # 실제 레버리지 및 투자금 계산
                    actual_leverage = pos_info.get('actualLeverage', pos_info.get('leverage', 1))
                    initial_margin = pos_info.get('initialMargin', 0)
                    
                    # 투자금: initialMargin이 있으면 사용, 없으면 notional/leverage로 계산
                    if initial_margin and initial_margin > 0:
                        investment_amount = initial_margin
                    else:
                        investment_amount = abs(pos_info['notional']) / actual_leverage if actual_leverage > 0 else abs(pos_info['notional'])
                    
                    # 수동 진입 포지션으로 DB에 추가
                    c.execute('''
                        INSERT INTO trades (
                            coin_symbol, action, entry_price, amount, leverage,
                            investment_amount, sl_price, tp_price, 
                            trading_style, ai_reasoning, status, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        coin,
                        pos_info['side'],
                        pos_info['entryPrice'],
                        pos_info['contracts'],
                        round(actual_leverage, 1),  # 실제 레버리지 (소수점 1자리)
                        investment_amount,
                        None,  # sl_price
                        None,  # tp_price
                        'MANUAL',
                        '[자동동기화: 바이낸스 수동 진입 포지션]',
                        'OPEN'
                    ))
                    
                    synced_count += 1
                    print(f"     ✅ {coin} DB 추가 완료")
                    print(f"        진입가: ${pos_info['entryPrice']:,.2f}")
                    print(f"        레버리지: {actual_leverage:.1f}x")
                    print(f"        투자금: ${investment_amount:.2f}")
        
        conn.commit()
        conn.close()
        
        if synced_count > 0:
            print(f"   ✅ 양방향 동기화 완료: {synced_count}개 포지션 처리")
        
        return synced_count
        
    except Exception as e:
        print(f"   ❌ DB 동기화 오류: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.close()
        return 0


def cleanup_orphaned_orders():
    """🆕 포지션 없는 TP/SL 주문 정리 - 간소화 버전"""
    try:
        print(f"\n{'='*70}")
        print(f"🧹 고아 TP/SL 주문 정리 시작")
        print(f"{'='*70}")
        
        # 1. 현재 오픈 포지션 조회
        open_positions = get_open_positions()
        position_symbols = {pos['symbol'] for pos in open_positions}
        
        print(f"   오픈 포지션: {len(position_symbols)}개")
        
        # 2. DB에서 거래한 심볼 조회 (최근 30일)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        cutoff = (get_utc_now() - timedelta(days=30)).strftime('%Y-%m-%d')
        c.execute("""
            SELECT DISTINCT coin_symbol FROM trades 
            WHERE timestamp >= ?
        """, (cutoff,))
        traded_coins = [row[0] for row in c.fetchall()]
        conn.close()
        
        print(f"   최근 거래 심볼: {len(traded_coins)}개")
        
        # 3. 각 심볼별 TP/SL 주문 확인 및 정리
        orphaned_count = 0
        for coin in traded_coins:
            symbol = f"{coin}/USDT:USDT"
            
            # 포지션 있으면 스킵
            if symbol in position_symbols:
                continue
            
            try:
                # 이 심볼의 미체결 주문 조회
                orders = get_open_orders(symbol)
                if not orders:
                    continue
                
                # TP/SL 주문만 필터링
                tp_sl_orders = [
                    order for order in orders 
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 
                                          'STOP_LOSS_MARKET', 'TAKE_PROFIT']
                ]
                
                if tp_sl_orders:
                    print(f"\n   🗑️ {coin}: TP/SL 주문 {len(tp_sl_orders)}개 발견")
                    
                    # 각 주문 상세 출력
                    for order in tp_sl_orders:
                        print(f"      - {order['type']} @ ${order.get('price', 0):,.2f}")
                    
                    # TP/SL 주문만 취소
                    cancelled = cancel_all_tpsl_orders(symbol)
                    orphaned_count += cancelled
                    
            except Exception as e:
                # 심볼별 오류는 스킵 (다음 심볼 계속 처리)
                if 'does not have market symbol' not in str(e):
                    print(f"   ⚠️ {coin} 처리 실패: {e}")
                continue
        
        if orphaned_count > 0:
            print(f"\n   ✅ 고아 주문 정리 완료: {orphaned_count}개 취소")
        else:
            print(f"   ✅ 고아 주문 없음")
        
        print(f"{'='*70}\n")
        
        return orphaned_count
        
    except Exception as e:
        print(f"❌ 고아 주문 정리 오류: {e}")
        import traceback
        traceback.print_exc()
        return 0


# ===== 데이터베이스 함수 =====

def setup_database():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 거래 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            amount REAL,
            leverage INTEGER,
            investment_amount REAL,
            pnl REAL,
            pnl_percentage REAL,
            sl_price REAL,
            tp_price REAL,
            trading_style TEXT,
            holding_time_estimate TEXT,
            status TEXT DEFAULT 'OPEN',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            close_timestamp DATETIME,
            ai_reasoning TEXT,
            market_conditions TEXT,
            binance_order_id TEXT,
            binance_pnl REAL,
            binance_close_price REAL,
            confidence_score INTEGER,
            reasoning TEXT,
            pattern_description TEXT,
            manual_trade INTEGER DEFAULT 0
        )
    ''')
    
    # 🆕 기존 DB에 manual_trade 컬럼 추가 (마이그레이션)
    try:
        c.execute("ALTER TABLE trades ADD COLUMN manual_trade INTEGER DEFAULT 0")
        print("✅ manual_trade 컬럼이 추가되었습니다.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            # 컬럼이 이미 존재함
            pass
        else:
            print(f"⚠️ DB 마이그레이션 오류: {e}")
    
    # 🆕 트레일링 스탑용 highest_price 컬럼 추가
    try:
        c.execute("ALTER TABLE trades ADD COLUMN highest_price REAL DEFAULT 0")
        print("✅ highest_price 컬럼이 추가되었습니다 (트레일링 스탑용).")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass
        else:
            print(f"⚠️ highest_price 컬럼 추가 오류: {e}")
    
    # AI 결정 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS ai_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_symbol TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            direction TEXT,
            leverage INTEGER,
            investment_percentage REAL,
            sl_percentage REAL,
            tp_percentage REAL,
            confidence_score INTEGER,
            reasoning TEXT,
            market_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            related_trade_id INTEGER,
            FOREIGN KEY (related_trade_id) REFERENCES trades(id)
        )
    ''')
    
    # 성과 리뷰 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS performance_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            total_pnl REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            current_balance REAL,
            ai_feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 🆕 전략별 성과 추적 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trading_style TEXT NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            avg_win REAL DEFAULT 0,
            avg_loss REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            consecutive_losses INTEGER DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(trading_style)
        )
    ''')
    
    # 🆕 성공/실패 패턴 학습 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS pattern_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_description TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0,
            avg_pnl REAL DEFAULT 0,
            confidence_level INTEGER DEFAULT 50,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            market_conditions TEXT,
            UNIQUE(pattern_description)
        )
    ''')
    
    # 🆕 리스크 조정 히스토리 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS risk_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            adjustment_type TEXT NOT NULL,
            previous_value REAL,
            new_value REAL,
            reason TEXT,
            consecutive_losses INTEGER,
            current_drawdown REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 초기화 완료 (강화학습 테이블 포함)")

def reset_database():
    """데이터베이스 완전 초기화"""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"✅ 기존 DB 삭제: {DB_FILE}")
    setup_database()

def save_trade_to_db(trade_data: Dict) -> int:
    """거래 저장"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO trades (
            coin_symbol, action, entry_price, amount, leverage,
            investment_amount, sl_price, tp_price, trading_style,
            holding_time_estimate, ai_reasoning, market_conditions, binance_order_id,
            confidence_score, manual_trade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data['coin_symbol'],
        trade_data['action'],
        trade_data['entry_price'],
        trade_data['amount'],
        trade_data['leverage'],
        trade_data['investment_amount'],
        trade_data['sl_price'],
        trade_data['tp_price'],
        trade_data.get('trading_style', 'DAY_TRADING'),
        trade_data.get('holding_time_estimate', 'Unknown'),
        trade_data['ai_reasoning'],
        trade_data.get('market_conditions', ''),
        trade_data.get('binance_order_id', ''),
        trade_data.get('confidence_score', 0),
        trade_data.get('manual_trade', 0)
    ))
    
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id

def register_manual_trade(coin_symbol: str, action: str, entry_price: float, amount: float, leverage: int = 1) -> int:
    """
    수동거래를 시스템에 등록
    - coin_symbol: 코인 심볼 (예: "BTC")
    - action: "LONG" 또는 "SHORT"
    - entry_price: 진입 가격
    - amount: 수량
    - leverage: 레버리지 (기본값 1)
    """
    print(f"\n{'='*60}")
    print(f"📝 수동거래 등록")
    print(f"{'='*60}")
    print(f"   코인: {coin_symbol}")
    print(f"   방향: {action}")
    print(f"   진입가: ${entry_price:,.4f}")
    print(f"   수량: {amount}")
    print(f"   레버리지: {leverage}x")
    
    trade_data = {
        'coin_symbol': coin_symbol,
        'action': action,
        'entry_price': entry_price,
        'amount': amount,
        'leverage': leverage,
        'investment_amount': entry_price * amount,
        'sl_price': 0,  # 수동거래는 초기 SL/TP 없음
        'tp_price': 0,
        'trading_style': 'MANUAL',
        'holding_time_estimate': 'Manual Control',
        'ai_reasoning': f'수동거래: {action} {coin_symbol} @ ${entry_price:,.4f}',
        'market_conditions': 'Manual Entry',
        'binance_order_id': '',
        'confidence_score': 100,  # 수동거래는 100% 신뢰도
        'manual_trade': 1  # 🔧 수동거래 플래그
    }
    
    trade_id = save_trade_to_db(trade_data)
    print(f"   ✅ 수동거래 등록 완료 (ID: {trade_id})")
    print(f"   🛡️ AI 자동 청산 방지 설정됨")
    print(f"{'='*60}")
    
    return trade_id

def mark_existing_trade_as_manual(trade_id: int) -> bool:
    """
    기존 거래를 수동거래로 표시 (AI 청산 방지)
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # 거래 존재 확인
        c.execute("SELECT coin_symbol, action FROM trades WHERE id = ? AND status = 'OPEN'", (trade_id,))
        result = c.fetchone()
        
        if not result:
            print(f"❌ 거래 ID {trade_id}를 찾을 수 없거나 이미 종료되었습니다.")
            conn.close()
            return False
        
        coin_symbol, action = result
        
        # manual_trade 플래그 설정
        c.execute("UPDATE trades SET manual_trade = 1 WHERE id = ?", (trade_id,))
        conn.commit()
        conn.close()
        
        print(f"✅ 거래 ID {trade_id} ({coin_symbol} {action})를 수동거래로 설정했습니다.")
        print(f"   🛡️ AI 자동 청산이 비활성화됩니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 수동거래 설정 오류: {e}")
        return False

def unmark_manual_trade(trade_id: int) -> bool:
    """
    수동거래 플래그 해제 (AI 청산 활성화)
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # 거래 존재 확인
        c.execute("SELECT coin_symbol, action FROM trades WHERE id = ? AND status = 'OPEN'", (trade_id,))
        result = c.fetchone()
        
        if not result:
            print(f"❌ 거래 ID {trade_id}를 찾을 수 없거나 이미 종료되었습니다.")
            conn.close()
            return False
        
        coin_symbol, action = result
        
        # manual_trade 플래그 해제
        c.execute("UPDATE trades SET manual_trade = 0 WHERE id = ?", (trade_id,))
        conn.commit()
        conn.close()
        
        print(f"✅ 거래 ID {trade_id} ({coin_symbol} {action})의 수동거래 플래그를 해제했습니다.")
        print(f"   🤖 AI 자동 청산이 활성화됩니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 수동거래 해제 오류: {e}")
        return False

def update_trade_close(trade_id: int, close_data: Dict):
    """거래 청산 정보 업데이트 (바이낸스 실제 결과 사용 + AI 청산 이유 추가)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 🆕 AI 청산 이유가 있으면 ai_reasoning에 추가
    if close_data.get('ai_close_reason'):
        c.execute('''
            UPDATE trades SET
                exit_price = ?,
                pnl = ?,
                pnl_percentage = ?,
                status = 'CLOSED',
                close_timestamp = CURRENT_TIMESTAMP,
                binance_pnl = ?,
                binance_close_price = ?,
                ai_reasoning = COALESCE(ai_reasoning, '') || ?
            WHERE id = ?
        ''', (
            close_data.get('close_price', 0),
            close_data.get('pnl', 0),
            close_data.get('pnl_percentage', 0),
            close_data.get('binance_pnl', 0),
            close_data.get('binance_close_price', 0),
            f"\n[청산 이유] {close_data.get('ai_close_reason')}",
            trade_id
        ))
    else:
        # AI 이유 없으면 기존 방식
        c.execute('''
            UPDATE trades SET
                exit_price = ?,
                pnl = ?,
                pnl_percentage = ?,
                status = 'CLOSED',
                close_timestamp = CURRENT_TIMESTAMP,
                binance_pnl = ?,
                binance_close_price = ?
            WHERE id = ?
        ''', (
            close_data.get('close_price', 0),
            close_data.get('pnl', 0),
            close_data.get('pnl_percentage', 0),
            close_data.get('binance_pnl', 0),
            close_data.get('binance_close_price', 0),
            trade_id
        ))
    
    conn.commit()
    conn.close()

# ===== 🆕 전략 성과 추적 함수 =====

def update_strategy_performance(trading_style: str, pnl: float, is_win: bool):
    """전략별 성과 업데이트"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 기존 데이터 조회
    c.execute('SELECT * FROM strategy_performance WHERE trading_style = ?', (trading_style,))
    existing = c.fetchone()
    
    if existing:
        # 업데이트
        total_trades = existing[2] + 1
        winning_trades = existing[3] + (1 if is_win else 0)
        losing_trades = existing[4] + (0 if is_win else 1)
        total_pnl = existing[5] + pnl
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 평균 계산
        if is_win and winning_trades > 0:
            avg_win = (existing[7] * (winning_trades - 1) + pnl) / winning_trades
        else:
            avg_win = existing[7]
        
        if not is_win and losing_trades > 0:
            avg_loss = (existing[8] * (losing_trades - 1) + pnl) / losing_trades
        else:
            avg_loss = existing[8]
        
        # Sharpe Ratio 간단 계산
        if abs(avg_win - avg_loss) > 0:
            avg_return = total_pnl / total_trades
            volatility = abs(avg_win - avg_loss)
            sharpe_ratio = avg_return / volatility
        else:
            sharpe_ratio = 0
        
        # 연속 손실 카운트
        consecutive_losses = existing[10] + 1 if not is_win else 0
        
        c.execute('''
            UPDATE strategy_performance SET
                total_trades = ?,
                winning_trades = ?,
                losing_trades = ?,
                total_pnl = ?,
                win_rate = ?,
                avg_win = ?,
                avg_loss = ?,
                sharpe_ratio = ?,
                consecutive_losses = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE trading_style = ?
        ''', (total_trades, winning_trades, losing_trades, total_pnl, win_rate,
              avg_win, avg_loss, sharpe_ratio, consecutive_losses, trading_style))
    else:
        # 신규 삽입
        c.execute('''
            INSERT INTO strategy_performance (
                trading_style, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, avg_win, avg_loss, sharpe_ratio, consecutive_losses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trading_style, 1, 1 if is_win else 0, 0 if is_win else 1,
              pnl, 100.0 if is_win else 0.0,
              pnl if is_win else 0, pnl if not is_win else 0, 0, 0 if is_win else 1))
    
    conn.commit()
    conn.close()

def get_strategy_performance(trading_style: str = None) -> Dict:
    """전략별 성과 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if trading_style:
        c.execute('SELECT * FROM strategy_performance WHERE trading_style = ?', (trading_style,))
        row = c.fetchone()
        if row:
            result = {
                'trading_style': row[1],
                'total_trades': row[2],
                'winning_trades': row[3],
                'losing_trades': row[4],
                'total_pnl': row[5],
                'win_rate': row[6],
                'avg_win': row[7],
                'avg_loss': row[8],
                'sharpe_ratio': row[9],
                'max_drawdown': row[10],
                'consecutive_losses': row[11]
            }
        else:
            result = None
    else:
        # 전체 조회
        c.execute('SELECT * FROM strategy_performance')
        rows = c.fetchall()
        result = []
        for row in rows:
            result.append({
                'trading_style': row[1],
                'total_trades': row[2],
                'winning_trades': row[3],
                'losing_trades': row[4],
                'total_pnl': row[5],
                'win_rate': row[6],
                'avg_win': row[7],
                'avg_loss': row[8],
                'sharpe_ratio': row[9],
                'max_drawdown': row[10],
                'consecutive_losses': row[11]
            })
    
    conn.close()
    return result

def update_pattern_learning(pattern_description: str, pattern_type: str, success: bool, pnl: float, market_conditions: Dict):
    """패턴 학습 업데이트"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 기존 패턴 조회
    c.execute('SELECT * FROM pattern_learning WHERE pattern_description = ?', (pattern_description,))
    existing = c.fetchone()
    
    if existing:
        success_count = existing[3] + (1 if success else 0)
        failure_count = existing[4] + (0 if success else 1)
        total_count = success_count + failure_count
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        # 평균 PnL 업데이트
        prev_avg_pnl = existing[6]
        prev_count = existing[3] + existing[4]
        new_avg_pnl = ((prev_avg_pnl * prev_count) + pnl) / total_count
        
        # Confidence Level 계산 (성공률 기반)
        confidence_level = int(success_rate) if total_count >= 5 else 50
        
        c.execute('''
            UPDATE pattern_learning SET
                success_count = ?,
                failure_count = ?,
                success_rate = ?,
                avg_pnl = ?,
                confidence_level = ?,
                last_seen = CURRENT_TIMESTAMP,
                market_conditions = ?
            WHERE pattern_description = ?
        ''', (success_count, failure_count, success_rate, new_avg_pnl,
              confidence_level, json.dumps(market_conditions), pattern_description))
    else:
        # 신규 삽입
        success_count = 1 if success else 0
        failure_count = 0 if success else 1
        success_rate = 100.0 if success else 0.0
        
        c.execute('''
            INSERT INTO pattern_learning (
                pattern_type, pattern_description, success_count, failure_count,
                success_rate, avg_pnl, confidence_level, market_conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pattern_type, pattern_description, success_count, failure_count,
              success_rate, pnl, 50, json.dumps(market_conditions)))
    
    conn.commit()
    conn.close()

def get_learned_patterns(min_occurrences: int = 3) -> List[Dict]:
    """학습된 패턴 조회 (최소 발생 횟수 필터)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM pattern_learning
        WHERE (success_count + failure_count) >= ?
        ORDER BY success_rate DESC, (success_count + failure_count) DESC
        LIMIT 20
    ''', (min_occurrences,))
    
    rows = c.fetchall()
    patterns = []
    for row in rows:
        patterns.append({
            'pattern_type': row[1],
            'pattern_description': row[2],
            'success_count': row[3],
            'failure_count': row[4],
            'success_rate': row[5],
            'avg_pnl': row[6],
            'confidence_level': row[7],
            'total_occurrences': row[3] + row[4]
        })
    
    conn.close()
    return patterns

def get_dynamic_confidence_threshold() -> int:
    """Confidence 기준 - 70% 고정"""
    return 70


def log_risk_adjustment(adjustment_type: str, previous_value: float, new_value: float, reason: str, consecutive_losses: int = 0, current_drawdown: float = 0):
    """리스크 조정 로그 저장"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO risk_adjustments (
            adjustment_type, previous_value, new_value, reason,
            consecutive_losses, current_drawdown
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (adjustment_type, previous_value, new_value, reason, consecutive_losses, current_drawdown))
    
    conn.commit()
    conn.close()

def get_all_open_trades() -> List[Dict]:
    """모든 오픈 거래 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, coin_symbol, action, entry_price, amount, leverage,
               investment_amount, sl_price, tp_price, timestamp, ai_reasoning,
               binance_order_id, trading_style, manual_trade
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
            'action': row[2],
            'entry_price': row[3],
            'amount': row[4],
            'leverage': row[5],
            'investment': row[6],
            'sl_price': row[7],
            'tp_price': row[8],
            'timestamp': row[9],
            'ai_reasoning': row[10],
            'binance_order_id': row[11],
            'trading_style': row[12] if len(row) > 12 else 'DAY_TRADING',
            'manual_trade': row[13] if len(row) > 13 else 0
        })
    
    return trades

def save_ai_decision(decision_data: Dict):
    """AI 결정 저장"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO ai_decisions (
            coin_symbol, decision_type, direction, leverage,
            investment_percentage, sl_percentage, tp_percentage,
            confidence_score, reasoning, market_data, related_trade_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        decision_data['coin_symbol'],
        decision_data['decision_type'],
        decision_data.get('direction'),
        decision_data.get('leverage'),
        decision_data.get('investment_percentage'),
        decision_data.get('sl_percentage'),
        decision_data.get('tp_percentage'),
        decision_data['confidence_score'],
        decision_data['reasoning'],
        decision_data.get('market_data', ''),
        decision_data.get('related_trade_id')
    ))
    
    conn.commit()
    conn.close()

def get_recent_performance(days: int = 7) -> Dict:
    """최근 성과 조회 (강화된 디버깅 버전)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 🔍 디버깅: 전체 거래 현황 먼저 확인
    c.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
    total_closed = c.fetchone()[0]
    
    print(f"   🔍 [디버그] 전체 청산 거래: {total_closed}개")
    
    if total_closed == 0:
        conn.close()
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    # 🔍 디버깅: 실제 PnL 값들 확인
    c.execute("""
        SELECT coin_symbol, pnl, binance_pnl, close_timestamp 
        FROM trades 
        WHERE status = 'CLOSED' 
        ORDER BY close_timestamp DESC 
        LIMIT 10
    """)
    
    recent_trades = c.fetchall()
    print(f"   🔍 [디버그] 최근 청산 거래:")
    for trade in recent_trades:
        coin, pnl, binance_pnl, close_time = trade
        actual_pnl = binance_pnl if binance_pnl is not None and binance_pnl != 0 else pnl
        print(f"   ├─ {coin}: PnL=${actual_pnl}, Binance=${binance_pnl}, DB=${pnl}, 시간={close_time}")
    
    # 🆕 개선된 쿼리: binance_pnl 우선, pnl을 fallback으로 사용
    c.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE 
                WHEN COALESCE(binance_pnl, pnl, 0) > 0 THEN 1 
                ELSE 0 
            END) as winning_trades,
            SUM(CASE 
                WHEN COALESCE(binance_pnl, pnl, 0) < 0 THEN 1 
                ELSE 0 
            END) as losing_trades,
            COALESCE(SUM(COALESCE(binance_pnl, pnl, 0)), 0) as total_pnl,
            COALESCE(AVG(CASE 
                WHEN COALESCE(binance_pnl, pnl, 0) > 0 
                THEN COALESCE(binance_pnl, pnl, 0) 
            END), 0) as avg_win,
            COALESCE(AVG(CASE 
                WHEN COALESCE(binance_pnl, pnl, 0) < 0 
                THEN COALESCE(binance_pnl, pnl, 0) 
            END), 0) as avg_loss
        FROM trades
        WHERE status = 'CLOSED'
        AND (
            close_timestamp >= datetime('now', '-' || ? || ' days')
            OR close_timestamp >= date('now', '-' || ? || ' days')
        )
    ''', (days, days))
    
    row = c.fetchone()
    conn.close()
    
    print(f"   🔍 [디버그] 쿼리 결과: {row}")
    
    if not row or row[0] == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    result = {
        'total_trades': row[0],
        'winning_trades': row[1],
        'losing_trades': row[2],
        'win_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
        'total_pnl': float(row[3]) if row[3] else 0,
        'avg_win': float(row[4]) if row[4] else 0,
        'avg_loss': abs(float(row[5])) if row[5] else 0  # 절댓값으로 표시
    }
    
    print(f"   🔍 [디버그] 최종 결과: {result}")
    
    return result

def get_consecutive_losses() -> int:
    """최근 연속 손실 횟수 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT pnl FROM trades
        WHERE status = 'CLOSED'
        ORDER BY close_timestamp DESC
        LIMIT 10
    ''')
    
    recent_trades = c.fetchall()
    conn.close()
    
    consecutive_losses = 0
    for trade in recent_trades:
        if trade[0] < 0:
            consecutive_losses += 1
        else:
            break
    
    return consecutive_losses

def get_current_drawdown() -> float:
    """현재 드로다운 % 계산"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT SUM(pnl) as cumulative_pnl
        FROM trades
        WHERE status = 'CLOSED'
        ORDER BY close_timestamp
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    if not rows or not rows[0][0]:
        return 0
    
    # 간단한 드로다운: 최근 총 손익의 음수 비율
    total_pnl = rows[0][0]
    if total_pnl < 0:
        return abs(total_pnl)
    return 0

def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """켈리 기준 계산 (최적 포지션 크기 비율)"""
    if avg_loss == 0 or win_rate == 0:
        return 0.05  # 기본 5%
    
    b = abs(avg_win / avg_loss)  # win/loss ratio
    p = win_rate / 100  # 승률 (0-1)
    q = 1 - p  # 패율
    
    kelly = (b * p - q) / b
    
    # Kelly를 너무 높게 사용하지 않도록 제한 (Half Kelly 또는 Quarter Kelly)
    # 보수적으로 Kelly의 25%만 사용
    conservative_kelly = max(0.03, min(kelly * 0.25, 0.20))  # 3%-20% 범위
    
    return conservative_kelly


    """모든 오픈 거래 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, coin_symbol, action, entry_price, amount, leverage,
               investment_amount, sl_price, tp_price, timestamp, ai_reasoning,
               binance_order_id
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
            'action': row[2],
            'entry_price': row[3],
            'amount': row[4],
            'leverage': row[5],
            'investment': row[6],
            'sl_price': row[7],
            'tp_price': row[8],
            'timestamp': row[9],
            'ai_reasoning': row[10],
            'binance_order_id': row[11]
        })
    
    return trades

# ===== 시장 데이터 수집 =====

def fetch_ohlcv(symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
    """OHLCV 데이터 가져오기"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"   ❌ OHLCV 조회 오류 ({timeframe}): {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산 (원본 복원)"""
    if df.empty or len(df) < 20:
        return df
    
    try:
        # DataFrame 정렬 및 인덱스 설정 (VWAP 계산을 위해 필수)
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # ===== 이동평균 =====
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.sma(length=20, append=True)
        
        # ===== 모멘텀 지표 =====
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.stoch(append=True)
        df.ta.cci(length=20, append=True)
        df.ta.roc(length=10, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.mom(length=10, append=True)
        
        # ===== 변동성 지표 =====
        df.ta.bbands(length=20, std=2.0, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.kc(length=20, append=True)
        df.ta.donchian(length=20, append=True)
        
        # ===== 추세 강도 지표 =====
        df.ta.adx(length=14, append=True)
        df.ta.aroon(length=25, append=True)
        df.ta.psar(append=True)
        
        # ===== 거래량 지표 =====
        df.ta.obv(append=True)
        df.ta.vwap(append=True)  # VWAP는 정렬된 DatetimeIndex 필요
        df.ta.mfi(length=14, append=True)
        df.ta.ad(append=True)
        
        # ===== 추가 계산 =====
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # 골든크로스/데드크로스
        has_ema = 'EMA_50' in df.columns and 'EMA_200' in df.columns
        df['golden_cross'] = (df['EMA_50'] > df['EMA_200']).astype(int) if has_ema else 0
        
        # 인덱스 리셋 (이후 처리를 위해)
        df.reset_index(inplace=True)
        
        return df
    except Exception as e:
        print(f"   ⚠️ 지표 계산 오류: {e}")
        # 오류 발생 시 인덱스 리셋 시도
        if 'timestamp' in df.index.names:
            df.reset_index(inplace=True)
        return df

def fetch_comprehensive_market_data(symbol: str) -> Dict:
    """여러 타임프레임의 시장 데이터 수집 + 선물 특화 지표"""
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']  # 원본 복원: 7개 타임프레임
    market_data = {}
    
    # 🆕 선물 특화 지표 수집
    print(f"   📊 선물 특화 지표 수집 중...")
    futures_indicators = fetch_futures_market_indicators(symbol)
    
    # 펀딩비 정보 출력
    if futures_indicators['funding_rate'] is not None:
        fr = futures_indicators['funding_rate']
        fr_status = futures_indicators['funding_rate_status']
        print(f"   💸 펀딩비: {fr:+.4f}% ({fr_status})")
    
    # OI 정보 출력
    if futures_indicators['open_interest'] is not None:
        oi = futures_indicators['open_interest']
        print(f"   📊 미결제약정: {oi:,.0f}")
    
    # market_data에 선물 지표 추가
    market_data['futures_indicators'] = futures_indicators
    
    for tf in timeframes:
        try:
            df = fetch_ohlcv(symbol, tf, 100)
            if df.empty:
                continue
            
            df = calculate_technical_indicators(df)
            
            if len(df) < 20:
                continue
            
            # 최근 캔들
            last_candle = df.iloc[-1]
            
            # 추세 판단 (다중 신호 기반)
            trend_signals = []
            if 'EMA_9' in last_candle.index and 'EMA_21' in last_candle.index:
                trend_signals.append(1 if last_candle['EMA_9'] > last_candle['EMA_21'] else -1)
            if 'EMA_50' in last_candle.index:
                trend_signals.append(1 if last_candle['close'] > last_candle['EMA_50'] else -1)
            if 'MACDh_12_26_9' in last_candle.index:
                trend_signals.append(1 if last_candle['MACDh_12_26_9'] > 0 else -1)
            
            avg_signal = sum(trend_signals) / len(trend_signals) if trend_signals else 0
            trend = 'bullish' if avg_signal > 0.3 else ('bearish' if avg_signal < -0.3 else 'neutral')
            
            market_data[tf] = {
                'summary': {
                    'current_price': float(last_candle['close']),
                    'price_change_24h': float(((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100)),
                    
                    # 추세
                    'ema_9': float(last_candle.get('EMA_9', 0)),
                    'ema_21': float(last_candle.get('EMA_21', 0)),
                    'ema_50': float(last_candle.get('EMA_50', 0)),
                    'ema_200': float(last_candle.get('EMA_200', 0)),
                    'golden_cross': bool(last_candle.get('golden_cross', 0)),
                    
                    # 모멘텀
                    'rsi': float(last_candle.get('RSI_14', 50)),
                    'macd': float(last_candle.get('MACD_12_26_9', 0)),
                    'macd_signal': float(last_candle.get('MACDs_12_26_9', 0)),
                    'macd_histogram': float(last_candle.get('MACDh_12_26_9', 0)),
                    'stoch_k': float(last_candle.get('STOCHk_14_3_3', 50)),
                    'stoch_d': float(last_candle.get('STOCHd_14_3_3', 50)),
                    'cci': float(last_candle.get('CCI_20_0.015', 0)),
                    'williams_r': float(last_candle.get('WILLR_14', -50)),
                    
                    # 변동성
                    'bb_upper': float(last_candle.get('BBU_20_2.0', 0)),
                    'bb_middle': float(last_candle.get('BBM_20_2.0', 0)),
                    'bb_lower': float(last_candle.get('BBL_20_2.0', 0)),
                    'bb_width': float((last_candle.get('BBU_20_2.0', 0) - last_candle.get('BBL_20_2.0', 0)) / last_candle.get('BBM_20_2.0', 1) * 100) if last_candle.get('BBM_20_2.0', 0) > 0 else 0,
                    'atr': float(last_candle.get('ATRr_14', 0)),
                    
                    # 추세 강도
                    'adx': float(last_candle.get('ADX_14', 0)),
                    'adx_trend': 'strong' if last_candle.get('ADX_14', 0) > 25 else 'weak',
                    
                    # 거래량
                    'volume': float(last_candle['volume']),
                    'volume_trend': 'increasing' if last_candle['volume'] > df.iloc[-10:-1]['volume'].mean() else 'decreasing',
                    'volume_ratio': float(last_candle.get('volume_ratio', 1)),
                    'obv': float(last_candle.get('OBV', 0)),
                    'mfi': float(last_candle.get('MFI_14', 50)),
                    
                    # 종합
                    'trend': trend,
                    'volatility': float(df['price_change_pct'].tail(20).std()) if len(df) > 20 else 0,
                    'support_level': float(df['low'].tail(20).min()),
                    'resistance_level': float(df['high'].tail(20).max()),
                }
            }
            
            s = market_data[tf]['summary']
            print(f"   ✅ {tf:3s} | RSI:{s['rsi']:5.1f} | MACD:{s['macd_histogram']:+6.2f} | ADX:{s['adx']:5.1f} | 추세:{s['trend']}")
            
        except Exception as e:
            print(f"   ⚠️ {tf} 오류: {e}")
            continue
    
    return market_data

def get_top_volume_coins(limit: int = 10) -> List[Dict]:
    """거래대금 상위 코인 조회"""
    try:
        tickers = exchange.fetch_tickers()
        usdt_futures = [
            {
                'symbol': symbol,
                'coin': symbol.split('/')[0],
                'volume': ticker.get('quoteVolume', 0),
                'price': ticker.get('last', 0),
                'change_24h': ticker.get('percentage', 0)
            }
            for symbol, ticker in tickers.items()
            if symbol.endswith('/USDT:USDT')
        ]
        
        sorted_coins = sorted(usdt_futures, key=lambda x: x['volume'], reverse=True)
        return sorted_coins[:limit]
    except Exception as e:
        print(f"❌ 거래대금 상위 코인 조회 오류: {e}")
        return []

# ===== AI 분석 =====

def call_ai_model(prompt: str, temperature: float = 0.7) -> str:
    """AI 모델 호출"""
    global ai_client, last_api_call_time
    provider = AI_MODEL_CONFIG["provider"].lower()
    model_name = AI_MODEL_CONFIG["models"][provider]
    
    rate_config = AI_MODEL_CONFIG["rate_limit"][provider]
    min_delay = rate_config["delay_between_requests"]
    
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    
    if time_since_last_call < min_delay:
        wait_time = min_delay - time_since_last_call
        print(f"   ⏳ Rate Limit: {wait_time:.1f}초 대기...")
        time.sleep(wait_time)
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=temperature)
                )
                last_api_call_time = time.time()
                return response.text
            elif provider == "deepseek":
                response = ai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "당신은 암호화폐 트레이딩 전문가 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                last_api_call_time = time.time()
                return response.choices[0].message.content
            else:
                raise Exception(f"지원하지 않는 제공자: {provider}")
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['429', 'rate limit', 'quota', 'exceeded']):
                wait_time = 60
                if 'retry_delay' in error_msg or 'seconds:' in error_msg:
                    import re
                    match = re.search(r'seconds:\s*(\d+)', error_msg)
                    if match:
                        wait_time = int(match.group(1)) + 5
                print(f"   ⚠️ Rate Limit 초과! {wait_time}초 대기 중...")
                time.sleep(wait_time)
                continue
            elif any(keyword in error_msg for keyword in ['402', 'insufficient balance', 'insufficient_balance']):
                raise Exception(f"⚠️ API 크레딧 부족! DeepSeek 계정을 충전하세요: https://platform.deepseek.com/usage")
            elif any(keyword in error_msg for keyword in ['overloaded', '503', 'timeout']):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"   ⚠️ API 과부하! {wait_time:.1f}초 후 재시도...")
                time.sleep(wait_time)
                continue
            else:
                print(f"   ❌ AI 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

def ai_comprehensive_analysis(coin_data: Dict, market_data: Dict, performance_history: Dict) -> Dict:
    """AI 종합 분석 (신규 진입) - 과거 성과 학습 추가"""
    
    try:
        # 🆕 선물 특화 지표 텍스트 생성
        futures_indicators_text = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        if 'futures_indicators' in market_data:
            fi = market_data['futures_indicators']
            
            # 1. 펀딩 비율
            if fi['funding_rate'] is not None:
                fr = fi['funding_rate']
                fr_status = fi['funding_rate_status']
                
                fr_emoji = "🔥" if abs(fr) > 0.05 else "⚠️" if abs(fr) > 0.01 else "✅"
                futures_indicators_text += f"{fr_emoji} **펀딩 비율 (Funding Rate)**: {fr:+.4f}% ({fr_status})\n"
                
                if fr_status == 'long_overheated':
                    futures_indicators_text += "   💡 롱 과열! 펀딩비 매우 높음 → SHORT 기회 or 롱 진입 회피\n"
                elif fr_status == 'short_overheated':
                    futures_indicators_text += "   💡 숏 과열! 펀딩비 매우 낮음 → LONG 기회 or 숏 진입 회피\n"
                elif fr_status == 'bullish':
                    futures_indicators_text += "   💡 롱 우세 → 상승 추세 가능성, 숏 신중\n"
                elif fr_status == 'bearish':
                    futures_indicators_text += "   💡 숏 우세 → 하락 추세 가능성, 롱 신중\n"
                else:
                    futures_indicators_text += "   💡 중립적 펀딩비 → 방향성 판단 어려움\n"
            else:
                futures_indicators_text += "⚠️ 펀딩 비율 데이터 없음\n"
            
            # 2. 미결제 약정 (Open Interest)
            if fi['open_interest'] is not None:
                oi = fi['open_interest']
                oi_status = fi['oi_status']
                
                futures_indicators_text += f"\n📊 **미결제 약정 (Open Interest)**: {oi:,.0f} ({oi_status})\n"
                futures_indicators_text += "   💡 OI 증가 + 가격 상승 = 강한 상승 추세\n"
                futures_indicators_text += "   💡 OI 증가 + 가격 하락 = 강한 하락 추세\n"
                futures_indicators_text += "   💡 OI 감소 = 포지션 청산 중, 추세 약화\n"
            else:
                futures_indicators_text += "\n⚠️ 미결제 약정 데이터 없음\n"
        else:
            futures_indicators_text += "⚠️ 선물 특화 지표를 가져올 수 없습니다.\n"
        
        futures_indicators_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 🆕 전략별 성과 조회
        strategy_stats = get_strategy_performance()
        strategy_summary = "\n【전략별 과거 성과 - 이 데이터로 전략 선택하세요】\n"
        
        if strategy_stats:
            strategy_summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for stat in strategy_stats:
                emoji = "⭐" if stat['sharpe_ratio'] >= 1.2 else "✅" if stat['sharpe_ratio'] >= 0.8 else "⚠️"
                strategy_summary += f"{emoji} {stat['trading_style']:15s} | "
                strategy_summary += f"거래:{stat['total_trades']:3d}회 | "
                strategy_summary += f"승률:{stat['win_rate']:5.1f}% | "
                strategy_summary += f"Sharpe:{stat['sharpe_ratio']:5.2f} | "
                strategy_summary += f"평균수익:${stat['avg_win']:6.1f} | "
                strategy_summary += f"평균손실:${stat['avg_loss']:6.1f} | "
                strategy_summary += f"연속손실:{stat['consecutive_losses']}회\n"
            strategy_summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            # 최고 성과 전략 강조
            best_strategy = max(strategy_stats, key=lambda x: x['sharpe_ratio']) if strategy_stats else None
            if best_strategy and best_strategy['sharpe_ratio'] > 0.8:
                strategy_summary += f"🌟 **최고 성과 전략: {best_strategy['trading_style']} (Sharpe: {best_strategy['sharpe_ratio']:.2f})**\n"
                strategy_summary += f"   → 이 전략을 우선적으로 고려하세요!\n"
            
            # 연속 손실 경고
            for stat in strategy_stats:
                if stat['consecutive_losses'] >= 3:
                    strategy_summary += f"⚠️ **{stat['trading_style']} 연속 {stat['consecutive_losses']}회 손실 중 - 이 전략 회피 권장**\n"
        else:
            strategy_summary += "아직 전략별 성과 데이터가 없습니다. 신중하게 판단하세요.\n"
        
        # 🆕 학습된 패턴 조회
        learned_patterns = get_learned_patterns(min_occurrences=3)
        pattern_summary = "\n【학습된 성공/실패 패턴 - 반드시 참고하세요】\n"
        
        if learned_patterns:
            pattern_summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            # 성공 패턴 (승률 70% 이상)
            success_patterns = [p for p in learned_patterns if p['success_rate'] >= 70]
            if success_patterns:
                pattern_summary += "✅ **성공 패턴 (승률 70% 이상) - 이런 신호 보이면 진입!**\n"
                for p in success_patterns[:5]:
                    pattern_summary += f"   • {p['pattern_description']} "
                    pattern_summary += f"(승률:{p['success_rate']:.0f}% | {p['total_occurrences']}회 | 평균:${p['avg_pnl']:+.1f})\n"
            
            # 실패 패턴 (승률 40% 미만)
            failure_patterns = [p for p in learned_patterns if p['success_rate'] < 40]
            if failure_patterns:
                pattern_summary += "\n❌ **실패 패턴 (승률 40% 미만) - 이런 신호 보이면 회피!**\n"
                for p in failure_patterns[:5]:
                    pattern_summary += f"   • {p['pattern_description']} "
                    pattern_summary += f"(승률:{p['success_rate']:.0f}% | {p['total_occurrences']}회 | 평균:${p['avg_pnl']:+.1f})\n"
            
            pattern_summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        else:
            pattern_summary += "아직 학습된 패턴이 없습니다 (최소 3회 이상 발생 필요).\n"
        
        # 🆕 리스크 조정 상태
        consecutive_losses = get_consecutive_losses()
        current_drawdown = get_current_drawdown()
        
        risk_adjustment_summary = "\n【현재 리스크 상태】\n"
        risk_adjustment_summary += f"연속 손실: {consecutive_losses}회\n"
        risk_adjustment_summary += f"현재 드로다운: ${current_drawdown:.2f}\n"
        
        # 시장 데이터 요약 (상세 버전)
        market_summary = "【타임프레임별 상세 분석】\n"
        for tf, data in market_data.items():
            if 'summary' in data:
                s = data['summary']
                market_summary += f"\n━━━ [{tf}] ━━━\n"
                market_summary += f"현재가: ${s.get('current_price', 0):,.4f} | 추세: {s['trend']}\n"
                market_summary += f"이동평균 - EMA9:{s.get('ema_9', 0):.2f} EMA21:{s.get('ema_21', 0):.2f} EMA50:{s.get('ema_50', 0):.2f} 골든크로스:{s.get('golden_cross', False)}\n"
                market_summary += f"모멘텀 - RSI:{s['rsi']:.1f} MACD:{s['macd_histogram']:+.4f} Stochastic:{s.get('stoch_k', 50):.1f}/{s.get('stoch_d', 50):.1f}\n"
                market_summary += f"추세강도 - ADX:{s['adx']:.1f}({s['adx_trend']}) CCI:{s.get('cci', 0):.1f} Williams%R:{s.get('williams_r', -50):.1f}\n"
                market_summary += f"변동성 - BB밴드폭:{s['bb_width']:.2f}% ATR:{s['atr']:.2f}\n"
                market_summary += f"거래량 - {s['volume_trend']} 비율:{s['volume_ratio']:.2f}x MFI:{s['mfi']:.1f}\n"
                market_summary += f"지지/저항 - 지지선:${s['support_level']:,.2f} 저항선:${s['resistance_level']:,.2f}\n"
        
        prompt = f"""
당신은 암호화폐 선물 거래 최고 전문가 AI입니다. [실거래 모드 - 신중]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **CORE OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【핵심 원칙】
✅ 단순 수익률이 아닌 **위험 대비 수익 (Risk/Reward Ratio) 최대화**
✅ Risk/Reward < 1:2 (리스크 1%, 리워드 2% 미만) → 절대 진입 금지
✅ Risk/Reward ≥ 1:3 (리스크 1%, 리워드 3% 이상) → 우선 고려
✅ 불확실성 높은 거래 회피 (Confidence < 70% → 거부)
✅ 손실 -5% 초과 가능성이 보이면 즉시 거부
✅ Sharpe Ratio 개념: 변동성 대비 안정적 수익 추구
✅ 높은 레버리지 + 낮은 확신 = 절대 금지
✅ 여러 타임프레임에서 일치하는 안전한 신호만 채택

【Risk-Adjusted 거래 체크리스트】
□ Risk/Reward Ratio ≥ 1:2 인가?
□ 손절 가능성 < 익절 가능성 인가?
□ 여러 지표가 합의(confluence)하는가?
□ 3개 이상 타임프레임이 일치하는가?
□ ADX > 25로 추세가 명확한가?
□ 변동성이 과도하지 않은가? (ATR/가격 < 5%)
□ 지지/저항선이 명확한가?
□ 거래량이 충분한가? (MFI > 40, OBV 상승)

**5개 이상 충족 + Risk/Reward ≥ 1:2 → 진입 고려**

【분석 대상】
코인: {coin_data['coin']}
현재가: ${coin_data['price']:,.4f}
24시간 변동: {coin_data.get('change_24h', 0):.2f}%
거래대금: ${coin_data['volume']:,.0f}

【선물 특화 지표 - 매우 중요】
{futures_indicators_text}

【전체 과거 성과】
총 거래: {performance_history['total_trades']}회
승률: {performance_history['win_rate']:.1f}%
총 수익: ${performance_history['total_pnl']:,.2f}

{strategy_summary}

{pattern_summary}

{risk_adjustment_summary}

{market_summary}

【제공된 20+ 보조지표】
✅ 추세: EMA(9,21,50,200), SMA(20), 골든/데드크로스
✅ 모멘텀: RSI, MACD, Stochastic, CCI, ROC, Williams%R, Momentum
✅ 변동성: 볼린저밴드, ATR, Keltner Channel, Donchian Channel
✅ 추세강도: ADX, Aroon, Parabolic SAR
✅ 거래량: OBV, VWAP, MFI, A/D
✅ 지지/저항선 자동 계산
✅ 🆕 선물 특화: 펀딩비, 미결제약정(OI)

【선물 특화 지표 활용법 - 코인 선물만의 강력한 도구】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **펀딩 비율 (Funding Rate)** - 시장 심리 판단
   ✅ 펀딩비 > +0.05% → 롱 과열 → SHORT 기회 or LONG 회피
   ✅ 펀딩비 < -0.05% → 숏 과열 → LONG 기회 or SHORT 회피
   ✅ 펀딩비 중립(-0.01% ~ +0.01%) → 방향성 불명확
   💡 극단적 펀딩비는 포지션 반전 신호 (롱→숏 or 숏→롱)

2. **미결제 약정 (Open Interest, OI)** - 추세 강도 판단
   ✅ OI 증가 + 가격 상승 = 강한 상승 추세 (신규 롱 유입)
   ✅ OI 증가 + 가격 하락 = 강한 하락 추세 (신규 숏 유입)
   ✅ OI 감소 + 가격 변동 = 포지션 청산, 추세 약화
   💡 OI 급증은 추세 강화 신호, OI 급락은 추세 전환 가능

🎯 **선물 지표 종합 전략**
- 펀딩비 + OI를 종합적으로 분석
- 2가지 지표가 같은 방향 → 강한 신호
- 예: 펀딩비 롱과열 + OI 증가 + 가격 상승 = SHORT 고려 (조정 가능성)
- 예: 펀딩비 숏과열 + OI 증가 + 가격 하락 = LONG 고려 (반등 가능성)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【분석 가이드 - 실거래 모드】
1. 타임프레임 크로스체크: 여러 타임프레임에서 같은 신호 → 신뢰도 ↑
2. 다중 지표 확인: RSI+MACD+Stochastic 모두 일치 → 강한 신호
3. 추세 강도 확인: ADX > 25 → 강한 추세, < 20 → 약한 추세 (진입 회피)
4. 거래량 확인: MFI, OBV 상승 → 매수세 강함
5. 변동성 고려: BB밴드폭 좁으면 → 큰 움직임 임박, 넓으면 조정 가능성
6. 지지/저항 근처: 반등 or 돌파 가능성 평가
7. 레버리지: BTC/ETH 최대 15배, 알트 8~12배 (실거래도 적극적으로)
8. 과거 실패 패턴 반드시 회피
9. 🆕 선물 특화 지표로 시장 심리와 추세 강도를 정확히 파악

【트레이딩 스타일 선택 - 매우 중요】
타임프레임 분석을 바탕으로 아래 중 하나를 선택하세요:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 **트레이딩 스타일 선택 가이드 - 타임프레임 매칭 필수**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **중요**: 트레이딩 스타일은 **현재 시장 신호에 가장 적합한 것**을 선택하세요!
- 스캘핑/데이/스윙 모두 활성화되어 있습니다
- 단기 신호 → SCALPING, 중기 신호 → DAY_TRADING, 장기 신호 → SWING_TRADING
- 타점이 오면 반드시 진입하되, **타임프레임과 스타일이 반드시 매칭되어야 함**

1. SCALPING (스캘핑) - 초단타
   ✅ **주 분석 타임프레임: 5m, 15m** (1m~15m 범위)
   ❌ **1h, 4h, 1d 타임프레임을 주 분석으로 사용 금지** (스캘핑에는 너무 긴 시간)
   - 목표 수익: 1~3%
   - 손절: 0.5~1.5%
   - 보유 시간: 수분~1시간
   - 레버리지: 높음 (10~15x)
   - 시그널 예: 5m/15m 급등/급락, 단기 과매수/과매도, 초단기 모멘텀
   - 타임프레임 검증: 5m과 15m이 일치하면 진입
   - 예: "5m과 15m 모두 RSI 과매도 + 볼린저밴드 하단 터치" ✅
   - 잘못된 예: "1h과 4h 기준으로 스캘핑 진입" ❌

2. DAY_TRADING (데이트레이딩) - 단기
   ✅ **주 분석 타임프레임: 15m, 1h, 4h** (15m~4h 범위)
   ❌ **5m 타임프레임 사용 금지** (너무 짧아 노이즈 많음)
   ❌ **1d, 1w 타임프레임 사용 금지** (데이트레이딩에는 너무 긴 시간)
   - 목표 수익: 3~8%
   - 손절: 1.5~3%
   - 보유 시간: 수시간~1일 (4~12시간 목표)
   - 레버리지: 중상 (8~12x)
   - 시그널 예: 1h/4h 추세 전환, 중기 모멘텀, 당일 변동성 트레이딩
   - 타임프레임 검증: 15m, 1h, 4h 중 최소 2개 이상 일치하면 진입
   - 예: "1h과 4h 추세 일치, 15m 진입 신호, ADX 상승" ✅
   - 잘못된 예: "5m 기준으로 데이트레이딩" ❌ (너무 짧음)
   - 잘못된 예: "1d 기준으로 데이트레이딩" ❌ (너무 김)

3. SWING_TRADING (스윙) - 중장기
   ✅ **주 분석 타임프레임: 4h, 1d, 1w** (4h~1w 범위)
   ❌ **5m, 15m, 1h 타임프레임 사용 금지** (스윙에는 너무 짧은 시간)
   - 목표 수익: 8~20%
   - 손절: 3~5%
   - 보유 시간: 수일~수주
   - 레버리지: 중간 (5~8x)
   - 시그널 예: 4h/1d/1w 장기 추세 전환, 주봉 패턴, 거시적 모멘텀
   - 타임프레임 검증: 4h, 1d, 1w 중 최소 2개 이상 일치하면 진입
   - 예: "1d와 1w 모두 상승 추세 전환, 4h 골든크로스" ✅
   - 잘못된 예: "15m과 1h 기준으로 스윙 진입" ❌ (잘못된 매칭)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **트레이딩 스타일 결정 방법**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 현재 가장 강한 신호가 나타나는 타임프레임 그룹을 확인
2. 5m/15m에서 신호 → SCALPING 선택
3. 15m/1h/4h에서 신호 → DAY_TRADING 선택
4. 4h/1d/1w에서 신호 → SWING_TRADING 선택
5. 타점이 오면 반드시 진입하되, 타임프레임이 매칭되는 스타일을 선택!
6. 손절/익절 범위도 반드시 트레이딩 스타일에 맞게 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 **SHORT 포지션 전용 규칙 - 매우 중요!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **SHORT는 위험하므로 빠른 진출입만 허용**

SHORT 포지션은 **SCALPING 또는 DAY_TRADING만** 허용:
❌ SHORT + SWING_TRADING = **절대 금지**
✅ SHORT + SCALPING = 허용 (1~3% 목표, 최대 1시간)
✅ SHORT + DAY_TRADING = 허용 (3~8% 목표, 최대 1일)

이유:
1. SHORT는 무한 손실 가능 (Long Squeeze 위험)
2. 하락은 빠르지만, 급등도 빠름 (변동성 ↑)
3. 긴 보유는 funding fee 누적
4. 빠른 익절이 안전

**SHORT 판단 시 반드시 체크:**
□ SCALPING 또는 DAY_TRADING 스타일인가?
□ 보유 시간 24시간 이내인가?
□ 빠른 익절 가능한 신호인가?
□ 위 조건 불충족 시 → SHORT 거부, HOLD 또는 LONG 고려
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【실거래 중요 원칙 - RISK-ADJUSTED RETURNS 중심】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **실제 돈이므로 Risk/Reward 철저히 계산하고 확신 없으면 거부**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **LONG과 SHORT를 동등하게 고려하세요!**
- 상승 신호 → LONG
- 하락 신호 → SHORT (동등한 기회)
- SHORT도 LONG만큼 수익 기회가 있습니다
- 편향 없이 객관적으로 판단하세요

【SHORT 포지션 진입 신호 - 적극 고려】
✅ RSI > 70 (과매수) + MACD 데드크로스 → SHORT 고려
✅ 볼린저밴드 상단 터치 + 거래량 감소 → SHORT 고려
✅ 저항선 터치 + 반전 신호 → SHORT 고려
✅ 하락 추세 + ADX > 25 → SHORT 적극 진입
✅ Stochastic 과매수 영역 + 하락 다이버전스 → SHORT 고려

【LONG 포지션 진입 신호】
✅ RSI < 30 (과매도) + MACD 골든크로스 → LONG 고려
✅ 볼린저밴드 하단 터치 + 거래량 증가 → LONG 고려
✅ 지지선 터치 + 반등 신호 → LONG 고려
✅ 상승 추세 + ADX > 25 → LONG 적극 진입
✅ Stochastic 과매도 영역 + 상승 다이버전스 → LONG 고려

1. **Risk/Reward Ratio 계산 필수**
   - 예상 손절 거리: A%
   - 예상 익절 거리: B%
   - Risk/Reward = A:B
   - **1:2 미만이면 절대 진입 금지**
   - **1:3 이상이면 우선 고려**

2. **다중 지표 합의(Confluence) 필수**
   - 최소 3개 이상의 서로 다른 지표군(추세/모멘텀/거래량) 일치
   - 단일 지표만 보지 말고 여러 지표의 합의를 찾으세요
   - 예: RSI 과매도 + MACD 골든크로스 + 거래량 급증 + 지지선 터치

3. **타임프레임 크로스체크 필수**
   - 모든 타임프레임이 같은 방향이면 높은 신뢰도
   - 3개 미만 타임프레임 일치 → 진입 회피
   - 단기/중기/장기 추세가 모두 일치할 때 진입

4. **Confidence 기준: 70% 이상만 진입**
   - 70-75%: 진입 허용 (투자비율 50-60%)
   - 75-80%: 적극 진입 (투자비율 60-70%)
   - 80-85%: 강력 진입 (투자비율 70-80%)  
   - 85-90%: 매우 강력 (투자비율 80-90%)
   - 90-95%: 최고 확신 (투자비율 90-100%)
   - 95%+: 올인 수준 (투자비율 100%)
   - 70% 미만: 거래 금지

5. **Investment Percentage = Confidence 완전 연동 (공격적)**
   - Confidence 70% → Investment 50%
   - Confidence 75% → Investment 60%
   - Confidence 80% → Investment 70%
   - Confidence 85% → Investment 80%
   - Confidence 90% → Investment 90%
   - Confidence 95% → Investment 100%
   - 공식: Investment% = (Confidence - 70) × 2 + 50
   - 신뢰도가 높을수록 공격적으로 투자하세요!

5. **변동성 리스크 관리**
   - 고변동성(ATR/가격 > 5%) → 레버리지 ↓, 투자금액 ↓
   - 저변동성 → 안정적 수익 기회

6. **추세 강도 확인 필수**
   - ADX < 20 → 약한 추세 → 진입 회피
   - ADX > 25 → 강한 추세 → 진입 고려 (LONG/SHORT 모두)

7. **과거 실패 패턴 반드시 회피**
   - 유사한 패턴에서 손실 발생 → 동일 실수 금지

8. **레버리지 적극적 운용**
   - 높은 확신(90+) + 강한 추세 → 높은 레버리지 적극 사용 (12~15x)
   - 중간 확신(80%) → 중상 레버리지 (8~12x)
   - 기본 확신(70%) → 중간 레버리지 (6~10x)
   - 낮은 확신 → 거래 금지

**핵심: 불확실한 거래는 절대 하지 않는다. 확실한 Risk/Reward만 공략한다. LONG과 SHORT는 동등한 기회다.**

🚨 **레버리지 선택 가이드 - 매우 중요!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **실거래에서도 적극적인 레버리지 사용으로 수익 극대화**

**트레이딩 스타일별 권장 레버리지:**
- SCALPING: 10~15x (초단기, 빠른 진출입)
- DAY_TRADING: 8~12x (중기, 안정적 수익)  
- SWING_TRADING: 5~8x (장기, 안전 우선)

**확신도별 레버리지 조정:**
- Confidence 95%+: 최대 레버리지 (12~15x)
- Confidence 85-95%: 중상 레버리지 (10~12x)
- Confidence 75-85%: 중간 레버리지 (8~10x)
- Confidence 70-75%: 기본 레버리지 (6~8x)

**코인별 레버리지 조정:**
- BTC/ETH: 최대 15x 가능 (안정적)
- 주요 알트코인: 10~12x 권장
- 신규/소형 알트: 8~10x 안전

**시장 상황별 레버리지:**
- 강한 추세 (ADX > 30): +2x 보너스
- 변동성 낮음 (ATR < 3%): +1x 보너스
- 다중 지표 합의: +1x 보너스

**예시:**
- BTC DAY_TRADING + Confidence 85% + 강한 추세 = 10x + 2x = 12x
- ETH SCALPING + Confidence 90% + 낮은 변동성 = 12x + 1x = 13x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**반드시 다음 JSON 형식으로만 답변:**

{{
  "trade": true/false,
  "direction": "LONG" or "SHORT",
  "risk_reward_ratio": "예: 1:3 (리스크 2%, 리워드 6%)",
  "expected_risk_pct": 0.5~5.0 (예상 손실 %),
  "expected_reward_pct": 1.0~20.0 (예상 수익 %),
  "risk_adjusted_score": 0~100 (Risk/Reward 고려한 점수, 80+ 진입),
  "trading_style": "SCALPING" or "DAY_TRADING" or "SWING_TRADING",
  "leverage": 5~{LIVE_TRADING_CONFIG['MAX_LEVERAGE']},
  "investment_percentage": 50~100 (신뢰도 연동: 70%=50%, 80%=70%, 90%=90%, 95%=100%),
  "sl_percentage": 0.5~5 (스타일에 맞게),
  "tp_percentage": 1~20 (스타일에 맞게),
  "confidence": 0~100,
  "reasoning": "분석 근거 (200자 이내) - Risk/Reward 계산 포함",
  "key_factors": ["핵심요인1", "핵심요인2", "핵심요인3"],
  "primary_timeframe": "주 분석 타임프레임",
  "timeframe_analysis": "타임프레임 종합 (100자)",
  "indicator_confluence": "일치하는 지표들 (예: RSI+MACD+Stochastic 모두 과매수)",
  "holding_time_estimate": "예상 보유 시간 (예: 30분, 4시간, 3일)",
  "checklist_passed": 5~8 (체크리스트 통과 개수)
}}

중요: 
- **risk_reward_ratio가 1:2 미만이면 trade: false 필수**
- **risk_adjusted_score < 80이면 trade: false 필수**
- **checklist_passed < 5이면 trade: false 필수**
- **트레이딩 스타일과 타임프레임이 매칭되어야 함:**
  - SCALPING이면 primary_timeframe이 "5m" 또는 "15m"이어야 함
  - DAY_TRADING이면 primary_timeframe이 "15m", "1h", "4h" 중 하나여야 함
  - SWING_TRADING이면 primary_timeframe이 "4h", "1d", "1w" 중 하나여야 함
  - 매칭 안 되면 trade: false로 거부
- **trading_style에 따라 sl_percentage와 tp_percentage를 적절히 설정:**
  
  🆕 **변동성 기반 동적 TP/SL 설정 (필수):**
  현재 변동성: {market_data.get('1h', {}).get('summary', {}).get('volatility', 3.0):.1f}%
  
  📊 **변동성 구간별 TP/SL 가이드:**
  
  **저변동성 (< 3%)** - BTC, ETH 같은 안정적 코인:
    - SCALPING: SL 0.5~1.0%, TP 1~2%
    - DAY_TRADING: SL 1.5~2.5%, TP 3~5%
    - SWING_TRADING: SL 3~4%, TP 8~12%
  
  **중변동성 (3~5%)** - 중형 알트코인:
    - SCALPING: SL 1.0~2.0%, TP 2~4%
    - DAY_TRADING: SL 2.5~4.0%, TP 5~10%
    - SWING_TRADING: SL 4~6%, TP 12~18%
  
  **고변동성 (> 5%)** - JELLY, MMT 같은 급등주:
    - SCALPING: SL 2.0~3.0%, TP 4~8%
    - DAY_TRADING: SL 4.0~6.0%, TP 10~15%
    - SWING_TRADING: SL 6~8%, TP 18~25%
  
  ⚠️ **중요:** 
  - 변동성이 높을수록 TP/SL을 넓게 설정해야 함
  - 좁은 TP/SL + 고변동성 = 즉시 청산 (1-2분 내)
  - 넓은 TP/SL = 추세를 충분히 탈 수 있음
  - **현재 변동성에 맞는 구간의 TP/SL 범위를 반드시 사용**
  
  - **expected_reward_pct / expected_risk_pct ≥ 2.0 되도록 설정**
- **레버리지 계산 공식:**
  - 기본 레버리지 = 트레이딩 스타일 기준값 (SCALPING: 12x, DAY: 10x, SWING: 6x)
  - + Confidence 보너스: (Confidence - 80) / 5 (85% = +1x, 90% = +2x, 95% = +3x)
  - + 추세 보너스: ADX > 30이면 +2x, ADX > 25면 +1x
  - + 안정성 보너스: BTC/ETH면 +1x
  - 최종 레버리지 = min(계산값, 15x)

주의: JSON 외 다른 텍스트 포함 금지
"""
        
        response_text = call_ai_model(prompt, temperature=0.7)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            decision = json.loads(json_str)
            
            if decision.get('trade'):
                max_lev = LIVE_TRADING_CONFIG['MAX_LEVERAGE']
                if decision.get('leverage', 0) > max_lev:
                    decision['leverage'] = max_lev
                
                # 🔥 Risk/Reward Ratio 검증
                expected_risk = decision.get('expected_risk_pct', 999)
                expected_reward = decision.get('expected_reward_pct', 0)
                risk_adjusted_score = decision.get('risk_adjusted_score', 0)
                
                # Risk/Reward가 1:2 미만이면 거부
                if expected_risk > 0 and (expected_reward / expected_risk) < 2.0:
                    print(f"   ❌ Risk/Reward 부족: {expected_risk:.1f}% 리스크 대비 {expected_reward:.1f}% 리워드 (1:2 미만)")
                    return {"trade": False, "reasoning": f"Risk/Reward 비율 부족 ({expected_risk:.1f}%:{expected_reward:.1f}%)", "confidence": 0}
                
                # Risk-Adjusted Score < 80이면 거부
                if risk_adjusted_score < 80:
                    print(f"   ❌ Risk-Adjusted Score 부족: {risk_adjusted_score} (최소 80 필요)")
                    return {"trade": False, "reasoning": f"위험 조정 점수 부족 ({risk_adjusted_score}/100)", "confidence": 0}
                
                # 체크리스트 통과 개수 < 6이면 거부
                checklist_passed = decision.get('checklist_passed', 0)
                if checklist_passed < 5:
                    print(f"   ❌ 체크리스트 미달: {checklist_passed}/8 (최소 5개 필요)")
                    return {"trade": False, "reasoning": f"안전 체크리스트 미달 ({checklist_passed}/8)", "confidence": 0}
                
                # 🆕 트레이딩 스타일 & 타임프레임 매칭 검증
                trading_style = decision.get('trading_style', '')
                primary_timeframe = decision.get('primary_timeframe', '')
                sl_pct = decision.get('sl_percentage', 0)
                tp_pct = decision.get('tp_percentage', 0)
                
                # 타임프레임 매칭 검증
                timeframe_valid = False
                if trading_style == 'SCALPING':
                    if primary_timeframe in ['5m', '15m']:
                        timeframe_valid = True
                    else:
                        print(f"   ❌ 타임프레임 불일치: SCALPING인데 {primary_timeframe} 사용 (5m/15m만 허용)")
                        return {"trade": False, "reasoning": f"SCALPING은 5m/15m 타임프레임만 사용 가능", "confidence": 0}
                elif trading_style == 'DAY_TRADING':
                    if primary_timeframe in ['15m', '1h', '4h']:
                        timeframe_valid = True
                    else:
                        print(f"   ❌ 타임프레임 불일치: DAY_TRADING인데 {primary_timeframe} 사용 (15m/1h/4h만 허용)")
                        return {"trade": False, "reasoning": f"DAY_TRADING은 15m/1h/4h 타임프레임만 사용 가능", "confidence": 0}
                elif trading_style == 'SWING_TRADING':
                    if primary_timeframe in ['4h', '1d', '1w']:
                        timeframe_valid = True
                    else:
                        print(f"   ❌ 타임프레임 불일치: SWING_TRADING인데 {primary_timeframe} 사용 (4h/1d/1w만 허용)")
                        return {"trade": False, "reasoning": f"SWING_TRADING은 4h/1d/1w 타임프레임만 사용 가능", "confidence": 0}
                
                # 🆕 변동성 기반 손절/익절 범위 검증
                # 1시간 변동성 가져오기
                volatility = market_data.get('1h', {}).get('summary', {}).get('volatility', 3.0)
                
                # 변동성 구간 판단
                if volatility < 3:
                    vol_tier = "저변동성"
                elif volatility <= 5:
                    vol_tier = "중변동성"
                else:
                    vol_tier = "고변동성"
                
                sl_tp_valid = False
                
                if trading_style == 'SCALPING':
                    if volatility < 3:  # 저변동성
                        if 0.5 <= sl_pct <= 1.0 and 1 <= tp_pct <= 2:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SCALPING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 0.5-1.0%, TP 1-2%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SCALPING 범위 부적합", "confidence": 0}
                    elif volatility <= 5:  # 중변동성
                        if 1.0 <= sl_pct <= 2.0 and 2 <= tp_pct <= 4:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SCALPING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 1.0-2.0%, TP 2-4%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SCALPING 범위 부적합", "confidence": 0}
                    else:  # 고변동성
                        if 2.0 <= sl_pct <= 3.0 and 4 <= tp_pct <= 8:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SCALPING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 2.0-3.0%, TP 4-8%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SCALPING 범위 부적합", "confidence": 0}
                
                elif trading_style == 'DAY_TRADING':
                    if volatility < 3:  # 저변동성
                        if 1.5 <= sl_pct <= 2.5 and 3 <= tp_pct <= 5:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} DAY_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 1.5-2.5%, TP 3-5%)")
                            return {"trade": False, "reasoning": f"{vol_tier} DAY_TRADING 범위 부적합", "confidence": 0}
                    elif volatility <= 5:  # 중변동성
                        if 2.5 <= sl_pct <= 4.0 and 5 <= tp_pct <= 10:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} DAY_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 2.5-4.0%, TP 5-10%)")
                            return {"trade": False, "reasoning": f"{vol_tier} DAY_TRADING 범위 부적합", "confidence": 0}
                    else:  # 고변동성
                        if 4.0 <= sl_pct <= 6.0 and 10 <= tp_pct <= 15:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} DAY_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 4.0-6.0%, TP 10-15%)")
                            return {"trade": False, "reasoning": f"{vol_tier} DAY_TRADING 범위 부적합", "confidence": 0}
                
                elif trading_style == 'SWING_TRADING':
                    if volatility < 3:  # 저변동성
                        if 3 <= sl_pct <= 4 and 8 <= tp_pct <= 12:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SWING_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 3-4%, TP 8-12%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SWING_TRADING 범위 부적합", "confidence": 0}
                    elif volatility <= 5:  # 중변동성
                        if 4 <= sl_pct <= 6 and 12 <= tp_pct <= 18:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SWING_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 4-6%, TP 12-18%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SWING_TRADING 범위 부적합", "confidence": 0}
                    else:  # 고변동성
                        if 6 <= sl_pct <= 8 and 18 <= tp_pct <= 25:
                            sl_tp_valid = True
                        else:
                            print(f"   ❌ {vol_tier} SWING_TRADING SL/TP 범위 초과: SL {sl_pct}%, TP {tp_pct}% (권장: SL 6-8%, TP 18-25%)")
                            return {"trade": False, "reasoning": f"{vol_tier} SWING_TRADING 범위 부적합", "confidence": 0}
                
                # 변동성 정보 출력
                print(f"   📊 변동성: {volatility:.1f}% ({vol_tier})")
                
                required_keys = ['direction', 'leverage', 'investment_percentage', 'sl_percentage', 'tp_percentage', 'confidence']
                if all(key in decision for key in required_keys):
                    # ✅ Risk/Reward 정보 출력
                    if 'risk_reward_ratio' in decision:
                        print(f"   📊 Risk/Reward: {decision['risk_reward_ratio']}")
                    if 'expected_risk_pct' in decision and 'expected_reward_pct' in decision:
                        print(f"   📉 예상 리스크: {decision['expected_risk_pct']:.1f}%")
                        print(f"   📈 예상 리워드: {decision['expected_reward_pct']:.1f}%")
                    if 'risk_adjusted_score' in decision:
                        print(f"   ⭐ Risk-Adjusted Score: {decision['risk_adjusted_score']}/100")
                    if 'checklist_passed' in decision:
                        print(f"   ✅ 체크리스트: {decision['checklist_passed']}/8 통과")
                    
                    # 분석 정보 출력
                    if 'trading_style' in decision:
                        style_emoji = {"SCALPING": "⚡", "DAY_TRADING": "📊", "SWING_TRADING": "📈"}
                        emoji = style_emoji.get(decision['trading_style'], "📊")
                        print(f"   {emoji} 트레이딩 스타일: {decision['trading_style']}")
                    if 'holding_time_estimate' in decision:
                        print(f"   ⏱️  예상 보유: {decision['holding_time_estimate']}")
                    if 'primary_timeframe' in decision:
                        print(f"   📊 주 타임프레임: {decision['primary_timeframe']}")
                    if 'indicator_confluence' in decision:
                        print(f"   🎯 지표 합의: {decision['indicator_confluence']}")
                    if 'timeframe_analysis' in decision:
                        print(f"   📈 타임프레임: {decision['timeframe_analysis']}")
                    
                    # 변동성 추가
                    if '1h' in market_data:
                        decision['volatility'] = market_data['1h']['summary'].get('volatility', 3.0)
                    else:
                        decision['volatility'] = 3.0
                    
                    return decision
            else:
                return decision
        
        return {"trade": False, "reasoning": "JSON 파싱 실패", "confidence": 0}
    except Exception as e:
        print(f"❌ AI 분석 오류: {e}")
        return {"trade": False, "reasoning": f"오류: {str(e)[:50]}", "confidence": 0}

def ai_position_management(trade: Dict, market_data: Dict, current_price: float) -> Dict:
    """AI 포지션 관리 (청산 판단) - 원본 복원"""
    
    try:
        entry_price = trade['entry_price']
        leverage = trade.get('leverage', 1)
        action = trade['action']
        
        # 🔧 가격 변화율 (레버리지 반영)
        if action == 'long':
            price_change_pct = ((current_price - entry_price) / entry_price * 100)
        else:
            price_change_pct = ((entry_price - current_price) / entry_price * 100)
        
        # 레버리지 적용된 수익률
        leveraged_pnl_pct = price_change_pct * leverage
        
        # 실제 투자금 대비 PnL
        investment = trade.get('investment', 0)
        actual_pnl_amount = investment * (leveraged_pnl_pct / 100)
        actual_pnl_pct = leveraged_pnl_pct
        
        # 타임프레임 요약
        market_summary = "【타임프레임별 현황】\n"
        for tf, data in market_data.items():
            if 'summary' in data:
                s = data['summary']
                market_summary += f"[{tf}] RSI:{s['rsi']:.0f} 추세:{s['trend']} 변동성:{s['volatility']:.1f}%\n"
        
        prompt = f"""
포지션 관리 AI [실거래 모드]

🎯 **OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS**
- **이익 보호가 최우선**: 수익 중일 때는 이익 보호에 집중
- **변동성 대응**: 큰 이익에서는 추세 약화 시 적극적 익절
- 손실 확대 방지 (손실 -12% 초과 시 즉시 청산 고려)  
- 추세 반전 신호가 보이면 수익 보호 우선

🚨 **이익 보호 우선 원칙**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **변동성 장세에서 이익 실현이 손실 방지보다 중요**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**수익 상황별 대응:**
- **+20% 이상**: 추세 약화/반전 신호 시 즉시 익절 고려 (이익 보호 최우선)
- **+15% 이상**: 여러 타임프레임 약화 시 익절 고려
- **+10% 이상**: 강한 반전 신호 시에만 익절 고려
- **+5% 이상**: 목표가 근처에서만 익절 고려

🔥 **실시간 시장 대응의 장점:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **급변 대응**: 뉴스, 청산 연쇄반응 등 즉시 대응
- **이익 보호**: +25% 수익을 시간 때문에 놓치는 일 없음
- **손실 최소화**: -10% 손실을 시간 채우려 기다리지 않음
- **AI 판단 존중**: 정교한 분석을 시간 제약이 무효화하지 않음
- **기회 포착**: 단기 변동에서도 이익 실현 가능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **10분 주기 평가의 효과:**
- 하루 144회 체크 (충분한 빈도)
- API 호출 비용 < 이익 보호 가치
- 과도하지 않은 적절한 간격
- 급변하는 암호화폐 시장에 최적화

【포지션】
코인: {trade['coin_symbol']} {action.upper()}
진입: ${entry_price:,.4f} → 현재: ${current_price:,.4f}
가격 변화: {price_change_pct:+.2f}%
레버리지: {leverage}x
실제 수익률: {actual_pnl_pct:+.2f}% (투자금 ${investment:,.2f} 대비)
실제 손익: ${actual_pnl_amount:+,.2f}
보유 시간: {(get_utc_now() - parse_db_timestamp(trade['timestamp'])).total_seconds() / 3600:.1f}시간
트레이딩 스타일: {trade.get('trading_style', 'DAY_TRADING')}

{market_summary}

【판단 기준 - Risk-Adjusted Returns 중심】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **AI 판단 우선주의로 전환**
✅ **변동성 고려한 동적 임계값 적용**
✅ **실시간 시장 대응 (시간 제약 없음)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔥 **변동성 기반 동적 임계값**
현재 변동성: {market_data.get('1h', {}).get('summary', {}).get('volatility', 3.0):.1f}%
- 고변동성 (5% 이상): 손실 임계값 -15%, 이익 보호 +12%
- 중변동성 (3-5%): 손실 임계값 -12%, 이익 보호 +15%  
- 저변동성 (3% 미만): 손실 임계값 -10%, 이익 보호 +18%

**현재 적용 임계값:** 변동성에 따라 자동 조정됨

1. **실시간 시장 대응 (시간 제약 없음)**
   현재 보유: {(get_utc_now() - parse_db_timestamp(trade['timestamp'])).total_seconds() / 3600:.1f}시간
   트레이딩 스타일: {trade.get('trading_style', 'DAY_TRADING')}
   
   **🔥 AI 판단 최우선 원칙:**
   - 시장 신호가 청산을 권고하면 즉시 실행
   - 보유시간과 무관하게 이익 보호 및 손실 방지 우선
   - 진입 직후라도 급변 시 대응 가능
   
   **더 이상 최소 보유시간 제약 없음!**

2. **즉시 청산 조건 (AI 판단 기준)**
   - 손실 -12% 초과 (큰 손실 방지) 
   - 설정된 손절가 정확히 도달
   - 3개 이상 타임프레임에서 강력한 추세 반전 + 손실 -8% 초과

3. **일반 청산 조건 (AI 종합 판단)**
   💰 **수익 상황 (이익 보호 우선):**
   - 수익 +20% 이상 + 추세 약화 신호 → 즉시 익절 고려
   - 수익 +15% 이상 + 2개 이상 타임프레임 약화 → 익절 고려  
   - 수익 +10% 이상 + 강한 반전 신호 → 익절 고려
   - 목표 수익 달성 + 추세 약화 → 익절
   
   📉 **손실 상황:**
   - 손실 -8% 초과 + 반등 근거 부족 → 청산 고려
   - 여러 타임프레임 추세 반전 + 손실 -5% 초과 → 청산 고려

4. **변동성 고려 (완화된 기준)**
   - 암호화폐 특성상 ±8% 변동은 정상 (기존 ±5%에서 완화)
   - **수익 중에는 단기 변동 무시, 이익 보호에 집중**
   - 손실 중에만 추세 반전에 민감하게 반응

5. **수익 실현 (Take-Profit) - 이익 보호 중심**
   🔥 **변동성 장세 대응 - 이익 보호 최우선**
   - 수익 +25% 이상 → 추세 약화 첫 신호에서 즉시 익절
   - 수익 +20% 이상 → 2개 타임프레임 약화 시 익절
   - 수익 +15% 이상 → 3개 타임프레임 약화 시 익절  
   - 수익 +10% 이상 → 강한 반전 + 거래량 급증 시 익절
   - 목표가 도달 + 추세 약화 신호 → 전량 청산
   
   💡 **핵심**: 큰 이익일수록 보수적으로 보호, TP 못 도달해도 이익 확보
   
6. **홀딩 vs 청산 결정 (수익 우선)**
   - **수익 중**: 이익 보호가 최우선 → 의심스러우면 익절
   - **손실 중**: 반등 근거 명확 → 홀드, 불명확 → 청산

**핵심: 수익 중에는 보수적으로 이익 보호, 손실 중에는 적극적 반등 대기**

📊 **구체적 판단 기준 (수익률별)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**수익 +30% 이상**: 어떤 약화 신호든 즉시 익절 (이익 보호 절대 우선)
**수익 +25% 이상**: RSI 과매수 + MACD 약화 → 즉시 익절
**수익 +20% 이상**: 2개 타임프레임 추세 약화 → 익절 고려
**수익 +15% 이상**: 3개 타임프레임 추세 약화 → 익절 고려
**수익 +10% 이상**: 강한 반전 신호만 익절 고려
**수익 +5~10%**: 목표가 근처에서만 익절, 기본은 홀드
**수익 0~5%**: 적극적 홀드, 반전 신호 무시

**손실 -5% 이내**: 적극적 홀드, 반등 대기
**손실 -5~10%**: 반등 근거 있으면 홀드, 없으면 청산 고려
**손실 -10~15%**: 명확한 반등 신호만 홀드, 기본은 청산
**손실 -15% 이상**: 즉시 청산 (변동성 무관)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JSON 형식:
{{
  "close_position": true/false,
  "reason": "이유 (100자 이내)",
  "adjust_sl": 숫자 or "KEEP",
  "adjust_tp": 숫자 or "KEEP",
  "confidence": 0~100,
  "analyzed_timeframes": "분석한 타임프레임 (예: 15m, 1h, 4h)"
}}
"""
        
        response_text = call_ai_model(prompt, temperature=0.6)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            decision = json.loads(json_str)
            
            # 분석 타임프레임 출력
            if 'analyzed_timeframes' in decision:
                print(f"   📊 분석 타임프레임: {decision['analyzed_timeframes']}")
            
            return decision
        
        return {"close_position": False, "reason": "파싱 실패"}
    except Exception as e:
        print(f"⚠️ 포지션 관리 AI 오류: {e}")
        return {"close_position": False, "reason": f"오류: {str(e)[:30]}"}

# ===== 포지션 사이징 =====

def calculate_position_size(available_balance: float, ai_investment_pct: float, volatility: float, open_positions: int, trading_style: str = 'DAY_TRADING') -> float:
    """🆕 동적 균등 분할 기반 스마트 포지션 사이징"""
    config = LIVE_TRADING_CONFIG
    max_positions = config['MAX_CONCURRENT_POSITIONS']
    
    print(f"   💰 가용 잔고: ${available_balance:,.2f}")
    print(f"   📊 현재 포지션: {open_positions}/{max_positions}")
    
    # 🆕 핵심: 동적 균등 분할 방식
    remaining_positions = max_positions - open_positions
    if remaining_positions <= 0:
        print(f"   ❌ 포지션 풀 - 더 이상 진입 불가")
        return 0
    
    # 가용 잔고를 남은 포지션 수로 균등 분할
    equal_split_amount = available_balance / remaining_positions
    print(f"   📊 균등 분할: ${available_balance:,.2f} ÷ {remaining_positions} = ${equal_split_amount:,.2f}")
    
    # 🆕 1. Kelly Criterion 기반 조정 (선택적)
    strategy_perf = get_strategy_performance(trading_style)
    
    # 🆕 동적 균등 분할 모드: 기본적으로 균등 분할 금액 100% 사용
    base_investment = equal_split_amount
    
    if strategy_perf and strategy_perf['total_trades'] >= 5:
        # 충분한 데이터가 있으면 Kelly Criterion으로 조정
        win_rate = strategy_perf['win_rate']
        avg_win = abs(strategy_perf['avg_win'])
        avg_loss = abs(strategy_perf['avg_loss'])
        
        kelly_pct = calculate_kelly_criterion(win_rate, avg_win, avg_loss) * 100
        
        print(f"   📊 Kelly Criterion: {kelly_pct:.1f}% (승률:{win_rate:.1f}%, R/R:{avg_win/avg_loss if avg_loss > 0 else 0:.2f})")
        
        # Kelly가 30% 이하면 보수적으로 조정 (너무 낮은 경우만)
        if kelly_pct < 30:
            kelly_multiplier = max(0.5, kelly_pct / 100)  # 최소 50%는 보장
            base_investment *= kelly_multiplier
            print(f"   💡 Kelly 조정: {kelly_multiplier:.2f}x → ${base_investment:,.2f}")
    else:
        # 🆕 데이터 부족 시에도 100% 사용 (AI 투자비율 무시)
        print(f"   💡 동적 균등 분할: 100% 사용 → ${base_investment:,.2f}")
    
    # 🆕 2. 연속 손실 페널티 (완화)
    consecutive_losses = get_consecutive_losses()
    if consecutive_losses >= 5:
        penalty = 0.7  # 70%로 축소 (기존 50%)
        base_investment *= penalty
        print(f"   ⚠️ 연속 손실 {consecutive_losses}회 → {penalty}x 투자")
        
        # 리스크 조정 로그
        log_risk_adjustment(
            adjustment_type='POSITION_SIZE',
            previous_value=equal_split_amount,
            new_value=base_investment,
            reason=f"연속 손실 {consecutive_losses}회",
            consecutive_losses=consecutive_losses,
            current_drawdown=get_current_drawdown()
        )
    elif consecutive_losses >= 3:
        penalty = 0.85  # 85%로 축소 (기존 75%)
        base_investment *= penalty
        print(f"   ⚠️ 연속 손실 {consecutive_losses}회 → {penalty}x 투자")
    
    # 🆕 3. 드로다운 페널티 (완화)
    current_drawdown = get_current_drawdown()
    if current_drawdown > 300:  # 기존 200 → 300
        penalty = 0.7  # 기존 0.5 → 0.7
        base_investment *= penalty
        print(f"   ⚠️ 드로다운 ${current_drawdown:.2f} → {penalty}x 투자")
    elif current_drawdown > 150:  # 기존 100 → 150
        penalty = 0.85  # 기존 0.75 → 0.85
        base_investment *= penalty
        print(f"   ⚠️ 드로다운 ${current_drawdown:.2f} → {penalty}x 투자")
    
    # 4. 최대/최소 제한 적용 (안전장치)
    max_investment = available_balance * (config['MAX_POSITION_SIZE_PCT'] / 100)
    min_investment = available_balance * (config['MIN_POSITION_SIZE_PCT'] / 100)
    
    investment = min(base_investment, max_investment)
    investment = max(investment, min_investment)
    
    # 5. 변동성 기반 조정 (완화)
    if config['VOLATILITY_BASED_SIZING']:
        if volatility > config['HIGH_VOLATILITY_THRESHOLD']:
            # 고변동성 = 리스크 높음 = 투자 줄임 (기존 0.6x → 0.8x)
            multiplier = 0.8
            investment *= multiplier
            print(f"   📉 고변동성 ({volatility:.1f}%) → {multiplier}x 투자")
        elif volatility < config['HIGH_VOLATILITY_THRESHOLD'] / 2:
            # 저변동성 = 리스크 낮음 = 투자 늘림 (기존 1.5x → 1.2x)
            multiplier = 1.2
            investment *= multiplier
            print(f"   📈 저변동성 ({volatility:.1f}%) → {multiplier}x 투자")
    
    # 6. 가용 잔고 초과 방지 (동적 균등 분할은 100% 사용 가능)
    if investment > available_balance:
        investment = available_balance
        print(f"   ⚠️ 가용 잔고 부족 → ${investment:.2f}로 조정")
    
    print(f"   ✅ 최종 투자금: ${investment:,.2f}")
    
    return investment

# ===== 실제 거래 실행 =====

def execute_live_trade(coin_data: Dict, decision: Dict, available_balance: float) -> bool:
    """실제 거래 실행"""
    
    coin = coin_data['coin']
    symbol = f"{coin}/USDT:USDT"
    
    print(f"\n{'='*70}")
    print(f"🔴 [실거래] AI 거래 실행: {coin} {decision['direction']}")
    print(f"{'='*70}")
    
    try:
        # 🆕 SHORT + SWING_TRADING 조합 차단
        if decision['direction'] == 'SHORT' and decision.get('trading_style') == 'SWING_TRADING':
            print(f"   ❌ SHORT + SWING_TRADING 조합은 금지되어 있습니다!")
            print(f"   ℹ️ SHORT는 SCALPING 또는 DAY_TRADING만 허용됩니다.")
            print(f"   💡 AI 오류: 프롬프트 규칙 위반")
            return False
        
        # 🆕 SHORT 보유 시간 제한 확인
        if decision['direction'] == 'SHORT':
            holding_time = decision.get('holding_time_estimate', '')
            trading_style = decision.get('trading_style', '')
            
            # SWING_TRADING 감지 (보유 시간으로도 체크)
            if any(keyword in holding_time.lower() for keyword in ['일', 'day', '주', 'week', '장기']):
                print(f"   ❌ SHORT 포지션은 장기 보유 불가!")
                print(f"   ℹ️ 예상 보유 시간: {holding_time}")
                print(f"   💡 SHORT는 최대 24시간 이내만 허용")
                return False
            
            print(f"   ✅ SHORT 검증 통과: {trading_style} ({holding_time})")
        
        # 이미 오픈된 포지션 확인
        open_positions = get_open_positions()
        for pos in open_positions:
            if pos['symbol'] == symbol:
                print(f"   ⚠️ 이미 {coin} 포지션이 열려있습니다. 스킵.")
                return False
        
        # 포지션 수 확인
        if len(open_positions) >= LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']:
            print(f"   ⚠️ 최대 포지션 수({LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}) 도달. 스킵.")
            return False
        
        # 변동성 가져오기
        volatility = decision.get('volatility', 3.0)
        
        print(f"   가용 잔고: ${available_balance:,.2f}")
        print(f"   오픈 포지션: {len(open_positions)}/{LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개")
        print(f"   변동성: {volatility:.1f}%")
        
        # 포지션 사이징
        investment = calculate_position_size(
            available_balance,
            decision['investment_percentage'],
            volatility,
            len(open_positions),
            decision.get('trading_style', 'DAY_TRADING')  # 🆕 전략 전달
        )
        
        actual_investment_pct = (investment / available_balance * 100) if available_balance > 0 else 0
        
        leverage = decision['leverage']
        current_price = coin_data['price']
        
        # 최소 주문 수량 확인
        markets = exchange.load_markets()
        market_info = markets.get(symbol, {})
        min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0.001)
        
        amount = (investment * leverage) / current_price
        
        if amount < min_amount:
            print(f"   ⚠️ 주문 수량({amount:.8f})이 최소 수량({min_amount:.8f})보다 작습니다. 스킵.")
            return False
        
        print(f"   💰 AI 제안: {decision['investment_percentage']:.0f}% → 실제: {actual_investment_pct:.1f}%")
        print(f"   투자금: ${investment:,.2f}")
        print(f"   레버리지: {leverage}x")
        print(f"   수량: {amount:.8f}")
        
        # 사용자 확인 (옵션)
        # confirm = input(f"\n   ⚠️ 실제 거래를 진행하시겠습니까? (y/N): ").strip().lower()
        # if confirm != 'y':
        #     print("   ❌ 거래 취소")
        #     return False
        
        # 실제 주문 실행
        side = 'buy' if decision['direction'] == 'LONG' else 'sell'
        order = create_market_order(symbol, side, amount, leverage)
        
        entry_price = order.get('average', order.get('price', current_price))
        print(f"   ✅ 진입: ${entry_price:,.4f}")
        
        # 손절/익절 주문
        if decision['direction'] == 'LONG':
            sl_price = entry_price * (1 - decision['sl_percentage'] / 100)
            tp_price = entry_price * (1 + decision['tp_percentage'] / 100)
            sl_side = 'sell'
            tp_side = 'sell'
        else:
            sl_price = entry_price * (1 + decision['sl_percentage'] / 100)
            tp_price = entry_price * (1 - decision['tp_percentage'] / 100)
            sl_side = 'buy'
            tp_side = 'buy'
        
        try:
            create_stop_loss_order(symbol, sl_side, amount, sl_price)
        except Exception as e:
            print(f"   ⚠️ 손절 주문 실패: {e}")
        
        try:
            create_take_profit_order(symbol, tp_side, amount, tp_price)
        except Exception as e:
            print(f"   ⚠️ 익절 주문 실패: {e}")
        
        # DB 저장
        trade_data = {
            'coin_symbol': coin,
            'action': decision['direction'].lower(),
            'entry_price': entry_price,
            'amount': amount,
            'leverage': leverage,
            'investment_amount': investment,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'trading_style': decision.get('trading_style', 'DAY_TRADING'),
            'holding_time_estimate': decision.get('holding_time_estimate', 'Unknown'),
            'ai_reasoning': decision['reasoning'],
            'market_conditions': json.dumps(decision.get('key_factors', [])),
            'binance_order_id': order['id'],
            'confidence_score': decision.get('confidence', 0)  # 🆕 신뢰도 추가
        }
        
        trade_id = save_trade_to_db(trade_data)
        
        decision_data = {
            'coin_symbol': coin,
            'decision_type': 'ENTRY',
            'direction': decision['direction'],
            'leverage': leverage,
            'investment_percentage': decision['investment_percentage'],
            'sl_percentage': decision['sl_percentage'],
            'tp_percentage': decision['tp_percentage'],
            'confidence_score': decision['confidence'],
            'reasoning': decision['reasoning'],
            'market_data': json.dumps(coin_data),
            'related_trade_id': trade_id
        }
        
        save_ai_decision(decision_data)
        print(f"   ✅ Trade ID: {trade_id}")
        print(f"{'='*70}\n")
        
        return True
    except Exception as e:
        print(f"   ❌ 거래 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_manual_trades() -> List[Dict]:
    """수동거래 목록 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, coin_symbol, action, entry_price, amount, leverage,
               investment_amount, timestamp, status
        FROM trades
        WHERE manual_trade = 1
        ORDER BY timestamp DESC
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    manual_trades = []
    for row in rows:
        manual_trades.append({
            'id': row[0],
            'coin_symbol': row[1],
            'action': row[2],
            'entry_price': row[3],
            'amount': row[4],
            'leverage': row[5],
            'investment_amount': row[6],
            'timestamp': row[7],
            'status': row[8]
        })
    
    return manual_trades

def display_manual_trades_status():
    """수동거래 현황 표시"""
    manual_trades = list_manual_trades()
    
    if not manual_trades:
        print("📝 등록된 수동거래가 없습니다.")
        return
    
    print(f"\n{'='*80}")
    print(f"📝 수동거래 현황 ({len(manual_trades)}개)")
    print(f"{'='*80}")
    
    open_count = 0
    closed_count = 0
    
    for trade in manual_trades:
        status_emoji = "🟢" if trade['status'] == 'OPEN' else "🔴"
        print(f"   {status_emoji} ID {trade['id']}: {trade['coin_symbol']} {trade['action']} "
              f"${trade['entry_price']:,.4f} ({trade['status']})")
        
        if trade['status'] == 'OPEN':
            open_count += 1
        else:
            closed_count += 1
    
    print(f"{'='*80}")
    print(f"   🟢 진행중: {open_count}개")
    print(f"   🔴 종료됨: {closed_count}개")
    print(f"{'='*80}")

# ===== 바이낸스 트레일링 스탑 기능 =====

def set_binance_trailing_stop(symbol: str, side: str, amount: float, entry_price: float, leverage: int, activation_pct: float = 20.0, callback_rate: float = 7.0) -> bool:
    """바이낸스 네이티브 트레일링 스탑 설정
    
    Args:
        symbol: 거래 심볼 (예: ZEC/USDT:USDT)
        side: 'long' 또는 'short'
        amount: 수량
        entry_price: 진입가
        leverage: 레버리지
        activation_pct: 활성화 수익률 (기본 20%)
        callback_rate: 콜백 비율 (기본 7%)
    
    Returns:
        성공 여부
    """
    try:
        # 활성화 가격 계산 (진입가 대비 +20%)
        if side == 'long':
            activation_price = entry_price * (1 + activation_pct / 100)
            close_side = 'sell'
        else:
            activation_price = entry_price * (1 - activation_pct / 100)
            close_side = 'buy'
        
        print(f"   🎯 바이낸스 트레일링 스탑 설정:")
        print(f"      진입가: ${entry_price:.4f}")
        print(f"      활성화가: ${activation_price:.4f} (+{activation_pct}%)")
        print(f"      콜백: {callback_rate}%")
        
        # 바이낸스 트레일링 스탑 주문
        order = exchange.create_order(
            symbol=symbol,
            type='TRAILING_STOP_MARKET',
            side=close_side,
            amount=amount,
            params={
                'activationPrice': activation_price,
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'workingType': 'MARK_PRICE'  # 마크 가격 기준
            }
        )
        
        print(f"   ✅ 트레일링 스탑 주문 완료: {order.get('id', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"   ❌ 트레일링 스탑 설정 실패: {e}")
        return False

def check_and_set_trailing_stops():
    """오픈 포지션을 체크하고 수익률 20% 이상이면 트레일링 스탑 설정"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # 오픈 포지션 조회
        c.execute("""
            SELECT id, coin_symbol, action, entry_price, amount, leverage, 
                   investment_amount, timestamp
            FROM trades 
            WHERE status = 'OPEN'
            AND manual_trade = 0
        """)
        
        trades = c.fetchall()
        conn.close()
        
        if not trades:
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 트레일링 스탑 체크 (5분마다)")
        print(f"{'='*80}")
        
        for trade_data in trades:
            try:
                trade_id, coin, action, entry_price, amount, leverage, investment, timestamp = trade_data
                
                symbol = f"{coin}/USDT:USDT"
                
                # 현재가 조회
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 수익률 계산 (레버리지 포함)
                if action == 'long':
                    price_change_pct = (current_price - entry_price) / entry_price
                else:
                    price_change_pct = (entry_price - current_price) / entry_price
                
                profit_pct = price_change_pct * leverage * 100
                
                print(f"\n   {coin} ({action.upper()}):")
                print(f"   ├─ 진입: ${entry_price:.4f} → 현재: ${current_price:.4f}")
                print(f"   └─ 수익률: {profit_pct:+.2f}% (Lev {leverage}x)")
                
                # 20% 이상 수익이면 트레일링 스탑 설정
                if profit_pct >= 20.0:
                    print(f"   🚀 수익률 20% 초과! 트레일링 스탑 전환")
                    
                    # 🔥 기존 TP/SL 주문 모두 삭제
                    print(f"   🧹 기존 TP/SL 주문 삭제 중...")
                    cancel_all_tpsl_orders(symbol)
                    time.sleep(0.5)
                    
                    # 다시 한 번 확인 (중요!)
                    open_orders = exchange.fetch_open_orders(symbol)
                    tpsl_orders = [o for o in open_orders if o.get('reduceOnly') == True or 
                                   o['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'STOP_LOSS_MARKET']]
                    
                    if tpsl_orders:
                        print(f"   ⚠️ TP/SL 주문 {len(tpsl_orders)}개 남아있음, 재삭제...")
                        for order in tpsl_orders:
                            try:
                                exchange.cancel_order(order['id'], symbol)
                                print(f"      ✅ 주문 {order['id']} 삭제")
                            except:
                                pass
                        time.sleep(0.5)
                    
                    # 바이낸스 트레일링 스탑 설정
                    success = set_binance_trailing_stop(
                        symbol=symbol,
                        side=action,
                        amount=amount,
                        entry_price=entry_price,
                        leverage=leverage,
                        activation_pct=20.0,
                        callback_rate=7.0
                    )
                    
                    if success:
                        # DB에 트레일링 스탑 설정 기록
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                        c.execute("""
                            UPDATE trades
                            SET ai_reasoning = COALESCE(ai_reasoning, '') || ?
                            WHERE id = ?
                        """, (f'\n[트레일링 스탑] 수익률 {profit_pct:.1f}% 도달, 바이낸스 트레일링 스탑 설정 (콜백 7%)', trade_id))
                        conn.commit()
                        conn.close()
                        
                        print(f"   ✅ {coin} 트레일링 스탑 전환 완료")
                
            except Exception as e:
                print(f"   ❌ {coin if 'coin' in locals() else 'Unknown'} 트레일링 스탑 체크 오류: {e}")
                continue
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"   ❌ 트레일링 스탑 체크 오류: {e}")

def trailing_stop_checker_loop():
    """5분마다 트레일링 스탑 체크 (별도 스레드)"""
    print(f"🎯 바이낸스 트레일링 스탑 체커 시작 (5분마다)")
    
    while True:
        try:
            check_and_set_trailing_stops()
            time.sleep(300)  # 5분 대기
            
        except Exception as e:
            print(f"❌ 트레일링 스탑 체커 오류: {e}")
            time.sleep(60)

# 🆕 수동거래 쉬운 사용을 위한 전역 함수들
def add_manual_long(coin: str, price: float, amount: float, leverage: int = 1):
    """수동 롱 포지션 등록"""
    return register_manual_trade(coin, "LONG", price, amount, leverage)

def add_manual_short(coin: str, price: float, amount: float, leverage: int = 1):
    """수동 숏 포지션 등록"""
    return register_manual_trade(coin, "SHORT", price, amount, leverage)

def protect_trade(trade_id: int):
    """거래를 수동거래로 보호 (AI 청산 방지)"""
    return mark_existing_trade_as_manual(trade_id)

def unprotect_trade(trade_id: int):
    """거래 보호 해제 (AI 청산 허용)"""
    return unmark_manual_trade(trade_id)

def check_trade_protection(trade_id: int) -> bool:
    """거래 보호 상태 확인"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("SELECT coin_symbol, action, manual_trade FROM trades WHERE id = ? AND status = 'OPEN'", (trade_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            print(f"❌ 거래 ID {trade_id}를 찾을 수 없거나 이미 종료되었습니다.")
            return False
        
        coin_symbol, action, manual_trade = result
        is_protected = manual_trade == 1
        
        status = "🛡️ 보호됨 (AI 청산 차단)" if is_protected else "🤖 AI 관리"
        print(f"거래 ID {trade_id} ({coin_symbol} {action}): {status}")
        
        return is_protected
        
    except Exception as e:
        print(f"❌ 보호 상태 확인 오류: {e}")
        return False

def show_all_trades_protection():
    """모든 진행중인 거래의 보호 상태 표시"""
    open_trades = get_all_open_trades()
    
    if not open_trades:
        print("📝 진행중인 거래가 없습니다.")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 전체 거래 보호 상태")
    print(f"{'='*80}")
    
    for trade in open_trades:
        is_manual = trade.get('manual_trade', 0) == 1
        status = "🛡️ 수동거래" if is_manual else "🤖 AI거래"
        print(f"   ID {trade['id']:2d}: {trade['coin_symbol']:6s} {trade['action']:5s} | {status}")
    
    print(f"{'='*80}")

def manage_live_positions():
    """실제 포지션 관리"""
    
    db_trades = get_all_open_trades()
    if not db_trades:
        # 포지션 없을 때도 고아 주문 체크
        cleanup_orphaned_orders()
        return
    
    print(f"\n{'='*70}")
    print(f"🔴 [실거래] AI 포지션 관리 ({len(db_trades)}개)")
    print(f"{'='*70}")
    
    # 실제 바이낸스 포지션 조회
    live_positions = get_open_positions()
    
    # 🆕 고아 TP/SL 주문 정리
    cleanup_orphaned_orders()
    
    for trade in db_trades:
        coin = trade['coin_symbol']
        symbol = f"{coin}/USDT:USDT"
        
        # 바이낸스에서 실제 포지션 확인
        live_pos = None
        for pos in live_positions:
            if pos['symbol'] == symbol:
                live_pos = pos
                break
        
        # 포지션이 이미 청산되었는지 확인
        if not live_pos:
            # 🛡️ 수동거래 보호 - 바이낸스에 없어도 DB 업데이트 안함
            is_manual_trade = trade.get('manual_trade', 0) == 1
            
            if is_manual_trade:
                print(f"\n   🛡️ {coin}: 수동거래 - 바이낸스 포지션 없어도 DB 유지")
                print(f"   ⚠️ 수동으로 청산했다면 unprotect_trade({trade['id']}) 호출하여 DB 정리하세요")
                continue
            
            print(f"\n   ⚠️ {coin}: DB에는 있으나 바이낸스에 포지션 없음 (AI거래)")
            
            # 🆕 청산된 포지션의 TP/SL 주문 즉시 취소
            cancel_all_tpsl_orders(symbol)
            
            # 바이낸스에서 청산된 경우 PnL 조회 후 DB 업데이트
            try:
                pnl_info = get_position_pnl(symbol)
                realized_pnl = pnl_info.get('realizedPnl', 0)
                
                # 청산 이유 판단
                if realized_pnl > 0:
                    close_reason = "익절 목표가 도달 (TP)"
                elif realized_pnl < 0:
                    close_reason = "손절 라인 도달 (SL)"
                else:
                    close_reason = "청산 완료 (무손익)"
                
                close_data = {
                    'close_price': 0,
                    'pnl': realized_pnl,
                    'pnl_percentage': 0,
                    'binance_pnl': realized_pnl,
                    'binance_close_price': 0,
                    'ai_close_reason': close_reason
                }
                update_trade_close(trade['id'], close_data)
                print(f"   ✅ DB 업데이트 완료 (실현 PnL: ${realized_pnl:,.2f})")
            except Exception as e:
                print(f"   ⚠️ PnL 조회 실패: {e}")
                # PnL 조회 실패해도 DB 업데이트는 해야 함
                try:
                    close_data = {
                        'close_price': 0,
                        'pnl': 0,
                        'pnl_percentage': 0,
                        'binance_pnl': 0,
                        'binance_close_price': 0
                    }
                    update_trade_close(trade['id'], close_data)
                    print(f"   ✅ DB 업데이트 완료 (PnL 조회 실패로 0으로 설정)")
                except Exception as db_error:
                    print(f"   ❌ DB 업데이트 실패: {db_error}")
            continue
        
        try:
            current_price = live_pos['markPrice']
            unrealized_pnl = live_pos['unrealizedPnl']
            pnl_percentage = live_pos['percentage']
            
            print(f"\n{'─'*70}")
            print(f"   {coin}")
            print(f"{'─'*70}")
            print(f"   방향: {live_pos['side'].upper()}")
            print(f"   진입가: ${live_pos['entryPrice']:,.4f}")
            print(f"   현재가: ${current_price:,.4f}")
            print(f"   미실현 PnL: ${unrealized_pnl:+,.2f} ({pnl_percentage:+.2f}%)")
            
            # 시장 데이터 수집
            print(f"   🔍 시장 데이터 수집 중...")
            market_data = fetch_comprehensive_market_data(symbol)
            
            if not market_data:
                print(f"   ⚠️ 시장 데이터 없음")
                continue
            
            # AI 포지션 관리 분석
            print(f"   🤖 AI 분석 중...")
            decision = ai_position_management(trade, market_data, current_price)
            print(f"   신뢰도: {decision.get('confidence', 0)}%")
            print(f"   판단: {decision.get('reason', 'N/A')}")
            
            # 청산 판단
            if decision.get('close_position'):
                # 🛡️ 수동거래 체크 - AI 청산 방지
                is_manual_trade = trade.get('manual_trade', 0) == 1
                
                if is_manual_trade:
                    print(f"\n   🛡️ 수동거래 보호 - AI 청산 차단")
                    print(f"   AI 권장: {decision.get('reason', 'N/A')}")
                    print(f"   ⚠️ 수동으로 관리하세요!")
                    continue  # 청산하지 않고 다음 포지션으로
                
                print(f"\n   🔴 AI 청산 권장")
                
                try:
                    # 실제 청산 실행
                    close_result = close_position(
                        symbol,
                        live_pos['side'],
                        live_pos['contracts']
                    )
                    
                    # close_result가 None이거나 order가 None이면 이미 청산된 것
                    if close_result is None or close_result.get('order') is None:
                        print(f"   ⚠️ 포지션이 이미 청산되었습니다 (TP/SL 체결?)")
                        # PnL만 있으면 DB 업데이트
                        if close_result and close_result.get('pnl') != 0:
                            close_data = {
                                'close_price': live_pos['entryPrice'],  # 진입가 사용
                                'pnl': close_result['pnl'],
                                'pnl_percentage': (close_result['pnl'] / trade['investment'] * 100) if trade.get('investment', 0) > 0 else 0,
                                'binance_pnl': close_result['pnl'],
                                'binance_close_price': live_pos['entryPrice'],
                                'ai_close_reason': decision.get('reason', 'AI 조기 청산')  # 🆕 AI 이유 추가
                            }
                            update_trade_close(trade['id'], close_data)
                            
                            # 전략 성과 업데이트
                            trading_style = trade.get('trading_style', 'DAY_TRADING')
                            is_win = close_result['pnl'] > 0
                            update_strategy_performance(trading_style, close_result['pnl'], is_win)
                            
                            print(f"   ✅ DB 업데이트 완료 (실현 PnL: ${close_result['pnl']:+,.2f})")
                        continue
                    
                    # 🆕 청산 성공 후 TP/SL 주문은 이미 close_position에서 취소됨
                    
                    # DB 업데이트 (바이낸스 실제 PnL 사용 + 🆕 AI 청산 이유)
                    close_data = {
                        'close_price': close_result['close_price'],
                        'pnl': close_result['pnl'],
                        'pnl_percentage': (close_result['pnl'] / trade['investment'] * 100) if trade.get('investment', 0) > 0 else 0,
                        'binance_pnl': close_result['pnl'],
                        'binance_close_price': close_result['close_price'],
                        'ai_close_reason': decision.get('reason', 'AI 조기 청산')  # 🆕 AI 이유 추가
                    }
                    
                    update_trade_close(trade['id'], close_data)
                    
                except Exception as close_error:
                    print(f"   ❌ 청산 오류: {close_error}")
                    # 청산 실패해도 계속 진행 (다음 포지션 체크)
                    # 하지만 이미 청산되었을 수도 있으므로 PnL 확인해보기
                    try:
                        pnl_info = get_position_pnl(symbol)
                        if pnl_info.get('realizedPnl', 0) != 0:
                            close_data = {
                                'close_price': current_price,
                                'pnl': pnl_info.get('realizedPnl', 0),
                                'pnl_percentage': (pnl_info.get('realizedPnl', 0) / trade['investment'] * 100) if trade.get('investment', 0) > 0 else 0,
                                'binance_pnl': pnl_info.get('realizedPnl', 0),
                                'binance_close_price': current_price
                            }
                            update_trade_close(trade['id'], close_data)
                            print(f"   ✅ 실제로는 청산 완료됨 (PnL: ${pnl_info.get('realizedPnl', 0):+,.2f})")
                    except Exception as pnl_error:
                        print(f"   ⚠️ PnL 확인도 실패: {pnl_error}")
                    continue
                
                # 🆕 전략 성과 업데이트
                trading_style = trade.get('trading_style', 'DAY_TRADING')
                is_win = close_result['pnl'] > 0
                update_strategy_performance(trading_style, close_result['pnl'], is_win)
                
                # 🆕 패턴 학습 업데이트
                pattern_desc = f"{coin} {trade['action'].upper()} - "
                
                # 진입 시 지표 상태 (AI reasoning에서 추출)
                ai_reasoning = trade.get('ai_reasoning', '')
                if 'RSI' in ai_reasoning:
                    pattern_desc += "RSI신호 + "
                if 'MACD' in ai_reasoning or '골든크로스' in ai_reasoning:
                    pattern_desc += "MACD골든 + "
                if 'ADX' in ai_reasoning:
                    pattern_desc += "ADX강세 + "
                if '볼린저' in ai_reasoning or 'BB' in ai_reasoning:
                    pattern_desc += "볼린저 + "
                
                pattern_desc += f"{trading_style}"
                
                # 시장 상황 저장
                market_conditions = {
                    'coin': coin,
                    'action': trade['action'],
                    'entry_price': trade['entry_price'],
                    'close_price': close_result['close_price'],
                    'leverage': trade.get('leverage', 1),
                    'trading_style': trading_style
                }
                
                update_pattern_learning(
                    pattern_description=pattern_desc,
                    pattern_type=trading_style,
                    success=is_win,
                    pnl=close_result['pnl'],
                    market_conditions=market_conditions
                )
                
                print(f"   ✅ 청산 완료")
                print(f"   ├─ 실현 PnL: ${close_result['pnl']:+,.2f}")
                print(f"   ├─ Trade ID: {trade['id']}")
                print(f"   └─ 학습: {trading_style} {'승리' if is_win else '손실'} 패턴 기록")
                
                # AI 결정 저장
                decision_data = {
                    'coin_symbol': coin,
                    'decision_type': 'EXIT',
                    'direction': None,
                    'leverage': None,
                    'investment_percentage': None,
                    'sl_percentage': None,
                    'tp_percentage': None,
                    'confidence_score': decision.get('confidence', 0),
                    'reasoning': decision.get('reason', ''),
                    'market_data': '',
                    'related_trade_id': trade['id']
                }
                save_ai_decision(decision_data)
            else:
                print(f"   ✅ 보유 유지")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ {coin} 관리 오류: {e}")
            import traceback
            traceback.print_exc()

# ===== 성과 리뷰 =====

def analyze_ai_decision_patterns(days: int = 7) -> Dict:
    """AI 결정 근거 패턴 분석"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # 최근 N일 봇 거래의 결정 근거 조회
        cutoff_date = (get_utc_now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            SELECT coin_symbol, action, pnl, confidence_score, ai_reasoning, 
                   reasoning, pattern_description, entry_price, exit_price,
                   close_timestamp, binance_pnl
            FROM trades 
            WHERE status = 'CLOSED' 
            AND close_timestamp >= ?
            ORDER BY close_timestamp DESC
        ''', (cutoff_date,))
        
        trades = c.fetchall()
        conn.close()
        
        if not trades:
            return {'total_trades': 0, 'patterns': {}}
        
        print(f"   🔍 AI 결정 근거 분석 중... ({len(trades)}건)")
        
        # 패턴 분석
        success_patterns = []
        failure_patterns = []
        confidence_analysis = {'high': [], 'medium': [], 'low': []}
        reasoning_frequency = {}
        
        for trade in trades:
            coin, action, pnl, confidence, ai_reasoning, reasoning, pattern, entry_price, exit_price, close_time, binance_pnl = trade
            
            # 실제 PnL 사용 (바이낸스 우선)
            actual_pnl = binance_pnl if binance_pnl is not None and binance_pnl != 0 else (pnl if pnl else 0)
            
            # confidence가 None이면 기본값 50 사용
            safe_confidence = confidence if confidence is not None else 50
            
            trade_data = {
                'coin': coin,
                'action': action,
                'pnl': actual_pnl,
                'confidence': safe_confidence,
                'ai_reasoning': ai_reasoning or 'N/A',
                'reasoning': reasoning or 'N/A',
                'pattern': pattern or 'N/A',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'timestamp': close_time
            }
            
            # 성공/실패 패턴 분류
            if actual_pnl > 0:
                success_patterns.append(trade_data)
            else:
                failure_patterns.append(trade_data)
            
            # 신뢰도별 분류 (None 체크 추가)
            if safe_confidence >= 80:
                confidence_analysis['high'].append(trade_data)
            elif safe_confidence >= 60:
                confidence_analysis['medium'].append(trade_data)
            else:
                confidence_analysis['low'].append(trade_data)
            
            # 추론 빈도 분석
            if reasoning and reasoning != 'N/A':
                reasoning_key = reasoning[:50]  # 처음 50자만 사용
                reasoning_frequency[reasoning_key] = reasoning_frequency.get(reasoning_key, 0) + 1
        
        return {
            'total_trades': len(trades),
            'success_patterns': success_patterns,
            'failure_patterns': failure_patterns,
            'confidence_analysis': confidence_analysis,
            'reasoning_frequency': reasoning_frequency
        }
        
    except Exception as e:
        print(f"   ❌ AI 결정 근거 분석 오류: {e}")
        return {'total_trades': 0, 'patterns': {}}

def ai_performance_review():
    """AI 성과 리뷰 - 결정 근거 기반 분석"""
    
    print(f"\n{'🔍'*10} 성과 리뷰 (AI 결정 근거 분석) {'🔍'*10}")
    
    # 🆕 봇 거래 성과 분석 (DB 우선, 바이낸스 보조)
    try:
        binance_performance = calculate_binance_performance(7)
        use_binance_data = binance_performance['total_trades'] > 0
        data_source = binance_performance.get('data_source', 'Unknown')
    except Exception as e:
        print(f"   ⚠️ 성과 데이터 조회 실패: {e}")
        use_binance_data = False
        binance_performance = None
        data_source = 'N/A'
    
    # DB 데이터도 참고용으로 조회
    try:
        db_performance = get_recent_performance(7)
    except Exception as e:
        print(f"   ⚠️ DB 데이터 조회 실패: {e}")
        db_performance = None
    
    # 🆕 성과 데이터 우선순위: DB 봇 거래 > 바이낸스 API > DB 일반
    if use_binance_data:
        performance = binance_performance
        print(f"   ✅ 데이터 소스: {data_source}")
    elif db_performance and db_performance['total_trades'] > 0:
        performance = db_performance
        data_source = "DB 일반 데이터"
        print(f"   ✅ 데이터 소스: {data_source}")
    else:
        print("\n   ⚠️ 사용 가능한 거래 데이터 없음")
        return
    
    current_balance = get_available_balance()
    
    print(f"\n{'='*70}")
    print(f"📊 AI 성과 리뷰 (최근 7일) - {data_source}")
    print(f"{'='*70}")
    
    # 현재 오픈 포지션도 확인
    try:
        open_trades = get_all_open_trades()
        open_count = len(open_trades)
        
        if open_count > 0:
            print(f"   ℹ️ 현재 오픈 포지션: {open_count}개 (아래 통계에서 제외)")
    except:
        open_count = 0
    
    print(f"   총 거래: {performance['total_trades']}회 (청산 완료)")
    print(f"   승률: {performance['win_rate']:.1f}%")
    print(f"   총 손익: ${performance['total_pnl']:+,.2f}")
    print(f"   평균 수익: ${performance['avg_win']:+,.2f}")
    print(f"   평균 손실: ${-performance['avg_loss']:+,.2f}")  # 음수로 표시하여 명확히
    print(f"   현재 잔고: ${current_balance:,.2f}")
    
    # 🆕 AI 결정 근거 분석 추가
    try:
        decision_analysis = analyze_ai_decision_patterns(7)
        has_decision_data = decision_analysis['total_trades'] > 0
    except Exception as e:
        print(f"   ⚠️ AI 결정 근거 분석 실패: {e}")
        has_decision_data = False
        decision_analysis = None
    
    # 🆕 바이낸스 데이터가 있는 경우 상세 내역 표시
    if use_binance_data and binance_performance.get('income_data'):
        print(f"\n{'─'*70}")
        print(f"   📋 최근 거래 내역 (바이낸스)")
        print(f"{'─'*70}")
        
        recent_trades = sorted(binance_performance['income_data'], 
                             key=lambda x: x['timestamp'], reverse=True)[:5]
        
        for i, trade in enumerate(recent_trades, 1):
            symbol = trade['symbol'].replace('USDT', '')
            income = trade['income']
            timestamp = trade['timestamp'].strftime('%m-%d %H:%M')
            status_icon = "✅" if income > 0 else "❌"
            
            print(f"   {i}. {status_icon} {symbol}: ${income:+,.2f} ({timestamp})")
    
    # 🆕 AI 결정 근거 분석 섹션
    if has_decision_data and decision_analysis:
        print(f"\n{'─'*70}")
        print(f"   🧠 AI 결정 근거 분석")
        print(f"{'─'*70}")
        
        # 신뢰도별 성과 분석
        conf_analysis = decision_analysis['confidence_analysis']
        
        for conf_level, trades in conf_analysis.items():
            if trades:
                level_name = {'high': '높음(80+)', 'medium': '중간(60-79)', 'low': '낮음(60미만)'}[conf_level]
                win_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(win_trades) / len(trades) * 100
                avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
                
                print(f"   📊 신뢰도 {level_name}: {len(trades)}회 | 승률: {win_rate:.1f}% | 평균 PnL: ${avg_pnl:+,.2f}")
        
        # 성공 패턴 분석
        if decision_analysis['success_patterns']:
            print(f"\n   ✅ 성공 패턴 분석:")
            success_patterns = decision_analysis['success_patterns']
            
            # 신뢰도 평균
            avg_confidence = sum(p['confidence'] for p in success_patterns) / len(success_patterns)
            print(f"   ├─ 성공 거래 평균 신뢰도: {avg_confidence:.1f}")
            
            # 상위 성공 패턴 표시
            top_success = sorted(success_patterns, key=lambda x: x['pnl'], reverse=True)[:2]
            for i, pattern in enumerate(top_success, 1):
                reasoning = pattern['reasoning'][:40] + "..." if len(pattern['reasoning']) > 40 else pattern['reasoning']
                print(f"   ├─ 성공 {i}: {pattern['coin']} (신뢰도: {pattern['confidence']}) - {reasoning}")
        
        # 실패 패턴 분석
        if decision_analysis['failure_patterns']:
            print(f"\n   ❌ 실패 패턴 분석:")
            failure_patterns = decision_analysis['failure_patterns']
            
            # 신뢰도 평균
            avg_confidence = sum(p['confidence'] for p in failure_patterns) / len(failure_patterns)
            print(f"   ├─ 실패 거래 평균 신뢰도: {avg_confidence:.1f}")
            
            # 주요 실패 패턴 표시
            worst_failures = sorted(failure_patterns, key=lambda x: x['pnl'])[:2]
            for i, pattern in enumerate(worst_failures, 1):
                reasoning = pattern['reasoning'][:40] + "..." if len(pattern['reasoning']) > 40 else pattern['reasoning']
                print(f"   ├─ 실패 {i}: {pattern['coin']} (신뢰도: {pattern['confidence']}) - {reasoning}")
        
        # 자주 사용되는 추론 패턴
        if decision_analysis['reasoning_frequency']:
            print(f"\n   🔍 자주 사용된 추론:")
            sorted_reasoning = sorted(decision_analysis['reasoning_frequency'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
            for reasoning, count in sorted_reasoning:
                print(f"   ├─ \"{reasoning}\" ({count}회)")
    
    # 🔥 Risk-Adjusted Metrics
    print(f"\n{'─'*70}")
    print(f"   📈 Risk-Adjusted Performance")
    print(f"{'─'*70}")
    
    if performance['total_trades'] < 3:
        print(f"   📊 충분한 데이터 수집 중... (현재 {performance['total_trades']}회)")
        print(f"   📈 3회 이상 거래 후 Risk/Reward 분석을 제공합니다")
    elif performance['avg_loss'] != 0:
        avg_risk_reward = abs(performance['avg_win'] / performance['avg_loss'])
        print(f"   평균 Risk/Reward: 1:{avg_risk_reward:.2f}")
        
        if avg_risk_reward < 1.5:
            print(f"   ⚠️ Risk/Reward 개선 필요 (목표: 1:2 이상)")
            print(f"   → 손절은 빠르게, 익절은 충분히 기다리세요")
        elif avg_risk_reward >= 2.0:
            print(f"   ✅ Risk/Reward 우수! 현재 전략 유지")
        else:
            print(f"   📊 Risk/Reward 양호 (개선 여지 있음)")
    else:
        print(f"   📊 Risk/Reward 분석 불가 (손실 거래 없음 또는 데이터 부족)")
    
    if performance['total_pnl'] != 0 and performance['total_trades'] > 5:
        # Sharpe Ratio 근사값 (간단 버전)
        avg_return = performance['total_pnl'] / performance['total_trades']
        volatility = abs(performance['avg_win'] - performance['avg_loss'])
        if volatility > 0:
            simple_sharpe = avg_return / volatility
            print(f"   Sharpe Ratio (근사): {simple_sharpe:.2f}")
            
            if simple_sharpe > 1.0:
                print(f"   ✅ 변동성 대비 수익 우수!")
            elif simple_sharpe < 0.5:
                print(f"   ⚠️ 변동성이 너무 큼 (레버리지 ↓ or 진입 신중)")
            else:
                print(f"   📊 안정적인 성과")
    elif performance['total_trades'] <= 5:
        print(f"   📊 Sharpe Ratio: 5회 이상 거래 후 분석 가능")
    
    # 💡 AI 피드백 (결정 근거 기반)
    print(f"\n{'─'*70}")
    print(f"   💡 AI 피드백 (결정 근거 기반)")
    print(f"{'─'*70}")
    
    # 거래 횟수가 적으면 통계적 의미 없음
    if performance['total_trades'] < 3:
        print(f"   📊 샘플 부족: 거래 {performance['total_trades']}회로는 성과 평가가 어렵습니다")
        print(f"   📈 더 많은 거래 후 피드백을 제공하겠습니다")
    else:
        # 기본 성과 피드백
        if performance['win_rate'] < 40:
            print(f"   ⚠️ 승률 낮음 ({performance['win_rate']:.1f}%): Confidence 85+ 거래만 진입하세요")
        elif performance['win_rate'] > 70:
            print(f"   ✅ 승률 우수 ({performance['win_rate']:.1f}%): 현재 선별 기준 유지")
        elif performance['win_rate'] < 50:
            print(f"   📊 승률 보통 ({performance['win_rate']:.1f}%): 신중한 진입 필요")
        
        # 🆕 신뢰도 기반 피드백
        if has_decision_data and decision_analysis:
            high_conf_trades = decision_analysis['confidence_analysis']['high']
            low_conf_trades = decision_analysis['confidence_analysis']['low']
            
            if high_conf_trades:
                high_win_rate = len([t for t in high_conf_trades if t['pnl'] > 0]) / len(high_conf_trades) * 100
                print(f"   📊 높은 신뢰도(80+) 거래 승률: {high_win_rate:.1f}%")
                
                if high_win_rate > 60:
                    print(f"   ✅ 높은 신뢰도 거래가 효과적! 신뢰도 80+ 위주로 진입하세요")
                else:
                    print(f"   ⚠️ 높은 신뢰도에도 성과 부진 → 신호 알고리즘 재검토 필요")
            
            if low_conf_trades:
                low_win_rate = len([t for t in low_conf_trades if t['pnl'] > 0]) / len(low_conf_trades) * 100
                if low_win_rate < 30:
                    print(f"   ⚠️ 낮은 신뢰도 거래는 피하세요 (승률: {low_win_rate:.1f}%)")
            
            # 🆕 실패 패턴 기반 피드백
            if decision_analysis['failure_patterns']:
                failure_reasons = [p['reasoning'] for p in decision_analysis['failure_patterns']]
                common_failures = {}
                for reason in failure_reasons:
                    if 'RSI' in reason:
                        common_failures['RSI'] = common_failures.get('RSI', 0) + 1
                    if 'MACD' in reason:
                        common_failures['MACD'] = common_failures.get('MACD', 0) + 1
                    if '볼린저' in reason or 'Bollinger' in reason:
                        common_failures['볼린저밴드'] = common_failures.get('볼린저밴드', 0) + 1
                
                if common_failures:
                    worst_indicator = max(common_failures, key=common_failures.get)
                    worst_count = common_failures[worst_indicator]
                    if worst_count >= 2:
                        print(f"   ⚠️ '{worst_indicator}' 신호 패턴이 자주 실패 ({worst_count}회) → 다른 지표 조합 고려")
        
        if performance['total_pnl'] < -50:
            print(f"   ⚠️ 손실 누적: 리스크 낮추고 더 확실한 기회만 공략")
        elif performance['total_pnl'] > 100:
            print(f"   ✅ 수익 실현 중: 현재 전략 계속 유지")
        elif performance['total_pnl'] < 0:
            print(f"   📊 소폭 손실: 전략 점검 후 계속 진행")
    
    print(f"{'='*70}")
    
    # 🆕 성과 리뷰 데이터를 DB에 저장
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # AI 피드백 텍스트 생성
        feedback_parts = []
        
        if performance['total_trades'] >= 3:
            feedback_parts.append(f"승률: {performance['win_rate']:.1f}%")
            
            if performance['win_rate'] < 40:
                feedback_parts.append("승률 개선 필요")
            elif performance['win_rate'] > 70:
                feedback_parts.append("승률 우수")
            
            if has_decision_data and decision_analysis:
                high_conf = decision_analysis['confidence_analysis'].get('high', [])
                if high_conf:
                    high_win_rate = len([t for t in high_conf if t['pnl'] > 0]) / len(high_conf) * 100
                    feedback_parts.append(f"높은 신뢰도 승률: {high_win_rate:.1f}%")
            
            if performance['avg_loss'] > 0:
                rr_ratio = abs(performance['avg_win'] / performance['avg_loss'])
                feedback_parts.append(f"Risk/Reward: 1:{rr_ratio:.2f}")
        else:
            feedback_parts.append(f"데이터 수집 중 ({performance['total_trades']}회)")
        
        ai_feedback = " | ".join(feedback_parts)
        
        # INSERT
        c.execute('''
            INSERT INTO performance_reviews (
                total_trades, winning_trades, losing_trades, total_pnl,
                win_rate, avg_win, avg_loss, current_balance, ai_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance['total_trades'],
            performance['winning_trades'],
            performance['losing_trades'],
            performance['total_pnl'],
            performance['win_rate'],
            performance['avg_win'],
            performance['avg_loss'],
            current_balance,
            ai_feedback
        ))
        
        conn.commit()
        conn.close()
        
        print(f"\n   💾 성과 리뷰 데이터 DB 저장 완료")
        
    except Exception as e:
        print(f"\n   ⚠️ DB 저장 실패: {e}")

def display_dashboard():
    """대시보드 표시 (바이낸스 실제 데이터 우선)"""
    
    current_balance = get_available_balance()
    open_trades = get_all_open_trades()
    live_positions = get_open_positions()
    
    # 🔧 포지션 불일치 체크 및 자동 동기화
    db_position_count = len(open_trades)
    binance_position_count = len(live_positions)
    
    if db_position_count != binance_position_count:
        print(f"   ⚠️ 포지션 불일치 감지: DB {db_position_count}개, 바이낸스 {binance_position_count}개")
        
        # 바이낸스에 없는 DB 포지션 찾기
        binance_coins = {pos['symbol'].split('/')[0] for pos in live_positions}
        db_coins = {trade['coin_symbol'] for trade in open_trades}
        
        missing_in_binance = db_coins - binance_coins
        if missing_in_binance:
            print(f"   🔍 바이낸스에 없는 DB 포지션: {', '.join(missing_in_binance)}")
            print(f"   🔄 자동 동기화 시작...")
            
            # 🆕 자동 동기화 실행
            synced = sync_db_with_binance()
            if synced > 0:
                # 동기화 후 다시 조회
                open_trades = get_all_open_trades()
                db_position_count = len(open_trades)
                print(f"   ✅ 동기화 완료: DB 포지션 {db_position_count}개로 업데이트")
    
    # 청산된 거래 조회 (바이낸스 실제 데이터 우선)
    try:
        # 바이낸스 실제 수익 데이터 시도
        binance_performance = calculate_binance_performance(30)  # 30일 데이터
        if binance_performance['total_trades'] > 0:
            realized_pnl = binance_performance['total_pnl']
            closed_trades_count = binance_performance['total_trades']
            winning_trades = binance_performance['winning_trades']
            losing_trades = binance_performance['losing_trades']
            win_rate = binance_performance['win_rate']
            avg_win = binance_performance['avg_win']
            avg_loss = -binance_performance['avg_loss']  # 음수로 표시
            data_source = "바이낸스"
        else:
            raise Exception("바이낸스 데이터 없음")
    except:
        # DB 데이터 fallback
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            SELECT pnl FROM trades
            WHERE status = 'CLOSED'
        ''')
        closed_trades = [{'pnl': row[0] if row[0] else 0} for row in c.fetchall()]
        conn.close()
        
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        closed_trades_count = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        losing_trades = closed_trades_count - winning_trades
        win_rate = (winning_trades / closed_trades_count * 100) if closed_trades_count > 0 else 0
        avg_win = sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) < 0) / losing_trades if losing_trades > 0 else 0
        data_source = "DB"
    
    unrealized_pnl = sum(pos['unrealizedPnl'] for pos in live_positions)
    total_pnl = realized_pnl + unrealized_pnl
    
    print(f"\n{'='*70}")
    print(f"   💼 계정 현황 ({data_source})")
    print(f"{'='*70}")
    print(f"   Available Balance: ${current_balance:,.2f}")
    print(f"   실현 손익: ${realized_pnl:+,.2f}")
    print(f"   미실현 손익: ${unrealized_pnl:+,.2f}")
    print(f"   총 손익: ${total_pnl:+,.2f}")
    print(f"   오픈 포지션: {binance_position_count}개 (바이낸스 실제)")
    if db_position_count != binance_position_count:
        print(f"   ⚠️ DB 기록: {db_position_count}개 (동기화 필요)")
    
    if live_positions:
        print(f"\n{'─'*70}")
        print(f"   📈 포지션 상세")
        print(f"{'─'*70}")
        for pos in live_positions:
            coin = pos['symbol'].split('/')[0]
            direction = "🟢 LONG" if pos['side'] == 'long' else "🔴 SHORT"
            print(f"   {coin:6s} {direction} | 진입: ${pos['entryPrice']:8,.2f} | 현재: ${pos['markPrice']:8,.2f} | PnL: ${pos['unrealizedPnl']:7,.2f} ({pos['percentage']:+6.2f}%)")
    
    if closed_trades_count > 0:
        print(f"\n{'─'*70}")
        print(f"   📊 거래 통계")
        print(f"{'─'*70}")
        print(f"   청산된 거래: {closed_trades_count}회")
        print(f"   ├─ 승리: {winning_trades}회 (평균: ${avg_win:,.2f})")
        print(f"   └─ 손실: {losing_trades}회 (평균: ${avg_loss:,.2f})")
        print(f"   승률: {win_rate:.1f}%")
        print(f"   실현 손익: ${realized_pnl:,.2f}")
    
    # 🆕 수동거래 현황 표시
    manual_trades = list_manual_trades()
    open_manual_trades = [t for t in manual_trades if t['status'] == 'OPEN']
    
    if open_manual_trades:
        print(f"\n{'─'*70}")
        print(f"   🛡️ 수동거래 현황 (AI 청산 차단됨)")
        print(f"{'─'*70}")
        for trade in open_manual_trades:
            direction = "🟢 LONG" if trade['action'] == 'LONG' else "🔴 SHORT"
            print(f"   ID {trade['id']:2d} {trade['coin_symbol']:6s} {direction} | 진입: ${trade['entry_price']:8,.4f} | 투자: ${trade['investment_amount']:8,.2f}")
    
    print(f"{'='*70}\n")

# ===== 메인 루프 =====

def main():
    provider = AI_MODEL_CONFIG["provider"]
    model_name = AI_MODEL_CONFIG["models"][provider]
    
    print(f"\n{'='*80}")
    print(f"  🔴 AI 실거래 트레이딩 봇 v2.0 - AI 결정 근거 분석")
    print(f"{'='*80}")
    print(f"  🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS")
    print(f"  ⚠️  WARNING: 실제 자금으로 거래합니다!")
    print(f"  ✅ AI: {provider.upper()} ({model_name})")
    print(f"  ✅ 마진 모드: {LIVE_TRADING_CONFIG['MARGIN_MODE'].upper()}")
    print(f"  ✅ 최대 포지션: {LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개")
    print(f"  ✅ 포지션당 투자: 동적 균등 분할 (남은 포지션 수로 나눔)")
    if LIVE_TRADING_CONFIG['VOLATILITY_BASED_SIZING']:
        print(f"  ✅ 변동성 기반 사이징: ON")
    print(f"  ✅ Risk/Reward 최소: 1:2")
    print(f"  ✅ Confidence 최소: 70 (투자비율 50-100% 연동)")
    print(f"  🧠 AI 결정 근거 분석: ON")
    print(f"  📊 DB-바이낸스 PnL 매칭: ON")
    print(f"{'='*80}\n")
    
    # 🔧 현재 포지션 수 확인
    print("📊 현재 포지션 상태 확인 중...")
    live_positions = get_open_positions()
    open_positions_count = len(live_positions)
    max_positions = LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']
    
    print(f"   현재 오픈 포지션: {open_positions_count}/{max_positions}개")
    
    # 잔고 확인
    available_balance = get_available_balance()
    
    # 🔧 포지션이 꽉 차있으면 잔고 체크 스킵
    if open_positions_count >= max_positions:
        print(f"\n⚠️  포지션이 이미 {open_positions_count}개로 꽉 차있습니다.")
        print(f"   가용 잔고(${available_balance:,.2f})가 적어도 기존 포지션 관리는 가능합니다.")
        print(f"   신규 진입은 불가하며, 기존 포지션 관리 모드로 실행합니다.\n")
    elif available_balance < LIVE_TRADING_CONFIG['MIN_CAPITAL_THRESHOLD']:
        print(f"\n❌ 가용 잔고(${available_balance:,.2f})가 최소 요구량(${LIVE_TRADING_CONFIG['MIN_CAPITAL_THRESHOLD']:,.2f})보다 작습니다.")
        print(f"   현재 포지션: {open_positions_count}개")
        print("   신규 진입을 위해서는 충분한 잔고를 확보하거나, 기존 포지션을 정리해주세요.")
        return
    else:
        print(f"✅ 가용 잔고: ${available_balance:,.2f} (신규 진입 가능)\n")
    
    # DB 설정 (기존 데이터 유지)
    if os.path.exists(DB_FILE):
        print(f"✅ 기존 DB 파일 사용: {DB_FILE} (거래 데이터 누적)")
        setup_database()  # 스키마만 확인/업데이트
    else:
        print(f"✅ 새 DB 생성: {DB_FILE}")
        setup_database()
    
    print(f"\n{'='*80}")
    print(f"🚀 실거래 봇 자동 시작")
    print(f"   Ctrl+C를 눌러 안전하게 종료할 수 있습니다.")
    print(f"{'='*80}\n")
    
    # 🆕 바이낸스 트레일링 스탑 체커 시작 (별도 스레드)
    if LIVE_TRADING_CONFIG.get("BINANCE_TRAILING_STOP_ENABLED", False):
        print(f"🎯 바이낸스 트레일링 스탑 활성화:")
        print(f"   ✅ 활성화 기준: +{LIVE_TRADING_CONFIG['TRAILING_STOP_ACTIVATION']}% 이익")
        print(f"   ✅ 콜백 비율: {LIVE_TRADING_CONFIG['TRAILING_STOP_CALLBACK_RATE']}% (최고가 대비)")
        print(f"   ✅ 체크 주기: {LIVE_TRADING_CONFIG['TRAILING_STOP_CHECK_INTERVAL']}초 (5분)\n")
        
        trailing_stop_thread = threading.Thread(
            target=trailing_stop_checker_loop,
            daemon=True,
            name="BinanceTrailingStopChecker"
        )
        trailing_stop_thread.start()
        time.sleep(1)
    
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
            
            # 🔧 포지션 관리 비활성화 (TP/SL 주문에만 의존)
            position_check_interval = LIVE_TRADING_CONFIG.get("POSITION_CHECK_INTERVAL", 0)
            if position_check_interval > 0 and current_time - last_position_check_time > position_check_interval:
                manage_live_positions()
                last_position_check_time = current_time
            elif position_check_interval == 0:
                # AI 중간평가 완전 비활성화
                pass
            
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
║        🔴 AI 실거래 트레이딩 봇 v2.0                         ║
║        🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS          ║
║        ⚠️  실제 자금으로 거래합니다!                         ║
║        ✅ AI: {provider:20s} ({model_name:20s})    ║
║        ✅ Isolated Margin 모드                                ║
║        ✅ Risk/Reward ≥ 1:2 전략                             ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  주의사항:")
    print("   1. 실제 바이낸스 계정의 자금을 사용합니다")
    print("   2. Isolated Margin 모드로 거래합니다")
    print("   3. 손실 가능성이 있으니 충분히 이해한 후 사용하세요")
    print("   4. 소액으로 테스트한 후 본격적으로 사용하세요")
    print(f"   5. 수수료: Maker 0.02%, Taker 0.05%\n")
    
    # 자동 시작 (확인 없음)
    print("🚀 백그라운드 모드 - 자동 시작...")
    main()
