"""
AI Live Trading Bot v1.0 (실거래 버전)
----------------------------------------------------------------------
⚠️ 실제 바이낸스 선물 거래 - 실제 자금 사용
- 실시간 바이낸스 선물 데이터 사용
- 실제 잔고로 거래 (Available Balance 기준)
- Isolated Margin 모드
- 실제 주문 체결 및 청산
- AI 보수적 리스크 관리
----------------------------------------------------------------------
"""

import ccxt
import os
import time
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas_ta as ta
import google.generativeai as genai
import random
import json
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import sys

load_dotenv()

# ===== 로깅 설정 =====
LOG_FILE = "trading_bot.log"

class Logger:
    """파일과 콘솔에 동시 로깅"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 로거 초기화
sys.stdout = Logger(LOG_FILE)
sys.stderr = Logger(LOG_FILE)

# ===== AI 모델 선택 =====
AI_MODEL_CONFIG = {
    "provider": "deepseek",  # 🔧 여기서 변경: "gemini" (무료) 또는 "deepseek" (유료, 정확)
    "models": {
        "gemini": "gemini-2.0-flash-exp",  # 무료, 빠름
        "openai": "gpt-4o",
        "qwen": "qwen-max",
        "claude": "claude-3-5-sonnet-20241022",
        "deepseek": "deepseek-chat"  # DeepSeek V3 (최신, 추천)
    },
    "rate_limit": {
        "gemini": {
            "requests_per_minute": 15,
            "delay_between_requests": 4.5
        },
        "openai": {
            "requests_per_minute": 60,
            "delay_between_requests": 1.0
        },
        "qwen": {
            "requests_per_minute": 60,
            "delay_between_requests": 1.0
        },
        "claude": {
            "requests_per_minute": 50,
            "delay_between_requests": 1.2
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
        elif provider == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            ai_client = openai
            print(f"✅ OpenAI ({AI_MODEL_CONFIG['models']['openai']}) 설정 완료")
            return True
        elif provider == "qwen":
            import dashscope
            dashscope.api_key = os.getenv("QWEN_API_KEY")
            ai_client = dashscope
            print(f"✅ Qwen AI ({AI_MODEL_CONFIG['models']['qwen']}) 설정 완료")
            return True
        elif provider == "claude":
            import anthropic
            ai_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            print(f"✅ Claude AI ({AI_MODEL_CONFIG['models']['claude']}) 설정 완료")
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
    print("\n사용 가능한 AI 제공자: gemini, openai, qwen, claude, deepseek")
    exit(1)

# ===== 실거래 설정 =====
LIVE_TRADING_CONFIG = {
    "MAX_CONCURRENT_POSITIONS": 5,  # 최대 동시 포지션 (실제 진입 제한)
    "MIN_CAPITAL_THRESHOLD": 100.0,  # 최소 잔고 (USDT)
    "AI_ANALYSIS_INTERVAL": 60,  # 신규 진입 분석 (1분마다)
    "PERFORMANCE_REVIEW_INTERVAL": 600,  # AI 성과 리뷰 (10분)
    "POSITION_CHECK_INTERVAL": 120,  # 포지션 관리 체크 간격 (2분)
    
    # 🔧 자금 관리 설정
    "MAX_POSITION_SIZE_PCT": 20,  # 가용 자금의 최대 20% (5개 포지션이므로)
    "MIN_POSITION_SIZE_PCT": 3,   # 최소 3% (너무 작은 포지션 방지)
    "VOLATILITY_BASED_SIZING": True,  # 변동성 기반 포지션 크기 조절
    "HIGH_VOLATILITY_THRESHOLD": 5.0,  # 5% 이상이면 고변동성
    "LOW_VOLATILITY_MULTIPLIER": 1.5,  # 저변동성 = 1.5배 투자
    "HIGH_VOLATILITY_MULTIPLIER": 0.6,  # 고변동성 = 0.6배 투자
    
    # 🔧 거래 수수료 (바이낸스 선물 일반회원)
    "MAKER_FEE": 0.02,  # 0.02%
    "TAKER_FEE": 0.05,  # 0.05%
    
    # 🔧 레버리지 설정
    "MAX_LEVERAGE": 10,
    "CONSERVATIVE_LEVERAGE": 7,
    
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

def cancel_all_orders(symbol: str):
    """심볼의 모든 오픈 주문 취소"""
    try:
        result = exchange.cancel_all_orders(symbol)
        print(f"   ✅ 모든 주문 취소: {symbol}")
        return result
    except Exception as e:
        print(f"   ⚠️ 주문 취소 오류: {e}")
        return None

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
                
                open_positions.append({
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'contracts': contracts,
                    'entryPrice': float(entry_price) if entry_price is not None else 0,
                    'markPrice': float(mark_price) if mark_price is not None else 0,
                    'unrealizedPnl': float(unrealized_pnl) if unrealized_pnl is not None else 0,
                    'percentage': float(percentage) if percentage is not None else 0,
                    'leverage': float(leverage) if leverage is not None else 1,
                    'notional': float(notional) if notional is not None else 0,
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

def cleanup_orphaned_orders():
    """🆕 포지션 없는 TP/SL 주문 정리 - 심볼별 조회"""
    try:
        print(f"\n{'='*70}")
        print(f"🧹 고아 TP/SL 주문 정리 시작")
        print(f"{'='*70}")
        
        # 1. 현재 오픈 포지션 조회
        open_positions = get_open_positions()
        position_symbols = {pos['symbol'] for pos in open_positions}
        
        print(f"   오픈 포지션: {len(position_symbols)}개")
        
        # 2. DB에서 거래한 심볼 조회
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT DISTINCT coin_symbol FROM trades")
        traded_coins = [row[0] for row in c.fetchall()]
        conn.close()
        
        # 3. 거래한 심볼만 체크
        all_symbols = [f"{coin}/USDT:USDT" for coin in traded_coins]
        
        orphaned_count = 0
        for symbol in all_symbols:
            # 포지션 있으면 스킵
            if symbol in position_symbols:
                continue
            
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
                coin = symbol.replace('/USDT:USDT', '')
                print(f"\n   🗑️ {coin}: TP/SL 주문 {len(tp_sl_orders)}개 정리 중...")
                
                try:
                    cancelled = cancel_all_orders(symbol)
                    orphaned_count += cancelled
                    print(f"     ✅ {cancelled}개 취소 완료")
                except Exception as e:
                    print(f"     ❌ 취소 실패: {e}")
        
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
            binance_close_price REAL
        )
    ''')
    
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
            holding_time_estimate, ai_reasoning, market_conditions, binance_order_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        trade_data.get('binance_order_id', '')
    ))
    
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id

def update_trade_close(trade_id: int, close_data: Dict):
    """거래 청산 정보 업데이트 (바이낸스 실제 결과 사용)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
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
               binance_order_id, trading_style
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
            'trading_style': row[12] if len(row) > 12 else 'DAY_TRADING'
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
    """최근 성과 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
            COALESCE(AVG(CASE WHEN pnl < 0 THEN pnl END), 0) as avg_loss
        FROM trades
        WHERE status = 'CLOSED'
        AND close_timestamp >= datetime('now', '-' || ? || ' days')
    ''', (days,))
    
    row = c.fetchone()
    conn.close()
    
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
    
    return {
        'total_trades': row[0],
        'winning_trades': row[1],
        'losing_trades': row[2],
        'win_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
        'total_pnl': row[3],
        'avg_win': row[4],
        'avg_loss': row[5]
    }

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
    """최근 성과 조회"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
            COALESCE(AVG(CASE WHEN pnl < 0 THEN pnl END), 0) as avg_loss
        FROM trades
        WHERE status = 'CLOSED'
        AND close_timestamp >= datetime('now', '-' || ? || ' days')
    ''', (days,))
    
    row = c.fetchone()
    conn.close()
    
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
    
    return {
        'total_trades': row[0],
        'winning_trades': row[1],
        'losing_trades': row[2],
        'win_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
        'total_pnl': row[3],
        'avg_win': row[4],
        'avg_loss': row[5]
    }

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
    """여러 타임프레임의 시장 데이터 수집"""
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']  # 원본 복원: 7개 타임프레임
    market_data = {}
    
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
            elif provider == "openai":
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
            elif provider == "qwen":
                from dashscope import Generation
                response = Generation.call(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "당신은 암호화폐 트레이딩 전문가 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    result_format='message',
                    temperature=temperature
                )
                if response.status_code == 200:
                    last_api_call_time = time.time()
                    return response.output.choices[0].message.content
                else:
                    raise Exception(f"Qwen API 오류: {response.message}")
            elif provider == "claude":
                response = ai_client.messages.create(
                    model=model_name,
                    max_tokens=2000,
                    temperature=temperature,
                    system="당신은 암호화폐 트레이딩 전문가 AI입니다.",
                    messages=[{"role": "user", "content": prompt}]
                )
                last_api_call_time = time.time()
                return response.content[0].text
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

【분석 가이드 - 실거래 모드】
1. 타임프레임 크로스체크: 여러 타임프레임에서 같은 신호 → 신뢰도 ↑
2. 다중 지표 확인: RSI+MACD+Stochastic 모두 일치 → 강한 신호
3. 추세 강도 확인: ADX > 25 → 강한 추세, < 20 → 약한 추세 (진입 회피)
4. 거래량 확인: MFI, OBV 상승 → 매수세 강함
5. 변동성 고려: BB밴드폭 좁으면 → 큰 움직임 임박, 넓으면 조정 가능성
6. 지지/저항 근처: 반등 or 돌파 가능성 평가
7. 레버리지: BTC/ETH 최대 10배, 알트 5~7배 (실거래는 보수적으로)
8. 과거 실패 패턴 반드시 회피

【트레이딩 스타일 선택 - 매우 중요】
타임프레임 분석을 바탕으로 아래 중 하나를 선택하세요:

1. SCALPING (스캘핑) - 초단타
   - 조건: 1m, 5m, 15m 타임프레임에서 강한 신호
   - 목표 수익: 1~3%
   - 손절: 0.5~1.5%
   - 보유 시간: 수분~1시간
   - 레버리지: 높음 (7~10x)
   - 예: 단기 RSI 과매도 + 볼린저밴드 하단 터치 + 거래량 급증

2. DAY_TRADING (데이트레이딩) - 단기
   - 조건: 15m, 1h, 4h 타임프레임에서 일치
   - 목표 수익: 3~8%
   - 손절: 1.5~3%
   - 보유 시간: 수시간~1일
   - 레버리지: 중간 (5~7x)
   - 예: 중단기 추세 형성 + ADX 상승 + 골든크로스

3. SWING_TRADING (스윙) - 중장기
   - 조건: 4h, 1d, 1w 타임프레임에서 강한 추세
   - 목표 수익: 8~20%
   - 손절: 3~5%
   - 보유 시간: 수일~수주
   - 레버리지: 낮음 (3~5x)
   - 예: 장기 추세 전환 + 주봉 패턴 + 거시적 모멘텀

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
   - 70 이상: 진입 고려
   - 70 미만: 거래 금지

5. **변동성 리스크 관리**
   - 고변동성(ATR/가격 > 5%) → 레버리지 ↓, 투자금액 ↓
   - 저변동성 → 안정적 수익 기회

6. **추세 강도 확인 필수**
   - ADX < 20 → 약한 추세 → 진입 회피
   - ADX > 25 → 강한 추세 → 진입 고려 (LONG/SHORT 모두)

7. **과거 실패 패턴 반드시 회피**
   - 유사한 패턴에서 손실 발생 → 동일 실수 금지

8. **레버리지 보수적 운용**
   - 높은 확신(90+) + 강한 추세 → 높은 레버리지 가능
   - 중간 확신(70%) → 중간 레버리지 (5~7x)
   - 낮은 확신 → 거래 금지

**핵심: 불확실한 거래는 절대 하지 않는다. 확실한 Risk/Reward만 공략한다. LONG과 SHORT는 동등한 기회다.**

**반드시 다음 JSON 형식으로만 답변:**

{{
  "trade": true/false,
  "direction": "LONG" or "SHORT",
  "risk_reward_ratio": "예: 1:3 (리스크 2%, 리워드 6%)",
  "expected_risk_pct": 0.5~5.0 (예상 손실 %),
  "expected_reward_pct": 1.0~20.0 (예상 수익 %),
  "risk_adjusted_score": 0~100 (Risk/Reward 고려한 점수, 80+ 진입),
  "trading_style": "SCALPING" or "DAY_TRADING" or "SWING_TRADING",
  "leverage": 3~{LIVE_TRADING_CONFIG['MAX_LEVERAGE']},
  "investment_percentage": 5~20,
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
- trading_style에 따라 sl_percentage와 tp_percentage를 적절히 설정
- 스캘핑: SL 0.5~1.5%, TP 1~3%
- 데이트레이딩: SL 1.5~3%, TP 3~8%  
- 스윙: SL 3~5%, TP 8~20%
- **expected_reward_pct / expected_risk_pct ≥ 2.0 되도록 설정**

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
- 손실 확대 방지가 최우선 (손실 -10% 초과 시 즉시 청산 고려)
- 수익 실현보다 손실 방지가 더 중요
- 추세 반전 신호가 보이면 수익 중이라도 청산
- 불확실성 증가 시 포지션 축소 or 청산

【포지션】
코인: {trade['coin_symbol']} {action.upper()}
진입: ${entry_price:,.4f} → 현재: ${current_price:,.4f}
가격 변화: {price_change_pct:+.2f}%
레버리지: {leverage}x
실제 수익률: {actual_pnl_pct:+.2f}% (투자금 ${investment:,.2f} 대비)
실제 손익: ${actual_pnl_amount:+,.2f}
보유 시간: {(datetime.now() - datetime.fromisoformat(trade['timestamp'])).total_seconds() / 3600:.1f}시간

{market_summary}

【판단 기준 - Risk-Adjusted Returns 중심】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **손실 방지가 수익 추구보다 우선**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **손실 관리 (Stop-Loss)**
   - 손실 -5% 초과 → 반등 근거 명확해야 유지
   - 손실 -10% 초과 → 즉시 청산 (손실 확대 방지)
   - 손실 확대 중 + 반등 신호 없음 → 청산

2. **추세 반전 체크**
   - 여러 타임프레임에서 추세 반전 신호 확인
   - MACD 크로스 + RSI 역전 + 지지선 붕괴 → 청산
   
3. **변동성 급증 대응**
   - 변동성 큰 코인은 과민 반응 금지
   - 하지만 ATR 급증 + 추세 반전 → 조기 청산

4. **수익 실현 (Take-Profit)**
   - 큰 수익(+15% 이상) → 일부 실현 고려
   - 목표 도달 + 추세 약화 신호 → 전량 청산
   
5. **홀딩 vs 청산 결정**
   - 손실 중: 반등 근거 명확 → 홀드, 불명확 → 청산
   - 수익 중: 추세 지속 → 홀드, 약화 → 청산

**핵심: 불확실하면 청산. 명확한 근거 있을 때만 홀드.**

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

def calculate_position_size(available_balance: float, ai_percentage: float, volatility: float, open_positions: int, trading_style: str = 'DAY_TRADING') -> float:
    """🆕 Kelly Criterion 기반 스마트 포지션 사이징"""
    config = LIVE_TRADING_CONFIG
    
    # 🆕 1. Kelly Criterion 기반 최적 비율 계산
    strategy_perf = get_strategy_performance(trading_style)
    
    if strategy_perf and strategy_perf['total_trades'] >= 5:
        # 충분한 데이터가 있으면 Kelly Criterion 사용
        win_rate = strategy_perf['win_rate']
        avg_win = abs(strategy_perf['avg_win'])
        avg_loss = abs(strategy_perf['avg_loss'])
        
        kelly_pct = calculate_kelly_criterion(win_rate, avg_win, avg_loss) * 100
        
        print(f"   📊 Kelly Criterion: {kelly_pct:.1f}% (승률:{win_rate:.1f}%, R/R:{avg_win/avg_loss if avg_loss > 0 else 0:.2f})")
        
        # AI 제안과 Kelly 중 보수적인 값 선택
        base_percentage = min(ai_percentage, kelly_pct)
        print(f"   💡 AI:{ai_percentage:.1f}% vs Kelly:{kelly_pct:.1f}% → 선택:{base_percentage:.1f}%")
    else:
        # 데이터 부족 시 AI 제안 사용
        base_percentage = ai_percentage
        print(f"   💡 데이터 부족 → AI 제안 {base_percentage:.1f}% 사용")
    
    base_investment = available_balance * (base_percentage / 100)
    
    # 🆕 2. 연속 손실 페널티
    consecutive_losses = get_consecutive_losses()
    if consecutive_losses >= 3:
        penalty = 0.5  # 50%로 축소
        base_investment *= penalty
        print(f"   ⚠️ 연속 손실 {consecutive_losses}회 → {penalty}x 투자")
        
        # 리스크 조정 로그
        log_risk_adjustment(
            adjustment_type='POSITION_SIZE',
            previous_value=base_percentage,
            new_value=base_percentage * penalty,
            reason=f"연속 손실 {consecutive_losses}회",
            consecutive_losses=consecutive_losses,
            current_drawdown=get_current_drawdown()
        )
    elif consecutive_losses >= 2:
        penalty = 0.75  # 75%로 축소
        base_investment *= penalty
        print(f"   ⚠️ 연속 손실 {consecutive_losses}회 → {penalty}x 투자")
    
    # 🆕 3. 드로다운 페널티
    current_drawdown = get_current_drawdown()
    if current_drawdown > 200:
        penalty = 0.5
        base_investment *= penalty
        print(f"   ⚠️ 드로다운 ${current_drawdown:.2f} → {penalty}x 투자")
    elif current_drawdown > 100:
        penalty = 0.75
        base_investment *= penalty
        print(f"   ⚠️ 드로다운 ${current_drawdown:.2f} → {penalty}x 투자")
    
    # 4. 최대/최소 제한 적용
    max_investment = available_balance * (config['MAX_POSITION_SIZE_PCT'] / 100)
    min_investment = available_balance * (config['MIN_POSITION_SIZE_PCT'] / 100)
    
    investment = min(base_investment, max_investment)
    investment = max(investment, min_investment)
    
    # 5. 변동성 기반 조정
    if config['VOLATILITY_BASED_SIZING']:
        if volatility > config['HIGH_VOLATILITY_THRESHOLD']:
            # 고변동성 = 리스크 높음 = 투자 줄임
            multiplier = config['HIGH_VOLATILITY_MULTIPLIER']
            investment *= multiplier
            print(f"   📉 고변동성 ({volatility:.1f}%) → {multiplier}x 투자")
        elif volatility < config['HIGH_VOLATILITY_THRESHOLD'] / 2:
            # 저변동성 = 리스크 낮음 = 투자 늘림
            multiplier = config['LOW_VOLATILITY_MULTIPLIER']
            investment *= multiplier
            print(f"   📈 저변동성 ({volatility:.1f}%) → {multiplier}x 투자")
    
    # 6. 포지션 수에 따른 조정 (포지션 많을수록 보수적으로)
    max_positions = config['MAX_CONCURRENT_POSITIONS']
    if open_positions >= max_positions * 0.8:  # 80% 이상 차면
        investment *= 0.7  # 30% 감소
        print(f"   ⚠️ 포지션 많음 ({open_positions}/{max_positions}) → 0.7x 투자")
    
    # 7. 최종 제한 재확인
    investment = min(investment, max_investment)
    investment = max(investment, min_investment)
    
    # 8. 가용 잔고 초과 방지
    if investment > available_balance * 0.95:  # 95% 이상 사용 방지
        investment = available_balance * 0.95
        print(f"   ⚠️ 가용 잔고 부족 → ${investment:.2f}로 조정")
    
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
            'binance_order_id': order['id']
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
            print(f"\n   ⚠️ {coin}: DB에는 있으나 바이낸스에 포지션 없음")
            
            # 🆕 해당 심볼의 미체결 TP/SL 주문 정리
            try:
                open_orders = get_open_orders(symbol)
                tp_sl_orders = [
                    order for order in open_orders 
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 
                                          'STOP_LOSS_MARKET', 'TAKE_PROFIT']
                ]
                
                if tp_sl_orders:
                    print(f"   🗑️ {coin}: TP/SL 주문 {len(tp_sl_orders)}개 정리 중...")
                    cancelled = cancel_all_orders(symbol)
                    print(f"   ✅ {cancelled}개 주문 취소 완료")
            except Exception as e:
                print(f"   ⚠️ TP/SL 주문 정리 실패: {e}")
            
            # 바이낸스에서 청산된 경우 PnL 조회 후 DB 업데이트
            try:
                pnl_info = get_position_pnl(symbol)
                close_data = {
                    'close_price': 0,
                    'pnl': pnl_info.get('realizedPnl', 0),
                    'pnl_percentage': 0,
                    'binance_pnl': pnl_info.get('realizedPnl', 0),
                    'binance_close_price': 0
                }
                update_trade_close(trade['id'], close_data)
                print(f"   ✅ DB 업데이트 완료 (실현 PnL: ${pnl_info.get('realizedPnl', 0):,.2f})")
            except Exception as e:
                print(f"   ⚠️ PnL 조회 실패: {e}")
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
                                'binance_close_price': live_pos['entryPrice']
                            }
                            update_trade_close(trade['id'], close_data)
                            
                            # 전략 성과 업데이트
                            trading_style = trade.get('trading_style', 'DAY_TRADING')
                            is_win = close_result['pnl'] > 0
                            update_strategy_performance(trading_style, close_result['pnl'], is_win)
                            
                            print(f"   ✅ DB 업데이트 완료 (실현 PnL: ${close_result['pnl']:+,.2f})")
                        continue
                    
                    # DB 업데이트 (바이낸스 실제 PnL 사용)
                    close_data = {
                        'close_price': close_result['close_price'],
                        'pnl': close_result['pnl'],
                        'pnl_percentage': (close_result['pnl'] / trade['investment'] * 100) if trade.get('investment', 0) > 0 else 0,
                        'binance_pnl': close_result['pnl'],
                        'binance_close_price': close_result['close_price']
                    }
                    
                    update_trade_close(trade['id'], close_data)
                    
                except Exception as close_error:
                    print(f"   ❌ 청산 실패: {close_error}")
                    # 청산 실패해도 계속 진행 (다음 포지션 체크)
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

def ai_performance_review():
    """AI 성과 리뷰 - Risk-Adjusted Returns 분석"""
    
    performance = get_recent_performance(7)
    
    if performance['total_trades'] == 0:
        print("\n   ⚠️ 거래 내역 없음")
        return
    
    current_balance = get_available_balance()
    
    print(f"\n{'='*70}")
    print(f"📊 AI 성과 리뷰 (최근 7일) - Risk-Adjusted Analysis")
    print(f"{'='*70}")
    print(f"   총 거래: {performance['total_trades']}회")
    print(f"   승률: {performance['win_rate']:.1f}%")
    print(f"   총 손익: ${performance['total_pnl']:+,.2f}")
    print(f"   평균 수익: ${performance['avg_win']:+,.2f}")
    print(f"   평균 손실: ${performance['avg_loss']:+,.2f}")
    print(f"   현재 잔고: ${current_balance:,.2f}")
    
    # 🔥 Risk-Adjusted Metrics
    print(f"\n{'─'*70}")
    print(f"   📈 Risk-Adjusted Performance")
    print(f"{'─'*70}")
    
    if performance['avg_loss'] != 0:
        avg_risk_reward = abs(performance['avg_win'] / performance['avg_loss'])
        print(f"   평균 Risk/Reward: 1:{avg_risk_reward:.2f}")
        
        if avg_risk_reward < 1.5:
            print(f"   ⚠️ Risk/Reward 개선 필요 (목표: 1:2 이상)")
            print(f"   → 손절은 빠르게, 익절은 충분히 기다리세요")
        elif avg_risk_reward >= 2.0:
            print(f"   ✅ Risk/Reward 우수! 현재 전략 유지")
        else:
            print(f"   📊 Risk/Reward 양호 (개선 여지 있음)")
    
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
    
    # 💡 AI 피드백
    print(f"\n{'─'*70}")
    print(f"   💡 AI 피드백")
    print(f"{'─'*70}")
    
    if performance['win_rate'] < 50:
        print(f"   ⚠️ 승률 낮음: Confidence 80+ 거래만 진입하세요")
    elif performance['win_rate'] > 70:
        print(f"   ✅ 승률 우수: 현재 선별 기준 유지")
    
    if performance['total_pnl'] < 0:
        print(f"   ⚠️ 손실 중: 리스크 낮추고 더 확실한 기회만 공략")
    elif performance['total_pnl'] > 100:
        print(f"   ✅ 수익 실현 중: 현재 전략 계속 유지")
    
    print(f"{'='*70}")

# ===== 대시보드 =====

def display_dashboard():
    """대시보드 표시"""
    
    current_balance = get_available_balance()
    open_trades = get_all_open_trades()
    live_positions = get_open_positions()
    
    # 청산된 거래 조회
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT pnl FROM trades
        WHERE status = 'CLOSED'
    ''')
    closed_trades = [{'pnl': row[0]} for row in c.fetchall()]
    conn.close()
    
    realized_pnl = sum(t['pnl'] for t in closed_trades if t['pnl'])
    unrealized_pnl = sum(pos['unrealizedPnl'] for pos in live_positions)
    total_pnl = realized_pnl + unrealized_pnl
    
    print(f"\n{'='*70}")
    print(f"   💼 계정 현황")
    print(f"{'='*70}")
    print(f"   Available Balance: ${current_balance:,.2f}")
    print(f"   실현 손익: ${realized_pnl:+,.2f}")
    print(f"   미실현 손익: ${unrealized_pnl:+,.2f}")
    print(f"   총 손익: ${total_pnl:+,.2f}")
    print(f"   오픈 포지션: {len(open_trades)}개")
    
    if live_positions:
        print(f"\n{'─'*70}")
        print(f"   📈 포지션 상세")
        print(f"{'─'*70}")
        for pos in live_positions:
            coin = pos['symbol'].split('/')[0]
            direction = "🟢 LONG" if pos['side'] == 'long' else "🔴 SHORT"
            print(f"   {coin:6s} {direction} | 진입: ${pos['entryPrice']:8,.2f} | 현재: ${pos['markPrice']:8,.2f} | PnL: ${pos['unrealizedPnl']:7,.2f} ({pos['percentage']:+6.2f}%)")
    
    if closed_trades:
        winning_trades = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        losing_trades = len(closed_trades) - winning_trades
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        avg_win = sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['pnl'] for t in closed_trades if t.get('pnl', 0) < 0) / losing_trades if losing_trades > 0 else 0
        
        print(f"\n{'─'*70}")
        print(f"   📊 거래 통계")
        print(f"{'─'*70}")
        print(f"   청산된 거래: {len(closed_trades)}회")
        print(f"   ├─ 승리: {winning_trades}회 (평균: ${avg_win:,.2f})")
        print(f"   └─ 손실: {losing_trades}회 (평균: ${avg_loss:,.2f})")
        print(f"   승률: {win_rate:.1f}%")
        print(f"   실현 손익: ${realized_pnl:,.2f}")
    
    print(f"{'='*70}\n")

# ===== 메인 루프 =====

def main():
    provider = AI_MODEL_CONFIG["provider"]
    model_name = AI_MODEL_CONFIG["models"][provider]
    
    print(f"\n{'='*80}")
    print(f"  🔴 AI 실거래 트레이딩 봇 v2.0 - Risk-Adjusted Returns")
    print(f"{'='*80}")
    print(f"  🎯 OBJECTIVE: MAXIMIZE RISK-ADJUSTED RETURNS")
    print(f"  ⚠️  WARNING: 실제 자금으로 거래합니다!")
    print(f"  ✅ AI: {provider.upper()} ({model_name})")
    print(f"  ✅ 마진 모드: {LIVE_TRADING_CONFIG['MARGIN_MODE'].upper()}")
    print(f"  ✅ 최대 포지션: {LIVE_TRADING_CONFIG['MAX_CONCURRENT_POSITIONS']}개")
    print(f"  ✅ 포지션당 최대: 가용자금의 {LIVE_TRADING_CONFIG['MAX_POSITION_SIZE_PCT']}%")
    if LIVE_TRADING_CONFIG['VOLATILITY_BASED_SIZING']:
        print(f"  ✅ 변동성 기반 사이징: ON")
    print(f"  ✅ Risk/Reward 최소: 1:2")
    print(f"  ✅ Confidence 최소: 80")
    print(f"{'='*80}\n")
    
    # 잔고 확인
    available_balance = get_available_balance()
    if available_balance < LIVE_TRADING_CONFIG['MIN_CAPITAL_THRESHOLD']:
        print(f"\n❌ 가용 잔고(${available_balance:,.2f})가 최소 요구량(${LIVE_TRADING_CONFIG['MIN_CAPITAL_THRESHOLD']:,.2f})보다 작습니다.")
        print("   충분한 잔고를 확보한 후 다시 시도하세요.")
        return
    
    print(f"✅ 가용 잔고: ${available_balance:,.2f}\n")
    
    # DB 설정 (자동)
    if os.path.exists(DB_FILE):
        print(f"✅ 기존 DB 파일 사용: {DB_FILE}")
        setup_database()  # 스키마만 확인
    else:
        print(f"✅ 새 DB 생성: {DB_FILE}")
        setup_database()
    
    print(f"\n{'='*80}")
    print(f"🚀 실거래 봇 자동 시작")
    print(f"   Ctrl+C를 눌러 안전하게 종료할 수 있습니다.")
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
            now = datetime.now()
            
            # 대시보드 표시
            if current_time - last_dashboard_time > 120:
                display_dashboard()
                last_dashboard_time = current_time
            
            print(f"\n{'='*80}")
            print(f"⏰ [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"{'='*80}")
            
            # 잔고 조회
            available_balance = get_available_balance()
            
            # 포지션 관리 (2분마다 AI 호출)
            position_check_interval = LIVE_TRADING_CONFIG.get("POSITION_CHECK_INTERVAL", 120)
            if current_time - last_position_check_time > position_check_interval:
                manage_live_positions()
                last_position_check_time = current_time
            
            open_trades = get_all_open_trades()
            open_positions_count = len(open_trades)
            
            # 성과 리뷰
            if current_time - last_review_time > LIVE_TRADING_CONFIG['PERFORMANCE_REVIEW_INTERVAL']:
                ai_performance_review()
                last_review_time = current_time
            
            # 포지션이 꽉 찼는지 체크
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
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
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