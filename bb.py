"""
Dual Bollinger Band Strategy Bot (v1.1 - Confirmation & SL)
----------------------------------------------------------------
전략:
- 멀티코인 스윙 트레이딩 (BTC, ETH 등 8개 코인)
- 분석 타임프레임: 1시간봉 (1h)
- 핵심 전략: 과매도/과매수 확인 후 반전 캔들에서 진입, 자동 손절매 설정

- 지표 설정:
    1. 메인 볼린저밴드 (BB1): 20 periods, 2 standard deviations
    2. 보조 볼린저밴드 (BB2): 4 periods, 4 standard deviations

- 진입 조건 (Entry):
    - 롱(Long):
        1. (Setup) 1시간봉이 BB1 하단과 BB2 하단을 모두 터치/돌파하며 마감
        2. (Confirm) 바로 다음 1시간봉이 양봉으로 마감
        3. Confirm 캔들 마감 시 진입
    - 숏(Short):
        1. (Setup) 1시간봉이 BB1 상단과 BB2 상단을 모두 터치/돌파하며 마감
        2. (Confirm) 바로 다음 1시간봉이 음봉으로 마감
        3. Confirm 캔들 마감 시 진입

- 청산 조건 (Exit):
    - 익절(Take Profit):
        - 롱 포지션: 현재가가 BB1 상단 또는 BB2 상단 중 하나라도 터치/돌파 시 전량 익절
        - 숏 포지션: 현재가가 BB1 하단 또는 BB2 하단 중 하나라도 터치/돌파 시 전량 익절
    - 손절(Stop Loss):
        - 포지션 진입 시 자동으로 'Setup 캔들'의 고점/저점을 기준으로 지정가 손절 주문 설정

- 관리:
    - 스캔 주기: 5분마다 모든 코인을 스캔하여 진입/청산 신호 확인
    - 포지션 관리: 최대 5개의 동시 포지션 유지
----------------------------------------------------------------
"""

# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import time
import pandas as pd
import numpy as np
import sqlite3
from dotenv import load_dotenv
from datetime import datetime
import pandas_ta as ta  # ATR 계산을 위해 추가 (pip install pandas_ta)

# .env 파일 로드
load_dotenv()

# ===== 멀티코인 설정 =====
TRADING_PAIRS = {
    "BTC": {"symbol": "BTC/USDT"},
    "ETH": {"symbol": "ETH/USDT"},
    "SOL": {"symbol": "SOL/USDT"},
    "XRP": {"symbol": "XRP/USDT"},
    "ADA": {"symbol": "ADA/USDT"},
    "AVAX": {"symbol": "AVAX/USDT"},
    "LINK": {"symbol": "LINK/USDT"},
    "DOGE": {"symbol": "DOGE/USDT"},
}

# ===== 동시 포지션 제한 설정 =====
MAX_CONCURRENT_POSITIONS = 5

# ===== 전략 설정 =====
STRATEGY_CONFIG = {
    "TIMEFRAME": '1h',
    "POSITION_SIZE": 0.20,  # 가용 자본의 20%를 각 포지션에 할당
    "LEVERAGE": 5,
    
    # 볼린저밴드 1 설정
    "BB1_WINDOW": 20,
    "BB1_STD_DEV": 2,
    
    # 볼린저밴드 2 설정
    "BB2_WINDOW": 4,
    "BB2_STD_DEV": 4,
    
    # 밴드 '터치'로 간주할 근접 버퍼 (0.1%)
    "TOUCH_BUFFER": 0.001,
    
    # 손절매 설정 버퍼 (Setup 캔들 고점/저점에서 추가할 % 0.2%)
    "SL_BUFFER": 0.002
}

# ===== API 및 DB 설정 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
DB_FILE = "multi_coin_daytrading.db"

# ===== 기술 지표 계산 함수 =====
def calculate_bollinger_bands(prices, window, std_dev):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

# ===== 데이터베이스 함수 (기존과 동일) =====
def setup_database():
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
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        profit_loss REAL,
        exit_reason TEXT,
        exit_timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print("Dual BB Trading 데이터베이스 설정 완료")

def save_trade_to_db(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, coin_symbol, action, entry_price, amount, leverage, status)
    VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (
        datetime.now().isoformat(),
        trade_data['coin_symbol'],
        trade_data['action'],
        trade_data['entry_price'],
        trade_data['amount'],
        trade_data['leverage']
    ))
    conn.commit()
    conn.close()

def close_trade_in_db(coin_symbol, exit_price, pnl, reason):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE trades 
    SET status = 'CLOSED', exit_price = ?, profit_loss = ?, exit_reason = ?, exit_timestamp = ?
    WHERE coin_symbol = ? AND status = 'OPEN'
    ''', (exit_price, pnl, reason, datetime.now().isoformat(), coin_symbol))
    conn.commit()
    conn.close()

# ===== 포지션 조회 함수 (기존과 동일) =====
def get_open_positions_from_binance():
    try:
        positions = exchange.fetch_positions()
        open_positions = []
        for pos in positions:
            if float(pos['info']['positionAmt']) != 0:
                coin_symbol = pos['symbol'].replace('/USDT:USDT', '')
                open_positions.append({
                    "coin_symbol": coin_symbol,
                    "action": 'long' if float(pos['info']['positionAmt']) > 0 else 'short',
                    "entry_price": float(pos['entryPrice']),
                    "amount": float(pos['contracts']),
                    "leverage": int(pos['info']['leverage'])
                })
        return open_positions
    except Exception as e:
        print(f"바이낸스 포지션 조회 오류: {e}")
        return []

# ===== 데이터 수집 및 분석 함수 (ATR 계산 추가) =====
def fetch_and_analyze_coin_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=STRATEGY_CONFIG['TIMEFRAME'], limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 두 개의 볼린저밴드 계산
        df['bb1_upper'], df['bb1_middle'], df['bb1_lower'] = calculate_bollinger_bands(df['close'], STRATEGY_CONFIG['BB1_WINDOW'], STRATEGY_CONFIG['BB1_STD_DEV'])
        df['bb2_upper'], df['bb2_middle'], df['bb2_lower'] = calculate_bollinger_bands(df['close'], STRATEGY_CONFIG['BB2_WINDOW'], STRATEGY_CONFIG['BB2_STD_DEV'])
        
        # ATR 계산 추가 (손절 로직 정보용)
        df.ta.atr(length=14, append=True) # 'ATRr_14' 컬럼 추가됨

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"데이터 수집/분석 오류 ({symbol}): {e}")
        return None

# ===== 신규 진입 신호 확인 함수 (전략 변경) =====
def check_for_entry_signal(df):
    if df is None or len(df) < 2: return None
    
    # Setup Candle은 -2 (직전 봉), Confirmation Candle은 -1 (최신 봉)
    setup = df.iloc[-2]
    confirm = df.iloc[-1]
    buffer = STRATEGY_CONFIG['TOUCH_BUFFER']
    sl_buffer = STRATEGY_CONFIG['SL_BUFFER']

    # 롱 진입 조건: 직전 봉(setup)이 양쪽 BB하단 터치 + 최신 봉(confirm)이 양봉
    is_setup_long = (setup['close'] <= setup['bb1_lower'] * (1 + buffer) and
                     setup['close'] <= setup['bb2_lower'] * (1 + buffer))
    is_confirm_long = confirm['close'] > confirm['open']

    if is_setup_long and is_confirm_long:
        stop_loss_price = setup['low'] * (1 - sl_buffer) # Setup 캔들의 저점보다 SL_BUFFER % 아래에 SL 설정
        return {
            "direction": "long", 
            "price": confirm['close'], 
            "stop_loss": stop_loss_price,
        }

    # 숏 진입 조건: 직전 봉(setup)이 양쪽 BB상단 터치 + 최신 봉(confirm)이 음봉
    is_setup_short = (setup['close'] >= setup['bb1_upper'] * (1 - buffer) and
                      setup['close'] >= setup['bb2_upper'] * (1 - buffer))
    is_confirm_short = confirm['close'] < confirm['open']

    if is_setup_short and is_confirm_short:
        stop_loss_price = setup['high'] * (1 + sl_buffer) # Setup 캔들의 고점보다 SL_BUFFER % 위에 SL 설정
        return {
            "direction": "short", 
            "price": confirm['close'], 
            "stop_loss": stop_loss_price,
        }
    
    return None

# ===== 거래 실행 함수 (SL 주문 추가) =====
def execute_trade(coin_name, signal, available_capital):
    symbol = TRADING_PAIRS[coin_name]['symbol']
    leverage = STRATEGY_CONFIG['LEVERAGE']
    investment = available_capital * STRATEGY_CONFIG['POSITION_SIZE']
    amount = (investment * leverage) / signal['price']
    order_side = 'buy' if signal['direction'] == 'long' else 'sell'
    sl_price = signal['stop_loss']

    try:
        print(f"\n--- {coin_name} {signal['direction'].upper()} 신규 포지션 진입 실행 ---")
        exchange.set_leverage(leverage, symbol)
        
        # 시장가 주문과 함께 손절(Stop-Loss) 주문 전송
        # 바이낸스 선물 API는 createOrder에 stopPrice 파라미터를 사용
        params = {
            'stopPrice': sl_price,
            'type': 'STOP_MARKET',
            'reduceOnly': True
        }
        
        # 1. 진입 주문
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order['price'] else signal['price']
        
        # 2. 손절 주문
        sl_side = 'sell' if signal['direction'] == 'long' else 'buy'
        exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, None, params)
        
        trade_data = {
            'coin_symbol': coin_name,
            'action': signal['direction'],
            'entry_price': entry_price,
            'amount': amount,
            'leverage': leverage,
        }
        save_trade_to_db(trade_data)
        
        print(f"✅ 포지션 진입 성공: {coin_name} @ ${entry_price:,.4f}")
        print(f"   - 투자금: ${investment:,.2f}, 수량: {amount:.4f}")
        print(f"   - 손절매(SL) 설정 완료: ${sl_price:,.4f}")
        return investment
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}")
        if 'binance' in str(e).lower() and 'stop price' in str(e).lower():
            print("   - 팁: SL 가격이 현재가와 너무 가깝거나(롱 포지션인데 SL이 현재가보다 높음 등), 주문 수량/가격 정밀도 문제일 수 있습니다.")
        return None


# ===== 오픈 포지션 관리 (청산) 함수 (기존 익절 로직 유지) =====
def manage_open_positions(open_positions):
    if not open_positions: return

    print("\n--- 오픈 포지션 익절 조건 확인 ---")
    for pos in open_positions:
        symbol = TRADING_PAIRS[pos['coin_symbol']]['symbol']
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            df = fetch_and_analyze_coin_data(symbol)
            if df is None: continue
            latest = df.iloc[-1]
            buffer = STRATEGY_CONFIG['TOUCH_BUFFER']

            exit_signal = False
            exit_reason = ""
            
            # 롱 포지션 익절 조건 확인
            if pos['action'] == 'long':
                is_touching_upper = (current_price >= latest['bb1_upper'] * (1 - buffer) or
                                     current_price >= latest['bb2_upper'] * (1 - buffer))
                if is_touching_upper:
                    exit_signal = True
                    reason_parts = []
                    if current_price >= latest['bb1_upper'] * (1-buffer): reason_parts.append("BB1 Upper Touch")
                    if current_price >= latest['bb2_upper'] * (1-buffer): reason_parts.append("BB2 Upper Touch")
                    exit_reason = " / ".join(reason_parts)

            # 숏 포지션 익절 조건 확인
            elif pos['action'] == 'short':
                is_touching_lower = (current_price <= latest['bb1_lower'] * (1 + buffer) or 
                                     current_price <= latest['bb2_lower'] * (1 + buffer))
                if is_touching_lower:
                    exit_signal = True
                    reason_parts = []
                    if current_price <= latest['bb1_lower'] * (1+buffer): reason_parts.append("BB1 Lower Touch")
                    if current_price <= latest['bb2_lower'] * (1+buffer): reason_parts.append("BB2 Lower Touch")
                    exit_reason = " / ".join(reason_parts)
                    
            # 익절 실행
            if exit_signal:
                print(f"🚨 익절 신호 포착: {pos['coin_symbol']} ({pos['action']}) - 이유: {exit_reason}")
                
                # 먼저 모든 오픈 주문(SL 주문 포함)을 취소
                exchange.cancel_all_orders(symbol)
                
                # 그 다음 포지션 종료
                close_side = 'sell' if pos['action'] == 'long' else 'buy'
                exchange.create_market_order(symbol, close_side, pos['amount'], params={'reduceOnly': True})
                
                pnl = (current_price - pos['entry_price']) * pos['amount'] if pos['action'] == 'long' else (pos['entry_price'] - current_price) * pos['amount']
                
                close_trade_in_db(pos['coin_symbol'], current_price, pnl, f"Take Profit: {exit_reason}")
                print(f"   - ✅ 포지션 익절 완료 @ ${current_price:,.4f}. P&L: ${pnl:,.2f}")
                time.sleep(3) 

        except Exception as e:
            print(f"포지션 관리 오류 ({pos['coin_symbol']}): {e}")

# ===== 메인 루프 (기존과 거의 동일) =====
def main():
    print("\n=== Dual Bollinger Band Strategy Bot (v1.1) Started ===")
    setup_database()

    while True:
        try:
            print(f"\n\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            # 1. 기존 포지션 익절 관리
            open_positions = get_open_positions_from_binance()
            manage_open_positions(open_positions)
            
            # 2. 신규 진입을 위해 포지션 정보 다시 로드
            current_positions = get_open_positions_from_binance()
            occupied_coins = {p['coin_symbol'] for p in current_positions}
            balance = exchange.fetch_balance()['USDT']
            available_capital = balance['free']
            
            print(f"\n--- 현재 상태 ---")
            print(f"가용 자본: ${available_capital:,.2f} | 보유 포지지션: {len(current_positions)}/{MAX_CONCURRENT_POSITIONS}개 ({', '.join(occupied_coins) if occupied_coins else '없음'})")

            # 3. 신규 포지션 진입 탐색
            if len(current_positions) < MAX_CONCURRENT_POSITIONS:
                print("\n--- 신규 진입 신호 탐색 ---")
                for coin_name, config in TRADING_PAIRS.items():
                    if coin_name in occupied_coins: continue
                    
                    if len(get_open_positions_from_binance()) >= MAX_CONCURRENT_POSITIONS:
                        print("   - 포지션 슬롯이 가득 찼습니다. 신규 탐색을 중단합니다.")
                        break

                    print(f"-> {coin_name} 스캔 중...")
                    df = fetch_and_analyze_coin_data(config['symbol'])
                    if df is None or df.empty: continue

                    signal = check_for_entry_signal(df)
                    
                    if signal:
                        print(f"🔥 진입 신호 포착! Coin: {coin_name}, Direction: {signal['direction']}")
                        used_capital = execute_trade(coin_name, signal, available_capital)
                        if used_capital:
                            available_capital -= used_capital
                            occupied_coins.add(coin_name)
                    else:
                        print(f"   - 조건 미충족.")
            
            wait_time = 300 # 5분 대기
            print(f"\n--- 사이클 완료. {int(wait_time/60)}분 후 다시 시작합니다. ---")
            time.sleep(wait_time)

        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()