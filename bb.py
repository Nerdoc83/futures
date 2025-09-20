"""
Dual Bollinger Band Strategy Bot (v1.4 - 15min + RSI Filter with Detailed Logging)
----------------------------------------------------------------
전략:
- 멀티코인 단기 트레이딩 (BTC, ETH 등 8개 코인)
- 분석 타임프레임: 15분봉 (15m)
- 핵심 전략: 15분봉의 과매도/과매수 상태를 듀얼 볼린저밴드와 RSI로 확인 후, 반전 캔들에서 정밀 진입

- 지표 설정:
    1. 메인 볼린저밴드 (BB1): 20 periods, 2 standard deviations
    2. 보조 볼린저밴드 (BB2): 4 periods, 4 standard deviations
    3. 상대강도지수 (RSI): 14 periods

- 진입 조건 (Entry):
    - 롱(Long):
        1. (Setup) 15분봉이 BB1 하단과 BB2 하단을 모두 터치/돌파
        2. (Filter) 동시에 해당 봉의 RSI가 30 미만 (과매도)
        3. (Confirm) 바로 다음 15분봉이 양봉으로 마감 시 진입
    - 숏(Short):
        1. (Setup) 15분봉이 BB1 상단과 BB2 상단을 모두 터치/돌파
        2. (Filter) 동시에 해당 봉의 RSI가 70 초과 (과매수)
        3. (Confirm) 바로 다음 15분봉이 음봉으로 마감 시 진입

- 청산 조건 (Exit):
    - 익절(Take Profit): 기존과 동일 (반대편 밴드 터치)
    - 손절(Stop Loss): 기존과 동일 (Setup 캔들의 고점/저점 기준 자동 설정)

- 관리:
    - 스캔 주기: 1분마다 모든 코인을 스캔하여 진입/청산 신호 확인
    - 포지션 관리: 최대 5개의 동시 포지션 유지
----------------------------------------------------------------
"""

# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import time
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from datetime import datetime
import pandas_ta as ta

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
    "TIMEFRAME": '15m',
    "POSITION_SIZE": 0.20,
    "LEVERAGE": 5,
    
    # 볼린저밴드 1 설정
    "BB1_WINDOW": 20,
    "BB1_STD_DEV": 2,
    
    # 볼린저밴드 2 설정
    "BB2_WINDOW": 4,
    "BB2_STD_DEV": 4,

    # RSI 설정 추가
    "RSI_PERIOD": 14,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
    
    "TOUCH_BUFFER": 0.001,
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
DB_FILE = "multi_coin_daytrading_v1_4.db" # DB 파일명 변경하여 이전 전략과 분리

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
    print(f"Trading DB '{DB_FILE}' 설정 완료")

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

# ===== 데이터 수집 및 분석 함수 (기존과 동일) =====
def fetch_and_analyze_coin_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=STRATEGY_CONFIG['TIMEFRAME'], limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['bb1_upper'], df['bb1_middle'], df['bb1_lower'] = calculate_bollinger_bands(df['close'], STRATEGY_CONFIG['BB1_WINDOW'], STRATEGY_CONFIG['BB1_STD_DEV'])
        df['bb2_upper'], df['bb2_middle'], df['bb2_lower'] = calculate_bollinger_bands(df['close'], STRATEGY_CONFIG['BB2_WINDOW'], STRATEGY_CONFIG['BB2_STD_DEV'])
        
        df.ta.rsi(length=STRATEGY_CONFIG['RSI_PERIOD'], append=True)

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"데이터 수집/분석 오류 ({symbol}): {e}")
        return None

# ===== [수정됨] 신규 진입 신호 확인 함수 (상세 사유 반환 기능 추가) =====
def check_for_entry_signal(df):
    if df is None or len(df) < 2:
        return None, "데이터프레임이 비어있거나 분석에 필요한 데이터 길이가 부족합니다."
    
    cfg = STRATEGY_CONFIG
    setup = df.iloc[-2]
    confirm = df.iloc[-1]
    
    rsi_col = f'RSI_{cfg["RSI_PERIOD"]}'

    # 롱 진입 조건 검사
    is_bb_touch_low = (setup['close'] <= setup['bb1_lower'] * (1 + cfg['TOUCH_BUFFER']) and
                       setup['close'] <= setup['bb2_lower'] * (1 + cfg['TOUCH_BUFFER']))
    
    if is_bb_touch_low:
        is_rsi_oversold = setup[rsi_col] < cfg['RSI_OVERSOLD']
        if not is_rsi_oversold:
            return None, f"BB 하단 터치, 그러나 RSI({setup[rsi_col]:.2f})가 과매도({cfg['RSI_OVERSOLD']}) 아님"
        
        is_confirm_bullish = confirm['close'] > confirm['open']
        if not is_confirm_bullish:
            return None, "BB 하단 터치 및 RSI 과매도, 그러나 다음 봉이 양봉 마감되지 않음"
            
        # 모든 롱 조건 충족
        stop_loss_price = setup['low'] * (1 - cfg['SL_BUFFER'])
        signal = {
            "direction": "long", 
            "price": confirm['close'], 
            "stop_loss": stop_loss_price,
        }
        return signal, "롱 진입 조건 충족"

    # 숏 진입 조건 검사
    is_bb_touch_high = (setup['close'] >= setup['bb1_upper'] * (1 - cfg['TOUCH_BUFFER']) and
                        setup['close'] >= setup['bb2_upper'] * (1 - cfg['TOUCH_BUFFER']))
    
    if is_bb_touch_high:
        is_rsi_overbought = setup[rsi_col] > cfg['RSI_OVERBOUGHT']
        if not is_rsi_overbought:
            return None, f"BB 상단 터치, 그러나 RSI({setup[rsi_col]:.2f})가 과매수({cfg['RSI_OVERBOUGHT']}) 아님"

        is_confirm_bearish = confirm['close'] < confirm['open']
        if not is_confirm_bearish:
            return None, "BB 상단 터치 및 RSI 과매수, 그러나 다음 봉이 음봉 마감되지 않음"

        # 모든 숏 조건 충족
        stop_loss_price = setup['high'] * (1 + cfg['SL_BUFFER'])
        signal = {
            "direction": "short", 
            "price": confirm['close'], 
            "stop_loss": stop_loss_price,
        }
        return signal, "숏 진입 조건 충족"
    
    # 위 조건들에 아무것도 해당되지 않을 경우
    return None, "BB 밴드 상/하단을 터치하지 않음"

# ===== 거래 실행 함수 (기존과 동일) =====
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
        
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order['price'] else signal['price']
        
        sl_side = 'sell' if signal['direction'] == 'long' else 'buy'
        params = {'stopPrice': sl_price, 'reduceOnly': True}
        exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, None, params)
        
        trade_data = {
            'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price,
            'amount': amount, 'leverage': leverage,
        }
        save_trade_to_db(trade_data)
        
        print(f"✅ 포지션 진입 성공: {coin_name} @ ${entry_price:,.4f}")
        print(f"   - 투자금: ${investment:,.2f}, 수량: {amount:.4f}")
        print(f"   - 손절매(SL) 설정 완료: ${sl_price:,.4f}")
        return investment
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}")
        return None

# ===== 오픈 포지션 관리 (청산) 함수 (기존과 동일) =====
def manage_open_positions(open_positions):
    if not open_positions: return

    print("\n--- 오픈 포지션 익절 조건 확인 ---")
    for pos in open_positions:
        symbol = TRADING_PAIRS[pos['coin_symbol']]['symbol']
        try:
            df = fetch_and_analyze_coin_data(symbol)
            if df is None: continue
            
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            latest = df.iloc[-1]
            buffer = STRATEGY_CONFIG['TOUCH_BUFFER']

            exit_signal = False
            exit_reason = ""
            
            if pos['action'] == 'long':
                if (current_price >= latest['bb1_upper'] * (1 - buffer) or
                    current_price >= latest['bb2_upper'] * (1 - buffer)):
                    exit_signal = True
                    exit_reason = "Take Profit: Upper BB Touch"

            elif pos['action'] == 'short':
                if (current_price <= latest['bb1_lower'] * (1 + buffer) or 
                    current_price <= latest['bb2_lower'] * (1 + buffer)):
                    exit_signal = True
                    exit_reason = "Take Profit: Lower BB Touch"
                    
            if exit_signal:
                print(f"🚨 익절 신호 포착: {pos['coin_symbol']} ({pos['action']}) - 이유: {exit_reason}")
                exchange.cancel_all_orders(symbol)
                close_side = 'sell' if pos['action'] == 'long' else 'buy'
                exchange.create_market_order(symbol, close_side, pos['amount'], params={'reduceOnly': True})
                
                pnl = (current_price - pos['entry_price']) * pos['amount'] if pos['action'] == 'long' else (pos['entry_price'] - current_price) * pos['amount']
                close_trade_in_db(pos['coin_symbol'], current_price, pnl, exit_reason)
                print(f"   - ✅ 포지션 익절 완료 @ ${current_price:,.4f}. P&L: ${pnl:,.2f}")
                time.sleep(3) 

        except Exception as e:
            print(f"포지션 관리 오류 ({pos['coin_symbol']}): {e}")

# ===== 메인 루프 =====
def main():
    print("\n=== Dual BB Strategy Bot (v1.4 - 15min + RSI + Log) Started ===")
    setup_database()

    while True:
        try:
            print(f"\n\n\n[{datetime.now().strftime('%Y-m-d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            open_positions = get_open_positions_from_binance()
            manage_open_positions(open_positions)
            
            current_positions = get_open_positions_from_binance()
            occupied_coins = {p['coin_symbol'] for p in current_positions}
            
            if len(current_positions) < MAX_CONCURRENT_POSITIONS:
                balance = exchange.fetch_balance()['USDT']
                available_capital = balance['free']
                print(f"\n--- 현재 상태 ---")
                print(f"가용 자본: ${available_capital:,.2f} | 보유 포지션: {len(current_positions)}/{MAX_CONCURRENT_POSITIONS}개")
                print("\n--- 신규 진입 신호 탐색 ---")

                for coin_name, config in TRADING_PAIRS.items():
                    if coin_name in occupied_coins: continue
                    if len(get_open_positions_from_binance()) >= MAX_CONCURRENT_POSITIONS:
                        break

                    print(f"-> {coin_name} 스캔 중...")
                    df = fetch_and_analyze_coin_data(config['symbol'])
                    if df is None or df.empty: continue
                    
                    # [수정됨] 반환값을 signal과 reason으로 함께 받음
                    signal, reason = check_for_entry_signal(df)
                    
                    if signal:
                        print(f"🔥 진입 신호 포착! Coin: {coin_name}, Direction: {signal['direction']} (이유: {reason})")
                        used_capital = execute_trade(coin_name, signal, available_capital)
                        if used_capital:
                            available_capital -= used_capital
                            occupied_coins.add(coin_name)
                    else:
                        # [수정됨] 진입 조건이 맞지 않았을 때, 구체적인 이유를 출력
                        print(f"   - {reason}")
            else:
                 print(f"\n--- 현재 상태: 포지션 슬롯({len(current_positions)}/{MAX_CONCURRENT_POSITIONS}) 가득 참 ---")

            wait_time = 60
            print(f"\n--- 사이클 완료. {wait_time}초 후 다시 시작합니다. ---")
            time.sleep(wait_time)

        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()