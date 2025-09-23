"""
Dual Bollinger Band Strategy Bot (v5.3 - Stop-Loss Adjusted to 2%)
----------------------------------------------------------------
전략:
- 거래대금 상위 코인 단기 트레이딩
- 분석 타임프레임: 1시간봉 (Real-time Setup/Filter) + 5분봉 (Confirmation)
- 핵심 전략: '실시간 가격'이 1시간봉 BB/RSI 조건을 만족하면 '초기 신호'로 감지.
             신호의 강도에 따라 투자금을 1배 또는 2배로 조절하여 최종 진입.

- 지표 설정:
    1. 메인 볼린저밴드 (BB1): 20 periods, 2 standard deviations
    2. 보조 볼린저밴드 (BB2): 4 periods, 4 standard deviations
    3. 상대강도지수 (RSI): 14 periods

- 진입 조건 (Entry):
    - 일반 신호 (1x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1 '또는' BB2 중 하나를 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 5분봉이 완성되어 반전을 '확증'하면 기본 투자금으로 진입
    - 강력 신호 (2x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1 '그리고' BB2를 '모두' 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 5분봉이 완성되어 반전을 '확증'하면 '2배'의 투자금으로 진입

- 청산 조건 (Exit):
    - 익절(Take Profit): '1시간봉' 기준 BB1 '또는' BB2 반대편 밴드 터치
    - 손절(Stop Loss): '실제 진입 가격' 기준으로 -2% 설정 (10배 레버리지 시 -20% 손실)

- 관리:
    - 포지션 크기: 첫 진입 시 '전체 자본의 20%'로 기본 투자금을 고정.
----------------------------------------------------------------
"""

# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import time
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas_ta as ta

# .env 파일 로드
load_dotenv()

# ===== 동시 포지션 제한 설정 =====
MAX_CONCURRENT_POSITIONS = 5

# ===== 전략 설정 =====
STRATEGY_CONFIG = {
    "POSITION_SIZE": 0.20, "LEVERAGE": 10,
    "BB1_WINDOW": 20, "BB1_STD_DEV": 2,
    "BB2_WINDOW": 4, "BB2_STD_DEV": 4,
    "RSI_PERIOD": 14, "RSI_OVERBOUGHT": 70, "RSI_OVERSOLD": 30,
    "SL_BUFFER": 0.02 # 2% (10x leverage = 20% loss)
}

# ===== API 및 DB 설정 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key, 'secret': secret, 'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
DB_FILE = "multi_coin_daytrading.db" # DB 파일명 고정

# ===== 확증 대기 신호 저장소 =====
pending_confirmation = {}

# ===== 고정 투자금 관리 변수 =====
fixed_investment_per_trade = 0

# ===== 거래대금 상위 코인 조회 함수 (기존과 동일) =====
def get_top_volume_coins(exchange, limit=10):
    try:
        tickers = exchange.fetch_tickers()
        usdt_futures = {s: t for s, t in tickers.items() if s.endswith('/USDT:USDT')}
        sorted_tickers = sorted(usdt_futures.values(), key=lambda x: x.get('quoteVolume', 0), reverse=True)
        top_coins = {}
        for ticker in sorted_tickers[:limit]:
            symbol_for_trade = ticker['symbol'].split(':')[0]
            coin_name = symbol_for_trade.split('/')[0]
            top_coins[coin_name] = {"symbol": symbol_for_trade}
        print(f"거래대금 상위 {limit}개 코인: {list(top_coins.keys())}")
        return top_coins
    except Exception as e:
        print(f"거래대금 상위 코인 조회 오류: {e}")
        return {}

# ===== 기술 지표 계산 함수 (기존과 동일) =====
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
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, coin_symbol TEXT NOT NULL,
        action TEXT NOT NULL, entry_price REAL NOT NULL, amount REAL NOT NULL, leverage INTEGER NOT NULL,
        status TEXT DEFAULT 'OPEN', exit_price REAL, profit_loss REAL, exit_reason TEXT, exit_timestamp TEXT
    )''')
    conn.commit(); conn.close()
    print(f"Trading DB '{DB_FILE}' 설정 완료")

def save_trade_to_db(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, coin_symbol, action, entry_price, amount, leverage, status)
    VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (datetime.now().isoformat(), trade_data['coin_symbol'], trade_data['action'],
          trade_data['entry_price'], trade_data['amount'], trade_data['leverage']))
    conn.commit(); conn.close()

def close_trade_in_db(coin_symbol, exit_price, pnl, reason):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE trades SET status = 'CLOSED', exit_price = ?, profit_loss = ?, exit_reason = ?, exit_timestamp = ?
    WHERE coin_symbol = ? AND status = 'OPEN'
    ''', (exit_price, pnl, reason, datetime.now().isoformat(), coin_symbol))
    conn.commit(); conn.close()

# ===== 포지션 조회 함수 (기존과 동일) =====
def get_open_positions_from_binance():
    try:
        positions = exchange.fetch_positions()
        open_positions = []
        for pos in positions:
            if float(pos['info'].get('positionAmt', 0)) != 0:
                coin_symbol = pos['symbol'].replace('/USDT:USDT', '')
                open_positions.append({
                    "coin_symbol": coin_symbol,
                    "action": 'long' if float(pos['info']['positionAmt']) > 0 else 'short',
                    "entry_price": float(pos.get('entryPrice', 0)),
                    "amount": float(pos.get('contracts', 0)),
                    "leverage": int(pos['info'].get('leverage', 0))
                })
        return open_positions
    except Exception as e:
        print(f"바이낸스 포지션 조회 오류: {e}"); return []

# ===== 데이터 수집 및 분석 함수 (기존과 동일) =====
def fetch_and_analyze_data(symbol, timeframe, cfg, calculate_bb=False, calculate_rsi=False):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if calculate_bb:
            df['bb1_upper'], _, df['bb1_lower'] = calculate_bollinger_bands(df['close'], cfg['BB1_WINDOW'], cfg['BB1_STD_DEV'])
            df['bb2_upper'], _, df['bb2_lower'] = calculate_bollinger_bands(df['close'], cfg['BB2_WINDOW'], cfg['BB2_STD_DEV'])
        if calculate_rsi:
            df.ta.rsi(length=cfg['RSI_PERIOD'], append=True)
        df.dropna(inplace=True); return df
    except Exception as e:
        print(f"데이터 수집/분석 오류 ({symbol}, {timeframe}): {e}"); return None

# ===== 초기 신호 감지 함수 (기존과 동일) =====
def check_for_initial_signal(df_1h, current_price):
    if df_1h is None or len(df_1h) < 1: return None, "1시간봉 데이터 부족"
    cfg = STRATEGY_CONFIG
    current_1h_candle = df_1h.iloc[-1]
    rsi_col = f'RSI_{cfg["RSI_PERIOD"]}'

    is_bb1_touch_low = current_price <= current_1h_candle['bb1_lower']
    is_bb2_touch_low = current_price <= current_1h_candle['bb2_lower']
    is_rsi_oversold = current_1h_candle[rsi_col] < cfg['RSI_OVERSOLD']

    if (is_bb1_touch_low or is_bb2_touch_low) and is_rsi_oversold:
        size_multiplier = 2.0 if is_bb1_touch_low and is_bb2_touch_low else 1.0
        signal_type = "강력(2x)" if size_multiplier == 2.0 else "일반(1x)"
        signal = {
            "direction": "long",
            "size_multiplier": size_multiplier
        }
        reason = f"초기 롱 신호({signal_type}) 감지 @ ${current_price:,.2f} (1h RSI: {current_1h_candle[rsi_col]:.2f})"
        return signal, reason

    is_bb1_touch_high = current_price >= current_1h_candle['bb1_upper']
    is_bb2_touch_high = current_price >= current_1h_candle['bb2_upper']
    is_rsi_overbought = current_1h_candle[rsi_col] > cfg['RSI_OVERBOUGHT']

    if (is_bb1_touch_high or is_bb2_touch_high) and is_rsi_overbought:
        size_multiplier = 2.0 if is_bb1_touch_high and is_bb2_touch_high else 1.0
        signal_type = "강력(2x)" if size_multiplier == 2.0 else "일반(1x)"
        signal = {
            "direction": "short",
            "size_multiplier": size_multiplier
        }
        reason = f"초기 숏 신호({signal_type}) 감지 @ ${current_price:,.2f} (1h RSI: {current_1h_candle[rsi_col]:.2f})"
        return signal, reason

    return None, "초기 신호 없음"

# ===== 거래 실행 함수 (기존과 동일) =====
def execute_trade(coin_name, signal, investment_amount):
    symbol = f"{coin_name}/USDT"
    cfg = STRATEGY_CONFIG
    leverage = cfg['LEVERAGE']
    current_price = exchange.fetch_ticker(symbol)['last']
    amount = (investment_amount * leverage) / current_price
    order_side = 'buy' if signal['direction'] == 'long' else 'sell'
    
    try:
        print(f"\n--- {coin_name} {signal['direction'].upper()} 신규 포지션 진입 실행 ---")
        exchange.set_leverage(leverage, symbol)
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order['price'] else current_price
        
        if signal['direction'] == 'long':
            sl_price = entry_price * (1 - cfg['SL_BUFFER'])
            sl_side = 'sell'
        else: # short
            sl_price = entry_price * (1 + cfg['SL_BUFFER'])
            sl_side = 'buy'

        exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
        
        trade_data = {'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price,
                      'amount': amount, 'leverage': leverage}
        save_trade_to_db(trade_data)
        
        print(f"✅ 포지션 진입 성공: {coin_name} @ ${entry_price:,.4f}")
        print(f"   - 투자금: ${investment_amount:,.2f}, 수량: {amount:.4f}")
        print(f"   - 손절매(SL) 설정 완료: ${sl_price:,.4f} (진입가 대비 -{cfg['SL_BUFFER']*100:.2f}%)")
        return True
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}"); return False

# ===== 오픈 포지션 관리 함수 (기존과 동일) =====
def manage_open_positions(open_positions):
    if not open_positions: return
    print("\n--- 오픈 포지션 익절 조건 확인 ---")
    for pos in open_positions:
        symbol = f"{pos['coin_symbol']}/USDT"
        try:
            df_1h = fetch_and_analyze_data(symbol, '1h', STRATEGY_CONFIG, calculate_bb=True)
            if df_1h is None: continue
            current_price = exchange.fetch_ticker(symbol)['last']
            latest = df_1h.iloc[-1]
            exit_signal, exit_reason = False, ""
            if pos['action'] == 'long' and (current_price >= latest['bb1_upper'] or current_price >= latest['bb2_upper']):
                exit_signal, exit_reason = True, "Take Profit: Upper BB Touch (1h)"
            elif pos['action'] == 'short' and (current_price <= latest['bb1_lower'] or current_price <= latest['bb2_lower']):
                exit_signal, exit_reason = True, "Take Profit: Lower BB Touch (1h)"
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

# ===== 메인 루프 (기존과 동일) =====
def main():
    global fixed_investment_per_trade
    print("\n=== Dual BB Strategy Bot (v5.3 - Stop-Loss Adjusted to 2%) Started ===")
    setup_database()
    while True:
        try:
            now = datetime.now()
            print(f"\n\n\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            coins_to_scan = get_top_volume_coins(exchange, limit=10)
            if not coins_to_scan:
                print("코인 리스트를 가져오지 못했습니다. 60초 후 재시도합니다.")
                time.sleep(60)
                continue

            open_positions = get_open_positions_from_binance()
            manage_open_positions(open_positions)
            
            current_positions = get_open_positions_from_binance()
            if not current_positions and not pending_confirmation:
                if fixed_investment_per_trade != 0:
                    print("모든 포지션 종료. 다음 사이클에서 투자금을 재계산합니다.")
                    fixed_investment_per_trade = 0
            
            confirmed_coins = []
            for coin, data in list(pending_confirmation.items()):
                if now >= data['confirmation_timestamp']:
                    print(f"-> {coin} 5분봉 확증 확인 중...")
                    symbol = f"{coin}/USDT"
                    df_5m = fetch_and_analyze_data(symbol, '5m', STRATEGY_CONFIG)
                    if df_5m is None or len(df_5m) < 2:
                        confirmed_coins.append(coin); continue
                    
                    last_5m_candle = df_5m.iloc[-2]
                    is_confirmed = (data['direction'] == 'long' and last_5m_candle['close'] > last_5m_candle['open']) or \
                                   (data['direction'] == 'short' and last_5m_candle['close'] < last_5m_candle['open'])

                    if is_confirmed:
                        base_investment = data['investment_amount']
                        multiplier = data.get('size_multiplier', 1.0)
                        final_investment = base_investment * multiplier
                        
                        balance = exchange.fetch_balance()['USDT']
                        if balance['free'] < final_investment:
                            print(f"   - ❌ 자본 부족으로 {coin} 진입 실패 (필요: ${final_investment:,.2f}, 보유: ${balance['free']:,.2f})")
                        else:
                            print(f"   - 🔥 확증 성공! {coin} {data['direction']} 포지션 진입 시도 (투자금: ${final_investment:,.2f}).")
                            execute_trade(coin, data, final_investment)
                    else:
                        print(f"   - ❌ 확증 실패. {coin} 신호 폐기.")
                    confirmed_coins.append(coin)

            for coin in confirmed_coins: del pending_confirmation[coin]
            
            current_positions = get_open_positions_from_binance()
            occupied_slots = len(current_positions) + len(pending_confirmation)
            
            if occupied_slots < MAX_CONCURRENT_POSITIONS:
                if fixed_investment_per_trade == 0 and occupied_slots == 0:
                    balance = exchange.fetch_balance()['USDT']
                    fixed_investment_per_trade = balance['total'] * STRATEGY_CONFIG['POSITION_SIZE']
                    print(f"새로운 거래 사이클 시작. 포지션당 기본 투자금 설정: ${fixed_investment_per_trade:,.2f}")

                if fixed_investment_per_trade > 0:
                    balance = exchange.fetch_balance()['USDT']
                    if balance['free'] < fixed_investment_per_trade:
                        print(f"\n--- 가용 자본 부족으로 신규 진입 탐색 중단 (최소 필요: ${fixed_investment_per_trade:,.2f}, 보유: ${balance['free']:,.2f}) ---")
                    else:
                        print(f"\n--- 현재 상태: 보유 {len(current_positions)}개, 대기 {len(pending_confirmation)}개. ({occupied_slots}/{MAX_CONCURRENT_POSITIONS}) ---")
                        print(f"--- 신규 초기 신호 탐색 (기본 투자금: ${fixed_investment_per_trade:,.2f}) ---")
                        for coin, config in coins_to_scan.items():
                            if coin in [p['coin_symbol'] for p in current_positions] or coin in pending_confirmation: continue
                            if len(get_open_positions_from_binance()) + len(pending_confirmation) >= MAX_CONCURRENT_POSITIONS: break
                            try:
                                print(f"-> {coin} 스캔 중...")
                                current_price = exchange.fetch_ticker(config['symbol'])['last']
                                df_1h = fetch_and_analyze_data(config['symbol'], '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True)
                                if df_1h is None: continue
                                initial_signal, reason = check_for_initial_signal(df_1h, current_price)
                                if initial_signal:
                                    print(f"   - 🚨 {reason}. 5분봉 확증 대기열에 추가.")
                                    minutes_to_next_5m = 5 - (now.minute % 5)
                                    confirm_time = now.replace(second=5, microsecond=0) + timedelta(minutes=minutes_to_next_5m)
                                    initial_signal['confirmation_timestamp'] = confirm_time
                                    initial_signal['investment_amount'] = fixed_investment_per_trade
                                    pending_confirmation[coin] = initial_signal
                                    print(f"   - (확증 예정 시각: {confirm_time.strftime('%H:%M:%S')})")
                                else:
                                    print(f"   - ({reason})")
                            except Exception as e:
                                print(f"스캔 오류 ({coin}): {e}")
            else:
                print(f"\n--- 현재 상태: 포지션 슬롯({occupied_slots}/{MAX_CONCURRENT_POSITIONS}) 가득 참 ---")

            print(f"\n--- 사이클 완료. 60초 후 다시 시작합니다. ---")
            time.sleep(60)
        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}"); time.sleep(60)

if __name__ == "__main__":
    main()
