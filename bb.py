"""
Dual Bollinger Band Strategy Bot (v5.30 - 15m Confirmation)
----------------------------------------------------------------
전략:
- 거래대금 상위 코인 단기 트레이딩
- 분석 타임프레임: 1시간봉 (Real-time Setup/Filter) + 15분봉 (Confirmation)
- 핵심 전략: '실시간 가격'이 1시간봉 BB/RSI 조건을 만족하면 '초기 신호'로 감지.
             신호의 강도에 따라 투자금을 1배 또는 2배로 조절하여 최종 진입.

- 지표 설정:
    1. 메인 볼린저밴드 (BB1): 20 periods, 2 standard deviations
    2. 보조 볼린저밴드 (BB2): 4 periods, 4 standard deviations
    3. 상대강도지수 (RSI): 14 periods

- 진입 조건 (Entry):
    - 일반 신호 (1x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1 '또는' BB2 중 하나를 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 15분봉이 완성되어 반전을 '확증'하면 기본 투자금으로 진입
    - 강력 신호 (2x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1 '그리고' BB2를 '모두' 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 15분봉이 완성되어 반전을 '확증'하면 '2배'의 투자금으로 진입

- 청산 조건 (Exit):
    - 분할 익절(Partial Take Profit):
        1. (1차 익절) '1시간봉' 기준 BB1 중앙선(20MA)에 1% 근접 시, 보유 물량의 50% 익절
        2. (최종 익절) '1시간봉' 기준 BB1/BB2 반대편 밴드에 1% 근접 시, 남은 물량 전체 익절
    - 손절(Stop Loss): '증거금' 기준 -20% 손실로 고정 (10배 레버리지 시, 진입 가격 대비 -2% 변동 시)

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
from datetime import datetime, timedelta, timezone
import pandas_ta as ta

# .env 파일 로드
load_dotenv()

# ===== 동시 포지션 제한 설정 =====
MAX_CONCURRENT_POSITIONS = 5

# ===== 전략 설정 =====
STRATEGY_CONFIG = {
    "MARGIN_SIZE": 0.20, "LEVERAGE": 10,
    "BB1_WINDOW": 20, "BB1_STD_DEV": 2,
    "BB2_WINDOW": 4, "BB2_STD_DEV": 4,
    "RSI_PERIOD": 14, "RSI_OVERBOUGHT": 70, "RSI_OVERSOLD": 30,
    "TP_BUFFER": 0.01, # 익절을 위한 1% 버퍼
    "MAX_DATA_DELAY_MINUTES": 120 # 데이터 신선도 허용 최대 지연 시간 (분)
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

# ===== 분할 익절 상태 관리 =====
partial_tp_status = {}

# ===== 거래대금 상위 코인 조회 함수 =====
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

# ===== 데이터베이스 함수 =====
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

# ===== 포지션 조회 함수 =====
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

# ===== 데이터 수집 및 분석 함수 =====
def fetch_and_analyze_data(symbol, timeframe, cfg, calculate_bb=False, calculate_rsi=False):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        if not ohlcv:
            print(f"      ㄴ ({symbol}) API로부터 0개의 캔들 데이터 수신")
            return None

        last_candle_timestamp_ms = ohlcv[-1][0]
        last_candle_dt = datetime.fromtimestamp(last_candle_timestamp_ms / 1000, tz=timezone.utc)
        current_dt_utc = datetime.now(timezone.utc)
        time_diff_minutes = (current_dt_utc - last_candle_dt).total_seconds() / 60

        if time_diff_minutes > cfg['MAX_DATA_DELAY_MINUTES']:
            print(f"      ㄴ ({symbol}) 수신된 데이터가 너무 오래되었습니다. (최신 캔들: {int(time_diff_minutes)}분 전)")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if len(df) < cfg['BB1_WINDOW']:
            print(f"      ㄴ ({symbol}) BB 계산에 필요한 데이터 부족. 수신: {len(df)}개, 필요: {cfg['BB1_WINDOW']}개")
            return None

        if calculate_bb:
            bb1_middle = df['close'].rolling(window=cfg['BB1_WINDOW']).mean()
            bb1_std = df['close'].rolling(window=cfg['BB1_WINDOW']).std()
            df['bb1_upper'] = bb1_middle + (bb1_std * cfg['BB1_STD_DEV'])
            df['bb1_middle'] = bb1_middle
            df['bb1_lower'] = bb1_middle - (bb1_std * cfg['BB1_STD_DEV'])

            bb2_middle = df['close'].rolling(window=cfg['BB2_WINDOW']).mean()
            bb2_std = df['close'].rolling(window=cfg['BB2_WINDOW']).std()
            df['bb2_upper'] = bb2_middle + (bb2_std * cfg['BB2_STD_DEV'])
            df['bb2_lower'] = bb2_middle - (bb2_std * cfg['BB2_STD_DEV'])

        if calculate_rsi:
            df.ta.rsi(length=cfg['RSI_PERIOD'], append=True)

        df.dropna(inplace=True)
        if df.empty:
            return None

        return df
    except Exception as e:
        print(f"      ㄴ 데이터 수집/분석 상세 오류 ({symbol}): {e}")
        return None

# ===== 초기 신호 감지 함수 =====
def check_for_initial_signal(df_1h, current_price):
    if df_1h is None or len(df_1h) < 1: return None, "1시간봉 데이터 부족"
    cfg = STRATEGY_CONFIG
    current_1h_candle = df_1h.iloc[-1]
    rsi_col = f'RSI_{cfg["RSI_PERIOD"]}'
    if rsi_col not in current_1h_candle:
        return None, "RSI 데이터를 계산할 수 없음"
    rsi_value = current_1h_candle[rsi_col]

    is_bb1_touch_low = current_price <= current_1h_candle['bb1_lower']
    is_bb2_touch_low = current_price <= current_1h_candle['bb2_lower']
    is_bb_touch_low = is_bb1_touch_low or is_bb2_touch_low
    is_rsi_oversold = rsi_value < cfg['RSI_OVERSOLD']

    if is_bb_touch_low and is_rsi_oversold:
        size_multiplier = 2.0 if is_bb1_touch_low and is_bb2_touch_low else 1.0
        signal_type = "강력(2x)" if size_multiplier == 2.0 else "일반(1x)"
        signal = { "direction": "long", "size_multiplier": size_multiplier }
        reason = f"초기 롱 신호({signal_type}) 감지 @ ${current_price:,.2f} (1h RSI: {rsi_value:.2f})"
        return signal, reason

    is_bb1_touch_high = current_price >= current_1h_candle['bb1_upper']
    is_bb2_touch_high = current_price >= current_1h_candle['bb2_upper']
    is_bb_touch_high = is_bb1_touch_high or is_bb2_touch_high
    is_rsi_overbought = rsi_value > cfg['RSI_OVERBOUGHT']

    if is_bb_touch_high and is_rsi_overbought:
        size_multiplier = 2.0 if is_bb1_touch_high and is_bb2_touch_high else 1.0
        signal_type = "강력(2x)" if size_multiplier == 2.0 else "일반(1x)"
        signal = { "direction": "short", "size_multiplier": size_multiplier }
        reason = f"초기 숏 신호({signal_type}) 감지 @ ${current_price:,.2f} (1h RSI: {rsi_value:.2f})"
        return signal, reason

    bb_long_status = "O" if is_bb_touch_low else "X"
    rsi_long_status = "O" if is_rsi_oversold else "X"
    bb_short_status = "O" if is_bb_touch_high else "X"
    rsi_short_status = "O" if is_rsi_overbought else "X"

    reason = f"롱[BB:{bb_long_status},RSI:{rsi_long_status}] 숏[BB:{bb_short_status},RSI:{rsi_short_status}] (RSI:{rsi_value:.2f})"
    return None, reason

# ===== [최종 수정] 거래 실행 함수 - 주문 수량 계산 로직 수정 =====
def execute_trade(coin_name, signal, base_investment, multiplier):
    global partial_tp_status
    symbol = f"{coin_name}/USDT"
    cfg = STRATEGY_CONFIG
    leverage = cfg['LEVERAGE']
    final_investment = base_investment * multiplier # 최종 증거금 (예: 9.99 USDT)

    current_price = exchange.fetch_ticker(symbol)['last']
    
    # [수정됨] 최종 포지션 크기에 해당하는 코인 수량을 계산하도록 변경
    # (기존) amount = final_investment / current_price
    amount = (final_investment * leverage) / current_price

    order_side = 'buy' if signal['direction'] == 'long' else 'sell'

    try:
        print(f"\n--- {coin_name} {signal['direction'].upper()} 신규 포지션 진입 실행 ---")
        exchange.set_leverage(leverage, symbol)
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order['price'] else current_price

        # 증거금의 20% 손실 지점을 가격으로 계산 (레버리지 반영)
        sl_price_change_percentage = 0.20 / leverage

        if signal['direction'] == 'long':
            sl_price = entry_price * (1 - sl_price_change_percentage)
            sl_side = 'sell'
        else: # short
            sl_price = entry_price * (1 + sl_price_change_percentage)
            sl_side = 'buy'

        exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, None, {'stopPrice': sl_price, 'reduceOnly': True})

        trade_data = {'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price,
                      'amount': amount, 'leverage': leverage}
        save_trade_to_db(trade_data)

        partial_tp_status[coin_name] = False

        log_msg_signal_type = f"(강력 신호: {multiplier}x)" if multiplier > 1 else "(일반 신호)"
        
        # 최종 포지션 크기를 정확히 표시하도록 print문 수정
        final_position_size = amount * entry_price

        print(f"✅ 포지션 진입 성공: {coin_name} @ ${entry_price:,.4f} {log_msg_signal_type}")
        print(f"   - 증거금(Margin): ${final_investment:,.2f}")
        print(f"   - 포지션 크기(Size): 약 ${final_position_size:,.2f}") # 정확한 포지션 크기 표시
        print(f"   - 수량: {amount:.4f} {coin_name.replace('USDT', '')}")
        print(f"   - 손절매(SL) 설정 완료: ${sl_price:,.4f} (증거금 대비 -20%)")
        return True
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}"); return False

# ===== 오픈 포지션 관리 함수 =====
def manage_open_positions(open_positions):
    global partial_tp_status
    print("\n--- 오픈 포지션 익절 조건 확인 ---")
    if not open_positions:
        print("   -> 현재 오픈된 포지션이 없습니다.")
        return

    for pos in open_positions:
        symbol = f"{pos['coin_symbol']}/USDT"
        cfg = STRATEGY_CONFIG
        tp_buffer = cfg.get("TP_BUFFER", 0)

        try:
            df_1h = fetch_and_analyze_data(symbol, '1h', cfg, calculate_bb=True)
            if df_1h is None: continue

            current_price = exchange.fetch_ticker(symbol)['last']
            latest = df_1h.iloc[-1]
            has_partial_tp_executed = partial_tp_status.get(pos['coin_symbol'], False)

            if not has_partial_tp_executed:
                partial_tp_triggered = False
                if pos['action'] == 'long':
                    middle_band_target = latest['bb1_middle'] * (1 - tp_buffer)
                    if current_price >= middle_band_target:
                        partial_tp_triggered = True
                elif pos['action'] == 'short':
                    middle_band_target = latest['bb1_middle'] * (1 + tp_buffer)
                    if current_price <= middle_band_target:
                        partial_tp_triggered = True

                if partial_tp_triggered:
                    current_pos_info = next((p for p in get_open_positions_from_binance() if p['coin_symbol'] == pos['coin_symbol']), None)
                    if not current_pos_info: continue
                    amount_to_close = current_pos_info['amount'] / 2

                    close_side = 'sell' if pos['action'] == 'long' else 'buy'
                    print(f"🚨 1차 익절(50%) 신호 포착: {pos['coin_symbol']} ({pos['action']}) - 이유: BB1 중앙선 근접")
                    exchange.create_market_order(symbol, close_side, amount_to_close, params={'reduceOnly': True})
                    partial_tp_status[pos['coin_symbol']] = True
                    print(f"   - ✅ 1차 익절 완료 @ ${current_price:,.4f}. (수량: {amount_to_close:.4f})")
                    time.sleep(3)
                    continue

            final_tp_triggered = False
            if pos['action'] == 'long':
                tp_target_bb1 = latest['bb1_upper'] * (1 - tp_buffer)
                tp_target_bb2 = latest['bb2_upper'] * (1 - tp_buffer)
                if current_price >= tp_target_bb1 or current_price >= tp_target_bb2:
                    final_tp_triggered = True
            elif pos['action'] == 'short':
                tp_target_bb1 = latest['bb1_lower'] * (1 - tp_buffer)
                tp_target_bb2 = latest['bb2_lower'] * (1 + tp_buffer)
                if current_price <= tp_target_bb1 or current_price <= tp_target_bb2:
                    final_tp_triggered = True

            if final_tp_triggered:
                reason = f"최종 익절: 반대편 BB 근접 (버퍼: {tp_buffer*100}%)"
                print(f"🚨 {reason} 신호 포착: {pos['coin_symbol']} ({pos['action']})")
                exchange.cancel_all_orders(symbol)
                remaining_pos = next((p for p in get_open_positions_from_binance() if p['coin_symbol'] == pos['coin_symbol']), None)
                if remaining_pos:
                    close_side = 'sell' if pos['action'] == 'long' else 'buy'
                    exchange.create_market_order(symbol, close_side, remaining_pos['amount'], params={'reduceOnly': True})
                    pnl = (current_price - pos['entry_price']) * pos['amount']
                    close_trade_in_db(pos['coin_symbol'], current_price, pnl, reason)
                    print(f"   - ✅ 최종 익절 완료 @ ${current_price:,.4f}.")
                    if pos['coin_symbol'] in partial_tp_status:
                        del partial_tp_status[pos['coin_symbol']]
                    time.sleep(3)
        except Exception as e:
            print(f"포지션 관리 오류 ({pos['coin_symbol']}): {e}")

# ===== 메인 루프 - 잔고 확인 코드 추가 =====
def main():
    global fixed_investment_per_trade, partial_tp_status
    print("\n=== Dual BB Strategy Bot (v5.30 - 15m Confirmation) Started ===")
    setup_database()
    while True:
        try:
            now = datetime.now()
            print(f"\n\n\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")

            coins_to_scan = get_top_volume_coins(exchange, limit=15)
            if not coins_to_scan:
                print("코인 리스트를 가져오지 못했습니다. 60초 후 재시도합니다.")
                time.sleep(60)
                continue

            open_positions = get_open_positions_from_binance()
            open_position_symbols = {p['coin_symbol'] for p in open_positions}
            for coin in list(partial_tp_status.keys()):
                if coin not in open_position_symbols:
                    del partial_tp_status[coin]

            manage_open_positions(open_positions)

            current_positions_after_manage = get_open_positions_from_binance()
            if not current_positions_after_manage and not pending_confirmation:
                if fixed_investment_per_trade != 0:
                    print("\n모든 포지션 종료. 다음 사이클에서 투자금을 재계산합니다.")
                    fixed_investment_per_trade = 0

            confirmed_coins = []
            if pending_confirmation:
                print("\n--- 15분봉 확증 확인 ---")
            for coin, data in list(pending_confirmation.items()):
                if now >= data['confirmation_timestamp']:
                    print(f"-> {coin} ({data['direction']}) 확증 확인 중...")
                    symbol = f"{coin}/USDT"
                    df_15m = fetch_and_analyze_data(symbol, '15m', STRATEGY_CONFIG)
                    if df_15m is None or len(df_15m) < 2:
                        confirmed_coins.append(coin); continue

                    last_15m_candle = df_15m.iloc[-2]
                    is_confirmed = (data['direction'] == 'long' and last_15m_candle['close'] > last_15m_candle['open']) or \
                                   (data['direction'] == 'short' and last_15m_candle['close'] < last_15m_candle['open'])

                    if is_confirmed:
                        base_investment = data['investment_amount']
                        multiplier = data.get('size_multiplier', 1.0)

                        balance = exchange.fetch_balance()['USDT']
                        if balance['free'] < (base_investment * multiplier):
                            print(f"   - ❌ 자본 부족으로 {coin} 진입 실패 (필요: ${base_investment * multiplier:,.2f}, 보유: ${balance['free']:,.2f})")
                        else:
                            execute_trade(coin, data, base_investment, multiplier)
                    else:
                        print(f"   - ❌ 확증 실패. {coin} 신호 폐기.")
                    confirmed_coins.append(coin)

            for coin in confirmed_coins: del pending_confirmation[coin]

            current_positions = get_open_positions_from_binance()
            occupied_slots = len(current_positions) + len(pending_confirmation)

            if fixed_investment_per_trade == 0:
                print("\n--- 선물 지갑 잔고 확인 중 ---")
                try:
                    balance = exchange.fetch_balance()['USDT']
                    total_capital = balance['total']
                    
                    print(f"✅ [디버그] API가 확인한 선물 지갑 상태:")
                    print(f"    - 총 자산 (Total): {total_capital:.4f} USDT")
                    print(f"    - 사용 가능 (Free): {balance['free']:.4f} USDT")
                    print(f"    - 주문 사용 중 (Used): {balance['used']:.4f} USDT")

                    fixed_investment_per_trade = total_capital * STRATEGY_CONFIG['MARGIN_SIZE']
                    
                    print(f"\n포지션당 기본 투자금(증거금) 설정 완료:")
                    print(f"   -> 계산: {total_capital:.4f} USDT (총 자산) * {STRATEGY_CONFIG['MARGIN_SIZE']} (20%)")
                    print(f"   -> 결과: ${fixed_investment_per_trade:.4f} USDT")

                except Exception as e:
                    print(f"❌ 잔고 조회 중 심각한 오류 발생: {e}")
                    print("60초 후 재시도합니다...")
                    time.sleep(60)
                    continue

            print(f"\n--- 현재 상태 요약 ---")
            print(f"포지션: {len(current_positions)}/{MAX_CONCURRENT_POSITIONS} | 확증 대기: {len(pending_confirmation)} | 총 슬롯: {occupied_slots}/{MAX_CONCURRENT_POSITIONS}")
            if current_positions:
                pos_summary = []
                for p in current_positions:
                    status = " (1차 익절)" if partial_tp_status.get(p['coin_symbol']) else ""
                    pos_summary.append(f"{p['coin_symbol']}({p['action']}){status}")
                print(f"   - 보유 포지션: {', '.join(pos_summary)}")
            if pending_confirmation:
                pend_summary = [f"{coin}({data['direction']})" for coin, data in pending_confirmation.items()]
                print(f"   - 확증 대기: {', '.join(pend_summary)}")

            if occupied_slots < MAX_CONCURRENT_POSITIONS:
                if fixed_investment_per_trade > 0:
                    balance = exchange.fetch_balance()['USDT']
                    if balance['free'] < fixed_investment_per_trade:
                        print(f"\n--- 신규 진입 탐색 중단: 가용 자본 부족 (최소 필요: ${fixed_investment_per_trade:,.2f}, 보유: ${balance['free']:,.2f}) ---")
                    else:
                        print(f"\n--- 신규 진입 탐색 시작 (기본 투자금: ${fixed_investment_per_trade:,.2f}) ---")
                        for coin, config in coins_to_scan.items():
                            if len(get_open_positions_from_binance()) + len(pending_confirmation) >= MAX_CONCURRENT_POSITIONS:
                                print("\n포지션 슬롯이 가득 차서 신규 탐색을 중단합니다.")
                                break

                            print(f"-> {coin} 스캔 중...")

                            if coin in [p['coin_symbol'] for p in current_positions] or coin in pending_confirmation:
                                print(f"   - (보유 또는 대기 중인 포지션, 건너뜁니다)")
                                continue

                            try:
                                current_price = exchange.fetch_ticker(config['symbol'])['last']
                                df_1h = fetch_and_analyze_data(config['symbol'], '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True)

                                if df_1h is None:
                                    print(f"   - (1시간봉 데이터 수집 실패, 상세 오류는 위 로그 확인)")
                                    continue

                                initial_signal, reason = check_for_initial_signal(df_1h, current_price)

                                if initial_signal:
                                    print(f"   - 🚨 {reason}. 15분봉 확증 대기열에 추가.")
                                    minutes_to_next_15m = 15 - (now.minute % 15)
                                    confirm_time = now.replace(second=5, microsecond=0) + timedelta(minutes=minutes_to_next_15m)
                                    initial_signal['confirmation_timestamp'] = confirm_time
                                    initial_signal['investment_amount'] = fixed_investment_per_trade
                                    pending_confirmation[coin] = initial_signal
                                    print(f"     (확증 예정 시각: {confirm_time.strftime('%H:%M:%S')})")
                                else:
                                    print(f"   - ({reason})")
                            except Exception as e:
                                print(f"   - (스캔 중 오류 발생: {e})")
            else:
                print(f"\n--- 신규 진입 탐색 중단: 모든 포지션 슬롯이 사용 중입니다. ---")

            print(f"\n--- 사이클 완료. 60초 후 다시 시작합니다. ---")
            time.sleep(60)
        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}"); time.sleep(60)

if __name__ == "__main__":
    main()

