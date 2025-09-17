"""
AI-Verified Multi-Coin Swing Trading Bot (v2.3 - AI Retry Logic)
----------------------------------------------------------------
전략:
- 멀티코인 스윙 트레이딩 (BTC, ETH, SOL, XRP, ADA, AVAX, LINK, DOGE)
- 분석 타임프레임: 일봉 (1D)
- 핵심 전략: 볼린저밴드 + RSI + MACD를 결합한 'Tiered' 평균 회귀 전략
- AI 역할: 규칙 기반으로 포착된 신호를 최종 검증하는 '최고 분석가' (Gemini 2.5 Pro)
- 자금 관리:
    - Tier 2 (BB+RSI): 가용 자본의 20% 투자
    - Tier 1 (BB+RSI+MACD): 가용 자본의 30% 투자 (집중 투자)
- 청산 전략:
    - 손절(SL): ATR 기반 동적 손절매 (1.5 * ATR), 거래소에 STOP_MARKET 주문
    - 익절(TP): 2단계 분할 익절 (BB중심선 50% 익절 -> 반대편 밴드 50% 익절), 봇이 실시간 감시
- 포지션 관리: 최대 5개의 동시 포지션 유지. DB가 아닌 바이낸스에서 직접 포지션 정보를 가져와 동기화
- 스캔 주기: 5분 고정 주기로 모든 포지션 관리 및 신규 기회 탐색

=== v2.3 변경 사항 ===
- AI 응답 실패 시, 즉시 포기하지 않고 최대 3회까지 자동으로 재시도하는 로직 추가
- 봇의 안정성을 높여 일시적인 네트워크/API 오류로 인한 기회 손실 방지
----------------------------------------------------------------
"""

# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import time
import pandas as pd
import numpy as np
import json
import sqlite3
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

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
    "TIMEFRAME": '1d',
    "TIER2_POSITION_SIZE": 0.20,
    "TIER1_POSITION_SIZE": 0.30,
    "LEVERAGE_TIER2": 5,
    "LEVERAGE_TIER1": 8,
    "ATR_SL_MULTIPLIER": 1.5,
    "BB_WINDOW": 20,
    "BB_STD_DEV": 2,
    "DYNAMIC_BB_BUFFER_RATIO": 0.25,
    "RSI_WINDOW": 14,
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "MACD_CHECK_WINDOW": 3,
    "AI_CONVICTION_THRESHOLD": 85,
    "AI_RETRY_ATTEMPTS": 3, # v2.3 추가: AI 요청 재시도 횟수
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
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')
DB_FILE = "multi_coin_daytrading.db"

# ===== 기술 지표 계산 함수 =====
def calculate_bollinger_bands(prices, window, std_dev):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

def calculate_rsi(prices, window):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast, slow, signal):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

# ===== 데이터베이스 함수 (기록용) =====
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        coin_symbol TEXT NOT NULL,
        tier TEXT NOT NULL,
        action TEXT NOT NULL,
        entry_price REAL NOT NULL,
        initial_amount REAL NOT NULL,
        current_amount REAL NOT NULL,
        leverage INTEGER NOT NULL,
        sl_price REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        binance_order_id TEXT,
        tp1_achieved INTEGER DEFAULT 0,
        profit_loss REAL DEFAULT 0,
        exit_timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print("Swing Trading 데이터베이스 설정 완료")

def save_trade_to_db(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, coin_symbol, tier, action, entry_price, initial_amount, current_amount, leverage, sl_price, binance_order_id, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (
        datetime.now().isoformat(),
        trade_data['coin_symbol'],
        trade_data['tier'],
        trade_data['action'],
        trade_data['entry_price'],
        trade_data['amount'],
        trade_data['amount'],
        trade_data['leverage'],
        trade_data['sl_price'],
        trade_data['sl_order_id']
    ))
    conn.commit()
    conn.close()

def find_trade_in_db(coin_symbol, action):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE coin_symbol = ? AND action = ? AND (status = 'OPEN' OR status = 'PARTIALLY_CLOSED') ORDER BY timestamp DESC LIMIT 1", (coin_symbol, action))
    trade = cursor.fetchone()
    conn.close()
    return dict(trade) if trade else None

def update_trade_in_db(trade_id, updates):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    fields = ', '.join([f"{key} = ?" for key in updates.keys()])
    values = list(updates.values())
    values.append(trade_id)
    cursor.execute(f"UPDATE trades SET {fields} WHERE id = ?", tuple(values))
    conn.commit()
    conn.close()

# ===== 바이낸스에서 직접 포지션 조회 =====
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
                })
        return open_positions
    except Exception as e:
        print(f"바이낸스 포지션 조회 오류: {e}")
        return []

# ===== 데이터 수집 및 분석 함수 =====
def fetch_and_analyze_coin_data(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=STRATEGY_CONFIG['TIMEFRAME'], limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(df['close'], STRATEGY_CONFIG['BB_WINDOW'], STRATEGY_CONFIG['BB_STD_DEV'])
        df['rsi'] = calculate_rsi(df['close'], STRATEGY_CONFIG['RSI_WINDOW'])
        df['macd'], df['macd_signal'] = calculate_macd(df['close'], STRATEGY_CONFIG['MACD_FAST'], STRATEGY_CONFIG['MACD_SLOW'], STRATEGY_CONFIG['MACD_SIGNAL'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"데이터 수집/분석 오류 ({symbol}): {e}")
        return None

def check_for_trading_signal(coin_name, df):
    if df is None or len(df) < 2: return None
    latest = df.iloc[-1]
    
    dynamic_buffer_ratio = STRATEGY_CONFIG['DYNAMIC_BB_BUFFER_RATIO']
    band_width_to_mid_lower = latest['middle_band'] - latest['lower_band']
    lower_band_zone = latest['lower_band'] + (band_width_to_mid_lower * dynamic_buffer_ratio)
    band_width_to_mid_upper = latest['upper_band'] - latest['middle_band']
    upper_band_zone = latest['upper_band'] - (band_width_to_mid_upper * dynamic_buffer_ratio)

    is_long_signal = latest['close'] <= lower_band_zone and latest['rsi'] < STRATEGY_CONFIG['RSI_OVERSOLD']
    is_short_signal = latest['close'] >= upper_band_zone and latest['rsi'] > STRATEGY_CONFIG['RSI_OVERBOUGHT']
    
    if not is_long_signal and not is_short_signal: return None

    signal = {
        "coin": coin_name, "tier": "Tier 2", "direction": "long" if is_long_signal else "short",
        "price": latest['close'], "atr": latest['atr'],
        "bb_middle": latest['middle_band'], "bb_opposite": latest['upper_band'] if is_long_signal else latest['lower_band']
    }
    
    recent_df = df.tail(STRATEGY_CONFIG['MACD_CHECK_WINDOW'])
    if signal['direction'] == 'long' and any((recent_df['macd'].shift(1) < recent_df['macd_signal'].shift(1)) & (recent_df['macd'] > recent_df['macd_signal'])):
        signal['tier'] = "Tier 1"
    elif signal['direction'] == 'short' and any((recent_df['macd'].shift(1) > recent_df['macd_signal'].shift(1)) & (recent_df['macd'] < recent_df['macd_signal'])):
        signal['tier'] = "Tier 1"
    return signal

# ===== AI 검증 함수 (v2.3 수정: 재시도 로직 추가) =====
def verify_signal_with_ai(signal):
    system_prompt = f"""당신은 암호화폐 스윙 트레이딩 전략을 검증하는 최고 리스크 분석가입니다. 당신의 역할은 '결정'이 아닌 '검증'입니다. "{signal['coin']}" 코인에 대해 "{signal['tier']}" 등급의 "{signal['direction']}" 신호가 포착되었습니다. 기술적 분석, 거시 경제, 시장 심리를 종합하여 이 거래의 신뢰도 점수(1-100), 핵심 근거, 잠재적 리스크를 분석하세요. 85점 이상이 권장됩니다. 반드시 아래의 JSON 형식으로만 응답해야 합니다: {{"conviction_score": <int>, "reasoning": "<요약>", "risk_analysis": "<분석>"}}"""
    
    attempts = STRATEGY_CONFIG['AI_RETRY_ATTEMPTS']
    for attempt in range(attempts):
        raw_response_text = ""
        try:
            print(f"   - AI 분석 요청 시도 ({attempt + 1}/{attempts})...")
            response = model.generate_content(system_prompt)
            raw_response_text = response.text.strip()

            if not raw_response_text:
                print(f"   - AI 분석 오류: 모델이 빈 응답을 반환했습니다. 재시도합니다.")
                time.sleep(2) # 재시도 전 2초 대기
                continue # 다음 재시도 실행

            ai_response = json.loads(raw_response_text)
            print(f"\n=== AI Signal Verification: {signal['coin']} ===")
            print(f"Tier: {signal['tier']}, Direction: {signal['direction']}")
            print(f"AI Conviction Score: {ai_response.get('conviction_score')}")
            print(f"Reasoning: {ai_response.get('reasoning')}")
            print(f"Risk Analysis: {ai_response.get('risk_analysis')}")
            return ai_response # 성공 시 즉시 결과 반환

        except json.JSONDecodeError as e:
            print(f"   - AI 분석 오류 (JSON 파싱 실패): {e}")
            print(f"   - AI가 반환한 원본 내용: '{raw_response_text}'")
            time.sleep(2)
            continue
            
        except Exception as e:
            print(f"   - AI 분석 중 예기치 않은 오류 발생: {e}")
            time.sleep(2)
            continue

    print(f"❌ AI 분석 최종 실패: {attempts}번의 시도 후에도 유효한 응답을 받지 못했습니다.")
    return {"conviction_score": 0}

# ===== 거래 실행 및 관리 함수 =====
def execute_trade(signal, available_capital):
    symbol = TRADING_PAIRS[signal['coin']]['symbol']
    position_size_pct = STRATEGY_CONFIG[f"TIER{signal['tier'][-1]}_POSITION_SIZE"]
    leverage = STRATEGY_CONFIG[f"LEVERAGE_TIER{signal['tier'][-1]}"]
    investment = available_capital * position_size_pct
    amount = (investment * leverage) / signal['price']
    
    if signal['direction'] == 'long':
        sl_price = signal['price'] - (signal['atr'] * STRATEGY_CONFIG['ATR_SL_MULTIPLIER'])
        order_side = 'buy'
        sl_side = 'sell'
    else:
        sl_price = signal['price'] + (signal['atr'] * STRATEGY_CONFIG['ATR_SL_MULTIPLIER'])
        order_side = 'sell'
        sl_side = 'buy'

    try:
        print(f"\n--- {signal['coin']} ({signal['tier']}) {signal['direction'].upper()} 거래 실행 ---")
        exchange.set_leverage(leverage, symbol)
        
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price'])
        
        sl_order = exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, params={'stopPrice': sl_price, 'reduceOnly': True})
        
        trade_data = {
            'coin_symbol': signal['coin'], 'tier': signal['tier'], 'action': signal['direction'],
            'entry_price': entry_price, 'amount': amount, 'leverage': leverage,
            'sl_price': sl_price, 'sl_order_id': sl_order['id']
        }
        save_trade_to_db(trade_data)
        
        print(f"✅ 포지션 진입 성공: {signal['coin']} @ ${entry_price:,.4f}")
        print(f"   - 투자금: ${investment:,.2f}, 수량: {amount:.4f}")
        print(f"   - SL 설정: ${sl_price:,.4f} (주문 ID: {sl_order['id']})")
        return investment
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}")
        return None

def manage_open_positions(open_positions):
    if not open_positions: return

    print("\n--- 오픈 포지션 관리 ---")
    for pos in open_positions:
        symbol = TRADING_PAIRS[pos['coin_symbol']]['symbol']
        db_trade = find_trade_in_db(pos['coin_symbol'], pos['action'])
        if not db_trade:
            print(f"   - 경고: {pos['coin_symbol']} 포지션이 DB에 없습니다. (수동 거래 가능성)")
            continue
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            df = fetch_and_analyze_coin_data(symbol)
            if df is None: continue
            latest_data = df.iloc[-1]

            is_long = db_trade['action'] == 'long'
            
            if db_trade['tp1_achieved'] == 0:
                tp1_price = latest_data['middle_band']
                if (is_long and current_price >= tp1_price) or (not is_long and current_price <= tp1_price):
                    print(f"🔥 TP1 도달: {db_trade['coin_symbol']} @ ${current_price:,.4f}")
                    close_amount = db_trade['initial_amount'] / 2
                    close_side = 'sell' if is_long else 'buy'
                    exchange.create_market_order(symbol, close_side, close_amount, params={'reduceOnly': True})
                    open_orders = exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        if order['id'] == db_trade['binance_order_id']:
                            exchange.cancel_order(db_trade['binance_order_id'], symbol)
                            break
                    new_sl_order = exchange.create_order(symbol, 'STOP_MARKET', close_side, close_amount, params={'stopPrice': db_trade['entry_price'], 'reduceOnly': True})
                    pnl = (current_price - db_trade['entry_price']) * close_amount if is_long else (db_trade['entry_price'] - current_price) * close_amount
                    update_trade_in_db(db_trade['id'], {
                        'status': 'PARTIALLY_CLOSED',
                        'current_amount': close_amount,
                        'binance_order_id': new_sl_order['id'],
                        'tp1_achieved': 1,
                        'profit_loss': db_trade['profit_loss'] + pnl,
                        'exit_timestamp': datetime.now().isoformat()
                    })
                    print(f"   - 50% 익절 완료, SL을 본절(${db_trade['entry_price']})로 이동.")

            else: # TP1 달성 후
                tp2_price = latest_data['upper_band'] if is_long else latest_data['lower_band']
                if (is_long and current_price >= tp2_price) or (not is_long and current_price <= tp2_price):
                    print(f"🚀 TP2 도달: {db_trade['coin_symbol']} @ ${current_price:,.4f}")
                    close_side = 'sell' if is_long else 'buy'
                    exchange.create_market_order(symbol, close_side, db_trade['current_amount'], params={'reduceOnly': True})
                    open_orders = exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        if order['id'] == db_trade['binance_order_id']:
                            exchange.cancel_order(db_trade['binance_order_id'], symbol)
                            break
                    pnl = (current_price - db_trade['entry_price']) * db_trade['current_amount'] if is_long else (db_trade['entry_price'] - current_price) * db_trade['current_amount']
                    update_trade_in_db(db_trade['id'], {
                        'status': 'CLOSED',
                        'current_amount': 0,
                        'profit_loss': db_trade['profit_loss'] + pnl,
                        'exit_timestamp': datetime.now().isoformat()
                    })
                    print(f"   - 최종 익절 완료. 거래 종료.")

        except Exception as e:
            print(f"포지션 관리 오류 ({db_trade['coin_symbol']}): {e}")

# ===== 메인 루프 =====
def main():
    print("\n=== AI-Verified Multi-Coin Swing Trading Bot (v2.3 AI Retry Logic) Started ===")
    setup_database()

    while True:
        try:
            print(f"\n\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            open_positions = get_open_positions_from_binance()
            manage_open_positions(open_positions)
            
            open_positions = get_open_positions_from_binance()
            occupied_coins = {p['coin_symbol'] for p in open_positions}
            balance = exchange.fetch_balance()['USDT']
            available_capital = balance['free']
            
            print(f"\n--- 현재 상태 ---")
            print(f"가용 자본: ${available_capital:,.2f} | 보유 포지션: {len(open_positions)}/{MAX_CONCURRENT_POSITIONS}개 ({occupied_coins})")

            if len(occupied_coins) < MAX_CONCURRENT_POSITIONS:
                print("\n--- 빈 슬롯 신호 탐색 ---")
                for coin_name, config in TRADING_PAIRS.items():
                    if coin_name in occupied_coins: continue
                    if len(get_open_positions_from_binance()) >= MAX_CONCURRENT_POSITIONS: break

                    print(f"-> {coin_name} 스캔 중...")
                    df = fetch_and_analyze_coin_data(config['symbol'])
                    if df is None: continue

                    signal = check_for_trading_signal(coin_name, df)
                    
                    if not signal:
                        latest = df.iloc[-1]
                        price = latest['close']
                        upper_band = latest['upper_band']
                        lower_band = latest['lower_band']
                        middle_band = latest['middle_band']
                        rsi = latest['rsi']
                        
                        dynamic_buffer_ratio = STRATEGY_CONFIG['DYNAMIC_BB_BUFFER_RATIO']
                        band_width_to_mid_lower = middle_band - lower_band
                        lower_band_zone = lower_band + (band_width_to_mid_lower * dynamic_buffer_ratio)
                        band_width_to_mid_upper = upper_band - middle_band
                        upper_band_zone = upper_band - (band_width_to_mid_upper * dynamic_buffer_ratio)

                        bb_status_text = f"밴드 내 위치 ({lower_band:,.2f} ~ {upper_band:,.2f})"
                        if price <= lower_band_zone:
                            bb_status_text = f"하단 존 진입 ({lower_band_zone:,.2f} 이하)"
                        elif price >= upper_band_zone:
                            bb_status_text = f"상단 존 진입 ({upper_band_zone:,.2f} 이상)"
                            
                        print(f"   - 조건 미충족. 현재가: ${price:,.2f}, BB: {bb_status_text}, RSI: {rsi:.2f}")
                        continue
                    
                    print(f"🔥 신호 포착! Coin: {coin_name}, Tier: {signal['tier']}, Direction: {signal['direction']}")
                    ai_verification = verify_signal_with_ai(signal)
                    
                    if ai_verification.get('conviction_score', 0) >= STRATEGY_CONFIG['AI_CONVICTION_THRESHOLD']:
                        print("✅ AI 검증 통과. 거래를 실행합니다.")
                        used_capital = execute_trade(signal, available_capital)
                        if used_capital:
                            available_capital -= used_capital
                    else:
                        print("❌ AI 검증 실패. 거래를 진행하지 않습니다.")
                    time.sleep(5)
            
            wait_time = 300
            print(f"\n--- 사이클 완료. {int(wait_time/60)}분 후 다시 시작합니다. ---")
            time.sleep(wait_time)

        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
