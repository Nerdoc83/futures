"""
AI-Verified Multi-Coin Swing Trading Bot (v1.5 - Fixed 5-Minute Scan Cycle)
----------------------------------------------------------------
전략:
- 멀티코인 스윙 트레이딩 (BTC, ETH, SOL)
- 분석 타임프레임: 일봉 (1D)
- 핵심 전략: 볼린저밴드 + RSI + MACD를 결합한 'Tiered' 평균 회귀 전략
- AI 역할: 규칙 기반으로 포착된 신호를 최종 검증하는 '최고 분석가'
- 자금 관리:
    - Tier 2 (BB+RSI): 가용 자본의 20% 투자
    - Tier 1 (BB+RSI+MACD): 가용 자본의 30% 투자 (집중 투자)
- 청산 전략:
    - 손절(SL): ATR 기반 동적 손절매 (1.5 * ATR), 거래소에 STOP_MARKET 주문
    - 익절(TP): 2단계 분할 익절 (BB중심선 50% 익절 -> 반대편 밴드 50% 익절), 봇이 실시간 감시
- 포지션 관리: 3개 코인 슬롯을 항상 채우기 위해 지속적으로 빈 슬롯 탐색
- 스캔 주기: 5분 고정 주기로 모든 포지션 관리 및 신규 기회 탐색 (v1.5)
- 상세 로깅: 신호 미발생 시 주요 지표(BB, RSI) 값과 미충족 사유 출력 (v1.3)
- 진입 조건 완화: 볼린저밴드 근접 시(0.5% 버퍼)를 진입 조건으로 인정 (v1.4)
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
}

# ===== 전략 설정 =====
STRATEGY_CONFIG = {
    "TIMEFRAME": '1d',
    "TIER2_POSITION_SIZE": 0.20,
    "TIER1_POSITION_SIZE": 0.30,
    "LEVERAGE_TIER2": 5, # Tier 2 고정 레버리지
    "LEVERAGE_TIER1": 8, # Tier 1 고정 레버리지
    "ATR_SL_MULTIPLIER": 1.5,
    "BB_WINDOW": 20,
    "BB_STD_DEV": 2,
    "BB_PROXIMITY_BUFFER": 0.005, # 0.5% 버퍼 (v1.4)
    "RSI_WINDOW": 14,
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "MACD_CHECK_WINDOW": 3,
    "AI_CONVICTION_THRESHOLD": 85,
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

# ===== 데이터베이스 함수 =====
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
        profit_loss REAL DEFAULT 0
    )
    ''')
    conn.commit()
    conn.close()
    print("Swing Trading 데이터베이스 설정 완료")

def save_trade_to_db(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, coin_symbol, tier, action, entry_price, initial_amount, current_amount, leverage, sl_price, binance_order_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

def update_trade_in_db(trade_id, updates):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    fields = ', '.join([f"{key} = ?" for key in updates.keys()])
    values = list(updates.values())
    values.append(trade_id)
    cursor.execute(f"UPDATE trades SET {fields} WHERE id = ?", tuple(values))
    conn.commit()
    conn.close()

def get_open_trades():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' OR status = 'PARTIALLY_CLOSED'")
    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return trades

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
    
    # v1.4 변경점: 버퍼를 적용한 진입 영역 계산
    buffer = STRATEGY_CONFIG['BB_PROXIMITY_BUFFER']
    lower_band_zone = latest['lower_band'] * (1 + buffer)
    upper_band_zone = latest['upper_band'] * (1 - buffer)

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

# ===== AI 검증 함수 =====
def verify_signal_with_ai(signal):
    system_prompt = f"""당신은 암호화폐 스윙 트레이딩 전략을 검증하는 최고 리스크 분석가입니다. 당신의 역할은 '결정'이 아닌 '검증'입니다. "{signal['coin']}" 코인에 대해 "{signal['tier']}" 등급의 "{signal['direction']}" 신호가 포착되었습니다. 기술적 분석, 거시 경제, 시장 심리를 종합하여 이 거래의 신뢰도 점수(1-100), 핵심 근거, 잠재적 리스크를 분석하세요. 85점 이상이 권장됩니다. 반드시 아래의 JSON 형식으로만 응답해야 합니다: {{"conviction_score": <int>, "reasoning": "<요약>", "risk_analysis": "<분석>"}}"""
    try:
        response = model.generate_content(system_prompt)
        ai_response = json.loads(response.text.strip())
        print(f"\n=== AI Signal Verification: {signal['coin']} ===")
        print(f"Tier: {signal['tier']}, Direction: {signal['direction']}")
        print(f"AI Conviction Score: {ai_response.get('conviction_score')}")
        print(f"Reasoning: {ai_response.get('reasoning')}")
        print(f"Risk Analysis: {ai_response.get('risk_analysis')}")
        return ai_response
    except Exception as e:
        print(f"AI 분석 오류: {e}")
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
        
        # 시장가 주문
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = order['price']
        
        # SL 주문
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

def manage_open_positions():
    open_trades = get_open_trades()
    if not open_trades: return

    print("\n--- 오픈 포지션 관리 ---")
    for trade in open_trades:
        symbol = TRADING_PAIRS[trade['coin_symbol']]['symbol']
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            df = fetch_and_analyze_coin_data(symbol) # 최신 BB값 확인
            if df is None: continue
            latest_data = df.iloc[-1]

            is_long = trade['action'] == 'long'
            
            # TP1: BB 중심선 도달 (분할 익절)
            if trade['tp1_achieved'] == 0:
                tp1_price = latest_data['middle_band']
                if (is_long and current_price >= tp1_price) or (not is_long and current_price <= tp1_price):
                    print(f"🔥 TP1 도달: {trade['coin_symbol']} @ ${current_price:,.4f}")
                    
                    # 50% 물량 익절
                    close_amount = trade['initial_amount'] / 2
                    close_side = 'sell' if is_long else 'buy'
                    exchange.create_market_order(symbol, close_side, close_amount, params={'reduceOnly': True})
                    
                    # 기존 SL 주문 취소
                    exchange.cancel_order(trade['binance_order_id'], symbol)
                    
                    # SL을 본절(entry_price)로 재설정
                    new_sl_order = exchange.create_order(symbol, 'STOP_MARKET', close_side, close_amount, params={'stopPrice': trade['entry_price'], 'reduceOnly': True})
                    
                    # DB 업데이트
                    pnl = (current_price - trade['entry_price']) * close_amount if is_long else (trade['entry_price'] - current_price) * close_amount
                    update_trade_in_db(trade['id'], {
                        'status': 'PARTIALLY_CLOSED',
                        'current_amount': close_amount,
                        'binance_order_id': new_sl_order['id'],
                        'tp1_achieved': 1,
                        'profit_loss': trade['profit_loss'] + pnl
                    })
                    print(f"   - 50% 익절 완료, SL을 본절(${trade['entry_price']})로 이동.")

            # TP2: 반대편 BB 도달 (최종 익절)
            else:
                tp2_price = latest_data['upper_band'] if is_long else latest_data['lower_band']
                if (is_long and current_price >= tp2_price) or (not is_long and current_price <= tp2_price):
                    print(f"🚀 TP2 도달: {trade['coin_symbol']} @ ${current_price:,.4f}")
                    
                    # 남은 물량 전량 익절
                    close_side = 'sell' if is_long else 'buy'
                    exchange.create_market_order(symbol, close_side, trade['current_amount'], params={'reduceOnly': True})
                    
                    # SL 주문 취소
                    exchange.cancel_order(trade['binance_order_id'], symbol)
                    
                    # DB 업데이트
                    pnl = (current_price - trade['entry_price']) * trade['current_amount'] if is_long else (trade['entry_price'] - current_price) * trade['current_amount']
                    update_trade_in_db(trade['id'], {
                        'status': 'CLOSED',
                        'current_amount': 0,
                        'profit_loss': trade['profit_loss'] + pnl
                    })
                    print(f"   - 최종 익절 완료. 거래 종료.")

        except Exception as e:
            print(f"포지션 관리 오류 ({trade['coin_symbol']}): {e}")

# ===== 메인 루프 =====
def main():
    print("\n=== AI-Verified Multi-Coin Swing Trading Bot (v1.5 Complete) Started ===")
    setup_database()

    while True:
        try:
            print(f"\n\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            # 1. 오픈된 포지션 관리 (매 사이클마다 가격 체크)
            manage_open_positions()

            # 2. 현재 포지션 및 가용 자본 확인
            open_trades = get_open_trades()
            occupied_coins = {t['coin_symbol'] for t in open_trades}
            balance = exchange.fetch_balance()['USDT']
            available_capital = balance['free']
            
            print(f"\n--- 현재 상태 ---")
            print(f"가용 자본: ${available_capital:,.2f} | 보유 포지션: {len(open_trades)}개 ({occupied_coins})")

            # 3. 빈 슬롯에 대해 신호 탐색
            if len(occupied_coins) < len(TRADING_PAIRS):
                print("\n--- 빈 슬롯 신호 탐색 ---")
                for coin_name, config in TRADING_PAIRS.items():
                    if coin_name in occupied_coins: continue

                    print(f"-> {coin_name} 스캔 중...")
                    df = fetch_and_analyze_coin_data(config['symbol'])
                    if df is None: continue

                    signal = check_for_trading_signal(coin_name, df)
                    
                    if not signal:
                        latest = df.iloc[-1]
                        price = latest['close']
                        upper_band = latest['upper_band']
                        lower_band = latest['lower_band']
                        rsi = latest['rsi']
                        
                        buffer = STRATEGY_CONFIG['BB_PROXIMITY_BUFFER']
                        lower_band_zone = lower_band * (1 + buffer)
                        upper_band_zone = upper_band * (1 - buffer)

                        bb_status_text = f"밴드 내 위치 ({lower_band:,.2f} ~ {upper_band:,.2f})"
                        if price <= lower_band_zone:
                            bb_status_text = f"하단 근접 ({lower_band_zone:,.2f})"
                        elif price >= upper_band_zone:
                            bb_status_text = f"상단 근접 ({upper_band_zone:,.2f})"
                            
                        print(f"   - 조건 미충족. 현재가: ${price:,.2f}, BB: {bb_status_text}, RSI: {rsi:.2f}")
                        continue
                    
                    print(f"🔥 신호 포착! Coin: {coin_name}, Tier: {signal['tier']}, Direction: {signal['direction']}")
                    ai_verification = verify_signal_with_ai(signal)
                    
                    if ai_verification.get('conviction_score', 0) >= STRATEGY_CONFIG['AI_CONVICTION_THRESHOLD']:
                        print("✅ AI 검증 통과. 거래를 실행합니다.")
                        used_capital = execute_trade(signal, available_capital)
                        if used_capital:
                            available_capital -= used_capital
                            occupied_coins.add(signal['coin'])
                    else:
                        print("❌ AI 검증 실패. 거래를 진행하지 않습니다.")
                    time.sleep(5)
            
            # 4. 고정 스캔 주기 설정 (v1.5 변경점)
            wait_time = 300  # 5분
            print(f"\n--- 사이클 완료. {int(wait_time/60)}분 후 다시 시작합니다. ---")
            time.sleep(wait_time)

        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

