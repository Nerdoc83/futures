"""
Dual Bollinger Band Strategy Bot (v6.2 - Enhanced Logging)
----------------------------------------------------------------
전략:
- 거래대금 상위 코인 단기 트레이딩
- 분석 타임프레임: 1시간봉 (Real-time Setup/Filter) + 15분봉 (Confirmation) + AI (Final Confirmation)
- 핵심 전략: '실시간 가격'이 1시간봉 BB/RSI 조건을 만족하면 '초기 신호'로 감지. AI의 최종 판단 후 진입.

- 지표 설정:
    1. 메인 볼린저밴드 (BB1): 20 periods, 2 standard deviations
    2. 보조 볼린저밴드 (BB2): 4 periods, 4 standard deviations
    3. 상대강도지수 (RSI): 14 periods
    4. 평균 실제 범위 (ATR): 14 periods (손절매 계산용)

- 진입 조건 (Entry):
    - 일반 신호 (1x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1을 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 15분봉이 완성되어 반전을 '확증'
        3. (AI Confirm) AI가 시장 상황을 분석하여 최종 진입을 '동의'하면 기본 투자금으로 진입
    - 강력 신호 (2x 투자):
        1. (Detect) '실시간 가격'이 1시간봉 BB1 '그리고' BB2를 '모두' 터치 & 1시간봉 RSI 조건 만족
        2. (Confirm) 다음 15분봉이 완성되어 반전을 '확증'
        3. (AI Confirm) AI가 시장 상황을 분석하여 최종 진입을 '동의'하면 '2배'의 투자금으로 진입

- 청산 조건 (Exit):
    - 분할 익절(Partial Take Profit):
        1. (1차 익절) '1시간봉' 기준 BB1 중앙선(20MA)에 1% 근접 시, 보유 물량의 50% 익절
    - 트레일링 스탑 (Trailing Stop for Remaining Position):
        1. 1차 익절 후, 손절 라인을 최소 본절(진입가)로 상향 조정.
        2. 이후 가격이 수익 방향으로 움직이면, 손절 라인을 '직전 3개 1시간봉 캔들의 저점/고점'을 따라 계속 상향/하향 조정하며 수익을 극대화.
    - 초기 손절(Initial Stop Loss): 진입 시점의 '1시간봉 ATR' 값에 기반하여 동적으로 손절 라인 설정. (시장 변동성에 연동)

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
import google.generativeai as genai

# .env 파일 로드
load_dotenv()

# ===== Gemini AI 설정 =====
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini AI가 성공적으로 설정되었습니다.")
except Exception as e:
    print(f"Gemini AI 설정 오류: {e}")

# ===== 동시 포지션 제한 설정 =====
MAX_CONCURRENT_POSITIONS = 5

# ===== 전략 설정 (ATR 추가) =====
STRATEGY_CONFIG = {
    "MARGIN_SIZE": 0.20, "LEVERAGE": 10,
    "BB1_WINDOW": 20, "BB1_STD_DEV": 2,
    "BB2_WINDOW": 4, "BB2_STD_DEV": 4,
    "RSI_PERIOD": 14, "RSI_OVERBOUGHT": 70, "RSI_OVERSOLD": 30,
    "ATR_PERIOD": 14, "ATR_MULTIPLIER": 2.0, # ATR 기반 손절 설정
    "TP_BUFFER": 0.01,
    "MAX_DATA_DELAY_MINUTES": 120
}

# ===== API 및 DB 설정 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key, 'secret': secret, 'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
DB_FILE = "multi_coin_daytrading.db"

# ===== 상태 관리 변수 (position_states로 통합) =====
pending_confirmation = {}
fixed_investment_per_trade = 0
position_states = {} # 1차 익절 여부, 트레일링 스탑 가격, SL 주문 ID 등을 관리

# ===== 데이터베이스 함수 (수정됨) =====
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, coin_symbol TEXT NOT NULL,
        action TEXT NOT NULL, entry_price REAL NOT NULL, amount REAL NOT NULL, leverage INTEGER NOT NULL,
        current_sl_price REAL, -- 현재 손절가 저장 컬럼 추가
        status TEXT DEFAULT 'OPEN', exit_price REAL, profit_loss REAL, exit_reason TEXT, exit_timestamp TEXT
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, coin_symbol TEXT NOT NULL,
        direction TEXT NOT NULL, ai_decision TEXT NOT NULL, reasoning TEXT, related_trade_id INTEGER
    )''')
    conn.commit(); conn.close()
    print(f"Trading DB '{DB_FILE}' 설정 완료 (AI 분석 테이블 포함)")

def save_trade_to_db(trade_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, coin_symbol, action, entry_price, amount, leverage, current_sl_price, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (datetime.now().isoformat(), trade_data['coin_symbol'], trade_data['action'],
          trade_data['entry_price'], trade_data['amount'], trade_data['leverage'], trade_data['current_sl_price']))
    trade_id = cursor.lastrowid
    conn.commit(); conn.close()
    return trade_id

def update_sl_price_in_db(trade_id, new_sl_price):
    """DB에 저장된 손절 가격을 업데이트하는 함수"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET current_sl_price = ? WHERE id = ?", (new_sl_price, trade_id))
    conn.commit(); conn.close()

def get_open_trade_from_db(coin_symbol):
    """코인 심볼로 DB에서 오픈된 거래 정보를 가져오는 함수"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Use a try-except block to handle cases where the table or columns might not exist yet
    try:
        cursor.execute("SELECT * FROM trades WHERE coin_symbol = ? AND status = 'OPEN' ORDER BY timestamp DESC LIMIT 1", (coin_symbol,))
        trade = cursor.fetchone()
        if trade:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, trade))
    except sqlite3.OperationalError as e:
        print(f"DB Error in get_open_trade_from_db: {e}")
        return None
    finally:
        conn.close()
    return None

def close_trade_in_db(coin_symbol, exit_price, pnl, reason):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE trades SET status = 'CLOSED', exit_price = ?, profit_loss = ?, exit_reason = ?, exit_timestamp = ?
    WHERE coin_symbol = ? AND status = 'OPEN'
    ''', (exit_price, pnl, reason, datetime.now().isoformat(), coin_symbol))
    conn.commit(); conn.close()

def save_ai_analysis_to_db(analysis_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO ai_analysis (timestamp, coin_symbol, direction, ai_decision, reasoning, related_trade_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), analysis_data['coin_symbol'], analysis_data['direction'],
          analysis_data['ai_decision'], analysis_data['reasoning'], analysis_data.get('related_trade_id')))
    conn.commit(); conn.close()

def fetch_and_analyze_data(symbol, timeframe, cfg, calculate_bb=False, calculate_rsi=False, calculate_macd=False, calculate_atr=False):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        if not ohlcv: return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if len(df) < cfg['BB1_WINDOW']: return None
        if calculate_bb:
            bb1 = df.ta.bbands(length=cfg['BB1_WINDOW'], std=cfg['BB1_STD_DEV'])
            df[f'bb1_upper'] = bb1[f'BBU_{cfg["BB1_WINDOW"]}_{cfg["BB1_STD_DEV"]:.1f}']
            df[f'bb1_middle'] = bb1[f'BBM_{cfg["BB1_WINDOW"]}_{cfg["BB1_STD_DEV"]:.1f}']
            df[f'bb1_lower'] = bb1[f'BBL_{cfg["BB1_WINDOW"]}_{cfg["BB1_STD_DEV"]:.1f}']
            bb2 = df.ta.bbands(length=cfg['BB2_WINDOW'], std=cfg['BB2_STD_DEV'])
            df[f'bb2_upper'] = bb2[f'BBU_{cfg["BB2_WINDOW"]}_{cfg["BB2_STD_DEV"]:.1f}']
            df[f'bb2_lower'] = bb2[f'BBL_{cfg["BB2_WINDOW"]}_{cfg["BB2_STD_DEV"]:.1f}']
        if calculate_rsi: df.ta.rsi(length=cfg['RSI_PERIOD'], append=True)
        if calculate_macd: df.ta.macd(append=True)
        if calculate_atr: df.ta.atr(length=cfg['ATR_PERIOD'], append=True)
        df.dropna(inplace=True)
        return df if not df.empty else None
    except Exception: return None

def get_ai_confirmation(coin_symbol, direction, df_1h, df_15m):
    analysis_data = {'coin_symbol': coin_symbol, 'direction': direction, 'ai_decision': '보류', 'reasoning': ''}
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        current_price = df_15m.iloc[-1]['close']
        cols_1h = [col for col in ['open', 'high', 'low', 'close', 'volume', f'RSI_{STRATEGY_CONFIG["RSI_PERIOD"]}', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if col in df_1h.columns]
        cols_15m = [col for col in ['open', 'high', 'low', 'close', 'volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if col in df_15m.columns]
        recent_1h_data = df_1h[cols_1h].tail(5).to_string()
        recent_15m_data = df_15m[cols_15m].tail(10).to_string()
        latest_15m_macd = df_15m.iloc[-1]
        macd_info = f"- 15분봉 MACD: {latest_15m_macd.get('MACD_12_26_9', 'N/A'):.4f}\n- 15분봉 MACD 시그널: {latest_15m_macd.get('MACDs_12_26_9', 'N/A'):.4f}\n- 15분봉 MACD 히스토그램: {latest_15m_macd.get('MACDh_12_26_9', 'N/A'):.4f}"
        prompt = f"""당신은 변동성이 큰 암호화폐 선물 시장의 단기 트레이딩을 전문으로 하는 AI 분석가입니다. 당신의 임무는 기술적 분석 신호의 함정 가능성을 판단하여 자산을 보호하는 것입니다.\n\n[분석 요청]\n- 코인: {coin_symbol}\n- 포지션 방향: {direction}\n- 현재 가격: ${current_price:.4f}\n\n[상황]\n1시간봉 차트에서 볼린저밴드와 RSI를 기반으로 한 과매수/과매도 반전 신호가 감지되었습니다. 이후 15분봉 차트에서 기술적 반전 캔들이 확인되어 1차 진입 조건이 충족되었습니다. 이제 최종 진입 여부를 결정하기 위해 당신의 종합적인 분석이 필요합니다.\n\n[분석 데이터]\n- 최신 15분봉 MACD 지표:\n{macd_info}\n\n- 최근 1시간봉 데이터:\n{recent_1h_data}\n\n- 최근 15분봉 데이터:\n{recent_15m_data}\n\n[지시사항]\n위 데이터를 바탕으로 지금 {direction} 포지션에 진입하는 것이 타당한지 최종 판단을 내려주세요. 특히 최근 가격 움직임, 거래량 변화, 캔들 패턴, 그리고 **MACD 지표(시그널선 교차, 히스토그램의 추세)를 종합적으로 고려하여** 이것이 진짜 반전 신호인지, 아니면 단기적인 속임수(Fakeout)일 가능성이 높은지 분석해주세요.\n\n답변은 반드시 다음 형식 중 하나로 시작해야 합니다.\n"결론: [진입 동의]" 또는 "결론: [진입 보류]"\n\n그 뒤에 구체적인 분석 이유를 2~3줄로 요약하여 설명해주세요."""
        response = model.generate_content(prompt)
        analysis = response.text
        analysis_data['reasoning'] = analysis
        if "결론: [진입 동의]" in analysis:
            analysis_data['ai_decision'] = '동의'
            save_ai_analysis_to_db(analysis_data)
            return True, analysis
        else:
            analysis_data['ai_decision'] = '보류'
            save_ai_analysis_to_db(analysis_data)
            return False, analysis
    except Exception as e:
        error_msg = f"AI 분석 중 오류 발생: {e}"
        analysis_data['reasoning'] = error_msg
        save_ai_analysis_to_db(analysis_data)
        return False, error_msg

def get_top_volume_coins(exchange, limit=15):
    try:
        tickers = exchange.fetch_tickers()
        usdt_futures = {s: t for s, t in tickers.items() if s.endswith('/USDT:USDT')}
        sorted_tickers = sorted(usdt_futures.values(), key=lambda x: x.get('quoteVolume', 0), reverse=True)
        top_coins = {ticker['symbol'].split('/')[0]: {"symbol": ticker['symbol'].split(':')[0]} for ticker in sorted_tickers[:limit]}
        return top_coins
    except Exception as e: return {}

def get_open_positions_from_binance():
    try:
        positions = exchange.fetch_positions()
        return [{ "coin_symbol": pos['symbol'].split('/')[0], "action": 'long' if float(pos['info']['positionAmt']) > 0 else 'short', "entry_price": float(pos.get('entryPrice', 0)), "amount": float(pos.get('contracts', 0)), "leverage": int(pos['info'].get('leverage', 0)) } for pos in positions if float(pos['info'].get('positionAmt', 0)) != 0]
    except Exception as e: return []

# ===== 초기 신호 감지 함수 (수정됨) =====
def check_for_initial_signal(df_1h, current_price):
    if df_1h is None or len(df_1h) < 1: return None, "데이터 부족"
    cfg = STRATEGY_CONFIG; candle = df_1h.iloc[-1]; rsi_val = candle.get(f'RSI_{cfg["RSI_PERIOD"]}')
    if rsi_val is None: return None, "RSI 계산 불가"
    
    bb1_low, bb1_high = candle.get('bb1_lower'), candle.get('bb1_upper')
    bb2_low, bb2_high = candle.get('bb2_lower'), candle.get('bb2_upper')

    is_bb1_touch_low = current_price <= bb1_low if bb1_low else False
    is_bb2_touch_low = current_price <= bb2_low if bb2_low else False
    is_rsi_oversold = rsi_val < cfg['RSI_OVERSOLD']
    
    is_bb1_touch_high = current_price >= bb1_high if bb1_high else False
    is_bb2_touch_high = current_price >= bb2_high if bb2_high else False
    is_rsi_overbought = rsi_val > cfg['RSI_OVERBOUGHT']

    # 롱 신호 조건
    if is_bb1_touch_low and is_rsi_oversold:
        mult = 2.0 if is_bb2_touch_low else 1.0
        return {"direction": "long", "size_multiplier": mult}, f"🚨 초기 롱 신호({'강력' if mult==2.0 else '일반'}) 감지"
    
    # 숏 신호 조건
    if is_bb1_touch_high and is_rsi_overbought:
        mult = 2.0 if is_bb2_touch_high else 1.0
        return {"direction": "short", "size_multiplier": mult}, f"🚨 초기 숏 신호({'강력' if mult==2.0 else '일반'}) 감지"

    # 신호 없음 상세 로그 반환
    bb_status = 'O' if is_bb1_touch_low or is_bb1_touch_high else 'X'
    rsi_status = 'O' if is_rsi_oversold or is_rsi_overbought else 'X'
    return None, f"BB:{bb_status}, RSI:{rsi_status}"

def execute_trade(coin_name, signal, base_investment, multiplier):
    global position_states
    symbol = f"{coin_name}/USDT"; cfg = STRATEGY_CONFIG; leverage = cfg['LEVERAGE']
    final_investment = base_investment * multiplier
    
    try:
        df_1h = fetch_and_analyze_data(symbol, '1h', cfg, calculate_atr=True)
        if df_1h is None or df_1h.empty:
            print(f"❌ {coin_name}의 ATR 데이터 조회 실패. 거래를 취소합니다.")
            return False
            
        atr_value = df_1h.iloc[-1][f'ATRr_{cfg["ATR_PERIOD"]}']
        current_price = exchange.fetch_ticker(symbol)['last']
        amount = (final_investment * leverage) / current_price
        order_side = 'buy' if signal['direction'] == 'long' else 'sell'

        exchange.set_leverage(leverage, symbol)
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order.get('price') else current_price
        
        sl_distance = atr_value * cfg['ATR_MULTIPLIER']
        sl_price = entry_price - sl_distance if signal['direction'] == 'long' else entry_price + sl_distance
        
        stop_loss_order = exchange.create_order(symbol, 'STOP_MARKET', 'sell' if signal['direction'] == 'long' else 'buy', amount, None, {'stopPrice': sl_price, 'reduceOnly': True})
        
        trade_data = {'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price, 'amount': amount, 'leverage': leverage, 'current_sl_price': sl_price}
        trade_id = save_trade_to_db(trade_data)

        position_states[coin_name] = {'partial_tp_hit': False, 'trailing_stop_price': sl_price, 'sl_order_id': stop_loss_order['id']}
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE ai_analysis SET related_trade_id = ? WHERE coin_symbol = ? AND ai_decision = '동의' ORDER BY timestamp DESC LIMIT 1", (trade_id, coin_name))
        conn.commit(); conn.close()
        
        print(f"✅ 포지션 진입 성공: {coin_name} @ ${entry_price:,.4f}")
        print(f"   - 손절매(ATR 기반) 설정 완료: ${sl_price:,.4f} (거리: ${sl_distance:,.4f})")
        return True
    except Exception as e:
        print(f"❌ 거래 실행 오류: {e}"); return False

def manage_open_positions(open_positions):
    global position_states
    if not open_positions: return
    
    for pos in open_positions:
        coin = pos['coin_symbol']
        symbol = f"{coin}/USDT"
        state = position_states.get(coin, {})

        try:
            db_trade = get_open_trade_from_db(coin)
            if not db_trade: continue
                
            df_1h = fetch_and_analyze_data(symbol, '1h', STRATEGY_CONFIG, calculate_bb=True)
            if df_1h is None or len(df_1h) < 4: continue
            
            current_price = exchange.fetch_ticker(symbol)['last']
            
            if not state.get('partial_tp_hit'):
                middle_band = df_1h.iloc[-1]['bb1_middle']
                target = middle_band * (1 - STRATEGY_CONFIG['TP_BUFFER']) if pos['action'] == 'long' else middle_band * (1 + STRATEGY_CONFIG['TP_BUFFER'])

                if (pos['action'] == 'long' and current_price >= target) or (pos['action'] == 'short' and current_price <= target):
                    print(f"🚨 1차 익절 신호: {coin} @ ${current_price:,.4f}")
                    if state.get('sl_order_id'):
                        try: exchange.cancel_order(state['sl_order_id'], symbol)
                        except Exception: pass
                    
                    amount_to_close = pos['amount'] / 2
                    exchange.create_market_order(symbol, 'sell' if pos['action'] == 'long' else 'buy', amount_to_close, params={'reduceOnly': True})
                    
                    trailing_sl_price = pos['entry_price']
                    remaining_amount = pos['amount'] - amount_to_close
                    sl_order = exchange.create_order(symbol, 'sell' if pos['action'] == 'long' else 'buy', remaining_amount, None, {'stopPrice': trailing_sl_price, 'reduceOnly': True})
                    
                    position_states[coin] = {'partial_tp_hit': True, 'trailing_stop_price': trailing_sl_price, 'sl_order_id': sl_order['id']}
                    update_sl_price_in_db(db_trade['id'], trailing_sl_price)
                    print(f"   - ✅ 1차 익절 완료, 트레일링 스탑 시작. (손실 방지선: ${trailing_sl_price:,.4f})")
                    time.sleep(3)
                    continue
            else: # 트레일링 스탑 관리
                current_sl_price = state.get('trailing_stop_price')
                new_sl_candidate = df_1h.iloc[-4:-1]['low'].min() if pos['action'] == 'long' else df_1h.iloc[-4:-1]['high'].max()
                
                should_update = (pos['action'] == 'long' and new_sl_candidate > current_sl_price) or \
                                (pos['action'] == 'short' and new_sl_candidate < current_sl_price)
                
                if should_update:
                    print(f"📈 트레일링 스탑 조정: {coin} ${current_sl_price:,.4f} -> ${new_sl_candidate:,.4f}")
                    try: exchange.cancel_order(state.get('sl_order_id'), symbol)
                    except Exception: pass
                    
                    remaining_amount = pos['amount'] / 2
                    sl_order = exchange.create_order(symbol, 'sell' if pos['action'] == 'long' else 'buy', remaining_amount, None, {'stopPrice': new_sl_candidate, 'reduceOnly': True})
                    
                    position_states[coin]['trailing_stop_price'] = new_sl_candidate
                    position_states[coin]['sl_order_id'] = sl_order['id']
                    update_sl_price_in_db(db_trade['id'], new_sl_candidate)

        except Exception as e:
            print(f"포지션 관리 오류 ({coin}): {e}")

# ===== 메인 루프 (수정됨) =====
def main():
    global fixed_investment_per_trade, position_states
    print("\n=== Dual BB Strategy Bot (v6.2 - Enhanced Logging) Started ===")
    setup_database()
    while True:
        try:
            now = datetime.now()
            print(f"\n\n\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            coins_to_scan = get_top_volume_coins(exchange, limit=15)
            open_positions = get_open_positions_from_binance()
            
            open_position_symbols = {p['coin_symbol'] for p in open_positions}
            for coin in list(position_states.keys()):
                if coin not in open_position_symbols:
                    del position_states[coin]
            
            manage_open_positions(open_positions)
            
            if not get_open_positions_from_binance() and not pending_confirmation:
                if fixed_investment_per_trade != 0: fixed_investment_per_trade = 0

            confirmed_coins = []
            if pending_confirmation:
                print("\n--- 확증 확인 ---")
            for coin, data in list(pending_confirmation.items()):
                if now >= data['confirmation_timestamp']:
                    log_prefix = f"-> {coin} ({data['direction']})"
                    symbol = f"{coin}/USDT"
                    df_15m = fetch_and_analyze_data(symbol, '15m', STRATEGY_CONFIG, calculate_macd=True)
                    if df_15m is None or len(df_15m) < 2:
                        confirmed_coins.append(coin); continue

                    last_15m_candle = df_15m.iloc[-2]
                    is_candle_confirmed = (data['direction'] == 'long' and last_15m_candle['close'] > last_15m_candle['open']) or \
                                          (data['direction'] == 'short' and last_15m_candle['close'] < last_15m_candle['open'])
                    
                    log_prefix += f" | 15분봉: {'✅' if is_candle_confirmed else '❌'}"

                    if is_candle_confirmed:
                        log_prefix += " | AI: ⏳"
                        print(log_prefix, end='\r') # AI 분석 중임을 표시
                        df_1h = fetch_and_analyze_data(symbol, '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True, calculate_macd=True)
                        if df_1h is not None:
                            is_ai_confirmed, reason = get_ai_confirmation(coin, data['direction'], df_1h, df_15m)
                            reason_summary = reason.split('결론: [')[-1].split(']')[-1].strip().split('\n')[0]
                            
                            if is_ai_confirmed:
                                log_prefix = f"-> {coin} ({data['direction']}) | 15분봉: ✅ | AI: ✅ 동의 ({reason_summary})"
                                print(log_prefix)
                                execute_trade(coin, data, data['investment_amount'], data.get('size_multiplier', 1.0))
                            else:
                                log_prefix = f"-> {coin} ({data['direction']}) | 15분봉: ✅ | AI: ❌ 보류 ({reason_summary})"
                                print(log_prefix)
                        else:
                            log_prefix += " | AI: ❌ (데이터 로드 실패)"
                            print(log_prefix)
                    else:
                        print(log_prefix) # 15분봉 실패 로그 출력
                    
                    confirmed_coins.append(coin)

            for coin in confirmed_coins:
                if coin in pending_confirmation: del pending_confirmation[coin]
            
            if fixed_investment_per_trade == 0:
                balance = exchange.fetch_balance()['USDT']
                total_capital = balance['total']
                fixed_investment_per_trade = total_capital * STRATEGY_CONFIG['MARGIN_SIZE']
                print(f"\n포지션당 기본 투자금(증거금) 설정: ${fixed_investment_per_trade:.2f}")

            occupied_slots = len(get_open_positions_from_binance()) + len(pending_confirmation)
            if occupied_slots < MAX_CONCURRENT_POSITIONS:
                print(f"\n--- 신규 진입 탐색 ---")
                for coin, config in coins_to_scan.items():
                    if len(get_open_positions_from_binance()) + len(pending_confirmation) >= MAX_CONCURRENT_POSITIONS: break
                    if coin in [p['coin_symbol'] for p in get_open_positions_from_binance()] or coin in pending_confirmation: continue
                    
                    try:
                        current_price = exchange.fetch_ticker(config['symbol'])['last']
                        df_1h = fetch_and_analyze_data(config['symbol'], '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True)
                        if df_1h is None: 
                            print(f"-> {coin} 스캔 중... (데이터 부족)")
                            continue
                            
                        initial_signal, reason = check_for_initial_signal(df_1h, current_price)

                        if initial_signal:
                            print(f"-> {coin} 스캔 중... {reason}")
                            minutes_to_next_15m = 15 - (now.minute % 15)
                            confirm_time = now.replace(second=5, microsecond=0) + timedelta(minutes=minutes_to_next_15m)
                            initial_signal['confirmation_timestamp'] = confirm_time
                            initial_signal['investment_amount'] = fixed_investment_per_trade
                            pending_confirmation[coin] = initial_signal
                            print(f"     ㄴ 확증 대기열 추가 (예정 시각: {confirm_time.strftime('%H:%M:%S')})")
                        else:
                            print(f"-> {coin} 스캔 중... ({reason})")

                    except Exception as e:
                        print(f"   - 스캔 중 오류 발생 ({coin}): {e}")

            print(f"\n--- 사이클 완료. 60초 후 다시 시작합니다. ---")
            time.sleep(60)
        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}"); time.sleep(60)

if __name__ == "__main__":
    main()

