"""
Dual Bollinger Band Strategy Bot (v8.5 - Robust State Management)
----------------------------------------------------------------
전략:
- 안정성과 거래 빈도를 모두 잡기 위한 '투트랙' 시스템 동시 운영.
- 각 트랙의 진입 타임프레임에 맞춰 청산(TP/SL) 타임프레임도 동적으로 변경.
- AI 모델 과부하 시, 지능적인 재시도 로직(Exponential Backoff)을 통해 안정성 강화.
- 확증 로직을 신호가 발생한 캔들의 '다음' 캔들을 확인하도록 수정하여 신뢰도 향상.
- 거래소 API의 일시적인 오류로 비정상적인 잔고가 조회될 경우, 거래를 중단하는 안전장치 추가.
- [개선] 포지션 상태와 관계없이 투자금 재설정 로직이 안정적으로 작동하도록 수정.

- Track 1: 안정적인 추세 반전 전략
    - 신호: 1시간봉 BB 터치 + 1시간봉 RSI 과매수/과매도
    - 확증: '다음' 15분봉 캔들 마감 + AI 최종 승인
    - 청산: '1시간봉' 기준 (초기 SL, 분할 TP, 트레일링 스탑)

- Track 2: 빠른 단기 변동성 전략
    - 신호: 15분봉 BB 터치 + 15분봉 RSI 과매수/과매도
    - 확증: '다음' 5분봉 캔들 마감 + AI 최종 승인
    - 청산: '15분봉' 기준 (초기 SL, 분할 TP, 트레일링 스탑)
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
import google.generativeai as genai
import random

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

# ===== 전략 설정 =====
STRATEGY_CONFIG = {
    "MARGIN_SIZE": 0.20, "LEVERAGE": 10,
    "BB1_WINDOW": 20, "BB1_STD_DEV": 2.0,
    "BB2_WINDOW": 4, "BB2_STD_DEV": 4.0,
    "RSI_PERIOD": 14, "RSI_OVERBOUGHT": 70, "RSI_OVERSOLD": 30,
    "ATR_PERIOD": 14, "ATR_MULTIPLIER": 2.0,
    "TP_BUFFER": 0.01,
    "MAX_LOSS_PERCENTAGE": 0.50,
    "MIN_CAPITAL_THRESHOLD": 1.0, 
}

# ===== API 및 DB 설정 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key, 'secret': secret, 'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
DB_FILE = "multi_coin_daytrading.db"

# ===== 상태 관리 변수 =====
pending_confirmation = {}
fixed_investment_per_trade = 0
position_states = {}

# ===== 데이터베이스 함수 =====
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, coin_symbol TEXT NOT NULL,
        action TEXT NOT NULL, entry_price REAL NOT NULL, amount REAL NOT NULL, leverage INTEGER NOT NULL,
        current_sl_price REAL, status TEXT DEFAULT 'OPEN', exit_price REAL, profit_loss REAL,
        exit_reason TEXT, exit_timestamp TEXT, partial_tp_hit INTEGER DEFAULT 0, entry_track TEXT
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
    INSERT INTO trades (timestamp, coin_symbol, action, entry_price, amount, leverage, current_sl_price, status, entry_track)
    VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    ''', (datetime.now().isoformat(), trade_data['coin_symbol'], trade_data['action'],
          trade_data['entry_price'], trade_data['amount'], trade_data['leverage'], 
          trade_data['current_sl_price'], trade_data['entry_track']))
    trade_id = cursor.lastrowid
    conn.commit(); conn.close()
    return trade_id

def update_db(query, params=()):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def get_all_open_trades_from_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
        trades = cursor.fetchall()
        return [dict(trade) for trade in trades]
    except Exception as e:
        print(f"DB Error in get_all_open_trades_from_db: {e}")
        return []
    finally:
        conn.close()

def close_trade_in_db(trade_id, exit_price, pnl, reason):
    update_db('''
        UPDATE trades SET status = 'CLOSED', exit_price = ?, profit_loss = ?, exit_reason = ?, exit_timestamp = ?
        WHERE id = ?
    ''', (exit_price, pnl, reason, datetime.now().isoformat(), trade_id))
    print(f"   - DB 업데이트 완료: Trade ID {trade_id} 포지션 CLOSED 처리.")

def save_ai_analysis_to_db(analysis_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO ai_analysis (timestamp, coin_symbol, direction, ai_decision, reasoning, related_trade_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), analysis_data['coin_symbol'], analysis_data['direction'],
          analysis_data['ai_decision'], analysis_data['reasoning'], analysis_data.get('related_trade_id')))
    conn.commit(); conn.close()

# ===== 데이터 분석 함수 =====
def fetch_and_analyze_data(symbol, timeframe, cfg, calculate_bb=False, calculate_rsi=False, calculate_macd=False, calculate_atr=False):
    try:
        limit = 150 if timeframe in ['5m', '15m'] else 100
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv: return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if len(df) < cfg['BB1_WINDOW']: return None

        if calculate_bb:
            bb1 = df.ta.bbands(length=cfg['BB1_WINDOW'], std=cfg['BB1_STD_DEV'])
            bbu1_col = next((col for col in bb1.columns if col.startswith('BBU_')), None)
            bbm1_col = next((col for col in bb1.columns if col.startswith('BBM_')), None)
            bbl1_col = next((col for col in bb1.columns if col.startswith('BBL_')), None)

            bb2 = df.ta.bbands(length=cfg['BB2_WINDOW'], std=cfg['BB2_STD_DEV'])
            bbu2_col = next((col for col in bb2.columns if col.startswith('BBU_')), None)
            bbl2_col = next((col for col in bb2.columns if col.startswith('BBL_')), None)

            if not all([bbu1_col, bbm1_col, bbl1_col, bbu2_col, bbl2_col]): return None

            df['bb1_upper'] = bb1[bbu1_col]
            df['bb1_middle'] = bb1[bbm1_col]
            df['bb1_lower'] = bb1[bbl1_col]
            df['bb2_upper'] = bb2[bbu2_col]
            df['bb2_lower'] = bb2[bbl2_col]

        if calculate_rsi: df.ta.rsi(length=cfg['RSI_PERIOD'], append=True)
        if calculate_macd: df.ta.macd(append=True)
        if calculate_atr: df.ta.atr(length=cfg['ATR_PERIOD'], append=True)

        df.dropna(inplace=True)
        return df if not df.empty else None
    except Exception as e:
        return None

# ===== AI 및 거래량 함수 =====
def get_ai_confirmation(coin_symbol, direction, df_1h, df_15m):
    analysis_data = {'coin_symbol': coin_symbol, 'direction': direction, 'ai_decision': '보류', 'reasoning': ''}
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            current_price = df_15m.iloc[-1]['close']
            cols_1h = [col for col in ['open', 'high', 'low', 'close', 'volume', f'RSI_{STRATEGY_CONFIG["RSI_PERIOD"]}', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if col in df_1h.columns]
            cols_15m = [col for col in ['open', 'high', 'low', 'close', 'volume', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if col in df_15m.columns]
            recent_1h_data = df_1h[cols_1h].tail(5).to_string()
            recent_15m_data = df_15m[cols_15m].tail(10).to_string()
            latest_15m_macd = df_15m.iloc[-1]
            macd_info = f"- 15분봉 MACD: {latest_15m_macd.get('MACD_12_26_9', 'N/A'):.4f}\n- 15분봉 MACD 시그널: {latest_15m_macd.get('MACDs_12_26_9', 'N/A'):.4f}\n- 15분봉 MACD 히스토그램: {latest_15m_macd.get('MACDh_12_26_9', 'N/A'):.4f}"
            prompt = f"""당신은 변동성이 큰 암호화폐 선물 시장의 단기 트레이딩을 전문으로 하는 AI 분석가입니다. 당신의 임무는 기술적 분석 신호의 함정 가능성을 판단하여 자산을 보호하는 것입니다.\n\n[분석 요청]\n- 코인: {coin_symbol}\n- 포지션 방향: {direction}\n- 현재 가격: ${current_price:.4f}\n\n[상황]\n기술적 지표(BB, RSI)를 기반으로 한 단기 과매수/과매도 반전 신호가 감지되었습니다. 이후 더 짧은 타임프레임에서 기술적 반전 캔들이 확인되어 1차 진입 조건이 충족되었습니다. 이제 최종 진입 여부를 결정하기 위해 당신의 종합적인 분석이 필요합니다.\n\n[분석 데이터]\n- 최신 15분봉 MACD 지표:\n{macd_info}\n\n- 최근 1시간봉 데이터:\n{recent_1h_data}\n\n- 최근 15분봉 데이터:\n{recent_15m_data}\n\n[지시사항]\n위 데이터를 바탕으로 지금 {direction} 포지션에 진입하는 것이 타당한지 최종 판단을 내려주세요. 특히 최근 가격 움직임, 거래량 변화, 캔들 패턴, 그리고 **MACD 지표(시그널선 교차, 히스토그램의 추세)를 종합적으로 고려하여** 이것이 진짜 반전 신호인지, 아니면 단기적인 속임수(Fakeout)일 가능성이 높은지 분석해주세요.\n\n답변은 반드시 다음 형식 중 하나로 시작해야 합니다.\n"결론: [진입 동의]" 또는 "결론: [진입 보류]"\n\n그 뒤에 구체적인 분석 이유를 2~3줄로 요약하여 설명해주세요."""
            
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
            error_msg = str(e)
            if "overloaded" in error_msg.lower() or "503" in error_msg:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"   - AI 모델 과부하 감지. {wait_time:.1f}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                final_error_msg = f"AI 분석 중 예상치 못한 오류 발생: {e}"
                analysis_data['reasoning'] = final_error_msg
                save_ai_analysis_to_db(analysis_data)
                return False, final_error_msg
    
    final_error_msg = f"AI 분석 실패: 모델이 계속 과부하 상태입니다 (최대 재시도 {max_retries}회 초과)."
    analysis_data['reasoning'] = final_error_msg
    save_ai_analysis_to_db(analysis_data)
    return False, final_error_msg

def get_top_volume_coins(exchange, limit=15):
    try:
        tickers = exchange.fetch_tickers()
        usdt_futures = {s: t for s, t in tickers.items() if s.endswith('/USDT:USDT')}
        sorted_tickers = sorted(usdt_futures.values(), key=lambda x: x.get('quoteVolume', 0), reverse=True)
        top_coins = {ticker['symbol'].split('/')[0]: {"symbol": ticker['symbol'].split(':')[0]} for ticker in sorted_tickers[:limit]}
        return top_coins
    except Exception as e:
        print(f"거래대금 상위 코인 조회 오류: {e}")
        return {}

def get_open_positions_from_binance():
    try:
        positions = exchange.fetch_positions()
        return [{
            "coin_symbol": pos['symbol'].split('/')[0],
            "action": 'long' if float(pos['info']['positionAmt']) > 0 else 'short',
            "entry_price": float(pos.get('entryPrice', 0)),
            "amount": float(pos.get('contracts', 0)),
            "leverage": int(pos['info'].get('leverage', 0)),
            "pnl": float(pos.get('unrealizedPnl', 0))
        } for pos in positions if float(pos['info'].get('positionAmt', 0)) != 0]
    except Exception as e:
        print(f"바이낸스 포지션 조회 오류: {e}")
        return []

# ===== 신호 감지 및 거래 실행 함수 =====
def check_for_initial_signal(df_realtime, current_price):
    if df_realtime is None or len(df_realtime) < 1: return None, "데이터 부족"
    cfg = STRATEGY_CONFIG
    last_row = df_realtime.iloc[-1]
    rsi_col_name = f'RSI_{cfg["RSI_PERIOD"]}'
    if rsi_col_name not in last_row.index: return None, f"{rsi_col_name} 컬럼 없음"
    rsi_val = last_row[rsi_col_name]
    if pd.isna(rsi_val): return None, "RSI 계산 불가"

    bb1_low, bb1_high = last_row.get('bb1_lower'), last_row.get('bb1_upper')
    bb2_low, bb2_high = last_row.get('bb2_lower'), last_row.get('bb2_upper')

    is_bb1_touch_low = current_price <= bb1_low if bb1_low else False
    is_bb2_touch_low = current_price <= bb2_low if bb2_low else False
    is_rsi_oversold = rsi_val < cfg['RSI_OVERSOLD']

    is_bb1_touch_high = current_price >= bb1_high if bb1_high else False
    is_bb2_touch_high = current_price >= bb2_high if bb2_high else False
    is_rsi_overbought = rsi_val > cfg['RSI_OVERBOUGHT']

    if is_bb1_touch_low and is_rsi_oversold:
        mult = 2.0 if is_bb2_touch_low else 1.0
        return {"direction": "long", "size_multiplier": mult}, f"🚨 초기 롱 신호({'강력' if mult==2.0 else '일반'}) 감지"
    if is_bb1_touch_high and is_rsi_overbought:
        mult = 2.0 if is_bb2_touch_high else 1.0
        return {"direction": "short", "size_multiplier": mult}, f"🚨 초기 숏 신호({'강력' if mult==2.0 else '일반'}) 감지"

    bb_status = 'O' if is_bb1_touch_low or is_bb1_touch_high else 'X'
    return None, f"BB:{bb_status}, RSI:{rsi_val:.1f}"

def execute_trade(coin_name, signal, base_investment, multiplier):
    global position_states
    symbol = f"{coin_name}/USDT"; cfg = STRATEGY_CONFIG; leverage = cfg['LEVERAGE']
    final_investment = base_investment * multiplier
    trade_id = None
    
    try:
        track_name = signal.get('track', {}).get('name', 'Track 1 (1h)')
        atr_tf = '15m' if track_name == 'Track 2 (15m)' else '1h'

        df_for_atr = fetch_and_analyze_data(symbol, atr_tf, cfg, calculate_atr=True)
        if df_for_atr is None or df_for_atr.empty:
            print(f"❌ {coin_name}의 {atr_tf} ATR 데이터 조회 실패. 거래를 취소합니다.")
            return False
            
        current_price = exchange.fetch_ticker(symbol)['last']
        amount = (final_investment * leverage) / current_price
        order_side = 'buy' if signal['direction'] == 'long' else 'sell'

        exchange.set_leverage(leverage, symbol)
        
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order.get('price') else current_price

        trade_data_preliminary = {
            'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price,
            'amount': amount, 'leverage': leverage, 'current_sl_price': None, 'entry_track': track_name
        }
        trade_id = save_trade_to_db(trade_data_preliminary)
        print(f"✅ 포지션 진입 및 DB 기록 성공 (ID: {trade_id}): {coin_name} @ ${entry_price:,.4f}")

        try:
            atr_value = df_for_atr.iloc[-1][f'ATRr_{cfg["ATR_PERIOD"]}']
            sl_distance_atr = atr_value * cfg['ATR_MULTIPLIER']
            sl_price_atr = entry_price - sl_distance_atr if signal['direction'] == 'long' else entry_price + sl_distance_atr
            
            max_loss_usdt = final_investment * cfg['MAX_LOSS_PERCENTAGE']
            price_change_per_coin = max_loss_usdt / amount
            sl_price_max_loss = entry_price - price_change_per_coin if signal['direction'] == 'long' else entry_price + price_change_per_coin

            if signal['direction'] == 'long':
                final_sl_price = max(sl_price_atr, sl_price_max_loss)
                sl_reason = f"{atr_tf} ATR" if final_sl_price == sl_price_atr else "최대손실"
            else:
                final_sl_price = min(sl_price_atr, sl_price_max_loss)
                sl_reason = f"{atr_tf} ATR" if final_sl_price == sl_price_atr else "최대손실"
            
            stop_loss_order = exchange.create_order(symbol, 'STOP_MARKET', 'sell' if signal['direction'] == 'long' else 'buy', amount, None, {'stopPrice': final_sl_price, 'reduceOnly': True})
            
            update_db("UPDATE trades SET current_sl_price = ? WHERE id = ?", (final_sl_price, trade_id))
            position_states[coin_name] = {'sl_order_id': stop_loss_order['id']}
            print(f"   - 손절매({sl_reason} 기반) 설정 완료: ${final_sl_price:,.4f}")

        except Exception as sl_error:
            print(f"🚨🚨🚨 중요 경고: {coin_name} 포지션은 오픈되었으나 손절 주문 설정에 실패했습니다. 수동 관리가 필요합니다! 🚨🚨🚨")
            print(f"   - 오류: {sl_error}")

        update_db("UPDATE ai_analysis SET related_trade_id = ? WHERE coin_symbol = ? AND ai_decision = '동의' ORDER BY timestamp DESC LIMIT 1", (trade_id, coin_name))
        
        return True

    except Exception as e:
        print(f"❌ 거래 실행 중 치명적 오류 (진입 주문 실패): {e}")
        if trade_id:
            update_db("DELETE FROM trades WHERE id = ?", (trade_id,))
            print(f"   - 불완전한 거래 기록(ID: {trade_id})이 DB에서 삭제되었습니다.")
        return False

# ===== 포지션 관리 함수 =====
def manage_open_positions():
    global position_states
    open_trades_db = get_all_open_trades_from_db()
    if not open_trades_db: return

    print("\n--- 오픈 포지션 관리 ---")
    for trade in open_trades_db:
        coin = trade['coin_symbol']; symbol = f"{coin}/USDT"; trade_id = trade['id']
        state = position_states.get(coin, {})
        action = trade['action']
        track_name = trade.get('entry_track', 'Track 1 (1h)')

        exit_tf = '15m' if track_name == 'Track 2 (15m)' else '1h'

        try:
            df_exit = fetch_and_analyze_data(symbol, exit_tf, STRATEGY_CONFIG, calculate_bb=True)
            if df_exit is None or len(df_exit) < 4: continue
            
            current_price = exchange.fetch_ticker(symbol)['last']
            last_candle = df_exit.iloc[-1]

            if not trade['partial_tp_hit']:
                middle_band = last_candle['bb1_middle']
                target = middle_band * (1 - STRATEGY_CONFIG['TP_BUFFER']) if action == 'long' else middle_band * (1 + STRATEGY_CONFIG['TP_BUFFER'])
                
                print(f"-> {coin}({action}) [{exit_tf}] | 현재가: ${current_price:,.4f} | 1차 익절 목표: ${target:,.4f}")

                if (action == 'long' and current_price >= target) or \
                   (action == 'short' and current_price <= target):
                    
                    reason = f"1차 익절 ({exit_tf} BB 중앙선 근접)"
                    print(f"   🚨 {reason} 실행 - {coin} @ ${current_price:,.4f}")
                    
                    if state.get('sl_order_id'):
                        try: exchange.cancel_order(state['sl_order_id'], symbol)
                        except Exception as e: print(f"   - 기존 SL 주문 취소 실패: {e}")
                    
                    amount_to_close = trade['amount'] / 2
                    exchange.create_market_order(symbol, 'sell' if action == 'long' else 'buy', amount_to_close, params={'reduceOnly': True})
                    
                    trailing_sl_price = trade['entry_price']
                    remaining_amount = trade['amount'] - amount_to_close
                    sl_order = exchange.create_order(symbol, 'STOP_MARKET', 'sell' if action == 'long' else 'buy', remaining_amount, None, {'stopPrice': trailing_sl_price, 'reduceOnly': True})
                    
                    position_states[coin] = {'sl_order_id': sl_order['id']}
                    update_db("UPDATE trades SET partial_tp_hit = 1, current_sl_price = ? WHERE id = ?", (trailing_sl_price, trade_id))
                    print(f"   - ✅ 1차 익절 완료, 트레일링 스탑 시작 (손실 방지선: ${trailing_sl_price:,.4f})")
                    time.sleep(3)
                    continue
            else:
                current_sl_price = trade['current_sl_price']
                print(f"-> {coin}({action}) [{exit_tf}] | 현재가: ${current_price:,.4f} | 트레일링 스탑 관리 중... | 현재 SL: ${current_sl_price:,.4f}")

                new_sl_candidate = df_exit.iloc[-4:-1]['low'].min() if action == 'long' else df_exit.iloc[-4:-1]['high'].max()
                
                should_update = (action == 'long' and new_sl_candidate > current_sl_price) or \
                                (action == 'short' and new_sl_candidate < current_sl_price)
                
                if should_update:
                    print(f"   📈 트레일링 스탑 조정 ({exit_tf} 기준): {coin} ${current_sl_price:,.4f} -> ${new_sl_candidate:,.4f}")
                    try: 
                        if state.get('sl_order_id'):
                            exchange.cancel_order(state.get('sl_order_id'), symbol)
                    except Exception as e: print(f"   - 기존 SL 주문 취소 실패: {e}")
                    
                    pos_info = next((p for p in get_open_positions_from_binance() if p['coin_symbol'] == coin), None)
                    if pos_info:
                        sl_order = exchange.create_order(symbol, 'STOP_MARKET', 'sell' if action == 'long' else 'buy', pos_info['amount'], None, {'stopPrice': new_sl_candidate, 'reduceOnly': True})
                        position_states[coin]['sl_order_id'] = sl_order['id']
                        update_db("UPDATE trades SET current_sl_price = ? WHERE id = ?", (new_sl_candidate, trade_id))
        
        except Exception as e:
            print(f"포지션 관리 오류 ({coin}): {e}")

# ===== 메인 루프 (투자금 재설정 로직 수정) =====
def main():
    global fixed_investment_per_trade, position_states
    print(f"\n=== Dual BB Strategy Bot (v8.5 - Robust State Management) Started ===")
    setup_database()

    tracks = {
        '1h': {'confirmation_tf': '15m', 'name': 'Track 1 (1h)'},
        '15m': {'confirmation_tf': '5m', 'name': 'Track 2 (15m)'}
    }

    while True:
        try:
            now = datetime.now()
            print(f"\n\n\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] --- 새로운 사이클 시작 ---")
            
            api_positions = get_open_positions_from_binance()
            db_trades = get_all_open_trades_from_db()
            api_symbols = {p['coin_symbol'] for p in api_positions}

            for trade in db_trades:
                if trade['coin_symbol'] not in api_symbols:
                    print(f"   - ⚠️ 포지션 불일치 감지: {trade['coin_symbol']} DB에 있지만 API에 없음 (외부/수동 종료 간주).")
                    close_trade_in_db(trade['id'], 0, 0, "외부 요인에 의해 포지션 종료됨")
            
            open_symbols_db = {t['coin_symbol'] for t in get_all_open_trades_from_db()}
            for coin in list(position_states.keys()):
                if coin not in open_symbols_db:
                    del position_states[coin]

            manage_open_positions()

            # [핵심 수정] 투자금 재설정 조건을 단순화하여 안정성 확보
            if not api_positions:
                if fixed_investment_per_trade != 0:
                    print("   - 모든 포지션이 청산되었습니다. 다음 사이클에서 투자금을 재설정합니다.")
                    fixed_investment_per_trade = 0

            coins_to_scan = get_top_volume_coins(exchange)
            if not coins_to_scan: 
                print("거래대금 상위 코인 조회 실패. 60초 후 재시도.")
                time.sleep(60); continue

            # --- 확증 프로세스 ---
            if pending_confirmation:
                print("\n--- 확증 확인 ---")
            confirmed_coins = []
            for coin, data in list(pending_confirmation.items()):
                if now >= data['confirmation_timestamp']:
                    track_info = data['track']
                    confirmation_tf = track_info['confirmation_tf']
                    log_prefix = f"-> [{track_info['name']}] {coin} ({data['direction']})"
                    symbol = f"{coin}/USDT"

                    df_confirm = fetch_and_analyze_data(symbol, confirmation_tf, STRATEGY_CONFIG, calculate_macd=True)
                    if df_confirm is None or len(df_confirm) < 2:
                        confirmed_coins.append(coin); continue

                    last_candle = df_confirm.iloc[-2]
                    is_confirmed = (data['direction'] == 'long' and last_candle['close'] > last_candle['open']) or \
                                   (data['direction'] == 'short' and last_candle['close'] < last_candle['open'])
                    
                    log_prefix += f" | {confirmation_tf}봉: {'✅' if is_confirmed else '❌'}"

                    if is_confirmed:
                        log_prefix += " | AI: ⏳"
                        print(log_prefix, end='\r')
                        df_1h = fetch_and_analyze_data(symbol, '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True, calculate_macd=True)
                        df_15m = fetch_and_analyze_data(symbol, '15m', STRATEGY_CONFIG, calculate_macd=True)
                        if df_1h is not None and df_15m is not None:
                            ai_ok, reason = get_ai_confirmation(coin, data['direction'], df_1h, df_15m)
                            reason_summary = reason.split('결론: [')[-1].split(']')[-1].strip().split('\n')[0]
                            
                            if ai_ok:
                                print(f"{log_prefix.split(' | AI:')[0]} | AI: ✅ 동의 ({reason_summary})")
                                execute_trade(coin, data, data['investment_amount'], data.get('size_multiplier', 1.0))
                            else:
                                print(f"{log_prefix.split(' | AI:')[0]} | AI: ❌ 보류 ({reason_summary})")
                        else:
                            print(f"{log_prefix.split(' | AI:')[0]} | AI: ❌ (데이터 로드 실패)")
                    else:
                        print(log_prefix)
                    
                    confirmed_coins.append(coin)

            for coin in confirmed_coins:
                if coin in pending_confirmation: del pending_confirmation[coin]
            
            # --- 투자금 설정 ---
            if fixed_investment_per_trade == 0:
                try:
                    balance = exchange.fetch_balance()['USDT']
                    total_capital = balance['total']
                    
                    if total_capital < STRATEGY_CONFIG["MIN_CAPITAL_THRESHOLD"]:
                        print(f"⚠️ 경고: 조회된 총 자본금(${total_capital:.2f})이 비정상적으로 낮습니다. 투자금 계산을 건너뜁니다.")
                        time.sleep(60); continue

                    fixed_investment_per_trade = total_capital * STRATEGY_CONFIG['MARGIN_SIZE']
                    print(f"\n포지션당 기본 투자금(증거금) 설정: ${fixed_investment_per_trade:.2f}")
                except Exception as e:
                    print(f"잔고 조회 오류: {e}")
                    time.sleep(60); continue
            
            # --- 신규 진입 탐색 (투트랙) ---
            open_pos_count = len(get_all_open_trades_from_db())
            occupied_slots = open_pos_count + len(pending_confirmation)

            if occupied_slots < MAX_CONCURRENT_POSITIONS:
                print(f"\n--- 신규 진입 탐색 (투트랙 시스템) ---")
                current_open_symbols = {t['coin_symbol'] for t in get_all_open_trades_from_db()}

                for coin, config in coins_to_scan.items():
                    if len(get_all_open_trades_from_db()) + len(pending_confirmation) >= MAX_CONCURRENT_POSITIONS: break
                    if coin in current_open_symbols or coin in pending_confirmation: continue
                    
                    for signal_tf, track_info in tracks.items():
                        try:
                            df_base = fetch_and_analyze_data(config['symbol'], signal_tf, STRATEGY_CONFIG)
                            if df_base is None: continue
                            
                            current_price = exchange.fetch_ticker(config['symbol'])['last']
                            
                            df_realtime = df_base.copy()
                            df_realtime.iloc[-1, df_realtime.columns.get_loc('close')] = current_price
                            
                            df_realtime.ta.rsi(length=STRATEGY_CONFIG['RSI_PERIOD'], append=True)
                            bb1 = df_realtime.ta.bbands(length=STRATEGY_CONFIG['BB1_WINDOW'], std=STRATEGY_CONFIG['BB1_STD_DEV'])
                            bb2 = df_realtime.ta.bbands(length=STRATEGY_CONFIG['BB2_WINDOW'], std=STRATEGY_CONFIG['BB2_STD_DEV'])
                            if bb1 is None or bb2 is None: continue

                            bbu1 = next((c for c in bb1.columns if c.startswith('BBU_')),None)
                            bbm1 = next((c for c in bb1.columns if c.startswith('BBM_')),None)
                            bbl1 = next((c for c in bb1.columns if c.startswith('BBL_')),None)
                            bbu2 = next((c for c in bb2.columns if c.startswith('BBU_')),None)
                            bbl2 = next((c for c in bb2.columns if c.startswith('BBL_')),None)
                            if not all([bbu1, bbm1, bbl1, bbu2, bbl2]): continue

                            df_realtime['bb1_upper'] = bb1[bbu1]; df_realtime['bb1_middle'] = bb1[bbm1]; df_realtime['bb1_lower'] = bb1[bbl1]
                            df_realtime['bb2_upper'] = bb2[bbu2]; df_realtime['bb2_lower'] = bb2[bbl2]
                            df_realtime.dropna(inplace=True)
                            if df_realtime.empty: continue
                                
                            signal, reason = check_for_initial_signal(df_realtime, current_price)
                            print(f"-> [{signal_tf}] {coin} 스캔 중... ({reason})")

                            if signal:
                                confirm_tf_str = track_info['confirmation_tf']
                                confirm_mins = int(confirm_tf_str.replace('m', ''))
                                
                                minutes_to_current_interval_end = confirm_mins - (now.minute % confirm_mins)
                                total_minutes_to_wait = minutes_to_current_interval_end + confirm_mins
                                confirm_time = now.replace(second=5, microsecond=0) + timedelta(minutes=total_minutes_to_wait)

                                signal['confirmation_timestamp'] = confirm_time
                                signal['investment_amount'] = fixed_investment_per_trade
                                signal['track'] = track_info
                                pending_confirmation[coin] = signal
                                print(f"     ㄴ [{track_info['name']}] 확증 대기열 추가 (예정 시각: {confirm_time.strftime('%H:%M:%S')})")
                                
                                break 

                        except Exception as e:
                            print(f"   - 스캔 중 오류 발생 ({coin}, {signal_tf}): {e}")

            print(f"\n--- 사이클 완료. 60초 후 다시 시작합니다. ---")
            time.sleep(60)
        except Exception as e:
            print(f"메인 루프 심각한 오류: {e}"); time.sleep(60)

if __name__ == "__main__":
    main()

