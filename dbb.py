"""
Bollinger Band Trend Strategy Bot (v10.0 - Trend Following Exit)
----------------------------------------------------------------
전략:
- 단일 볼린저 밴드(20, 2.0)와 RSI를 활용한 안정적인 추세 반전 포착
- AI 모델 과부하 시, 지능적인 재시도 로직(Exponential Backoff)을 통해 안정성 강화
- 확증 로직을 신호가 발생한 캔들의 '다음' 캔들을 확인하도록 수정하여 신뢰도 향상
- [v10.0 변경사항]
  * BB2 관련 로직 전체 제거 및 코드 단순화
  * 익절 로직을 추세 추종형으로 변경:
    - 1차 TP: BB 중앙선 터치 → 50% 판매 + 손절가를 진입가로 상향
    - 2차 TP: BB 중앙선을 종가 이탈 시 → 나머지 50% 판매 (트레일링 스탑)
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
    "MARGIN_SIZE": 0.20,
    "LEVERAGE": 10,
    "BB_WINDOW": 20,
    "BB_STD_DEV": 2.0,
    "RSI_PERIOD": 14,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
    "ATR_PERIOD": 14,
    "ATR_MULTIPLIER": 2.0,
    "MAX_LOSS_PERCENTAGE": 0.80,
    "MIN_CAPITAL_THRESHOLD": 1.0,
    "AI_MAX_TOTAL_WAIT_SECONDS": 15,
    "SIGNAL_TIMEFRAME": "1h",
    "CONFIRMATION_TIMEFRAME": "15m",
    "EXIT_TIMEFRAME": "1h"
}

# ===== API 및 DB 설정 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key, 'secret': secret, 'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
DB_FILE = "trend_following_trading.db"

# ===== 상태 관리 변수 =====
pending_confirmation = {}
fixed_investment_per_trade = 0
position_states = {}

# ===== 로깅 함수 =====
def log_trade_failure(coin, reason, details=""):
    timestamp = datetime.now().isoformat()
    log_msg = f"[{timestamp}] 거래 실패: {coin} - {reason}"
    if details:
        log_msg += f"\n   상세: {details}"
    print(log_msg)
    try:
        with open("trade_failures.log", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    except:
        pass

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
    print(f"Trading DB '{DB_FILE}' 설정 완료")

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
    conn.commit(); conn.close()

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
        if len(df) < cfg['BB_WINDOW']: return None

        if calculate_bb:
            bb = df.ta.bbands(length=cfg['BB_WINDOW'], std=cfg['BB_STD_DEV'])
            bbu_col = next((col for col in bb.columns if col.startswith('BBU_')), None)
            bbm_col = next((col for col in bb.columns if col.startswith('BBM_')), None)
            bbl_col = next((col for col in bb.columns if col.startswith('BBL_')), None)

            if not all([bbu_col, bbm_col, bbl_col]): return None

            df['bb_upper'] = bb[bbu_col]
            df['bb_middle'] = bb[bbm_col]
            df['bb_lower'] = bb[bbl_col]

        if calculate_rsi: df.ta.rsi(length=cfg['RSI_PERIOD'], append=True)
        if calculate_macd: df.ta.macd(append=True)
        if calculate_atr: df.ta.atr(length=cfg['ATR_PERIOD'], append=True)

        df.dropna(inplace=True)
        return df if not df.empty else None
    except Exception as e:
        print(f"데이터 조회 오류 ({symbol}, {timeframe}): {e}")
        return None

# ===== AI 및 거래량 함수 =====
def get_ai_confirmation(coin_symbol, direction, df_1h, df_15m):
    analysis_data = {'coin_symbol': coin_symbol, 'direction': direction, 'ai_decision': '보류', 'reasoning': ''}
    max_retries = 3
    total_wait_time = 0
    max_total_wait = STRATEGY_CONFIG['AI_MAX_TOTAL_WAIT_SECONDS']
    
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
                wait_time = min((2 ** attempt) + random.uniform(0, 1), max_total_wait - total_wait_time)
                
                if total_wait_time + wait_time > max_total_wait:
                    final_error_msg = f"AI 분석 실패: 총 대기시간 초과 ({total_wait_time:.1f}초)"
                    analysis_data['reasoning'] = final_error_msg
                    save_ai_analysis_to_db(analysis_data)
                    return False, final_error_msg
                
                print(f"   - AI 모델 과부하. {wait_time:.1f}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                total_wait_time += wait_time
                continue
            else:
                final_error_msg = f"AI 분석 오류: {e}"
                analysis_data['reasoning'] = final_error_msg
                save_ai_analysis_to_db(analysis_data)
                return False, final_error_msg
    
    final_error_msg = f"AI 분석 실패: 최대 재시도 초과"
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
            "amount": abs(float(pos.get('contracts', 0))),
            "leverage": int(pos['info'].get('leverage', 0)),
            "pnl": float(pos.get('unrealizedPnl', 0))
        } for pos in positions if float(pos['info'].get('positionAmt', 0)) != 0]
    except Exception as e:
        print(f"바이낸스 포지션 조회 오류: {e}")
        return []

# ===== 신호 감지 및 거래 실행 함수 =====
def check_for_initial_signal(df_signal, cfg):
    if df_signal is None or len(df_signal) < 1: return None, "데이터 부족"
    
    last_row = df_signal.iloc[-1]
    rsi_col_name = f'RSI_{cfg["RSI_PERIOD"]}'
    if rsi_col_name not in last_row.index: return None, f"{rsi_col_name} 컬럼 없음"
    rsi_val = last_row[rsi_col_name]
    if pd.isna(rsi_val): return None, "RSI 계산 불가"

    bb_low, bb_high = last_row.get('bb_lower'), last_row.get('bb_upper')
    close_price = last_row['close']

    is_bb_touch_low = close_price <= bb_low if bb_low else False
    is_rsi_oversold = rsi_val < cfg['RSI_OVERSOLD']

    is_bb_touch_high = close_price >= bb_high if bb_high else False
    is_rsi_overbought = rsi_val > cfg['RSI_OVERBOUGHT']

    if is_bb_touch_low and is_rsi_oversold:
        return {"direction": "long"}, "롱 신호 감지"
    if is_bb_touch_high and is_rsi_overbought:
        return {"direction": "short"}, "숏 신호 감지"

    bb_status = 'O' if is_bb_touch_low or is_bb_touch_high else 'X'
    return None, f"BB:{bb_status}, RSI:{rsi_val:.1f}"

def set_leverage_safe(symbol, leverage):
    try:
        try:
            positions = exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol:
                    current_leverage = int(pos['info'].get('leverage', 0))
                    if current_leverage == leverage:
                        return True
        except:
            pass
        
        exchange.set_leverage(leverage, symbol)
        print(f"   - 레버리지 {leverage}x 설정 완료")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'no need to change' in error_msg or 'leverage not modified' in error_msg:
            return True
        else:
            log_trade_failure(symbol.split('/')[0], "레버리지 설정 실패", str(e))
            return False

def execute_trade(coin_name, signal, base_investment):
    global position_states
    symbol = f"{coin_name}/USDT"
    cfg = STRATEGY_CONFIG
    leverage = cfg['LEVERAGE']
    trade_id = None
    order = None
    
    print(f"\n{'='*60}")
    print(f"[거래 시작] {coin_name} {signal['direction'].upper()} | 투입금: ${base_investment:.2f}")
    print(f"{'='*60}")
    
    try:
        df_for_atr = fetch_and_analyze_data(symbol, cfg['EXIT_TIMEFRAME'], cfg, calculate_atr=True)
        if df_for_atr is None or df_for_atr.empty:
            log_trade_failure(coin_name, "ATR 데이터 조회 실패", f"Timeframe: {cfg['EXIT_TIMEFRAME']}")
            return False
        
        try:
            market_info = exchange.market(symbol)
            min_amount = market_info['limits']['amount']['min']
            min_cost = market_info['limits']['cost']['min']
            print(f"   - 시장 정보: 최소 수량={min_amount}, 최소 금액={min_cost}")
        except Exception as e:
            log_trade_failure(coin_name, "시장 정보 조회 실패", str(e))
            return False
        
        current_price = exchange.fetch_ticker(symbol)['last']
        amount = (base_investment * leverage) / current_price
        print(f"   - 현재가: ${current_price:.4f}")
        print(f"   - 계산된 수량: {amount:.8f} (레버리지 {leverage}x 적용)")
        
        if amount < min_amount:
            log_trade_failure(coin_name, "최소 수량 미달", f"주문수량={amount:.8f} < 최소수량={min_amount}")
            return False
        
        if amount * current_price < min_cost:
            log_trade_failure(coin_name, "최소 거래금액 미달", f"주문금액={amount * current_price:.2f} < 최소금액={min_cost}")
            return False

        order_side = 'buy' if signal['direction'] == 'long' else 'sell'

        if not set_leverage_safe(symbol, leverage): return False
        
        print(f"   - 주문 실행 중... ({order_side} {amount:.8f})")
        order = exchange.create_market_order(symbol, order_side, amount)
        entry_price = float(order['price']) if order.get('price') else current_price
        print(f"   - ✅ 주문 체결: ${entry_price:.4f}")

        trade_data_preliminary = {
            'coin_symbol': coin_name, 'action': signal['direction'], 'entry_price': entry_price,
            'amount': amount, 'leverage': leverage, 'current_sl_price': None, 'entry_track': 'Track 1 (1h)'
        }
        trade_id = save_trade_to_db(trade_data_preliminary)
        print(f"   - DB 기록 완료 (Trade ID: {trade_id})")

        try:
            atr_col = f'ATRr_{cfg["ATR_PERIOD"]}'
            if atr_col not in df_for_atr.columns: raise Exception(f"ATR 컬럼 없음: {atr_col}")
            atr_value = df_for_atr.iloc[-1][atr_col]
            sl_distance_atr = atr_value * cfg['ATR_MULTIPLIER']
            sl_price_atr = entry_price - sl_distance_atr if signal['direction'] == 'long' else entry_price + sl_distance_atr
            max_loss_usdt = base_investment * cfg['MAX_LOSS_PERCENTAGE']
            price_change_per_coin = max_loss_usdt / amount
            sl_price_max_loss = entry_price - price_change_per_coin if signal['direction'] == 'long' else entry_price + price_change_per_coin

            if signal['direction'] == 'long':
                final_sl_price = max(sl_price_atr, sl_price_max_loss)
                sl_reason = "ATR" if final_sl_price == sl_price_atr else "최대손실 80%"
            else:
                final_sl_price = min(sl_price_atr, sl_price_max_loss)
                sl_reason = "ATR" if final_sl_price == sl_price_atr else "최대손실 80%"
            
            print(f"   - 손절가 계산:\n     * ATR 기반: ${sl_price_atr:.4f}\n     * 80% 손실 기반: ${sl_price_max_loss:.4f}\n     * 최종 선택: ${final_sl_price:.4f} ({sl_reason})")
            
            stop_loss_order = exchange.create_order(
                symbol, 'STOP_MARKET', 'sell' if signal['direction'] == 'long' else 'buy', amount, None, {'stopPrice': final_sl_price, 'reduceOnly': True}
            )
            update_db("UPDATE trades SET current_sl_price = ? WHERE id = ?", (final_sl_price, trade_id))
            position_states[coin_name] = {'sl_order_id': stop_loss_order['id']}
            print(f"   - ✅ 손절매 설정 완료 (주문ID: {stop_loss_order['id']})")
        except Exception as sl_error:
            print(f"   - ❌ 손절 주문 설정 실패: {sl_error}")
            log_trade_failure(coin_name, "손절 주문 설정 실패", str(sl_error))
            try:
                print(f"   - 긴급 청산 시도...")
                emergency_close_side = 'sell' if signal['direction'] == 'long' else 'buy'
                emergency_order = exchange.create_market_order(symbol, emergency_close_side, amount, params={'reduceOnly': True})
                emergency_price = float(emergency_order.get('price')) if emergency_order.get('price') else current_price
                close_trade_in_db(trade_id, emergency_price, 0, "손절 주문 설정 실패로 긴급 청산")
                print(f"   - ✅ 긴급 청산 완료: ${emergency_price:.4f}")
                return False
            except Exception as close_error:
                print(f"   - ❌ 긴급 청산마저 실패! 수동 개입 필요!")
                log_trade_failure(coin_name, "긴급 청산 실패", str(close_error))
                close_trade_in_db(trade_id, entry_price, 0, "손절 실패 + 긴급청산 실패 (수동 확인 필요)")
                return False

        update_db("UPDATE ai_analysis SET related_trade_id = ? WHERE coin_symbol = ? AND ai_decision = '동의' ORDER BY timestamp DESC LIMIT 1", (trade_id, coin_name))
        
        print(f"{'='*60}\n[거래 완료] {coin_name} 포지션 오픈 성공\n{'='*60}\n")
        return True

    except Exception as e:
        print(f"   - ❌ 거래 실행 중 오류: {e}")
        log_trade_failure(coin_name, "거래 실행 오류", str(e))
        if order and trade_id:
            try:
                print(f"   - 불완전 포지션 청산 시도...")
                emergency_close_side = 'sell' if signal['direction'] == 'long' else 'buy'
                amount_to_close = abs(float(order.get('amount', amount)))
                exchange.create_market_order(symbol, emergency_close_side, amount_to_close, params={'reduceOnly': True})
                print(f"   - 불완전 포지션 청산 완료")
            except Exception as cleanup_error:
                log_trade_failure(coin_name, "불완전 포지션 청산 실패", str(cleanup_error))
        if trade_id:
            update_db("DELETE FROM trades WHERE id = ?", (trade_id,))
            print(f"   - 불완전한 거래 기록(ID: {trade_id}) 삭제")
        return False

# ===== 포지션 관리 함수 (추세 추종 익절 로직 적용) =====
def manage_open_positions():
    global position_states
    open_trades_db = get_all_open_trades_from_db()
    if not open_trades_db: return

    print("\n--- 오픈 포지션 관리 (추세 추종 로직 적용) ---")
    for trade in open_trades_db:
        coin = trade['coin_symbol']
        symbol = f"{coin}/USDT"
        trade_id = trade['id']
        state = position_states.get(coin, {})
        action = trade['action']

        try:
            df_exit = fetch_and_analyze_data(symbol, STRATEGY_CONFIG['EXIT_TIMEFRAME'], STRATEGY_CONFIG, calculate_bb=True)
            if df_exit is None or len(df_exit) < 2: continue
            
            current_price = exchange.fetch_ticker(symbol)['last']
            last_closed_candle = df_exit.iloc[-2] # 마감된 이전 캔들 사용
            
            # --- 1차 익절 로직 ---
            if not trade['partial_tp_hit']:
                target_tp1 = last_closed_candle['bb_middle']
                print(f"-> {coin}({action}) [1차 익절 대기] | 현재가: ${current_price:,.4f} | 목표(중앙선): ${target_tp1:,.4f}")

                if (action == 'long' and current_price >= target_tp1) or \
                   (action == 'short' and current_price <= target_tp1):
                    
                    print(f"   🎯 1차 익절 실행 (BB 중앙선 터치) - {coin}")
                    if state.get('sl_order_id'):
                        try: 
                            exchange.cancel_order(state['sl_order_id'], symbol)
                            print(f"   - 기존 손절 주문 취소")
                        except Exception as e: 
                            print(f"   - 손절 주문 취소 실패: {e}")
                    
                    amount_to_close = trade['amount'] / 2
                    remaining_amount = trade['amount'] - amount_to_close
                    
                    exchange.create_market_order(symbol, 'sell' if action == 'long' else 'buy', amount_to_close, params={'reduceOnly': True})
                    print(f"   - ✅ 50% 판매 완료 ({amount_to_close:.8f})")
                    
                    if remaining_amount > 0:
                        breakeven_price = trade['entry_price']
                        sl_order = exchange.create_order(
                            symbol, 'STOP_MARKET', 'sell' if action == 'long' else 'buy', remaining_amount, None, 
                            {'stopPrice': breakeven_price, 'reduceOnly': True}
                        )
                        position_states[coin] = {'sl_order_id': sl_order['id']}
                        update_db("UPDATE trades SET partial_tp_hit = 1, current_sl_price = ?, amount = ? WHERE id = ?", 
                                (breakeven_price, remaining_amount, trade_id))
                        print(f"   - ✅ 손절가 상향 완료: ${breakeven_price:,.4f} (진입가 = 손실 방지)")
                    else:
                        close_trade_in_db(trade_id, current_price, 0, "1차 익절 (전체 청산)")
                        if coin in position_states: del position_states[coin]
                    time.sleep(2)
                    continue
                    
            # --- 2차 익절 로직 (추세 추종) ---
            else:
                current_sl_price = trade['current_sl_price']
                trailing_stop_price = last_closed_candle['bb_middle']
                
                print(f"-> {coin}({action}) [추세 추종 중] | 현재가: ${current_price:,.4f} | 트레일링 스탑(중앙선): ${trailing_stop_price:,.4f} | 손절(본절): ${current_sl_price:,.4f}")
                
                is_trend_broken = (action == 'long' and last_closed_candle['close'] < trailing_stop_price) or \
                                  (action == 'short' and last_closed_candle['close'] > trailing_stop_price)

                if is_trend_broken:
                    print(f"   🎯 최종 익절 실행 (추세 이탈 감지: 종가 ${last_closed_candle['close']:.4f}) - {coin}")
                    if state.get('sl_order_id'):
                        try: exchange.cancel_order(state['sl_order_id'], symbol)
                        except Exception as e: print(f"   - 손절 주문 취소 실패: {e}")
                    
                    pos_info = next((p for p in get_open_positions_from_binance() if p['coin_symbol'] == coin), None)
                    if pos_info:
                        exchange.create_market_order(
                            symbol, 'sell' if action == 'long' else 'buy', pos_info['amount'], params={'reduceOnly': True}
                        )
                        close_trade_in_db(trade_id, current_price, 0, "최종 익절 (BB 중앙선 이탈)")
                        if coin in position_states: del position_states[coin]
                        print(f"   - ✅ 최종 익절 완료: ${current_price:,.4f}")
        
        except Exception as e:
            print(f"포지션 관리 오류 ({coin}): {e}")

# ===== 메인 루프 =====
def main():
    global fixed_investment_per_trade, position_states
    print(f"\n{'='*70}")
    print(f"  Bollinger Band Trend Strategy Bot v10.0")
    print(f"{'='*70}")
    print(f"  전략: 1시간봉 BB(20,2) + RSI 과매도/수 진입")
    print(f"  투입: 총 자산의 {STRATEGY_CONFIG['MARGIN_SIZE']*100}% 고정")
    print(f"  익절 전략: BB 중앙선(50%) → BB 중앙선 이탈(나머지 50%)")
    print(f"{'='*70}\n")
    setup_database()

    while True:
        try:
            now = datetime.now()
            print(f"\n{'='*70}\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 새로운 사이클 시작\n{'='*70}")
            
            api_positions = get_open_positions_from_binance()
            db_trades = get_all_open_trades_from_db()
            api_symbols = {p['coin_symbol'] for p in api_positions}
            api_positions_dict = {p['coin_symbol']: p for p in api_positions}

            for trade in db_trades:
                coin_sym = trade['coin_symbol']
                if coin_sym not in api_symbols:
                    print(f"   - ⚠️ 포지션 청산 감지: {coin_sym} (손절 또는 외부 종료)")
                    close_trade_in_db(trade['id'], trade.get('current_sl_price', trade['entry_price']), 0, "포지션 청산됨")
                else:
                    api_action = api_positions_dict[coin_sym]['action']
                    if trade['action'] != api_action:
                        print(f"   - ⚠️ 방향 불일치: {coin_sym} DB({trade['action']}) vs API({api_action})")
                        close_trade_in_db(trade['id'], api_positions_dict[coin_sym]['entry_price'], 0, "방향 불일치")
            
            open_symbols_db = {t['coin_symbol'] for t in get_all_open_trades_from_db()}
            for coin in list(position_states.keys()):
                if coin not in open_symbols_db:
                    del position_states[coin]

            manage_open_positions()

            if not api_positions and fixed_investment_per_trade != 0:
                print("   - 모든 포지션 청산됨. 투자금 재설정 예정")
                fixed_investment_per_trade = 0

            coins_to_scan = get_top_volume_coins(exchange)
            if not coins_to_scan: 
                print("거래대금 상위 코인 조회 실패. 60초 후 재시도")
                time.sleep(60); continue

            if pending_confirmation: print("\n--- 확증 확인 ---")
            confirmed_coins = []
            for coin, data in list(pending_confirmation.items()):
                if now >= data['confirmation_timestamp']:
                    confirmation_tf = STRATEGY_CONFIG['CONFIRMATION_TIMEFRAME']
                    log_prefix = f"-> {coin} ({data['direction']})"
                    symbol = f"{coin}/USDT"

                    df_confirm = fetch_and_analyze_data(symbol, confirmation_tf, STRATEGY_CONFIG, calculate_macd=True)
                    if df_confirm is None or len(df_confirm) < 2:
                        print(f"{log_prefix} | 확증 데이터 없음")
                        confirmed_coins.append(coin); continue
                    
                    last_candle = df_confirm.iloc[-2]
                    is_confirmed = (data['direction'] == 'long' and last_candle['close'] > last_candle['open']) or \
                                   (data['direction'] == 'short' and last_candle['close'] < last_candle['open'])
                    
                    log_prefix += f" | {confirmation_tf}봉: {'✅' if is_confirmed else '❌'}"

                    if is_confirmed:
                        log_prefix += " | AI: ⏳"; print(log_prefix, end='\r')
                        df_1h = fetch_and_analyze_data(symbol, '1h', STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True, calculate_macd=True)
                        df_15m = fetch_and_analyze_data(symbol, '15m', STRATEGY_CONFIG, calculate_macd=True)
                        if df_1h is not None and df_15m is not None:
                            ai_ok, reason = get_ai_confirmation(coin, data['direction'], df_1h, df_15m)
                            if ai_ok:
                                print(f"{log_prefix.split(' | AI:')[0]} | AI: ✅ 동의")
                                execute_trade(coin, data, data['investment_amount'])
                            else:
                                print(f"{log_prefix.split(' | AI:')[0]} | AI: ❌ 보류")
                        else:
                            print(f"{log_prefix.split(' | AI:')[0]} | AI: ❌ (데이터 로드 실패)")
                            log_trade_failure(coin, "AI 분석용 데이터 로드 실패", "1h 또는 15m 데이터 없음")
                    else:
                        print(log_prefix)
                    
                    confirmed_coins.append(coin)

            for coin in confirmed_coins:
                if coin in pending_confirmation: del pending_confirmation[coin]
            
            if fixed_investment_per_trade == 0:
                try:
                    balance = exchange.fetch_balance()['USDT']
                    total_capital = balance['total']
                    if total_capital < STRATEGY_CONFIG["MIN_CAPITAL_THRESHOLD"]:
                        print(f"⚠️ 경고: 총 자본금(${total_capital:.2f})이 비정상적으로 낮습니다")
                        time.sleep(60); continue
                    fixed_investment_per_trade = total_capital * STRATEGY_CONFIG['MARGIN_SIZE']
                    print(f"\n💰 포지션당 투입금 설정: ${fixed_investment_per_trade:.2f} (총자산의 {STRATEGY_CONFIG['MARGIN_SIZE']*100}%)")
                except Exception as e:
                    print(f"잔고 조회 오류: {e}"); time.sleep(60); continue
            
            occupied_slots = len(get_all_open_trades_from_db()) + len(pending_confirmation)
            if occupied_slots < MAX_CONCURRENT_POSITIONS:
                print(f"\n--- 신규 진입 탐색 ({occupied_slots}/{MAX_CONCURRENT_POSITIONS} 슬롯 사용중) ---")
                current_positions_with_direction = {(t['coin_symbol'], t['action']) for t in get_all_open_trades_from_db()}

                for coin, config in coins_to_scan.items():
                    if len(get_all_open_trades_from_db()) + len(pending_confirmation) >= MAX_CONCURRENT_POSITIONS: break
                    if coin in pending_confirmation: continue
                    
                    try:
                        signal_tf = STRATEGY_CONFIG['SIGNAL_TIMEFRAME']
                        df_signal = fetch_and_analyze_data(config['symbol'], signal_tf, STRATEGY_CONFIG, calculate_bb=True, calculate_rsi=True)
                        if df_signal is None or len(df_signal) < 2: continue
                        
                        signal, reason = check_for_initial_signal(df_signal, STRATEGY_CONFIG)
                        print(f"-> [{signal_tf}] {coin}: {reason}")

                        if signal:
                            if (coin, signal['direction']) in current_positions_with_direction:
                                print(f"     ㄴ 이미 {signal['direction']} 포지션 보유중"); continue
                            opposite_direction = 'short' if signal['direction'] == 'long' else 'long'
                            if (coin, opposite_direction) in current_positions_with_direction:
                                print(f"     ㄴ 반대 방향 포지션 보유중 (헷징 방지)"); continue
                            
                            confirm_tf_str = STRATEGY_CONFIG['CONFIRMATION_TIMEFRAME']
                            confirm_mins = int(confirm_tf_str.replace('m', ''))
                            next_interval_start = (now.minute // confirm_mins + 1) * confirm_mins
                            confirm_time = (now.replace(minute=next_interval_start, second=5, microsecond=0) + timedelta(minutes=confirm_mins)) if next_interval_start < 60 else (now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1, minutes=confirm_mins))

                            signal['confirmation_timestamp'] = confirm_time
                            signal['investment_amount'] = fixed_investment_per_trade
                            pending_confirmation[coin] = signal
                            print(f"     ㄴ 확증 대기열 추가 (예정: {confirm_time.strftime('%H:%M:%S')})")
                    except Exception as e:
                        print(f"   - 스캔 오류 ({coin}): {e}")

            print(f"\n{'='*70}\n사이클 완료. 60초 대기...\n{'='*70}")
            time.sleep(60)
            
        except Exception as e:
            print(f"메인 루프 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
