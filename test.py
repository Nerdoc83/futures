"""
AI 이더리움 데이트레이딩 봇 - Gemini + 고급 지표 + 뉴스 분석 (v2.3 - Trend Analysis + Daily/Weekly)
--------------------------------------------------------
기능:
- 데이트레이딩 최적화 (5분, 15분, 1시간, 일봉, 주봉 차트)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 추세 분석 시스템 (장기 추세 vs 단기 신호)
- 추세 순응/역추세 거래 전략
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- 실시간 이더리움 뉴스 분석 (SERP API)
- Gemini API 기반 AI 분석
- 동적 레버리지 및 포지션 사이징 (5-50배)
- 개선된 SL/TP 설정 (ATR 기반 동적 설정 + 추세별 조정)
- 24시간 무제한 거래
- 이더리움 선물 거래 최적화
- 최소 투자금액: 100 USDT
--------------------------------------------------------
"""
# ===== 필요한 라이브러리 임포트 =====
import ccxt  # 암호화폐 거래소 API 라이브러리
import os  # 환경 변수 및 파일 시스템 접근
import math  # 수학 연산
import time  # 시간 지연 및 타임스탬프
import pandas as pd  # 데이터 분석 및 조작
import numpy as np  # 수치 계산
import requests  # HTTP 요청
import json  # JSON 데이터 처리
import sqlite3  # 로컬 데이터베이스
from dotenv import load_dotenv  # 환경 변수 로드
load_dotenv()  # .env 파일에서 환경 변수 로드
import google.generativeai as genai  # Gemini API
from datetime import datetime  # 날짜 및 시간 처리

# ===== 모멘텀 지표 계산 함수들 =====
def calculate_rsi(prices, window=14):
    """RSI 계산 함수"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산 함수"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator 계산"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, window=14):
    """Williams %R 계산"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_atr(df, window=14):
    """Average True Range 계산"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_ema(prices, window):
    """지수이동평균 계산"""
    return prices.ewm(span=window).mean()

def calculate_sma(prices, window):
    """단순이동평균 계산"""
    return prices.rolling(window=window).mean()

# ===== 추세 분석 함수들 =====
def analyze_trend_direction(df, short_ema=20, long_ema=50):
    """추세 방향 분석 (EMA 기반) - 안전한 처리"""
    try:
        if len(df) < long_ema :  # 충분한 데이터가 없으면
            print(f"Insufficient data for trend analysis: {len(df)} < {long_ema + 5}")
            return {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
        
        df['EMA_short'] = calculate_ema(df['close'], short_ema)
        df['EMA_long'] = calculate_ema(df['close'], long_ema)
        
        # NaN 제거 후 다시 체크
        df_clean = df.dropna()
        if len(df_clean) < 3:
            print(f"Too little data after EMA calculation: {len(df_clean)}")
            return {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
        
        current_short = df_clean['EMA_short'].iloc[-1]
        current_long = df_clean['EMA_long'].iloc[-1]
        current_price = df_clean['close'].iloc[-1]
        
        # 안전한 계산
        if current_long == 0:
            ema_distance_pct = 0
        else:
            ema_distance_pct = ((current_short - current_long) / current_long) * 100
        
        # 추세 방향 결정
        if current_short > current_long and current_price > current_short:
            if ema_distance_pct > 2.0:
                trend = "STRONG_UPTREND"
            elif ema_distance_pct > 0.5:
                trend = "UPTREND"
            else:
                trend = "WEAK_UPTREND"
        elif current_short < current_long and current_price < current_short:
            if ema_distance_pct < -2.0:
                trend = "STRONG_DOWNTREND"
            elif ema_distance_pct < -0.5:
                trend = "DOWNTREND"
            else:
                trend = "WEAK_DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        return {
            "trend_direction": trend,
            "ema_distance_pct": ema_distance_pct,
            "current_price": current_price,
            "ema_short": current_short,
            "ema_long": current_long
        }
        
    except Exception as e:
        print(f"Error in trend analysis: {e}")
        return {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}

def determine_trend_strategy(short_term_signal, daily_trend, weekly_trend):
    """단기 신호와 장기 추세를 비교하여 전략 결정"""
    try:
        # 장기 추세 통합 판단 (주봉 > 일봉 우선순위)
        weekly_direction = weekly_trend.get('trend_direction', 'UNKNOWN')
        daily_direction = daily_trend.get('trend_direction', 'UNKNOWN')
        
        # 주봉 추세가 강하면 주봉 우선, 아니면 일봉 우선
        if 'STRONG' in weekly_direction:
            long_term_trend = weekly_direction
            primary_timeframe = "WEEKLY"
        else:
            long_term_trend = daily_direction
            primary_timeframe = "DAILY"
        
        # 단기 신호 방향 결정
        short_signal_direction = "BULLISH" if short_term_signal.lower() == "long" else "BEARISH"
        
        # 장기 추세 방향 결정
        long_trend_direction = "BULLISH" if "UP" in long_term_trend else "BEARISH" if "DOWN" in long_term_trend else "NEUTRAL"
        
        # 전략 결정
        if short_signal_direction == long_trend_direction and long_trend_direction != "NEUTRAL":
            # 추세 순응 거래
            strategy_type = "WITH_TREND"
            conviction_level = "HIGH"
            
            # 강한 추세일수록 더 높은 확신
            if "STRONG" in long_term_trend:
                conviction_level = "VERY_HIGH"
                position_size_multiplier = 1.3
                tp_atr_multiplier = 4.0  # 더 큰 익절 목표
            else:
                position_size_multiplier = 1.1
                tp_atr_multiplier = 3.0
            
            sl_atr_multiplier = 1.8  # 표준 손절
            
        elif short_signal_direction != long_trend_direction and long_trend_direction != "NEUTRAL":
            # 역추세 거래 (스캘핑)
            strategy_type = "COUNTER_TREND"
            conviction_level = "LOW"
            position_size_multiplier = 0.7  # 작은 포지션
            tp_atr_multiplier = 1.8  # 빠른 익절
            sl_atr_multiplier = 1.2  # 타이트한 손절
            
        else:
            # 중립 상황
            strategy_type = "NEUTRAL"
            conviction_level = "MEDIUM"
            position_size_multiplier = 0.9
            tp_atr_multiplier = 2.5
            sl_atr_multiplier = 1.5
        
        return {
            "strategy_type": strategy_type,
            "conviction_level": conviction_level,
            "long_term_trend": long_term_trend,
            "short_signal_direction": short_signal_direction,
            "primary_timeframe": primary_timeframe,
            "position_size_multiplier": position_size_multiplier,
            "tp_atr_multiplier": tp_atr_multiplier,
            "sl_atr_multiplier": sl_atr_multiplier,
            "trend_alignment": short_signal_direction == long_trend_direction
        }
        
    except Exception as e:
        print(f"Error in trend strategy determination: {e}")
        return {
            "strategy_type": "NEUTRAL",
            "conviction_level": "MEDIUM",
            "position_size_multiplier": 1.0,
            "tp_atr_multiplier": 2.5,
            "sl_atr_multiplier": 1.5
        }

# ===== 긴급 이벤트 감지 및 대응 시스템 =====
def detect_sudden_market_event(current_price, timeframe_data):
    """갑작스러운 시장 이벤트 감지 (1분봉 우선, 급등/급락, 볼륨 급증)"""
    try:
        # 1분봉이 있으면 우선 사용 (긴급 모드), 없으면 5분봉 사용
        primary_tf = '1m' if '1m' in timeframe_data else '5m'
        secondary_tf = '5m' if '1m' in timeframe_data else '15m'
        
        if primary_tf not in timeframe_data:
            return {"event_detected": False}
        
        # 주요 타임프레임 데이터 분석
        recent_primary = timeframe_data[primary_tf]['recent_candles']
        recent_secondary = timeframe_data.get(secondary_tf, {}).get('recent_candles', [])
        
        if len(recent_primary) < 5:
            return {"event_detected": False}
        
        # 현재 ATR 기준 정상 변동성 범위
        current_atr = timeframe_data[primary_tf]['current_indicators']['atr']
        
        # 1분봉 기준으로 더 민감한 감지
        if primary_tf == '1m':
            normal_move_threshold = 1.0  # 1분 내 1% 이상 변동
            volume_threshold = 2.5  # 볼륨 2.5배 이상
            sustained_threshold = 2.0  # 지속적 변동 2% 이상
        else:
            normal_move_threshold = 2.0  # 5분 내 2% 이상 변동
            volume_threshold = 3.0  # 볼륨 3배 이상
            sustained_threshold = 3.0  # 지속적 변동 3% 이상
        
        # 최근 캔들들의 가격 변동 및 볼륨 변화 분석
        latest_candles = recent_primary[-5:]  # 최근 5개 캔들
        price_changes = []
        volume_changes = []
        consecutive_moves = 0
        move_direction = None
        
        for i in range(1, len(latest_candles)):
            prev_close = latest_candles[i-1]['close']
            curr_close = latest_candles[i]['close']
            price_change_pct = ((curr_close - prev_close) / prev_close) * 100
            abs_price_change = abs(price_change_pct)
            
            prev_volume = latest_candles[i-1]['volume']
            curr_volume = latest_candles[i]['volume']
            volume_ratio = curr_volume / (prev_volume + 1) if prev_volume > 0 else 1
            
            price_changes.append(abs_price_change)
            volume_changes.append(volume_ratio)
            
            # 연속적인 방향성 움직임 체크
            if abs_price_change > (normal_move_threshold * 0.5):  # 절반 기준으로 방향성 체크
                current_direction = 'UP' if price_change_pct > 0 else 'DOWN'
                if move_direction is None:
                    move_direction = current_direction
                    consecutive_moves = 1
                elif move_direction == current_direction:
                    consecutive_moves += 1
                else:
                    consecutive_moves = 1
                    move_direction = current_direction
        
        # 이벤트 감지 조건들
        max_price_change = max(price_changes) if price_changes else 0
        max_volume_ratio = max(volume_changes) if volume_changes else 1
        avg_volume_ratio = sum(volume_changes) / len(volume_changes) if volume_changes else 1
        
        # 급격한 변동 감지
        sudden_price_move = max_price_change > normal_move_threshold
        volume_spike = max_volume_ratio > volume_threshold or avg_volume_ratio > (volume_threshold * 0.7)
        sustained_directional_move = consecutive_moves >= 3  # 3캔들 연속 같은 방향
        
        # 누적 변동률 계산 (최근 5캔들)
        total_price_change = abs((latest_candles[-1]['close'] - latest_candles[0]['close']) / latest_candles[0]['close']) * 100
        severe_cumulative_move = total_price_change > sustained_threshold
        
        # 1분봉 특별 감지: 매우 빠른 변동
        if primary_tf == '1m':
            # 1분 내 0.8% 이상도 긴급상황으로 간주
            ultra_fast_move = max_price_change > 0.8
            event_detected = ultra_fast_move or (sudden_price_move and volume_spike)
        else:
            # 기존 5분봉 로직
            event_detected = sudden_price_move and (volume_spike or sustained_directional_move)
        
        # 추가 조건: 누적 변동이 큰 경우도 이벤트로 간주
        if severe_cumulative_move and volume_spike:
            event_detected = True
        
        if event_detected:
            # 최종 방향 결정 (최근 2캔들 기준)
            recent_direction = "UP" if latest_candles[-1]['close'] > latest_candles[-2]['close'] else "DOWN"
            
            # 심각도 계산 (1분봉은 더 민감하게)
            if primary_tf == '1m':
                severity = "EXTREME" if max_price_change > 2.0 else "HIGH" if max_price_change > 1.2 else "MEDIUM"
            else:
                severity = "EXTREME" if max_price_change > 6.0 else "HIGH" if max_price_change > 4.0 else "MEDIUM"
            
            return {
                "event_detected": True,
                "event_type": "SUDDEN_SURGE" if recent_direction == "UP" else "SUDDEN_DROP",
                "primary_timeframe": primary_tf,
                "price_change_max": max_price_change,
                "price_change_cumulative": total_price_change,
                "volume_spike_ratio": max_volume_ratio,
                "consecutive_moves": consecutive_moves,
                "direction": recent_direction,
                "severity": severity,
                "action_needed": True,
                "ultra_fast": primary_tf == '1m' and max_price_change > 1.5
            }
        
        return {"event_detected": False, "action_needed": False}
        
    except Exception as e:
        print(f"Error in event detection: {e}")
        return {"event_detected": False}

def check_high_impact_news_keywords(news_data):
    """고임팩트 뉴스 키워드 감지"""
    if not news_data:
        return {"high_impact": False}
    
    # 고임팩트 키워드들 (이더리움/암호화폐)
    high_impact_keywords = [
        "fed", "powell", "jackson hole", "interest rate", "monetary policy",
        "regulation", "sec", "etf", "institutional", "blackrock", "grayscale",
        "upgrade", "merge", "staking", "burn", "fork", "ethereum 2.0",
        "defi", "hack", "exploit", "crash", "surge", "breakout",
        "whale", "liquidation", "manipulation", "ban", "legal"
    ]
    
    high_impact_found = []
    for news in news_data[:5]:  # 최신 5개 뉴스만 확인
        title_lower = news.get("title", "").lower()
        for keyword in high_impact_keywords:
            if keyword in title_lower:
                high_impact_found.append({
                    "keyword": keyword,
                    "title": news.get("title", ""),
                    "source": news.get("source", "")
                })
                break
    
    return {
        "high_impact": len(high_impact_found) > 0,
        "impact_count": len(high_impact_found),
        "found_keywords": high_impact_found
    }

def handle_emergency_position_adjustment(current_price, event_data, current_trade, news_impact):
    """긴급 상황 시 포지션 조정"""
    if not current_trade or not event_data.get("action_needed"):
        return False
    
    try:
        trade_id = current_trade['id']
        action = current_trade['action']
        entry_price = current_trade['entry_price']
        leverage = current_trade['leverage']
        amount = current_trade['amount']
        
        # 현재 P/L 계산
        if action == 'long':
            current_pnl_pct = ((current_price / entry_price) - 1) * leverage * 100
        else:
            current_pnl_pct = ((entry_price / current_price) - 1) * leverage * 100
        
        print(f"\n🚨 EMERGENCY EVENT DETECTED! 🚨")
        print(f"Event Type: {event_data.get('event_type', 'UNKNOWN')}")
        print(f"Timeframe: {event_data.get('primary_timeframe', '5m').upper()}")
        print(f"Max Price Change: {event_data.get('price_change_max', 0):.2f}%")
        print(f"Cumulative Change: {event_data.get('price_change_cumulative', 0):.2f}%")
        print(f"Current Position P/L: {current_pnl_pct:.2f}%")
        
        # 긴급 대응 로직
        emergency_action = None
        
        # 1. 수익 중인 포지션 + 방향이 맞는 경우 → 부분 익절
        if current_pnl_pct > 10:  # 10% 이상 수익
            if (action == 'long' and event_data.get('direction') == 'UP') or \
               (action == 'short' and event_data.get('direction') == 'DOWN'):
                emergency_action = "PARTIAL_PROFIT"
                
        # 2. 손실 중인 포지션 + 방향이 반대인 경우 → 즉시 손절
        elif current_pnl_pct < -5:  # 5% 이상 손실
            if (action == 'long' and event_data.get('direction') == 'DOWN') or \
               (action == 'short' and event_data.get('direction') == 'UP'):
                emergency_action = "EMERGENCY_EXIT"
        
        # 3. 고임팩트 뉴스가 있는 경우 추가 고려
        if news_impact.get("high_impact") and current_pnl_pct > 5:
            emergency_action = "PARTIAL_PROFIT"
        
        # 긴급 액션 실행
        if emergency_action == "PARTIAL_PROFIT":
            # 50% 부분 익절
            partial_amount = amount * 0.5
            if action == 'long':
                exchange.create_market_sell_order(symbol, partial_amount)
            else:
                exchange.create_market_buy_order(symbol, partial_amount)
            
            print(f"✅ PARTIAL PROFIT TAKEN: {partial_amount} ETH at ${current_price:,.2f}")
            print(f"Profit Secured: ~{current_pnl_pct * 0.5:.1f}%")
            
            return True
            
        elif emergency_action == "EMERGENCY_EXIT":
            # 전체 포지션 즉시 청산
            if action == 'long':
                exchange.create_market_sell_order(symbol, amount)
            else:
                exchange.create_market_buy_order(symbol, amount)
            
            # 기존 SL/TP 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    exchange.cancel_order(order['id'], symbol)
            except:
                pass
            
            # DB 업데이트
            handle_position_closure(current_price, action, amount, trade_id)
            
            print(f"🚨 EMERGENCY EXIT: Full position closed at ${current_price:,.2f}")
            print(f"Emergency P/L: {current_pnl_pct:.1f}%")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error in emergency position adjustment: {e}")
        return False

def detect_flash_opportunity(current_price, event_data, market_sentiment):
    """급변동 시 새로운 기회 포착"""
    if not event_data.get("event_detected"):
        return None
    
    try:
        # 플래시 기회 조건들
        price_move = event_data.get('price_change_max', 0)
        cumulative_move = event_data.get('price_change_cumulative', 0)
        direction = event_data.get('direction', '')
        severity = event_data.get('severity', 'LOW')
        timeframe = event_data.get('primary_timeframe', '5m')
        ultra_fast = event_data.get('ultra_fast', False)
        
        # 급변동 후 리바운드 또는 연속 모멘텀 기회
        flash_opportunity = None
        
        # 1분봉 기준 초고속 거래 기회
        if timeframe == '1m' and ultra_fast:
            print("⚡ ULTRA-FAST 1MIN OPPORTUNITY DETECTED")
            
            # 초고속 스캘핑 기회 (더 보수적)
            if direction == 'UP' and price_move > 1.5:
                flash_opportunity = {
                    "direction": "SHORT",
                    "position_size": 0.15,  # 매우 보수적
                    "leverage": 8,  # 낮은 레버리지
                    "reasoning": f"Ultra-fast short after {price_move:.1f}% 1min spike"
                }
            elif direction == 'DOWN' and price_move > 1.5:
                flash_opportunity = {
                    "direction": "LONG", 
                    "position_size": 0.15,
                    "leverage": 8,
                    "reasoning": f"Ultra-fast long after {price_move:.1f}% 1min drop"
                }
        
        # 기존 5분봉+ 기준 플래시 기회
        else:
            # 1. 과도한 상승 후 숏기회 (오버익스텐션)
            if direction == 'UP' and price_move > 5.0 and severity == 'HIGH':
                # 펀딩비가 매우 높거나 롱숏비율이 극단적인 경우
                funding_rate = market_sentiment['funding_rate']['funding_rate_percentage']
                ls_ratio = market_sentiment['long_short_ratio']['latest_long_short_ratio']
                
                if funding_rate > 0.02 or ls_ratio > 2.5:  # 과열 신호
                    flash_opportunity = {
                        "direction": "SHORT",
                        "position_size": 0.2,  # 보수적
                        "leverage": 10,  # 낮은 레버리지
                        "reasoning": f"Flash crash opportunity after {price_move:.1f}% surge - overextended market"
                    }
            
            # 2. 과도한 하락 후 롱 기회 (오버솔드)
            elif direction == 'DOWN' and price_move > 5.0 and severity == 'HIGH':
                funding_rate = market_sentiment['funding_rate']['funding_rate_percentage']
                ls_ratio = market_sentiment['long_short_ratio']['latest_long_short_ratio']
                
                if funding_rate < -0.02 or ls_ratio < 0.6:  # 과매도 신호
                    flash_opportunity = {
                        "direction": "LONG",
                        "position_size": 0.2,
                        "leverage": 10,
                        "reasoning": f"Bounce opportunity after {price_move:.1f}% crash - oversold market"
                    }
            
            # 3. 연속 모멘텀 기회 (브레이크아웃)
            elif price_move > 3.0 and price_move < 8.0:  # 적당한 급변동
                # 볼륨과 함께 온 건전한 브레이크아웃
                volume_ratio = event_data.get('volume_spike_ratio', 1)
                consecutive = event_data.get('consecutive_moves', 0)
                
                if volume_ratio > 2.0 and consecutive >= 3:
                    flash_opportunity = {
                        "direction": "LONG" if direction == 'UP' else "SHORT",
                        "position_size": 0.25,
                        "leverage": 12,
                        "reasoning": f"Momentum continuation after {price_move:.1f}% move with {consecutive} consecutive candles"
                    }
        
        return flash_opportunity
        
    except Exception as e:
        print(f"Error detecting flash opportunity: {e}")
        return None

def fetch_ethereum_news():
    """최신 이더리움 뉴스 가져오기 함수"""
    try:
        # SERP API를 사용해 이더리움 관련 최신 뉴스 가져오기
        serp_api_key = os.getenv("SERP_API_KEY")
        
        if not serp_api_key:
            print("SERP API 키가 설정되지 않았습니다. 뉴스 분석을 건너뜁니다.")
            return []
        
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_news",
            "q": "ethereum ETH price trading",  # 이더리움 관련 키워드
            "gl": "us",
            "hl": "en",
            "api_key": serp_api_key,
            "num": 15  # 더 많은 뉴스 수집
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            # 최신 뉴스 10개만 추출하고 title과 date만 포함
            recent_news = []
            for i, news in enumerate(news_results[:10]):
                news_item = {
                    "title": news.get("title", ""),
                    "date": news.get("date", ""),
                    "source": news.get("source", {}).get("name", "Unknown")
                }
                recent_news.append(news_item)
            
            print(f"Collected {len(recent_news)} recent Ethereum news articles")
            return recent_news
        else:
            print(f"Error fetching Ethereum news: Status code {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error fetching Ethereum news: {e}")
        return []

def analyze_news_sentiment(news_data):
    """뉴스 제목 기반 간단한 감정 분석"""
    if not news_data:
        return {"sentiment": "NEUTRAL", "news_count": 0}
    
    bullish_keywords = ["surge", "rise", "bull", "pump", "moon", "breakout", "rally", "gains", "up", "positive", "institutional", "adoption"]
    bearish_keywords = ["crash", "dump", "bear", "drop", "fall", "decline", "down", "negative", "sell-off", "correction", "fears"]
    
    bullish_count = 0
    bearish_count = 0
    
    for news in news_data:
        title_lower = news.get("title", "").lower()
        
        for keyword in bullish_keywords:
            if keyword in title_lower:
                bullish_count += 1
                break
        
        for keyword in bearish_keywords:
            if keyword in title_lower:
                bearish_count += 1
                break
    
    total_sentiment_news = bullish_count + bearish_count
    
    if total_sentiment_news == 0:
        sentiment = "NEUTRAL"
    elif bullish_count > bearish_count * 1.5:
        sentiment = "BULLISH"
    elif bearish_count > bullish_count * 1.5:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"
    
    return {
        "sentiment": sentiment,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "news_count": len(news_data),
        "sentiment_strength": abs(bullish_count - bearish_count) / max(total_sentiment_news, 1)
    }

# ===== 설정 및 초기화 =====
# 바이낸스 API 설정
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})
symbol = "ETH/USDT"  # 이더리움 선물 페어

# Gemini API 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# SQLite 데이터베이스 설정
DB_FILE = "ethereum_daytrading.db"  # 이더리움용 데이터베이스 파일

# ===== 데이터베이스 관련 함수 =====
def setup_database():
    """데이터베이스 및 필요한 테이블 생성"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 거래 기록 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        entry_price REAL NOT NULL,
        amount REAL NOT NULL,
        leverage INTEGER NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        sl_percentage REAL,
        tp_percentage REAL,
        position_size_percentage REAL NOT NULL,
        investment_amount REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp TEXT,
        profit_loss REAL,
        profit_loss_percentage REAL,
        trend_strategy TEXT,
        conviction_level TEXT
    )
    ''')
    
    # AI 분석 결과 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        current_price REAL NOT NULL,
        direction TEXT NOT NULL,
        recommended_position_size REAL NOT NULL,
        recommended_leverage INTEGER NOT NULL,
        stop_loss_percentage REAL,
        take_profit_percentage REAL,
        sl_atr_multiplier REAL,
        tp_atr_multiplier REAL,
        reasoning TEXT NOT NULL,
        news_sentiment TEXT,
        trend_strategy TEXT,
        conviction_level TEXT,
        daily_trend TEXT,
        weekly_trend TEXT,
        trade_id INTEGER,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    
    # 뉴스 데이터 테이블 추가
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        news_sentiment TEXT NOT NULL,
        bullish_count INTEGER NOT NULL,
        bearish_count INTEGER NOT NULL,
        news_count INTEGER NOT NULL,
        sentiment_strength REAL NOT NULL,
        news_titles TEXT NOT NULL
    )
    ''')
    
    # 기존 테이블에 새로운 컬럼 추가 (마이그레이션)
    try:
        cursor.execute("PRAGMA table_info(ai_analysis)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'news_sentiment' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN news_sentiment TEXT DEFAULT "NEUTRAL"')
        if 'sl_atr_multiplier' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN sl_atr_multiplier REAL')
        if 'tp_atr_multiplier' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN tp_atr_multiplier REAL')
        if 'trend_strategy' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN trend_strategy TEXT')
        if 'conviction_level' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN conviction_level TEXT')
        if 'daily_trend' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN daily_trend TEXT')
        if 'weekly_trend' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN weekly_trend TEXT')
            
        # trades 테이블에도 추세 관련 컬럼 추가
        cursor.execute("PRAGMA table_info(trades)")
        trade_columns = [column[1] for column in cursor.fetchall()]
        
        if 'trend_strategy' not in trade_columns:
            cursor.execute('ALTER TABLE trades ADD COLUMN trend_strategy TEXT')
        if 'conviction_level' not in trade_columns:
            cursor.execute('ALTER TABLE trades ADD COLUMN conviction_level TEXT')
            
    except sqlite3.Error as e:
        print(f"Database migration note: {e}")
    
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, current_price, direction, recommended_position_size, 
        recommended_leverage, sl_atr_multiplier, tp_atr_multiplier,
        reasoning, news_sentiment, trend_strategy, conviction_level,
        daily_trend, weekly_trend, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        analysis_data.get('current_price', 0),
        analysis_data.get('direction', 'NO_POSITION'),
        analysis_data.get('recommended_position_size', 0),
        analysis_data.get('recommended_leverage', 0),
        analysis_data.get('sl_atr_multiplier', 0),
        analysis_data.get('tp_atr_multiplier', 0),
        analysis_data.get('reasoning', ''),
        analysis_data.get('news_sentiment', 'NEUTRAL'),
        analysis_data.get('trend_strategy', 'NEUTRAL'),
        analysis_data.get('conviction_level', 'MEDIUM'),
        analysis_data.get('daily_trend', 'UNKNOWN'),
        analysis_data.get('weekly_trend', 'UNKNOWN'),
        trade_id
    ))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def save_news_analysis(news_sentiment_data):
    """뉴스 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO news_analysis (
        timestamp, news_sentiment, bullish_count, bearish_count, 
        news_count, sentiment_strength, news_titles
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        news_sentiment_data.get('sentiment', 'NEUTRAL'),
        news_sentiment_data.get('bullish_count', 0),
        news_sentiment_data.get('bearish_count', 0),
        news_sentiment_data.get('news_count', 0),
        news_sentiment_data.get('sentiment_strength', 0),
        json.dumps(news_sentiment_data.get('news_titles', []))
    ))
    
    conn.commit()
    conn.close()

def save_trade(trade_data):
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp, action, entry_price, amount, leverage, sl_price, tp_price,
        position_size_percentage, investment_amount, trend_strategy, conviction_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        trade_data.get('action', ''),
        trade_data.get('entry_price', 0),
        trade_data.get('amount', 0),
        trade_data.get('leverage', 0),
        trade_data.get('sl_price', 0),
        trade_data.get('tp_price', 0),
        trade_data.get('position_size_percentage', 0),
        trade_data.get('investment_amount', 0),
        trade_data.get('trend_strategy', 'NEUTRAL'),
        trade_data.get('conviction_level', 'MEDIUM')
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def update_trade_status(trade_id, status, exit_price=None, exit_timestamp=None, profit_loss=None, profit_loss_percentage=None):
    """거래 상태를 업데이트합니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    update_fields = ["status = ?"]
    update_values = [status]
    
    if exit_price is not None:
        update_fields.append("exit_price = ?")
        update_values.append(exit_price)
    
    if exit_timestamp is not None:
        update_fields.append("exit_timestamp = ?")
        update_values.append(exit_timestamp)
    
    if profit_loss is not None:
        update_fields.append("profit_loss = ?")
        update_values.append(profit_loss)
    
    if profit_loss_percentage is not None:
        update_fields.append("profit_loss_percentage = ?")
        update_values.append(profit_loss_percentage)
    
    update_sql = f"UPDATE trades SET {', '.join(update_fields)} WHERE id = ?"
    update_values.append(trade_id)
    
    cursor.execute(update_sql, update_values)
    conn.commit()
    conn.close()

def get_latest_open_trade():
    """가장 최근의 열린 거래 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC
    LIMIT 1
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'action': result[1],
            'entry_price': result[2],
            'amount': result[3],
            'leverage': result[4],
            'sl_price': result[5],
            'tp_price': result[6]
        }
    return None

def get_trade_summary(days=7):
    """지정된 일수 동안의 거래 요약 정보를 가져옵니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage
    FROM trades
    WHERE exit_timestamp IS NOT NULL
    AND timestamp >= datetime('now', ?)
    ''', (f'-{days} days',))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'total_trades': result[0] or 0,
            'winning_trades': result[1] or 0,
            'losing_trades': result[2] or 0,
            'total_profit_loss': result[3] or 0,
            'avg_profit_loss_percentage': result[4] or 0
        }
    return None

def get_historical_trading_data(limit=5):
    """과거 거래 내역 가져오기"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        t.action, t.entry_price, t.exit_price, t.leverage,
        t.profit_loss_percentage, t.trend_strategy, t.conviction_level,
        a.reasoning
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status = 'CLOSED'
    ORDER BY t.timestamp DESC
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    historical_data = []
    for row in results:
        historical_data.append({k: row[k] for k in row.keys()})
    
    conn.close()
    return historical_data

def get_performance_metrics():
    """거래 성과 메트릭스를 계산합니다"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage,
        MAX(profit_loss_percentage) as max_profit_percentage,
        MIN(profit_loss_percentage) as max_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    ''')
    
    overall_metrics = cursor.fetchone()
    conn.close()
    
    metrics = {
        "total_trades": overall_metrics[0] or 0,
        "winning_trades": overall_metrics[1] or 0,
        "avg_profit_loss_percentage": overall_metrics[2] or 0,
        "max_profit_percentage": overall_metrics[3] or 0,
        "max_loss_percentage": overall_metrics[4] or 0
    }
    
    if metrics["total_trades"] > 0:
        metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
    else:
        metrics["win_rate"] = 0
    
    return metrics

# ===== 데이터 수집 함수 (이더리움 최적화 + 일봉/주봉 추가) =====
def fetch_multi_timeframe_data(emergency_mode=False):
    """이더리움 데이트레이딩 최적화된 멀티 타임프레임 데이터 수집 (일봉/주봉 포함)"""
    if emergency_mode:
        # 긴급 모드: 1분봉 추가로 더 빠른 감지
        timeframes = {
            "1m": {"timeframe": "1m", "limit": 60},    # 1시간 데이터 (긴급시에만)
            "5m": {"timeframe": "5m", "limit": 60},    # 5시간 데이터
            "15m": {"timeframe": "15m", "limit": 48},  # 12시간 데이터
            "1h": {"timeframe": "1h", "limit": 24},    # 24시간 데이터
            "1d": {"timeframe": "1d", "limit": 100},    # 60일 데이터 (추세 분석용) - 증가
            "1w": {"timeframe": "1w", "limit": 100}     # 50주 데이터 (장기 추세) - 증가
        }
        print("🚨 EMERGENCY MODE: 1분봉 포함 + 일봉/주봉 추세 분석")
    else:
        # 일반 모드: 기존 + 일봉/주봉 추가
        timeframes = {
            "5m": {"timeframe": "5m", "limit": 100},   # 8시간 20분 데이터
            "15m": {"timeframe": "15m", "limit": 96},  # 24시간 데이터
            "1h": {"timeframe": "1h", "limit": 48},    # 48시간 데이터
            "1d": {"timeframe": "1d", "limit": 100},    # 60일 데이터 (추세 분석용) - 증가
            "1w": {"timeframe": "1w", "limit": 100}     # 50주 데이터 (장기 추세) - 증가
        }
    
    multi_tf_data = {}
    
    for tf_name, tf_params in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf_params["timeframe"], 
                limit=tf_params["limit"]
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # ===== 모멘텀 지표 계산 =====
            # RSI (14, 21)
            df['RSI_14'] = calculate_rsi(df['close'], window=14)
            df['RSI_21'] = calculate_rsi(df['close'], window=21)
            
            # MACD
            macd, signal, histogram = calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_histogram'] = histogram
            
            # Stochastic Oscillator
            stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Williams %R
            df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
            
            # ATR (변동성 측정)
            df['ATR'] = calculate_atr(df)
            
            # NaN 값 제거
            df.dropna(inplace=True)
            
            # 데이터 충분성 체크 (특히 일봉/주봉)
            if len(df) < 10:
                print(f"Warning: {tf_name} has insufficient data ({len(df)} candles) after indicator calculation")
                if tf_name in ['1d', '1w']:
                    # 일봉/주봉의 경우 최소한의 추세 분석이라도 시도
                    if len(df) < 3:
                        print(f"Skipping {tf_name} - too little data")
                        continue
                else:
                    # 단기 타임프레임은 건너뛰기
                    continue
            
            # ===== 추세 분석 (일봉/주봉용) =====
            trend_analysis = None
            if tf_name in ['1d', '1w'] and len(df) >= 10:  # 최소 10개 데이터 필요
                try:
                    trend_analysis = analyze_trend_direction(df.copy())
                    print(f"{tf_name.upper()} Trend: {trend_analysis.get('trend_direction', 'UNKNOWN')}")
                except Exception as trend_error:
                    print(f"Error in {tf_name} trend analysis: {trend_error}")
                    trend_analysis = {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
            
            # 현재 지표 안전하게 추출
            try:
                current_indicators = {
                    "current_price": float(df['close'].iloc[-1]),
                    "rsi_14": float(df['RSI_14'].iloc[-1]) if not pd.isna(df['RSI_14'].iloc[-1]) else 50.0,
                    "rsi_21": float(df['RSI_21'].iloc[-1]) if not pd.isna(df['RSI_21'].iloc[-1]) else 50.0,
                    "macd": float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else 0.0,
                    "macd_signal": float(df['MACD_signal'].iloc[-1]) if not pd.isna(df['MACD_signal'].iloc[-1]) else 0.0,
                    "macd_histogram": float(df['MACD_histogram'].iloc[-1]) if not pd.isna(df['MACD_histogram'].iloc[-1]) else 0.0,
                    "stoch_k": float(df['Stoch_K'].iloc[-1]) if not pd.isna(df['Stoch_K'].iloc[-1]) else 50.0,
                    "stoch_d": float(df['Stoch_D'].iloc[-1]) if not pd.isna(df['Stoch_D'].iloc[-1]) else 50.0,
                    "williams_r": float(df['Williams_R'].iloc[-1]) if not pd.isna(df['Williams_R'].iloc[-1]) else -50.0,
                    "atr": float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.02,
                    "volume": float(df['volume'].iloc[-1])
                }

                if trend_analysis:
                    current_indicators['trend_analysis'] = trend_analysis
                    
            except IndexError as idx_error:
                print(f"Index error in {tf_name} indicators extraction: {idx_error}")
                continue
            except Exception as ind_error:
                print(f"Error extracting {tf_name} indicators: {ind_error}")
                continue
            
            # 긴급 모드에서는 더 많은 최근 캔들 데이터 저장
            candle_count = 10 if emergency_mode and tf_name == "1m" else 5
            recent_candles = df.tail(candle_count).copy()
            recent_candles['timestamp'] = recent_candles['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_candles_dict = recent_candles.to_dict('records')
            
            # float64를 일반 float로 변환
            for candle in recent_candles_dict:
                for key, value in candle.items():
                    if isinstance(value, (np.integer, np.floating)):
                        candle[key] = float(value)
            
            multi_tf_data[tf_name] = {
                "current_indicators": current_indicators,
                "recent_candles": recent_candles_dict
            }
            
            timeframe_display = tf_name.upper()
            if emergency_mode and tf_name == "1m":
                timeframe_display += " (EMERGENCY)"
            elif tf_name in ['1d', '1w']:
                timeframe_display += " (TREND)"
            
            print(f"Collected {timeframe_display} data: {len(df)} candles")
            
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수 (이더리움용) =====
def fetch_funding_rate():
    """이더리움 펀딩비 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {'symbol': 'ETHUSDT'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            funding_rate = float(data.get('lastFundingRate', 0))
            next_funding_time = datetime.fromtimestamp(int(data.get('nextFundingTime', 0)) / 1000).isoformat()
            
            return {
                "current_funding_rate": funding_rate,
                "funding_rate_percentage": funding_rate * 100,
                "next_funding_time": next_funding_time,
                "funding_sentiment": "HIGH_LONG_LEVERAGE" if funding_rate > 0.01 else 
                                  "HIGH_SHORT_LEVERAGE" if funding_rate < -0.01 else "NEUTRAL"
            }
    except Exception as e:
        print(f"Error fetching funding rate: {e}")
        return {"current_funding_rate": 0, "funding_rate_percentage": 0, "funding_sentiment": "UNKNOWN"}

def fetch_open_interest():
    """이더리움 미결제약정 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {'symbol': 'ETHUSDT'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            current_data = response.json()
            current_oi = float(current_data.get('openInterest', 0))
            
            # 과거 데이터도 가져오기 (24시간 전 비교용)
            url_historical = "https://fapi.binance.com/futures/data/openInterestHist"
            params_hist = {
                'symbol': 'ETHUSDT',
                'period': '1h',
                'limit': 24
            }
            hist_response = requests.get(url_historical, params=params_hist)
            
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if hist_data and len(hist_data) > 1:
                    previous_oi = float(hist_data[-2].get('sumOpenInterest', current_oi))
                    first_oi = float(hist_data[0].get('sumOpenInterest', current_oi))
                    
                    oi_change_percentage = ((current_oi - previous_oi) / previous_oi) * 100 if previous_oi > 0 else 0
                    oi_change_24h_percentage = ((current_oi - first_oi) / first_oi) * 100 if first_oi > 0 else 0
                    
                    return {
                        "latest_open_interest": current_oi,
                        "oi_change_1h_percentage": oi_change_percentage,
                        "oi_change_24h_percentage": oi_change_24h_percentage,
                        "oi_trend": "INCREASING" if oi_change_percentage > 2 else "DECREASING" if oi_change_percentage < -2 else "STABLE"
                    }
            
            # 과거 데이터를 가져올 수 없는 경우 현재 데이터만 반환
            return {
                "latest_open_interest": current_oi,
                "oi_change_1h_percentage": 0,
                "oi_change_24h_percentage": 0,
                "oi_trend": "STABLE"
            }
        
    except Exception as e:
        print(f"Error fetching open interest: {e}")
        return {"latest_open_interest": 0, "oi_change_1h_percentage": 0, "oi_trend": "UNKNOWN"}

def fetch_long_short_ratio():
    """이더리움 롱숏 비율 데이터 수집"""
    try:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {
            'symbol': 'ETHUSDT',
            'period': '5m',
            'limit': 50
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            
            latest_ratio = df['longShortRatio'].iloc[-1]
            avg_ratio = df['longShortRatio'].mean()
            
            sentiment = "EXTREME_LONG_BIAS" if latest_ratio > 2.5 else \
                       "LONG_BIAS" if latest_ratio > 1.5 else \
                       "SHORT_BIAS" if latest_ratio < 0.8 else \
                       "EXTREME_SHORT_BIAS" if latest_ratio < 0.5 else "BALANCED"
            
            return {
                "latest_long_short_ratio": latest_ratio,
                "average_long_short_ratio": avg_ratio,
                "ls_sentiment": sentiment,
                "contrarian_signal": "SHORT_OPPORTUNITY" if latest_ratio > 2.0 else 
                                   "LONG_OPPORTUNITY" if latest_ratio < 0.8 else "NO_CLEAR_SIGNAL"
            }
    except Exception as e:
        print(f"Error fetching long/short ratio: {e}")
        return {"latest_long_short_ratio": 1.0, "ls_sentiment": "UNKNOWN", "contrarian_signal": "NO_DATA"}

def fetch_liquidation_data():
    """이더리움 청산 데이터 수집"""
    try:
        url = "https://fapi.binance.com/fapi/v1/forceOrders"
        params = {
            'symbol': 'ETHUSDT',
            'limit': 100
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            long_liquidations = [order for order in data if order['side'] == 'SELL']
            short_liquidations = [order for order in data if order['side'] == 'BUY']
            
            total_long_liq = sum([float(order['executedQty']) for order in long_liquidations])
            total_short_liq = sum([float(order['executedQty']) for order in short_liquidations])
            
            liq_ratio = total_long_liq / (total_short_liq + 0.0001)  # 0으로 나누기 방지
            
            return {
                "long_liquidations_eth": total_long_liq,
                "short_liquidations_eth": total_short_liq,
                "liquidation_ratio": liq_ratio,
                "liquidation_pressure": "LONG_SQUEEZE" if liq_ratio > 2 else 
                                      "SHORT_SQUEEZE" if liq_ratio < 0.5 else "BALANCED"
            }
    except Exception as e:
        print(f"Error fetching liquidation data: {e}")
        return {"liquidation_pressure": "UNKNOWN", "liquidation_ratio": 1.0}

def fetch_market_sentiment():
    """모든 시장 심리 지표를 통합 수집"""
    print("Collecting market sentiment data...")
    
    funding_data = fetch_funding_rate()
    oi_data = fetch_open_interest()
    ls_ratio_data = fetch_long_short_ratio()
    liquidation_data = fetch_liquidation_data()
    
    market_sentiment = {
        "funding_rate": funding_data,
        "open_interest": oi_data,
        "long_short_ratio": ls_ratio_data,
        "liquidations": liquidation_data
    }
    
    print(f"Market Sentiment - Funding: {funding_data['funding_rate_percentage']:.4f}%, "
          f"LS Ratio: {ls_ratio_data['latest_long_short_ratio']:.2f}, "
          f"OI Trend: {oi_data['oi_trend']}")
    
    return market_sentiment

# ===== 포지션 관리 함수 =====
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    """포지션 종료 시 데이터베이스를 업데이트하고 결과를 표시합니다"""
    if current_trade_id is None:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            current_trade_id = latest_trade['id']
    
    if current_trade_id:
        latest_trade = get_latest_open_trade()
        if latest_trade:
            entry_price = latest_trade['entry_price']
            action = latest_trade['action']
            leverage = latest_trade['leverage']
            
            if action == 'long':
                profit_loss = (current_price - entry_price) * amount
                # 원금 대비 수익률 계산
                profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
            else: # short
                profit_loss = (entry_price - current_price) * amount
                # 원금 대비 수익률 계산
                profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                
            update_trade_status(
                current_trade_id,
                'CLOSED',
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            
            print(f"\n=== ETH Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("============================")
            
            summary = get_trade_summary(days=7)
            if summary:
                print("\n=== 7-Day Trading Summary ===")
                print(f"Total Trades: {summary['total_trades']}")
                print(f"Win/Loss: {summary['winning_trades']}/{summary['losing_trades']}")
                if summary['total_trades'] > 0:
                    win_rate = (summary['winning_trades'] / summary['total_trades']) * 100
                    print(f"Win Rate: {win_rate:.2f}%")
                print(f"Total P/L: ${summary['total_profit_loss']:,.2f}")
                print(f"Avg P/L %: {summary['avg_profit_loss_percentage']:.2f}%")
                print("=============================")

# ===== 메인 프로그램 시작 =====
print("\n=== Ethereum Day Trading Bot Started (v2.3 - Trend Analysis + Daily/Weekly) ===") 
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Strategy: Day Trading + Trend Analysis (5m/15m/1h/1d/1w analysis)")
print("AI Engine: Google Gemini 2.5 Flash")
print("Asset: Ethereum Futures")
print("Leverage Range: 5-50x (Dynamic)")
print("SL/TP Range: Dynamic based on ATR + Trend Strategy")
print("Trend Strategies: WITH_TREND (High Conviction) vs COUNTER_TREND (Scalping)")
print("Market Sentiment: Funding Rate, OI, L/S Ratio, Liquidations")
print("News Analysis: Real-time Ethereum news sentiment")
print("Emergency System: Sudden event detection & response")
print("Flash Trading: Rapid opportunity capture")
print("Momentum Indicators: RSI, MACD, Stochastic, Williams %R")
print("Trend Analysis: Daily & Weekly EMA-based trend detection")
print("Execution Frequency: Every 2 minutes (30s during events)")
print("Trading Hours: 24/7 Unlimited")
print("Min Margin: 100 USDT")
print("====================================================================\n")

# 데이터베이스 설정
setup_database()

# ===== 메인 트레이딩 루프 =====
while True:
    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current ETH Price: ${current_price:,.2f}")

        # ===== 1. 현재 포지션 확인 =====
        current_side = None
        amount = 0
        
        positions = exchange.fetch_positions([symbol])
        for position in positions:
            if position['symbol'] == 'ETH/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if amt > 0:
                    current_side = 'long'
                    amount = amt
                elif amt < 0:
                    current_side = 'short'
                    amount = abs(amt)
        
        current_trade = get_latest_open_trade()
        current_trade_id = current_trade['id'] if current_trade else None
        
        # ===== 2. 긴급 이벤트 감지 시스템 =====
        # 먼저 일반 모드로 시장 데이터 수집 (빠른 스캔)
        multi_tf_data = fetch_multi_timeframe_data(emergency_mode=False)
        
        # 갑작스러운 시장 이벤트 감지
        market_event = detect_sudden_market_event(current_price, multi_tf_data)
        
        # 이벤트가 감지되면 1분봉 포함 긴급 모드로 재수집
        if market_event.get("event_detected"):
            print(f"\n🚨 MARKET EVENT DETECTED - SWITCHING TO 1MIN MODE 🚨")
            multi_tf_data = fetch_multi_timeframe_data(emergency_mode=True)  # 1분봉 포함 재수집
            market_event = detect_sudden_market_event(current_price, multi_tf_data)  # 재분석
            
            print(f"Type: {market_event.get('event_type')}")
            print(f"Timeframe: {market_event.get('primary_timeframe', '5m').upper()}")
            print(f"Severity: {market_event.get('severity')}")
            print(f"Max Price Change: {market_event.get('price_change_max', 0):.2f}%")
            print(f"Cumulative Change: {market_event.get('price_change_cumulative', 0):.2f}%")
            print(f"Volume Spike: {market_event.get('volume_spike_ratio', 1):.1f}x")
            print(f"Consecutive Moves: {market_event.get('consecutive_moves', 0)}")
            if market_event.get('ultra_fast'):
                print("⚡ ULTRA-FAST MOVEMENT DETECTED (1min)")
            
            # 뉴스 체크 (이벤트 시에만)
            ethereum_news = fetch_ethereum_news()
            news_impact = check_high_impact_news_keywords(ethereum_news)
            
            if news_impact.get("high_impact"):
                print(f"🔥 HIGH IMPACT NEWS DETECTED: {news_impact.get('impact_count')} keywords found")
                for item in news_impact.get('found_keywords', [])[:2]:
                    print(f"   • {item['keyword'].upper()}: {item['title'][:60]}...")
        
        # ===== 3. 포지션이 있는 경우 처리 (긴급 대응 포함) =====
        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} ETH")
            
            if not current_trade:
                temp_trade_data = {
                    'action': current_side,
                    'entry_price': current_price,
                    'amount': amount,
                    'leverage': 1,
                    'sl_price': 0,
                    'tp_price': 0,
                    'position_size_percentage': 0,
                    'investment_amount': 0
                }
                current_trade_id = save_trade(temp_trade_data)
                current_trade = get_latest_open_trade()  # 업데이트된 정보 가져오기
                print("새로운 거래 기록 생성 (기존 포지션)")
            
            # 🚨 긴급 이벤트 감지 시 포지션 조정
            if market_event.get("event_detected"):
                market_sentiment = fetch_market_sentiment()
                news_impact = check_high_impact_news_keywords(ethereum_news if 'ethereum_news' in locals() else [])
                
                emergency_handled = handle_emergency_position_adjustment(
                    current_price, market_event, current_trade, news_impact
                )
                
                if emergency_handled:
                    print("Emergency action completed. Continuing monitoring...")
                    time.sleep(30)  # 30초 후 재분석
                    continue
        
        # ===== 4. 포지션이 없는 경우 처리 =====
        else:
            if current_trade:
                handle_position_closure(current_price, current_trade['action'], current_trade['amount'], current_trade_id)
            
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)
            
            # 🚨 긴급 이벤트 감지 시 플래시 기회 포착
            if market_event.get("event_detected"):
                market_sentiment = fetch_market_sentiment()
                ethereum_news = fetch_ethereum_news() if 'ethereum_news' not in locals() else ethereum_news
                
                flash_opportunity = detect_flash_opportunity(current_price, market_event, market_sentiment)
                
                if flash_opportunity:
                    print(f"\n⚡ FLASH OPPORTUNITY DETECTED! ⚡")
                    print(f"Direction: {flash_opportunity['direction']}")
                    print(f"Reasoning: {flash_opportunity['reasoning']}")
                    
                    # 1분봉 기반 초고속 거래인지 확인
                    is_ultra_fast = market_event.get('ultra_fast', False) and market_event.get('primary_timeframe') == '1m'
                    
                    # 플래시 거래 실행
                    try:
                        balance = exchange.fetch_balance()
                        available_capital = balance['USDT']['free']
                        
                        position_size = flash_opportunity['position_size']
                        leverage = flash_opportunity['leverage']
                        investment_amount = available_capital * position_size
                        
                        if investment_amount >= 100:  # 최소 투자금액 확인
                            if is_ultra_fast:
                                print(f"⚡ ULTRA-FAST 1MIN SCALP: {flash_opportunity['direction']} with {investment_amount:.0f} USDT")
                                # 초고속 거래용 매우 타이트한 SL/TP (원금대비 %로 유지)
                                sl_percentage = 0.10
                                tp_percentage = 0.15
                            else:
                                print(f"🚀 Executing flash trade: {flash_opportunity['direction']} with {investment_amount:.0f} USDT")
                                # 일반 플래시 거래용 SL/TP (원금대비 %로 유지)
                                sl_percentage = 0.12
                                tp_percentage = 0.20
                            
                            total_position_value = investment_amount * leverage
                            amount = math.ceil((total_position_value / current_price) * 10000) / 10000
                            
                            exchange.set_leverage(leverage, symbol)
                            
                            sl_price_change_ratio = sl_percentage / leverage
                            tp_price_change_ratio = tp_percentage / leverage
                            
                            if flash_opportunity['direction'] == 'LONG':
                                order = exchange.create_market_buy_order(symbol, amount)
                                sl_price = round(current_price * (1 - sl_price_change_ratio), 2)
                                tp_price = round(current_price * (1 + tp_price_change_ratio), 2)
                                
                                exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                                
                            else:  # SHORT
                                order = exchange.create_market_sell_order(symbol, amount)
                                sl_price = round(current_price * (1 + sl_price_change_ratio), 2)
                                tp_price = round(current_price * (1 - tp_price_change_ratio), 2)
                                
                                exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                            
                            # 플래시 거래 기록
                            trade_data = {
                                'action': flash_opportunity['direction'].lower(),
                                'entry_price': current_price,
                                'amount': amount,
                                'leverage': leverage,
                                'sl_price': sl_price,
                                'tp_price': tp_price,
                                'position_size_percentage': position_size,
                                'investment_amount': investment_amount,
                                'trend_strategy': 'FLASH_TRADE',
                                'conviction_level': 'HIGH' if is_ultra_fast else 'MEDIUM'
                            }
                            trade_id = save_trade(trade_data)
                            
                            if is_ultra_fast:
                                print(f"⚡ ULTRA-FAST 1MIN {flash_opportunity['direction']} EXECUTED!")
                                print(f"Entry: ${current_price:,.2f}")
                                print(f"SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")
                                print(f"Ultra-tight scalp - 1min monitoring")
                                time.sleep(15)
                            else:
                                print(f"⚡ FLASH {flash_opportunity['direction']} EXECUTED!")
                                print(f"Entry: ${current_price:,.2f}")
                                print(f"SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")
                                print(f"Quick scalp target - monitoring closely...")
                                time.sleep(30)
                            
                            continue
                            
                    except Exception as e:
                        print(f"Flash trade execution error: {e}")
                
                time.sleep(60)
                continue
                
            time.sleep(5)
            print("No position. Analyzing market for ethereum day trading opportunities (with trend analysis)...")

            # ===== 5. 일반적인 이더리움 데이트레이딩 분석 (추세 분석 포함) =====
            multi_tf_data = fetch_multi_timeframe_data()
            market_sentiment = fetch_market_sentiment()
            
            # ===== 6. 이더리움 뉴스 분석 추가 =====
            print("Fetching latest Ethereum news...")
            ethereum_news = fetch_ethereum_news()
            news_sentiment = analyze_news_sentiment(ethereum_news)
            
            if ethereum_news:
                print(f"News Sentiment: {news_sentiment['sentiment']} "
                      f"(Bullish: {news_sentiment['bullish_count']}, "
                      f"Bearish: {news_sentiment['bearish_count']}, "
                      f"Strength: {news_sentiment['sentiment_strength']:.2f})")
                
                news_sentiment_with_titles = news_sentiment.copy()
                news_sentiment_with_titles['news_titles'] = [news['title'] for news in ethereum_news]
                save_news_analysis(news_sentiment_with_titles)
            else:
                print("No news data available - proceeding with technical analysis only")
            
            historical_trading_data = get_historical_trading_data(limit=3)
            performance_metrics = get_performance_metrics()
            
            # ===== 7. 추세 분석 및 전략 결정 =====
            # 일봉과 주봉 추세 분석 데이터 추출 (안전한 처리)
            daily_trend = {}
            weekly_trend = {}
            
            if '1d' in multi_tf_data and 'trend_analysis' in multi_tf_data['1d']['current_indicators']:
                daily_trend = multi_tf_data['1d']['current_indicators']['trend_analysis']
            else:
                daily_trend = {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
            
            if '1w' in multi_tf_data and 'trend_analysis' in multi_tf_data['1w']['current_indicators']:
                weekly_trend = multi_tf_data['1w']['current_indicators']['trend_analysis']
            else:
                weekly_trend = {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
            
            print(f"\n=== TREND ANALYSIS ===")
            print(f"Daily Trend: {daily_trend.get('trend_direction', 'UNKNOWN')}")
            print(f"Weekly Trend: {weekly_trend.get('trend_direction', 'UNKNOWN')}")
            if daily_trend.get('ema_distance_pct') is not None:
                print(f"Daily EMA Distance: {daily_trend['ema_distance_pct']:.2f}%")
            if weekly_trend.get('ema_distance_pct') is not None:
                print(f"Weekly EMA Distance: {weekly_trend['ema_distance_pct']:.2f}%")
            print("=====================")
            
            # ===== 8. Gemini 분석을 위한 데이터 준비 =====
            # 15분봉 ATR 값을 AI 분석에 포함 
            main_tf_atr = multi_tf_data.get('15m', {}).get('current_indicators', {}).get('atr', 0)

            market_analysis = {
                "current_price": current_price,
                "current_atr_15m": main_tf_atr,
                "timeframes": multi_tf_data,
                "market_sentiment": market_sentiment,
                "news_analysis": {
                    "sentiment": news_sentiment['sentiment'],
                    "bullish_count": news_sentiment['bullish_count'],
                    "bearish_count": news_sentiment['bearish_count'],
                    "sentiment_strength": news_sentiment['sentiment_strength'],
                    "recent_headlines": [news['title'] for news in ethereum_news[:5]]
                },
                "trend_analysis": {
                    "daily_trend": daily_trend,
                    "weekly_trend": weekly_trend
                },
                "recent_trades": historical_trading_data,
                "performance_summary": {
                    "total_trades": performance_metrics["total_trades"],
                    "win_rate": performance_metrics["win_rate"],
                    "avg_profit_loss_pct": performance_metrics["avg_profit_loss_percentage"]
                }
            }
            
            # ===== 9. Gemini AI 트레이딩 결정 요청 (추세 분석 특화) =====
            system_prompt = """
You are an expert Ethereum day trader with advanced trend analysis capabilities. You MUST respond with ONLY a valid JSON object, no other text.

CRITICAL: Your response must be ONLY valid JSON. No explanations, no markdown, no code blocks, no additional text.

ANALYSIS PROCESS:
1. TREND ANALYSIS FIRST: Check daily and weekly trend data to determine long-term direction
2. TREND STRATEGY SELECTION:
   - WITH_TREND: Short-term signal matches long-term trend → High conviction, bigger position, wider TP
   - COUNTER_TREND: Short-term signal opposes long-term trend → Low conviction scalping, smaller position, tight TP/SL
   - NEUTRAL: Unclear trends → Medium conviction, standard approach
3. MARKET EVENTS: Check if sudden event detected - adjust strategy accordingly
4. TIMEFRAME PRIORITY: If 1m data available (emergency mode), prioritize 1m signals for ultra-fast response
5. NEWS SENTIMENT: BULLISH news supports LONG, BEARISH supports SHORT
6. MOMENTUM: Use RSI, MACD, Stochastic for entry timing
7. MARKET SENTIMENT: Funding rate, OI, L/S ratio for contrarian signals
8. ETH VOLATILITY: Can move 5-10% in hours, 1-3% in minutes during events

TREND STRATEGY RULES:
A) WITH_TREND (追势交易):
   - When short-term signal aligns with daily/weekly trend
   - High conviction: 0.4-0.7 margin, 20-40x leverage
   - Wider TP: 3.5-5.0 * ATR (let profits run with trend)
   - Standard SL: 1.5-2.0 * ATR

B) COUNTER_TREND (逆势交易/短线回调):
   - When short-term signal opposes daily/weekly trend
   - Low conviction scalping: 0.1-0.25 margin, 8-15x leverage  
   - Tight TP: 1.5-2.5 * ATR (quick profits against the tide)
   - Tight SL: 1.0-1.5 * ATR (fast exit if wrong)

C) NEUTRAL (中性):
   - Unclear or sideways trends
   - Medium conviction: 0.2-0.4 margin, 10-25x leverage
   - Standard TP: 2.5-3.5 * ATR
   - Standard SL: 1.2-1.8 * ATR

POSITION SIZING MODIFIERS:
- Strong trends (STRONG_UPTREND/STRONG_DOWNTREND): +0.1 to position size
- Event detected: +0.1 to position size if signals align
- High news sentiment strength (>0.7): +0.05 to position size

SL/TP (ATR-based + Trend-adjusted):
- Use the provided 'current_atr_15m' as baseline volatility measure
- Adjust multipliers based on trend strategy as outlined above
- Ensure at least 1:1.5 Risk/Reward ratio for counter-trend, 1:2+ for with-trend

RESPOND WITH ONLY THIS JSON FORMAT:
{
  "direction": "LONG",
  "recommended_position_size": 0.3,
  "recommended_leverage": 20,
  "sl_atr_multiplier": 1.5,
  "tp_atr_multiplier": 3.0,
  "trend_strategy": "WITH_TREND",
  "conviction_level": "HIGH", 
  "reasoning": "Brief analysis mentioning trend alignment and ATR-based levels."
}

Valid trend_strategy values: "WITH_TREND", "COUNTER_TREND", "NEUTRAL"
Valid conviction_level values: "VERY_HIGH", "HIGH", "MEDIUM", "LOW"
"""
            
            try:
                market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
                
                response = model.generate_content([
                    system_prompt,
                    f"Ethereum Market Analysis Data (with Daily/Weekly Trends): {market_analysis_json}"
                ])
                
                response_content = response.text.strip()
                print(f"Raw Gemini response (first 500 chars): {response_content[:500]}...")
                
                def extract_json_from_response(text):
                    import re
                    if "```" in text:
                        start_markers = ["```json", "```"]
                        start_pos = -1
                        for marker in start_markers:
                            pos = text.find(marker)
                            if pos != -1:
                                start_pos = pos + len(marker)
                                break
                        if start_pos != -1:
                            end_pos = text.rfind("```")
                            if end_pos > start_pos:
                                text = text[start_pos:end_pos]
                    
                    brace_count = 0
                    start_idx = -1
                    end_idx = -1
                    for i, char in enumerate(text):
                        if char == '{':
                            if brace_count == 0:
                                start_idx = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_idx != -1:
                                end_idx = i + 1
                                break
                    if start_idx != -1 and end_idx != -1:
                        json_text = text[start_idx:end_idx]
                        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
                        return json_text.strip()
                    return text.strip()

                cleaned_response = extract_json_from_response(response_content)
                print(f"Cleaned JSON: {cleaned_response[:300]}...")

                trading_decision = json.loads(cleaned_response)

                print(f"\nAI 거래 결정 (Ethereum Gemini + Trend Analysis + News + Events + ATR):")
                print(f"방향: {trading_decision['direction']}")
                print(f"추세 전략: {trading_decision.get('trend_strategy', 'NEUTRAL')}")
                print(f"확신 수준: {trading_decision.get('conviction_level', 'MEDIUM')}")
                print(f"추천 포지션 크기: {trading_decision['recommended_position_size']*100:.1f}%")
                print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                print(f"스탑로스 ATR 배수: {trading_decision.get('sl_atr_multiplier', 'N/A')}")
                print(f"테이크프로핏 ATR 배수: {trading_decision.get('tp_atr_multiplier', 'N/A')}")
                print(f"분석 근거: {trading_decision['reasoning']}")
                
                if market_event.get("event_detected"):
                    print(f"⚠️ Event Influence: {market_event.get('event_type')} considered in decision")
                
                # ===== 10. 추세 전략에 따른 최종 조정 =====
                # AI 결과를 바탕으로 추세 전략 결정 함수 호출 (필요시 추가 조정용)
                trend_strategy_data = determine_trend_strategy(
                    trading_decision['direction'], daily_trend, weekly_trend
                )
                
                print(f"\n=== TREND STRATEGY CONFIRMATION ===")
                print(f"Strategy Type: {trend_strategy_data['strategy_type']}")
                print(f"Conviction Level: {trend_strategy_data['conviction_level']}")
                print(f"Trend Alignment: {trend_strategy_data['trend_alignment']}")
                print(f"Primary Timeframe: {trend_strategy_data['primary_timeframe']}")
                print(f"Long-term Trend: {trend_strategy_data['long_term_trend']}")
                print("===================================")
                
                analysis_data = {
                    'current_price': current_price,
                    'direction': trading_decision['direction'],
                    'recommended_position_size': trading_decision['recommended_position_size'],
                    'recommended_leverage': trading_decision['recommended_leverage'],
                    'sl_atr_multiplier': trading_decision.get('sl_atr_multiplier'),
                    'tp_atr_multiplier': trading_decision.get('tp_atr_multiplier'),
                    'reasoning': trading_decision['reasoning'],
                    'news_sentiment': news_sentiment['sentiment'],
                    'trend_strategy': trading_decision.get('trend_strategy', 'NEUTRAL'),
                    'conviction_level': trading_decision.get('conviction_level', 'MEDIUM'),
                    'daily_trend': daily_trend.get('trend_direction', 'UNKNOWN'),
                    'weekly_trend': weekly_trend.get('trend_direction', 'UNKNOWN')
                }
                analysis_id = save_ai_analysis(analysis_data)
                
                action = trading_decision['direction'].lower()
                
                if action == "no_position":
                    print("현재 ETH 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    print(f"이유: {trading_decision['reasoning']}")
                    time.sleep(120)
                    continue
                    
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']
                
                position_size_percentage = trading_decision['recommended_position_size']
                recommended_leverage = trading_decision['recommended_leverage']
                
                # 추세 전략에 따른 포지션 사이즈 조정
                trend_multiplier = trend_strategy_data.get('position_size_multiplier', 1.0)
                final_position_size = position_size_percentage * trend_multiplier
                final_position_size = min(final_position_size, 0.8)  # 최대 80% 제한
                
                investment_amount = available_capital * final_position_size
                
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"최소 투입 마진(100 USDT)으로 조정됨")

                print(f"투입 마진: {investment_amount:.2f} USDT (Trend Adjusted: {trend_multiplier:.2f}x)")
                
                total_position_value = investment_amount * recommended_leverage
                print(f"총 포지션 가치: {total_position_value:.2f} USDT")

                amount = math.ceil((total_position_value / current_price) * 10000) / 10000
                if amount <= 0:
                    print("계산된 주문 수량이 0보다 작거나 같아 주문을 진행하지 않습니다.")
                    time.sleep(60)
                    continue
                print(f"주문 수량: {amount} ETH")

                exchange.set_leverage(recommended_leverage, symbol)
                print(f"레버리지 설정: {recommended_leverage}x")

                # AI가 반환한 ATR 배수 사용 (추세 전략에 따라 조정)
                base_sl_atr_multiplier = trading_decision.get('sl_atr_multiplier', 1.5)
                base_tp_atr_multiplier = trading_decision.get('tp_atr_multiplier', 3.0)
                
                # 추세 전략에 따른 SL/TP 조정
                final_sl_atr_multiplier = base_sl_atr_multiplier * trend_strategy_data.get('sl_atr_multiplier', 1.0) / 1.5
                final_tp_atr_multiplier = base_tp_atr_multiplier * trend_strategy_data.get('tp_atr_multiplier', 1.0) / 2.5
                
                if main_tf_atr <= 0:
                    print("ATR 값이 유효하지 않아 이번 사이클은 건너뜁니다.")
                    time.sleep(60)
                    continue

                if action == "long":
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    
                    # ATR 기반 + 추세 조정된 SL/TP 가격 계산
                    sl_price = round(entry_price - (final_sl_atr_multiplier * main_tf_atr), 2)
                    tp_price = round(entry_price + (final_tp_atr_multiplier * main_tf_atr), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'long',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'position_size_percentage': final_position_size,
                        'investment_amount': investment_amount,
                        'trend_strategy': trading_decision.get('trend_strategy', 'NEUTRAL'),
                        'conviction_level': trading_decision.get('conviction_level', 'MEDIUM')
                    }
                    trade_id = save_trade(trade_data)
                    
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    # 결과 출력문
                    print(f"\n=== ETH LONG Position Opened (Trend-Adjusted ATR Dynamic) ===")
                    print(f"Entry: ${entry_price:,.2f} (15m ATR: ${main_tf_atr:.2f})")
                    print(f"Stop Loss: ${sl_price:,.2f} (Entry - {final_sl_atr_multiplier:.2f} * ATR)")
                    print(f"Take Profit: ${tp_price:,.2f} (Entry + {final_tp_atr_multiplier:.2f} * ATR)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"Trend Strategy: {trading_decision.get('trend_strategy', 'NEUTRAL')}")
                    print(f"Conviction Level: {trading_decision.get('conviction_level', 'MEDIUM')}")
                    print(f"Daily Trend: {daily_trend.get('trend_direction', 'UNKNOWN')}")
                    print(f"Weekly Trend: {weekly_trend.get('trend_direction', 'UNKNOWN')}")
                    print(f"News Sentiment: {news_sentiment['sentiment']}")
                    if market_event.get("event_detected"):
                        print(f"Event Context: {market_event.get('event_type')}")
                    print("=============================================================")

                elif action == "short":
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    
                    # ATR 기반 + 추세 조정된 SL/TP 가격 계산
                    sl_price = round(entry_price + (final_sl_atr_multiplier * main_tf_atr), 2)
                    tp_price = round(entry_price - (final_tp_atr_multiplier * main_tf_atr), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'short',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'position_size_percentage': final_position_size,
                        'investment_amount': investment_amount,
                        'trend_strategy': trading_decision.get('trend_strategy', 'NEUTRAL'),
                        'conviction_level': trading_decision.get('conviction_level', 'MEDIUM')
                    }
                    trade_id = save_trade(trade_data)
                    
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    # 결과 출력문
                    print(f"\n=== ETH SHORT Position Opened (Trend-Adjusted ATR Dynamic) ===")
                    print(f"Entry: ${entry_price:,.2f} (15m ATR: ${main_tf_atr:.2f})")
                    print(f"Stop Loss: ${sl_price:,.2f} (Entry + {final_sl_atr_multiplier:.2f} * ATR)")
                    print(f"Take Profit: ${tp_price:,.2f} (Entry - {final_tp_atr_multiplier:.2f} * ATR)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"Trend Strategy: {trading_decision.get('trend_strategy', 'NEUTRAL')}")
                    print(f"Conviction Level: {trading_decision.get('conviction_level', 'MEDIUM')}")
                    print(f"Daily Trend: {daily_trend.get('trend_direction', 'UNKNOWN')}")
                    print(f"Weekly Trend: {weekly_trend.get('trend_direction', 'UNKNOWN')}")
                    print(f"News Sentiment: {news_sentiment['sentiment']}")
                    if market_event.get("event_detected"):
                        print(f"Event Context: {market_event.get('event_type')}")
                    print("==============================================================")
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"Raw Gemini 응답: {response_content[:1000]}...")
                print("안전상 이번 사이클은 건너뜁니다.")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Gemini API 오류: {e}")
                time.sleep(30)
                continue

        # ===== 11. 대기 시간 (추세 분석 최적화) =====
        if current_side:
            if market_event.get("event_detected"):
                if market_event.get('ultra_fast') or market_event.get('primary_timeframe') == '1m':
                    time.sleep(15)
                    print("⚡ Ultra-fast monitoring (15s intervals)")
                else:
                    time.sleep(30)
                    print("🚨 Event monitoring (30s intervals)")
            else:
                # 추세 순응 거래는 더 긴 시간 대기 (추세가 이어질 시간 제공)
                current_trade = get_latest_open_trade()
                if current_trade:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("SELECT trend_strategy FROM trades WHERE id = ?", (current_trade['id'],))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result and result[0] == 'WITH_TREND':
                        time.sleep(90)  # 추세 순응 거래는 90초 대기
                        print("📈 Trend-following position monitoring (90s intervals)")
                    elif result and result[0] == 'COUNTER_TREND':
                        time.sleep(45)  # 역추세 거래는 45초 대기 (빠른 모니터링)
                        print("🔄 Counter-trend scalping monitoring (45s intervals)")
                    else:
                        time.sleep(60)  # 기본 60초
                else:
                    time.sleep(60)
        else:
            if market_event.get("event_detected"):
                if market_event.get('ultra_fast'):
                    time.sleep(30)
                    print("⚡ Ultra-fast opportunity scanning (30s)")
                else:
                    time.sleep(60)
                    print("🚨 Event opportunity scanning (60s)")
            else:
                time.sleep(120)  # 포지션이 없을 때는 2분 대기

    except Exception as e:
        print(f"\nMain Loop Error: {e}")
        time.sleep(10)