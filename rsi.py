"""
AI 이더리움 데이트레이딩 봇 - 전략 1: RSI + 볼린저 밴드 + 반전 캔들 (v2.3)
--------------------------------------------------------
전략 설명:
- RSI로 과매수/과매도 구간 필터링
- 볼린저 밴드로 극단적 가격 확인
- 반전 캔들 패턴으로 정확한 진입 타이밍 포착
- 3분봉 기준 분석
--------------------------------------------------------
"""
# ===== 필요한 라이브러리 임포트 =====
import ccxt
import os
import math
import time
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from datetime import datetime

# ===== 볼린저 밴드 계산 함수 =====
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """볼린저 밴드 계산"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

# ===== 캔들 패턴 인식 함수들 =====
def is_bullish_engulfing(current_candle, previous_candle):
    """상승 장악형 캔들 패턴 확인"""
    # 이전 캔들은 음봉, 현재 캔들은 양봉
    prev_is_bearish = previous_candle['close'] < previous_candle['open']
    curr_is_bullish = current_candle['close'] > current_candle['open']
    
    # 현재 양봉이 이전 음봉을 완전히 장악
    engulfs = (current_candle['open'] < previous_candle['close'] and 
               current_candle['close'] > previous_candle['open'])
    
    return prev_is_bearish and curr_is_bullish and engulfs

def is_bearish_engulfing(current_candle, previous_candle):
    """하락 장악형 캔들 패턴 확인"""
    # 이전 캔들은 양봉, 현재 캔들은 음봉
    prev_is_bullish = previous_candle['close'] > previous_candle['open']
    curr_is_bearish = current_candle['close'] < current_candle['open']
    
    # 현재 음봉이 이전 양봉을 완전히 장악
    engulfs = (current_candle['open'] > previous_candle['close'] and 
               current_candle['close'] < previous_candle['open'])
    
    return prev_is_bullish and curr_is_bearish and engulfs

def check_second_bottom_inside_bb(recent_candles, bb_lower, bb_upper):
    """두 번째 바닥이 볼린저 밴드 안에서 형성되는지 확인"""
    if len(recent_candles) < 5:
        return False
    
    # 최근 5개 캔들에서 저점들을 찾아 분석
    lows = [candle['low'] for candle in recent_candles[-5:]]
    
    # 첫 번째 저점(볼린저 밴드 하단 근처)과 두 번째 저점(밴드 안쪽) 비교
    first_low = min(lows[:3])  # 초기 3개 캔들에서 최저점
    second_low = min(lows[2:])  # 나중 3개 캔들에서 최저점
    
    # 두 번째 저점이 볼린저 밴드 안에 있고, 첫 번째 저점보다 높으면 true
    bb_lower_val = bb_lower.iloc[-1] if hasattr(bb_lower, 'iloc') else bb_lower
    bb_upper_val = bb_upper.iloc[-1] if hasattr(bb_upper, 'iloc') else bb_upper
    
    return (second_low > bb_lower_val and second_low < bb_upper_val and 
            second_low > first_low * 0.999)  # 약간의 여유

def check_second_top_inside_bb(recent_candles, bb_lower, bb_upper):
    """두 번째 고점이 볼린저 밴드 안에서 형성되는지 확인 (쌍봉 패턴)"""
    if len(recent_candles) < 5:
        return False
    
    # 최근 5개 캔들에서 고점들을 찾아 분석
    highs = [candle['high'] for candle in recent_candles[-5:]]
    
    # 첫 번째 고점(볼린저 밴드 상단 근처)과 두 번째 고점(밴드 안쪽) 비교
    first_high = max(highs[:3])  # 초기 3개 캔들에서 최고점
    second_high = max(highs[2:])  # 나중 3개 캔들에서 최고점
    
    # 두 번째 고점이 볼린저 밴드 안에 있고, 첫 번째 고점보다 낮으면 true
    bb_lower_val = bb_lower.iloc[-1] if hasattr(bb_lower, 'iloc') else bb_lower
    bb_upper_val = bb_upper.iloc[-1] if hasattr(bb_upper, 'iloc') else bb_upper
    
    return (second_high < bb_upper_val and second_high > bb_lower_val and 
            second_high < first_high * 1.001)  # 약간의 여유

# ===== 전략 1 시그널 확인 함수들 =====
def check_strategy1_long_signal(df_3m):
    """전략 1: RSI + 볼린저 밴드 + 반전 캔들 매수 신호 확인"""
    try:
        if len(df_3m) < 25:  # 충분한 데이터 확인
            return False, "Insufficient data for Strategy 1 analysis"
        
        # 최신 데이터 추출
        current_rsi = df_3m['RSI_14'].iloc[-1]
        current_candle = df_3m.iloc[-1].to_dict()
        previous_candle = df_3m.iloc[-2].to_dict()
        
        bb_upper = df_3m['BB_upper'].iloc[-1]
        bb_lower = df_3m['BB_lower'].iloc[-1]
        
        recent_candles = df_3m.tail(10).to_dict('records')
        
        # 조건 1: RSI가 30 이하 (과매도 구간)
        if current_rsi > 30:
            return False, f"RSI not oversold: {current_rsi:.2f} > 30"
        
        # 조건 2: 볼린저 밴드 하단 터치/이탈 확인
        bb_touch = (current_candle['low'] <= bb_lower or 
                   previous_candle['low'] <= bb_lower)
        
        if not bb_touch:
            return False, "Price not touching Bollinger Lower Band"
        
        # 조건 3: 상승 장악형 양봉 패턴 확인
        bullish_engulfing = is_bullish_engulfing(current_candle, previous_candle)
        
        if not bullish_engulfing:
            return False, "No bullish engulfing pattern detected"
        
        # 조건 4: 두 번째 바닥이 볼린저 밴드 안에서 형성
        second_bottom = check_second_bottom_inside_bb(
            recent_candles, bb_lower, bb_upper
        )
        
        if not second_bottom:
            return False, "Second bottom not formed inside Bollinger Bands"
        
        # 모든 조건 충족
        reasoning = (f"Strategy 1 LONG: RSI oversold ({current_rsi:.1f}), "
                    f"BB lower touch, bullish engulfing, second bottom inside BB")
        
        return True, reasoning
        
    except Exception as e:
        return False, f"Error in Strategy 1 long signal: {e}"

def check_strategy1_short_signal(df_3m):
    """전략 1: RSI + 볼린저 밴드 + 반전 캔들 매도 신호 확인"""
    try:
        if len(df_3m) < 25:
            return False, "Insufficient data for Strategy 1 analysis"
        
        # 최신 데이터 추출
        current_rsi = df_3m['RSI_14'].iloc[-1]
        current_candle = df_3m.iloc[-1].to_dict()
        previous_candle = df_3m.iloc[-2].to_dict()
        
        bb_upper = df_3m['BB_upper'].iloc[-1]
        bb_lower = df_3m['BB_lower'].iloc[-1]
        
        recent_candles = df_3m.tail(10).to_dict('records')
        
        # 조건 1: RSI가 70 이상 (과매수 구간)
        if current_rsi < 70:
            return False, f"RSI not overbought: {current_rsi:.2f} < 70"
        
        # 조건 2: 볼린저 밴드 상단 터치/돌파 확인
        bb_touch = (current_candle['high'] >= bb_upper or 
                   previous_candle['high'] >= bb_upper)
        
        if not bb_touch:
            return False, "Price not touching Bollinger Upper Band"
        
        # 조건 3: 하락 장악형 음봉 패턴 확인
        bearish_engulfing = is_bearish_engulfing(current_candle, previous_candle)
        
        if not bearish_engulfing:
            return False, "No bearish engulfing pattern detected"
        
        # 조건 4: 두 번째 고점이 볼린저 밴드 안에서 형성 (쌍봉)
        second_top = check_second_top_inside_bb(
            recent_candles, bb_lower, bb_upper
        )
        
        if not second_top:
            return False, "Second top not formed inside Bollinger Bands"
        
        # 모든 조건 충족
        reasoning = (f"Strategy 1 SHORT: RSI overbought ({current_rsi:.1f}), "
                    f"BB upper touch, bearish engulfing, second top inside BB")
        
        return True, reasoning
        
    except Exception as e:
        return False, f"Error in Strategy 1 short signal: {e}"

# ===== 기존 모멘텀 지표 계산 함수들 (유지) =====
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

# ===== 추세 분석 함수들 (기존 유지) =====
def analyze_trend_direction(df, short_ema=20, long_ema=50):
    """추세 방향 분석 (EMA 기반)"""
    try:
        if len(df) < long_ema:
            return {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
        
        df['EMA_short'] = calculate_ema(df['close'], short_ema)
        df['EMA_long'] = calculate_ema(df['close'], long_ema)
        
        df_clean = df.dropna()
        if len(df_clean) < 3:
            return {"trend_direction": "UNKNOWN", "ema_distance_pct": 0}
        
        current_short = df_clean['EMA_short'].iloc[-1]
        current_long = df_clean['EMA_long'].iloc[-1]
        current_price = df_clean['close'].iloc[-1]
        
        if current_long == 0:
            ema_distance_pct = 0
        else:
            ema_distance_pct = ((current_short - current_long) / current_long) * 100
        
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

# ===== 데이터 수집 함수 (3분봉 추가) =====
def fetch_multi_timeframe_data(emergency_mode=False):
    """3분봉 포함 멀티 타임프레임 데이터 수집"""
    if emergency_mode:
        timeframes = {
            "1m": {"timeframe": "1m", "limit": 60},
            "3m": {"timeframe": "3m", "limit": 100},  # 전략 1의 주요 타임프레임
            "5m": {"timeframe": "5m", "limit": 60},
            "15m": {"timeframe": "15m", "limit": 48},
            "1h": {"timeframe": "1h", "limit": 24},
            "1d": {"timeframe": "1d", "limit": 100},
            "1w": {"timeframe": "1w", "limit": 100}
        }
        print("🚨 EMERGENCY MODE: 1분봉 포함 + 3분봉 전략 1 분석")
    else:
        timeframes = {
            "3m": {"timeframe": "3m", "limit": 100},  # 전략 1의 주요 타임프레임
            "5m": {"timeframe": "5m", "limit": 100},
            "15m": {"timeframe": "15m", "limit": 96},
            "1h": {"timeframe": "1h", "limit": 48},
            "1d": {"timeframe": "1d", "limit": 100},
            "1w": {"timeframe": "1w", "limit": 100}
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
            
            # ATR
            df['ATR'] = calculate_atr(df)
            
            # ===== 볼린저 밴드 계산 (전략 1 전용) =====
            if tf_name == "3m":  # 3분봉에서만 볼린저 밴드 계산
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
                df['BB_upper'] = bb_upper
                df['BB_middle'] = bb_middle
                df['BB_lower'] = bb_lower
                print(f"Bollinger Bands added to {tf_name} data")
            
            # NaN 값 제거
            df.dropna(inplace=True)
            
            # 데이터 충분성 체크
            if len(df) < 10:
                print(f"Warning: {tf_name} has insufficient data ({len(df)} candles)")
                if tf_name in ['1d', '1w']:
                    if len(df) < 3:
                        continue
                else:
                    continue
            
            # 추세 분석 (일봉/주봉용)
            trend_analysis = None
            if tf_name in ['1d', '1w'] and len(df) >= 10:
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

                # 3분봉에서 볼린저 밴드 지표도 추가
                if tf_name == "3m" and 'BB_upper' in df.columns:
                    current_indicators.update({
                        "bb_upper": float(df['BB_upper'].iloc[-1]) if not pd.isna(df['BB_upper'].iloc[-1]) else current_indicators["current_price"] * 1.02,
                        "bb_middle": float(df['BB_middle'].iloc[-1]) if not pd.isna(df['BB_middle'].iloc[-1]) else current_indicators["current_price"],
                        "bb_lower": float(df['BB_lower'].iloc[-1]) if not pd.isna(df['BB_lower'].iloc[-1]) else current_indicators["current_price"] * 0.98,
                    })

                if trend_analysis:
                    current_indicators['trend_analysis'] = trend_analysis
                    
            except Exception as ind_error:
                print(f"Error extracting {tf_name} indicators: {ind_error}")
                continue
            
            # 최근 캔들 데이터 저장
            candle_count = 10 if emergency_mode and tf_name == "1m" else 10  # 패턴 분석을 위해 10개로 증가
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
                "recent_candles": recent_candles_dict,
                "dataframe": df  # 전략 분석을 위해 DataFrame도 저장
            }
            
            timeframe_display = tf_name.upper()
            if emergency_mode and tf_name == "1m":
                timeframe_display += " (EMERGENCY)"
            elif tf_name in ['1d', '1w']:
                timeframe_display += " (TREND)"
            elif tf_name == "3m":
                timeframe_display += " (STRATEGY 1 MAIN)"
            
            print(f"Collected {timeframe_display} data: {len(df)} candles")
            
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    
    return multi_tf_data

# ===== 시장 심리 지표 수집 함수들 (기존 유지) =====
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
            
            # 과거 데이터도 가져오기
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
            
            liq_ratio = total_long_liq / (total_short_liq + 0.0001)
            
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

# ===== 뉴스 관련 함수들 (기존 유지) =====
def fetch_ethereum_news():
    """최신 이더리움 뉴스 가져오기"""
    try:
        serp_api_key = os.getenv("SERP_API_KEY")
        
        if not serp_api_key:
            print("SERP API 키가 설정되지 않았습니다.")
            return []
        
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_news",
            "q": "ethereum ETH price trading",
            "gl": "us",
            "hl": "en",
            "api_key": serp_api_key,
            "num": 15
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            recent_news = []
            for news in news_results[:10]:
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
symbol = "ETH/USDT"

# Gemini API 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# SQLite 데이터베이스 설정
DB_FILE = "ethereum_strategy1_trading.db"

# ===== 데이터베이스 관련 함수들 (기존과 동일하지만 테이블명 변경) =====
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
        strategy_type TEXT DEFAULT 'STRATEGY_1'
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
        reasoning TEXT NOT NULL,
        strategy_signal TEXT,
        strategy_type TEXT DEFAULT 'STRATEGY_1',
        trade_id INTEGER,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Strategy 1 Database setup completed")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, current_price, direction, recommended_position_size, 
        recommended_leverage, reasoning, strategy_signal, strategy_type, trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        analysis_data.get('current_price', 0),
        analysis_data.get('direction', 'NO_POSITION'),
        analysis_data.get('recommended_position_size', 0),
        analysis_data.get('recommended_leverage', 0),
        analysis_data.get('reasoning', ''),
        analysis_data.get('strategy_signal', ''),
        'STRATEGY_1',
        trade_id
    ))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def save_trade(trade_data):
    """거래 정보를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp, action, entry_price, amount, leverage, sl_price, tp_price,
        position_size_percentage, investment_amount, strategy_type
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        'STRATEGY_1'
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
                profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
            else:
                profit_loss = (entry_price - current_price) * amount
                profit_loss_percentage = ((entry_price / current_price) - 1) * leverage * 100
                
            update_trade_status(
                current_trade_id,
                'CLOSED',
                exit_price=current_price,
                exit_timestamp=datetime.now().isoformat(),
                profit_loss=profit_loss,
                profit_loss_percentage=profit_loss_percentage
            )
            
            print(f"\n=== Strategy 1 ETH Position Closed ===")
            print(f"Entry: ${entry_price:,.2f}")
            print(f"Exit: ${current_price:,.2f}")
            print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
            print("============================")

# ===== 메인 프로그램 시작 =====
print("\n=== Ethereum Strategy 1 Trading Bot Started ===") 
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Strategy: RSI + Bollinger Bands + Reversal Candles")
print("Primary Timeframe: 3m")
print("AI Engine: Gemini 2.5 Flash (Position Sizing & Risk Management)")
print("Manual Entry Signals: Strategy 1 Logic")
print("========================================================\n")

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

        # ===== 2. 포지션이 있는 경우 모니터링 =====
        if current_side:
            print(f"Current Strategy 1 Position: {current_side.upper()} {amount} ETH")
            time.sleep(60)  # 1분 대기
            continue

        # ===== 3. 포지션이 없는 경우 - 전략 1 분석 =====
        else:
            if current_trade:
                handle_position_closure(current_price, current_trade['action'], current_trade['amount'], current_trade_id)
            
            # 기존 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders")
            except Exception as e:
                print("Error cancelling orders:", e)
            
            time.sleep(5)
            print("No position. Analyzing Strategy 1 signals...")

            # ===== 4. 멀티 타임프레임 데이터 수집 =====
            multi_tf_data = fetch_multi_timeframe_data()
            
            if '3m' not in multi_tf_data:
                print("Error: 3m data not available for Strategy 1")
                time.sleep(60)
                continue
            
            # 3분봉 DataFrame 추출
            df_3m = multi_tf_data['3m']['dataframe']
            
            # ===== 5. 전략 1 시그널 확인 =====
            long_signal, long_reasoning = check_strategy1_long_signal(df_3m)
            short_signal, short_reasoning = check_strategy1_short_signal(df_3m)
            
            print(f"\n=== Strategy 1 Signal Analysis ===")
            print(f"Long Signal: {long_signal}")
            if long_signal:
                print(f"Long Reasoning: {long_reasoning}")
            else:
                print(f"Long Check: {long_reasoning}")
                
            print(f"Short Signal: {short_signal}")
            if short_signal:
                print(f"Short Reasoning: {short_reasoning}")
            else:
                print(f"Short Check: {short_reasoning}")
            print("==================================")
            
            # 시그널이 없으면 대기
            if not long_signal and not short_signal:
                print("No Strategy 1 signals detected. Waiting...")
                time.sleep(120)  # 2분 대기
                continue
            
            # ===== 6. 시그널이 있으면 AI에게 포지션 사이징 요청 =====
            determined_direction = "LONG" if long_signal else "SHORT"
            signal_reasoning = long_reasoning if long_signal else short_reasoning
            
            # 시장 심리 및 뉴스 데이터 수집
            market_sentiment = fetch_market_sentiment()
            ethereum_news = fetch_ethereum_news()
            news_sentiment = analyze_news_sentiment(ethereum_news)
            
            # AI 분석을 위한 데이터 준비
            market_analysis = {
                "current_price": current_price,
                "strategy_1_signal": {
                    "direction": determined_direction,
                    "reasoning": signal_reasoning
                },
                "timeframes": multi_tf_data,
                "market_sentiment": market_sentiment,
                "news_analysis": {
                    "sentiment": news_sentiment['sentiment'],
                    "bullish_count": news_sentiment['bullish_count'],
                    "bearish_count": news_sentiment['bearish_count'],
                    "sentiment_strength": news_sentiment['sentiment_strength']
                }
            }
            
            # ===== 7. Gemini AI 포지션 사이징 요청 =====
            system_prompt = f"""
You are an expert position sizing and risk management AI for Ethereum futures trading. 

CRITICAL: Your response must be ONLY valid JSON. No explanations, no markdown, no code blocks.

The trading signal has already been determined by Strategy 1: {determined_direction}
Signal Reasoning: {signal_reasoning}

Your job is to determine:
1. Position size (0.1 to 0.8 of available capital)
2. Leverage (5x to 50x)
3. Stop loss and take profit levels based on volatility

Consider:
- Current ETH volatility and ATR
- Market sentiment (funding rates, long/short ratios)
- News sentiment
- Risk management principles

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "recommended_position_size": 0.3,
  "recommended_leverage": 20,
  "risk_assessment": "MEDIUM",
  "reasoning": "Brief explanation of sizing decision"
}}

Valid risk_assessment values: "LOW", "MEDIUM", "HIGH", "VERY_HIGH"
"""
            
            try:
                market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
                
                response = model.generate_content([
                    system_prompt,
                    f"Market Analysis Data: {market_analysis_json}"
                ])
                
                response_content = response.text.strip()
                
                # JSON 추출 함수
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
                ai_decision = json.loads(cleaned_response)

                print(f"\nAI Position Sizing Decision:")
                print(f"Direction: {determined_direction} (Strategy 1)")
                print(f"Position Size: {ai_decision['recommended_position_size']*100:.1f}%")
                print(f"Leverage: {ai_decision['recommended_leverage']}x")
                print(f"Risk Assessment: {ai_decision.get('risk_assessment', 'MEDIUM')}")
                print(f"AI Reasoning: {ai_decision['reasoning']}")
                
                # ===== 8. 거래 실행 =====
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']
                
                position_size_percentage = ai_decision['recommended_position_size']
                recommended_leverage = ai_decision['recommended_leverage']
                
                investment_amount = available_capital * position_size_percentage
                
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"Minimum investment adjusted to 100 USDT")

                print(f"Investment Amount: {investment_amount:.2f} USDT")
                
                total_position_value = investment_amount * recommended_leverage
                amount = math.ceil((total_position_value / current_price) * 10000) / 10000
                
                if amount <= 0:
                    print("Invalid amount calculated. Skipping trade.")
                    time.sleep(60)
                    continue
                    
                print(f"Order Amount: {amount} ETH")

                exchange.set_leverage(recommended_leverage, symbol)
                print(f"Leverage Set: {recommended_leverage}x")

                # ATR 기반 SL/TP 설정 (15분봉 ATR 사용)
                main_tf_atr = multi_tf_data.get('15m', {}).get('current_indicators', {}).get('atr', 0)
                if main_tf_atr <= 0:
                    main_tf_atr = current_price * 0.02  # 2% 기본값

                # 전략 1 특화 SL/TP (보수적)
                sl_atr_multiplier = 1.5
                tp_atr_multiplier = 2.5  # 1:1.67 위험보상비율

                if determined_direction == "LONG":
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    
                    sl_price = round(entry_price - (sl_atr_multiplier * main_tf_atr), 2)
                    tp_price = round(entry_price + (tp_atr_multiplier * main_tf_atr), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    print(f"\n=== Strategy 1 LONG Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f}")
                    print(f"Take Profit: ${tp_price:,.2f}")
                    print(f"Strategy Signal: {signal_reasoning}")
                    print("=====================================")

                elif determined_direction == "SHORT":
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    
                    sl_price = round(entry_price + (sl_atr_multiplier * main_tf_atr), 2)
                    tp_price = round(entry_price - (tp_atr_multiplier * main_tf_atr), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    print(f"\n=== Strategy 1 SHORT Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f}")
                    print(f"Take Profit: ${tp_price:,.2f}")
                    print(f"Strategy Signal: {signal_reasoning}")
                    print("=====================================")
                
                # 거래 데이터 저장
                trade_data = {
                    'action': determined_direction.lower(),
                    'entry_price': entry_price,
                    'amount': amount,
                    'leverage': recommended_leverage,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'position_size_percentage': position_size_percentage,
                    'investment_amount': investment_amount
                }
                trade_id = save_trade(trade_data)
                
                # AI 분석 데이터 저장
                analysis_data = {
                    'current_price': current_price,
                    'direction': determined_direction,
                    'recommended_position_size': position_size_percentage,
                    'recommended_leverage': recommended_leverage,
                    'reasoning': ai_decision['reasoning'],
                    'strategy_signal': signal_reasoning
                }
                analysis_id = save_ai_analysis(analysis_data, trade_id)
                    
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                print("Skipping this cycle for safety.")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"AI Analysis Error: {e}")
                time.sleep(30)
                continue

        # ===== 9. 대기 시간 =====
        if current_side:
            time.sleep(60)  # 포지션 있을 때 1분
        else:
            time.sleep(120)  # 포지션 없을 때 2분

    except Exception as e:
        print(f"\nMain Loop Error: {e}")
        time.sleep(10)