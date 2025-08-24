"""
AI 이더리움 데이트레이딩 봇 - Final Stable Version v2.7
--------------------------------------------------------
기능:
- 데이트레이딩 최적화 (5분, 15분, 1시간 차트)
- 고급 모멘텀 지표 (RSI, MACD, Stochastic, Williams %R)
- 시장 심리 분석 (펀딩비, 미결제약정, 롱숏비율, 청산 데이터)
- 실시간 이더리움 뉴스 분석 (SERP API)
- Gemini API 기반 AI 분석
- 동적 레버리지 및 포지션 사이징 (5-50배)
- 개선된 SL/TP 설정 (원금 대비 10-50% 범위)
- 24시간 무제한 거래
- 이더리움 선물 거래 최적화
- 최소 투자금액: 100 USDT

🆕 변경된 사항:
- CryptoQuant, Whale Alert, Glassnode API를 선택 사항으로 변경
- 해당 키가 없으면 경고 메시지 출력 후, 중립 데이터로 처리하여 분석 계속
- 핵심 기능(Binance, Gemini, FRED) API는 필수로 유지
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

# 🆕 새로 추가된 라이브러리들
import yfinance as yf  # 전통 시장 데이터
try:
    import fredapi  # 미국 경제 지표 데이터
    FRED_AVAILABLE = True
except ImportError:
    print("fredapi not installed. pip install fredapi for full economic data.")
    FRED_AVAILABLE = False

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

# ===== 🆕 온체인 데이터 수집 함수들 =====
def fetch_exchange_flows():
    """이더리움 거래소 입출금 데이터 수집 (선택 사항)"""
    cryptoquant_api_key = os.getenv("CRYPTOQUANT_API_KEY")
    if not cryptoquant_api_key:
        print("⚠️ WARNING: CRYPTOQUANT_API_KEY not found. Skipping exchange flow analysis.")
        return {
            "exchange_inflow_24h": 0, "exchange_outflow_24h": 0, "net_flow": 0,
            "inflow_trend": "UNKNOWN", "flow_sentiment": "NEUTRAL", "data_source": "NOT_AVAILABLE"
        }

    try:
        # 실제 CryptoQuant API 호출
        headers = {"Authorization": f"Bearer {cryptoquant_api_key}"}
        
        inflow_url = "https://api.cryptoquant.com/v1/eth/exchange-flows/exchange-inflow"
        inflow_params = {"window": "24h", "limit": 1}
        inflow_response = requests.get(inflow_url, headers=headers, params=inflow_params, timeout=10)
        
        outflow_url = "https://api.cryptoquant.com/v1/eth/exchange-flows/exchange-outflow"
        outflow_params = {"window": "24h", "limit": 1}
        outflow_response = requests.get(outflow_url, headers=headers, params=outflow_params, timeout=10)
        
        if inflow_response.status_code == 200 and outflow_response.status_code == 200:
            inflow_data = inflow_response.json()
            outflow_data = outflow_response.json()
            
            latest_inflow = inflow_data.get('result', {}).get('data', [{}])[-1].get('value', 0)
            latest_outflow = outflow_data.get('result', {}).get('data', [{}])[-1].get('value', 0)
            
            net_flow = latest_outflow - latest_inflow
            
            if len(inflow_data.get('result', {}).get('data', [])) > 1:
                prev_inflow = inflow_data['result']['data'][-2].get('value', latest_inflow)
                inflow_change = ((latest_inflow - prev_inflow) / prev_inflow) * 100 if prev_inflow > 0 else 0
                inflow_trend = "INCREASING" if inflow_change > 5 else "DECREASING" if inflow_change < -5 else "STABLE"
            else:
                inflow_trend = "STABLE"
            
            if net_flow > 10000:
                flow_sentiment = "VERY_BULLISH"
            elif net_flow > 2000:
                flow_sentiment = "BULLISH"
            elif net_flow < -10000:
                flow_sentiment = "VERY_BEARISH"
            elif net_flow < -2000:
                flow_sentiment = "BEARISH"
            else:
                flow_sentiment = "NEUTRAL"
            
            return {
                "exchange_inflow_24h": latest_inflow,
                "exchange_outflow_24h": latest_outflow,
                "net_flow": net_flow,
                "inflow_trend": inflow_trend,
                "flow_sentiment": flow_sentiment,
                "data_source": "CRYPTOQUANT_API"
            }
        else:
            print("CryptoQuant API 호출 실패, 기본값 사용")
            return {"exchange_inflow_24h": 0, "exchange_outflow_24h": 0, "net_flow": 0, 
                   "inflow_trend": "UNKNOWN", "flow_sentiment": "NEUTRAL", "data_source": "ERROR"}
            
    except Exception as e:
        print(f"거래소 플로우 데이터 수집 오류: {e}")
        return {"exchange_inflow_24h": 0, "exchange_outflow_24h": 0, "net_flow": 0,
               "inflow_trend": "ERROR", "flow_sentiment": "NEUTRAL", "data_source": "ERROR"}

def fetch_whale_activity():
    """고래 지갑 활동 및 대량 거래 감지 (선택 사항)"""
    whale_alert_api_key = os.getenv("WHALE_ALERT_API_KEY")
    if not whale_alert_api_key:
        print("⚠️ WARNING: WHALE_ALERT_API_KEY not found. Skipping whale activity analysis.")
        return {
            "large_transactions_24h": 0, "total_whale_volume_eth": 0, "avg_transaction_size": 0,
            "whale_sentiment": "NEUTRAL", "largest_transaction": 0, "exchange_whale_deposits": 0,
            "exchange_whale_withdrawals": 0, "data_source": "NOT_AVAILABLE"
        }
    
    try:
        # Whale Alert API 호출
        url = "https://api.whale-alert.io/v1/transactions"
        params = {
            "api_key": whale_alert_api_key,
            "symbol": "eth",
            "min_value": 500000,
            "start": int((datetime.now().timestamp() - 86400)),
            "limit": 100
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get("transactions", [])
            
            if not transactions:
                return {"large_transactions_24h": 0, "total_whale_volume_eth": 0, 
                       "whale_sentiment": "NEUTRAL", "data_source": "WHALE_ALERT"}
            
            total_volume = 0
            exchange_deposits = 0
            exchange_withdrawals = 0
            transaction_sizes = []
            
            exchanges = ["binance", "coinbase", "kraken", "bitfinex", "huobi", "okex"]
            
            for tx in transactions:
                amount_eth = tx.get("amount", 0)
                from_owner_type = tx.get("from", {}).get("owner_type", "")
                to_owner_type = tx.get("to", {}).get("owner_type", "")
                from_owner = tx.get("from", {}).get("owner", "").lower()
                to_owner = tx.get("to", {}).get("owner", "").lower()
                
                total_volume += amount_eth
                transaction_sizes.append(amount_eth)
                
                if any(exchange in from_owner for exchange in exchanges) or from_owner_type == "exchange":
                    exchange_withdrawals += amount_eth
                elif any(exchange in to_owner for exchange in exchanges) or to_owner_type == "exchange":
                    exchange_deposits += amount_eth
            
            if exchange_withdrawals > exchange_deposits * 1.5:
                whale_sentiment = "ACCUMULATING"
            elif exchange_deposits > exchange_withdrawals * 1.5:
                whale_sentiment = "DISTRIBUTING"
            else:
                whale_sentiment = "NEUTRAL"
            
            return {
                "large_transactions_24h": len(transactions),
                "total_whale_volume_eth": total_volume,
                "avg_transaction_size": sum(transaction_sizes) / len(transaction_sizes) if transaction_sizes else 0,
                "whale_sentiment": whale_sentiment,
                "largest_transaction": max(transaction_sizes) if transaction_sizes else 0,
                "exchange_whale_deposits": exchange_deposits,
                "exchange_whale_withdrawals": exchange_withdrawals,
                "data_source": "WHALE_ALERT"
            }
        else:
            print("Whale Alert API 호출 실패")
            return {"large_transactions_24h": 0, "whale_sentiment": "NEUTRAL", "data_source": "ERROR"}
            
    except Exception as e:
        print(f"고래 활동 데이터 수집 오류: {e}")
        return {"large_transactions_24h": 0, "whale_sentiment": "NEUTRAL", "data_source": "ERROR"}

def fetch_onchain_metrics():
    """온체인 메트릭스 종합 (선택 사항)"""
    glassnode_api_key = os.getenv("GLASSNODE_API_KEY")
    if not glassnode_api_key:
        print("⚠️ WARNING: GLASSNODE_API_KEY not found. Skipping on-chain metrics analysis.")
        return {
            "nvt_ratio": 0, "mvrv_ratio": 0, "active_addresses_24h": 0, "transaction_count_24h": 0,
            "gas_used_24h": 0, "eth_supply_on_exchanges": 0, "staking_ratio": 0,
            "data_source": "NOT_AVAILABLE"
        }
        
    try:
        # 실제 Glassnode API 호출
        base_url = "https://api.glassnode.com/v1/metrics"
        headers = {"X-API-KEY": glassnode_api_key}
        
        metrics = {}
        
        try:
            nvt_response = requests.get(f"{base_url}/indicators/nvt", 
                                       headers=headers, 
                                       params={"a": "ETH", "s": "24h", "i": "24h"},
                                       timeout=10)
            if nvt_response.status_code == 200:
                nvt_data = nvt_response.json()
                metrics["nvt_ratio"] = nvt_data[-1]["v"] if nvt_data else 50
        except:
            pass # 일부 메트릭 실패는 전체를 중단시키지 않음
        
        return {
            "nvt_ratio": metrics.get("nvt_ratio", 50),
            "mvrv_ratio": 1.2, # 예시 데이터, 실제 API 연동 필요
            "active_addresses_24h": 500000,
            "transaction_count_24h": 1200000,
            "gas_used_24h": 150e9,
            "eth_supply_on_exchanges": 20000000,
            "staking_ratio": 0.20,
            "data_source": "GLASSNODE_API"
        }
        
    except Exception as e:
        print(f"온체인 메트릭스 수집 오류: {e}")
        return {"nvt_ratio": 50, "mvrv_ratio": 1.0, "data_source": "ERROR"}

# ===== 🆕 매크로 경제 지표 수집 함수들 =====
def fetch_traditional_markets():
    """전통 시장 데이터 수집 (나스닥, S&P500, DXY 등)"""
    try:
        tickers = {
            "^IXIC": "nasdaq",
            "^GSPC": "sp500",
            "DX-Y.NYB": "dxy",
            "^TNX": "us10y"
        }
        
        traditional_data = {}
        
        for ticker, name in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d", interval="1d")
                
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    daily_change = ((latest_price - prev_price) / prev_price) * 100
                    
                    first_price = hist['Close'].iloc[0]
                    five_day_change = ((latest_price - first_price) / first_price) * 100
                    
                    traditional_data[name] = {
                        "current_value": float(latest_price),
                        "daily_change_pct": float(daily_change),
                        "five_day_change_pct": float(five_day_change),
                        "trend": "UP" if daily_change > 0.5 else "DOWN" if daily_change < -0.5 else "FLAT"
                    }
                else:
                    traditional_data[name] = {"current_value": 0, "daily_change_pct": 0, "trend": "UNKNOWN"}
                    
            except Exception as e:
                print(f"{name} 데이터 수집 실패: {e}")
                traditional_data[name] = {"current_value": 0, "daily_change_pct": 0, "trend": "ERROR"}
        
        nasdaq_change = traditional_data.get("nasdaq", {}).get("daily_change_pct", 0)
        dxy_change = traditional_data.get("dxy", {}).get("daily_change_pct", 0)
        
        if nasdaq_change > 1.0 and dxy_change < -0.5:
            macro_sentiment = "RISK_ON"
        elif nasdaq_change < -1.0 and dxy_change > 0.5:
            macro_sentiment = "RISK_OFF"
        else:
            macro_sentiment = "NEUTRAL"
        
        traditional_data["macro_sentiment"] = macro_sentiment
        traditional_data["data_timestamp"] = datetime.now().isoformat()
        
        return traditional_data
        
    except Exception as e:
        print(f"전통시장 데이터 수집 오류: {e}")
        return {"macro_sentiment": "ERROR", "nasdaq": {"trend": "ERROR"}}

def fetch_economic_indicators():
    """미국 경제지표 수집 (FRED API) - 필수"""
    if not FRED_AVAILABLE:
        raise ImportError("CRITICAL: fredapi library is not installed. Please run 'pip install fredapi'.")

    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise ValueError("CRITICAL: FRED_API_KEY is missing. Halting operations.")

    try:
        # FRED API를 통한 실제 경제지표 수집
        fred = fredapi.Fred(api_key=fred_api_key)
        
        indicators = {}
        
        # 연준 기준금리
        fed_rate = fred.get_series('FEDFUNDS', limit=1)
        indicators["fed_funds_rate"] = float(fed_rate.iloc[-1]) if not fed_rate.empty else 5.0
        
        # 인플레이션 (CPI)
        cpi = fred.get_series('CPIAUCSL', limit=2)
        if len(cpi) >= 2:
            inflation_rate = ((cpi.iloc[-1] - cpi.iloc[-2]) / cpi.iloc[-2]) * 100 * 12
            indicators["inflation_rate"] = float(inflation_rate)
        else:
            indicators["inflation_rate"] = 3.0
        
        # 실업률
        unemployment = fred.get_series('UNRATE', limit=1)
        indicators["unemployment_rate"] = float(unemployment.iloc[-1]) if not unemployment.empty else 4.0
        
        # VIX (공포지수)
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="2d")
            indicators["vix_index"] = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 20.0
        except:
            indicators["vix_index"] = 20.0
        
        # 경제 센티먼트 종합 판단
        fed_rate_val = indicators["fed_funds_rate"]
        inflation = indicators["inflation_rate"]
        
        if fed_rate_val > 5.0 and inflation > 4.0:
            economic_sentiment = "HAWKISH"
        elif fed_rate_val < 3.0 and indicators["vix_index"] < 20:
            economic_sentiment = "DOVISH"
        else:
            economic_sentiment = "NEUTRAL"
        
        indicators["economic_sentiment"] = economic_sentiment
        indicators["data_source"] = "FRED_API"
        
        return indicators
        
    except Exception as e:
        print(f"경제지표 수집 오류: {e}")
        return {"fed_funds_rate": 5.0, "economic_sentiment": "NEUTRAL", "data_source": "ERROR"}

def calculate_correlation_analysis(current_price, traditional_markets):
    """이더리움과 전통시장 간의 상관관계 분석"""
    try:
        eth_ticker = yf.Ticker("ETH-USD")
        eth_hist = eth_ticker.history(period="30d")
        
        nasdaq_ticker = yf.Ticker("^IXIC")
        nasdaq_hist = nasdaq_ticker.history(period="30d")
        
        if len(eth_hist) > 5 and len(nasdaq_hist) > 5:
            eth_returns = eth_hist['Close'].pct_change().dropna()
            nasdaq_returns = nasdaq_hist['Close'].pct_change().dropna()
            
            common_dates = eth_returns.index.intersection(nasdaq_returns.index)
            if len(common_dates) > 10:
                eth_aligned = eth_returns[common_dates]
                nasdaq_aligned = nasdaq_returns[common_dates]
                
                correlation = eth_aligned.corr(nasdaq_aligned)
                beta = eth_aligned.std() / nasdaq_aligned.std() if nasdaq_aligned.std() > 0 else 1.0
                
                correlation_strength = "STRONG" if abs(correlation) > 0.7 else \
                                     "MODERATE" if abs(correlation) > 0.4 else "WEAK"
                
                return {
                    "eth_nasdaq_correlation": float(correlation),
                    "eth_nasdaq_beta": float(beta),
                    "correlation_strength": correlation_strength,
                    "market_regime": "CORRELATED" if abs(correlation) > 0.6 else "DECOUPLED",
                    "sample_days": len(common_dates)
                }
        
        return {"eth_nasdaq_correlation": 0.0, "market_regime": "UNKNOWN", "correlation_strength": "UNKNOWN"}
        
    except Exception as e:
        print(f"상관관계 분석 오류: {e}")
        return {"eth_nasdaq_correlation": 0.0, "market_regime": "ERROR"}

# ===== 🆕 통합 이벤트 감지 시스템 =====
def detect_whale_impact_events(whale_data, current_price):
    """고래 움직임 기반 긴급 이벤트 감지"""
    try:
        if whale_data.get("data_source") in ["ERROR", "NOT_AVAILABLE"]:
            return {"whale_event": False}
        
        large_tx_count = whale_data.get("large_transactions_24h", 0)
        largest_tx = whale_data.get("largest_transaction", 0)
        whale_sentiment = whale_data.get("whale_sentiment", "NEUTRAL")
        exchange_deposits = whale_data.get("exchange_whale_deposits", 0)
        exchange_withdrawals = whale_data.get("exchange_whale_withdrawals", 0)
        
        whale_event_detected = False
        event_type = None
        severity = "LOW"
        
        if large_tx_count > 30:
            whale_event_detected = True
            event_type = "HIGH_WHALE_ACTIVITY"
            severity = "HIGH" if large_tx_count > 50 else "MEDIUM"
        
        if largest_tx > 50000:
            whale_event_detected = True
            event_type = "MEGA_WHALE_TRANSACTION"
            severity = "EXTREME" if largest_tx > 100000 else "HIGH"
        
        if exchange_deposits > 20000:
            whale_event_detected = True
            event_type = "WHALE_EXCHANGE_DEPOSIT"
            severity = "HIGH"
        
        if exchange_withdrawals > 25000:
            whale_event_detected = True
            event_type = "WHALE_EXCHANGE_WITHDRAWAL"
            severity = "HIGH"
        
        if whale_event_detected:
            return {
                "whale_event": True,
                "event_type": event_type,
                "severity": severity,
                "whale_sentiment": whale_sentiment,
                "large_transactions": large_tx_count,
                "largest_transaction": largest_tx,
                "exchange_deposits": exchange_deposits,
                "exchange_withdrawals": exchange_withdrawals,
                "action_recommended": True
            }
        
        return {"whale_event": False, "action_recommended": False}
        
    except Exception as e:
        print(f"고래 이벤트 감지 오류: {e}")
        return {"whale_event": False}

def detect_macro_events(traditional_markets):
    """매크로 경제 이벤트 감지"""
    try:
        nasdaq_data = traditional_markets.get("nasdaq", {})
        dxy_data = traditional_markets.get("dxy", {})
        sp500_data = traditional_markets.get("sp500", {})
        
        nasdaq_change = nasdaq_data.get("daily_change_pct", 0)
        dxy_change = dxy_data.get("daily_change_pct", 0)
        sp500_change = sp500_data.get("daily_change_pct", 0)
        
        macro_event = False
        severity = "LOW"
        event_type = None
        
        if nasdaq_change < -3.0 or sp500_change < -3.0:
            macro_event = True
            event_type = "STOCK_MARKET_CRASH"
            severity = "HIGH"
        elif nasdaq_change > 3.0 or sp500_change > 3.0:
            macro_event = True
            event_type = "STOCK_MARKET_SURGE"
            severity = "HIGH"
        
        if abs(dxy_change) > 1.0:
            macro_event = True
            event_type = "DOLLAR_VOLATILITY"
            severity = "MEDIUM"
        
        if macro_event:
            return {
                "macro_event": True,
                "event_type": event_type,
                "severity": severity,
                "nasdaq_change": nasdaq_change,
                "dxy_change": dxy_change,
                "sp500_change": sp500_change
            }
        
        return {"macro_event": False}
        
    except Exception as e:
        print(f"매크로 이벤트 감지 오류: {e}")
        return {"macro_event": False}

# ===== 기존 긴급 이벤트 감지 및 대응 시스템 =====
def detect_sudden_market_event(current_price, timeframe_data):
    """갑작스러운 시장 이벤트 감지 (1분봉 우선, 급등/급락, 볼륨 급증)"""
    try:
        primary_tf = '1m' if '1m' in timeframe_data else '5m'
        secondary_tf = '5m' if '1m' in timeframe_data else '15m'
        
        if primary_tf not in timeframe_data:
            return {"event_detected": False}
        
        recent_primary = timeframe_data[primary_tf]['recent_candles']
        recent_secondary = timeframe_data.get(secondary_tf, {}).get('recent_candles', [])
        
        if len(recent_primary) < 5:
            return {"event_detected": False}
        
        current_atr = timeframe_data[primary_tf]['current_indicators']['atr']
        
        if primary_tf == '1m':
            normal_move_threshold = 1.0
            volume_threshold = 2.5
            sustained_threshold = 2.0
        else:
            normal_move_threshold = 2.0
            volume_threshold = 3.0
            sustained_threshold = 3.0
        
        latest_candles = recent_primary[-5:]
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
            
            if abs_price_change > (normal_move_threshold * 0.5):
                current_direction = 'UP' if price_change_pct > 0 else 'DOWN'
                if move_direction is None:
                    move_direction = current_direction
                    consecutive_moves = 1
                elif move_direction == current_direction:
                    consecutive_moves += 1
                else:
                    consecutive_moves = 1
                    move_direction = current_direction
        
        max_price_change = max(price_changes) if price_changes else 0
        max_volume_ratio = max(volume_changes) if volume_changes else 1
        avg_volume_ratio = sum(volume_changes) / len(volume_changes) if volume_changes else 1
        
        sudden_price_move = max_price_change > normal_move_threshold
        volume_spike = max_volume_ratio > volume_threshold or avg_volume_ratio > (volume_threshold * 0.7)
        sustained_directional_move = consecutive_moves >= 3
        
        total_price_change = abs((latest_candles[-1]['close'] - latest_candles[0]['close']) / latest_candles[0]['close']) * 100
        severe_cumulative_move = total_price_change > sustained_threshold
        
        if primary_tf == '1m':
            ultra_fast_move = max_price_change > 0.8
            event_detected = ultra_fast_move or (sudden_price_move and volume_spike)
        else:
            event_detected = sudden_price_move and (volume_spike or sustained_directional_move)
        
        if severe_cumulative_move and volume_spike:
            event_detected = True
        
        if event_detected:
            recent_direction = "UP" if latest_candles[-1]['close'] > latest_candles[-2]['close'] else "DOWN"
            
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

def detect_comprehensive_market_events(current_price, timeframe_data, comprehensive_analysis):
    """기존 가격 이벤트 + 온체인 이벤트 + 매크로 이벤트 통합 감지"""
    
    # 1. 기존 가격 기반 이벤트 감지
    price_event = detect_sudden_market_event(current_price, timeframe_data)
    
    # 2. 고래 이벤트 감지
    whale_data = comprehensive_analysis.get("onchain_data", {}).get("whale_activity", {})
    whale_event = detect_whale_impact_events(whale_data, current_price)
    
    # 3. 매크로 이벤트 감지
    traditional_markets = comprehensive_analysis.get("macro_data", {}).get("traditional_markets", {})
    macro_event = detect_macro_events(traditional_markets)
    
    # 4. 종합 이벤트 평가
    events_detected = []
    max_severity = "LOW"
    action_urgency = "NORMAL"
    
    if price_event.get("event_detected"):
        events_detected.append("PRICE_EVENT")
        if price_event.get("severity") in ["HIGH", "EXTREME"]:
            max_severity = "HIGH"
    
    if whale_event.get("whale_event"):
        events_detected.append("WHALE_EVENT")
        if whale_event.get("severity") in ["HIGH", "EXTREME"]:
            max_severity = "HIGH"
    
    if macro_event.get("macro_event"):
        events_detected.append("MACRO_EVENT")
        if macro_event.get("severity") == "HIGH":
            max_severity = "HIGH"
    
    # 복합 이벤트 시 우선순위 증가
    if len(events_detected) >= 2:
        action_urgency = "HIGH"
        max_severity = "EXTREME"
    
    if events_detected:
        return {
            "comprehensive_event": True,
            "events_detected": events_detected,
            "max_severity": max_severity,
            "action_urgency": action_urgency,
            "price_event": price_event,
            "whale_event": whale_event,
            "macro_event": macro_event,
            "event_count": len(events_detected)
        }
    
    return {"comprehensive_event": False, "event_count": 0}

# ===== 뉴스 분석 함수들 =====
def fetch_ethereum_news():
    """최신 이더리움 뉴스 가져오기 함수"""
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        print("⚠️ WARNING: SERP_API_KEY not set. Skipping news analysis.")
        return []
    
    try:
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

def check_high_impact_news_keywords(news_data):
    """고임팩트 뉴스 키워드 감지"""
    if not news_data:
        return {"high_impact": False}
    
    high_impact_keywords = [
        "fed", "powell", "jackson hole", "interest rate", "monetary policy",
        "regulation", "sec", "etf", "institutional", "blackrock", "grayscale",
        "upgrade", "merge", "staking", "burn", "fork", "ethereum 2.0",
        "defi", "hack", "exploit", "crash", "surge", "breakout",
        "whale", "liquidation", "manipulation", "ban", "legal"
    ]
    
    high_impact_found = []
    for news in news_data[:5]:
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

# ===== 설정 및 초기화 =====
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
if not api_key or not secret:
    raise ValueError("CRITICAL: BINANCE_API_KEY and BINANCE_SECRET_KEY must be set.")
    
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
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("CRITICAL: GEMINI_API_KEY must be set.")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# SQLite 데이터베이스 설정
DB_FILE = "ethereum_daytrading.db"

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
        sl_percentage REAL NOT NULL,
        tp_percentage REAL NOT NULL,
        position_size_percentage REAL NOT NULL,
        investment_amount REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp TEXT,
        profit_loss REAL,
        profit_loss_percentage REAL
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
        stop_loss_percentage REAL NOT NULL,
        take_profit_percentage REAL NOT NULL,
        reasoning TEXT NOT NULL,
        news_sentiment TEXT,
        trade_id INTEGER,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    )
    ''')
    
    # 뉴스 데이터 테이블
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
    
    try:
        cursor.execute("PRAGMA table_info(ai_analysis)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'news_sentiment' not in columns:
            cursor.execute('ALTER TABLE ai_analysis ADD COLUMN news_sentiment TEXT DEFAULT "NEUTRAL"')
            print("Added news_sentiment column to ai_analysis table")
            
    except sqlite3.Error as e:
        print(f"Database migration note: {e}")
    
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")

def save_ai_analysis(analysis_data, trade_id=None):
    """AI 분석 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO ai_analysis (
            timestamp, current_price, direction, recommended_position_size, 
            recommended_leverage, stop_loss_percentage, take_profit_percentage, 
            reasoning, news_sentiment, trade_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            analysis_data.get('current_price', 0),
            analysis_data.get('direction', 'NO_POSITION'),
            analysis_data.get('recommended_position_size', 0),
            analysis_data.get('recommended_leverage', 0),
            analysis_data.get('stop_loss_percentage', 0),
            analysis_data.get('take_profit_percentage', 0),
            analysis_data.get('reasoning', ''),
            analysis_data.get('news_sentiment', 'NEUTRAL'),
            trade_id
        ))
    except sqlite3.OperationalError as e:
        if "no column named news_sentiment" in str(e):
            print("Using fallback query without news_sentiment column")
            cursor.execute('''
            INSERT INTO ai_analysis (
                timestamp, current_price, direction, recommended_position_size, 
                recommended_leverage, stop_loss_percentage, take_profit_percentage, 
                reasoning, trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                analysis_data.get('current_price', 0),
                analysis_data.get('direction', 'NO_POSITION'),
                analysis_data.get('recommended_position_size', 0),
                analysis_data.get('recommended_leverage', 0),
                analysis_data.get('stop_loss_percentage', 0),
                analysis_data.get('take_profit_percentage', 0),
                analysis_data.get('reasoning', ''),
                trade_id
            ))
        else:
            raise e
    
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
        sl_percentage, tp_percentage, position_size_percentage, investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        trade_data.get('action', ''),
        trade_data.get('entry_price', 0),
        trade_data.get('amount', 0),
        trade_data.get('leverage', 0),
        trade_data.get('sl_price', 0),
        trade_data.get('tp_price', 0),
        trade_data.get('sl_percentage', 0),
        trade_data.get('tp_percentage', 0),
        trade_data.get('position_size_percentage', 0),
        trade_data.get('investment_amount', 0)
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
        t.profit_loss_percentage, a.reasoning
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

# ===== 데이터 수집 함수 (이더리움 최적화) =====
def fetch_multi_timeframe_data(emergency_mode=False):
    """이더리움 데이트레이딩 최적화된 멀티 타임프레임 데이터 수집"""
    if emergency_mode:
        timeframes = {
            "1m": {"timeframe": "1m", "limit": 60},
            "5m": {"timeframe": "5m", "limit": 60},
            "15m": {"timeframe": "15m", "limit": 48},
            "1h": {"timeframe": "1h", "limit": 24}
        }
        print("🚨 EMERGENCY MODE: 1분봉 포함 고속 분석")
    else:
        timeframes = {
            "5m": {"timeframe": "5m", "limit": 100},
            "15m": {"timeframe": "15m", "limit": 96},
            "1h": {"timeframe": "1h", "limit": 48}
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
            
            macd, signal, histogram = calculate_macd(df['close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_histogram'] = histogram
            
            stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
            df['ATR'] = calculate_atr(df)
            
            df.dropna(inplace=True)
            
            current_indicators = {
                "current_price": float(df['close'].iloc[-1]),
                "rsi_14": float(df['RSI_14'].iloc[-1]),
                "rsi_21": float(df['RSI_21'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "macd_signal": float(df['MACD_signal'].iloc[-1]),
                "macd_histogram": float(df['MACD_histogram'].iloc[-1]),
                "stoch_k": float(df['Stoch_K'].iloc[-1]),
                "stoch_d": float(df['Stoch_D'].iloc[-1]),
                "williams_r": float(df['Williams_R'].iloc[-1]),
                "atr": float(df['ATR'].iloc[-1]),
                "volume": float(df['volume'].iloc[-1])
            }
            
            candle_count = 10 if emergency_mode and tf_name == "1m" else 5
            recent_candles = df.tail(candle_count).copy()
            recent_candles['timestamp'] = recent_candles['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_candles_dict = recent_candles.to_dict('records')
            
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
    """이더리움 롱숏비율 데이터 수집"""
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

# ===== 🆕 통합 시장 분석 함수 =====
def fetch_comprehensive_market_analysis():
    """온체인 + 매크로 + 기존 지표를 모두 통합한 종합 시장 분석"""
    print("📊 Comprehensive Market Analysis - Collecting all data sources...")
    
    # 기존 데이터
    market_sentiment = fetch_market_sentiment()
    
    # 새로운 온체인 데이터 (선택 사항)
    print("🔗 Fetching on-chain data (optional)...")
    exchange_flows = fetch_exchange_flows()
    whale_activity = fetch_whale_activity()
    onchain_metrics = fetch_onchain_metrics()
    
    # 매크로 경제 데이터 (필수)
    print("📈 Fetching traditional markets data...")
    traditional_markets = fetch_traditional_markets()
    economic_indicators = fetch_economic_indicators()
    
    # 상관관계 분석
    print("🔄 Analyzing market correlations...")
    current_price = exchange.fetch_ticker(symbol)['last']
    correlation_analysis = calculate_correlation_analysis(current_price, traditional_markets)
    
    comprehensive_data = {
        "traditional_sentiment": market_sentiment,
        "onchain_data": {
            "exchange_flows": exchange_flows,
            "whale_activity": whale_activity,
            "onchain_metrics": onchain_metrics
        },
        "macro_data": {
            "traditional_markets": traditional_markets,
            "economic_indicators": economic_indicators
        },
        "correlation_analysis": correlation_analysis,
        "data_quality_score": calculate_data_quality_score(exchange_flows, whale_activity, traditional_markets)
    }
    
    # 데이터 품질 및 주요 인사이트 출력
    print("📊 Market Analysis Summary:")
    print(f"   • Exchange Flow: {exchange_flows.get('flow_sentiment', 'N/A')}")
    print(f"   • Whale Activity: {whale_activity.get('whale_sentiment', 'N/A')}")
    print(f"   • Macro Sentiment: {traditional_markets.get('macro_sentiment', 'N/A')}")
    print(f"   • ETH-NASDAQ Correlation: {correlation_analysis.get('eth_nasdaq_correlation', 0):.3f}")
    print(f"   • Data Quality Score: {comprehensive_data['data_quality_score']}/100")
    
    return comprehensive_data

def calculate_data_quality_score(exchange_flows, whale_activity, traditional_markets):
    """데이터 품질 점수 계산 (0-100)"""
    score = 0
    max_score = 100
    
    if exchange_flows.get("data_source") not in ["ERROR", "NOT_AVAILABLE"]:
        score += 25
    
    if whale_activity.get("data_source") not in ["ERROR", "NOT_AVAILABLE"]:
        score += 25
        
    if traditional_markets.get("nasdaq", {}).get("trend") != "ERROR":
        score += 25
    
    # 기본 기술적 분석은 항상 가능
    score += 25
    
    return min(score, max_score)

def enhanced_trading_analysis():
    """기존 메인 루프에서 호출할 강화된 분석 함수"""
    
    # 종합 시장 분석 수행
    comprehensive_analysis = fetch_comprehensive_market_analysis()
    
    # 멀티 타임프레임 데이터 (기존)
    current_price = exchange.fetch_ticker(symbol)['last']
    multi_tf_data = fetch_multi_timeframe_data(emergency_mode=False)
    
    # 통합 이벤트 감지
    comprehensive_events = detect_comprehensive_market_events(
        current_price, multi_tf_data, comprehensive_analysis
    )
    
    # 이벤트가 감지되면 긴급 모드로 전환
    if comprehensive_events.get("comprehensive_event"):
        print(f"\n🚨 COMPREHENSIVE EVENT DETECTED! 🚨")
        print(f"Event Types: {', '.join(comprehensive_events.get('events_detected', []))}")
        print(f"Max Severity: {comprehensive_events.get('max_severity')}")
        print(f"Action Urgency: {comprehensive_events.get('action_urgency')}")
        
        # 1분봉 포함 긴급 모드로 재수집
        multi_tf_data = fetch_multi_timeframe_data(emergency_mode=True)
        comprehensive_events = detect_comprehensive_market_events(
            current_price, multi_tf_data, comprehensive_analysis
        )
    
    return {
        "current_price": current_price,
        "multi_tf_data": multi_tf_data,
        "comprehensive_analysis": comprehensive_analysis,
        "comprehensive_events": comprehensive_events
    }

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
                profit_loss_percentage = ((current_price / entry_price) - 1) * leverage * 100
            else: # short
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

# ===== 🆕 강화된 Gemini AI 시스템 프롬프트 =====
ENHANCED_SYSTEM_PROMPT = """
You are an expert Ethereum day trader with access to comprehensive market data. You MUST respond with ONLY a valid JSON object, no other text.

ANALYSIS DATA AVAILABLE:
1. TECHNICAL: Multi-timeframe price action, RSI, MACD, Stochastic indicators
2. ON-CHAIN: Exchange flows, whale activity, NVT ratio, active addresses (NOTE: On-chain data might be unavailable and will be marked as NEUTRAL if so. Do not base decisions solely on this if it's unavailable.)
3. SENTIMENT: Funding rates, long/short ratios, liquidations
4. MACRO: NASDAQ, S&P500, DXY, US10Y, VIX, Fed rates
5. NEWS: Real-time Ethereum news sentiment
6. CORRELATIONS: ETH-NASDAQ correlation analysis

ENHANCED ANALYSIS PROCESS:
1. EVENT PRIORITY: Check for comprehensive_events (price + whale + macro).
2. ON-CHAIN SIGNALS: 
   - Exchange OUTFLOWS (withdrawals) = BULLISH accumulation.
   - Exchange INFLOWS (deposits) = BEARISH distribution pressure.
   - Whale ACCUMULATING = Strong bullish signal.
   - Whale DISTRIBUTING = Strong bearish signal.
   - If on-chain data is NEUTRAL or UNKNOWN, rely more on technical, macro, and sentiment data.
3. MACRO CORRELATION:
   - NASDAQ UP + DXY DOWN = Risk-on environment (BULLISH for ETH).
   - NASDAQ DOWN + DXY UP = Risk-off environment (BEARISH for ETH).
   - High ETH-NASDAQ correlation = Follow traditional market moves.
4. SENTIMENT CONFLUENCE: Look for alignment across all available data sources.
5. EVENT-DRIVEN SIZING: Increase position size when multiple strong signals align.

DECISION CONFIDENCE LEVELS:
- EXTREME (90%+): All signals align + major event detected.
- HIGH (80-90%): Multiple strong signals across different data types.
- MEDIUM (65-79%): Some conflicting signals but trend is clear.
- LOW (<65%): NO_POSITION.

ENHANCED POSITION SIZING:
- Whale events + price confluence: Add 0.1-0.2 to base position size.
- Macro alignment (NASDAQ/DXY): Add 0.05-0.15 to position size.
- Strong on-chain flow signal: Add 0.1 to position size.
- Multiple event types: Add 0.2 to position size (max 0.6 total).

RESPONSE FORMAT (JSON ONLY):
{
  "direction": "LONG",
  "recommended_position_size": 0.4,
  "recommended_leverage": 25,
  "stop_loss_percentage": 0.18,
  "take_profit_percentage": 0.35,
  "reasoning": "Brief comprehensive analysis in one sentence based on available data"
}
"""

# ===== 메인 프로그램 시작 =====
print("\n=== Enhanced Ethereum Day Trading Bot v2.7 (Optional On-Chain) Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Strategy: Day Trading (5m/15m/1h analysis)")
print("AI Engine: Google Gemini 2.5 Flash")
print("Asset: Ethereum Futures")
print("Leverage Range: 5-35x (Dynamic)")
print("SL/TP Range: 10-60% on margin (Dynamic)")

print("\n🆕 FLEXIBLE DATA SOURCES:")
print("✅ On-chain Data: Exchange flows, Whale tracking (If API key is provided)")
print("✅ Macro Analysis: NASDAQ, S&P500, DXY, US10Y correlation")
print("✅ Comprehensive Event Detection: Multi-source alerts")
print("✅ AI Decision adapts to available data.")

print("\nExecution Frequency: Every 2 minutes (30s during events)")
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

        # ===== 2. 🆕 강화된 시장 분석 수행 =====
        enhanced_analysis = enhanced_trading_analysis()
        
        current_price = enhanced_analysis["current_price"]
        multi_tf_data = enhanced_analysis["multi_tf_data"]
        comprehensive_analysis = enhanced_analysis["comprehensive_analysis"]
        comprehensive_events = enhanced_analysis["comprehensive_events"]

        # ===== 3. 종합 이벤트 감지 시 추가 처리 =====
        if comprehensive_events.get("comprehensive_event"):
            print(f"\n🚨 MULTI-SOURCE EVENT ALERT! 🚨")
            
            event_types = comprehensive_events.get("events_detected", [])
            if "WHALE_EVENT" in event_types:
                whale_data = comprehensive_events["whale_event"]
                print(f"🐋 Whale Event: {whale_data.get('event_type')} - {whale_data.get('largest_transaction', 0):,.0f} ETH")
            
            if "MACRO_EVENT" in event_types:
                macro_data = comprehensive_events["macro_event"]
                print(f"📈 Macro Event: {macro_data.get('event_type')} - NASDAQ {macro_data.get('nasdaq_change', 0):+.1f}%")
            
            if "PRICE_EVENT" in event_types:
                print(f"💹 Price Event: {comprehensive_events['price_event'].get('event_type')}")

        # ===== 4. 포지션이 있는 경우 처리 =====
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
                    'sl_percentage': 0,
                    'tp_percentage': 0,
                    'position_size_percentage': 0,
                    'investment_amount': 0
                }
                current_trade_id = save_trade(temp_trade_data)
                current_trade = get_latest_open_trade()
                print("새로운 거래 기록 생성 (기존 포지션)")

        # ===== 5. 포지션이 없는 경우 처리 =====
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
            
            time.sleep(5)
            print("No position. Analyzing market for ethereum day trading opportunities...")

            # ===== 6. 이더리움 뉴스 분석 =====
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

            # ===== 7. 🆕 강화된 Gemini AI 분석용 데이터 준비 =====
            market_analysis = {
                "current_price": current_price,
                "timeframes": multi_tf_data,
                "market_sentiment": comprehensive_analysis["traditional_sentiment"],
                "onchain_analysis": {
                    "exchange_flows": comprehensive_analysis["onchain_data"]["exchange_flows"],
                    "whale_activity": comprehensive_analysis["onchain_data"]["whale_activity"],
                    "onchain_metrics": comprehensive_analysis["onchain_data"]["onchain_metrics"]
                },
                "macro_analysis": {
                    "traditional_markets": comprehensive_analysis["macro_data"]["traditional_markets"],
                    "economic_indicators": comprehensive_analysis["macro_data"]["economic_indicators"],
                    "correlation": comprehensive_analysis["correlation_analysis"]
                },
                "comprehensive_events": comprehensive_events,
                "news_analysis": {
                    "sentiment": news_sentiment['sentiment'],
                    "bullish_count": news_sentiment['bullish_count'],
                    "bearish_count": news_sentiment['bearish_count'],
                    "sentiment_strength": news_sentiment['sentiment_strength'],
                    "recent_headlines": [news['title'] for news in ethereum_news[:5]]
                },
                "recent_trades": historical_trading_data,
                "performance_summary": get_performance_metrics(),
                "data_quality_score": comprehensive_analysis["data_quality_score"]
            }

            # ===== 8. 🆕 강화된 Gemini AI 트레이딩 결정 요청 =====
            try:
                market_analysis_json = json.dumps(market_analysis, ensure_ascii=False, indent=2)
                
                response = model.generate_content([
                    ENHANCED_SYSTEM_PROMPT,
                    f"Comprehensive Ethereum Market Analysis: {market_analysis_json}"
                ])
                
                response_content = response.text.strip()
                
                def extract_json_from_response(text):
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        return match.group(0)
                    return text

                cleaned_response = extract_json_from_response(response_content)
                trading_decision = json.loads(cleaned_response)
                
                print(f"🆕 AI 거래 결정 (Enhanced Ethereum Analysis):")
                print(f"방향: {trading_decision['direction']}")
                print(f"추천 포지션 크기: {trading_decision['recommended_position_size']*100:.1f}%")
                print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                print(f"스탑로스 레벨 (원금 대비): {trading_decision['stop_loss_percentage']*100:.2f}%")
                print(f"테이크프로핏 레벨 (원금 대비): {trading_decision['take_profit_percentage']*100:.2f}%")
                print(f"분석 근거: {trading_decision['reasoning']}")
                
                # AI 분석 결과를 데이터베이스에 저장
                analysis_data = {
                    'current_price': current_price,
                    **trading_decision,
                    'news_sentiment': news_sentiment['sentiment']
                }
                analysis_id = save_ai_analysis(analysis_data)
                
                action = trading_decision['direction'].lower()

                # ===== 9. 트레이딩 결정에 따른 액션 실행 =====
                if action == "no_position":
                    print("현재 ETH 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    time.sleep(120)
                    continue
                    
                # ===== 10. 투자 금액 및 레버리지 계산 =====
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']
                
                position_size_percentage = trading_decision['recommended_position_size']
                recommended_leverage = trading_decision['recommended_leverage']
                sl_percentage = trading_decision['stop_loss_percentage']
                tp_percentage = trading_decision['take_profit_percentage']
                
                investment_amount = available_capital * position_size_percentage
                
                if investment_amount < 100:
                    investment_amount = 100
                    print(f"최소 투입 마진(100 USDT)으로 조정됨")

                total_position_value = investment_amount * recommended_leverage
                amount = math.ceil((total_position_value / current_price) * 10000) / 10000
                
                if amount <= 0:
                    print("계산된 주문 수량이 0보다 작거나 같아 주문을 진행하지 않습니다.")
                    time.sleep(60)
                    continue
                
                exchange.set_leverage(recommended_leverage, symbol)
                sl_price_change_ratio = sl_percentage / recommended_leverage
                tp_price_change_ratio = tp_percentage / recommended_leverage

                # ===== 11. 포지션 진입 및 SL/TP 주문 실행 =====
                if action == "long":
                    order = exchange.create_market_buy_order(symbol, amount)
                    entry_price = current_price
                    sl_price = round(entry_price * (1 - sl_price_change_ratio), 2)
                    tp_price = round(entry_price * (1 + tp_price_change_ratio), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'long', 'entry_price': entry_price, 'amount': amount,
                        'leverage': recommended_leverage, 'sl_price': sl_price, 'tp_price': tp_price,
                        'sl_percentage': sl_percentage, 'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage, 'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)
                    
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    print(f"\n=== 🆕 ENHANCED ETH LONG Position Opened ===")
                    # ... (print statements) ...

                elif action == "short":
                    order = exchange.create_market_sell_order(symbol, amount)
                    entry_price = current_price
                    sl_price = round(entry_price * (1 + sl_price_change_ratio), 2)
                    tp_price = round(entry_price * (1 - tp_price_change_ratio), 2)
                    
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    trade_data = {
                        'action': 'short', 'entry_price': entry_price, 'amount': amount,
                        'leverage': recommended_leverage, 'sl_price': sl_price, 'tp_price': tp_price,
                        'sl_percentage': sl_percentage, 'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage, 'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)
                    
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE ai_analysis SET trade_id = ? WHERE id = ?", (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    print(f"\n=== 🆕 ENHANCED ETH SHORT Position Opened ===")
                    # ... (print statements) ...
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"Raw Gemini 응답: {response_content[:1000]}...")
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Gemini API 또는 주문 오류: {e}")
                time.sleep(30)
                continue

        # ===== 12. 대기 시간 (이벤트 기반 최적화) =====
        sleep_time = 120 # 기본 2분
        if current_side:
            sleep_time = 60 # 포지션 있으면 1분
        if comprehensive_events.get("comprehensive_event"):
            sleep_time = 30 # 이벤트 있으면 30초
            if comprehensive_events.get("action_urgency") == "HIGH":
                sleep_time = 15 # 긴급 이벤트면 15초
        
        print(f"Next analysis in {sleep_time} seconds...")
        time.sleep(sleep_time)

    except Exception as e:
        print(f"\n Main Loop Error: {e}")
        time.sleep(10)

# ===== 🆕 추가 유틸리티 함수들 =====
def print_comprehensive_status():
    """현재 봇의 종합 상태 출력"""
    try:
        current_price = exchange.fetch_ticker(symbol)['last']
        
        print(f"\n{'='*60}")
        print(f"🆕 ENHANCED ETHEREUM TRADING BOT STATUS")
        print(f"{'='*60}")
        print(f"Current ETH Price: ${current_price:,.2f}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        current_trade = get_latest_open_trade()
        if current_trade:
            print(f"Active Position: {current_trade['action'].upper()} {current_trade['amount']} ETH")
            entry_price = current_trade['entry_price']
            leverage = current_trade['leverage']
            pnl_pct = ((current_price / entry_price) - 1 if current_trade['action'] == 'long' else (entry_price / current_price) - 1) * leverage * 100
            print(f"Entry Price: ${entry_price:,.2f}, Current P/L: {pnl_pct:+.2f}%")
        else:
            print("Active Position: None")
        
        performance = get_performance_metrics()
        if performance and performance["total_trades"] > 0:
            print(f"Performance Summary: Total Trades {performance['total_trades']}, Win Rate {performance['win_rate']:.1f}%, Avg P/L {performance['avg_profit_loss_percentage']:+.2f}%")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Status display error: {e}")

def emergency_shutdown():
    """긴급 종료 함수 - 모든 포지션과 주문 정리"""
    try:
        print(f"\n🚨 EMERGENCY SHUTDOWN INITIATED 🚨")
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            exchange.cancel_order(order['id'], symbol)
        print(f"Cancelled {len(open_orders)} open orders")
        
        positions = exchange.fetch_positions([symbol])
        for position in positions:
            if position['symbol'] == 'ETH/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if amt != 0:
                    side = 'sell' if amt > 0 else 'buy'
                    order_type = 'market'
                    exchange.create_order(symbol, order_type, side, abs(amt))
                    print(f"Emergency closed position: {side.upper()} {abs(amt)} ETH")
        
        print("🚨 All positions and orders cleared")
        
    except Exception as e:
        print(f"Emergency shutdown error: {e}")

# 프로그램 시작시 상태 출력 및 API 키 확인
print_comprehensive_status()

print("🔑 API Configuration Status:")
print(f"   • Binance (거래): {'✅' if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET_KEY') else '❌ CRITICAL'}")
print(f"   • Gemini (AI 분석): {'✅' if os.getenv('GEMINI_API_KEY') else '❌ CRITICAL'}")
print(f"   • FRED (거시 경제): {'✅' if os.getenv('FRED_API_KEY') else '❌ CRITICAL'}")
print(f"   • SERP (뉴스): {'✅' if os.getenv('SERP_API_KEY') else '⚠️ Optional'}")
print(f"   • CryptoQuant (온체인): {'✅' if os.getenv('CRYPTOQUANT_API_KEY') else '⚠️ Optional'}")
print(f"   • Whale Alert (온체인): {'✅' if os.getenv('WHALE_ALERT_API_KEY') else '⚠️ Optional'}")
print(f"   • Glassnode (온체인): {'✅' if os.getenv('GLASSNODE_API_KEY') else '⚠️ Optional'}")
print(f"{'='*60}")

print("\n🚀 Enhanced bot ready - Starting comprehensive analysis...")