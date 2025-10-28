"""
AI Trading Dashboard - Streamlit
실시간 거래 모니터링 대시보드
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

import ccxt
from dotenv import load_dotenv

load_dotenv()

# 바이낸스 거래소 초기화
try:
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })
    BINANCE_AVAILABLE = True
except Exception as e:
    print(f"바이낸스 API 초기화 실패: {e}")
    exchange = None
    BINANCE_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit {
        color: #00cc00;
        font-weight: bold;
    }
    .loss {
        color: #ff0000;
        font-weight: bold;
    }
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 600px;
        overflow-y: auto;
    }
    .log-line {
        margin: 0.2rem 0;
        line-height: 1.4;
    }
    .log-timestamp {
        color: #6a9fb5;
    }
    .log-info {
        color: #4ec9b0;
    }
    .log-warning {
        color: #dcdcaa;
    }
    .log-error {
        color: #f48771;
    }
    .log-success {
        color: #4ec9b0;
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 경로
DB_FILE = "live_trading.db"
LOG_FILE = "trading_bot.log"

# 봇 프로세스 관리
BOT_SCRIPT = "live_trading.py"

def get_db_connection():
    """DB 연결"""
    try:
        return sqlite3.connect(DB_FILE)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return None

def load_trades():
    """거래 데이터 로드"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # 테이블 존재 확인
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not c.fetchone():
            conn.close()
            return pd.DataFrame()
        
        # 안전한 쿼리
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'close_timestamp' in df.columns:
                df['close_timestamp'] = pd.to_datetime(df['close_timestamp'])
        
        return df
    except Exception as e:
        st.error(f"거래 데이터 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def load_open_positions():
    """오픈 포지션 로드 (DB + 바이낸스 실제 포지션)"""
    # DB에서 봇이 잡은 포지션
    conn = get_db_connection()
    if not conn:
        db_positions = pd.DataFrame()
    else:
        try:
            # 테이블 존재 확인
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not c.fetchone():
                db_positions = pd.DataFrame()
            else:
                # 안전한 쿼리 (exit_price 사용)
                db_positions = pd.read_sql_query("""
                    SELECT * FROM trades 
                    WHERE exit_price IS NULL OR exit_price = 0
                    ORDER BY timestamp DESC
                """, conn)
                
                if not db_positions.empty:
                    db_positions['timestamp'] = pd.to_datetime(db_positions['timestamp'])
            
            conn.close()
        except Exception as e:
            st.error(f"DB 포지션 로드 오류: {e}")
            db_positions = pd.DataFrame()
            if conn:
                conn.close()
    
    # 바이낸스 실제 포지션
    if BINANCE_AVAILABLE and exchange:
        try:
            positions = exchange.fetch_positions()
            real_positions = []
            
            for pos in positions:
                contracts = pos.get('contracts', 0)
                if contracts is None:
                    contracts = 0
                contracts = float(contracts)
                
                if contracts != 0:
                    symbol = pos['symbol'].replace('/USDT:USDT', '')
                    entry_price = pos.get('entryPrice', 0)
                    mark_price = pos.get('markPrice', 0)
                    unrealized_pnl = pos.get('unrealizedPnl', 0)
                    notional = pos.get('notional', 0)
                    initial_margin = pos.get('initialMargin', 0)
                    
                    # 실제 레버리지 계산
                    # 방법 1: notional / initialMargin
                    if initial_margin and initial_margin != 0:
                        actual_leverage = abs(float(notional) / float(initial_margin))
                    else:
                        # 방법 2: API에서 제공하는 레버리지 사용
                        actual_leverage = float(pos.get('leverage', 1))
                    
                    # 투자금 계산 (증거금)
                    if initial_margin:
                        investment_amount = float(initial_margin)
                    else:
                        investment_amount = abs(float(notional) / actual_leverage) if actual_leverage else abs(float(notional))
                    
                    real_positions.append({
                        'coin_symbol': symbol,
                        'action': pos['side'],
                        'entry_price': float(entry_price) if entry_price is not None else 0,
                        'current_price': float(mark_price) if mark_price is not None else 0,
                        'amount': contracts,
                        'leverage': round(actual_leverage, 1),  # 실제 계산된 레버리지
                        'unrealized_pnl': float(unrealized_pnl) if unrealized_pnl is not None else 0,
                        'notional': abs(float(notional)) if notional is not None else 0,
                        'investment_amount': investment_amount,
                        'sl_price': None,
                        'tp_price': None,
                        'trading_style': 'MANUAL',
                        'holding_time_estimate': 'Unknown',
                        'timestamp': None,
                        'ai_reasoning': '수동 진입 포지션',
                        'source': 'BINANCE'
                    })
            
            if real_positions:
                real_df = pd.DataFrame(real_positions)
                
                # DB 포지션과 병합
                if not db_positions.empty:
                    # 바이낸스 포지션을 딕셔너리로 변환 (심볼을 키로)
                    binance_dict = {row['coin_symbol']: row for _, row in real_df.iterrows()}
                    
                    # DB 포지션에 바이낸스 데이터 추가
                    db_positions['source'] = 'BOT'
                    db_positions['current_price'] = db_positions['coin_symbol'].apply(
                        lambda x: binance_dict[x]['current_price'] if x in binance_dict else None
                    )
                    db_positions['unrealized_pnl'] = db_positions['coin_symbol'].apply(
                        lambda x: binance_dict[x]['unrealized_pnl'] if x in binance_dict else None
                    )
                    db_positions['notional'] = db_positions['coin_symbol'].apply(
                        lambda x: binance_dict[x]['notional'] if x in binance_dict else None
                    )
                    
                    # DB에 없는 바이낸스 포지션만 추가 (수동 포지션)
                    db_coins = set(db_positions['coin_symbol'])
                    manual_positions = real_df[~real_df['coin_symbol'].isin(db_coins)]
                    
                    # 병합 (빈 데이터프레임 체크)
                    if manual_positions.empty:
                        combined = db_positions
                    else:
                        combined = pd.concat([db_positions, manual_positions], ignore_index=True)
                    return combined
                else:
                    # DB 포지션 없으면 바이낸스만
                    return real_df
        except Exception as e:
            st.warning(f"바이낸스 포지션 조회 오류: {e}")
    
    # 바이낸스 조회 실패 시 DB만 반환
    if not db_positions.empty:
        db_positions['source'] = 'BOT'
        db_positions['current_price'] = None
        db_positions['unrealized_pnl'] = None
        db_positions['notional'] = None
    return db_positions

def load_ai_decisions():
    """AI 결정 로드"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
        SELECT 
            coin_symbol, decision_type, direction, leverage,
            confidence_score, reasoning, timestamp
        FROM ai_decisions
        ORDER BY timestamp DESC
        LIMIT 50
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"AI 결정 로드 오류: {e}")
        return pd.DataFrame()

def get_binance_account_info():
    """바이낸스 계정 정보 조회"""
    if not BINANCE_AVAILABLE or not exchange:
        return None
    
    try:
        balance = exchange.fetch_balance()
        usdt_info = balance.get('USDT', {})
        
        return {
            'total_balance': usdt_info.get('total', 0),
            'available_balance': usdt_info.get('free', 0),
            'used_balance': usdt_info.get('used', 0),
            'unrealized_pnl': balance.get('info', {}).get('totalUnrealizedProfit', 0)
        }
    except Exception as e:
        st.warning(f"계정 정보 조회 오류: {e}")
        return None

def get_binance_trade_history(days=30, limit=1000):
    """바이낸스 거래 내역 조회"""
    if not BINANCE_AVAILABLE or not exchange:
        return pd.DataFrame()
    
    try:
        # 최근 거래 내역 가져오기
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        
        # 모든 심볼의 거래 내역
        all_trades = []
        markets = exchange.load_markets()
        
        # USDT 선물 마켓만 필터링
        future_symbols = [symbol for symbol in markets.keys() if '/USDT:USDT' in symbol]
        
        # 상위 거래량 코인만 (시간 절약)
        top_symbols = future_symbols[:50]  # 상위 50개만
        
        for symbol in top_symbols:
            try:
                trades = exchange.fetch_my_trades(symbol, since=since, limit=100)
                if trades:
                    all_trades.extend(trades)
            except Exception as e:
                continue  # 해당 심볼 오류는 건너뛰기
        
        if not all_trades:
            return pd.DataFrame()
        
        # DataFrame으로 변환
        trades_data = []
        for trade in all_trades:
            trades_data.append({
                'timestamp': pd.to_datetime(trade['timestamp'], unit='ms'),
                'symbol': trade['symbol'].replace('/USDT:USDT', ''),
                'side': trade['side'],
                'price': trade['price'],
                'amount': trade['amount'],
                'cost': trade['cost'],
                'fee': trade['fee']['cost'] if trade.get('fee') else 0,
                'realized_pnl': trade.get('info', {}).get('realizedPnl', 0)
            })
        
        df = pd.DataFrame(trades_data)
        df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.warning(f"거래 내역 조회 오류: {e}")
        return pd.DataFrame()

def get_binance_income_history(days=30):
    """바이낸스 수익 내역 조회 (realized PnL)"""
    if not BINANCE_AVAILABLE or not exchange:
        return pd.DataFrame()
    
    try:
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        
        # 실현 손익 내역
        income = exchange.fapiPrivateGetIncome({
            'incomeType': 'REALIZED_PNL',
            'startTime': since,
            'limit': 1000
        })
        
        if not income:
            return pd.DataFrame()
        
        income_data = []
        for item in income:
            income_data.append({
                'timestamp': pd.to_datetime(int(item['time']), unit='ms'),
                'symbol': item['symbol'].replace('USDT', ''),
                'income': float(item['income']),
                'asset': item['asset']
            })
        
        return pd.DataFrame(income_data)
        
    except Exception as e:
        st.warning(f"수익 내역 조회 오류: {e}")
        return pd.DataFrame()

def calculate_binance_metrics(income_df, trades_df, open_positions_df):
    """바이낸스 실제 데이터로 메트릭 계산"""
    metrics = {
        'total_trades': 0,
        'open_trades': len(open_positions_df),
        'closed_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'best_trade': 0,
        'worst_trade': 0,
        'avg_holding_time': 0
    }
    
    # 미실현 손익 (오픈 포지션)
    if not open_positions_df.empty and 'unrealized_pnl' in open_positions_df.columns:
        metrics['unrealized_pnl'] = open_positions_df['unrealized_pnl'].sum()
    
    # 실현 손익 (income_df 사용)
    if not income_df.empty:
        metrics['realized_pnl'] = income_df['income'].sum()
        
        winning = income_df[income_df['income'] > 0]
        losing = income_df[income_df['income'] < 0]
        
        metrics['winning_trades'] = len(winning)
        metrics['losing_trades'] = len(losing)
        metrics['closed_trades'] = len(income_df)
        
        if metrics['closed_trades'] > 0:
            metrics['win_rate'] = (metrics['winning_trades'] / metrics['closed_trades']) * 100
        
        if not winning.empty:
            metrics['avg_win'] = winning['income'].mean()
            metrics['best_trade'] = winning['income'].max()
        
        if not losing.empty:
            metrics['avg_loss'] = losing['income'].mean()
            metrics['worst_trade'] = losing['income'].min()
    
    metrics['total_trades'] = metrics['closed_trades'] + metrics['open_trades']
    metrics['total_pnl'] = metrics['realized_pnl'] + metrics['unrealized_pnl']
    
    # 평균 보유 시간 (거래 내역에서 계산)
    if not trades_df.empty:
        # 같은 심볼의 연속된 거래를 매칭
        trades_df = trades_df.sort_values('timestamp')
        holding_times = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
            
            for i in range(1, len(symbol_trades)):
                prev_trade = symbol_trades.iloc[i-1]
                curr_trade = symbol_trades.iloc[i]
                
                # 진입 후 청산
                if prev_trade['side'] != curr_trade['side']:
                    time_diff = (curr_trade['timestamp'] - prev_trade['timestamp']).total_seconds() / 3600
                    holding_times.append(time_diff)
        
        if holding_times:
            metrics['avg_holding_time'] = sum(holding_times) / len(holding_times)
    
    return metrics
def calculate_db_metrics(trades_df):
    """DB 데이터로 성과 지표 계산 (백업용)"""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'open_trades': 0,
            'closed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_holding_time': 0
        }
    
    total_trades = len(trades_df)
    open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
    closed_trades = len(trades_df[trades_df['status'] == 'CLOSED'])
    
    closed_df = trades_df[trades_df['status'] == 'CLOSED'].copy()
    
    if not closed_df.empty:
        # 바이낸스 실제 PnL 우선 사용
        closed_df['actual_pnl'] = closed_df['binance_pnl'].fillna(closed_df['pnl'])
        
        winning_trades = len(closed_df[closed_df['actual_pnl'] > 0])
        losing_trades = len(closed_df[closed_df['actual_pnl'] < 0])
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
        
        realized_pnl = closed_df['actual_pnl'].sum()
        avg_win = closed_df[closed_df['actual_pnl'] > 0]['actual_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = closed_df[closed_df['actual_pnl'] < 0]['actual_pnl'].mean() if losing_trades > 0 else 0
        best_trade = closed_df['actual_pnl'].max()
        worst_trade = closed_df['actual_pnl'].min()
        
        # 평균 보유 시간 계산
        closed_df['timestamp'] = pd.to_datetime(closed_df['timestamp'])
        closed_df['close_timestamp'] = pd.to_datetime(closed_df['close_timestamp'])
        closed_df['holding_time'] = (closed_df['close_timestamp'] - closed_df['timestamp']).dt.total_seconds() / 3600
        avg_holding_time = closed_df['holding_time'].mean()
    else:
        winning_trades = losing_trades = 0
        win_rate = realized_pnl = avg_win = avg_loss = best_trade = worst_trade = avg_holding_time = 0
    
    # 미실현 손익 (오픈 포지션이 있다면 별도 계산 필요)
    unrealized_pnl = 0  # 실시간 가격 필요
    
    total_pnl = realized_pnl + unrealized_pnl
    
    return {
        'total_trades': total_trades,
        'open_trades': open_trades,
        'closed_trades': closed_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_holding_time': avg_holding_time
    }

def read_log_file(num_lines=200):
    """로그 파일 읽기"""
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-num_lines:] if len(lines) > num_lines else lines
    except Exception as e:
        return [f"로그 읽기 오류: {e}"]

def format_log_line(line):
    """로그 라인 포맷팅"""
    line = line.strip()
    
    # 로그 레벨 감지
    if '❌' in line or 'ERROR' in line.upper() or '오류' in line:
        css_class = 'log-error'
    elif '⚠️' in line or 'WARNING' in line.upper() or '경고' in line:
        css_class = 'log-warning'
    elif '✅' in line or 'SUCCESS' in line.upper() or '성공' in line or '완료' in line:
        css_class = 'log-success'
    else:
        css_class = 'log-info'
    
    return f'<div class="log-line {css_class}">{line}</div>'

def check_bot_running():
    """봇 실행 여부 확인 (Windows/Linux 호환)"""
    try:
        import psutil
        
        # 현재 실행 중인 모든 Python 프로세스 확인
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any(BOT_SCRIPT in arg for arg in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return False
    except ImportError:
        # psutil이 없으면 대체 방법
        try:
            if sys.platform == 'win32':
                # Windows
                result = subprocess.run(
                    ['tasklist', '/FI', f'IMAGENAME eq python.exe'],
                    capture_output=True,
                    text=True
                )
                # 더 정확한 체크를 위해 로그 파일 수정 시간 확인
                if os.path.exists(LOG_FILE):
                    mtime = os.path.getmtime(LOG_FILE)
                    # 로그가 최근 5분 내에 수정되었으면 실행 중으로 판단
                    return (time.time() - mtime) < 300
                return False
            else:
                # Linux/Mac
                result = subprocess.run(
                    ['pgrep', '-f', BOT_SCRIPT],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
        except:
            # 모두 실패하면 로그 파일로 판단
            if os.path.exists(LOG_FILE):
                mtime = os.path.getmtime(LOG_FILE)
                return (time.time() - mtime) < 300
            return False

def start_bot():
    """봇 시작 (Windows/Linux 호환)"""
    try:
        if sys.platform == 'win32':
            # Windows - 새 프로세스로 시작
            subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                stdout=open(LOG_FILE, 'a', encoding='utf-8'),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                stdout=open(LOG_FILE, 'a'),
                stderr=subprocess.STDOUT
            )
        return True
    except Exception as e:
        st.error(f"봇 시작 오류: {e}")
        return False

def stop_bot():
    """봇 중지 (Windows/Linux 호환)"""
    try:
        if psutil:
            # psutil 사용 (더 정확함)
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any(BOT_SCRIPT in arg for arg in cmdline):
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
        else:
            # psutil 없으면 OS별 명령어
            if sys.platform == 'win32':
                # Windows - taskkill 사용
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'])
            else:
                # Linux/Mac
                subprocess.run(['pkill', '-f', BOT_SCRIPT])
        
        return True
    except Exception as e:
        st.error(f"봇 중지 오류: {e}")
        return False

# ===== 메인 대시보드 =====

def main():
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # 사이드바 - 봇 컨트롤
    with st.sidebar:
        st.header("⚙️ 봇 컨트롤")
        
        bot_running = check_bot_running()
        
        if bot_running:
            st.success("🟢 봇 실행 중")
            if st.button("⏹️ 봇 중지", use_container_width=True):
                if stop_bot():
                    st.success("봇이 중지되었습니다.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.error("🔴 봇 중지됨")
            if st.button("▶️ 봇 시작", use_container_width=True):
                if start_bot():
                    st.success("봇이 시작되었습니다.")
                    time.sleep(2)
                    st.rerun()
        
        st.markdown("---")
        
        st.header("🔄 새로고침")
        auto_refresh = st.checkbox("자동 새로고침 (10초)", value=False)
        
        if st.button("🔄 수동 새로고침", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        st.header("📊 표시 옵션")
        show_closed = st.checkbox("청산된 거래 표시", value=True)
        show_ai_decisions = st.checkbox("AI 결정 표시", value=False)
        log_lines = st.slider("로그 라인 수", 50, 500, 200, 50)
    
    # 데이터 로드
    trades_df = load_trades()
    open_positions_df = load_open_positions()
    
    # 바이낸스 실제 데이터 로드
    use_binance_data = st.sidebar.checkbox("바이낸스 실제 데이터 사용", value=True, 
                                           help="체크 시 바이낸스 API에서 실제 거래 내역과 손익을 가져옵니다")
    
    if use_binance_data and BINANCE_AVAILABLE:
        with st.spinner("바이낸스 데이터 로딩 중..."):
            # 계정 정보
            account_info = get_binance_account_info()
            
            # 거래 내역
            binance_trades_df = get_binance_trade_history(days=30)
            
            # 수익 내역
            binance_income_df = get_binance_income_history(days=30)
            
            # 메트릭 계산
            metrics = calculate_binance_metrics(binance_income_df, binance_trades_df, open_positions_df)
            
            # 계정 정보 표시
            if account_info:
                st.sidebar.markdown("---")
                st.sidebar.subheader("💰 바이낸스 계정")
                st.sidebar.metric("총 잔고", f"${account_info['total_balance']:,.2f}")
                st.sidebar.metric("가용 잔고", f"${account_info['available_balance']:,.2f}")
                st.sidebar.metric("사용 중", f"${account_info['used_balance']:,.2f}")
    else:
        # DB 데이터 사용
        metrics = calculate_db_metrics(trades_df)
        binance_trades_df = pd.DataFrame()
        binance_income_df = pd.DataFrame()
        account_info = None
    
    # 상단 메트릭
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "총 거래",
            f"{metrics['total_trades']}회",
            delta=f"오픈: {metrics['open_trades']}"
        )
    
    with col2:
        pnl_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
        st.metric(
            "총 손익",
            f"${metrics['total_pnl']:,.2f}",
            delta=f"{metrics['total_pnl']:+.2f}",
            delta_color=pnl_color
        )
    
    with col3:
        st.metric(
            "실현 손익",
            f"${metrics['realized_pnl']:,.2f}",
            delta=f"미실현: ${metrics['unrealized_pnl']:,.2f}"
        )
    
    with col4:
        st.metric(
            "승률",
            f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['winning_trades']}승 {metrics['losing_trades']}패"
        )
    
    with col5:
        st.metric(
            "평균 보유",
            f"{metrics['avg_holding_time']:.1f}시간",
            delta=f"청산: {metrics['closed_trades']}회"
        )
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 오픈 포지션",
        "📜 거래 내역",
        "📈 성과 분석",
        "🤖 AI 결정",
        "📋 실시간 로그"
    ])
    
    # 탭 1: 오픈 포지션
    with tab1:
        st.header("📊 현재 오픈 포지션")
        
        if open_positions_df.empty:
            st.info("현재 오픈된 포지션이 없습니다.")
        else:
            # 봇/수동 포지션 구분
            bot_positions = open_positions_df[open_positions_df['source'] == 'BOT']
            manual_positions = open_positions_df[open_positions_df['source'] == 'BINANCE']
            
            if not manual_positions.empty:
                st.subheader(f"🔧 수동 포지션 ({len(manual_positions)}개)")
                for idx, row in manual_positions.iterrows():
                    side_emoji = "🟢" if row['action'] == 'long' else "🔴"
                    
                    # 현재 수익률 계산
                    if row['current_price'] and row['entry_price']:
                        if row['action'] == 'long':
                            pnl_pct = (row['current_price'] - row['entry_price']) / row['entry_price'] * 100 * row['leverage']
                        else:
                            pnl_pct = (row['entry_price'] - row['current_price']) / row['entry_price'] * 100 * row['leverage']
                        pnl_color = "🟢" if pnl_pct > 0 else "🔴"
                    else:
                        pnl_pct = 0
                        pnl_color = "⚪"
                    
                    with st.expander(f"{side_emoji} {row['coin_symbol']} {row['action'].upper()} - 수동 진입", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("진입가", f"${row['entry_price']:,.4f}")
                            if row['current_price']:
                                st.metric("현재가", f"${row['current_price']:,.4f}")
                        
                        with col2:
                            st.metric("레버리지", f"{row['leverage']}x")
                            st.metric("수량", f"{row['amount']:.8f}")
                        
                        with col3:
                            if row['unrealized_pnl'] is not None:
                                pnl_display = f"${row['unrealized_pnl']:,.2f}"
                                st.metric("미실현 손익", pnl_display, delta=f"{pnl_pct:+.2f}%")
                            else:
                                st.metric("손익률", f"{pnl_pct:+.2f}%")
                        
                        with col4:
                            if row['investment_amount']:
                                st.metric("투자금", f"${row['investment_amount']:,.2f}")
                            if row['notional']:
                                st.metric("포지션 크기", f"${row['notional']:,.2f}")
                        
                        st.info("💡 이 포지션은 수동으로 진입한 포지션입니다. 봇이 관리하지 않습니다.")
                
                st.markdown("---")
            
            if not bot_positions.empty:
                st.subheader(f"🤖 봇 포지션 ({len(bot_positions)}개)")
                for idx, row in bot_positions.iterrows():
                    side_emoji = "🟢" if row['action'] == 'long' else "🔴"
                    style_emoji = {"SCALPING": "⚡", "DAY_TRADING": "📊", "SWING_TRADING": "📈"}.get(row['trading_style'], "📊")
                    
                    # 수익률 계산
                    pnl_display = ""
                    if pd.notna(row['current_price']) and row['current_price'] > 0:
                        if row['action'] == 'long':
                            pnl_pct = (row['current_price'] - row['entry_price']) / row['entry_price'] * 100 * row['leverage']
                        else:
                            pnl_pct = (row['entry_price'] - row['current_price']) / row['entry_price'] * 100 * row['leverage']
                        pnl_color = "🟢" if pnl_pct > 0 else "🔴"
                        pnl_display = f" | {pnl_color} {pnl_pct:+.2f}%"
                    
                    with st.expander(f"{side_emoji} {row['coin_symbol']} {row['action'].upper()} - {style_emoji} {row['trading_style']}{pnl_display}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("진입가", f"${row['entry_price']:,.4f}")
                            if pd.notna(row['current_price']) and row['current_price'] > 0:
                                st.metric("현재가", f"${row['current_price']:,.4f}")
                            st.metric("레버리지", f"{row['leverage']}x")
                        
                        with col2:
                            st.metric("투자금", f"${row['investment_amount']:,.2f}")
                            st.metric("수량", f"{row['amount']:.8f}")
                            if pd.notna(row['notional']) and row['notional'] > 0:
                                st.caption(f"포지션 크기: ${row['notional']:,.2f}")
                        
                        with col3:
                            # 미실현 손익 표시
                            if pd.notna(row['unrealized_pnl']) and row['unrealized_pnl'] != 0:
                                pnl_pct = (row['unrealized_pnl'] / row['investment_amount'] * 100) if row['investment_amount'] > 0 else 0
                                st.metric("미실현 손익", f"${row['unrealized_pnl']:,.2f}", delta=f"{pnl_pct:+.2f}%")
                            elif pd.notna(row['sl_price']):
                                st.metric("손절가", f"${row['sl_price']:,.4f}")
                                sl_pct = abs((row['sl_price'] - row['entry_price']) / row['entry_price'] * 100)
                                st.caption(f"(-{sl_pct:.2f}%)")
                        
                        with col4:
                            if pd.notna(row['tp_price']):
                                st.metric("익절가", f"${row['tp_price']:,.4f}")
                                tp_pct = abs((row['tp_price'] - row['entry_price']) / row['entry_price'] * 100)
                                st.caption(f"(+{tp_pct:.2f}%)")
                        
                        if pd.notna(row['holding_time_estimate']):
                            st.markdown(f"**⏱️ 예상 보유:** {row['holding_time_estimate']}")
                        if pd.notna(row['timestamp']):
                            st.markdown(f"**📅 진입 시간:** {row['timestamp']}")
                        st.markdown(f"**💭 AI 분석:** {row['ai_reasoning']}")
            
            # 전체 포지션 수 표시
            st.markdown("---")
            total_investment = open_positions_df['investment_amount'].sum()
            st.info(f"📊 총 {len(open_positions_df)}개 포지션 (봇: {len(bot_positions)}개, 수동: {len(manual_positions)}개) | 총 투자금: ${total_investment:,.2f}")
    
    # 탭 2: 거래 내역
    with tab2:
        st.header("📜 전체 거래 내역")
        
        if trades_df.empty:
            st.info("거래 내역이 없습니다.")
        else:
            # 필터
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.multiselect(
                    "상태 필터",
                    options=['OPEN', 'CLOSED'],
                    default=['OPEN', 'CLOSED'] if show_closed else ['OPEN']
                )
            
            with col2:
                # 트레이딩 스타일 필터
                all_styles = ['SCALPING', 'DAY_TRADING', 'SWING_TRADING']
                available_styles = trades_df['trading_style'].dropna().unique().tolist() if 'trading_style' in trades_df.columns else all_styles
                if not available_styles:
                    available_styles = all_styles
                
                style_filter = st.multiselect(
                    "스타일 필터",
                    options=available_styles,
                    default=available_styles
                )
            
            with col3:
                direction_filter = st.multiselect(
                    "방향 필터",
                    options=['long', 'short'],
                    default=['long', 'short']
                )
            
            # 필터 적용
            filter_conditions = (
                (trades_df['status'].isin(status_filter)) &
                (trades_df['action'].isin(direction_filter))
            )
            
            # trading_style 컬럼이 있는 경우에만 필터링
            if 'trading_style' in trades_df.columns and style_filter:
                filter_conditions = filter_conditions & (trades_df['trading_style'].isin(style_filter))
            
            filtered_df = trades_df[filter_conditions].copy()
            
            # 표시할 컬럼 선택
            display_columns = [
                'coin_symbol', 'action', 'trading_style', 'entry_price', 
                'exit_price', 'leverage', 'investment_amount'
            ]
            
            # binance_pnl이 있으면 우선 표시
            if 'binance_pnl' in filtered_df.columns:
                filtered_df['actual_pnl'] = filtered_df['binance_pnl'].fillna(filtered_df['pnl'])
                display_columns.append('actual_pnl')
            else:
                display_columns.append('pnl')
            
            display_columns.extend(['pnl_percentage', 'status', 'timestamp'])
            
            # 손절/익절가 추가 (있는 경우)
            if 'sl_price' in filtered_df.columns:
                display_columns.insert(display_columns.index('leverage') + 1, 'sl_price')
            if 'tp_price' in filtered_df.columns:
                display_columns.insert(display_columns.index('leverage') + 2, 'tp_price')
            
            display_df = filtered_df[display_columns].copy()
            
            # 컬럼명 한글화
            column_names = {
                'coin_symbol': '코인',
                'action': '방향',
                'trading_style': '스타일',
                'entry_price': '진입가',
                'exit_price': '청산가',
                'leverage': '레버리지',
                'sl_price': '손절가',
                'tp_price': '익절가',
                'investment_amount': '투자금',
                'pnl': '손익',
                'actual_pnl': '손익',
                'binance_pnl': '실제손익',
                'pnl_percentage': '수익률%',
                'status': '상태',
                'timestamp': '시간'
            }
            display_df = display_df.rename(columns=column_names)
            
            # PnL 색상 적용
            def highlight_pnl(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}; font-weight: bold'
            
            # 손익 컬럼에 스타일 적용 (pandas 최신 버전 호환)
            pnl_columns = [col for col in display_df.columns if '손익' in col or '수익률' in col]
            
            # pandas 버전 체크
            try:
                # pandas >= 2.1.0
                styled_df = display_df.style.map(
                    highlight_pnl,
                    subset=pnl_columns
                )
            except AttributeError:
                # pandas < 2.1.0 (구버전 호환)
                styled_df = display_df.style.applymap(
                    highlight_pnl,
                    subset=pnl_columns
                )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # CSV 다운로드
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # 탭 3: 성과 분석
    with tab3:
        st.header("📈 성과 분석")
        
        # 데이터 소스 표시
        if use_binance_data and BINANCE_AVAILABLE:
            st.info("📊 바이낸스 실제 데이터로 분석 중 (최근 30일)")
            
            if not binance_income_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # PnL 분포
                    st.subheader("💰 손익 분포")
                    fig_pnl = px.histogram(
                        binance_income_df,
                        x='income',
                        nbins=20,
                        title='거래별 손익 분포 (실제 데이터)',
                        labels={'income': '손익 (USDT)', 'count': '거래 수'}
                    )
                    fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    # 코인별 성과
                    st.subheader("🪙 코인별 성과")
                    if 'symbol' in binance_income_df.columns:
                        coin_stats = binance_income_df.groupby('symbol').agg({
                            'income': ['sum', 'mean', 'count']
                        }).round(2)
                        coin_stats.columns = ['총 손익', '평균 손익', '거래 수']
                        coin_stats = coin_stats.sort_values('총 손익', ascending=False)
                        st.dataframe(coin_stats.head(10), use_container_width=True)
                
                with col2:
                    # 누적 손익
                    st.subheader("📈 누적 손익")
                    income_sorted = binance_income_df.sort_values('timestamp')
                    income_sorted['cumulative_pnl'] = income_sorted['income'].cumsum()
                    
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=income_sorted['timestamp'],
                        y=income_sorted['cumulative_pnl'],
                        mode='lines+markers',
                        name='누적 손익',
                        line=dict(color='blue', width=2)
                    ))
                    fig_cum.update_layout(
                        title='시간별 누적 손익 (실제 데이터)',
                        xaxis_title='시간',
                        yaxis_title='누적 손익 (USDT)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                    
                    # 일별 손익
                    st.subheader("📅 일별 손익")
                    daily_pnl = binance_income_df.copy()
                    daily_pnl['date'] = daily_pnl['timestamp'].dt.date
                    daily_stats = daily_pnl.groupby('date')['income'].sum().reset_index()
                    
                    fig_daily = px.bar(
                        daily_stats,
                        x='date',
                        y='income',
                        title='일별 실현 손익',
                        labels={'income': '손익 (USDT)', 'date': '날짜'},
                        color='income',
                        color_continuous_scale=['red', 'gray', 'green']
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # 상세 메트릭
                st.subheader("📊 상세 메트릭")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("평균 수익", f"${metrics['avg_win']:,.2f}")
                    st.metric("최대 수익", f"${metrics['best_trade']:,.2f}")
                
                with col2:
                    st.metric("평균 손실", f"${metrics['avg_loss']:,.2f}")
                    st.metric("최대 손실", f"${metrics['worst_trade']:,.2f}")
                
                with col3:
                    profit_factor = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                    
                    expectancy = (metrics['win_rate'] / 100 * metrics['avg_win']) + \
                                ((1 - metrics['win_rate'] / 100) * metrics['avg_loss'])
                    st.metric("Expectancy", f"${expectancy:.2f}")
                
                with col4:
                    if account_info:
                        roi = (metrics['realized_pnl'] / account_info['total_balance'] * 100) if account_info['total_balance'] > 0 else 0
                        st.metric("ROI (30일)", f"{roi:.2f}%")
                    st.metric("총 거래", f"{metrics['closed_trades']}회")
            else:
                st.warning("최근 30일 내 거래 내역이 없습니다.")
        
        else:
            # DB 데이터 사용
            if trades_df.empty or metrics['closed_trades'] == 0:
                st.info("청산된 거래가 없어 성과 분석을 표시할 수 없습니다.")
            else:
                closed_df = trades_df[trades_df['status'] == 'CLOSED'].copy()
                closed_df['actual_pnl'] = closed_df['binance_pnl'].fillna(closed_df['pnl'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # PnL 분포
                    st.subheader("💰 손익 분포")
                    fig_pnl = px.histogram(
                        closed_df,
                        x='actual_pnl',
                        nbins=20,
                        title='거래별 손익 분포',
                        labels={'actual_pnl': '손익 (USDT)', 'count': '거래 수'}
                    )
                    fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    # 트레이딩 스타일별 성과
                    st.subheader("📊 스타일별 성과")
                    if 'trading_style' in closed_df.columns and not closed_df['trading_style'].isna().all():
                        style_stats = closed_df.groupby('trading_style').agg({
                            'actual_pnl': ['sum', 'mean', 'count']
                        }).round(2)
                        style_stats.columns = ['총 손익', '평균 손익', '거래 수']
                        
                        # 승률 추가
                        style_win_rate = closed_df.groupby('trading_style').apply(
                            lambda x: (x['actual_pnl'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
                        ).round(1)
                        style_stats['승률%'] = style_win_rate
                        
                        # 스타일별 이모지 추가
                        style_emoji_map = {
                            'SCALPING': '⚡',
                            'DAY_TRADING': '📊',
                            'SWING_TRADING': '📈'
                        }
                        style_stats.index = [f"{style_emoji_map.get(idx, '📊')} {idx}" for idx in style_stats.index]
                        
                        st.dataframe(style_stats, use_container_width=True)
                    else:
                        st.info("트레이딩 스타일 정보가 없습니다.")
                
                with col2:
                    # 누적 손익
                    st.subheader("📈 누적 손익")
                    closed_df = closed_df.sort_values('timestamp')
                    closed_df['cumulative_pnl'] = closed_df['actual_pnl'].cumsum()
                    
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=closed_df['timestamp'],
                        y=closed_df['cumulative_pnl'],
                        mode='lines+markers',
                        name='누적 손익',
                        line=dict(color='blue', width=2)
                    ))
                    fig_cum.update_layout(
                        title='시간별 누적 손익',
                        xaxis_title='시간',
                        yaxis_title='누적 손익 (USDT)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                    
                    # 코인별 성과
                    st.subheader("🪙 코인별 성과")
                    coin_stats = closed_df.groupby('coin_symbol').agg({
                        'actual_pnl': ['sum', 'mean', 'count']
                    }).round(2)
                    coin_stats.columns = ['총 손익', '평균 손익', '거래 수']
                    coin_stats = coin_stats.sort_values('총 손익', ascending=False)
                    st.dataframe(coin_stats.head(10), use_container_width=True)
                
                # 상세 메트릭
                st.subheader("📊 상세 메트릭")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("평균 수익", f"${metrics['avg_win']:,.2f}")
                    st.metric("최대 수익", f"${metrics['best_trade']:,.2f}")
                
                with col2:
                    st.metric("평균 손실", f"${metrics['avg_loss']:,.2f}")
                    st.metric("최대 손실", f"${metrics['worst_trade']:,.2f}")
                
                with col3:
                    profit_factor = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                    
                    expectancy = (metrics['win_rate'] / 100 * metrics['avg_win']) + \
                                ((1 - metrics['win_rate'] / 100) * metrics['avg_loss'])
                    st.metric("Expectancy", f"${expectancy:.2f}")
                
                with col4:
                    total_invested = closed_df['investment_amount'].sum()
                    roi = (metrics['realized_pnl'] / total_invested * 100) if total_invested > 0 else 0
                    st.metric("ROI", f"{roi:.2f}%")
                    st.metric("총 투자금", f"${total_invested:,.2f}")
    
    # 탭 4: AI 결정
    with tab4:
        if show_ai_decisions:
            st.header("🤖 AI 결정 내역")
            
            ai_decisions_df = load_ai_decisions()
            
            if ai_decisions_df.empty:
                st.info("AI 결정 내역이 없습니다.")
            else:
                for idx, row in ai_decisions_df.iterrows():
                    decision_type = row['decision_type']
                    
                    if decision_type == 'ENTRY':
                        icon = "🟢"
                        color = "#00cc00"
                    elif decision_type == 'EXIT':
                        icon = "🔴"
                        color = "#ff0000"
                    else:
                        icon = "🟡"
                        color = "#ffcc00"
                    
                    with st.expander(f"{icon} {row['coin_symbol']} - {decision_type} (신뢰도: {row['confidence_score']}%)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**방향:** {row['direction']}")
                            st.markdown(f"**레버리지:** {row['leverage']}x")
                        
                        with col2:
                            st.markdown(f"**신뢰도:** {row['confidence_score']}%")
                            st.markdown(f"**시간:** {row['timestamp']}")
                        
                        with col3:
                            st.markdown(f"**타입:** {decision_type}")
                        
                        st.markdown(f"**💭 AI 분석:**")
                        st.text(row['reasoning'])
        else:
            st.info("사이드바에서 'AI 결정 표시'를 활성화하세요.")
    
    # 탭 5: 실시간 로그
    with tab5:
        st.header("📋 실시간 거래 로그")
        
        log_lines_list = read_log_file(log_lines)
        
        if not log_lines_list:
            st.info("로그 파일이 없거나 비어있습니다.")
        else:
            # 로그 검색
            search_term = st.text_input("🔍 로그 검색", "")
            
            if search_term:
                log_lines_list = [line for line in log_lines_list if search_term.lower() in line.lower()]
                st.info(f"'{search_term}' 검색 결과: {len(log_lines_list)}줄")
            
            # 로그 표시
            log_html = '<div class="log-container">'
            for line in log_lines_list:
                log_html += format_log_line(line)
            log_html += '</div>'
            
            st.markdown(log_html, unsafe_allow_html=True)
            
            # 로그 다운로드
            if st.button("📥 전체 로그 다운로드"):
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    st.download_button(
                        label="💾 로그 파일 저장",
                        data=log_content,
                        file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        mime="text/plain"
                    )
    
    # 자동 새로고침
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
