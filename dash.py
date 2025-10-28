"""
AI Trading Dashboard - Streamlit
실시간 거래 모니터링 대시보드 (DB 전용)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation.*')

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

# API 초기화 코드를 제거했으므로, 관련된 라이브러리 임포트 제거 (ccxt, dotenv)

# 바이낸스 API 초기화 제거
# exchange = None
# BINANCE_AVAILABLE = False # 이제 항상 False와 동일하게 동작

# 페이지 설정
st.set_page_config(
    page_title="AI Trading Dashboard (DB Only)",
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

@st.cache_data(ttl=30)  # 30초간 캐시 (거래 데이터는 자주 변경)
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

@st.cache_data(ttl=60)  # 60초간 캐시
def load_open_positions():
    """오픈 포지션 로드 (DB만 사용)"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # 테이블 존재 확인
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not c.fetchone():
            db_positions = pd.DataFrame()
        else:
            # 안전한 쿼리 (exit_price가 NULL 또는 0인 경우)
            db_positions = pd.read_sql_query("""
                SELECT * FROM trades 
                WHERE exit_price IS NULL OR exit_price = 0
                ORDER BY timestamp DESC
            """, conn)
            
            if not db_positions.empty:
                db_positions['timestamp'] = pd.to_datetime(db_positions['timestamp'])
                # DB 전용 대시보드이므로 source를 'BOT'으로 설정
                db_positions['source'] = 'BOT'
                # 바이낸스 데이터가 없으므로 관련 컬럼 초기화
                db_positions['current_price'] = None
                db_positions['unrealized_pnl'] = None
                db_positions['notional'] = None
        
        conn.close()
        return db_positions
    except Exception as e:
        st.error(f"DB 포지션 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)  # 60초간 캐시
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

# 바이낸스 계정 정보 조회 함수 제거
# def get_binance_account_info():
#     """바이낸스 계정 정보 조회"""
#     ...

# 바이낸스 거래 내역 조회 함수 제거
# def get_binance_trade_history(days=30, limit=1000):
#     """바이낸스 거래 내역 조회"""
#     ...

# 바이낸스 수익 내역 조회 함수 제거
# def get_binance_income_history(days=30):
#     """바이낸스 수익 내역 조회 (realized PnL)"""
#     ...

# 바이낸스 실제 데이터로 메트릭 계산 함수 제거
# def calculate_binance_metrics(income_df, trades_df, open_positions_df):
#     """바이낸스 실제 데이터로 메트릭 계산"""
#     ...

def calculate_db_metrics(trades_df):
    """DB 데이터로 성과 지표 계산"""
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
    
    # status 컬럼을 기준으로 OPEN/CLOSED 분류 (trades 테이블의 일반적인 스키마를 가정)
    if 'status' in trades_df.columns:
        open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
        closed_trades = len(trades_df[trades_df['status'] == 'CLOSED'])
        closed_df = trades_df[trades_df['status'] == 'CLOSED'].copy()
    else:
        # status 컬럼이 없으면 exit_price 기준으로 판단
        open_trades = len(trades_df[trades_df['exit_price'].isna() | (trades_df['exit_price'] == 0)])
        closed_trades = len(trades_df[trades_df['exit_price'].notna() & (trades_df['exit_price'] != 0)])
        closed_df = trades_df[trades_df['exit_price'].notna() & (trades_df['exit_price'] != 0)].copy()

    
    if not closed_df.empty:
        # DB의 pnl 컬럼 사용 (pnl 또는 binance_pnl 중 사용 가능한 것)
        if 'binance_pnl' in closed_df.columns:
            # binance_pnl이 있다면 우선 사용 (DB에 저장된 실제 PnL 가정)
            closed_df['actual_pnl'] = closed_df['binance_pnl'].fillna(closed_df['pnl'])
        elif 'pnl' in closed_df.columns:
            closed_df['actual_pnl'] = closed_df['pnl']
        else:
            # PnL 컬럼이 없으면 계산 불가
            closed_df['actual_pnl'] = 0 
            
        closed_df['actual_pnl'] = closed_df['actual_pnl'].fillna(0) # 결측값 0 처리
        
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
        # close_timestamp가 있는 경우에만 계산
        if 'close_timestamp' in closed_df.columns and closed_df['close_timestamp'].notna().any():
            closed_df['close_timestamp'] = pd.to_datetime(closed_df['close_timestamp'])
            # close_timestamp가 유효한 값인 경우에만 계산
            valid_holding_times = (closed_df['close_timestamp'] - closed_df['timestamp']).dt.total_seconds() / 3600
            avg_holding_time = valid_holding_times.mean()
        else:
            avg_holding_time = 0
            
    else:
        winning_trades = losing_trades = 0
        win_rate = realized_pnl = avg_win = avg_loss = best_trade = worst_trade = avg_holding_time = 0
    
    # 미실현 손익 (DB 데이터만으로는 정확한 실시간 미실현 손익 계산 불가. 0으로 처리.)
    unrealized_pnl = 0 
    
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
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard (DB 전용)</h1>', unsafe_allow_html=True)
    
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
        auto_refresh = st.checkbox("자동 새로고침 (60초)", value=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 자동 새로고침은 서버 부하를 유발할 수 있습니다. 필요시에만 활성화하세요.")
        with col2:
            if st.button("🔄 수동 새로고침", use_container_width=True):
                st.rerun()
        
        # 이전에 버튼이 중복되어 있었으므로 하나 제거
        # if st.button("🔄 수동 새로고침", use_container_width=True):
        #     st.rerun()
        
        st.markdown("---")
        
        st.header("📊 표시 옵션")
        show_closed = st.checkbox("청산된 거래 표시", value=True)
        show_ai_decisions = st.checkbox("AI 결정 표시", value=False)
        log_lines = st.slider("로그 라인 수", 50, 500, 200, 50)
        
        # 바이낸스 계정 정보 표시 제거
        # if account_info:
        #     st.sidebar.markdown("---")
        #     st.sidebar.subheader("💰 바이낸스 계정")
        #     st.sidebar.metric("총 잔고", f"${account_info['total_balance']:,.2f}")
        #     st.sidebar.metric("가용 잔고", f"${account_info['available_balance']:,.2f}")
        #     st.sidebar.metric("사용 중", f"${account_info['used_balance']:,.2f}")
    
    # 데이터 로드 (DB만 사용)
    trades_df = load_trades()
    open_positions_df = load_open_positions()
    
    # 바이낸스 실제 데이터 사용 체크박스 제거
    # use_binance_data = st.sidebar.checkbox(...)
    
    # DB 데이터 사용으로 통일
    metrics = calculate_db_metrics(trades_df)
    
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
            "총 손익 (DB)",
            f"${metrics['total_pnl']:,.2f}",
            delta=f"{metrics['total_pnl']:+.2f}",
            delta_color=pnl_color
        )
    
    with col3:
        # 미실현 손익은 0으로 표시됨
        st.metric(
            "실현 손익 (DB)",
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
        st.header("📊 현재 오픈 포지션 (DB 기반)")
        
        if open_positions_df.empty:
            st.info("현재 오픈된 포지션이 없습니다.")
        else:
            # DB 기반이므로 모든 포지션은 'BOT' 소스로 간주
            bot_positions = open_positions_df
            
            # 수동 포지션 관련 UI 제거
            # if not manual_positions.empty: ...
            # st.markdown("---")
            
            if not bot_positions.empty:
                st.subheader(f"🤖 봇 포지션 ({len(bot_positions)}개)")
                for idx, row in bot_positions.iterrows():
                    side_emoji = "🟢" if row['action'] == 'long' else "🔴"
                    style_emoji = {"SCALPING": "⚡", "DAY_TRADING": "📊", "SWING_TRADING": "📈"}.get(row['trading_style'], "📊")
                    
                    # DB만 사용하므로 현재 가격과 미실현 손익은 표시 불가 (None)
                    pnl_display = ""
                    
                    with st.expander(f"{side_emoji} {row['coin_symbol']} {row['action'].upper()} - {style_emoji} {row['trading_style']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("진입가", f"${row['entry_price']:,.4f}")
                            # 현재가, 미실현 손익은 DB만으로는 표시 불가
                            st.caption("현재가: (API 없음)")
                            st.metric("레버리지", f"{row['leverage']}x")
                        
                        with col2:
                            st.metric("투자금", f"${row['investment_amount']:,.2f}")
                            st.metric("수량", f"{row['amount']:.8f}")
                            # 포지션 크기 (notional)는 DB만으로는 표시 불가
                            st.caption("포지션 크기: (API 없음)")
                        
                        with col3:
                            # 미실현 손익은 0으로 표시
                            st.caption("미실현 손익: $0.00 (API 없음)")
                            
                            if pd.notna(row['sl_price']):
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
            total_investment = open_positions_df['investment_amount'].sum() if not open_positions_df.empty else 0
            st.info(f"📊 총 {len(open_positions_df)}개 포지션 (DB 기반) | 총 투자금: ${total_investment:,.2f}")
    
    # 탭 2: 거래 내역 (DB만 사용)
    with tab2:
        st.header("📜 전체 거래 내역 (DB 기반)")
        
        if trades_df.empty:
            st.info("거래 내역이 없습니다.")
        else:
            # 필터
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # status 컬럼이 없는 경우를 대비하여 기본값 처리
                status_options = trades_df['status'].dropna().unique().tolist() if 'status' in trades_df.columns else ['OPEN', 'CLOSED']
                status_filter = st.multiselect(
                    "상태 필터",
                    options=status_options,
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
                (trades_df['action'].isin(direction_filter))
            )
            
            # status 컬럼 필터링
            if 'status' in trades_df.columns:
                filter_conditions = filter_conditions & (trades_df['status'].isin(status_filter))
            # trading_style 컬럼 필터링
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
            elif 'pnl' in filtered_df.columns:
                display_columns.append('pnl')
            
            if 'pnl_percentage' in filtered_df.columns:
                display_columns.append('pnl_percentage')
            
            if 'status' in filtered_df.columns:
                display_columns.append('status')
                
            display_columns.append('timestamp')
            
            # 손절/익절가 추가 (있는 경우)
            if 'sl_price' in filtered_df.columns:
                display_columns.insert(display_columns.index('leverage') + 1, 'sl_price')
            if 'tp_price' in filtered_df.columns:
                display_columns.insert(display_columns.index('leverage') + 2, 'tp_price')
            
            # 최종적으로 필터링된 컬럼만 선택
            display_df = filtered_df[[col for col in display_columns if col in filtered_df.columns]].copy()
            
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
                'actual_pnl': '손익', # DB pnl/binance_pnl 통합
                'pnl_percentage': '수익률%',
                'status': '상태',
                'timestamp': '시간'
            }
            display_df = display_df.rename(columns={k: v for k, v in column_names.items() if k in display_df.columns})
            
            # PnL 색상 적용
            def highlight_pnl(val):
                if pd.isna(val):
                    return ''
                # 문자열이 아닌 경우에만 비교
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                    return f'color: {color}; font-weight: bold'
                return ''
            
            # 손익 컬럼에 스타일 적용
            pnl_columns = [col for col in display_df.columns if '손익' in col or '수익률' in col]
            
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
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}_db_only.csv",
                mime="text/csv"
            )
    
    # 탭 3: 성과 분석 (DB만 사용)
    with tab3:
        st.header("📈 성과 분석 (DB 기반)")
        
        # 데이터 소스 표시 제거
        # if use_binance_data and BINANCE_AVAILABLE: ...
        
        if trades_df.empty or metrics['closed_trades'] == 0:
            st.info("청산된 거래가 없어 성과 분석을 표시할 수 없습니다.")
        else:
            closed_df = trades_df[trades_df['status'] == 'CLOSED'].copy()
            
            # PnL 컬럼 처리
            if 'binance_pnl' in closed_df.columns:
                closed_df['actual_pnl'] = closed_df['binance_pnl'].fillna(closed_df['pnl'])
            elif 'pnl' in closed_df.columns:
                closed_df['actual_pnl'] = closed_df['pnl']
            else:
                closed_df['actual_pnl'] = 0
            
            closed_df['actual_pnl'] = closed_df['actual_pnl'].fillna(0) # 결측값 0 처리
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PnL 분포
                st.subheader("💰 손익 분포")
                fig_pnl = px.histogram(
                    closed_df,
                    x='actual_pnl',
                    nbins=20,
                    title='거래별 손익 분포 (DB 데이터)',
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
                    title='시간별 누적 손익 (DB 데이터)',
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
                total_invested = closed_df['investment_amount'].sum() if 'investment_amount' in closed_df.columns else 0
                roi = (metrics['realized_pnl'] / total_invested * 100) if total_invested > 0 else 0
                st.metric("ROI (DB 기반)", f"{roi:.2f}%")
                st.metric("총 투자금", f"${total_invested:,.2f}")
    
    # 탭 4: AI 결정 (DB만 사용)
    with tab4:
        if show_ai_decisions:
            st.header("🤖 AI 결정 내역 (DB 기반)")
            
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
                        file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_db_only.log",
                        mime="text/plain"
                    )
    
    # 자동 새로고침 (60초 간격)
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()