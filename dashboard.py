"""
AI Trading Dashboard - Streamlit
실시간 거래 모니터링 대시보드 (DB 전용)
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
    """오픈 포지션 로드 (DB 전용)"""
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
        
        # DB에서 오픈 포지션 로드
        db_positions = pd.read_sql_query("""
            SELECT * FROM trades 
            WHERE exit_price IS NULL OR exit_price = 0
            ORDER BY timestamp DESC
        """, conn)
        
        conn.close()
        
        if not db_positions.empty:
            db_positions['timestamp'] = pd.to_datetime(db_positions['timestamp'])
        
        return db_positions
        
    except Exception as e:
        st.error(f"포지션 데이터 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def load_account_info():
    """계정 정보 로드 (DB 전용)"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # 테이블 존재 확인
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='account_info'")
        if not c.fetchone():
            conn.close()
            return None
        
        # DB에서 최신 계정 정보 로드
        df = pd.read_sql_query("""
            SELECT * FROM account_info 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, conn)
        
        conn.close()
        
        if not df.empty:
            return df.iloc[0].to_dict()
        
        return None
        
    except Exception as e:
        st.error(f"계정 정보 로드 오류: {e}")
        if conn:
            conn.close()
        return None

def load_ai_decisions():
    """AI 결정 내역 로드"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # 테이블 존재 확인
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_decisions'")
        if not c.fetchone():
            conn.close()
            return pd.DataFrame()
        
        df = pd.read_sql_query("""
            SELECT * FROM ai_decisions 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, conn)
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        st.error(f"AI 결정 내역 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def calculate_metrics(trades_df):
    """거래 메트릭 계산"""
    metrics = {
        'total_trades': len(trades_df),
        'open_positions': 0,
        'closed_trades': 0,
        'win_rate': 0,
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'total_pnl': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'best_trade': 0,
        'worst_trade': 0
    }
    
    if trades_df.empty:
        return metrics
    
    # 오픈 포지션과 청산된 거래 구분
    open_mask = (trades_df['exit_price'].isna()) | (trades_df['exit_price'] == 0)
    closed_mask = ~open_mask
    
    open_trades = trades_df[open_mask]
    closed_trades = trades_df[closed_mask]
    
    metrics['open_positions'] = len(open_trades)
    metrics['closed_trades'] = len(closed_trades)
    
    # 미실현 손익 (오픈 포지션)
    if not open_trades.empty:
        metrics['unrealized_pnl'] = open_trades['pnl'].sum()
    
    # 실현 손익 (청산된 거래)
    if not closed_trades.empty:
        # 바이낸스 실제 PnL 우선 사용, 없으면 계산된 PnL 사용
        actual_pnl = closed_trades['binance_pnl'].fillna(closed_trades['pnl'])
        metrics['realized_pnl'] = actual_pnl.sum()
        
        # 승률 계산
        wins = actual_pnl > 0
        metrics['win_rate'] = (wins.sum() / len(actual_pnl) * 100) if len(actual_pnl) > 0 else 0
        
        # 수익/손실 통계
        win_trades = actual_pnl[wins]
        loss_trades = actual_pnl[~wins]
        
        metrics['avg_win'] = win_trades.mean() if len(win_trades) > 0 else 0
        metrics['avg_loss'] = loss_trades.mean() if len(loss_trades) > 0 else 0
        metrics['best_trade'] = actual_pnl.max() if len(actual_pnl) > 0 else 0
        metrics['worst_trade'] = actual_pnl.min() if len(actual_pnl) > 0 else 0
    
    metrics['total_pnl'] = metrics['realized_pnl'] + metrics['unrealized_pnl']
    
    return metrics

def check_bot_status():
    """봇 상태 확인"""
    try:
        # psutil이 있으면 사용
        if psutil:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(BOT_SCRIPT in cmd for cmd in cmdline):
                        return {
                            "running": True,
                            "pid": proc.info['pid'],
                            "name": proc.info['name']
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return {"running": False}
        
        # psutil이 없으면 대체 방법 사용 (Linux/Unix)
        else:
            try:
                result = subprocess.run(
                    ['pgrep', '-f', BOT_SCRIPT], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    return {
                        "running": True,
                        "pid": pids[0],
                        "name": "python"
                    }
                else:
                    return {"running": False}
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # pgrep이 없는 경우 ps 명령어 사용
                try:
                    result = subprocess.run(
                        ['ps', 'aux'], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if BOT_SCRIPT in line and 'python' in line:
                                parts = line.split()
                                if len(parts) > 1:
                                    return {
                                        "running": True,
                                        "pid": parts[1],
                                        "name": "python"
                                    }
                    
                    return {"running": False}
                    
                except Exception:
                    return {"running": False, "error": "프로세스 상태 확인 불가"}
    
    except Exception as e:
        return {"running": False, "error": str(e)}

def start_bot():
    """봇 시작"""
    try:
        if not os.path.exists(BOT_SCRIPT):
            return False, f"{BOT_SCRIPT} 파일을 찾을 수 없습니다."
        
        result = subprocess.Popen([sys.executable, BOT_SCRIPT], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
        
        return True, f"봇이 시작되었습니다. PID: {result.pid}"
    
    except Exception as e:
        return False, f"봇 시작 실패: {e}"

def stop_bot():
    """봇 중지"""
    try:
        # psutil이 있으면 사용
        if psutil:
            stopped = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(BOT_SCRIPT in cmd for cmd in cmdline):
                        proc.terminate()
                        stopped = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if stopped:
                return True, "봇이 중지되었습니다."
            else:
                return False, "실행 중인 봇을 찾을 수 없습니다."
        
        # psutil이 없으면 대체 방법 사용
        else:
            try:
                # pgrep으로 PID 찾기
                result = subprocess.run(
                    ['pgrep', '-f', BOT_SCRIPT], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    
                    # 찾은 PID들을 종료
                    for pid in pids:
                        try:
                            subprocess.run(['kill', pid], timeout=5)
                        except Exception:
                            pass
                    
                    return True, f"봇이 중지되었습니다. (PID: {', '.join(pids)})"
                else:
                    return False, "실행 중인 봇을 찾을 수 없습니다."
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False, "봇 중지 기능을 사용할 수 없습니다. 수동으로 프로세스를 종료해주세요."
    
    except Exception as e:
        return False, f"봇 중지 실패: {e}"

def read_log_file(num_lines=100):
    """로그 파일 읽기"""
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 최근 N줄만 반환
        return lines[-num_lines:] if len(lines) > num_lines else lines
    
    except Exception as e:
        st.error(f"로그 파일 읽기 오류: {e}")
        return []

def format_log_line(line):
    """로그 라인 포맷팅"""
    line = line.strip()
    if not line:
        return ""
    
    # 로그 레벨에 따른 색상 적용
    css_class = "log-line"
    if "ERROR" in line:
        css_class += " log-error"
    elif "WARNING" in line:
        css_class += " log-warning"
    elif "INFO" in line:
        css_class += " log-info"
    elif "SUCCESS" in line:
        css_class += " log-success"
    
    # 타임스탬프 강조
    if line.startswith("2024") or line.startswith("2025"):
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            timestamp = f"{parts[0]} {parts[1]}"
            rest = parts[2] if len(parts) > 2 else ""
            line = f'<span class="log-timestamp">{timestamp}</span> {rest}'
    
    return f'<div class="{css_class}">{line}</div>'

def main():
    """메인 함수"""
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard (DB 전용)</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 새로고침 설정
        auto_refresh = st.checkbox("자동 새로고침 (10초)", value=False)
        show_ai_decisions = st.checkbox("AI 결정 표시", value=True)
        log_lines = st.slider("로그 라인 수", 50, 500, 100)
        
        st.divider()
        
        # 봇 상태
        st.header("🤖 봇 상태")
        bot_status = check_bot_status()
        
        if bot_status.get("running", False):
            st.success(f"✅ 실행 중 (PID: {bot_status.get('pid', 'N/A')})")
            if st.button("🛑 봇 중지", type="secondary"):
                success, message = stop_bot()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.error("❌ 중지됨")
            if st.button("▶️ 봇 시작", type="primary"):
                success, message = start_bot()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        if "error" in bot_status:
            st.warning(f"상태 확인 오류: {bot_status['error']}")
        
        st.divider()
        
        # 새로고침 버튼
        if st.button("🔄 수동 새로고침"):
            st.rerun()
    
    # 데이터 로드
    trades_df = load_trades()
    positions_df = load_open_positions()
    account_info = load_account_info()
    
    # 메트릭 계산
    metrics = calculate_metrics(trades_df)
    
    # 메인 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 총 손익", 
            f"${metrics['total_pnl']:,.2f}",
            delta=f"${metrics['realized_pnl']:,.2f}" if metrics['realized_pnl'] != 0 else None
        )
    
    with col2:
        st.metric(
            "📊 승률", 
            f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['closed_trades']}회 거래"
        )
    
    with col3:
        st.metric(
            "📈 오픈 포지션", 
            f"{metrics['open_positions']}개",
            delta=f"${metrics['unrealized_pnl']:,.2f}" if metrics['unrealized_pnl'] != 0 else None
        )
    
    with col4:
        if account_info:
            balance = account_info.get('total_balance', 0)
            st.metric(
                "💳 잔고", 
                f"${balance:,.2f}",
                delta=f"사용 가능: ${account_info.get('available_balance', 0):,.2f}"
            )
        else:
            st.metric("💳 잔고", "정보 없음")
    
    # 탭 레이아웃
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 대시보드", "💼 오픈 포지션", "📈 성과 분석", "🤖 AI 결정", "📋 로그"])
    
    # 탭 1: 대시보드
    with tab1:
        if trades_df.empty:
            st.info("거래 데이터가 없습니다. 봇이 실행되면 데이터가 표시됩니다.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # 최근 거래 현황
                st.subheader("🕒 최근 거래")
                recent_trades = trades_df.head(10)
                
                for idx, trade in recent_trades.iterrows():
                    with st.container():
                        profit_loss = trade.get('binance_pnl', trade.get('pnl', 0))
                        color = "profit" if profit_loss > 0 else "loss"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{trade['coin_symbol']}</strong> - {trade['action']} 
                            ({trade.get('trading_style', 'N/A')})
                            <br>
                            <small>{trade['timestamp']}</small>
                            <br>
                            <span class="{color}">
                                ${profit_loss:,.2f}
                            </span>
                            {' ✅' if trade.get('status') == 'CLOSED' else ' 🔄'}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # 코인별 성과
                st.subheader("🪙 코인별 성과")
                if not trades_df.empty:
                    coin_performance = trades_df.groupby('coin_symbol').agg({
                        'pnl': 'sum',
                        'action': 'count'
                    }).round(2)
                    coin_performance.columns = ['총 손익', '거래 수']
                    coin_performance = coin_performance.sort_values('총 손익', ascending=False)
                    
                    st.dataframe(coin_performance.head(10), use_container_width=True)
                else:
                    st.info("거래 데이터가 없습니다.")
    
    # 탭 2: 오픈 포지션
    with tab2:
        st.header("💼 현재 오픈 포지션")
        
        if positions_df.empty:
            st.info("현재 오픈 포지션이 없습니다.")
        else:
            st.success(f"총 {len(positions_df)}개의 오픈 포지션")
            
            for idx, pos in positions_df.iterrows():
                with st.expander(f"{pos['coin_symbol']} - {pos['action']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("진입가", f"${pos['entry_price']:,.4f}")
                        st.metric("수량", f"{pos['amount']:,.4f}")
                    
                    with col2:
                        current_price = pos.get('current_price', 0)
                        st.metric("현재가", f"${current_price:,.4f}" if current_price > 0 else "정보 없음")
                        st.metric("레버리지", f"{pos.get('leverage', 1)}x")
                    
                    with col3:
                        pnl = pos.get('pnl', 0)
                        pnl_color = "normal" if pnl >= 0 else "inverse"
                        st.metric("미실현 손익", f"${pnl:,.2f}", delta_color=pnl_color)
                        
                        investment = pos.get('investment_amount', 0)
                        if investment > 0:
                            roi = (pnl / investment * 100)
                            st.metric("수익률", f"{roi:,.2f}%")
                    
                    with col4:
                        st.metric("거래 스타일", pos.get('trading_style', 'N/A'))
                        st.metric("진입 시간", pos['timestamp'].strftime('%m-%d %H:%M'))
    
    # 탭 3: 성과 분석
    with tab3:
        st.header("📈 거래 성과 분석")
        
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
                style_stats = closed_df.groupby('trading_style').agg({
                    'actual_pnl': ['sum', 'mean', 'count']
                }).round(2)
                style_stats.columns = ['총 손익', '평균 손익', '거래 수']
                st.dataframe(style_stats, use_container_width=True)
            
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
