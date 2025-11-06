"""
AI Trading Dashboard - Streamlit (Fixed)
실시간 거래 모니터링 대시보드 (auf_auto_fixed.py DB 스키마에 맞춤)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import subprocess
import sys

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
    .status-open {
        color: #ffa500;
        font-weight: bold;
    }
    .status-closed {
        color: #808080;
        font-weight: bold;
    }
    .manual-trade {
        background-color: #e6f3ff;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        color: #0066cc;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .auto-trade {
        background-color: #e6ffe6;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        color: #008000;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 경로
DB_FILE = "live_trading.db"
LOG_FILE = "trading_bot.log"
BOT_SCRIPT = "auf_auto_fixed.py"

def get_db_connection():
    """DB 연결"""
    try:
        if not os.path.exists(DB_FILE):
            st.error(f"데이터베이스 파일이 없습니다: {DB_FILE}")
            return None
        return sqlite3.connect(DB_FILE)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return None

@st.cache_data(ttl=30)
def load_open_positions():
    """오픈 포지션 로드 (auf_auto_fixed.py DB 스키마용)"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT 
                id,
                coin_symbol,
                side,
                entry_price,
                quantity,
                leverage,
                position_size,
                stop_loss_price,
                take_profit_price,
                CASE WHEN manual_trade = 1 THEN 'MANUAL' ELSE 'AUTO' END as trade_type,
                timestamp,
                ai_reasoning,
                ai_confidence
            FROM trades 
            WHERE status = 'OPEN'
            ORDER BY timestamp DESC
        """, conn)
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 포지션 값 계산
            df['position_value'] = df['position_size'] * df['leverage']
            
            # 수동/자동 거래 구분
            df['trade_source'] = df['trade_type'].apply(
                lambda x: '🛡️ 수동거래' if x == 'MANUAL' else '🤖 AI거래'
            )
        
        return df
        
    except Exception as e:
        st.error(f"포지션 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_closed_trades(days=30):
    """청산된 거래 로드 (auf_auto_fixed.py DB 스키마용)"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # 최근 N일간의 거래만 조회
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        df = pd.read_sql_query("""
            SELECT 
                id,
                coin_symbol,
                side,
                entry_price,
                exit_price,
                quantity,
                leverage,
                position_size,
                pnl,
                pnl_percent,
                exit_reason,
                CASE WHEN manual_trade = 1 THEN 'MANUAL' ELSE 'AUTO' END as trade_type,
                timestamp as entry_time,
                ai_reasoning,
                ai_confidence
            FROM trades 
            WHERE status = 'CLOSED'
            AND timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        if not df.empty:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            
            # 거래 결과 분류
            df['result'] = df['pnl'].apply(lambda x: '💰 수익' if x > 0 else '💸 손실' if x < 0 else '⚖️ 무승부')
            
            # 청산 이유 이모지 추가
            def add_emoji_to_reason(reason):
                if pd.isna(reason) or reason == '':
                    return "✅ 청산 완료"
                
                reason_lower = str(reason).lower()
                
                if '트레일링' in reason_lower or 'trailing' in reason_lower:
                    return f"🎯 {reason}"
                elif 'ai' in reason_lower and ('손절' in reason_lower or '청산' in reason_lower):
                    return f"🤖 {reason}"
                elif '동기화' in reason_lower:
                    return f"🔄 {reason}"
                elif '수동' in reason_lower:
                    return f"👤 {reason}"
                else:
                    return f"✅ {reason}"
            
            df['exit_reason_emoji'] = df['exit_reason'].apply(add_emoji_to_reason)
            
            # 거래 소스
            df['trade_source'] = df['trade_type'].apply(
                lambda x: '🛡️ 수동거래' if x == 'MANUAL' else '🤖 AI거래'
            )
        
        return df
        
    except Exception as e:
        st.error(f"거래 내역 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def calculate_statistics(df_closed):
    """거래 통계 계산"""
    if df_closed.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'avg_pnl_percent': 0,
            'auto_trades': 0,
            'manual_trades': 0
        }
    
    total_trades = len(df_closed)
    winning_trades = len(df_closed[df_closed['pnl'] > 0])
    losing_trades = len(df_closed[df_closed['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_closed['pnl'].sum()
    avg_pnl_percent = df_closed['pnl_percent'].mean()
    
    wins = df_closed[df_closed['pnl'] > 0]['pnl']
    losses = df_closed[df_closed['pnl'] < 0]['pnl']
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0  # 음수 그대로
    largest_win = wins.max() if len(wins) > 0 else 0
    largest_loss = losses.min() if len(losses) > 0 else 0
    
    # Profit Factor 계산
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
    
    # 거래 타입별 분류
    auto_trades = len(df_closed[df_closed['trade_type'] == 'AUTO'])
    manual_trades = len(df_closed[df_closed['trade_type'] == 'MANUAL'])
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'avg_pnl_percent': avg_pnl_percent,
        'auto_trades': auto_trades,
        'manual_trades': manual_trades
    }

def get_bot_status():
    """봇 실행 상태 확인"""
    try:
        if psutil is None:
            return "❓ 상태 확인 불가 (psutil 없음)"
        
        # auf_auto_fixed.py 프로세스 찾기
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('auf_auto_fixed.py' in arg for arg in cmdline):
                    return f"✅ 실행 중 (PID: {proc.info['pid']})"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return "🔴 정지됨"
    except Exception as e:
        return f"❓ 상태 확인 오류: {e}"

def display_bot_control():
    """봇 제어 패널"""
    st.sidebar.header("🤖 봇 제어")
    
    # 봇 상태 표시
    bot_status = get_bot_status()
    st.sidebar.markdown(f"**상태:** {bot_status}")
    
    # 봇 시작/정지 버튼
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 봇 시작", disabled=("실행 중" in bot_status)):
            try:
                subprocess.Popen([sys.executable, BOT_SCRIPT])
                st.sidebar.success("봇을 시작했습니다!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"봇 시작 실패: {e}")
    
    with col2:
        if st.button("⏹️ 봇 정지", disabled=("정지됨" in bot_status)):
            try:
                if psutil:
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            cmdline = proc.info['cmdline']
                            if cmdline and any('auf_auto_fixed.py' in arg for arg in cmdline):
                                proc.terminate()
                                st.sidebar.success("봇을 정지했습니다!")
                                time.sleep(2)
                                st.rerun()
                                break
                    else:
                        st.sidebar.warning("실행 중인 봇을 찾을 수 없습니다.")
                else:
                    st.sidebar.error("psutil이 설치되지 않아 봇을 정지할 수 없습니다.")
            except Exception as e:
                st.sidebar.error(f"봇 정지 실패: {e}")

def display_dashboard():
    """메인 대시보드 표시"""
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 자동 새로고침
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=False)
    refresh_interval = st.sidebar.slider("새로고침 간격 (초)", 10, 300, 30)
    
    # 기간 선택
    period_days = st.sidebar.selectbox(
        "거래 내역 기간",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"최근 {x}일"
    )
    
    # 봇 제어 패널
    display_bot_control()
    
    # 데이터 로드
    df_open = load_open_positions()
    df_closed = load_closed_trades(days=period_days)
    
    # 통계 계산
    stats = calculate_statistics(df_closed)
    
    # === 상단 메트릭 ===
    st.header("📊 거래 요약")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("총 거래", f"{stats['total_trades']}회")
        if stats['total_trades'] > 0:
            st.caption(f"🤖 AI: {stats['auto_trades']} | 🛡️ 수동: {stats['manual_trades']}")
    
    with col2:
        st.metric("승률", f"{stats['win_rate']:.1f}%")
        st.caption(f"✅ {stats['winning_trades']} / ❌ {stats['losing_trades']}")
    
    with col3:
        pnl_delta = f"{stats['avg_pnl_percent']:+.2f}%" if stats['total_trades'] > 0 else None
        st.metric("총 손익", f"${stats['total_pnl']:,.2f}", delta=pnl_delta)
    
    with col4:
        st.metric("평균 수익", f"${stats['avg_win']:,.2f}")
        st.metric("최대 수익", f"${stats['largest_win']:,.2f}")
    
    with col5:
        st.metric("평균 손실", f"${stats['avg_loss']:,.2f}")
        st.metric("최대 손실", f"${stats['largest_loss']:,.2f}")
    
    # Profit Factor
    if stats['profit_factor'] > 0:
        st.info(f"📈 Profit Factor: {stats['profit_factor']:.2f} " + 
                ("(우수)" if stats['profit_factor'] > 1.5 else "(보통)" if stats['profit_factor'] > 1.0 else "(개선필요)"))
    
    # === 오픈 포지션 ===
    st.header("🎯 오픈 포지션")
    if not df_open.empty:
        # 포지션별 현재 상태 (가격은 실시간이 아님을 안내)
        st.info("💡 현재 가격과 PnL은 실시간이 아닙니다. 바이낸스에서 실시간 확인하세요.")
        
        # 테이블 표시
        display_df = df_open[['id', 'coin_symbol', 'side', 'entry_price', 'quantity', 
                              'leverage', 'position_size', 'trade_source', 'timestamp']].copy()
        
        display_df = display_df.rename(columns={
            'id': 'ID',
            'coin_symbol': '코인',
            'side': '방향',
            'entry_price': '진입가',
            'quantity': '수량',
            'leverage': '레버리지',
            'position_size': '포지션 크기 ($)',
            'trade_source': '거래 타입',
            'timestamp': '진입 시간'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # 포지션 요약
        total_positions = len(df_open)
        total_investment = df_open['position_size'].sum()
        auto_positions = len(df_open[df_open['trade_type'] == 'AUTO'])
        manual_positions = len(df_open[df_open['trade_type'] == 'MANUAL'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 포지션", f"{total_positions}개")
        with col2:
            st.metric("총 투자금", f"${total_investment:,.2f}")
        with col3:
            st.caption(f"🤖 AI: {auto_positions}개 | 🛡️ 수동: {manual_positions}개")
    else:
        st.info("현재 오픈된 포지션이 없습니다.")
    
    # === 거래 내역 ===
    st.header("📜 거래 내역")
    if not df_closed.empty:
        # 필터 옵션
        col1, col2 = st.columns(2)
        with col1:
            trade_type_filter = st.selectbox(
                "거래 타입",
                options=['전체', 'AI거래', '수동거래'],
                index=0
            )
        with col2:
            result_filter = st.selectbox(
                "거래 결과",
                options=['전체', '수익', '손실'],
                index=0
            )
        
        # 필터 적용
        filtered_df = df_closed.copy()
        
        if trade_type_filter == 'AI거래':
            filtered_df = filtered_df[filtered_df['trade_type'] == 'AUTO']
        elif trade_type_filter == '수동거래':
            filtered_df = filtered_df[filtered_df['trade_type'] == 'MANUAL']
        
        if result_filter == '수익':
            filtered_df = filtered_df[filtered_df['pnl'] > 0]
        elif result_filter == '손실':
            filtered_df = filtered_df[filtered_df['pnl'] < 0]
        
        # 거래 내역 테이블
        if not filtered_df.empty:
            display_df = filtered_df[['id', 'coin_symbol', 'side', 'entry_price', 'exit_price', 
                                     'leverage', 'pnl', 'pnl_percent', 'exit_reason_emoji', 
                                     'trade_source', 'entry_time']].copy()
            
            display_df = display_df.rename(columns={
                'id': 'ID',
                'coin_symbol': '코인',
                'side': '방향',
                'entry_price': '진입가',
                'exit_price': '청산가',
                'leverage': '레버리지',
                'pnl': '손익 ($)',
                'pnl_percent': '수익률 (%)',
                'exit_reason_emoji': '청산 이유',
                'trade_source': '거래 타입',
                'entry_time': '진입 시간'
            })
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("필터 조건에 맞는 거래가 없습니다.")
        
        # === 손익 차트 ===
        st.subheader("📈 손익 추이")
        
        if not df_closed.empty:
            # 시간순 정렬
            df_sorted = df_closed.sort_values('entry_time')
            df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
            
            # 라인 차트
            fig = go.Figure()
            
            # 누적 손익
            fig.add_trace(go.Scatter(
                x=df_sorted['entry_time'],
                y=df_sorted['cumulative_pnl'],
                mode='lines+markers',
                name='누적 손익',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{y:.2f} USDT</b><br>%{x}<extra></extra>'
            ))
            
            # 제로라인
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title='누적 손익 추이',
                xaxis_title='시간',
                yaxis_title='손익 (USDT)',
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # === 거래 분포 차트 ===
            col1, col2 = st.columns(2)
            
            with col1:
                # 수익/손실 분포
                profit_dist = df_closed['pnl'].apply(lambda x: '수익' if x > 0 else '손실' if x < 0 else '무승부')
                fig_pie = px.pie(
                    values=profit_dist.value_counts().values,
                    names=profit_dist.value_counts().index,
                    title='거래 결과 분포',
                    color_discrete_map={'수익': '#00cc00', '손실': '#ff0000', '무승부': '#808080'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # AI vs 수동거래 분포
                trade_type_dist = df_closed['trade_source']
                fig_pie2 = px.pie(
                    values=trade_type_dist.value_counts().values,
                    names=trade_type_dist.value_counts().index,
                    title='거래 타입 분포',
                    color_discrete_map={'🤖 AI거래': '#4CAF50', '🛡️ 수동거래': '#2196F3'}
                )
                st.plotly_chart(fig_pie2, use_container_width=True)
        
    else:
        st.info(f"최근 {period_days}일 동안의 거래 내역이 없습니다.")
    
    # === 로그 확인 ===
    with st.expander("📋 최근 로그 (최근 50줄)", expanded=False):
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    log_content = ''.join(recent_lines)
                    st.text_area("로그 내용", value=log_content, height=300)
            except Exception as e:
                st.error(f"로그 파일 읽기 오류: {e}")
        else:
            st.info("로그 파일이 없습니다.")
    
    # 자동 새로고침
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    display_dashboard()
