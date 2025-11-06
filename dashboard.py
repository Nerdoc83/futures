"""
AI Trading Dashboard - Streamlit
실시간 거래 모니터링 대시보드 (DB 전용) - auf_auto_fixed.py용
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
</style>
""", unsafe_allow_html=True)

# 데이터베이스 경로
DB_FILE = "live_trading.db"
LOG_FILE = "trading_bot.log"
BOT_SCRIPT = "auf_auto_fixed.py"

def get_db_connection():
    """DB 연결"""
    try:
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
                coin_symbol,
                side,  /* **수정: 'side as action' 대신 'side' 사용** */
                entry_price,
                quantity as amount,
                leverage,
                position_size as investment_amount,
                stop_loss_price as sl_price,
                take_profit_price as tp_price,
                CASE WHEN manual_trade = 1 THEN 'MANUAL' ELSE 'AUTO' END as trading_style,
                timestamp,
                ai_reasoning,
                ai_confidence as confidence_score
            FROM trades 
            WHERE status = 'OPEN'
            ORDER BY timestamp DESC
        """, conn)
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['source'] = df['trading_style'].apply(
                lambda x: 'MANUAL' if x == 'MANUAL' else 'BOT'
            )
            # # 수정: DataFrame에서 컬럼 이름을 'action'으로 변경
            df.rename(columns={'side': 'action'}, inplace=True) 
        
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
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        df = pd.read_sql_query("""
            SELECT 
                coin_symbol,
                side,  /* **수정: 'side as action' 대신 'side' 사용** */
                entry_price,
                exit_price,
                quantity as amount,
                leverage,
                position_size as investment_amount,
                pnl,
                pnl as binance_pnl,
                pnl_percent as pnl_percentage,
                CASE WHEN manual_trade = 1 THEN 'MANUAL' ELSE 'AUTO' END as trading_style,
                timestamp,
                closed_at as close_timestamp,
                ai_reasoning,
                ai_confidence as confidence_score,
                reason
            FROM trades 
            WHERE status = 'CLOSED'
            AND closed_at >= ?
            ORDER BY closed_at DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['close_timestamp'] = pd.to_datetime(df['close_timestamp'])
            
            # # 수정: DataFrame에서 컬럼 이름을 'action'으로 변경
            df.rename(columns={'side': 'action'}, inplace=True)
            
            # PnL 처리
            df['final_pnl'] = df['pnl']
            df['pnl_source'] = '✅ 바이낸스'
            
            # 청산 이유 추출
            df['close_reason'] = df['reason'].apply(lambda x: x if pd.notna(x) else "청산 완료")
            
            # 이모지 추가
            def add_emoji_to_reason(reason):
                if pd.isna(reason):
                    return "✅ 청산 완료"
                reason_lower = reason.lower()
                
                if 'tp' in reason_lower or '익절' in reason or '목표' in reason or '수익 달성' in reason_lower:
                    return f"🎯 {reason}"
                elif 'sl' in reason_lower or '손절' in reason or '손실' in reason or '손실 제한' in reason_lower:
                    return f"🛑 {reason}"
                elif 'ai' in reason_lower and ('추세' in reason or '변화' in reason):
                    return f"🤖 {reason}"
                elif '동기화' in reason_lower:
                    return f"🔄 {reason}"
                else:
                    return f"✅ {reason}"
            
            df['close_reason_emoji'] = df['close_reason'].apply(add_emoji_to_reason)
        
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
            'profit_factor': 0
        }
    
    total_trades = len(df_closed)
    winning_trades = len(df_closed[df_closed['final_pnl'] > 0])
    losing_trades = len(df_closed[df_closed['final_pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_closed['final_pnl'].sum()
    
    wins = df_closed[df_closed['final_pnl'] > 0]['final_pnl']
    losses = df_closed[df_closed['final_pnl'] < 0]['final_pnl']
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.abs().mean() if len(losses) > 0 else 0 
    largest_win = wins.max() if len(wins) > 0 else 0
    largest_loss = losses.min() if len(losses) > 0 else 0
    
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
    
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
        'profit_factor': profit_factor
    }

def display_dashboard():
    """메인 대시보드 표시"""
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # 자동 새로고침
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # 기간 선택
    period_days = st.sidebar.selectbox(
        "거래 내역 기간",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"최근 {x}일"
    )
    
    # 데이터 로드
    df_open = load_open_positions()
    df_closed = load_closed_trades(days=period_days)
    
    # 통계 계산
    stats = calculate_statistics(df_closed)
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 거래", f"{stats['total_trades']}회")
        st.metric("승률", f"{stats['win_rate']:.1f}%")
    
    with col2:
        pnl_color = "profit" if stats['total_pnl'] >= 0 else "loss"
        st.metric("총 손익", f"${stats['total_pnl']:,.2f}")
        st.metric("승/패", f"{stats['winning_trades']} / {stats['losing_trades']}")
    
    with col3:
        st.metric("평균 수익", f"${stats['avg_win']:,.2f}")
        st.metric("최대 수익", f"${stats['largest_win']:,.2f}")
    
    with col4:
        st.metric("평균 손실", f"${abs(stats['avg_loss']):,.2f}")
        st.metric("최대 손실", f"${stats['largest_loss']:,.2f}")
    
    # 오픈 포지션
    st.header("📊 오픈 포지션")
    if not df_open.empty:
        st.dataframe(
            df_open[['coin_symbol', 'action', 'entry_price', 'amount', 'leverage', 
                     'investment_amount', 'sl_price', 'tp_price', 'source', 'timestamp']],
            use_container_width=True
        )
    else:
        st.info("현재 오픈된 포지션이 없습니다.")
    
    # 거래 내역
    st.header("📜 거래 내역")
    if not df_closed.empty:
        # 거래 내역 테이블
        display_df = df_closed[['coin_symbol', 'action', 'entry_price', 'exit_price', 
                                 'leverage', 'final_pnl', 'pnl_percentage', 
                                 'close_reason_emoji', 'close_timestamp']]
        
        display_df = display_df.rename(columns={
            'coin_symbol': '코인',
            'action': '방향',
            'entry_price': '진입가',
            'exit_price': '청산가',
            'leverage': '레버리지',
            'final_pnl': '손익 ($)',
            'pnl_percentage': '수익률 (%)',
            'close_reason_emoji': '청산 이유',
            'close_timestamp': '청산 시간'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # 차트
        st.subheader("📈 손익 추이")
        df_closed_sorted = df_closed.sort_values('close_timestamp')
        df_closed_sorted['cumulative_pnl'] = df_closed_sorted['final_pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_closed_sorted['close_timestamp'],
            y=df_closed_sorted['cumulative_pnl'],
            mode='lines+markers',
            name='누적 손익',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='누적 손익 추이',
            xaxis_title='시간',
            yaxis_title='손익 (USDT)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info(f"최근 {period_days}일 동안의 거래 내역이 없습니다.")

if __name__ == "__main__":
    display_dashboard()