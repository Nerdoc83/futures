# dashboard.py

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="ETH Trading AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터베이스 연결 및 데이터 로딩 함수 ---
@st.cache_data(ttl=60)  # 60초마다 데이터 캐시 만료
def load_data():
    """
    ethereum_daytrading.db에서 'trades'와 'ai_analysis' 테이블 데이터를 로드하고 전처리합니다.
    """
    try:
        conn = sqlite3.connect('ethereum_daytrading.db', timeout=10)
        
        # SQL 쿼리
        trades_query = "SELECT * FROM trades ORDER BY timestamp DESC"
        ai_analysis_query = "SELECT * FROM ai_analysis ORDER BY timestamp DESC"
        
        # 데이터 로드
        trades_df = pd.read_sql_query(trades_query, conn)
        ai_analysis_df = pd.read_sql_query(ai_analysis_query, conn)
        
    except sqlite3.OperationalError as e:
        st.error(f"❌ 데이터베이스 연결에 실패했습니다: {e}")
        st.info("데이터베이스 파일('ethereum_daytrading.db')이 스크립트와 동일한 디렉토리에 있는지 확인하세요.")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    # --- 데이터 전처리 ---
    if not trades_df.empty:
        # 타임스탬프 변환
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
        trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'], errors='coerce')
        
        # 숫자형 데이터 변환 및 오류 처리
        numeric_cols = ['entry_price', 'exit_price', 'amount', 'leverage', 'investment_amount', 'profit_loss', 'profit_loss_percentage']
        for col in numeric_cols:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

    if not ai_analysis_df.empty:
        ai_analysis_df['timestamp'] = pd.to_datetime(ai_analysis_df['timestamp'], errors='coerce')

    return trades_df.dropna(subset=['timestamp']), ai_analysis_df.dropna(subset=['timestamp'])

# --- 핵심 성과 지표(KPI) 계산 함수 ---
def calculate_metrics(df):
    """
    거래 데이터프레임을 기반으로 주요 성과 지표를 계산합니다.
    """
    metrics = {
        'total_pnl': 0, 'win_rate': 0, 'total_trades': len(df),
        'profit_factor': 0, 'avg_pnl_percent': 0, 'winning_trades': 0, 'losing_trades': 0
    }
    
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    if closed_trades.empty:
        return metrics

    # Null 값이 아닌 손익 데이터만 사용
    pnl_data = closed_trades['profit_loss'].dropna()
    
    metrics['total_pnl'] = pnl_data.sum()
    
    winning_trades = pnl_data[pnl_data > 0]
    losing_trades = pnl_data[pnl_data < 0]
    
    metrics['winning_trades'] = len(winning_trades)
    metrics['losing_trades'] = len(losing_trades)
    
    if (metrics['winning_trades'] + metrics['losing_trades']) > 0:
        metrics['win_rate'] = (metrics['winning_trades'] / (metrics['winning_trades'] + metrics['losing_trades'])) * 100

    total_profit = winning_trades.sum()
    total_loss = abs(losing_trades.sum())

    if total_loss > 0:
        metrics['profit_factor'] = total_profit / total_loss
    elif total_profit > 0:
        metrics['profit_factor'] = float('inf')  # 손실이 없으면 무한대

    if not closed_trades['profit_loss_percentage'].dropna().empty:
        metrics['avg_pnl_percent'] = closed_trades['profit_loss_percentage'].dropna().mean()

    return metrics
    
# --- 메인 대시보드 ---
def run_dashboard():
    st.title("🤖 이더리움 AI 트레이딩 대시보드")

    # 데이터 로드
    trades_df, ai_analysis_df = load_data()

    if trades_df.empty:
        st.warning("⚠️ 거래 데이터가 없습니다. 트레이딩 봇이 실행 중인지 확인해주세요.")
        return

    # --- 사이드바 ---
    st.sidebar.header("⚙️ 필터 및 설정")
    
    # 날짜 범위 필터
    min_date = trades_df['timestamp'].min().date()
    max_date = trades_df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "📅 날짜 범위 선택",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) != 2:
        st.sidebar.warning("올바른 날짜 범위를 선택하세요.")
        return

    start_date, end_date = date_range
    
    # 데이터 필터링
    filtered_df = trades_df[
        (trades_df['timestamp'].dt.date >= start_date) &
        (trades_df['timestamp'].dt.date <= end_date)
    ]
    
    # --- 핵심 성과 지표(KPI) 표시 ---
    st.subheader("📈 주요 성과 지표")
    metrics = calculate_metrics(filtered_df)
    
    cols = st.columns(4)
    cols[0].metric("💰 총 손익 (USDT)", f"${metrics['total_pnl']:,.2f}")
    cols[1].metric("🎯 승률", f"{metrics['win_rate']:.2f}%", f"{metrics['winning_trades']}승 / {metrics['losing_trades']}패")
    cols[2].metric("⚡ 수익비 (Profit Factor)", f"{metrics['profit_factor']:.2f}")
    cols[3].metric("📊 평균 수익률", f"{metrics['avg_pnl_percent']:.2f}%")

    st.markdown("---")

    # --- 시각화 ---
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("💹 누적 손익 그래프")
        closed_trades = filtered_df[filtered_df['status'] == 'CLOSED'].sort_values('timestamp')
        if not closed_trades.empty:
            closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
            fig = px.line(
                closed_trades, 
                x='timestamp', 
                y='cumulative_pnl', 
                title='시간에 따른 누적 손익 변화',
                labels={'timestamp': '시간', 'cumulative_pnl': '누적 손익 (USDT)'},
                template='plotly_white'
            )
            fig.update_traces(line=dict(color='royalblue', width=2))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("표시할 완료된 거래 데이터가 없습니다.")

    with col2:
        st.subheader("📊 거래 유형 분석")
        if not filtered_df.empty:
            action_counts = filtered_df['action'].value_counts()
            fig = px.pie(
                action_counts, 
                values=action_counts.values, 
                names=action_counts.index, 
                title='Long vs Short 비율',
                color_discrete_map={'long': 'forestgreen', 'short': 'crimson'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("거래 데이터가 없습니다.")
            
    st.markdown("---")

    # --- 현재 포지션 및 AI 분석 ---
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💼 현재 오픈 포지션")
        open_positions = filtered_df[filtered_df['status'] == 'OPEN']
        if not open_positions.empty:
            st.dataframe(
                open_positions[['timestamp', 'action', 'entry_price', 'leverage', 'investment_amount', 'sl_price', 'tp_price']],
                use_container_width=True
            )
        else:
            st.success("✅ 현재 진행 중인 거래가 없습니다.")

    with col4:
        st.subheader("🤖 최신 AI 분석")
        if not ai_analysis_df.empty:
            latest_analysis = ai_analysis_df.iloc[0]
            direction_color = "green" if latest_analysis['direction'] == "LONG" else "red"
            st.markdown(f"**방향 예측**: <span style='color:{direction_color};'>{latest_analysis['direction']}</span>", unsafe_allow_html=True)
            st.text(f"분석 시간: {latest_analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            with st.expander("자세한 분석 내용 보기"):
                st.write(latest_analysis['reasoning'])
        else:
            st.info("AI 분석 데이터가 없습니다.")
            
    st.markdown("---")

    # --- 전체 거래 내역 ---
    st.subheader("📋 전체 거래 내역")
    
    display_cols = [
        'timestamp', 'action', 'status', 'entry_price', 'exit_price', 
        'leverage', 'investment_amount', 'profit_loss', 'profit_loss_percentage'
    ]
    # 데이터프레임에 실제 존재하는 컬럼만 선택
    display_cols_exist = [col for col in display_cols if col in filtered_df.columns]
    
    st.dataframe(filtered_df[display_cols_exist].style.format({
        'entry_price': '${:,.2f}',
        'exit_price': '${:,.2f}',
        'investment_amount': '${:,.2f}',
        'profit_loss': '{:,.2f}',
        'profit_loss_percentage': '{:.2f}%'
    }), use_container_width=True)

    # --- 자동 새로고침 ---
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("🔄 30초마다 자동 새로고침"):
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    run_dashboard()