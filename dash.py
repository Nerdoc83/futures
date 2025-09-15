"""
AI-Verified Swing Trading Bot Dashboard - Streamlit
실행 방법: streamlit run dash.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ===== 기본 설정 (v2.0 - 스윙 트레이딩 봇에 맞게 수정) =====
DB_FILE = "multi_coin_daytrading.db" # << 수정: 새로운 DB 파일명으로 변경
LOG_FILE = "output.log" # 스윙 트레이딩 봇 실행 시 생성되는 로그 파일명

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="AI Swing Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 데이터베이스 연결 및 로드 함수 =====
@st.cache_data(ttl=30)  # 캐시 시간을 30초로 조정
def load_data_from_db(db_path):
    """데이터베이스에서 데이터 로드 (안정성 강화)"""
    if not os.path.exists(db_path):
        st.error(f"⚠️ 데이터베이스 파일을 찾을 수 없습니다: '{db_path}'. `swing.py`를 먼저 실행해주세요.")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        # << 수정: ai_analysis 테이블은 더 이상 사용하지 않음
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
        
        return trades_df
    
    except sqlite3.OperationalError as e:
        st.warning(f"DB가 잠겨있습니다. 잠시 후 새로고침 됩니다. (오류: {e})")
        time.sleep(5)
        st.rerun()
    except Exception as e:
        st.error(f"데이터베이스 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# ===== 로그 파일 읽기 함수 =====
@st.cache_data(ttl=10)
def load_log_file(log_path):
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-200:]
    except Exception as e:
        return [f"로그 파일 읽기 오류: {e}"]

# ===== 메트릭 계산 함수 (v2.0 - 스윙 트레이딩 봇에 맞게 수정) =====
def calculate_performance_metrics(trades_df):
    metrics = {
        'total_trades': 0, 'closed_trades': 0, 'open_trades': 0, 'partially_closed_trades': 0,
        'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0,
        'total_pnl': 0.0, 'profit_factor': 0.0,
        'coin_breakdown': {}
    }
    
    if trades_df.empty:
        return metrics
    
    metrics['total_trades'] = len(trades_df)
    metrics['open_trades'] = len(trades_df[trades_df['status'] == 'OPEN'])
    metrics['partially_closed_trades'] = len(trades_df[trades_df['status'] == 'PARTIALLY_CLOSED'])
    
    closed_df = trades_df[trades_df['status'].isin(['CLOSED', 'PARTIALLY_CLOSED'])].copy()
    metrics['closed_trades'] = len(trades_df[trades_df['status'] == 'CLOSED'])
    
    # 코인별 통계
    coin_stats = {}
    for coin in trades_df['coin_symbol'].unique():
        coin_trades = trades_df[trades_df['coin_symbol'] == coin]
        coin_closed = coin_trades[coin_trades['status'].isin(['CLOSED', 'PARTIALLY_CLOSED'])]
        
        pnl_sum = 0
        win_rate = 0
        if not coin_closed.empty and 'profit_loss' in coin_closed.columns:
            coin_pnl = coin_closed[coin_closed['status'] == 'CLOSED']['profit_loss'].dropna()
            if not coin_pnl.empty:
                pnl_sum = coin_closed['profit_loss'].sum()
                win_count = len(coin_pnl[coin_pnl > 0])
                if len(coin_pnl) > 0:
                    win_rate = (win_count / len(coin_pnl)) * 100
        
        coin_stats[coin] = {'total': len(coin_trades), 'pnl': pnl_sum, 'win_rate': win_rate}
    metrics['coin_breakdown'] = coin_stats
    
    if not closed_df.empty and 'profit_loss' in closed_df.columns:
        final_pnl_data = closed_df[closed_df['status'] == 'CLOSED']['profit_loss'].dropna()
        if not final_pnl_data.empty:
            winning_trades = final_pnl_data[final_pnl_data > 0]
            losing_trades = final_pnl_data[final_pnl_data < 0]
            
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            if (metrics['winning_trades'] + metrics['losing_trades']) > 0:
                metrics['win_rate'] = (metrics['winning_trades'] / (metrics['winning_trades'] + metrics['losing_trades'])) * 100
            
            metrics['total_pnl'] = closed_df['profit_loss'].sum()
            
            total_profit = winning_trades.sum()
            total_loss = abs(losing_trades.sum())
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else total_profit
    
    return metrics

# ===== 메인 대시보드 UI =====
def main():
    st.title("🚀 AI-Verified Swing Trading Dashboard")
    st.markdown("현재 **스윙 트레이딩 봇 (`m3_swing_trader_complete.py`)**의 실시간 상태를 모니터링합니다.")
    
    col_refresh, col_time = st.columns([1, 4])
    if col_refresh.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
    col_time.write(f"**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")
    
    trades_df = load_data_from_db(DB_FILE)
    
    if trades_df.empty:
        st.warning("📊 표시할 데이터가 없습니다. `m3_swing_trader_complete.py` 봇이 거래를 시작할 때까지 기다려주세요.")
        return

    # 사이드바
    st.sidebar.header("📊 데이터 상태")
    st.sidebar.write(f"**총 거래 기록**: {len(trades_df)}건")
    if not trades_df.empty:
        st.sidebar.write("**거래 상태 분포:**")
        st.sidebar.write(trades_df['status'].value_counts())

    # 필터
    filtered_trades = trades_df.copy()
    if not trades_df.empty:
        st.sidebar.header("🔍 필터")
        coin_options = sorted(trades_df['coin_symbol'].unique().tolist())
        if coin_options:
            coin_filter = st.sidebar.multiselect("💰 코인", options=coin_options, default=coin_options)
            filtered_trades = filtered_trades[filtered_trades['coin_symbol'].isin(coin_filter)]

    # 메트릭 계산 및 표시
    metrics = calculate_performance_metrics(filtered_trades)
    st.subheader("📈 실시간 성과 모니터링")
    
    cols = st.columns(4)
    cols[0].metric("🎯 총 거래", f"{metrics['total_trades']:,}", f"완전 종료: {metrics['closed_trades']}")
    cols[1].metric("💼 오픈 포지션", f"{metrics['open_trades']}", f"부분 익절: {metrics['partially_closed_trades']}")
    cols[2].metric("🎖️ 승률 (완전종료 기준)", f"{metrics['win_rate']:.1f}%", f"{metrics['winning_trades']}승 {metrics['losing_trades']}패")
    cols[3].metric("💰 누적 손익 (USDT)", f"${metrics['total_pnl']:,.2f}", f"Profit Factor: {metrics['profit_factor']:.2f}")

    # 코인별 성과
    if metrics['coin_breakdown']:
        st.subheader("🪙 코인별 성과 요약")
        coin_cols = st.columns(len(metrics['coin_breakdown']))
        coin_map = {"BTC": "🟠", "ETH": "🔵", "SOL": "🟣"}
        for i, (coin, stats) in enumerate(metrics['coin_breakdown'].items()):
            with coin_cols[i]:
                st.metric(f"{coin_map.get(coin, '⚪')} {coin}", f"${stats['pnl']:,.2f}", f"{stats['win_rate']:.1f}% 승률")
                st.caption(f"총 {stats['total']}건")
    st.markdown("---")
    
    # 현재 오픈 포지션 상세 정보
    st.subheader("💼 현재 오픈 포지션 상세")
    open_positions = trades_df[trades_df['status'].isin(['OPEN', 'PARTIALLY_CLOSED'])]
    if not open_positions.empty:
        for _, pos in open_positions.iterrows():
            col1, col2, col3 = st.columns(3)
            coin, action, tier = pos.get('coin_symbol', 'N/A'), pos.get('action', 'N/A'), pos.get('tier', 'N/A')
            color = 'green' if action == 'long' else 'red'
            
            with col1:
                st.markdown(f"<h5>{coin} <span style='color:{color};'>{action.upper()}</span> ({tier})</h5>", unsafe_allow_html=True)
                if pos['status'] == 'PARTIALLY_CLOSED':
                    st.success("🔥 1차 익절 완료 (TP1)")

            with col2:
                st.write(f"**진입가:** ${pos.get('entry_price', 0):,.4f}")
                st.write(f"**레버리지:** {pos.get('leverage', 0)}x")
            
            with col3:
                sl_price = pos.get('sl_price', 0) if pos['status'] == 'OPEN' else pos.get('entry_price', 0)
                sl_text = "손절(SL)" if pos['status'] == 'OPEN' else "본절(BE)"
                st.write(f"**{sl_text}:** ${sl_price:,.4f}")
                st.write(f"**수량:** {pos.get('current_amount', 0):.4f} ({((pos.get('current_amount', 0)/pos.get('initial_amount', 1))*100):.0f}%)")

            st.caption(f"진입시간: {pos['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.divider()
    else:
        st.info("💼 현재 오픈된 포지션이 없습니다.")

    # 차트와 거래 내역
    tab1, tab2 = st.tabs(["📈 누적 손익 차트", "📋 전체 거래 내역"])
    with tab1:
        if not filtered_trades.empty and 'profit_loss' in filtered_trades.columns:
            # 부분 익절 포함한 누적 손익 계산
            pnl_events = []
            for _, row in filtered_trades.iterrows():
                if row['status'] == 'PARTIALLY_CLOSED' and row['tp1_achieved'] == 1:
                     pnl_events.append({'timestamp': row['timestamp'], 'pnl': row['profit_loss']}) # 이 시점의 PNL
                elif row['status'] == 'CLOSED':
                     pnl_events.append({'timestamp': row['timestamp'], 'pnl': row['profit_loss']})
            
            if pnl_events:
                pnl_df = pd.DataFrame(pnl_events).sort_values('timestamp')
                pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()
                fig = px.line(pnl_df, x='timestamp', y='cumulative_pnl', title="누적 손익(PnL) 추이", markers=True)
                fig.update_layout(xaxis_title="시간", yaxis_title="누적 손익 (USDT)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 표시할 손익 데이터가 없습니다.")
        else:
            st.info("📊 표시할 거래가 없습니다.")
            
    with tab2:
        if not filtered_trades.empty:
            display_cols = ['timestamp', 'coin_symbol', 'tier', 'action', 'entry_price', 'sl_price', 
                           'leverage', 'initial_amount', 'current_amount', 'profit_loss', 'status']
            available_cols = [col for col in display_cols if col in filtered_trades.columns]
            st.dataframe(filtered_trades[available_cols].head(100), use_container_width=True)
        else:
            st.info("📋 표시할 거래 내역이 없습니다.")

    # 실시간 로그
    st.subheader("📜 실시간 로그")
    auto_refresh_log = st.toggle("📡 로그 실시간 새로고침 (10초)", value=True, help="10초마다 봇의 최신 로그를 자동으로 불러옵니다.")
    log_lines = load_log_file(LOG_FILE)
    if log_lines:
        log_text = "".join(reversed(log_lines))
        st.text_area("최근 200줄 로그", value=log_text, height=300, key="log_display")
    else:
        st.warning(f"'{LOG_FILE}' 파일을 찾을 수 없습니다. 아래와 같이 봇을 실행하면 로그가 여기에 표시됩니다.")
        st.code(f"# Linux/macOS\npython m3_swing_trader_complete.py > {LOG_FILE} 2>&1 &\n\n# Windows (PowerShell)\nStart-Process python -ArgumentList 'm3_swing_trader_complete.py' -RedirectStandardOutput {LOG_FILE} -RedirectStandardError {LOG_FILE} -NoNewWindow", language='bash')

    # 자동 새로고침 로직
    if auto_refresh_log:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
