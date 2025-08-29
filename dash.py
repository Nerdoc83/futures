"""
멀티코인 데이트레이딩 봇 대시보드 - Streamlit (BTC/ETH/SOL)
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

# ===== 기본 설정 =====
DB_FILE = "multi_coin_daytrading.db"
LOG_FILE = "output.log"

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="Multi-Coin Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 데이터베이스 연결 및 로드 함수 =====
@st.cache_data(ttl=15)  # 캐시 시간을 15초로 조정
def load_data_from_db(db_path):
    """데이터베이스에서 데이터 로드 (안정성 강화)"""
    if not os.path.exists(db_path):
        st.error(f"⚠️ 데이터베이스 파일을 찾을 수 없습니다: '{db_path}'. `m1.py`를 먼저 실행해주세요.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        # timeout을 설정하여 DB 잠금 문제 완화
        conn = sqlite3.connect(db_path, timeout=10)
        
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        ai_analysis_df = pd.read_sql_query("SELECT * FROM ai_analysis ORDER BY timestamp DESC", conn)
        
        conn.close()
        
        # 타임스탬프 변환 (오류 처리 강화)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
            if 'exit_timestamp' in trades_df.columns:
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'], errors='coerce')
        
        if not ai_analysis_df.empty:
            ai_analysis_df['timestamp'] = pd.to_datetime(ai_analysis_df['timestamp'], errors='coerce')
            
        return trades_df, ai_analysis_df
    
    except sqlite3.OperationalError as e:
        st.warning(f"DB가 잠겨있습니다. 잠시 후 새로고침 됩니다. (오류: {e})")
        time.sleep(5)
        st.rerun()
    except Exception as e:
        st.error(f"데이터베이스 로드 중 오류 발생: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ===== 로그 파일 읽기 함수 =====
@st.cache_data(ttl=10)
def load_log_file(log_path):
    """output.log 파일을 읽어서 반환"""
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-200:] # 최근 200줄
    except Exception as e:
        return [f"로그 파일 읽기 오류: {e}"]

# ===== 메트릭 계산 함수 =====
def calculate_performance_metrics(trades_df):
    """성과 메트릭 계산 (안정성 강화)"""
    metrics = {
        'total_trades': 0, 'closed_trades': 0, 'open_trades': 0,
        'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0,
        'total_pnl': 0.0, 'avg_pnl_percent': 0.0, 'max_profit': 0.0,
        'max_loss': 0.0, 'profit_factor': 0.0, 'avg_hold_time': 'N/A',
        'coin_breakdown': {}
    }
    
    if trades_df.empty:
        return metrics
    
    metrics['total_trades'] = len(trades_df)
    metrics['open_trades'] = len(trades_df[trades_df['status'] == 'OPEN'])
    
    # [수정] 'OPEN'이 아닌 모든 상태를 종료된 거래로 간주 (수동 종료 등 포함)
    closed_df = trades_df[trades_df['status'] != 'OPEN'].copy()
    metrics['closed_trades'] = len(closed_df)
    
    # 코인별 통계
    if 'coin_symbol' in trades_df.columns:
        coin_stats = {}
        for coin in trades_df['coin_symbol'].unique():
            coin_trades = trades_df[trades_df['coin_symbol'] == coin]
            # [수정] 'OPEN'이 아닌 모든 상태를 종료된 거래로 간주
            coin_closed = coin_trades[coin_trades['status'] != 'OPEN']
            
            pnl_sum = 0
            win_rate = 0
            if not coin_closed.empty and 'profit_loss' in coin_closed.columns:
                coin_pnl = coin_closed['profit_loss'].dropna()
                if not coin_pnl.empty:
                    pnl_sum = coin_pnl.sum()
                    win_count = len(coin_pnl[coin_pnl > 0])
                    if len(coin_pnl) > 0:
                        win_rate = (win_count / len(coin_pnl)) * 100
            
            coin_stats[coin] = {'total': len(coin_trades), 'closed': len(coin_closed), 'pnl': pnl_sum, 'win_rate': win_rate}
        metrics['coin_breakdown'] = coin_stats
    
    if closed_df.empty or 'profit_loss' not in closed_df.columns:
        return metrics
    
    pnl_data = closed_df['profit_loss'].dropna()
    if not pnl_data.empty:
        winning_trades = pnl_data[pnl_data > 0]
        losing_trades = pnl_data[pnl_data < 0]
        
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        if metrics['closed_trades'] > 0:
            metrics['win_rate'] = (metrics['winning_trades'] / metrics['closed_trades']) * 100
        
        metrics['total_pnl'] = pnl_data.sum()
        
        total_profit = winning_trades.sum()
        total_loss = abs(losing_trades.sum())
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else total_profit
    
    if 'profit_loss_percentage' in closed_df.columns:
        pnl_pct_data = closed_df['profit_loss_percentage'].dropna()
        if not pnl_pct_data.empty:
            metrics['avg_pnl_percent'] = pnl_pct_data.mean()
            metrics['max_profit'] = pnl_pct_data.max()
            metrics['max_loss'] = pnl_pct_data.min()

    if 'exit_timestamp' in closed_df.columns:
        valid_times = closed_df.dropna(subset=['timestamp', 'exit_timestamp'])
        if not valid_times.empty:
            hold_times = (valid_times['exit_timestamp'] - valid_times['timestamp']).dt.total_seconds() / 3600
            avg_hours = hold_times.mean()
            if pd.notna(avg_hours):
                metrics['avg_hold_time'] = f"{avg_hours:.1f} hours"
    
    return metrics

# ===== 메인 대시보드 UI =====
def main():
    st.title("🚀 Multi-Coin Day Trading Dashboard")
    st.markdown("**BTC • ETH • SOL** 동시 거래 모니터링")
    
    col_refresh, col_time = st.columns([1, 4])
    if col_refresh.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
    col_time.write(f"**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")
    
    trades_df, ai_analysis_df = load_data_from_db(DB_FILE)
    
    if trades_df.empty and ai_analysis_df.empty:
        st.warning("📊 표시할 데이터가 없습니다. `m1.py` 봇이 거래를 시작할 때까지 기다려주세요.")
        return

    # 사이드바
    st.sidebar.header("📊 데이터 상태")
    st.sidebar.write(f"**거래 기록**: {len(trades_df)}건")
    st.sidebar.write(f"**AI 분석**: {len(ai_analysis_df)}건")
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
    
    cols = st.columns(5)
    cols[0].metric("🎯 총 거래", f"{metrics['total_trades']:,}", f"오픈: {metrics['open_trades']}")
    cols[1].metric("🎖️ 승률", f"{metrics['win_rate']:.1f}%", f"완료: {metrics['closed_trades']}건")
    cols[2].metric("💰 총 손익 (USDT)", f"${metrics['total_pnl']:,.2f}", f"{metrics.get('avg_pnl_percent', 0):.2f}% 평균")
    cols[3].metric("📊 최고/최저 수익률", f"{metrics['max_profit']:.2f}%", f"최저: {metrics['max_loss']:.2f}%")
    cols[4].metric("⚡ Profit Factor", f"{metrics['profit_factor']:.2f}", f"보유: {metrics['avg_hold_time']}")

    # 코인별 성과
    if metrics['coin_breakdown']:
        st.subheader("🪙 코인별 성과 요약")
        coin_cols = st.columns(len(metrics['coin_breakdown']))
        coin_map = {"BTC": "🟠", "ETH": "🔵", "SOL": "🟣"}
        for i, (coin, stats) in enumerate(metrics['coin_breakdown'].items()):
            with coin_cols[i]:
                st.metric(f"{coin_map.get(coin, '⚪')} {coin}", f"${stats['pnl']:,.2f}", f"{stats['win_rate']:.1f}% 승률")
                st.caption(f"총 {stats['total']}건 (완료 {stats['closed']}건)")
    st.markdown("---")

    # 차트와 실시간 상태
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("📊 거래 분석 차트")
        tab1, tab2 = st.tabs(["📈 수익률 추이", "🪙 코인/방향 분석"])
        
        # [수정] 'OPEN'이 아닌 모든 상태를 종료된 거래로 간주하여 차트 생성
        closed_trades = filtered_trades[filtered_trades['status'] != 'OPEN'].copy()
        
        with tab1:
            if not closed_trades.empty and 'profit_loss' in closed_trades.columns:
                valid_pnl = closed_trades.dropna(subset=['profit_loss']).sort_values('timestamp')
                if not valid_pnl.empty:
                    valid_pnl['cumulative_pnl'] = valid_pnl['profit_loss'].cumsum()
                    fig = px.line(valid_pnl, x='timestamp', y='cumulative_pnl', title="누적 손익(PnL) 추이", markers=True)
                    fig.update_layout(xaxis_title="시간", yaxis_title="누적 손익 (USDT)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 표시할 손익 데이터가 없습니다.")
            else:
                st.info("📊 표시할 완료된 거래가 없습니다.")

        with tab2:
            if not filtered_trades.empty:
                c1, c2 = st.columns(2)
                if 'coin_symbol' in filtered_trades.columns:
                    coin_counts = filtered_trades['coin_symbol'].value_counts()
                    fig_coin = px.pie(values=coin_counts.values, names=coin_counts.index, title="코인별 거래 수", color_discrete_map={'BTC': '#FF9500', 'ETH': '#627EEA', 'SOL': '#9945FF'})
                    c1.plotly_chart(fig_coin, use_container_width=True)
                if 'action' in filtered_trades.columns:
                    action_counts = filtered_trades['action'].value_counts()
                    fig_action = px.pie(values=action_counts.values, names=action_counts.index, title="거래 방향 (Long/Short)", color_discrete_map={'long': '#00CC96', 'short': '#EF553B'})
                    c2.plotly_chart(fig_action, use_container_width=True)

    with col_right:
        st.subheader("🔴 실시간 상태")
        # 현재 오픈 포지션
        open_positions = trades_df[trades_df['status'] == 'OPEN']
        if not open_positions.empty:
            st.write("**💼 현재 오픈 포지션**")
            for _, pos in open_positions.iterrows():
                coin, action = pos.get('coin_symbol', 'N/A'), pos.get('action', 'N/A')
                color = 'green' if action == 'long' else 'red'
                st.markdown(f"<h3 style='color:{color};'>{coin} {action.upper()}</h3>", unsafe_allow_html=True)
                st.write(f"진입가: ${pos.get('entry_price', 0):,.2f} | 레버리지: {pos.get('leverage', 0)}x")
                st.write(f"손절(SL): ${pos.get('sl_price', 0):,.2f} | 익절(TP): ${pos.get('tp_price', 0):,.2f}")
                st.caption(f"진입시간: {pos['timestamp'].strftime('%m/%d %H:%M')}")
                st.divider()
        else:
            st.info("💼 현재 오픈된 포지션이 없습니다.")
        
        # 최근 AI 분석
        if not ai_analysis_df.empty:
            st.write("**🤖 최근 AI 분석**")
            recent_ai = ai_analysis_df.head(1).iloc[0]
            direction = recent_ai.get('direction', 'N/A')
            coin = recent_ai.get('selected_coin', 'N/A')
            color = 'green' if direction == 'LONG' else 'red' if direction == 'SHORT' else 'gray'
            st.markdown(f"<h3 style='color:{color};'>{coin} {direction}</h3>", unsafe_allow_html=True)
            st.write(f"**AI 스코어:** 🟠BTC {recent_ai.get('btc_score',0)} 🔵ETH {recent_ai.get('eth_score',0)} 🟣SOL {recent_ai.get('sol_score',0)}")
            st.caption(f"분석시간: {recent_ai['timestamp'].strftime('%m/%d %H:%M')}")
            with st.expander("AI 분석 이유 보기"):
                st.write(recent_ai.get('reasoning', '분석 이유 없음'))

    # 거래 내역 테이블
    st.subheader("📋 거래 내역")
    if not filtered_trades.empty:
        display_cols = ['timestamp', 'coin_symbol', 'action', 'entry_price', 'exit_price', 
                       'leverage', 'profit_loss', 'profit_loss_percentage', 'status']
        available_cols = [col for col in display_cols if col in filtered_trades.columns]
        st.dataframe(filtered_trades[available_cols].head(50), use_container_width=True)
    else:
        st.info("📋 표시할 거래 내역이 없습니다.")

    # 실시간 로그
    st.subheader("📜 실시간 로그")
    
    # 로그 자동 새로고침 토글 스위치
    auto_refresh_log = st.toggle("📡 로그 실시간 새로고침 (10초)", value=True, help="10초마다 봇의 최신 로그를 자동으로 불러옵니다.")

    log_lines = load_log_file(LOG_FILE)
    if log_lines:
        log_text = "".join(reversed(log_lines))
        st.text_area("최근 200줄 로그", value=log_text, height=300, key="log_display")
    else:
        st.info(f"'{LOG_FILE}' 파일을 찾을 수 없습니다.")
        st.code(f"# Linux/macOS\npython m1.py > {LOG_FILE} 2>&1\n\n# Windows (PowerShell)\npython m1.py | Tee-Object -FilePath {LOG_FILE}", language='bash')
        st.caption(f"위와 같이 봇을 실행하면 로그가 여기에 표시됩니다.")

    # 자동 새로고침 로직
    if auto_refresh_log:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
