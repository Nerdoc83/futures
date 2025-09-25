"""
멀티코인 데이트레이딩 봇 대시보드 - Streamlit
실행 방법: streamlit run streamlit_app_multi_coin.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import os # 로그 파일 확인을 위해 os 모듈 추가
import warnings
warnings.filterwarnings('ignore')

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="Multi-Coin Day Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 데이터베이스 연결 및 데이터 로드 함수 (수정됨) =====
@st.cache_data(ttl=60)  # 1분간 캐시
def load_data_from_db():
    """데이터베이스에서 멀티코인 거래 데이터를 로드하고 필요한 값을 계산"""
    try:
        # 새로운 DB 파일명으로 변경
        conn = sqlite3.connect('multi_coin_daytrading.db')
        
        # 새로운 테이블 구조에 맞는 쿼리
        query = "SELECT * FROM trades ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['exit_timestamp'].notna().any():
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
            
        # 대시보드 분석에 필요한 컬럼 계산
        # 레버리지가 0인 경우 방지
        df['leverage'] = df['leverage'].replace(0, 1) 
        # 투입 증거금 계산
        df['investment_amount'] = (df['entry_price'] * df['amount']) / df['leverage']
        
        # 손익률(%) 계산 (0으로 나누기 방지)
        df['profit_loss_percentage'] = np.where(
            df['investment_amount'] > 0,
            (df['profit_loss'] / df['investment_amount']) * 100,
            0
        )
        
        return df
    
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return pd.DataFrame()

# ===== 로그 파일 읽기 함수 (추가됨) =====
def read_log_file(log_file='output.log', lines=100):
    """지정된 로그 파일의 마지막 N줄을 읽어옴"""
    if not os.path.exists(log_file):
        return f"'{log_file}' 파일을 찾을 수 없습니다. 봇을 실행하여 로그를 생성해주세요."
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        return "".join(log_lines[-lines:])
    except Exception as e:
        return f"로그 파일 읽기 오류: {e}"

# ===== 메트릭 계산 함수 (기존과 거의 동일, profit_loss_percentage 사용) =====
def calculate_performance_metrics(trades_df):
    """성과 메트릭 계산"""
    if trades_df.empty:
        return {}
    
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty:
        return {
            'total_trades': len(trades_df), 'closed_trades': 0,
            'open_trades': len(trades_df[trades_df['status'] == 'OPEN']),
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl_percent': 0, 'max_profit': 0,
            'max_loss': 0, 'profit_factor': 0, 'avg_hold_time': '0 hours'
        }
    
    winning_trades = closed_trades[closed_trades['profit_loss'] > 0]
    losing_trades = closed_trades[closed_trades['profit_loss'] < 0]
    
    closed_trades['hold_time'] = closed_trades['exit_timestamp'] - closed_trades['timestamp']
    avg_hold_time = closed_trades['hold_time'].mean()
    avg_hold_hours = avg_hold_time.total_seconds() / 3600 if pd.notna(avg_hold_time) else 0
    
    total_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
    total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return {
        'total_trades': len(trades_df),
        'closed_trades': len(closed_trades),
        'open_trades': len(trades_df[trades_df['status'] == 'OPEN']),
        'win_rate': (len(winning_trades) / len(closed_trades)) * 100,
        'total_pnl': closed_trades['profit_loss'].sum(),
        'avg_pnl_percent': closed_trades['profit_loss_percentage'].mean(),
        'max_profit': closed_trades['profit_loss_percentage'].max(),
        'max_loss': closed_trades['profit_loss_percentage'].min(),
        'profit_factor': profit_factor,
        'avg_hold_time': f"{avg_hold_hours:.1f} hours"
    }

# ===== 메인 대시보드 (수정됨) =====
def main():
    st.title("🚀 Multi-Coin Day Trading Dashboard")
    st.markdown("---")
    
    trades_df = load_data_from_db()
    
    if trades_df.empty:
        st.warning("데이터베이스에서 데이터를 찾을 수 없습니다. 봇이 실행되고 있는지 확인해주세요.")
        # DB가 없어도 로그는 보여줄 수 있으므로 return하지 않음
    
    # ===== 사이드바 필터 (수정됨) =====
    st.sidebar.header("📊 필터 설정")
    
    if not trades_df.empty:
        # 코인 필터 (추가됨)
        unique_coins = sorted(trades_df['coin_symbol'].unique())
        coin_filter = st.sidebar.multiselect(
            "코인 선택",
            options=unique_coins,
            default=unique_coins
        )
        
        min_date = trades_df['timestamp'].min().date()
        max_date = trades_df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "날짜 범위 선택", value=(max_date - timedelta(days=7), max_date),
            min_value=min_date, max_value=max_date
        )
        
        status_filter = st.sidebar.multiselect(
            "거래 상태", options=['OPEN', 'CLOSED'], default=['OPEN', 'CLOSED']
        )
        
        action_filter = st.sidebar.multiselect(
            "거래 방향", options=['long', 'short'], default=['long', 'short']
        )
        
        # 데이터 필터링 (수정됨)
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_trades = trades_df[
                (trades_df['coin_symbol'].isin(coin_filter)) &
                (trades_df['timestamp'].dt.date >= start_date) &
                (trades_df['timestamp'].dt.date <= end_date) &
                (trades_df['status'].isin(status_filter)) &
                (trades_df['action'].isin(action_filter))
            ]
        else:
            filtered_trades = trades_df[
                (trades_df['coin_symbol'].isin(coin_filter)) &
                (trades_df['status'].isin(status_filter)) &
                (trades_df['action'].isin(action_filter))
            ]
    else:
        filtered_trades = pd.DataFrame() # 빈 데이터프레임으로 초기화

    # ===== 상단 메트릭 카드 =====
    metrics = calculate_performance_metrics(filtered_trades)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="총 거래 수", value=metrics.get('total_trades', 0), delta=f"진행중: {metrics.get('open_trades', 0)}")
    with col2:
        st.metric(label="승률", value=f"{metrics.get('win_rate', 0):.1f}%", delta="완료된 거래 기준")
    with col3:
        pnl = metrics.get('total_pnl', 0)
        st.metric(label="총 수익/손실", value=f"${pnl:.2f}", delta=f"{metrics.get('avg_pnl_percent', 0):.2f}% 평균")
    with col4:
        st.metric(label="최대 수익률", value=f"{metrics.get('max_profit', 0):.2f}%", delta=f"최대 손실: {metrics.get('max_loss', 0):.2f}%")
    with col5:
        pf = metrics.get('profit_factor', 0)
        pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric(label="Profit Factor", value=pf_display, delta=f"평균 보유: {metrics.get('avg_hold_time', '0 hours')}")
    
    st.markdown("---")
    
    # ===== 메인 차트 섹션 =====
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 거래 성과 분석")
        tab1, tab2, tab3 = st.tabs(["수익률 추이", "거래 분포", "레버리지 분석"])
        
        with tab1:
            if not filtered_trades.empty and 'profit_loss' in filtered_trades.columns:
                closed_trades = filtered_trades[filtered_trades['status'] == 'CLOSED'].copy()
                if not closed_trades.empty:
                    closed_trades = closed_trades.sort_values('timestamp')
                    closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
                    
                    fig = px.line(closed_trades, x='timestamp', y='cumulative_pnl', title='누적 수익/손실 (USDT)')
                    fig.update_traces(line=dict(color='blue', width=2))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("선택된 기간에 완료된 거래가 없습니다.")
            else:
                st.info("표시할 거래 데이터가 없습니다.")
                
        with tab2:
            if not filtered_trades.empty:
                col_a, col_b = st.columns(2)
                with col_a:
                    action_counts = filtered_trades['action'].value_counts()
                    fig_pie = px.pie(values=action_counts.values, names=action_counts.index, title="거래 방향 분포",
                                     color_discrete_map={'long': '#00CC96', 'short': '#EF553B'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_b:
                    closed_trades = filtered_trades[filtered_trades['status'] == 'CLOSED']
                    if not closed_trades.empty:
                        fig_hist = px.histogram(closed_trades, x='profit_loss_percentage', nbins=20, title="수익률 분포",
                                                color_discrete_sequence=['#636EFA'])
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_hist.update_xaxes(title="수익률 (%)")
                        st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            if not filtered_trades.empty:
                closed_trades = filtered_trades[filtered_trades['status'] == 'CLOSED']
                if not closed_trades.empty:
                    fig_scatter = px.scatter(closed_trades, x='leverage', y='profit_loss_percentage', color='action',
                                             size='investment_amount', title="레버리지별 수익률",
                                             color_discrete_map={'long': '#00CC96', 'short': '#EF553B'},
                                             hover_data=['coin_symbol', 'entry_price', 'exit_price'])
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("📊 실시간 상태")
        open_trades = filtered_trades[filtered_trades['status'] == 'OPEN'] if not filtered_trades.empty else pd.DataFrame()
        if not open_trades.empty:
            st.write("**🔴 현재 오픈 포지션**")
            for _, trade in open_trades.iterrows():
                with st.container():
                    # 손절가 계산 (증거금 10% 손실 기준)
                    sl_change = 0.10 / trade['leverage']
                    sl_price = trade['entry_price'] * (1 - sl_change) if trade['action'] == 'long' else trade['entry_price'] * (1 + sl_change)
                    
                    st.markdown(f"**{trade['coin_symbol']} | {trade['action'].upper()} | {trade['leverage']}x**")
                    st.text(f"  - 진입가: ${trade['entry_price']:,.4f}")
                    st.text(f"  - 투입마진: ${trade['investment_amount']:,.2f} USDT")
                    st.text(f"  - 손절가(계산됨): ${sl_price:,.4f}")
                    st.markdown("---")
        else:
            st.info("현재 오픈된 포지션이 없습니다.")
    
    # ===== 상세 거래 내역 테이블 (수정됨) =====
    st.subheader("📋 거래 내역 상세")
    if not filtered_trades.empty:
        display_columns = [
            'timestamp', 'coin_symbol', 'action', 'entry_price', 'exit_price', 
            'amount', 'leverage', 'investment_amount',
            'profit_loss', 'profit_loss_percentage', 'status'
        ]
        display_trades = filtered_trades[display_columns].copy()
        display_trades['timestamp'] = display_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        column_mapping = {
            'timestamp': '시간', 'coin_symbol': '코인', 'action': '방향', 'entry_price': '진입가',
            'exit_price': '청산가', 'amount': '수량', 'leverage': '레버리지',
            'investment_amount': '투입마진(USDT)', 'profit_loss': '손익(USDT)',
            'profit_loss_percentage': '손익률(%)', 'status': '상태'
        }
        display_trades = display_trades.rename(columns=column_mapping)
        
        def highlight_pnl(row):
            if row['상태'] == 'CLOSED' and pd.notna(row['손익(USDT)']):
                color = 'background-color: #d4edda' if row['손익(USDT)'] > 0 else 'background-color: #f8d7da'
                return [color] * len(row)
            return [''] * len(row)
        
        st.dataframe(display_trades.style.apply(highlight_pnl, axis=1).format({
            "진입가": "${:,.4f}", "청산가": "${:,.4f}", "투입마진(USDT)": "{:,.2f}",
            "손익(USDT)": "{:,.2f}", "손익률(%)": "{:.2f}%"
        }), use_container_width=True)
        
        csv = display_trades.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드", data=csv,
            file_name=f"multi_coin_trading_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    else:
        st.info("선택한 필터 조건에 맞는 거래 내역이 없습니다.")

    # ===== 실시간 로그 표시 기능 (추가됨) =====
    st.markdown("---")
    st.subheader("🤖 실시간 봇 로그")
    log_content = read_log_file()
    st.code(log_content, language='log', line_numbers=True)
    
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (60초)")
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()

