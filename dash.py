"""
멀티코인 데이트레이딩 봇 대시보드 - Streamlit
실행 방법: streamlit run dash.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="AI-Powered Multi-Coin Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 설정 =====
# !!! 중요: 이 파일이 봇이 생성한 'multi_coin_daytrading.db' 파일과 동일한 폴더에 있는지 확인하세요.
# 또는, 아래에 데이터베이스 파일의 전체 경로를 지정하세요.
# 예: DB_FILE = "C:/Users/YourUser/trading_bot/multi_coin_daytrading.db"
DB_FILE = 'multi_coin_daytrading.db'


# ===== 데이터 로드 함수 (DB 스키마 호환성 수정) =====
@st.cache_data(ttl=30)
def load_data():
    """거래 및 AI 분석 데이터를 DB에서 로드하고 필요한 값을 계산"""
    if not os.path.exists(DB_FILE):
        return None, None

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
        if cursor.fetchone() is None:
            conn.close()
            return pd.DataFrame(), pd.DataFrame(columns=['timestamp'])

        trades_query = "SELECT * FROM trades ORDER BY timestamp DESC"
        trades_df = pd.read_sql_query(trades_query, conn)
        
        ai_query = "SELECT * FROM ai_analysis ORDER BY timestamp DESC"
        ai_df = pd.read_sql_query(ai_query, conn)
        
        conn.close()

        # DB 스키마 차이 대응
        if 'initial_amount' in trades_df.columns and 'amount' not in trades_df.columns:
            trades_df = trades_df.rename(columns={'initial_amount': 'amount'})
        if 'sl_price' in trades_df.columns and 'current_sl_price' not in trades_df.columns:
            trades_df = trades_df.rename(columns={'sl_price': 'current_sl_price'})
        if 'tp1_achieved' in trades_df.columns and 'partial_tp_hit' not in trades_df.columns:
            trades_df = trades_df.rename(columns={'tp1_achieved': 'partial_tp_hit'})

        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            if 'exit_timestamp' in trades_df.columns:
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'], errors='coerce')
            
            trades_df['leverage'] = trades_df['leverage'].replace(0, 1)
            
            # profit_loss가 0으로만 되어 있는 구버전 DB를 위해 재계산 로직 추가
            if 'profit_loss' in trades_df.columns and trades_df[trades_df['status'] == 'CLOSED']['profit_loss'].sum() == 0:
                 trades_df['profit_loss'] = np.where(
                    trades_df['status'] == 'CLOSED',
                    (trades_df['exit_price'] - trades_df['entry_price']) * trades_df['amount'] * np.where(trades_df['action'] == 'long', 1, -1),
                    0
                )

            if 'investment_amount' not in trades_df.columns:
                 trades_df['investment_amount'] = (trades_df['entry_price'] * trades_df['amount']) / trades_df['leverage']

            if 'profit_loss_percentage' not in trades_df.columns:
                trades_df['profit_loss_percentage'] = np.where(
                    trades_df['investment_amount'] > 0,
                    (trades_df['profit_loss'] / trades_df['investment_amount']) * 100,
                    0
                )
        
        if not ai_df.empty:
            ai_df['timestamp'] = pd.to_datetime(ai_df['timestamp'])

        return trades_df, ai_df
    
    except Exception as e:
        st.error(f"데이터베이스 로딩 중 오류 발생: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ===== 로그 파일 읽기 함수 =====
def read_log_file(log_file='output.log', lines=100):
    if not os.path.exists(log_file):
        return f"'{log_file}' 파일을 찾을 수 없습니다."
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        return "".join(log_lines[-lines:])
    except Exception as e:
        return f"로그 파일 읽기 오류: {e}"

# ===== 메트릭 계산 함수 =====
def calculate_performance_metrics(trades_df):
    if trades_df.empty: return {}
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
    
    metrics = {
        'total_trades': len(trades_df),
        'open_trades': len(trades_df[trades_df['status'] == 'OPEN']),
        'win_rate': 0,
        'total_pnl': 0,
        'profit_factor': 0
    }

    if not closed_trades.empty and 'profit_loss' in closed_trades.columns:
        winning_trades = closed_trades[closed_trades['profit_loss'] > 0]
        total_profit = winning_trades['profit_loss'].sum()
        total_loss = abs(closed_trades[closed_trades['profit_loss'] <= 0]['profit_loss'].sum())
        
        metrics['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0
        metrics['total_pnl'] = closed_trades['profit_loss'].sum()
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

    return metrics

# ===== 메인 대시보드 =====
def main():
    st.title("🤖 AI-Powered Multi-Coin Trading Dashboard")
    st.markdown("---")
    
    trades_df, ai_df = load_data()

    if trades_df is None:
        st.error(f"🚨 데이터베이스 파일을 찾을 수 없습니다: '{DB_FILE}'")
        st.info("봇이 실행되는 폴더에 있는 `multi_coin_daytrading.db` 파일을 이 대시보드(`dash.py`)가 있는 폴더로 복사해주세요.\n\n또는 `dash.py` 코드 상단의 `DB_FILE` 변수에 데이터베이스 파일의 전체 경로를 직접 지정할 수 있습니다.")
        return

    if trades_df.empty:
        st.warning("📊 거래 데이터가 없습니다. 봇이 아직 거래를 기록하지 않았을 수 있습니다.")
    
    st.sidebar.header("📊 필터 설정")
    if not trades_df.empty:
        unique_coins = sorted(trades_df['coin_symbol'].unique())
        coin_filter = st.sidebar.multiselect("코인 선택", options=unique_coins, default=unique_coins)
        min_date, max_date = trades_df['timestamp'].min().date(), trades_df['timestamp'].max().date()
        date_range = st.sidebar.date_input("날짜 범위 선택", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        status_filter = st.sidebar.multiselect("거래 상태", options=['OPEN', 'CLOSED'], default=['OPEN', 'CLOSED'])
        action_filter = st.sidebar.multiselect("거래 방향", options=['long', 'short'], default=['long', 'short'])
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_trades = trades_df[(trades_df['coin_symbol'].isin(coin_filter)) & (trades_df['timestamp'].dt.date >= start_date) & (trades_df['timestamp'].dt.date <= end_date) & (trades_df['status'].isin(status_filter)) & (trades_df['action'].isin(action_filter))]
        else:
            filtered_trades = trades_df
    else:
        filtered_trades = pd.DataFrame()

    metrics = calculate_performance_metrics(filtered_trades)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 거래 수", f"{metrics.get('total_trades', 0)} (진행중: {metrics.get('open_trades', 0)})")
    col2.metric("승률", f"{metrics.get('win_rate', 0):.1f}%")
    col3.metric("총 손익", f"${metrics.get('total_pnl', 0):,.2f}")
    pf = metrics.get('profit_factor', 0)
    col4.metric("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 누적 손익 추이")
        if not filtered_trades.empty and 'status' in filtered_trades.columns:
            closed_trades = filtered_trades[filtered_trades['status'] == 'CLOSED'].sort_values('timestamp')
            if not closed_trades.empty and 'profit_loss' in closed_trades.columns:
                closed_trades['cumulative_pnl'] = closed_trades['profit_loss'].cumsum()
                
                # exit_timestamp가 없는 구버전 DB 호환
                x_axis = 'exit_timestamp' if 'exit_timestamp' in closed_trades.columns and not closed_trades['exit_timestamp'].isnull().all() else 'timestamp'
                
                fig = px.area(closed_trades, x=x_axis, y='cumulative_pnl', labels={'cumulative_pnl': '누적 손익 (USDT)', x_axis: '시간'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("표시할 완료된 거래가 없습니다.")
    
    with col_right:
        st.subheader("📊 실시간 상태")
        open_trades = filtered_trades[filtered_trades['status'] == 'OPEN'] if not filtered_trades.empty and 'status' in filtered_trades.columns else pd.DataFrame()
        if not open_trades.empty:
            st.write("**🔴 현재 오픈 포지션**")
            for _, trade in open_trades.iterrows():
                with st.container(border=True):
                    is_trailing = trade.get('partial_tp_hit') == 1 or trade.get('tp1_achieved') == 1
                    trailing_icon = "🛡️ Trailing" if is_trailing else ""
                    
                    st.markdown(f"**{trade['coin_symbol']} | {trade['action'].upper()} | {trade.get('leverage', 'N/A')}x {trailing_icon}**")
                    st.text(f"  - 진입: ${trade['entry_price']:,.4f} | 손절: ${trade.get('current_sl_price', 0):,.4f}")
        else:
            st.info("현재 오픈된 포지션이 없습니다.")

        st.write("**🤖 최근 AI 분석**")
        if ai_df is not None and not ai_df.empty:
            for _, analysis in ai_df.head(5).iterrows():
                with st.container(border=True):
                    decision_color = "green" if analysis['ai_decision'] == '동의' else "orange"
                    st.markdown(f"**{analysis['coin_symbol']} | <span style='color:{decision_color};'>{analysis['ai_decision']}</span>** ({analysis['direction']})", unsafe_allow_html=True)
                    st.caption(f"{analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    with st.expander("분석 근거 보기"):
                        st.markdown(analysis['reasoning'])
        else:
            st.info("AI 분석 기록이 없습니다.")

    st.subheader("📋 거래 내역 상세")
    if not filtered_trades.empty:
        default_cols = ['timestamp', 'coin_symbol', 'action', 'entry_price', 'status']
        available_cols = [col for col in default_cols if col in filtered_trades.columns]
        
        optional_cols = ['exit_price', 'profit_loss', 'profit_loss_percentage', 'exit_reason']
        available_cols.extend([col for col in optional_cols if col in filtered_trades.columns])
        
        display_trades = filtered_trades[available_cols].copy()
        
        column_mapping = {
            'timestamp': '시간', 'coin_symbol': '코인', 'action': '방향', 'entry_price': '진입가',
            'exit_price': '청산가', 'profit_loss': '손익(USDT)', 
            'profit_loss_percentage': '손익률(%)', 'status': '상태', 'exit_reason': '청산 이유'
        }
        display_trades = display_trades.rename(columns=column_mapping)
        st.dataframe(display_trades, use_container_width=True, hide_index=True)

    st.subheader("📜 실시간 봇 로그")
    log_content = read_log_file()
    st.code(log_content, language='log', line_numbers=False)
    
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("자동 새로고침 (30초)"):
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

