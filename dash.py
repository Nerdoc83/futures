"""
멀티코인 데이트레이딩 봇 대시보드 - Streamlit (BTC/ETH/SOL)
실행 방법: streamlit run dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
warnings.filterwarnings('ignore')

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="Multi-Coin Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 데이터베이스 연결 함수 =====
@st.cache_data(ttl=30)  # 30초간 캐시
def load_data_from_db():
    """데이터베이스에서 데이터 로드"""
    try:
        conn = sqlite3.connect('multi_coin_daytrading.db')
        
        # 거래 데이터
        trades_query = """
        SELECT * FROM trades ORDER BY timestamp DESC
        """
        trades_df = pd.read_sql_query(trades_query, conn)
        
        # AI 분석 데이터
        ai_analysis_query = """
        SELECT * FROM ai_analysis ORDER BY timestamp DESC
        """
        ai_analysis_df = pd.read_sql_query(ai_analysis_query, conn)
        
        conn.close()
        
        # 타임스탬프 변환
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            if 'exit_timestamp' in trades_df.columns:
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'], errors='coerce')
        
        if not ai_analysis_df.empty:
            ai_analysis_df['timestamp'] = pd.to_datetime(ai_analysis_df['timestamp'])
            
        return trades_df, ai_analysis_df
    
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ===== 로그 파일 읽기 함수 =====
@st.cache_data(ttl=10)  # 10초간 캐시
def load_log_file(log_path="output.log"):
    """output.log 파일을 읽어서 반환"""
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 최근 100줄만 반환
            return lines[-100:] if len(lines) > 100 else lines
        else:
            return ["로그 파일을 찾을 수 없습니다."]
    except Exception as e:
        return [f"로그 파일 읽기 오류: {e}"]

# ===== 메트릭 계산 함수 =====
def calculate_performance_metrics(trades_df):
    """성과 메트릭 계산 - 실제 DB 스키마 기반"""
    # 기본 메트릭 초기화
    metrics = {
        'total_trades': 0,
        'closed_trades': 0,
        'open_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'avg_pnl_percent': 0.0,
        'max_profit': 0.0,
        'max_loss': 0.0,
        'profit_factor': 0.0,
        'avg_hold_time': '0 hours',
        'coin_breakdown': {}
    }
    
    if trades_df.empty:
        return metrics
    
    # 기본 통계
    total_trades = len(trades_df)
    open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
    
    # 종료된 거래 (실제 DB에서 사용하는 상태값들)
    closed_trades_df = trades_df[trades_df['status'].isin(['CLOSED', 'CLOSED_TIMEOUT'])].copy()
    closed_count = len(closed_trades_df)
    
    metrics.update({
        'total_trades': total_trades,
        'open_trades': open_trades,
        'closed_trades': closed_count
    })
    
    # 코인별 통계
    if 'coin_symbol' in trades_df.columns:
        coin_stats = {}
        for coin in trades_df['coin_symbol'].unique():
            coin_trades = trades_df[trades_df['coin_symbol'] == coin]
            coin_closed = coin_trades[coin_trades['status'].isin(['CLOSED', 'CLOSED_TIMEOUT'])]
            
            # profit_loss 컬럼 사용 (실제 DB 스키마)
            if not coin_closed.empty and 'profit_loss' in coin_closed.columns:
                coin_pnl = coin_closed['profit_loss'].dropna()
                if not coin_pnl.empty:
                    coin_stats[coin] = {
                        'total': len(coin_trades),
                        'closed': len(coin_closed),
                        'pnl': coin_pnl.sum(),
                        'win_rate': (len(coin_pnl[coin_pnl > 0]) / len(coin_pnl)) * 100
                    }
                else:
                    coin_stats[coin] = {'total': len(coin_trades), 'closed': 0, 'pnl': 0, 'win_rate': 0}
            else:
                coin_stats[coin] = {'total': len(coin_trades), 'closed': 0, 'pnl': 0, 'win_rate': 0}
        
        metrics['coin_breakdown'] = coin_stats
    
    # 종료된 거래가 없으면 여기서 반환
    if closed_trades_df.empty:
        return metrics
    
    # 손익 계산 (실제 컬럼명 사용)
    if 'profit_loss' in closed_trades_df.columns:
        profit_data = closed_trades_df['profit_loss'].dropna()
        
        if not profit_data.empty:
            winning_trades = profit_data[profit_data > 0]
            losing_trades = profit_data[profit_data < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / len(profit_data)) * 100 if len(profit_data) > 0 else 0
            
            total_pnl = profit_data.sum()
            
            # Profit Factor 계산
            total_profit = winning_trades.sum() if win_count > 0 else 0
            total_loss = abs(losing_trades.sum()) if loss_count > 0 else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit
            
            metrics.update({
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor
            })
    
    # 퍼센트 데이터 계산 (실제 컬럼명 사용)
    if 'profit_loss_percentage' in closed_trades_df.columns:
        profit_pct_data = closed_trades_df['profit_loss_percentage'].dropna()
        
        if not profit_pct_data.empty:
            metrics.update({
                'avg_pnl_percent': profit_pct_data.mean(),
                'max_profit': profit_pct_data.max(),
                'max_loss': profit_pct_data.min()
            })
    
    # 보유 시간 계산
    if 'exit_timestamp' in closed_trades_df.columns:
        valid_times = closed_trades_df.dropna(subset=['exit_timestamp'])
        if not valid_times.empty:
            try:
                hold_times = (valid_times['exit_timestamp'] - valid_times['timestamp']).dt.total_seconds() / 3600
                avg_hours = hold_times.mean()
                if pd.notna(avg_hours):
                    metrics['avg_hold_time'] = f"{avg_hours:.1f} hours"
            except Exception:
                metrics['avg_hold_time'] = "계산불가"
    
    return metrics

# ===== 메인 대시보드 =====
def main():
    # 헤더
    st.title("🚀 Multi-Coin Day Trading Dashboard")
    st.markdown("**BTC • ETH • SOL** 동시 거래 모니터링")
    st.markdown("---")
    
    # 새로고침 버튼과 시간
    col_refresh, col_time = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 새로고침"):
            st.cache_data.clear()
            st.rerun()
    
    with col_time:
        st.write(f"**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    trades_df, ai_analysis_df = load_data_from_db()
    
    # 데이터 상태 확인
    st.sidebar.header("📊 데이터 상태")
    st.sidebar.write(f"**거래 기록**: {len(trades_df)}건")
    st.sidebar.write(f"**AI 분석**: {len(ai_analysis_df)}건")
    
    # 디버깅 정보 (사이드바)
    if not trades_df.empty:
        st.sidebar.write("**거래 데이터 컬럼:**")
        st.sidebar.write(list(trades_df.columns))
        
        st.sidebar.write("**거래 상태 분포:**")
        status_dist = trades_df['status'].value_counts()
        st.sidebar.write(status_dist)
        
        st.sidebar.write("**코인 분포:**")
        if 'coin_symbol' in trades_df.columns:
            coin_dist = trades_df['coin_symbol'].value_counts()
            st.sidebar.write(coin_dist)
        
        # 손익 데이터 확인
        if 'profit_loss' in trades_df.columns:
            non_null_pnl = trades_df['profit_loss'].notna().sum()
            st.sidebar.write(f"**손익 데이터**: {non_null_pnl}개 기록")
    
    if trades_df.empty and ai_analysis_df.empty:
        st.warning("⚠️ 데이터베이스에서 데이터를 찾을 수 없습니다.")
        st.info("💡 봇이 실행되고 있는지, 데이터베이스 파일 경로가 올바른지 확인해주세요.")
        return
    
    # ===== 사이드바 필터 =====
    st.sidebar.header("🔍 필터 설정")
    
    filtered_trades = trades_df.copy()
    
    if not trades_df.empty:
        # 코인 필터
        if 'coin_symbol' in trades_df.columns:
            coin_options = sorted(trades_df['coin_symbol'].unique().tolist())
            coin_filter = st.sidebar.multiselect(
                "💰 코인 선택",
                options=coin_options,
                default=coin_options,
                help="BTC, ETH, SOL 중 분석할 코인을 선택하세요"
            )
            
            filtered_trades = filtered_trades[filtered_trades['coin_symbol'].isin(coin_filter)]
        
        # 날짜 필터
        if not filtered_trades.empty:
            min_date = filtered_trades['timestamp'].min().date()
            max_date = filtered_trades['timestamp'].max().date()
            
            date_range = st.sidebar.date_input(
                "📅 날짜 범위",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # 상태 필터
            status_options = filtered_trades['status'].unique().tolist()
            status_filter = st.sidebar.multiselect(
                "📊 거래 상태",
                options=status_options,
                default=status_options
            )
            
            # 방향 필터
            if 'action' in filtered_trades.columns:
                action_options = filtered_trades['action'].unique().tolist()
                action_filter = st.sidebar.multiselect(
                    "🎯 거래 방향",
                    options=action_options,
                    default=action_options
                )
            else:
                action_filter = []
            
            # 필터 적용
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_trades = filtered_trades[
                    (filtered_trades['timestamp'].dt.date >= start_date) &
                    (filtered_trades['timestamp'].dt.date <= end_date) &
                    (filtered_trades['status'].isin(status_filter))
                ]
                
                if action_filter and 'action' in filtered_trades.columns:
                    filtered_trades = filtered_trades[filtered_trades['action'].isin(action_filter)]
    
    # ===== 성과 메트릭 =====
    metrics = calculate_performance_metrics(filtered_trades)
    
    st.subheader("📈 실시간 성과 모니터링")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 총 거래 수",
            value=f"{metrics['total_trades']:,}",
            delta=f"오픈: {metrics['open_trades']}"
        )
    
    with col2:
        win_rate = metrics['win_rate']
        closed = metrics['closed_trades']
        st.metric(
            label="🎖️ 승률",
            value=f"{win_rate:.1f}%",
            delta=f"완료: {closed}건"
        )
    
    with col3:
        pnl = metrics['total_pnl']
        st.metric(
            label="💰 총 손익",
            value=f"${pnl:,.2f}",
            delta=f"{metrics['avg_pnl_percent']:.2f}% 평균"
        )
    
    with col4:
        max_profit = metrics['max_profit']
        max_loss = metrics['max_loss']
        st.metric(
            label="📊 최고/최저 수익률",
            value=f"{max_profit:.2f}%",
            delta=f"최저: {max_loss:.2f}%"
        )
    
    with col5:
        pf = metrics['profit_factor']
        st.metric(
            label="⚡ Profit Factor",
            value=f"{pf:.2f}",
            delta=f"보유시간: {metrics['avg_hold_time']}"
        )
    
    # 코인별 성과 요약
    if metrics['coin_breakdown']:
        st.subheader("🪙 코인별 성과 요약")
        coin_cols = st.columns(len(metrics['coin_breakdown']))
        
        for i, (coin, stats) in enumerate(metrics['coin_breakdown'].items()):
            with coin_cols[i]:
                coin_emoji = {"BTC": "🟠", "ETH": "🔵", "SOL": "🟣"}.get(coin, "⚪")
                
                st.metric(
                    label=f"{coin_emoji} {coin}",
                    value=f"${stats['pnl']:,.2f}",
                    delta=f"{stats['win_rate']:.1f}% 승률"
                )
                st.caption(f"총 {stats['total']}건 (완료 {stats['closed']}건)")
    
    st.markdown("---")
    
    # ===== 차트 섹션과 실시간 상태 =====
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 거래 분석")
        
        tab1, tab2, tab3 = st.tabs(["📈 수익률 추이", "🪙 코인별 분석", "🎯 거래 분포"])
        
        with tab1:
            # 종료된 거래만 표시
            closed_trades = filtered_trades[filtered_trades['status'].isin(['CLOSED', 'CLOSED_TIMEOUT'])].copy()
            
            if not closed_trades.empty and 'profit_loss' in closed_trades.columns:
                valid_pnl = closed_trades.dropna(subset=['profit_loss']).sort_values('timestamp')
                
                if not valid_pnl.empty:
                    valid_pnl['cumulative_pnl'] = valid_pnl['profit_loss'].cumsum()
                    
                    fig = go.Figure()
                    
                    # 누적 PnL 라인
                    fig.add_trace(go.Scatter(
                        x=valid_pnl['timestamp'],
                        y=valid_pnl['cumulative_pnl'],
                        mode='lines+markers',
                        name='누적 손익',
                        line=dict(color='#00CC96', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # 코인별 색상으로 개별 거래 포인트
                    if 'coin_symbol' in valid_pnl.columns:
                        coin_colors = {'BTC': '#FF9500', 'ETH': '#627EEA', 'SOL': '#9945FF'}
                        for coin in valid_pnl['coin_symbol'].unique():
                            coin_data = valid_pnl[valid_pnl['coin_symbol'] == coin]
                            fig.add_trace(go.Scatter(
                                x=coin_data['timestamp'],
                                y=coin_data['profit_loss'],
                                mode='markers',
                                name=f'{coin} 거래',
                                marker=dict(
                                    color=coin_colors.get(coin, '#636EFA'),
                                    size=8,
                                    opacity=0.7
                                ),
                                yaxis='y2'
                            ))
                    
                    fig.update_layout(
                        title="거래 성과 추이",
                        xaxis_title="시간",
                        yaxis=dict(title="누적 손익 (USDT)", side='left'),
                        yaxis2=dict(title="거래별 손익 (USDT)", side='right', overlaying='y'),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 표시할 손익 데이터가 없습니다.")
            else:
                st.info("📊 표시할 완료된 거래가 없습니다.")
        
        with tab2:
            if not filtered_trades.empty and 'coin_symbol' in filtered_trades.columns:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # 코인별 거래 수 분포
                    coin_counts = filtered_trades['coin_symbol'].value_counts()
                    fig_coin = px.pie(
                        values=coin_counts.values,
                        names=coin_counts.index,
                        title="코인별 거래 수 분포",
                        color_discrete_map={'BTC': '#FF9500', 'ETH': '#627EEA', 'SOL': '#9945FF'}
                    )
                    fig_coin.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_coin, use_container_width=True)
                
                with col_b:
                    # 코인별 손익 비교
                    closed_trades = filtered_trades[filtered_trades['status'].isin(['CLOSED', 'CLOSED_TIMEOUT'])]
                    if not closed_trades.empty and 'profit_loss' in closed_trades.columns:
                        coin_pnl = closed_trades.groupby('coin_symbol')['profit_loss'].sum().dropna()
                        
                        if not coin_pnl.empty:
                            fig_pnl = px.bar(
                                x=coin_pnl.index,
                                y=coin_pnl.values,
                                title="코인별 총 손익",
                                labels={'x': '코인', 'y': '손익 (USDT)'},
                                color=coin_pnl.values,
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_pnl, use_container_width=True)
                        else:
                            st.info("코인별 손익 데이터가 없습니다.")
                    else:
                        st.info("완료된 거래가 없습니다.")
        
        with tab3:
            if not filtered_trades.empty:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # 거래 방향 분포
                    if 'action' in filtered_trades.columns:
                        action_counts = filtered_trades['action'].value_counts()
                        fig_pie = px.pie(
                            values=action_counts.values,
                            names=action_counts.index,
                            title="거래 방향 분포",
                            color_discrete_map={'long': '#00CC96', 'short': '#EF553B'}
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("거래 방향 데이터가 없습니다.")
                
                with col_b:
                    # 거래 상태 분포
                    status_counts = filtered_trades['status'].value_counts()
                    fig_status = px.bar(
                        x=status_counts.index,
                        y=status_counts.values,
                        title="거래 상태 분포",
                        color=status_counts.values,
                        color_continuous_scale='viridis'
                    )
                    fig_status.update_xaxes(title="상태")
                    fig_status.update_yaxes(title="거래 수")
                    st.plotly_chart(fig_status, use_container_width=True)
    
    with col_right:
        st.subheader("🔴 실시간 상태")
        
        # 현재 오픈 포지션
        open_positions = filtered_trades[filtered_trades['status'] == 'OPEN']
        
        if not open_positions.empty:
            st.write("**💼 현재 오픈 포지션**")
            for _, pos in open_positions.iterrows():
                with st.container():
                    coin = pos.get('coin_symbol', 'UNKNOWN')
                    coin_info = {
                        'BTC': {'emoji': '🟠', 'color': '#FF9500'},
                        'ETH': {'emoji': '🔵', 'color': '#627EEA'},
                        'SOL': {'emoji': '🟣', 'color': '#9945FF'}
                    }
                    
                    coin_emoji = coin_info.get(coin, {}).get('emoji', '⚪')
                    action_emoji = "🟢" if pos.get('action') == 'long' else "🔴"
                    
                    st.markdown(f"### {coin_emoji} {coin} {action_emoji} {pos.get('action', '').upper()}")
                    
                    st.write(f"레버리지: {pos.get('leverage', 'N/A')}x")
                    st.write(f"진입가: ${pos.get('entry_price', 0):,.2f}")
                    
                    if pd.notna(pos.get('sl_price')):
                        st.write(f"손절가: ${pos['sl_price']:,.2f}")
                    if pd.notna(pos.get('tp_price')):
                        st.write(f"익절가: ${pos['tp_price']:,.2f}")
                    if pd.notna(pos.get('investment_amount')):
                        st.write(f"투입금: ${pos['investment_amount']:,.0f}")
                    
                    st.write(f"⏰ {pos['timestamp'].strftime('%m/%d %H:%M')}")
                    st.divider()
        else:
            st.info("💼 현재 오픈된 포지션이 없습니다.")
        
        # 최근 AI 분석
        if not ai_analysis_df.empty:
            st.write("**🤖 최근 AI 분석**")
            recent_ai = ai_analysis_df.head(3)
            
            for _, analysis in recent_ai.iterrows():
                direction = analysis.get('direction', 'N/A')
                selected_coin = analysis.get('selected_coin', 'N/A')
                direction_emoji = "🟢" if direction == 'LONG' else "🔴" if direction == 'SHORT' else "⚪"
                
                with st.container():
                    st.markdown(f"### {direction_emoji} {selected_coin} {direction}")
                    
                    # 멀티코인 스코어 표시
                    btc_score = analysis.get('btc_score', 0)
                    eth_score = analysis.get('eth_score', 0)
                    sol_score = analysis.get('sol_score', 0)
                    
                    if btc_score or eth_score or sol_score:
                        st.write("**AI 스코어:**")
                        score_cols = st.columns(3)
                        with score_cols[0]:
                            st.write(f"🟠 BTC: {btc_score}")
                        with score_cols[1]:
                            st.write(f"🔵 ETH: {eth_score}")
                        with score_cols[2]:
                            st.write(f"🟣 SOL: {sol_score}")
                    
                    if pd.notna(analysis.get('recommended_leverage')):
                        st.write(f"레버리지: {analysis['recommended_leverage']}x")
                    if pd.notna(analysis.get('recommended_position_size')):
                        st.write(f"포지션: {analysis['recommended_position_size']*100:.0f}%")
                    
                    st.write(f"⏰ {analysis['timestamp'].strftime('%m/%d %H:%M')}")
                    st.divider()
    
    # ===== 실시간 로그 모니터링 =====
    st.subheader("📋 실시간 로그 모니터링")
    
    log_tab1, log_tab2 = st.tabs(["📄 최근 로그", "⚙️ 로그 설정"])
    
    with log_tab1:
        col_log_refresh, col_log_auto = st.columns([1, 3])
        
        with col_log_refresh:
            if st.button("🔄 로그 새로고침"):
                st.cache_data.clear()
        
        with col_log_auto:
            auto_refresh_log = st.checkbox("📡 로그 자동 새로고침 (10초)")
        
        # 로그 파일 읽기
        log_lines = load_log_file()
        
        # 로그 표시
        st.subheader("📜 Output.log")
        log_container = st.container()
        
        with log_container:
            if log_lines:
                # 로그를 역순으로 표시 (최신 로그가 위로)
                log_text = "".join(reversed(log_lines))
                st.text_area(
                    "로그 내용",
                    value=log_text,
                    height=400,
                    key="log_display"
                )
            else:
                st.info("📄 로그 파일이 비어있거나 읽을 수 없습니다.")
        
        # 로그 통계
        if log_lines:
            st.write(f"**로그 통계**: {len(log_lines)}줄 표시 중")
    
    with log_tab2:
        st.write("**로그 파일 경로 설정**")
        log_path = st.text_input("로그 파일 경로", value="output.log")
        
        if st.button("경로 테스트"):
            if os.path.exists(log_path):
                st.success(f"✅ 파일 찾음: {log_path}")
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    st.info(f"📄 총 {line_count}줄")
                except Exception as e:
                    st.error(f"❌ 파일 읽기 오류: {e}")
            else:
                st.error(f"❌ 파일을 찾을 수 없습니다: {log_path}")
    
    # ===== 거래 내역 테이블 =====
    st.subheader("📋 거래 내역")
    
    if not filtered_trades.empty:
        # 표시할 컬럼 선택
        display_cols = ['timestamp', 'coin_symbol', 'action', 'entry_price', 'exit_price', 'amount', 
                       'leverage', 'investment_amount', 'profit_loss', 'profit_loss_percentage', 'status']
        
        # 존재하는 컬럼만 선택
        available_cols = [col for col in display_cols if col in filtered_trades.columns]
        display_df = filtered_trades[available_cols].copy()
        
        # 타임스탬프 포맷
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 컬럼명 한글화
        column_names = {
            'timestamp': '시간',
            'coin_symbol': '코인',
            'action': '방향', 
            'entry_price': '진입가',
            'exit_price': '청산가',
            'amount': '수량',
            'leverage': '레버리지',
            'investment_amount': '투입마진',
            'profit_loss': '손익(USDT)',
            'profit_loss_percentage': '수익률(%)',
            'status': '상태'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # 스타일 적용
        def style_dataframe(df):
            def highlight_rows(row):
                styles = [''] * len(row)
                
                # 코인별 배경색
                if '코인' in row.index:
                    coin = row['코인']
                    if coin == 'BTC':
                        styles = ['background-color: rgba(255, 149, 0, 0.1)'] * len(row)
                    elif coin == 'ETH':
                        styles = ['background-color: rgba(98, 126, 234, 0.1)'] * len(row)
                    elif coin == 'SOL':
                        styles = ['background-color: rgba(153, 69, 255, 0.1)'] * len(row)
                
                # 손익 색상
                if '손익(USDT)' in row.index and pd.notna(row['손익(USDT)']):
                    try:
                        pnl_value = float(row['손익(USDT)'])
                        if pnl_value > 0:
                            styles = ['background-color: rgba(212, 237, 218, 0.7)'] * len(row)
                        elif pnl_value < 0:
                            styles = ['background-color: rgba(248, 215, 218, 0.7)'] * len(row)
                    except:
                        pass
                        
                return styles
            
            return df.style.apply(highlight_rows, axis=1)
        
        # 손익 컬럼이나 코인 컬럼이 있으면 스타일 적용
        if '손익(USDT)' in display_df.columns or '코인' in display_df.columns:
            styled_df = style_dataframe(display_df)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        # 다운로드 버튼
        csv = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name=f"multi_coin_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    
    else:
        st.info("📋 선택한 필터 조건에 맞는 거래가 없습니다.")
    
    # ===== 자동 새로고침 =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 설정")
    auto_refresh = st.sidebar.checkbox("🔄 대시보드 자동 새로고침 (30초)")
    
    # 코인별 통계 표시 (사이드바)
    if metrics['coin_breakdown']:
        st.sidebar.markdown("### 🪙 코인별 요약")
        for coin, stats in metrics['coin_breakdown'].items():
            emoji = {"BTC": "🟠", "ETH": "🔵", "SOL": "🟣"}.get(coin, "⚪")
            st.sidebar.markdown(f"""
            **{emoji} {coin}**  
            - 총 거래: {stats['total']}건  
            - 완료: {stats['closed']}건  
            - 손익: ${stats['pnl']:,.2f}  
            - 승률: {stats['win_rate']:.1f}%
            """)
    
    # 자동 새로고침
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()