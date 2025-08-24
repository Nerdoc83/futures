#!/usr/bin/env python3
"""
Ethereum Day Trading Bot - Streamlit Dashboard
기존 ethereum_daytrading.db를 활용한 실시간 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import os
import time
import warnings
import logging
from pathlib import Path

# 모든 경고와 로그 메시지 숨기기
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# 환경 변수 설정으로 추가 경고 방지
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# 페이지 설정
st.set_page_config(
    page_title="ETH Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1F2937, #374151);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00D4AA;
        margin: 0.5rem 0;
    }
    .profit-positive {
        color: #10B981;
        font-weight: bold;
    }
    .profit-negative {
        color: #EF4444;
        font-weight: bold;
    }
    .status-running {
        color: #10B981;
    }
    .status-stopped {
        color: #EF4444;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(145deg, #1F2937, #374151);
    }
</style>
""", unsafe_allow_html=True)

def setup_streamlit_config():
    """Streamlit 설정을 통해 경고 메시지 최소화"""
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    config_content = """[global]
developmentMode = false

[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#00D4AA"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1F2937"
textColor = "#FAFAFA"
font = "sans serif"
"""
    
    config_file = config_dir / "config.toml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)

def check_database_exists():
    """데이터베이스 파일 존재 확인"""
    db_path = "ethereum_daytrading.db"
    return os.path.exists(db_path)

def get_database_info():
    """데이터베이스 스키마 정보 조회"""
    if not check_database_exists():
        return None
    
    conn = sqlite3.connect("ethereum_daytrading.db")
    cursor = conn.cursor()
    
    # 모든 테이블 조회
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    db_info = {
        'tables': {},
        'total_size': os.path.getsize("ethereum_daytrading.db")
    }
    
    for table in tables:
        table_name = table[0]
        # 테이블 스키마 조회
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # 테이블 레코드 수 조회
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        db_info['tables'][table_name] = {
            'columns': [col[1] for col in columns],
            'column_details': columns,
            'row_count': count
        }
    
    conn.close()
    return db_info

# 캐시 데코레이터를 조건부로 적용
def conditional_cache(func):
    try:
        return st.cache_data(ttl=30)(func)
    except:
        return func

@conditional_cache
def load_trades_data(limit=200):
    """거래 데이터 로드"""
    if not check_database_exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect("ethereum_daytrading.db")
    
    try:
        # 일반적인 거래 테이블 이름들을 시도
        possible_tables = ['trades', 'trade_history', 'trading_history', 'transactions']
        trades_df = pd.DataFrame()
        
        for table_name in possible_tables:
            try:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
                trades_df = pd.read_sql_query(query, conn)
                break
            except:
                continue
        
        if not trades_df.empty and 'timestamp' in trades_df.columns:
            # 타임스탬프 컬럼을 datetime으로 변환
            try:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            except:
                pass
        
        conn.close()
        return trades_df
    
    except Exception as e:
        conn.close()
        st.error(f"거래 데이터 로드 중 오류: {e}")
        return pd.DataFrame()

@conditional_cache
def load_price_data(limit=500):
    """가격 데이터 로드"""
    if not check_database_exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect("ethereum_daytrading.db")
    
    try:
        # 일반적인 가격 데이터 테이블 이름들을 시도
        possible_tables = ['price_data', 'prices', 'market_data', 'ohlcv']
        price_df = pd.DataFrame()
        
        for table_name in possible_tables:
            try:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}"
                price_df = pd.read_sql_query(query, conn)
                break
            except:
                continue
        
        if not price_df.empty and 'timestamp' in price_df.columns:
            try:
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            except:
                pass
        
        conn.close()
        return price_df
    
    except Exception as e:
        conn.close()
        return pd.DataFrame()

@conditional_cache
def load_balance_data():
    """잔액/포트폴리오 데이터 로드"""
    if not check_database_exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect("ethereum_daytrading.db")
    
    try:
        possible_tables = ['portfolio', 'balance', 'account', 'wallet']
        balance_df = pd.DataFrame()
        
        for table_name in possible_tables:
            try:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100"
                balance_df = pd.read_sql_query(query, conn)
                break
            except:
                continue
        
        if not balance_df.empty and 'timestamp' in balance_df.columns:
            try:
                balance_df['timestamp'] = pd.to_datetime(balance_df['timestamp'])
            except:
                pass
        
        conn.close()
        return balance_df
    
    except Exception as e:
        conn.close()
        return pd.DataFrame()

def get_latest_metrics(trades_df):
    """최신 거래 메트릭스 계산"""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'total_profit': 0,
            'win_rate': 0,
            'current_balance': 0,
            'last_trade_time': 'N/A'
        }
    
    metrics = {}
    
    # 총 거래 수
    metrics['total_trades'] = len(trades_df)
    
    # 총 수익 계산
    if 'profit' in trades_df.columns:
        metrics['total_profit'] = trades_df['profit'].sum()
    elif 'pnl' in trades_df.columns:
        metrics['total_profit'] = trades_df['pnl'].sum()
    else:
        metrics['total_profit'] = 0
    
    # 승률 계산
    if 'profit' in trades_df.columns:
        winning_trades = len(trades_df[trades_df['profit'] > 0])
    elif 'pnl' in trades_df.columns:
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
    else:
        winning_trades = 0
    
    metrics['win_rate'] = (winning_trades / len(trades_df) * 100) if len(trades_df) > 0 else 0
    
    # 현재 잔액
    if 'balance' in trades_df.columns:
        metrics['current_balance'] = trades_df.iloc[0]['balance']
    else:
        metrics['current_balance'] = 0
    
    # 마지막 거래 시간
    if 'timestamp' in trades_df.columns:
        metrics['last_trade_time'] = trades_df.iloc[0]['timestamp']
    else:
        metrics['last_trade_time'] = 'N/A'
    
    return metrics

def create_price_chart(price_df, trades_df):
    """가격 차트 생성"""
    fig = go.Figure()
    
    if not price_df.empty:
        # 가격 컬럼 찾기
        price_col = None
        for col in ['price', 'close', 'last_price', 'current_price']:
            if col in price_df.columns:
                price_col = col
                break
        
        if price_col and 'timestamp' in price_df.columns:
            # 시간순 정렬
            price_df_sorted = price_df.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=price_df_sorted['timestamp'],
                y=price_df_sorted[price_col],
                mode='lines',
                name='ETH Price',
                line=dict(color='#00D4AA', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 212, 170, 0.1)'
            ))
    
    # 거래 포인트 추가
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        action_col = None
        price_col = None
        
        for col in ['action', 'side', 'type']:
            if col in trades_df.columns:
                action_col = col
                break
        
        for col in ['price', 'executed_price', 'fill_price']:
            if col in trades_df.columns:
                price_col = col
                break
        
        if action_col and price_col:
            for _, trade in trades_df.iterrows():
                try:
                    action = str(trade[action_col]).upper()
                    color = '#10B981' if 'BUY' in action else '#EF4444'
                    symbol = 'triangle-up' if 'BUY' in action else 'triangle-down'
                    
                    fig.add_trace(go.Scatter(
                        x=[trade['timestamp']],
                        y=[trade[price_col]],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=10,
                            symbol=symbol
                        ),
                        name=f"{action} - ${trade[price_col]:.2f}",
                        showlegend=False,
                        hovertemplate=f"<b>{action}</b><br>" +
                                      f"Price: ${trade[price_col]:.2f}<br>" +
                                      f"Time: {trade['timestamp']}<extra></extra>"
                    ))
                except:
                    continue
    
    fig.update_layout(
        title="Ethereum Price with Trading Points",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=450,
        showlegend=True
    )
    
    return fig

def main():
    # Streamlit 설정 적용 (경고 메시지 제거)
    setup_streamlit_config()
    
    # 헤더
    st.markdown('<h1 class="main-header">🚀 Ethereum Day Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # 데이터베이스 존재 확인
    if not check_database_exists():
        st.error("❌ ethereum_daytrading.db 파일을 찾을 수 없습니다!")
        st.info("💡 거래 봇을 먼저 실행하여 데이터베이스를 생성해주세요.")
        
        # 현재 디렉토리의 파일들 표시
        current_files = [f for f in os.listdir('.') if f.endswith('.db')]
        if current_files:
            st.write("📁 현재 디렉토리의 데이터베이스 파일들:")
            for file in current_files:
                st.write(f"  • {file}")
        return
    
    # 데이터베이스 정보 표시
    db_info = get_database_info()
    
    # 사이드바
    with st.sidebar:
        st.header("📊 Dashboard Control")
        
        # 데이터베이스 정보
        st.subheader("🗄️ Database Info")
        if db_info:
            st.write(f"📁 Size: {db_info['total_size']:,} bytes")
            st.write("📋 Tables:")
            for table_name, table_info in db_info['tables'].items():
                st.write(f"  • {table_name}: {table_info['row_count']:,} records")
        
        st.markdown("---")
        
        # 새로고침 버튼
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # 자동 새로고침 설정
        auto_refresh = st.checkbox("⚡ Auto Refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        st.markdown("---")
        
        # 필터 옵션
        st.subheader("🔍 Data Filters")
        
        time_range = st.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"]
        )
        
        max_records = st.slider("Max Records", 50, 1000, 200, 50)
    
    # 데이터 로드
    trades_df = load_trades_data(max_records)
    price_df = load_price_data(max_records)
    balance_df = load_balance_data()
    
    # 메트릭스 계산
    metrics = get_latest_metrics(trades_df)
    
    # 메인 메트릭스 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Current Balance",
            value=f"${metrics['current_balance']:,.2f}",
            delta=None
        )
    
    with col2:
        profit_color = "normal" if metrics['total_profit'] >= 0 else "inverse"
        st.metric(
            label="📈 Total Profit/Loss",
            value=f"${metrics['total_profit']:.2f}",
            delta_color=profit_color
        )
    
    with col3:
        st.metric(
            label="🔄 Total Trades",
            value=f"{metrics['total_trades']:,}"
        )
    
    with col4:
        st.metric(
            label="🎯 Win Rate",
            value=f"{metrics['win_rate']:.1f}%"
        )
    
    # 봇 상태 표시
    bot_status = "🟢 Running" if metrics['total_trades'] > 0 else "🔴 Stopped"
    last_activity = metrics['last_trade_time']
    
    if isinstance(last_activity, pd.Timestamp):
        time_since = datetime.now() - last_activity.to_pydatetime()
        if time_since.total_seconds() < 3600:  # 1시간 미만
            status_text = f"{bot_status} | Last Activity: {int(time_since.total_seconds()/60)}m ago"
        else:
            status_text = f"{bot_status} | Last Activity: {last_activity.strftime('%Y-%m-%d %H:%M')}"
    else:
        status_text = f"{bot_status} | Last Activity: {last_activity}"
    
    st.info(f"🤖 Bot Status: {status_text}")
    
    st.markdown("---")
    
    # 차트 섹션
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("📊 Price Chart & Trading Points")
        
        if not price_df.empty or not trades_df.empty:
            fig = create_price_chart(price_df, trades_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📊 가격 데이터를 찾을 수 없습니다.")
    
    with chart_col2:
        st.subheader("💰 Portfolio Performance")
        
        # 잔액 변화 차트
        if not trades_df.empty and 'balance' in trades_df.columns:
            trades_sorted = trades_df.sort_values('timestamp')
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=trades_sorted['timestamp'],
                y=trades_sorted['balance'],
                mode='lines+markers',
                name='Balance',
                line=dict(color='#8B5CF6', width=3),
                marker=dict(size=6)
            ))
            
            fig2.update_layout(
                title="Balance Over Time",
                xaxis_title="Time",
                yaxis_title="Balance (USD)",
                template="plotly_dark",
                height=450
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📊 잔액 데이터를 찾을 수 없습니다.")
    
    # 최근 거래 내역
    st.subheader("📋 Recent Trading Activity")
    
    if not trades_df.empty:
        # 표시할 컬럼 선택 (데이터에 따라 동적으로)
        display_columns = []
        column_mapping = {
            'timestamp': 'Time',
            'action': 'Action',
            'side': 'Side',
            'type': 'Type',
            'price': 'Price',
            'executed_price': 'Executed Price',
            'amount': 'Amount',
            'quantity': 'Quantity',
            'profit': 'Profit/Loss',
            'pnl': 'P&L',
            'balance': 'Balance',
            'fee': 'Fee'
        }
        
        for col in trades_df.columns:
            if col in column_mapping:
                display_columns.append(col)
        
        if display_columns:
            display_df = trades_df[display_columns].copy()
            
            # 데이터 포맷팅
            for col in display_df.columns:
                if 'price' in col.lower() or 'balance' in col.lower():
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                elif 'profit' in col.lower() or 'pnl' in col.lower():
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x >= 0 else f"-${abs(x):.2f}" if pd.notna(x) else "N/A"
                    )
                elif 'timestamp' in col.lower() and 'time' in col.lower():
                    display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 컬럼명 변경
            display_df.columns = [column_mapping.get(col, col.title()) for col in display_df.columns]
            
            st.dataframe(
                display_df.head(20),  # 최신 20개만 표시
                use_container_width=True,
                height=400
            )
        else:
            st.write(trades_df.head(10))
    else:
        st.info("📋 거래 데이터를 찾을 수 없습니다.")
    
    # 상세 통계
    if not trades_df.empty:
        st.markdown("---")
        st.subheader("📈 Detailed Statistics")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.write("**📊 Trade Analysis**")
            
            # 거래 유형별 분석
            if 'action' in trades_df.columns:
                action_counts = trades_df['action'].value_counts()
                for action, count in action_counts.items():
                    st.write(f"• {action}: {count}")
            
            # 평균 거래 크기
            if 'amount' in trades_df.columns:
                avg_amount = trades_df['amount'].mean()
                st.write(f"• Avg Trade Size: {avg_amount:.4f} ETH")
        
        with stats_col2:
            st.write("**💰 Performance Metrics**")
            
            profit_col = 'profit' if 'profit' in trades_df.columns else 'pnl' if 'pnl' in trades_df.columns else None
            
            if profit_col:
                profitable_trades = trades_df[trades_df[profit_col] > 0]
                losing_trades = trades_df[trades_df[profit_col] < 0]
                
                st.write(f"• Profitable: {len(profitable_trades)}")
                st.write(f"• Losing: {len(losing_trades)}")
                
                if len(profitable_trades) > 0:
                    avg_win = profitable_trades[profit_col].mean()
                    st.write(f"• Avg Win: ${avg_win:.2f}")
                
                if len(losing_trades) > 0:
                    avg_loss = abs(losing_trades[profit_col].mean())
                    st.write(f"• Avg Loss: ${avg_loss:.2f}")
        
        with stats_col3:
            st.write("**📈 Best/Worst**")
            
            if profit_col:
                max_profit = trades_df[profit_col].max()
                max_loss = trades_df[profit_col].min()
                
                st.write(f"• Best Trade: ${max_profit:.2f}")
                st.write(f"• Worst Trade: ${max_loss:.2f}")
                
                # 샤프 비율 등 추가 메트릭스
                if len(trades_df) > 1:
                    returns = trades_df[profit_col].pct_change().dropna()
                    if len(returns) > 0 and returns.std() != 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(252)
                        st.write(f"• Sharpe Ratio: {sharpe:.2f}")
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
        "🚀 Ethereum Day Trading Bot Dashboard | "
        f"Database: ethereum_daytrading.db | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()