#!/usr/bin/env python3
"""
Ethereum Day Trading Bot - Streamlit Dashboard with Log Monitoring
기존 ethereum_daytrading.db를 활용한 실시간 대시보드 + output.log 모니터링
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
import re

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

# CSS 스타일 (로그 스타일 추가)
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
    .log-container {
        background: #0F172A;
        border-radius: 8px;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        border: 1px solid #334155;
    }
    .log-error {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px 0;
    }
    .log-warning {
        color: #F59E0B;
        background: rgba(245, 158, 11, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px 0;
    }
    .log-info {
        color: #10B981;
        background: rgba(16, 185, 129, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px 0;
    }
    .log-debug {
        color: #6B7280;
        background: rgba(107, 114, 128, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px 0;
    }
    .log-trade {
        color: #8B5CF6;
        background: rgba(139, 92, 246, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px 0;
        font-weight: bold;
    }
    .log-default {
        color: #E5E7EB;
        padding: 2px 4px;
        margin: 1px 0;
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

def check_log_file_exists():
    """로그 파일 존재 확인"""
    log_path = "output.log"
    return os.path.exists(log_path)

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

# 로그 관련 함수들
def read_log_file(file_path="output.log", max_lines=500):
    """로그 파일을 읽어서 최근 라인들을 반환"""
    try:
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # 최근 max_lines 만큼만 반환
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]

def categorize_log_line(line):
    """로그 라인의 종류를 분류"""
    line_upper = line.upper()
    
    if 'ERROR' in line_upper or '에러' in line:
        return 'error'
    elif 'WARNING' in line_upper or 'WARN' in line_upper or '경고' in line:
        return 'warning'
    elif 'INFO' in line_upper or '정보' in line:
        return 'info'
    elif 'DEBUG' in line_upper or 'TRACE' in line_upper:
        return 'debug'
    elif any(word in line_upper for word in ['매수', '매도', 'BUY', 'SELL', 'TRADE', 'ORDER']):
        return 'trade'
    else:
        return 'default'

def extract_price_from_logs(log_lines, max_prices=30):
    """로그에서 이더리움 가격 정보 추출"""
    price_data = []
    
    # 가격 추출 패턴들
    price_patterns = [
        r'현재가[:\s]+\$?([0-9,]+\.?[0-9]*)',  # 현재가: $3,245.67
        r'ETH\s*Price[:\s]+\$?([0-9,]+\.?[0-9]*)',  # ETH Price: 3245.67
        r'price[:\s]+\$?([0-9,]+\.?[0-9]*)',  # price: 3245.67
        r'ETH[:\s]+\$([0-9,]+\.?[0-9]*)',  # ETH: $3,245.67
        r'이더리움[:\s]+\$?([0-9,]+\.?[0-9]*)',  # 이더리움: 3245.67
        r'\$([0-9,]+\.?[0-9]*)\s*(USD|USDT)',  # $3245.67 USD
        r'([0-9,]+\.?[0-9]*)\s*USD',  # 3245.67 USD
    ]
    
    # 시간 패턴들
    time_patterns = [
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
        r'(\d{2}:\d{2}:\d{2})',
        r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})',
        r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
    ]
    
    for line in log_lines:
        # 시간 정보 추출
        timestamp = None
        for pattern in time_patterns:
            time_match = re.search(pattern, line)
            if time_match:
                time_str = time_match.group(1)
                try:
                    if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', time_str):
                        timestamp = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    elif re.match(r'\d{2}:\d{2}:\d{2}', time_str):
                        today = datetime.now().date()
                        timestamp = datetime.combine(today, datetime.strptime(time_str, '%H:%M:%S').time())
                    elif re.match(r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', time_str):
                        timestamp = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                    break
                except:
                    continue
        
        # 가격 정보 추출
        for pattern in price_patterns:
            price_match = re.search(pattern, line, re.IGNORECASE)
            if price_match:
                try:
                    price_str = price_match.group(1).replace(',', '')
                    price = float(price_str)
                    
                    # 합리적인 이더리움 가격 범위 체크 ($100 ~ $10,000)
                    if 100 <= price <= 10000:
                        price_data.append({
                            'timestamp': timestamp or datetime.now(),
                            'price': price,
                            'log_line': line.strip()
                        })
                        break
                except:
                    continue
    
    # 최신 데이터부터 최대 max_prices개 반환
    return price_data[-max_prices:] if len(price_data) > max_prices else price_data

def create_log_price_chart(price_data):
    """로그에서 추출한 가격 데이터로 차트 생성"""
    if not price_data:
        return None
    
    df = pd.DataFrame(price_data)
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines+markers',
        name='ETH Price (from logs)',
        line=dict(color='#00D4AA', width=2),
        marker=dict(size=4, color='#00D4AA'),
        hovertemplate='<b>ETH Price</b><br>' +
                      'Time: %{x}<br>' +
                      'Price: $%{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Real-time ETH Price from Logs",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=300,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def parse_log_statistics(log_lines):
    """로그에서 통계 정보 추출"""
    stats = {
        'total_lines': len(log_lines),
        'error_count': 0,
        'warning_count': 0,
        'trade_count': 0,
        'last_update': datetime.now(),
        'bot_status': 'Unknown'
    }
    
    current_time = datetime.now()
    recent_logs = []
    
    # 로그 파일 자체의 최종 수정 시간도 확인
    try:
        if os.path.exists("output.log"):
            file_mtime = datetime.fromtimestamp(os.path.getmtime("output.log"))
            recent_logs.append(file_mtime)
    except:
        pass
    
    for line in log_lines[-50:]:  # 최근 50개 라인만 분석
        category = categorize_log_line(line)
        
        if category == 'error':
            stats['error_count'] += 1
        elif category == 'warning':
            stats['warning_count'] += 1
        elif category == 'trade':
            stats['trade_count'] += 1
        
        # 다양한 시간 형식 파싱 시도
        time_patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',  # 2024-08-24 13:36:59
            r'(\d{2}:\d{2}:\d{2})',  # 13:36:59
            r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})',  # 2024/08/24 13:36:59
            r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',  # Aug 24 13:36:59
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, line)
            if time_match:
                time_str = time_match.group(1)
                try:
                    # 다양한 형식으로 파싱 시도
                    if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', time_str):
                        log_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    elif re.match(r'\d{2}:\d{2}:\d{2}', time_str):
                        # 오늘 날짜로 가정
                        today = datetime.now().date()
                        log_time = datetime.combine(today, datetime.strptime(time_str, '%H:%M:%S').time())
                    elif re.match(r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', time_str):
                        log_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                    else:
                        continue
                    
                    recent_logs.append(log_time)
                    break
                except:
                    continue
    
    # 봇 상태 판단
    if recent_logs:
        latest_log = max(recent_logs)
        time_diff = current_time - latest_log
        
        if time_diff.total_seconds() < 300:  # 5분
            stats['bot_status'] = 'Running'
        elif time_diff.total_seconds() < 1800:  # 30분
            stats['bot_status'] = 'Idle'
        else:
            stats['bot_status'] = 'Stopped'
        
        stats['last_update'] = latest_log
    else:
        # 로그에서 시간을 파싱할 수 없지만 로그가 있다면
        if len(log_lines) > 0:
            # 파일 수정 시간 기준으로 판단
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime("output.log"))
                time_diff = current_time - file_mtime
                if time_diff.total_seconds() < 300:  # 5분
                    stats['bot_status'] = 'Running'
                elif time_diff.total_seconds() < 1800:  # 30분
                    stats['bot_status'] = 'Idle'
                else:
                    stats['bot_status'] = 'Stopped'
                stats['last_update'] = file_mtime
            except:
                stats['bot_status'] = 'Unknown'
    
    return stats

def format_log_line_html(line):
    """로그 라인을 HTML로 포맷팅"""
    category = categorize_log_line(line)
    escaped_line = line.replace('<', '&lt;').replace('>', '&gt;').strip()
    
    return f'<div class="log-{category}">{escaped_line}</div>'

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
    
    # 사이드바
    with st.sidebar:
        st.header("📊 Dashboard Control")
        
        # 로그 파일 상태
        log_exists = check_log_file_exists()
        db_exists = check_database_exists()
        
        st.subheader("📁 File Status")
        st.write(f"🗄️ Database: {'✅' if db_exists else '❌'}")
        st.write(f"📋 Log File: {'✅' if log_exists else '❌'}")
        
        # 데이터베이스 정보 표시
        if db_exists:
            db_info = get_database_info()
            if db_info:
                st.write(f"📊 DB Size: {db_info['total_size']:,} bytes")
                st.write("📋 Tables:")
                for table_name, table_info in db_info['tables'].items():
                    st.write(f"  • {table_name}: {table_info['row_count']:,} records")
        
        if log_exists:
            log_size = os.path.getsize("output.log")
            st.write(f"📝 Log Size: {log_size:,} bytes")
        
        st.markdown("---")
        
        # 새로고침 및 자동 업데이트 설정
        refresh_col1, refresh_col2 = st.columns(2)
        with refresh_col1:
            if st.button("🔄 Refresh"):
                st.cache_data.clear()
                st.rerun()
        
        with refresh_col2:
            auto_refresh = st.checkbox("⚡ Auto (30s)", value=False)
        
        # 로그 설정
        st.subheader("📋 Log Settings")
        log_lines = st.slider("Log Lines", 50, 1000, 200, 50)
        
        log_filter = st.selectbox(
            "Log Filter",
            ["All", "Errors Only", "Warnings+", "Trading Only", "Info+"]
        )
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        st.markdown("---")
        
        # 데이터 필터 옵션
        st.subheader("🔍 Data Filters")
        
        time_range = st.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"]
        )
        
        max_records = st.slider("Max Records", 50, 1000, 200, 50)
    
    # 탭 구성
    tab1, tab2 = st.tabs(["📊 Trading Dashboard", "📋 Live Logs"])
    
    with tab1:
        # 기존 대시보드 기능
        
        # 데이터베이스 존재 확인
        if not db_exists:
            st.error("❌ ethereum_daytrading.db 파일을 찾을 수 없습니다!")
            st.info("💡 거래 봇을 먼저 실행하여 데이터베이스를 생성해주세요.")
            
            # 현재 디렉토리의 파일들 표시
            current_files = [f for f in os.listdir('.') if f.endswith('.db')]
            if current_files:
                st.write("📁 현재 디렉토리의 데이터베이스 파일들:")
                for file in current_files:
                    st.write(f"  • {file}")
        else:
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
            
            # 봇 상태 표시 (로그 기반으로 업데이트)
            if log_exists:
                log_lines_data = read_log_file("output.log", 100)
                log_stats = parse_log_statistics(log_lines_data)
                
                # 상태별 아이콘과 색상
                status_icons = {
                    'Running': '🟢',
                    'Idle': '🟡', 
                    'Stopped': '🔴',
                    'Unknown': '⚫'
                }
                
                bot_status = f"{status_icons.get(log_stats['bot_status'], '⚫')} {log_stats['bot_status']}"
                last_activity = log_stats['last_update'].strftime('%H:%M:%S')
                
                # 마지막 활동으로부터 경과시간 계산
                time_diff = datetime.now() - log_stats['last_update']
                if time_diff.total_seconds() < 60:
                    time_ago = f"{int(time_diff.total_seconds())}s ago"
                elif time_diff.total_seconds() < 3600:
                    time_ago = f"{int(time_diff.total_seconds()/60)}m ago"
                else:
                    time_ago = f"{int(time_diff.total_seconds()/3600)}h ago"
                
                status_text = f"{bot_status} | Last Activity: {last_activity} ({time_ago}) | Errors: {log_stats['error_count']} | Warnings: {log_stats['warning_count']}"
                
                # 상태별 색상 적용하여 표시
                if log_stats['bot_status'] == 'Running':
                    st.success(f"🤖 Bot Status: {status_text}")
                elif log_stats['bot_status'] == 'Idle':
                    st.warning(f"🤖 Bot Status: {status_text}")
                elif log_stats['bot_status'] == 'Stopped':
                    st.error(f"🤖 Bot Status: {status_text}")
                else:
                    st.info(f"🤖 Bot Status: {status_text}")
                
            else:
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
            
            # 차트 섹션 (원래대로 2컬럼)
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
    
    with tab2:
        # 로그 모니터링 탭
        st.subheader("📋 Real-time Log Monitoring")
        
        if not log_exists:
            st.error("❌ output.log 파일을 찾을 수 없습니다!")
            st.info("💡 자동매매 봇을 실행하여 로그 파일을 생성해주세요.")
            
            # 현재 디렉토리의 로그 파일들 표시
            current_logs = [f for f in os.listdir('.') if f.endswith('.log') or f.endswith('.out')]
            if current_logs:
                st.write("📁 현재 디렉토리의 로그 파일들:")
                for log_file in current_logs:
                    st.write(f"  • {log_file}")
        else:
            # 로그 데이터 로드
            log_lines_data = read_log_file("output.log", log_lines)
            log_stats = parse_log_statistics(log_lines_data)
            
            # 상단에 가격 차트 추가
            price_data = extract_price_from_logs(log_lines_data, 50)
            if price_data:
                st.subheader("📈 Real-time Price Tracking")
                fig_price = create_log_price_chart(price_data)
                if fig_price:
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # 가격 통계
                    prices = [p['price'] for p in price_data]
                    price_col1, price_col2, price_col3, price_col4 = st.columns(4)
                    
                    with price_col1:
                        st.metric("Current", f"${prices[-1]:,.2f}")
                    with price_col2:
                        st.metric("High", f"${max(prices):,.2f}")
                    with price_col3:
                        st.metric("Low", f"${min(prices):,.2f}")
                    with price_col4:
                        price_change = prices[-1] - prices[0] if len(prices) > 1 else 0
                        st.metric("Change", f"${price_change:+.2f}")
            
            st.markdown("---")
            
            # 로그 통계 표시
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("📄 Total Lines", log_stats['total_lines'])
            
            with stat_col2:
                st.metric("❌ Errors", log_stats['error_count'])
            
            with stat_col3:
                st.metric("⚠️ Warnings", log_stats['warning_count'])
            
            with stat_col4:
                st.metric("💱 Trades", log_stats['trade_count'])
            
            st.markdown("---")
            
            # 로그 필터링
            filtered_logs = log_lines_data
            
            if log_filter == "Errors Only":
                filtered_logs = [line for line in log_lines_data if categorize_log_line(line) == 'error']
            elif log_filter == "Warnings+":
                filtered_logs = [line for line in log_lines_data if categorize_log_line(line) in ['error', 'warning']]
            elif log_filter == "Trading Only":
                filtered_logs = [line for line in log_lines_data if categorize_log_line(line) == 'trade']
            elif log_filter == "Info+":
                filtered_logs = [line for line in log_lines_data if categorize_log_line(line) in ['error', 'warning', 'info', 'trade']]
            
            # 로그 표시 영역
            st.subheader(f"📝 Live Logs ({len(filtered_logs)} lines)")
            
            # HTML 형식으로 로그 표시
            log_html = ""
            for line in filtered_logs[-100:]:  # 최신 100라인만 표시
                log_html += format_log_line_html(line) + "\n"
            
            st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
            
            # 최근 중요 이벤트 요약
            if log_stats['error_count'] > 0 or log_stats['warning_count'] > 0:
                st.markdown("---")
                st.subheader("⚠️ Recent Issues")
                
                error_logs = [line for line in log_lines_data[-50:] if categorize_log_line(line) in ['error', 'warning']]
                
                for line in error_logs[-10:]:  # 최근 10개 문제만 표시
                    category = categorize_log_line(line)
                    if category == 'error':
                        st.error(line.strip())
                    elif category == 'warning':
                        st.warning(line.strip())
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
        "🚀 Ethereum Day Trading Bot Dashboard with Live Log Monitoring | "
        f"Database: {'✅' if db_exists else '❌'} | "
        f"Logs: {'✅' if log_exists else '❌'} | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()