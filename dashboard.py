"""
AI Trading Dashboard - Streamlit
실시간 거래 모니터링 대시보드 (DB 전용)
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
BOT_SCRIPT = "auf_auto_fixed_logging.py"

def get_db_connection():
    """DB 연결"""
    try:
        return sqlite3.connect(DB_FILE)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return None

@st.cache_data(ttl=30)
def load_open_positions():
    """오픈 포지션 로드 (DB 전용)"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT 
                coin_symbol,
                action,
                entry_price,
                amount,
                leverage,
                investment_amount,
                sl_price,
                tp_price,
                trading_style,
                timestamp,
                ai_reasoning,
                confidence_score
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
        
        return df
        
    except Exception as e:
        st.error(f"포지션 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_closed_trades(days=30):
    """청산된 거래 로드"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        df = pd.read_sql_query("""
            SELECT 
                coin_symbol,
                action,
                entry_price,
                exit_price,
                amount,
                leverage,
                investment_amount,
                pnl,
                binance_pnl,
                pnl_percentage,
                trading_style,
                timestamp,
                close_timestamp,
                ai_reasoning,
                confidence_score
            FROM trades 
            WHERE status = 'CLOSED'
            AND close_timestamp >= ?
            ORDER BY close_timestamp DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['close_timestamp'] = pd.to_datetime(df['close_timestamp'])
            # binance_pnl 우선, 없으면 pnl
            df['final_pnl'] = df.apply(
                lambda x: x['binance_pnl'] if pd.notna(x['binance_pnl']) and x['binance_pnl'] != 0 else x['pnl'],
                axis=1
            )
        
        return df
        
    except Exception as e:
        st.error(f"거래 내역 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_ai_decisions(days=7):
    """AI 결정 로드"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        df = pd.read_sql_query("""
            SELECT 
                coin_symbol,
                decision_type,
                direction,
                leverage,
                confidence_score,
                reasoning,
                timestamp,
                related_trade_id
            FROM ai_decisions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        st.error(f"AI 결정 로드 오류: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_performance_data(days=30):
    """성과 데이터 분석"""
    closed_trades = load_closed_trades(days)
    
    if closed_trades.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    winning = closed_trades[closed_trades['final_pnl'] > 0]
    losing = closed_trades[closed_trades['final_pnl'] < 0]
    
    return {
        'total_trades': len(closed_trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0,
        'total_pnl': closed_trades['final_pnl'].sum(),
        'avg_win': winning['final_pnl'].mean() if len(winning) > 0 else 0,
        'avg_loss': abs(losing['final_pnl'].mean()) if len(losing) > 0 else 0
    }

@st.cache_data(ttl=60)
def analyze_confidence_performance(days=30):
    """신뢰도별 성과 분석"""
    closed_trades = load_closed_trades(days)
    
    if closed_trades.empty or 'confidence_score' not in closed_trades.columns:
        return {}
    
    # 유효한 신뢰도 데이터만
    valid_trades = closed_trades[closed_trades['confidence_score'].notna()].copy()
    
    if valid_trades.empty:
        return {}
    
    # 신뢰도 구간별 분류
    def categorize_confidence(score):
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        else:
            return 'low'
    
    valid_trades['conf_level'] = valid_trades['confidence_score'].apply(categorize_confidence)
    
    # 구간별 통계
    result = {}
    for level in ['high', 'medium', 'low']:
        level_trades = valid_trades[valid_trades['conf_level'] == level]
        if not level_trades.empty:
            winning = level_trades[level_trades['final_pnl'] > 0]
            result[level] = {
                'count': len(level_trades),
                'win_rate': len(winning) / len(level_trades) * 100,
                'avg_pnl': level_trades['final_pnl'].mean(),
                'total_pnl': level_trades['final_pnl'].sum()
            }
    
    return result

def get_bot_status():
    """봇 실행 상태 확인"""
    if psutil is None:
        return "Unknown"
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any(BOT_SCRIPT in cmd for cmd in cmdline):
                return f"Running (PID: {proc.info['pid']})"
        except:
            continue
    return "Stopped"

def start_bot():
    """봇 시작"""
    try:
        subprocess.Popen(['python', '-u', BOT_SCRIPT], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        st.error(f"봇 시작 실패: {e}")
        return False

def stop_bot():
    """봇 중지"""
    if psutil is None:
        return False
    
    stopped = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any(BOT_SCRIPT in cmd for cmd in cmdline):
                proc.terminate()
                stopped = True
        except:
            continue
    return stopped

def main():
    st.markdown('<h1 class="main-header">🤖 AI Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.header("⚙️ 설정")
    
    # 봇 상태 및 제어
    bot_status = get_bot_status()
    st.sidebar.metric("봇 상태", bot_status)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ 봇 시작", use_container_width=True, key="start_bot"):
            if start_bot():
                st.success("봇 시작됨")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.button("⏹️ 봇 중지", use_container_width=True, key="stop_bot"):
            if stop_bot():
                st.success("봇 중지됨")
                time.sleep(1)
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # 새로고침 설정
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (60초)", value=False)
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    
    col1, col2 = st.sidebar.columns(2)
    with col2:
        if st.button("🔄 수동 새로고침", use_container_width=True, key="manual_refresh"):
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("📊 데이터 소스: DB (바이낸스 API 미사용)")
    
    # 데이터 로드
    open_positions = load_open_positions()
    closed_trades = load_closed_trades(days=30)
    performance = load_performance_data(days=30)
    
    # 계정 정보
    total_invested = open_positions['investment_amount'].sum() if not open_positions.empty else 0
    realized_pnl = performance['total_pnl']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 계정 요약")
    st.sidebar.metric("투자 중", f"${total_invested:,.2f}")
    st.sidebar.metric("실현 손익", f"${realized_pnl:,.2f}")
    
    # 메인 메트릭
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "총 거래",
            f"{performance['total_trades']}회",
            delta=f"오픈: {len(open_positions)}"
        )
    
    with col2:
        st.metric(
            "승률",
            f"{performance['win_rate']:.1f}%",
            delta=f"승: {performance['winning_trades']}"
        )
    
    with col3:
        pnl_color = "normal" if realized_pnl >= 0 else "inverse"
        st.metric(
            "총 손익",
            f"${realized_pnl:,.2f}",
            delta=None,
            delta_color=pnl_color
        )
    
    with col4:
        st.metric(
            "평균 수익",
            f"${performance['avg_win']:,.2f}"
        )
    
    with col5:
        st.metric(
            "평균 손실",
            f"${performance['avg_loss']:,.2f}"
        )
    
    # 탭
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 포지션", 
        "📊 거래 내역", 
        "🧠 AI 진입 결정",
        "🔴 AI 포지션 평가",
        "📈 성과 분석",
        "📋 로그"
    ])
    
    # 탭 1: 포지션
    with tab1:
        st.header("📈 현재 포지션")
        
        if open_positions.empty:
            st.info("현재 오픈된 포지션이 없습니다.")
        else:
            # 포지션 테이블
            display_df = open_positions[[
                'coin_symbol', 'action', 'entry_price', 'amount', 
                'leverage', 'investment_amount', 'source', 'timestamp'
            ]].copy()
            
            display_df.columns = [
                '코인', '방향', '진입가', '수량', 
                '레버리지', '투자금', '소스', '시간'
            ]
            
            display_df['진입가'] = display_df['진입가'].apply(lambda x: f"${x:,.2f}")
            display_df['투자금'] = display_df['투자금'].apply(lambda x: f"${x:,.2f}")
            display_df['레버리지'] = display_df['레버리지'].apply(lambda x: f"{x}x")
            display_df['방향'] = display_df['방향'].str.upper()
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 포지션별 상세 정보
            st.subheader("📋 포지션 상세")
            
            for idx, row in open_positions.iterrows():
                with st.expander(f"{row['coin_symbol']} - {row['action'].upper()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("진입가", f"${row['entry_price']:,.2f}")
                        st.metric("수량", f"{row['amount']:.4f}")
                    
                    with col2:
                        st.metric("레버리지", f"{row['leverage']}x")
                        st.metric("투자금", f"${row['investment_amount']:,.2f}")
                    
                    with col3:
                        if pd.notna(row.get('confidence_score')):
                            st.metric("신뢰도", f"{row['confidence_score']}")
                        st.metric("소스", row['source'])
                    
                    if pd.notna(row.get('sl_price')):
                        st.write(f"**손절가:** ${row['sl_price']:,.2f}")
                    if pd.notna(row.get('tp_price')):
                        st.write(f"**익절가:** ${row['tp_price']:,.2f}")
                    
                    if pd.notna(row.get('ai_reasoning')):
                        st.write("**AI 분석:**")
                        st.info(row['ai_reasoning'])
    
    # 탭 2: 거래 내역
    with tab2:
        st.header("📊 거래 내역")
        
        if closed_trades.empty:
            st.info("청산된 거래가 없습니다.")
        else:
            # 거래 테이블
            display_df = closed_trades[[
                'coin_symbol', 'action', 'entry_price', 'exit_price',
                'leverage', 'final_pnl', 'close_timestamp'
            ]].copy()
            
            display_df.columns = [
                '코인', '방향', '진입가', '청산가',
                '레버리지', '손익', '청산 시간'
            ]
            
            display_df['진입가'] = display_df['진입가'].apply(lambda x: f"${x:,.2f}" if x > 0 else "N/A")
            display_df['청산가'] = display_df['청산가'].apply(lambda x: f"${x:,.2f}" if x > 0 else "N/A")
            display_df['레버리지'] = display_df['레버리지'].apply(lambda x: f"{x}x")
            display_df['방향'] = display_df['방향'].str.upper()
            
            # 손익 색상
            def color_pnl(val):
                try:
                    num_val = float(val.replace('$', '').replace(',', ''))
                    color = 'green' if num_val > 0 else 'red'
                    return f'color: {color}'
                except:
                    return ''
            
            display_df['손익'] = display_df['손익'].apply(lambda x: f"${x:+,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 차트
            col1, col2 = st.columns(2)
            
            with col1:
                # 코인별 손익
                coin_pnl = closed_trades.groupby('coin_symbol')['final_pnl'].sum().reset_index()
                coin_pnl = coin_pnl.sort_values('final_pnl', ascending=False)
                
                fig = px.bar(
                    coin_pnl,
                    x='coin_symbol',
                    y='final_pnl',
                    title='코인별 총 손익',
                    labels={'coin_symbol': '코인', 'final_pnl': '손익 (USDT)'},
                    color='final_pnl',
                    color_continuous_scale=['red', 'gray', 'green']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 일별 손익
                daily = closed_trades.copy()
                daily['date'] = daily['close_timestamp'].dt.date
                daily_pnl = daily.groupby('date')['final_pnl'].sum().reset_index()
                
                fig = px.line(
                    daily_pnl,
                    x='date',
                    y='final_pnl',
                    title='일별 손익',
                    labels={'date': '날짜', 'final_pnl': '손익 (USDT)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
    
    # 탭 3: AI 진입 결정
    with tab3:
        st.header("🧠 AI 진입 결정 분석")
        
        # 신뢰도별 성과
        conf_analysis = analyze_confidence_performance(days=30)
        
        if conf_analysis:
            st.subheader("📊 신뢰도별 성과")
            
            conf_data = []
            for level, stats in conf_analysis.items():
                level_name = {
                    'high': '높음 (80+)',
                    'medium': '중간 (60-79)',
                    'low': '낮음 (60미만)'
                }[level]
                
                conf_data.append({
                    '신뢰도': level_name,
                    '거래 수': stats['count'],
                    '승률': f"{stats['win_rate']:.1f}%",
                    '평균 손익': f"${stats['avg_pnl']:+,.2f}",
                    '총 손익': f"${stats['total_pnl']:+,.2f}"
                })
            
            st.table(pd.DataFrame(conf_data))
            
            # 신뢰도별 승률 차트
            chart_data = pd.DataFrame([
                {'신뢰도': level, '승률': stats['win_rate']}
                for level, stats in conf_analysis.items()
            ])
            
            fig = px.bar(
                chart_data,
                x='신뢰도',
                y='승률',
                title='신뢰도별 승률',
                labels={'신뢰도': '신뢰도 구간', '승률': '승률 (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 🆕 ENTRY 결정만 필터링
        st.subheader("🔍 최근 진입 결정 (ENTRY)")
        
        ai_decisions = load_ai_decisions(days=7)
        
        if ai_decisions.empty:
            st.info("AI 진입 결정 기록이 없습니다.")
        else:
            # ENTRY만 필터링
            entry_decisions = ai_decisions[ai_decisions['decision_type'] == 'ENTRY']
            
            if entry_decisions.empty:
                st.info("최근 7일간 진입 결정이 없습니다.")
            else:
                for idx, row in entry_decisions.head(10).iterrows():
                    # 결과 확인
                    result_emoji = ""
                    if pd.notna(row.get('related_trade_id')):
                        trade_id = row['related_trade_id']
                        # 해당 거래의 결과 조회
                        closed_trades = load_closed_trades(days=30)
                        if not closed_trades.empty:
                            trade_result = closed_trades[closed_trades.index == trade_id]
                            if not trade_result.empty:
                                pnl = trade_result.iloc[0]['final_pnl']
                                result_emoji = " ✅" if pnl > 0 else " ❌"
                    
                    with st.expander(f"{row['coin_symbol']} - {row['direction'].upper()}{result_emoji} ({row['timestamp'].strftime('%m-%d %H:%M')})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**방향:** {row['direction']}")
                            st.write(f"**레버리지:** {row['leverage']}x")
                        
                        with col2:
                            st.write(f"**신뢰도:** {row['confidence_score']}")
                            if pd.notna(row.get('related_trade_id')):
                                st.write(f"**거래 ID:** {row['related_trade_id']}")
                        
                        st.write("**AI 진입 분석:**")
                        st.info(row['reasoning'])
    
    # 탭 4: AI 포지션 평가 (조기 청산)
    with tab4:
        st.header("🔴 AI 포지션 평가 (조기 청산)")
        
        st.info("1시간마다 실행되는 AI 중간평가 결과")
        
        # EXIT 결정만 필터링
        ai_decisions = load_ai_decisions(days=7)
        
        if ai_decisions.empty:
            st.warning("AI 포지션 평가 기록이 없습니다.")
        else:
            exit_decisions = ai_decisions[ai_decisions['decision_type'] == 'EXIT']
            
            if exit_decisions.empty:
                st.success("최근 7일간 조기 청산이 없습니다. (모든 포지션이 TP/SL로 청산됨)")
            else:
                # 통계
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("총 조기 청산", f"{len(exit_decisions)}회")
                
                with col2:
                    # 조기 청산의 평균 신뢰도
                    avg_conf = exit_decisions['confidence_score'].mean()
                    st.metric("평균 신뢰도", f"{avg_conf:.0f}")
                
                with col3:
                    # 조기 청산의 결과 (trades 테이블에서 확인)
                    closed_trades = load_closed_trades(days=7)
                    if not closed_trades.empty:
                        # ai_reasoning에 "[청산 이유]" 포함된 거래
                        early_exits = closed_trades[
                            closed_trades['ai_reasoning'].str.contains('[청산 이유]', na=False)
                        ]
                        if not early_exits.empty:
                            profitable = len(early_exits[early_exits['final_pnl'] > 0])
                            exit_win_rate = profitable / len(early_exits) * 100
                            st.metric("조기청산 승률", f"{exit_win_rate:.1f}%")
                
                st.markdown("---")
                st.subheader("🔍 최근 포지션 평가 기록")
                
                for idx, row in exit_decisions.head(10).iterrows():
                    # 결과 확인
                    result_text = ""
                    result_color = "blue"
                    
                    if pd.notna(row.get('related_trade_id')):
                        trade_id = row['related_trade_id']
                        closed_trades = load_closed_trades(days=30)
                        if not closed_trades.empty:
                            # related_trade_id가 인덱스가 아닐 수 있으므로 수정
                            conn = get_db_connection()
                            trade_result = pd.read_sql_query("""
                                SELECT coin_symbol, pnl, binance_pnl, close_timestamp
                                FROM trades
                                WHERE id = ? AND status = 'CLOSED'
                            """, conn, params=(trade_id,))
                            conn.close()
                            
                            if not trade_result.empty:
                                pnl = trade_result.iloc[0]['binance_pnl']
                                if pd.isna(pnl) or pnl == 0:
                                    pnl = trade_result.iloc[0]['pnl']
                                
                                if pnl > 0:
                                    result_text = f" → 결과: +${pnl:,.2f} ✅"
                                    result_color = "green"
                                else:
                                    result_text = f" → 결과: ${pnl:,.2f} ❌"
                                    result_color = "red"
                    
                    with st.expander(f"{row['coin_symbol']} - 조기 청산 ({row['timestamp'].strftime('%m-%d %H:%M')}){result_text}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**결정:** 포지션 청산")
                            st.write(f"**신뢰도:** {row['confidence_score']}")
                        
                        with col2:
                            st.write(f"**시간:** {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                            if pd.notna(row.get('related_trade_id')):
                                st.write(f"**거래 ID:** {row['related_trade_id']}")
                        
                        st.write("**AI 평가 (청산 이유):**")
                        st.warning(row['reasoning'])
                        
                        # 원본 거래의 진입 분석도 표시
                        if pd.notna(row.get('related_trade_id')):
                            conn = get_db_connection()
                            original_trade = pd.read_sql_query("""
                                SELECT ai_reasoning, entry_price, action
                                FROM trades
                                WHERE id = ?
                            """, conn, params=(trade_id,))
                            conn.close()
                            
                            if not original_trade.empty and pd.notna(original_trade.iloc[0]['ai_reasoning']):
                                ai_reasoning = original_trade.iloc[0]['ai_reasoning']
                                # [청산 이유] 이전 부분만 (진입 분석)
                                if '[청산 이유]' in ai_reasoning:
                                    entry_reason = ai_reasoning.split('[청산 이유]')[0].strip()
                                else:
                                    entry_reason = ai_reasoning
                                
                                st.write("**원래 진입 분석:**")
                                st.info(entry_reason)
    
    # 탭 5: 성과 분석
    with tab5:
        st.header("📈 성과 분석")
        
        st.info("📊 DB 데이터 기반 분석")
        
        if closed_trades.empty:
            st.warning("분석할 거래 데이터가 없습니다.")
        else:
            # 손익 분포
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 손익 분포")
                fig = px.histogram(
                    closed_trades,
                    x='final_pnl',
                    nbins=20,
                    title='거래별 손익 분포',
                    labels={'final_pnl': '손익 (USDT)', 'count': '거래 수'}
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📅 누적 손익")
                cumulative = closed_trades.sort_values('close_timestamp').copy()
                cumulative['cumulative_pnl'] = cumulative['final_pnl'].cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative['close_timestamp'],
                    y=cumulative['cumulative_pnl'],
                    mode='lines+markers',
                    name='누적 손익',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title='시간별 누적 손익',
                    xaxis_title='시간',
                    yaxis_title='누적 손익 (USDT)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 상세 메트릭
            st.subheader("📊 상세 메트릭")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("평균 수익", f"${performance['avg_win']:,.2f}")
                best_trade = closed_trades['final_pnl'].max()
                st.metric("최대 수익", f"${best_trade:+,.2f}")
            
            with col2:
                st.metric("평균 손실", f"${performance['avg_loss']:,.2f}")
                worst_trade = closed_trades['final_pnl'].min()
                st.metric("최대 손실", f"${worst_trade:+,.2f}")
            
            with col3:
                # Risk/Reward
                if performance['avg_loss'] > 0:
                    rr_ratio = performance['avg_win'] / performance['avg_loss']
                    st.metric("Risk/Reward", f"1:{rr_ratio:.2f}")
                else:
                    st.metric("Risk/Reward", "N/A")
            
            with col4:
                # 수익 팩터
                total_wins = closed_trades[closed_trades['final_pnl'] > 0]['final_pnl'].sum()
                total_losses = abs(closed_trades[closed_trades['final_pnl'] < 0]['final_pnl'].sum())
                if total_losses > 0:
                    profit_factor = total_wins / total_losses
                    st.metric("수익 팩터", f"{profit_factor:.2f}")
                else:
                    st.metric("수익 팩터", "∞")
    
    # 탭 6: 로그
    with tab6:
        st.header("📋 로그")
        
        if os.path.exists(LOG_FILE):
            log_lines = st.slider("표시할 라인 수", 50, 500, 200)
            
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_lines = lines[-log_lines:]
                    log_text = ''.join(recent_lines)
                
                st.text_area("로그 내용", log_text, height=600, key="log_area")
                
                if st.button("📥 전체 로그 다운로드", key="download_log"):
                    with open(LOG_FILE, 'rb') as f:
                        st.download_button(
                            label="다운로드",
                            data=f,
                            file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                            mime="text/plain"
                        )
            
            except Exception as e:
                st.error(f"로그 읽기 오류: {e}")
        else:
            st.warning(f"로그 파일을 찾을 수 없습니다: {LOG_FILE}")

if __name__ == "__main__":
    main()
