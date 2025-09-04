import sqlite3
import pandas as pd

DB_FILE = "multi_coin_daytrading.db"

def analyze_trading_performance():
    conn = sqlite3.connect(DB_FILE)
    
    print("=== 1. 코인별 성과 분석 ===")
    query1 = """
    SELECT 
        coin_symbol,
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
        ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
        ROUND(AVG(profit_loss_percentage), 2) as avg_profit_pct,
        ROUND(MAX(profit_loss_percentage), 2) as max_win_pct,
        ROUND(MIN(profit_loss_percentage), 2) as max_loss_pct,
        status
    FROM trades 
    WHERE status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    GROUP BY coin_symbol, status
    ORDER BY coin_symbol, status;
    """
    df1 = pd.read_sql_query(query1, conn)
    print(df1.to_string(index=False))
    
    print("\n=== 2. 종료 방식별 분석 ===")
    query2 = """
    SELECT 
        status,
        COUNT(*) as trades,
        ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
        ROUND(AVG(profit_loss_percentage), 2) as avg_return_pct
    FROM trades
    WHERE status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    GROUP BY status;
    """
    df2 = pd.read_sql_query(query2, conn)
    print(df2.to_string(index=False))
    
    print("\n=== 3. 롱/숏 방향별 분석 ===")
    query3 = """
    SELECT 
        action,
        COUNT(*) as trades,
        ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
        ROUND(AVG(profit_loss_percentage), 2) as avg_return_pct
    FROM trades
    WHERE status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    GROUP BY action;
    """
    df3 = pd.read_sql_query(query3, conn)
    print(df3.to_string(index=False))
    
    print("\n=== 4. 레버리지별 분석 ===")
    query4 = """
    SELECT 
        leverage,
        COUNT(*) as trades,
        ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
        ROUND(AVG(profit_loss_percentage), 2) as avg_return_pct
    FROM trades
    WHERE status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    GROUP BY leverage
    ORDER BY leverage;
    """
    df4 = pd.read_sql_query(query4, conn)
    print(df4.to_string(index=False))
    
    print("\n=== 5. AI 시장 조건 판단별 분석 ===")
    query5 = """
    SELECT 
        COALESCE(a.market_conditions, 'Unknown') as market_condition,
        COUNT(*) as trades,
        ROUND(SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
        ROUND(AVG(t.profit_loss_percentage), 2) as avg_return_pct
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    GROUP BY a.market_conditions;
    """
    df5 = pd.read_sql_query(query5, conn)
    print(df5.to_string(index=False))
    
    print("\n=== 6. 최근 10개 거래 상세 ===")
    query6 = """
    SELECT 
        datetime(t.timestamp) as trade_time,
        t.coin_symbol,
        t.action,
        t.leverage,
        ROUND(t.profit_loss_percentage, 2) as return_pct,
        t.status,
        COALESCE(a.detailed_reasoning, 'No reasoning') as ai_reasoning
    FROM trades t
    LEFT JOIN ai_analysis a ON t.id = a.trade_id
    WHERE t.status IN ('CLOSED', 'CLOSED_RAPID_STOP')
    ORDER BY t.timestamp DESC
    LIMIT 10;
    """
    df6 = pd.read_sql_query(query6, conn)
    print(df6.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    analyze_trading_performance()