#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 거래 PnL 정정 스크립트
바이낸스 API를 통해 실제 realized PnL을 조회하여 DB 업데이트
"""

import ccxt
import os
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing import Tuple

load_dotenv()

# 설정
DB_FILE = "live_trading.db"

# 바이낸스 연결
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})

LIVE_TRADING_CONFIG = {
    'MAKER_FEE': 0.02,  # 0.02%
    'TAKER_FEE': 0.05,  # 0.05%
}

def get_utc_now():
    """UTC 현재 시간 반환"""
    return datetime.now(timezone.utc)

def parse_db_timestamp(timestamp_str):
    """DB timestamp를 UTC datetime으로 변환"""
    if not timestamp_str:
        return None
    iso_str = timestamp_str.replace(' ', 'T')
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def calculate_pnl_from_price(entry_price: float, exit_price: float, position_size: float, side: str, leverage: int) -> Tuple[float, float]:
    """
    바이낸스 선물 PnL 정확한 계산 (수수료 포함)
    """
    try:
        # 1. 실제 계약 수량 계산
        quantity = (position_size * leverage) / entry_price
        
        # 2. 가격 차이로 손익 계산
        if side.upper() == 'LONG':
            pnl_before_fees = (exit_price - entry_price) * quantity
        else:  # SHORT
            pnl_before_fees = (entry_price - exit_price) * quantity
        
        # 3. 총 거래금액 계산
        entry_notional = quantity * entry_price
        exit_notional = quantity * exit_price
        
        # 4. 수수료 계산
        maker_fee = LIVE_TRADING_CONFIG.get('MAKER_FEE', 0.02)
        taker_fee = LIVE_TRADING_CONFIG.get('TAKER_FEE', 0.05)
        
        entry_fee = entry_notional * (maker_fee / 100)
        exit_fee = exit_notional * (taker_fee / 100)
        total_fees = entry_fee + exit_fee
        
        # 5. 최종 실현 손익
        realized_pnl = pnl_before_fees - total_fees
        
        # 6. 수익률 계산
        final_pnl_pct = (realized_pnl / position_size) * 100
        
        return realized_pnl, final_pnl_pct
        
    except Exception as e:
        print(f"   ❌ PnL 계산 오류: {e}")
        return 0.0, 0.0

def update_closed_trades_pnl():
    """
    기존 청산 거래들의 PnL을 바이낸스에서 정확하게 가져와 업데이트
    """
    print(f"\n{'='*80}")
    print(f"🔧 청산 거래 PnL 정정 (바이낸스 실제 데이터)")
    print(f"{'='*80}")
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 최근 30일간 청산된 거래 조회
    since_date = (get_utc_now() - timedelta(days=30)).isoformat()
    c.execute('''
        SELECT id, coin_symbol, side, entry_price, exit_price, quantity, 
               position_size, leverage, timestamp, pnl, pnl_percent
        FROM trades 
        WHERE status = 'CLOSED' AND timestamp >= ?
        ORDER BY timestamp DESC
    ''', (since_date,))
    
    closed_trades = c.fetchall()
    conn.close()
    
    if not closed_trades:
        print("   📊 업데이트할 청산 거래가 없습니다")
        print(f"{'='*80}\n")
        return 0
    
    print(f"   📊 {len(closed_trades)}개 청산 거래 PnL 정정 시작...\n")
    
    updated_count = 0
    
    for trade_data in closed_trades:
        trade_id, coin, side, entry_price, exit_price, quantity, position_size, leverage, timestamp, old_pnl, old_pnl_pct = trade_data
        symbol = f"{coin}/USDT:USDT"
        binance_symbol = f"{coin}USDT"
        
        try:
            print(f"   🔍 ID {trade_id}: {coin} {side}")
            print(f"      기존 PnL: ${old_pnl:+.2f} ({old_pnl_pct:+.2f}%)")
            
            # 거래 시간 파싱
            trade_dt = parse_db_timestamp(timestamp)
            if not trade_dt:
                print(f"      ❌ 시간 파싱 실패")
                continue
            
            trade_timestamp = int(trade_dt.timestamp() * 1000)
            
            # 🥇 방법 1: userTrades API (가장 정확)
            try:
                start_time = trade_timestamp - 3600000  # 1시간 전
                end_time = trade_timestamp + 3600000    # 1시간 후
                
                params = {
                    'symbol': binance_symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 100
                }
                
                user_trades = exchange.fapiprivate_get_usertrades(params)
                
                # reduceOnly 거래 찾기
                total_pnl = 0
                total_commission = 0
                found_trades = 0
                
                for t in user_trades:
                    if t.get('reduceOnly') or t.get('positionSide') == 'BOTH':
                        realized_pnl = float(t.get('realizedPnl', 0))
                        if realized_pnl != 0:
                            total_pnl += realized_pnl
                            total_commission += abs(float(t.get('commission', 0)))
                            found_trades += 1
                
                if found_trades > 0:
                    new_pnl = total_pnl
                    new_pnl_pct = (new_pnl / position_size) * 100
                    
                    print(f"      ✅ userTrades 발견: {found_trades}건")
                    print(f"      💰 실제 PnL: ${new_pnl:+.2f} ({new_pnl_pct:+.2f}%)")
                    print(f"      📊 변동: ${new_pnl - old_pnl:+.2f}")
                    
                    # DB 업데이트
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute('''
                        UPDATE trades 
                        SET pnl = ?, pnl_percent = ?
                        WHERE id = ?
                    ''', (new_pnl, new_pnl_pct, trade_id))
                    conn.commit()
                    conn.close()
                    
                    updated_count += 1
                    continue
                    
            except Exception as e:
                print(f"      ⚠️ userTrades 실패: {e}")
            
            # 🥈 방법 2: Income History API
            try:
                params = {
                    'symbol': binance_symbol,
                    'incomeType': 'REALIZED_PNL',
                    'startTime': trade_timestamp - 3600000,
                    'endTime': trade_timestamp + 3600000,
                    'limit': 50
                }
                
                income_history = exchange.fapiprivate_get_income(params)
                
                total_pnl = 0
                found_income = 0
                
                for income in income_history:
                    if income['symbol'] == binance_symbol and income['incomeType'] == 'REALIZED_PNL':
                        total_pnl += float(income['income'])
                        found_income += 1
                
                if found_income > 0:
                    new_pnl = total_pnl
                    new_pnl_pct = (new_pnl / position_size) * 100
                    
                    print(f"      ✅ Income API 발견: {found_income}건")
                    print(f"      💰 실제 PnL: ${new_pnl:+.2f} ({new_pnl_pct:+.2f}%)")
                    print(f"      📊 변동: ${new_pnl - old_pnl:+.2f}")
                    
                    # DB 업데이트
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute('''
                        UPDATE trades 
                        SET pnl = ?, pnl_percent = ?
                        WHERE id = ?
                    ''', (new_pnl, new_pnl_pct, trade_id))
                    conn.commit()
                    conn.close()
                    
                    updated_count += 1
                    continue
                    
            except Exception as e:
                print(f"      ⚠️ Income API 실패: {e}")
            
            # 🥉 방법 3: 가격 기반 재계산
            if exit_price and exit_price > 0:
                print(f"      📊 가격 기반 재계산 사용")
                
                new_pnl, new_pnl_pct = calculate_pnl_from_price(
                    entry_price, exit_price, position_size, side, leverage
                )
                
                print(f"      💰 재계산 PnL: ${new_pnl:+.2f} ({new_pnl_pct:+.2f}%)")
                print(f"      📊 변동: ${new_pnl - old_pnl:+.2f}")
                
                # 변동이 크면 업데이트
                if abs(new_pnl - old_pnl) > 0.5:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute('''
                        UPDATE trades 
                        SET pnl = ?, pnl_percent = ?
                        WHERE id = ?
                    ''', (new_pnl, new_pnl_pct, trade_id))
                    conn.commit()
                    conn.close()
                    
                    updated_count += 1
                else:
                    print(f"      ✅ 변동 미미 - 업데이트 불필요")
            else:
                print(f"      ❌ 청산가 없음 - 건너뜀")
                
        except Exception as e:
            print(f"      ❌ PnL 정정 실패: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # 빈 줄
    
    print(f"{'='*80}")
    print(f"   📊 정정 완료: {updated_count}/{len(closed_trades)}개 거래 업데이트됨")
    print(f"   ✅ 이제 대시보드와 바이낸스 PnL이 일치합니다!")
    print(f"{'='*80}\n")
    
    return updated_count

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              🔧 거래 PnL 정정 스크립트                        ║
║              바이낸스 실제 데이터로 DB 업데이트               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("이 스크립트는 기존 청산 거래의 PnL을 바이낸스 API로 다시 조회하여")
    print("데이터베이스를 업데이트합니다.\n")
    
    confirm = input("계속하시겠습니까? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        try:
            updated = update_closed_trades_pnl()
            print(f"\n✅ 완료! {updated}개 거래가 업데이트되었습니다.")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n취소되었습니다.")
