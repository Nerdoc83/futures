import sqlite3
import os

# 봇과 대시보드가 사용하는 DB 파일명과 동일해야 합니다.
DB_FILE = "multi_coin_daytrading.db"

def check_db_contents():
    """
    지정된 SQLite 데이터베이스 파일의 내용을 검사하고 각 테이블의 레코드 수를 출력합니다.
    """
    if not os.path.exists(DB_FILE):
        print(f"오류: '{DB_FILE}' 파일을 현재 폴더에서 찾을 수 없습니다.")
        print("이 스크립트를 봇, 대시보드와 동일한 폴더에 놓고 실행해주세요.")
        return

    print(f"'{DB_FILE}' 파일 분석을 시작합니다...\n")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 데이터베이스의 모든 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("데이터베이스에 테이블이 존재하지 않습니다.")
            conn.close()
            return
            
        print(f"발견된 테이블: {[table[0] for table in tables]}\n")

        # 'trades' 테이블 레코드 수 확인
        print("--- 'trades' 테이블 검사 ---")
        try:
            cursor.execute("SELECT COUNT(*) FROM trades;")
            trade_count = cursor.fetchone()[0]
            print(f"✅ 총 거래 기록 수: {trade_count}")
            if trade_count > 0:
                print("-> 최근 3개 거래 기록:")
                cursor.execute("PRAGMA table_info(trades);")
                col_names = [info[1] for info in cursor.fetchall()]
                print(f"   컬럼: {col_names}")
                cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 3;")
                for row in cursor.fetchall():
                    print(f"   -> {row}")
        except sqlite3.OperationalError:
            print("❗️ 'trades' 테이블을 찾을 수 없거나 테이블 구조에 문제가 있습니다.")
        
        # 'ai_analysis' 테이블 레코드 수 확인
        print("\n--- 'ai_analysis' 테이블 검사 ---")
        try:
            cursor.execute("SELECT COUNT(*) FROM ai_analysis;")
            ai_count = cursor.fetchone()[0]
            print(f"✅ 총 AI 분석 기록 수: {ai_count}")
            if ai_count > 0:
                 print("-> 최근 3개 AI 분석 기록:")
                 cursor.execute("SELECT * FROM ai_analysis ORDER BY timestamp DESC LIMIT 3;")
                 for row in cursor.fetchall():
                    print(f"   -> {row}")
        except sqlite3.OperationalError:
            print("❗️ 'ai_analysis' 테이블을 찾을 수 없거나 테이블 구조에 문제가 있습니다.")

        conn.close()

    except Exception as e:
        print(f"데이터베이스 분석 중 예상치 못한 오류 발생: {e}")

if __name__ == "__main__":
    check_db_contents()
