#!/bin/bash
# AI Trading Bot - Background Runner

LOG_FILE="trading_bot.log"
PID_FILE="trading_bot.pid"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "❌ 봇이 이미 실행 중입니다 (PID: $PID)"
                exit 1
            fi
        fi
        
        echo "🚀 거래봇 시작 중..."
        nohup python3 aul_fixed.py > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "✅ 거래봇 시작 완료 (PID: $(cat $PID_FILE))"
        echo "📄 로그: tail -f $LOG_FILE"
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "❌ 실행 중인 봇이 없습니다"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⏹️  거래봇 종료 중 (PID: $PID)..."
            kill "$PID"
            sleep 2
            
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "⚠️  강제 종료..."
                kill -9 "$PID"
            fi
            
            rm -f "$PID_FILE"
            echo "✅ 거래봇 종료 완료"
        else
            echo "❌ 프로세스가 이미 종료되었습니다"
            rm -f "$PID_FILE"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "✅ 거래봇 실행 중 (PID: $PID)"
                echo ""
                ps -p "$PID" -o pid,etime,cmd
            else
                echo "❌ 거래봇 종료됨 (PID 파일만 존재)"
                rm -f "$PID_FILE"
            fi
        else
            echo "❌ 실행 중인 봇이 없습니다"
        fi
        ;;
        
    log)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ 로그 파일이 없습니다"
        fi
        ;;
        
    *)
        echo "사용법: $0 {start|stop|restart|status|log}"
        echo ""
        echo "  start   - 거래봇 시작"
        echo "  stop    - 거래봇 종료"
        echo "  restart - 거래봇 재시작"
        echo "  status  - 상태 확인"
        echo "  log     - 실시간 로그 보기"
        exit 1
        ;;
esac
