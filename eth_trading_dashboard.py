#!/usr/bin/env python3
"""
Ethereum Day Trading Bot - Streamlit Dashboard 실행 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """필요한 의존성 확인 및 설치"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"필요한 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        install = input("자동으로 설치하시겠습니까? (y/n): ")
        
        if install.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("필요한 패키지를 먼저 설치해주세요:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def setup_streamlit_config():
    """Streamlit 설정 디렉토리 생성"""
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
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Streamlit 설정 파일이 생성되었습니다: {config_file}")

def check_database():
    """데이터베이스 파일 존재 확인"""
    db_file = "ethereum_daytrading.db"
    
    if not os.path.exists(db_file):
        print(f"⚠️  데이터베이스 파일을 찾을 수 없습니다: {db_file}")
        print("먼저 거래 봇을 실행하여 데이터베이스를 생성해주세요.")
        return False
    
    print(f"✅ 데이터베이스 파일을 찾았습니다: {db_file}")
    return True

def main():
    print("=" * 60)
    print("🚀 Ethereum Day Trading Bot - Streamlit Dashboard")
    print("=" * 60)
    
    # 의존성 확인
    print("📦 필요한 패키지 확인 중...")
    if not check_dependencies():
        return
    
    # Streamlit 설정
    print("⚙️  Streamlit 설정 중...")
    setup_streamlit_config()
    
    # 데이터베이스 확인
    print("🗄️  데이터베이스 확인 중...")
    if not check_database():
        create_anyway = input("데이터베이스가 없어도 대시보드를 실행하시겠습니까? (y/n): ")
        if create_anyway.lower() != 'y':
            return
    
    # Streamlit 앱 실행
    print("\n🌟 Streamlit 대시보드를 시작합니다...")
    print("📱 브라우저에서 http://localhost:8501 를 열어주세요")
    print("🛑 종료하려면 Ctrl+C를 눌러주세요")
    print("-" * 60)
    
    try:
        # 대시보드 파일이 있는지 확인
        dashboard_file = "eth_trading_dashboard.py"
        if not os.path.exists(dashboard_file):
            print(f"❌ 대시보드 파일을 찾을 수 없습니다: {dashboard_file}")
            print("eth_trading_dashboard.py 파일을 같은 디렉토리에 저장해주세요.")
            return
        
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_file,
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 대시보드를 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {e}")
        print("문제가 지속되면 수동으로 실행해보세요:")
        print(f"streamlit run {dashboard_file}")

if __name__ == "__main__":
    main()