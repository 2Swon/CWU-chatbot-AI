@echo off
echo RAGAS 평가 시스템 오류 수정 및 재설치
echo =====================================
echo.

echo 1단계: pip 패키지 재설치 중...
pip uninstall -y ragas
pip install -r requirements.txt
echo.

echo 2단계: 설치된 RAGAS 버전 확인 중...
pip show ragas
echo.

echo 설치 완료! 이제 애플리케이션을 실행할 수 있습니다.
echo 실행 방법: python main.py
echo.

pause
