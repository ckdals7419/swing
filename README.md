# swing — KCM 백테스트 & 매수·매도 조건 체크

이 저장소는 **KCM(regime + RSI dip)** 전략을 백테스트로 검증하고, 오늘 종가 기준으로 **매수/매도/홀드 조건이 맞는지** 확인하는 도구 모음입니다.

실전 신호는 QQQ→TQQQ, SMH→USD(2x semis)처럼 **신호 종목과 실행 종목을 분리**합니다. 신호는 **T일 종가**, 체결은 기본적으로 **T+1 시가**입니다.

## KCM 규칙 (현행 기준)

| 항목 | 내용 |
|------|------|
| 레짐 ON | 가격이 SMA200 **+1%** 이상이거나 SMA50 이상 |
| 레짐 OFF(청산) | 가격이 SMA200 **-1%** 미만 **그리고** SMA50 미만 |
| 임시보유 (레짐 OFF일 때) | **단순평균 RSI(14) &lt; 26** 또는 **VIX&gt;35 이면서 10일 평균보다 하락** |
| 포지션 | 기본 **0% 또는 100%** |
| VT25 | **옵션**. `min(1, 0.25 / 20일 연환산 변동성)` 로 사이즈만 줄임. 기본 신호에는 넣지 않음 |

RSI는 Wilder가 아니라 **단순이동평균(Cutler) RSI**를 씁니다. 구현·상수: `notify_kcm_telegram.py` (`RSI_DIP=26`, `rsi_sma`, `kcm_weights`, `vt_scale`).

## 자주 쓰는 명령

```bash
# 가상환경 (최초 1회)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 오늘 신호 확인 (텔레그램 없이 출력만)
python notify_kcm_telegram.py --dry-run

# VT25 옵션까지 반영해서 보기
python notify_kcm_telegram.py --dry-run --vt25

# 변형 백테스트 (비용 9bps, 합성 TQQQ, 세후 가정 등)
python backtest_kcm_variants.py

# 레짐 OFF 규칙 민감도
python review_off_regime.py

# 급락 유사 구간 비교
python analog_regime_break.py
```

텔레그램 전송은 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 환경변수가 필요합니다. GitHub Actions 워크플로: `.github/workflows/kcm_daily_telegram.yml`.

## Git 사용법

이 프로젝트의 변경(신호 로직, 백테스트 스크립트, 차트, 연도별 MDD 표 등)은 git으로 버전 관리합니다.

```bash
# 상태 보기
git status
git diff

# 반영할 파일만 스테이징 후 커밋
git add -A
git commit -m "$(cat <<'EOF'
짧은 요약: 왜 바꿨는지.

EOF
)"

# 원격에 올리기 (필요할 때만)
git push origin main

# 최근 기록
git log --oneline -10
```

주의:

- `.env`, API 키, `.venv/`, `data/`, 영상(`*.mp4`)은 `.gitignore`로 제외합니다.
- **커밋은 요청이 있을 때만** 하는 것을 원칙으로 합니다.
- `main`에 force push / `--no-verify` 는 쓰지 않습니다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `notify_kcm_telegram.py` | **기준 KCM** + 일일 신호 / 텔레그램 |
| `backtest_kcm_variants.py` | KCM·변형·MOC·VT25(옵션) 백테스트 |
| `review_off_regime.py` | 레짐 OFF 규칙 스윕 |
| `analog_regime_break.py` | 레짐 붕괴+RSI 딥 유사 사례 |
| `kcm_yearly_mdd.csv` | 연도별 성과·MDD 요약 |
| `chart_*.png` | 구간별 시각화 |

그 외 `strategy_qld.py`, `crossval_qld.py`, ML 파이프라인 등은 QLD/실험용 확장입니다. 실전 KCM 기준 로직은 위 표의 파일을 우선하세요.
