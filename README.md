# swing — KCM 백테스트 & 매수·매도 조건 체크

이 저장소는 **KCM(regime + RSI dip)** 전략을 백테스트로 검증하고, 오늘 종가 기준으로 **매수/매도/홀드 조건이 맞는지** 확인하는 도구 모음입니다.

실전 신호는 QQQ→TQQQ·QLD, SMH→USD(2x semis)처럼 **신호 종목과 실행 종목을 분리**합니다.

## KCM 규칙 (현행 기준)

| 항목 | 내용 |
|------|------|
| 레짐 ON | 가격이 SMA200 **+1%** 이상이거나 SMA50 이상 |
| 레짐 OFF(청산) | 가격이 SMA200 **-1%** 미만 **그리고** SMA50 미만 |
| 임시보유 (레짐 OFF일 때) | **단순평균 RSI(14) &lt; 26** 또는 **VIX&gt;35 이면서 10일 평균보다 하락** |
| 포지션 | 기본 **0% 또는 100%** |
| VT25 | **옵션**. `min(1, 0.25 / 20일 연환산 변동성)` 로 사이즈만 줄임 |

RSI는 Wilder가 아니라 **단순이동평균(Cutler) RSI**입니다.  
구현: `notify_kcm_telegram.py` (`RSI_DIP=26`, `rsi_sma`, `kcm_weights`, `vt_scale`).  
체결 규칙: `kcm_fill_rules.py`.

## 매수·매도 체결 원칙 (트리거별)

신호는 항상 **신호 자산(QQQ 등) 종가**로 판정합니다.  
**체결 타이밍만** 트리거 종류에 따라 다르게 갑니다.

| 트리거 | 구분 | 체결 방법 | 이유 |
|--------|------|-----------|------|
| **SMA200 +1% 진입** | 매수 | **장중 터치** 지정가 | 가격 레벨이 장중 관측 가능, 시가·LOC보다 유리 |
| **SMA50 진입** | 매수 | **장중 터치** 지정가 | 동일 |
| **SMA200 탈출** (−1% & SMA50↓) | 매도 | **장중 터치** 지정가 | 동일 (SMA50 단독 탈출 없음) |
| **SMA-RSI &lt; 26 진입** | 매수 | **종가 LOC** | RSI는 종가 확정 → 장중 추측 금지 |
| **SMA-RSI 탈출** (≥26) | 매도 | **종가 LOC** | 동일 |
| **VIX&gt;35 & 하락 진입** | 매수 | **종가 LOC** | VIX 조건도 종가 기준 |
| **VIX 탈출** | 매도 | **종가 LOC** | 동일 |

실무 메모:

- SMA 재진입 장중 지정가 예: QQQ가 SMA200+1%일 때 `QLD/TQQQ ≈ 현재가 × (1 + 배수 × QQQ상승률)`.
- RSI/VIX는 LOC가 어려우면 **다음날 시가**로 대체 가능 (성과 차이 작음).
- VT25 사용 시 신호가 켜져도 비중은 `VT scale`만큼만 (진입일 스케일 고정 권장).

구현 상수: `kcm_fill_rules.FILL_RULES`.

## 체결 원칙 백테스트 (2000+)

`python backtest_kcm_fills.py` — 비용 9bps, 합성 3x→실물 TQQQ.

### VT 없음

| 방식 | 2000+ CAGR / MDD | 2010+ CAGR / MDD |
|------|------------------|------------------|
| **hybrid(원칙)** | **+53.4% / -67.0%** | **+60.2% / -52.6%** |
| 전부 시가 | +38.7% / -75.9% | +48.5% / -57.2% |
| 전부 LOC | +37.0% / -78.4% | +49.1% / -55.0% |
| 전부 장중 | +26.3% / -94.5% | +40.7% / -59.9% |

### VT25 (진입일 스케일 고정)

| 방식 | 2000+ CAGR / MDD | 2010+ CAGR / MDD |
|------|------------------|------------------|
| **hybrid(원칙)** | **+35.3% / -56.7%** | **+38.9% / -56.7%** |
| 전부 시가 | +27.8% / -58.8% | +30.6% / -58.8% |
| 전부 LOC | +26.1% / -66.6% | +30.6% / -59.0% |
| 전부 장중 | +18.4% / -72.1% | +25.7% / -54.4% |

→ **트리거별 체결(hybrid)이 단일 방식보다 우위**.

## 자주 쓰는 명령

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 오늘 신호 (텔레그램 없이)
python notify_kcm_telegram.py --dry-run
python notify_kcm_telegram.py --dry-run --vt25

# 트리거별 체결 원칙 백테스트
python backtest_kcm_fills.py

# 기타
python backtest_kcm_variants.py
python review_off_regime.py
python analog_regime_break.py
```

텔레그램: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.  
워크플로: `.github/workflows/kcm_daily_telegram.yml`.

## Git 사용법

```bash
git status
git diff
git add -A
git commit -m "$(cat <<'EOF'
짧은 요약: 왜 바꿨는지.

EOF
)"
git push origin main
git log --oneline -10
```

- `.env`, `.venv/`, `data/`, `*.mp4` 는 `.gitignore`
- 커밋은 요청 시에만, force push / `--no-verify` 금지

## 주요 파일

| 파일 | 역할 |
|------|------|
| `notify_kcm_telegram.py` | 기준 KCM + 일일 신호 / 텔레그램 |
| `kcm_fill_rules.py` | **트리거별 체결 규칙** + hybrid 엔진 |
| `backtest_kcm_fills.py` | hybrid vs 단일 체결 방식 백테스트 |
| `backtest_kcm_variants.py` | KCM 변형·MOC·VT 옵션 |
| `review_off_regime.py` | 레짐 OFF 규칙 스윕 |
| `analog_regime_break.py` | 레짐 붕괴 유사 사례 |

그 외 `strategy_qld.py`, ML 파이프라인 등은 실험용입니다. 실전 KCM은 위 표를 우선하세요.
