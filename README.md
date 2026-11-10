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
| **과열 트림 (기본)** | 종가가 SMA200 **+15% 초과 → 50%**, 이후 **+10% 미만 → 100% 복귀** (다음날 시가) |
| VT25 | **옵션**. `min(1, 0.25 / 20일 연환산 변동성)` 로 사이즈만 줄임 |

RSI는 Wilder가 아니라 **단순이동평균(Cutler) RSI**입니다.  
구현: `notify_kcm_telegram.py` (`RSI_DIP=26`, `rsi_sma`, `kcm_weights`, `trim_scale`, `vt_scale`).
트림 끄기: `kcm_weights(..., trim=False)`, `simulate(..., trim=None)`, `--no-trim`.  
실전 체결·세금 시뮬레이션: `kcm_sim.py`.

## 체결 방법

신호는 항상 **신호 자산(QQQ 등) 종가**로 판정. 체결은 카테고리별로 고름 (`kcm_sim.simulate(sma_fill=, temp_fill=)`).

| 카테고리 | 권장 | 비고 |
|----------|------|------|
| **SMA200·SMA50 레짐** 진입/청산 | **다음날 시가** (LOC도 거의 동일) | 장중 감시주문은 휩쏘로 거래 252→400회, 성과 최하 |
| **RSI·VIX 임시보유** 진입/청산 | LOC 또는 다음날 시가 | 차이 ±1%p 이내 |

> 구 hybrid 결과(+53.4% / -67.0%)는 무효. 종가로 확정된 날만 장중 터치가로 체결해
> look-ahead가 있었고, 구간 시작 시 현금으로 출발하는 버그도 있었음.

## 체결 방법 백테스트 (실전 체결 + 세금)

`python backtest_kcm_fills.py` — 비용 9bps, 합성 3x→실물 TQQQ,
**세금 기본 반영**: 해외주식 양도세 22%, 연 250만원 공제, 같은 해 손익 상계, 원금 1억 가정.
장중 = 전일 SMA 레벨 감시주문, 터치 시 무조건 체결.

과열 트림 기본 적용.

### VT 없음 (세후 CAGR / MDD)

| SMA | RSI·VIX | 2000+ | 2010+ | 거래 |
|-----|---------|-------|-------|------|
| 장중 | LOC | +23.9% / -84.9% | +41.9% / -59.2% | 441 |
| 장중 | 시가 | +24.6% / -86.8% | +41.2% / -59.2% | 441 |
| LOC | LOC | +30.0% / -77.2% | +43.3% / -53.1% | 293 |
| LOC | 시가 | +30.7% / -71.0% | +42.6% / -53.6% | 293 |
| **시가** | **LOC** | +30.7% / -73.6% | **+43.5% / -53.1%** | 293 |
| **시가** | **시가** | **+31.4% / -75.9%** | +42.7% / -53.6% | 293 |

### VT25 (진입일 스케일 고정, 세후 CAGR / MDD)

| SMA | RSI·VIX | 2000+ | 2010+ | 거래 |
|-----|---------|-------|-------|------|
| 장중 | LOC | +21.2% / -66.3% | +31.4% / -57.1% | 441 |
| 장중 | 시가 | +21.6% / -64.8% | +30.9% / -56.4% | 441 |
| LOC | LOC | +25.5% / -65.8% | +33.9% / -46.2% | 293 |
| LOC | 시가 | +25.8% / -57.7% | +33.2% / -46.6% | 293 |
| **시가** | **LOC** | +26.8% / -61.1% | **+34.6% / -43.4%** | 293 |
| **시가** | **시가** | **+27.0% / -52.0%** | +33.9% / -45.3% | 293 |

## 과열 트림 (SMA200 위 과열 시 비중 축소) 테스트

`python backtest_kcm_trim.py` — 종가 판정, 다음날 시가 리밸런스(목표 10%p 이상 변할 때), 세후.
IS 2000-2012에서 고르고 OOS 2013+로 검증.

| 규칙 | VT | IS00-12 | OOS13+ | 2000+ |
|------|----|---------|--------|-------|
| 기준 | 없음 | +9.4% / -87.0% | +47.4% / -56.9% | +27.7% / -87.0% |
| **vs200&gt;15% → 50%** (10% 밑이면 복귀) | 없음 | +14.4% / -73.6% | +48.0% / -53.1% | +30.7% / -73.6% |
| 기준 | VT25 | +11.0% / -78.9% | +37.3% / -49.7% | +23.9% / -78.9% |
| **vs200&gt;15% → 70%** | VT25 | +14.9% / -62.0% | +39.1% / -45.5% | +26.8% / -62.0% |

- vs200(SMA200 이격) 계열은 임계 15~30% 전반에서 OOS 수익·MDD가 함께 소폭 개선 → 비교적 견고.
- RSI 과열 트림은 거래만 늘고 OOS 수익 악화. 1년 수익률 트림은 IS 1위지만 OOS 악화(과적합).
- 2000+ MDD 개선의 대부분은 닷컴 한 번에서 나옴.
- 검증: 독립 구현(단위·평단·세금 별도 계산)과 총수익·MDD 완전 일치. 신호=종가, 체결=다음날 시가.
- **기본값으로 채택** (VT 없음 기준 50%). VT25와 함께 쓸 때는 70%가 약간 나음(`trim_scale(close, w=0.7)`).

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
| `kcm_sim.py` | **실전 체결(장중 감시/LOC/다음날 시가) + 양도세** 시뮬레이터 |
| `kcm_fill_rules.py` | 트리거 분류 + 구 hybrid 엔진(look-ahead, 참고용) |
| `backtest_kcm_fills.py` | 체결 방식 비교 (세후, VT 없음/VT25) |
| `backtest_kcm_trim.py` | 과열 트림 규칙 비교 (IS/OOS) |
| `backtest_kcm_variants.py` | KCM 변형·MOC·VT 옵션 |
| `review_off_regime.py` | 레짐 OFF 규칙 스윕 |
| `analog_regime_break.py` | 레짐 붕괴 유사 사례 |

그 외 `strategy_qld.py`, ML 파이프라인 등은 실험용입니다. 실전 KCM은 위 표를 우선하세요.
