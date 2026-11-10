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

`python backtest_kcm_fills.py` — 비용 9bps, 합성 3x(`lev_synth` TQQQ 프리셋)→실물 TQQQ,
**세금 기본 반영**: 해외주식 양도세 22%, 연 250만원 공제, 같은 해 손익 상계, 원금 1억 가정.
**현금**: 외화RP 연 **3.45%** 일할 (`cash_rate`, 이자소득세 미반영).
장중 = 전일 SMA 레벨 감시주문, 터치 시 무조건 체결.
과열 트림 기본 적용.

### VT 없음 (세후 CAGR / MDD)

| SMA | RSI·VIX | 2000+ | 2010+ | 거래 |
|-----|---------|-------|-------|------|
| 장중 | LOC | +23.4% / -86.1% | +42.8% / -59.0% | 441 |
| 장중 | 시가 | +24.1% / -86.8% | +42.1% / -58.6% | 441 |
| LOC | LOC | +29.9% / -74.7% | +44.2% / -51.9% | 293 |
| LOC | 시가 | +30.7% / -70.6% | +43.5% / -52.5% | 293 |
| **시가** | **LOC** | +30.7% / -70.7% | **+44.4% / -52.0%** | 293 |
| **시가** | **시가** | **+31.4% / -75.7%** | +43.7% / -52.5% | 293 |

### VT25 (진입일 스케일 고정, 세후 CAGR / MDD)

| SMA | RSI·VIX | 2000+ | 2010+ | 거래 |
|-----|---------|-------|-------|------|
| 장중 | LOC | +21.5% / -64.3% | +32.7% / -56.9% | 441 |
| 장중 | 시가 | +21.9% / -63.9% | +32.2% / -56.2% | 441 |
| LOC | LOC | +26.0% / -63.4% | +35.3% / -45.7% | 293 |
| LOC | 시가 | +26.3% / -54.7% | +34.5% / -46.1% | 293 |
| **시가** | **LOC** | +27.3% / -58.3% | **+36.0% / -42.9%** | 293 |
| **시가** | **시가** | **+27.5% / -48.5%** | +35.2% / -44.7% | 293 |

## 과열 트림 (SMA200 위 과열 시 비중 축소) 테스트

`python backtest_kcm_trim.py` — 종가 판정, 다음날 시가 리밸런스(목표 10%p 이상 변할 때), 세후.
IS 2000-2012에서 고르고 OOS 2013+로 검증.

| 규칙 | VT | IS00-12 | OOS13+ | 2000+ |
|------|----|---------|--------|-------|
| 기준(트림없음) | 없음 | +8.3% / -87.4% | +48.0% / -56.8% | +27.3% / -87.4% |
| **vs200&gt;15% → 50%** (10% 밑이면 복귀) | 없음 | +13.6% / -70.7% | +48.9% / -52.0% | +30.7% / -70.7% |
| 기준(트림없음) | VT25 | +10.5% / -78.3% | +38.3% / -49.7% | +24.1% / -78.3% |
| **vs200&gt;15% → 50%** | VT25 | +16.2% / -58.3% | +38.4% / -42.9% | +27.3% / -58.3% |

현금 RP 3.45%를 넣으면 트림 우위가 더 커짐 (현금 구간·50% 구간에서 이자 가산).

### 트림 깊이: →50% vs →0%(전량) (RP 3.45% 포함, VT 없음)

| 규칙 | 2000+ | OOS13+ | 닷컴 | 08위기 | 코로나20 |
|------|-------|--------|------|--------|----------|
| 트림 없음 | +27.3% / -87.4% | +48.0% / -56.8% | -25.0% / -87.4% | +26.9% / -72.5% | +188% / -56.8% |
| **→50% (채택)** | **+30.7% / -70.7%** | **+48.9% / -52.0%** | -0.5% / -68.5% | +18.4% / -70.0% | +123% / -50.5% |
| →0% (전량) | +30.5% / -71.6% | +48.5% / -50.2% | +19.1% / -70.6% | +2.6% / -70.3% | +43% / -42.8% |

전량은 닷컴에서만 우위. 08·코로나 반등을 놓쳐 장기 CAGR은 50%와 비슷하거나 약간 열위 → **기본 유지: →50%**.

- vs200(SMA200 이격) 계열은 임계 15~30% 전반에서 OOS 수익·MDD가 함께 소폭 개선 → 비교적 견고.
- RSI 과열 트림은 거래만 늘고 OOS 수익 악화. 1년 수익률 트림은 IS 1위지만 OOS 악화(과적합).
- 2000+ MDD 개선의 대부분은 닷컴 한 번에서 나옴.
- 검증: 독립 구현(단위·평단·세금 별도 계산)과 총수익·MDD 완전 일치. 신호=종가, 체결=다음날 시가.
- **기본값으로 채택** (VT 없음 기준 50%).

### 실험 결과 저장 위치

| 파일 | 내용 |
|------|------|
| `README.md` | 요약 표 (채택 규칙·핵심 숫자) |
| `results_kcm_fills.csv` | 체결 방식 6조합 × 구간 (`backtest_kcm_fills.py` 생성) |
| `results_kcm_trim.csv` | 과열 트림 규칙 전체 IS/OOS (`backtest_kcm_trim.py` 생성) |
| `backtest_nav*.csv`, `backtest_xval_*.csv` | 예전 QLD/교차검증 NAV (참고용, 구엔진) |

## 3배 지수 합성 (`lev_synth.py`)

실물 레버리지 ETF 상장 전 구간을 만드는 범용 모듈. S&P500(UPRO), SMH, 해외 지수에도 같은 방식 사용.

일별 모델 (종가→종가, 달력일 기준 금리 누적):

\[
r_{ETF} = L \cdot r_{TR} - \left[ER + (L-1)(r_f + s)\right]\frac{days}{365} + c_2\, r_{TR}^2
\]

| 항목 | 내용 |
|------|------|
| \(r_{TR}\) | **총수익(배당 포함) 지수**: 추종 ETF 수정종가 + 그 ETF 보수 (예: QQQ + 0.20%). ETF 상장 전은 가격지수 + 가정 배당수익률 |
| \(r_f\) | ^IRX(13주 T-bill), 전일 값 |
| ER | 레버리지 ETF 보수 (TQQQ 0.84%) |
| \(s\), \(c_2\) | 실물 ETF 구간에 맞춘 스왑 스프레드·변동성 항. \(c_2\)는 일별 회귀, \(s\)는 **누적수익 일치**로 보정 |
| 시가 | 전일 합성 종가 × (1 + L × 지수 갭) |

TQQQ 보정 결과 (`python lev_synth.py`, 학습 2010-02~2018-06 / 검증 2018-06~):

| 모델 | 학습 오차/년 | 검증 오차/년 | 검증 최대괴리 |
|------|-------------|-------------|---------------|
| 가격지수 − 1bp/일 (구) | -3.46% | +3.36% | 27.7% |
| 가격지수 + ER + 2×금리 | -2.38% | -0.45% | 5.5% |
| TR + ER + 2×금리 | +1.19% | +2.40% | 19.9% |
| TR + 스프레드 + r² (학습만 보정) | 0.00% | +1.11% | 9.3% |
| **TR + 스프레드 + r² (전체 보정, 채택)** | **-0.21%** | **+0.21%** | **5.4%** |

- 가격지수는 배당(연 ~1%×3)을 빠뜨려 스프레드가 음수로 왜곡됨 → **TR 지수 필수**.
- 일별 OLS는 장기 드리프트를 못 맞춤 → 스프레드는 누적수익 일치로 보정.
- 채택 프리셋: `PRESETS["TQQQ"]` (s=0.335%, c₂=-0.253). 구합성 대비 고금리기 수익이 낮아짐(2000-02 연 -11.7%, 1990s 연 -8.6%).
- KCM 영향: 2000+ CAGR 31.9% → 30.7% (현금 비중이 높은 시기라 영향 작음). 합성 변형 간 차이 0.4%p 이내.

새 지수 추가 방법:

```python
import lev_synth as L
tri, rf = L.load_tr_index("^GSPC", "SPY", 0.0009, pre_div_yield=0.02)
real = L._dl("UPRO")["Close"]
p = L.calibrate_drift(tri, real, rf, lev=3, er=0.0091)   # 실물 구간 전체로 보정
L.PRESETS["UPRO"] = {"index": "^GSPC", "tr_etf": "SPY", "tr_er": 0.0009, "pre_div": 0.02,
                     "lev": 3, **p, "real": "UPRO"}
```

`python lev_synth.py` 의 `_report(name)` 로 학습/검증 오차를 반드시 확인.

## 파라미터 민감도

`python backtest_kcm_sensitivity.py` → `results_kcm_sensitivity.csv` + `sensitivity_heatmaps/*.png`.
기본 체결·세금·RP·트림 유지, 축만 변경. ★ = 현행 기본값.

| 그리드 | 2000+ 인접 CAGR σ | 기본 주변(대략) | 판정 |
|--------|-------------------|-----------------|------|
| RSI × VIX | **2.2%p** | +27.3% ~ +30.9% (3×3) | 평탄. RSI≤20만 급락 |
| trim ON × weight | **0.9%p** | +29.2% ~ +31.0% | 매우 평탄. 트림 없음(+27.3%)보다 우위 |
| trim ON × gap | 1.3%p | 기본 15%/5%p 근처 유지 | 히스테리시스 폭에 덜 민감 |
| SMA ±buffer | — | **±1%가 2000+ 최고** | 0%·2%는 수 %p 열위 |

- 2013+만 보면 RSI30/VIX25가 +55.5%로 보이지만 2000+에서는 기본(26/35)이 더 나음 → 최근 구간 과적합 위험.
- 결론: **현행 파라미터를 유지해도 됨.** 미세 튜닝 이득보다 안정성이 큼.

## 교차 자산 검증 (TQQQ / SOXL / UPRO)

`python backtest_kcm_cross.py` — KCM 기본 규칙 동일, 신호·실행만 바꿈.
세후·RP 3.45%·트림 50%·SMA 시가·RSI LOC. VT 없음.

| 구간 | NDX→TQQQ | SOXX→SOXL | SPX→UPRO |
|------|----------|-----------|----------|
| 2000+ | **+30.7% / -70.7%** | +13.4% / -95.4% | +15.9% / -88.0% |
| 2010+ | **+42.6% / -52.0%** | +34.5% / -83.6% | +29.8% / -55.3% |
| 2013+ | **+48.9% / -52.0%** | +44.3% / -83.6% | +31.9% / -55.3% |
| 08위기 | **+18.4% / -70.0%** | -19.3% / -85.6% | -24.5% / -88.0% |
| 코로나20 | +122.7% / -50.5% | +128.0% / -55.2% | **+175.1% / -48.4%** |
| 2022 | -45.0% / -46.8% | -68.7% / -68.2% | **-41.7% / -41.3%** |

- S&P500 3배: **UPRO**(ProShares) / SPXL(Direxion). 프리셋은 UPRO (`lev_synth.PRESETS["UPRO"]`, TE≈3%/yr).
- UPRO는 TQQQ보다 CAGR이 낮고, 08위기를 못 피함. 그래도 SOXL보다는 MDD·장기 수익이 나음.
- 같은 규칙의 **나스닥에서 가장 잘 작동**. 결과: `results_kcm_cross.csv`.

## 자주 쓰는 명령

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 오늘 신호 (텔레그램 없이)
python notify_kcm_telegram.py --dry-run
python notify_kcm_telegram.py --dry-run --vt25

# 트리거별 체결 원칙 백테스트 → results_kcm_fills.csv
python backtest_kcm_fills.py

# 과열 트림 비교 → results_kcm_trim.csv
python backtest_kcm_trim.py
python backtest_kcm_cross.py  # TQQQ vs SOXL vs UPRO
python backtest_kcm_sensitivity.py

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
| `backtest_kcm_fills.py` | 체결 방식 비교 → `results_kcm_fills.csv` |
| `backtest_kcm_trim.py` | 과열 트림 비교 → `results_kcm_trim.csv` |
| `lev_synth.py` | **레버리지 ETF 합성** (TR + 금리 + 스프레드, 프리셋·보정·검증) |
| `backtest_kcm_cross.py` | TQQQ/SOXL/UPRO 교차 → `results_kcm_cross.csv` |
| `backtest_kcm_sensitivity.py` | 파라미터 히트맵 → `results_kcm_sensitivity.csv` |
| `results_kcm_fills.csv` | 체결 실험 결과 (재생: `python backtest_kcm_fills.py`) |
| `results_kcm_trim.csv` | 트림 실험 결과 (재생: `python backtest_kcm_trim.py`) |
| `backtest_kcm_variants.py` | KCM 변형·MOC·VT 옵션 |
| `review_off_regime.py` | 레짐 OFF 규칙 스윕 |
| `analog_regime_break.py` | 레짐 붕괴 유사 사례 |

그 외 `strategy_qld.py`, ML 파이프라인 등은 실험용입니다. 실전 KCM은 위 표를 우선하세요.
