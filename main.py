"""
전력 피크 분석 대시보드 (선형회귀 전용)

- 데이터: GitHub 에 올려 둔 전력수급 자료 (연도 x 1~12월, 단위 MW)
  https://github.com/tkp12301/smartgrid  (전력수급자료)
- 사용 모델: 선형회귀(LinearRegression) 하나만 사용
- 실행: streamlit run main.py
"""

import io
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="전력 피크 분석 (선형회귀)", page_icon="⚡", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/tkp12301/smartgrid/refs/heads/main/"
    "%EC%A0%84%EB%A0%A5%EC%88%98%EA%B8%89%EC%9E%90%EB%A3%8C"
)
UNIT = "MW"
DAYS_PER_MONTH = 30.4375


# ----------------------------------------------------------------------------
# 1. 데이터 불러오기 (GitHub -> 표 -> [날짜, 전력, 연도, 월])
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="GitHub에서 데이터를 불러오는 중...")
def load_raw(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("파일 인코딩을 읽을 수 없습니다.")
    return pd.read_csv(io.StringIO(text))


def to_tidy(wide: pd.DataFrame, keep_zero: bool = False) -> pd.DataFrame:
    """연도 x 1~12월 표를 [연도, 월, 전력, 날짜] 긴 표로 바꾼다.
    값 0 = 집계 없음이라 기본은 제외하고, keep_zero=True 이면 0도 남긴다(결측 점검용)."""
    wide = wide.copy()
    wide.columns = [str(c).strip() for c in wide.columns]
    month_cols = [c for c in wide.columns if c.endswith("월")]
    long = wide.melt(id_vars="연도", value_vars=month_cols, var_name="월", value_name="전력")
    long["월"] = long["월"].str.replace("월", "", regex=False).astype(int)
    long["전력"] = pd.to_numeric(
        long["전력"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    long = long.dropna(subset=["전력"])
    if not keep_zero:
        long = long[long["전력"] > 0]
    long = long.copy()
    long["날짜"] = pd.to_datetime(dict(year=long["연도"].astype(int), month=long["월"], day=1))
    return long.sort_values("날짜").reset_index(drop=True)[["연도", "월", "전력", "날짜"]]


# ----------------------------------------------------------------------------
# 2. 선형회귀
# ----------------------------------------------------------------------------
def elapsed_months(dates, start) -> np.ndarray:
    return ((pd.to_datetime(dates) - start).dt.total_seconds() / 86400 / DAYS_PER_MONTH).values


def fit_linear_regression(df: pd.DataFrame):
    """시작 월로부터 지난 개월 수(t)를 X, 전력(MW)을 y로 선형회귀."""
    start = df["날짜"].min()
    X = elapsed_months(df["날짜"], start).reshape(-1, 1)
    y = df["전력"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    metrics = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }
    return model, start, pred, metrics


def calc_threshold(values: pd.Series, mode: str, k: float, pct: float, manual: float) -> float:
    if mode == "평균 + k × 표준편차":
        return float(values.mean() + k * values.std())
    if mode == "상위 N% 분위수":
        return float(np.percentile(values, 100 - pct))
    return float(manual)


# ----------------------------------------------------------------------------
# 3. 화면
# ----------------------------------------------------------------------------
st.title("⚡ 전력 피크 분석 대시보드 (선형회귀)")
st.caption(
    "GitHub에 올려 둔 월별 전력수급 자료를 선형회귀로 분석하고, "
    "기준값 이상인 달을 전력 피크(빨간 점)로 표시합니다."
)

try:
    raw_df = load_raw(DATA_URL)
    all_data = to_tidy(raw_df)
except Exception as e:  # noqa: BLE001
    st.error(f"GitHub에서 데이터를 불러오지 못했습니다: {e}")
    st.stop()

min_year, max_year = int(all_data["연도"].min()), int(all_data["연도"].max())

with st.sidebar:
    st.header("1. 분석 기간")
    # 자료가 거의 다 채워진 첫 연도를 기본 시작 연도로 (초기에 몇 달만 있는 연도 제외)
    per_year = all_data.groupby("연도").size()
    dense = per_year[per_year >= 0.75 * per_year.max()]
    default_start = int(dense.index.min()) if len(dense) else min_year
    default_start = min(max(default_start, min_year), max_year - 1)
    start_year = st.slider("분석 시작 연도", min_year, max_year - 1, default_start)
    st.caption(
        "값이 0인 칸은 집계 자료가 없는 것으로 보고 제외했습니다. "
        "1993~2004년은 일부 월만 있어 기본 시작 연도를 2005년으로 잡았습니다."
    )

    st.header("2. 피크 기준")
    mode = st.radio(
        "기준값 정하는 방식",
        ["평균 + k × 표준편차", "상위 N% 분위수", "직접 입력"],
    )
    k = st.slider("k (표준편차 배수)", 0.0, 3.0, 1.0, 0.1) if mode.startswith("평균") else 1.0
    pct = st.slider("상위 N%", 1, 50, 10) if mode.startswith("상위") else 10
    manual = (
        st.number_input(
            f"기준값 ({UNIT})", min_value=0.0,
            value=float(round(all_data["전력"].quantile(0.9), -2)), step=1000.0,
        )
        if mode.startswith("직접")
        else 0.0
    )

    st.header("3. 예측")
    horizon = st.slider("추세선 연장 개월 수", 0, 60, 12)

data = all_data[all_data["연도"] >= start_year].reset_index(drop=True)
if len(data) < 3:
    st.error("선형회귀에는 최소 3개 이상의 데이터가 필요합니다. 시작 연도를 낮춰 주세요.")
    st.stop()

with st.expander("불러온 원본 데이터 (상위 10행)"):
    st.caption("값 0 = 집계 자료 없음")
    st.dataframe(raw_df.head(10), hide_index=True)

# 모델 학습 및 피크 판정
model, start, pred, m = fit_linear_regression(data)
threshold = calc_threshold(data["전력"], mode, k, pct, manual)

data = data.assign(
    추세선예측=pred,
    잔차=data["전력"].values - pred,
    피크=data["전력"] >= threshold,
)
peaks = data[data["피크"]]

# ----------------------------------------------------------------------------
# 데이터 품질 점검 계산 (결측치 · 이상치) - 🧹 탭과 🔍 탭에서 함께 사용
# ----------------------------------------------------------------------------
rd = raw_df.copy()
rd.columns = [str(c).strip() for c in rd.columns]
month_cols = [c for c in rd.columns if c.endswith("월")]
raw_vals = rd.set_index("연도")[month_cols].apply(pd.to_numeric, errors="coerce").sort_index()
n_rows, n_cols = rd.shape
n_cells = int(raw_vals.size)
n_zero = int((raw_vals == 0).sum().sum())
n_nan = int(raw_vals.isna().sum().sum())
n_dup = int(rd["연도"].duplicated().sum())
valid_per_year = (raw_vals > 0).sum(axis=1)
n_sparse_years = int((valid_per_year < 12).sum())
full_idx = valid_per_year[valid_per_year == 12].index
first_full_year = int(full_idx.min()) if len(full_idx) else int(raw_vals.index.min())

# --- 결측치: 0 이 어디에 있는지 분류 ---
grid = to_tidy(rd, keep_zero=True)  # 0 을 포함한 전체 달력 [연도, 월, 전력, 날짜]
valid_dates = grid.loc[grid["전력"] > 0, "날짜"]
first_valid, last_valid = valid_dates.min(), valid_dates.max()
zero_grid = grid[grid["전력"] == 0]
n_before = int((zero_grid["날짜"] < first_valid).sum())  # 첫 기록 이전
n_after = int((zero_grid["날짜"] > last_valid).sum())  # 마지막 기록 이후
n_inner = int(len(zero_grid) - n_before - n_after)  # 기록 기간 중간 공백

inner = grid[(grid["날짜"] >= first_valid) & (grid["날짜"] <= last_valid)].copy()
inner["결측"] = inner["전력"] == 0
run_id = (inner["결측"] != inner["결측"].shift()).cumsum()
gap_runs = (
    inner[inner["결측"]]
    .groupby(run_id[inner["결측"]])
    .agg(시작=("날짜", "min"), 끝=("날짜", "max"), 개월수=("날짜", "size"))
    .reset_index(drop=True)
)
continuous_from = (
    gap_runs["끝"].max() + pd.offsets.MonthBegin(1) if len(gap_runs) else first_valid
)


def trend_stats(df_: pd.DataFrame):
    """선형회귀로 본 연간 기울기, R², 데이터 개수 (처리 방식 비교용)."""
    t0 = df_["날짜"].min()
    X_ = elapsed_months(df_["날짜"], t0).reshape(-1, 1)
    y_ = df_["전력"].values
    mdl = LinearRegression().fit(X_, y_)
    return float(mdl.coef_[0] * 12), float(r2_score(y_, mdl.predict(X_))), len(df_)


sel = grid[grid["날짜"] >= pd.Timestamp(year=int(start_year), month=1, day=1)]
interp = (
    grid.set_index("날짜")["전력"].replace(0, np.nan).loc[first_valid:last_valid].interpolate()
    .reset_index()
)
interp.columns = ["날짜", "전력"]
cmp_opts = [
    ("① 0 제외 · 현재 분석 구간 (채택)", sel[sel["전력"] > 0]),
    ("② 0 제외 · 전체 유효값", grid[grid["전력"] > 0]),
    ("③ 0 그대로 포함 · 현재 분석 구간", sel),
    ("④ 0 그대로 포함 · 전체 기간", grid),
    ("⑤ 선형보간으로 공백을 채움 · 전체 유효 구간 (참고)", interp),
]
cmp_df = pd.DataFrame(
    [
        {"처리 방식": name, f"기울기({UNIT}/년)": sl, "R²": r2_, "데이터 수": n_}
        for name, d_ in cmp_opts
        for sl, r2_, n_ in [trend_stats(d_)]
    ]
)

# --- 이상치: 4가지 방법 ---
s_ = data["전력"]
q1_, q3_ = s_.quantile(0.25), s_.quantile(0.75)
iqr_ = q3_ - q1_
iqr_lo, iqr_hi = q1_ - 1.5 * iqr_, q3_ + 1.5 * iqr_
flag_iqr = (s_ < iqr_lo) | (s_ > iqr_hi)  # ① 전체 IQR
z_all = (s_ - s_.mean()) / s_.std()
flag_z = z_all.abs() > 3  # ② 전체 Z-점수
adj = data["잔차"] - data.groupby("월")["잔차"].transform("mean")  # 추세와 월별 평균을 뺀 나머지
z_adj = adj / adj.std()
flag_adj = z_adj.abs() > 3  # ③ 추세·계절 보정 잔차 Z-점수
prev_ok = data.groupby("월")["연도"].diff() == 1  # 바로 전 해의 같은 달이 있는 경우만
yoy = (data.groupby("월")["전력"].pct_change() * 100).where(prev_ok)
yoy_z = (yoy - yoy.mean()) / yoy.std()
flag_yoy = yoy_z.abs() > 3  # ④ 전년 동월 대비 변화율 Z-점수
n_iqr, n_z3, n_rz3, n_yoy = int(flag_iqr.sum()), int(flag_z.sum()), int(flag_adj.sum()), int(flag_yoy.sum())


# 요약 지표
c1, c2, c3, c4 = st.columns(4)
c1.metric("피크 기준값", f"{threshold:,.0f} {UNIT}")
c2.metric("피크 발생 횟수", f"{len(peaks)}회 / {len(data)}개월")
c3.metric("연간 증가 추세", f"{m['slope'] * 12:+,.0f} {UNIT}/년")
c4.metric("결정계수 R²", f"{m['r2']:.3f}")

tab0, tabq, tab1, tab2, tab3, tab4 = st.tabs(
    ["🧹 전처리·탐색", "🔍 결측·이상치 점검", "📈 회귀 그래프", "📋 회귀 결과 도표", "📊 데이터 정리 차트", "🔴 피크 목록"]
)

# --- 탭 0: 데이터 전처리 및 탐색 -------------------------------------------
with tab0:
    # ===================== A. 데이터 전처리 =====================
    st.header("A. 데이터 전처리")

    st.subheader("A-1. 원본 구조 확인")
    st.markdown(
        f"- 원본은 **{n_rows}행 × {n_cols}열**입니다. 행은 연도({int(raw_vals.index.min())}~{int(raw_vals.index.max())}), "
        f"열은 1월~12월이며 값은 월별 평균 최대전력({UNIT})입니다.\n"
        f"- 중복된 연도는 **{n_dup}개**, 빈 칸(NaN)은 **{n_nan}개**이고 값은 모두 숫자입니다.\n"
        f"- 다만 월별 칸 {n_cells}개 중 **{n_zero}개({n_zero / n_cells:.1%})가 0**입니다. "
        f"전력 사용량이 실제로 0일 수는 없으므로 0은 *집계 자료 없음*으로 해석했습니다."
    )

    st.subheader("A-2. 결측치·이상치 점검 결과")
    st.markdown(
        f"- **결측치**: 빈 칸은 {n_nan}개지만 값이 없는 달이 0으로 채워져 있어 **0인 칸 {n_zero}개**가 실제 결측입니다. "
        f"모두 제외했고, 값이 연속으로 있는 {first_full_year}년 이후를 기본 분석 구간으로 삼았습니다.\n"
        f"- **이상치**: IQR·Z-점수·추세/계절 보정 방법에서는 **{n_iqr + n_z3 + n_rz3}건**, "
        f"전년 동월 대비 변화율에서는 **{n_yoy}건**이 의심되었고 삭제·수정하지 않았습니다.\n"
        f"- 조사 방법과 그래프, 처리 방식별 결과 비교는 **🔍 결측·이상치 점검** 탭에 정리했습니다."
    )

    st.subheader("A-3. 형태 변환 (넓은 표 → 긴 표)")
    st.markdown(
        "- 선형회귀와 시계열 그래프를 그리려면 '연도 × 12개월' 표를 **한 행에 한 달**씩 쌓은 긴 표로 바꿔야 합니다.\n"
        "- 이때 `연도`, `월`, `전력` 컬럼을 만들고 연도와 월을 합쳐 `날짜`(YYYY-MM-01)를 만들었습니다. 아래는 변환 결과 앞부분입니다."
    )
    preview = all_data.head(8).copy()
    preview["날짜"] = preview["날짜"].dt.strftime("%Y-%m-%d")
    st.dataframe(preview, hide_index=True)

    st.subheader("A-4. 분석 구간 선정")
    n_expected = (data["날짜"].max().year - data["날짜"].min().year) * 12 + (
        data["날짜"].max().month - data["날짜"].min().month
    ) + 1
    n_gap = n_expected - len(data)
    st.markdown(
        f"- 현재 분석 구간은 **{data['날짜'].min():%Y-%m} ~ {data['날짜'].max():%Y-%m}**, "
        f"**{len(data)}개월**입니다. 이 구간에서 빠진 달은 **{n_gap}개**입니다"
        f"{' (매달 연속이라 시간 간격이 일정합니다).' if n_gap == 0 else ' (빠진 달이 있어 자기상관 해석에 주의하세요).'}\n"
        f"- 값이 일부 달에만 있는 초기 연도를 넣으면 시간 간격이 들쭉날쭉해져 추세 기울기가 왜곡될 수 있어, "
        f"기본 시작 연도를 자료가 모두 채워진 해로 잡았습니다. (사이드바에서 바꿀 수 있습니다.)"
    )

    st.divider()

    # ===================== B. 데이터 탐색 (EDA) =====================
    st.header("B. 데이터 탐색 (EDA)")
    st.caption(f"분석 구간({data['날짜'].min():%Y-%m} ~ {data['날짜'].max():%Y-%m}) 기준입니다.")

    # B-1 기술통계
    st.subheader("B-1. 기술통계")
    s = data["전력"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    desc = pd.DataFrame(
        {
            "항목": ["개수", "평균", "표준편차", "최소", "25%", "중앙값", "75%", "최대", "왜도", "첨도"],
            "값": [
                f"{len(s):,}", f"{s.mean():,.0f}", f"{s.std():,.0f}", f"{s.min():,.0f}",
                f"{q1:,.0f}", f"{s.median():,.0f}", f"{q3:,.0f}", f"{s.max():,.0f}",
                f"{s.skew():.3f}", f"{s.kurt():.3f}",
            ],
        }
    )
    imax, imin = s.idxmax(), s.idxmin()
    cl, cr = st.columns([1, 2])
    with cl:
        st.dataframe(desc, hide_index=True)
    with cr:
        st.markdown(
            f"- 평균({s.mean():,.0f} {UNIT})과 중앙값({s.median():,.0f} {UNIT})은 "
            f"{'서로 가깝습니다' if abs(s.mean() - s.median()) < 0.1 * s.std() else '차이가 있습니다'}. "
            f"왜도({s.skew():.2f})로 보면 분포는 {'거의 좌우 대칭입니다' if abs(s.skew()) < 0.5 else '한쪽으로 치우쳐 있습니다'}.\n"
            f"- 첨도({s.kurt():.2f})는 {'0보다 작아 봉우리가 완만하고 넓게 퍼진 분포' if s.kurt() < 0 else '0보다 커 중심이 뾰족하고 꼬리가 긴 분포'}입니다.\n"
            f"- 최대는 {data.loc[imax, '날짜']:%Y-%m}의 {s.max():,.0f} {UNIT}, "
            f"최소는 {data.loc[imin, '날짜']:%Y-%m}의 {s.min():,.0f} {UNIT}입니다. "
            f"최대와 최소는 {s.max() / s.min():.1f}배 차이입니다."
        )

    # B-2 분포
    st.subheader("B-2. 분포")
    fig_hist = px.histogram(data, x="전력", nbins=30, title="최대전력 분포 (히스토그램)",
                            labels={"전력": f"전력 ({UNIT})"})
    fig_hist.update_layout(yaxis_title="개월 수")
    st.plotly_chart(fig_hist)

    # B-3 시계열
    st.subheader("B-3. 시간에 따른 변화 (장기 추세)")
    corr_time = float(np.corrcoef(elapsed_months(data["날짜"], start), s.values)[0, 1])
    ma12 = s.rolling(12).mean()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=data["날짜"], y=s, mode="lines", name="월별 최대전력",
                                line=dict(color="#9ecae1", width=1.5)))
    fig_ts.add_trace(go.Scatter(x=data["날짜"], y=ma12, mode="lines", name="12개월 이동평균",
                                line=dict(color="#d62728", width=3)))
    fig_ts.update_layout(title="월별 최대전력과 12개월 이동평균", xaxis_title="날짜",
                         yaxis_title=f"전력 ({UNIT})", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_ts)
    st.markdown(
        f"- 시간과 전력의 상관계수는 **{corr_time:.2f}**로, {'뚜렷한 우상향 추세가 있습니다' if corr_time > 0.5 else '추세가 약한 편입니다'}. "
        f"12개월 이동평균은 계절 변동을 걷어낸 장기 흐름으로, 이 선이 오르는 만큼 선형회귀 기울기"
        f"(연 {m['slope'] * 12:+,.0f} {UNIT})가 나타납니다."
    )

    # B-4 월별 계절성
    st.subheader("B-4. 월별 계절성")
    mm = data.groupby("월")["전력"].mean()
    top3 = mm.sort_values(ascending=False).head(3)
    best_m, worst_m = int(mm.idxmax()), int(mm.idxmin())
    cl, cr = st.columns(2)
    with cl:
        fig_mbox = px.box(data, x="월", y="전력", title="월별 분포 (박스플롯)",
                          labels={"전력": f"전력 ({UNIT})"})
        fig_mbox.update_xaxes(dtick=1)
        st.plotly_chart(fig_mbox)
    with cr:
        mmdf = mm.reset_index()
        fig_mmean = px.bar(mmdf, x="월", y="전력", title="월별 평균 최대전력",
                           labels={"전력": f"전력 ({UNIT})"})
        fig_mmean.update_xaxes(dtick=1)
        st.plotly_chart(fig_mmean)
    st.markdown(
        f"- 월 평균이 가장 높은 달은 **{best_m}월({mm.max():,.0f} {UNIT})**, 가장 낮은 달은 **{worst_m}월({mm.min():,.0f} {UNIT})**이며 "
        f"그 차이는 {mm.max() - mm.min():,.0f} {UNIT}입니다.\n"
        f"- 높은 달 상위 3개는 {', '.join(f'{int(k)}월' for k in top3.index)}입니다. "
        f"{'여름과 겨울에 한 번씩 오르고 봄·가을에 내려가는, 두 번 솟는 모양입니다(냉·난방 수요 영향으로 추정).' if ({7, 8} & set(top3.index.astype(int))) and ({12, 1, 2} & set(top3.index.astype(int))) else '계절에 따라 값이 반복해서 오르내리는 모양입니다.'}"
    )

    # B-5 계절별 비교
    def season_of(m_):
        if m_ in (3, 4, 5):
            return "봄(3~5월)"
        if m_ in (6, 7, 8):
            return "여름(6~8월)"
        if m_ in (9, 10, 11):
            return "가을(9~11월)"
        return "겨울(12~2월)"

    st.subheader("B-5. 계절별 비교")
    seasons = ["봄(3~5월)", "여름(6~8월)", "가을(9~11월)", "겨울(12~2월)"]
    sdf = data.assign(계절=data["월"].map(season_of))
    sm = sdf.groupby("계절")["전력"].mean().reindex(seasons)
    fig_season = px.box(sdf, x="계절", y="전력", category_orders={"계절": seasons},
                        title="계절별 최대전력 분포", labels={"전력": f"전력 ({UNIT})"})
    st.plotly_chart(fig_season)
    st.markdown(
        f"- 계절 평균은 봄 {sm.iloc[0]:,.0f}, 여름 {sm.iloc[1]:,.0f}, 가을 {sm.iloc[2]:,.0f}, 겨울 {sm.iloc[3]:,.0f} {UNIT}이며 "
        f"**{sm.idxmax()}**이 가장 높고 **{sm.idxmin()}**이 가장 낮습니다."
    )

    # B-6 연도별 변화
    st.subheader("B-6. 연도별 변화")
    cnt_y = data.groupby("연도").size()
    full = cnt_y[cnt_y == 12].index
    yearly = data[data["연도"].isin(full)].groupby("연도")["전력"].agg(연최대="max", 연평균="mean")
    if len(yearly) >= 2:
        peak_month = data[data["연도"].isin(full)].loc[
            data[data["연도"].isin(full)].groupby("연도")["전력"].idxmax(), "월"
        ]
        vc = peak_month.value_counts().sort_index()
        yearly["전년 대비 연최대 증가율(%)"] = yearly["연최대"].pct_change() * 100
        n_y = len(yearly)
        cagr = ((yearly["연최대"].iloc[-1] / yearly["연최대"].iloc[0]) ** (1 / (n_y - 1)) - 1) * 100
        yr_df = yearly.reset_index()
        cl, cr = st.columns(2)
        with cl:
            fig_yr = px.line(yr_df, x="연도", y=["연최대", "연평균"], markers=True,
                             title="연도별 연 최대·연 평균 (12개월 모두 있는 해)",
                             labels={"value": f"전력 ({UNIT})", "variable": ""})
            st.plotly_chart(fig_yr)
        with cr:
            fig_g = px.bar(yr_df.dropna(), x="연도", y="전년 대비 연최대 증가율(%)",
                           title="전년 대비 연 최대전력 증가율(%)")
            fig_g.add_hline(y=0, line_color="black")
            st.plotly_chart(fig_g)
        vc_text = ", ".join(f"{int(k)}월 {int(v)}회" for k, v in vc.sort_values(ascending=False).items())
        st.markdown(
            f"- 12개월이 모두 있는 {n_y}개 연도({int(yearly.index.min())}~{int(yearly.index.max())})를 비교했습니다. "
            f"연 최대전력은 해마다 평균 약 **{cagr:.2f}%** 늘었습니다(연복리 기준).\n"
            f"- 매년 가장 높았던 달은 {vc_text}입니다. 즉 연 최대는 여름·겨울에 집중됩니다.\n"
            f"- 증가율이 해마다 일정하지 않고 오르내리므로, 직선 추세는 평균적인 방향만 보여줍니다."
        )
    else:
        st.write("12개월이 모두 있는 연도가 2개 미만이라 연도별 비교를 건너뜁니다.")

    # B-7 자기상관
    st.subheader("B-7. 이전 값과의 관계 (자기상관)")
    max_lag = int(min(24, len(s) // 2))
    ac = pd.DataFrame({"간격(개월)": range(1, max_lag + 1),
                       "자기상관": [s.autocorr(lag) for lag in range(1, max_lag + 1)]})
    fig_ac = px.bar(ac, x="간격(개월)", y="자기상관", title="자기상관 (n개월 전 값과의 상관)")
    fig_ac.update_xaxes(dtick=1)
    st.plotly_chart(fig_ac)
    lag1 = s.autocorr(1)
    ac_txt = f"- 한 달 전 값과의 상관은 **{lag1:.2f}**입니다.\n"
    if max_lag >= 12:
        lag12 = s.autocorr(12)
        ac_txt += (
            f"- 12개월 전 값과의 상관은 **{lag12:.2f}**로, 같은 달끼리 거의 같은 패턴이 해마다 반복됩니다 "
            f"(계절성이 강함).\n"
        )
    st.markdown(ac_txt)

    st.divider()

    # ===================== C. 정리 =====================
    st.header("C. 탐색 결과 요약과 모델링 시사점")
    st.markdown(
        f"1. **전처리**: 0({n_zero}칸)을 결측으로 제외하고 긴 표로 바꿨으며, 분석 구간 {len(data)}개월에는 빠진 달 {n_gap}개입니다. "
        f"이상치 의심은 IQR·Z·보정 잔차 기준 {n_iqr + n_z3 + n_rz3}건, 전년 동월 대비 기준 {n_yoy}건이었고 모두 그대로 사용했습니다.\n"
        f"2. **장기 추세**: 시간과 전력의 상관계수가 {corr_time:.2f}이므로 "
        f"{'시간(t)을 독립변수로 쓰는 **선형회귀가 적합**합니다' if corr_time > 0.5 else '선형회귀의 설명력이 낮을 수 있습니다'}. "
        f"추세는 연 {m['slope'] * 12:+,.0f} {UNIT}입니다.\n"
        f"3. **계절성**: {best_m}월이 가장 높고 {worst_m}월이 가장 낮으며 12개월 자기상관이 "
        f"{s.autocorr(12) if max_lag >= 12 else float('nan'):.2f}입니다. 직선 하나는 이 계절 변동을 설명하지 못해 "
        f"R²가 {m['r2']:.3f}에 머뭅니다.\n"
        f"4. **피크 기준**: 추세 때문에 기준값을 하나로 고정하면 뒤쪽 연도에 피크가 몰립니다. "
        f"그래서 평균+k×표준편차, 상위 N% 등 기준을 바꿔 가며 피크 횟수를 비교하도록 만들었습니다."
    )

# --- 탭 Q: 결측치·이상치 점검 ------------------------------------------------
with tabq:
    st.header("결측치·이상치 점검")
    st.caption("결측치와 이상치가 있는지 조사하고, 어떻게 처리했는지 근거와 함께 정리했습니다.")

    # ===================== 1. 결측치 =====================
    st.subheader("1. 결측치 조사")
    miss_tbl = pd.DataFrame(
        {
            "구분": [
                "빈 칸(NaN)", "값이 0인 칸 (전체)", "  · 첫 기록 이전 (수집 시작 전)",
                "  · 마지막 기록 이후 (아직 집계되지 않은 달)", "  · 기록 기간 중간의 공백",
            ],
            "개수": [n_nan, n_zero, n_before, n_after, n_inner],
            "전체 칸 대비": [f"{v / n_cells:.1%}" for v in [n_nan, n_zero, n_before, n_after, n_inner]],
        }
    )
    st.dataframe(miss_tbl, hide_index=True)
    longest = int(gap_runs["개월수"].max()) if len(gap_runs) else 0
    st.markdown(
        f"- **빈 칸(NaN)은 {n_nan}개**입니다. 하지만 값이 없는 달을 비워 두지 않고 **0으로 채워 놓았기 때문**에, "
        f"실제 결측은 0인 칸 **{n_zero}개({n_zero / n_cells:.1%})**입니다. "
        f"전력 사용량이 0일 수는 없으므로 0은 *자료 없음*으로 해석했습니다.\n"
        f"- 첫 기록은 **{first_valid:%Y-%m}**, 마지막 기록은 **{last_valid:%Y-%m}**입니다.\n"
        f"- 0인 칸의 종류: 첫 기록 이전 **{n_before}개**, 마지막 기록 이후 **{n_after}개**"
        f"({'아직 집계되지 않은 달로 보입니다' if n_after else '없음'}), "
        f"기록 기간 중간의 공백 **{n_inner}개**({len(gap_runs)}개 구간, 가장 긴 공백 {longest}개월)입니다.\n"
        + (
            f"- 중간 공백은 마지막이 **{gap_runs['끝'].max():%Y-%m}**에서 끝나고, "
            f"**{continuous_from:%Y-%m}부터는 매달 값이 연속**으로 있습니다."
            if len(gap_runs)
            else "- 중간에 빠진 달은 없고 첫 기록부터 매달 값이 연속으로 있습니다."
        )
    )

    cl, cr = st.columns(2)
    with cl:
        present = (raw_vals > 0).astype(int)
        present.index = present.index.astype(str)
        fig_na = px.imshow(
            present, aspect="auto", color_continuous_scale=[[0, "#d9d9d9"], [1, "#4C78A8"]],
            labels=dict(x="월", y="연도"), title="자료 유무 (파랑 = 있음, 회색 = 없음(0))",
        )
        fig_na.update_coloraxes(showscale=False)
        st.plotly_chart(fig_na)
    with cr:
        vpy = valid_per_year.reset_index()
        vpy.columns = ["연도", "값이 있는 달 수"]
        fig_vpy = px.bar(vpy, x="연도", y="값이 있는 달 수", title="연도별 값이 있는 달 수")
        fig_vpy.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="12개월")
        st.plotly_chart(fig_vpy)

    if len(gap_runs):
        st.markdown("**기록 기간 중간의 공백 구간**")
        gt = gap_runs.copy()
        gt["시작"] = gt["시작"].dt.strftime("%Y-%m")
        gt["끝"] = gt["끝"].dt.strftime("%Y-%m")
        st.dataframe(gt, hide_index=True)

    st.markdown("**결측(0) 처리 방식에 따른 결과 비교 (선형회귀 기울기)**")
    st.dataframe(
        cmp_df.style.format({f"기울기({UNIT}/년)": "{:,.0f}", "R²": "{:.3f}", "데이터 수": "{:,.0f}"}),
        hide_index=True,
    )
    fig_cmp = px.bar(cmp_df, x="처리 방식", y=f"기울기({UNIT}/년)", title="처리 방식별 연간 기울기")
    st.plotly_chart(fig_cmp)
    c1_slope = float(cmp_df.iloc[0][f"기울기({UNIT}/년)"])
    c4_slope = float(cmp_df.iloc[3][f"기울기({UNIT}/년)"])
    c5_slope = float(cmp_df.iloc[4][f"기울기({UNIT}/년)"])
    c5_r2 = float(cmp_df.iloc[4]["R²"])
    st.markdown(
        f"- 0을 값으로 그대로 두면(④) 연간 기울기가 **{c4_slope:,.0f} {UNIT}**로 계산되어, "
        f"0을 제외한 결과(①, {c1_slope:,.0f} {UNIT})의 **{c4_slope / c1_slope:.1f}배**가 됩니다. "
        f"자료 없음을 0으로 읽어 추세가 크게 왜곡되는 것입니다.\n"
        f"- 선형보간으로 공백을 채우면(⑤) 기울기 {c5_slope:,.0f} {UNIT}, R² {c5_r2:.2f}가 나옵니다. "
        f"하지만 채운 값은 실제로 관측한 값이 아니라 만들어 낸 값이고 공백이 {len(gap_runs)}개 구간에 걸쳐 있어 쓰지 않았습니다.\n"
        f"- **처리 결과**: 0은 결측으로 보고 제외했고, 값이 연속으로 있는 구간만 분석에 사용했으며(기본 {start_year}년 이후) "
        f"보간은 하지 않았습니다."
    )

    st.divider()

    # ===================== 2. 이상치 =====================
    st.subheader("2. 이상치 조사")
    st.markdown(
        f"- 분석 구간({data['날짜'].min():%Y-%m} ~ {data['날짜'].max():%Y-%m}, {len(data)}개월)에서 찾았습니다.\n"
        "- **전력 피크는 이상치가 아니라 분석 대상**입니다. 여기서 찾는 것은 입력 오류나 비정상적으로 튀는 값입니다.\n"
        "- 이 자료는 **추세(해마다 증가)와 계절성(여름·겨울 상승)**이 있어 값 하나만 보면 정상적인 여름 값이 튀어 보이거나 "
        "겨울 값이 묻힐 수 있습니다. 그래서 값 자체를 보는 방법 2가지와, 추세·계절을 걷어낸 뒤 보는 방법 2가지를 함께 썼습니다."
    )
    meth = pd.DataFrame(
        {
            "방법": [
                "① 전체 IQR", "② 전체 Z-점수", "③ 추세·계절 보정 잔차 Z-점수", "④ 전년 동월 대비 변화율 Z-점수",
            ],
            "무엇을 보나": [
                f"{iqr_lo:,.0f} ~ {iqr_hi:,.0f} {UNIT} 밖의 값",
                "평균에서 표준편차 3배 이상 떨어진 값",
                "추세선과 월별 평균을 뺀 나머지가 3배 이상 큰 달",
                "같은 달끼리 전년 대비 증감률이 3배 이상 튀는 달",
            ],
            "의심 값 개수": [n_iqr, n_z3, n_rz3, n_yoy],
            "가장 큰 |점수|": [
                "-", f"{z_all.abs().max():.2f}", f"{z_adj.abs().max():.2f}", f"{yoy_z.abs().max():.2f}",
            ],
        }
    )
    st.dataframe(meth, hide_index=True)

    cl, cr = st.columns(2)
    with cl:
        fig_box2 = px.box(data, y="전력", points="outliers", title="① 최대전력 박스플롯",
                          labels={"전력": f"전력 ({UNIT})"})
        st.plotly_chart(fig_box2)
    with cr:
        zadf = pd.DataFrame({"날짜": data["날짜"], "보정 잔차 Z": z_adj.values})
        fig_zadj = px.scatter(zadf, x="날짜", y="보정 잔차 Z", title="③ 추세·계절 보정 잔차 Z (±3 밖이면 의심)")
        fig_zadj.add_hline(y=3, line_dash="dash", line_color="red")
        fig_zadj.add_hline(y=-3, line_dash="dash", line_color="red")
        st.plotly_chart(fig_zadj)

    yoydf = pd.DataFrame({"날짜": data["날짜"], "전년 동월 대비(%)": yoy.values}).dropna()
    fig_yoy = px.scatter(yoydf, x="날짜", y="전년 동월 대비(%)", title="④ 전년 동월 대비 변화율(%)")
    fig_yoy.add_hline(y=float(yoy.mean() + 3 * yoy.std()), line_dash="dash", line_color="red",
                      annotation_text="평균 + 3σ")
    fig_yoy.add_hline(y=float(yoy.mean() - 3 * yoy.std()), line_dash="dash", line_color="red",
                      annotation_text="평균 − 3σ")
    st.plotly_chart(fig_yoy)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**③ 보정 잔차 |Z| 상위 5개**")
        idx_a = z_adj.abs().nlargest(5).index
        ta = pd.DataFrame(
            {"날짜": data.loc[idx_a, "날짜"].dt.strftime("%Y-%m"), f"전력({UNIT})": data.loc[idx_a, "전력"],
             "보정 잔차 Z": z_adj.loc[idx_a]}
        )
        st.dataframe(ta.style.format({f"전력({UNIT})": "{:,.0f}", "보정 잔차 Z": "{:+.2f}"}), hide_index=True)
    with cr:
        st.markdown("**④ 전년 동월 대비 |Z| 상위 5개**")
        idx_y = yoy_z.abs().nlargest(5).index
        ty = pd.DataFrame(
            {"날짜": data.loc[idx_y, "날짜"].dt.strftime("%Y-%m"), f"전력({UNIT})": data.loc[idx_y, "전력"],
             "전년 동월 대비(%)": yoy.loc[idx_y], "Z": yoy_z.loc[idx_y]}
        )
        st.dataframe(
            ty.style.format({f"전력({UNIT})": "{:,.0f}", "전년 동월 대비(%)": "{:+.1f}", "Z": "{:+.2f}"}),
            hide_index=True,
        )

    # --- 이상치 결론 ---
    n_basic = n_iqr + n_z3 + n_rz3
    yoy_list = ", ".join(
        f"{d:%Y-%m}(전년 동월 대비 {v:+.1f}%)"
        for d, v in zip(data.loc[flag_yoy, "날짜"], yoy[flag_yoy])
    )
    in_range = bool(not (flag_iqr | flag_z)[flag_yoy].any()) if n_yoy else True
    lines = [
        f"- **값 자체를 본 방법(①②)과 추세·계절 보정 방법(③)**에서는 의심 값이 "
        f"{'**없습니다**' if n_basic == 0 else f'**{n_basic}건** 나왔습니다'} "
        f"(가장 큰 점수: Z {z_all.abs().max():.2f}, 보정 잔차 Z {z_adj.abs().max():.2f}).",
    ]
    if n_yoy:
        lines.append(
            f"- **전년 동월 대비(④)** 기준으로는 **{n_yoy}건**이 기준을 넘었습니다: {yoy_list}. "
            f"전년 같은 달 값이 평소보다 낮거나 높아서 생긴 변화일 수 있으며, 이 자료만으로 입력 오류인지는 알 수 없습니다. "
            + ("값 자체는 전체 분포 범위 안에 있습니다." if in_range else "값 자체도 전체 분포 범위를 벗어납니다.")
        )
    else:
        lines.append("- **전년 동월 대비(④)** 기준으로도 의심 값이 없습니다.")
    lines.append(
        "- **처리 결과**: 확실한 이상치로 볼 값이 없어 **삭제하거나 고치지 않고 그대로 사용**했습니다. "
        "전력 사용량이 크게 달라지는 것은 실제 변동일 수 있고, 지우면 전력 피크 분석 자체가 왜곡되기 때문입니다."
    )
    st.markdown("\n".join(lines))

    last_row = data.iloc[-1]
    same = all_data[(all_data["월"] == last_row["월"]) & (all_data["연도"] < last_row["연도"])].tail(5)["전력"]
    if len(same) > 0:
        ratio = last_row["전력"] / same.mean()
        st.markdown(
            f"- **마지막 달 점검**: 가장 최근 값은 {last_row['날짜']:%Y-%m}의 {last_row['전력']:,.0f} {UNIT}이며, "
            f"같은 달 직전 {len(same)}년 평균의 **{ratio:.1%}** 수준입니다. "
            f"{'비슷한 수준이라 눈에 띄는 이상은 없습니다.' if 0.9 <= ratio <= 1.1 else '차이가 커서 아직 집계가 끝나지 않은 값인지 확인이 필요합니다.'} "
            f"(최종 확정값인지는 자료만으로 알 수 없습니다.)"
        )

    st.divider()
    st.subheader("3. 점검 결과 한눈에 보기")
    summ = pd.DataFrame(
        {
            "항목": ["빈 칸(NaN)", "값이 0인 칸", "이상치(IQR·Z·보정 잔차)", "이상치(전년 동월 대비)"],
            "조사 결과": [
                f"{n_nan}개",
                f"{n_zero}개 (첫 기록 이전 {n_before}, 마지막 기록 이후 {n_after}, 중간 공백 {n_inner})",
                f"{n_basic}건",
                f"{n_yoy}건",
            ],
            "처리": [
                "처리할 것 없음",
                "결측으로 제외 (보간하지 않음), 연속 구간만 사용",
                "삭제·수정하지 않고 사용",
                "삭제·수정하지 않고 사용 (원인은 자료로 확인 불가)" if n_yoy else "해당 없음",
            ],
        }
    )
    st.dataframe(summ, hide_index=True)

# --- 탭 1: 산점도 + 추세선 + 피크 ------------------------------------------
with tab1:
    normal = data[~data["피크"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal["날짜"], y=normal["전력"], mode="markers",
            name="최대전력(일반)", marker=dict(color="#4C78A8", size=7),
            hovertemplate=f"%{{x|%Y-%m}}<br>%{{y:,.0f}} {UNIT}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=peaks["날짜"], y=peaks["전력"], mode="markers",
            name="전력 피크", marker=dict(color="red", size=10, line=dict(color="darkred", width=1)),
            hovertemplate=f"%{{x|%Y-%m}}<br>%{{y:,.0f}} {UNIT} (피크)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["날짜"], y=data["추세선예측"], mode="lines",
            name="선형회귀 추세선", line=dict(color="black", width=2),
        )
    )

    future_dates, future_pred = None, None
    if horizon > 0:
        future_dates = pd.date_range(
            data["날짜"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        future_pred = model.predict(elapsed_months(pd.Series(future_dates), start).reshape(-1, 1))
        fig.add_trace(
            go.Scatter(
                x=future_dates, y=future_pred, mode="lines",
                name="추세선 연장(예측)", line=dict(color="black", width=2, dash="dot"),
            )
        )

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"피크 기준 {threshold:,.0f} {UNIT}", annotation_position="top left",
    )
    fig.update_layout(
        title="월별 최대전력 산점도와 선형회귀 추세선",
        xaxis_title="날짜", yaxis_title=f"전력 ({UNIT})",
        legend=dict(orientation="h", y=-0.2), height=560,
    )
    st.plotly_chart(fig)

    sign = "+" if m["intercept"] >= 0 else "-"
    st.info(
        f"**회귀식:** 전력({UNIT}) = {m['slope']:,.2f} × t {sign} {abs(m['intercept']):,.2f}  "
        f"(t = {start:%Y-%m}부터 지난 개월 수)"
    )

# --- 탭 2: 회귀 결과 도표 ---------------------------------------------------
with tab2:
    st.subheader("회귀 계수 및 성능")
    summary = pd.DataFrame(
        {
            "항목": [
                f"기울기 ({UNIT}/월)", f"기울기 ({UNIT}/년)", f"절편 ({UNIT})", "R²",
                f"RMSE ({UNIT})", "데이터 개수", f"피크 기준값 ({UNIT})", "피크 횟수",
            ],
            "값": [
                f"{m['slope']:,.2f}", f"{m['slope'] * 12:,.2f}", f"{m['intercept']:,.2f}",
                f"{m['r2']:.4f}", f"{m['rmse']:,.2f}", f"{len(data):,}",
                f"{threshold:,.0f}", f"{len(peaks):,}",
            ],
        }
    )
    st.dataframe(summary, hide_index=True)

    st.subheader("실측값 vs 회귀 추세선 값")
    table = data[["날짜", "전력", "추세선예측", "잔차", "피크"]].copy()
    table["날짜"] = table["날짜"].dt.strftime("%Y-%m")
    table = table.rename(
        columns={"전력": f"실측({UNIT})", "추세선예측": f"추세선({UNIT})", "잔차": f"잔차({UNIT})"}
    )
    st.dataframe(
        table.style.format(
            {f"실측({UNIT})": "{:,.0f}", f"추세선({UNIT})": "{:,.0f}", f"잔차({UNIT})": "{:+,.0f}"}
        ),
        hide_index=True,
    )
    st.download_button(
        "회귀 결과 CSV 다운로드",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="regression_result.csv",
        mime="text/csv",
    )

    if future_dates is not None:
        st.subheader(f"향후 {horizon}개월 추세선 예측값")
        fut = pd.DataFrame(
            {"날짜": future_dates.strftime("%Y-%m"), f"예측({UNIT})": future_pred}
        )
        st.dataframe(fut.style.format({f"예측({UNIT})": "{:,.0f}"}), hide_index=True)

# --- 탭 3: 데이터 정리 차트 -------------------------------------------------
with tab3:
    left, right = st.columns(2)

    with left:
        fig_line = px.line(
            data, x="월", y="전력", color=data["연도"].astype(str), markers=True,
            labels={"color": "연도", "전력": f"전력 ({UNIT})"},
            title="연도별 월간 추이",
        )
        fig_line.update_xaxes(dtick=1)
        st.plotly_chart(fig_line)

    with right:
        cnt = data.groupby("연도")["피크"].sum().reset_index().rename(columns={"피크": "피크 횟수"})
        fig_bar = px.bar(cnt, x="연도", y="피크 횟수", title="연도별 피크 발생 횟수",
                         color_discrete_sequence=["red"])
        fig_bar.update_xaxes(dtick=1)
        st.plotly_chart(fig_bar)

    pivot = data.pivot_table(index="연도", columns="월", values="전력")
    fig_heat = px.imshow(
        pivot, aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="월", y="연도", color=UNIT), title="연도 × 월 히트맵",
    )
    st.plotly_chart(fig_heat)

    left2, right2 = st.columns(2)
    with left2:
        monthly_avg = data.groupby("월")["전력"].mean().reset_index()
        fig_m = px.bar(monthly_avg, x="월", y="전력", title="월별 평균 (분석 기간 전체)",
                       labels={"전력": f"전력 ({UNIT})"})
        fig_m.update_xaxes(dtick=1)
        st.plotly_chart(fig_m)
    with right2:
        fig_res = px.scatter(data, x="날짜", y="잔차", color="피크",
                             color_discrete_map={True: "red", False: "#4C78A8"},
                             title="회귀 잔차 (실측 − 추세선)")
        fig_res.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_res)

# --- 탭 4: 피크 목록 --------------------------------------------------------
with tab4:
    st.subheader(f"전력 피크 발생 월 ({len(peaks)}건)")
    if peaks.empty:
        st.write("기준값 이상인 달이 없습니다. 왼쪽에서 기준을 낮춰 보세요.")
    else:
        pk = peaks[["날짜", "전력", "추세선예측"]].copy()
        pk["날짜"] = pk["날짜"].dt.strftime("%Y-%m")
        pk[f"기준 초과({UNIT})"] = peaks["전력"].values - threshold
        pk = pk.rename(columns={"전력": f"실측({UNIT})", "추세선예측": f"추세선({UNIT})"})
        st.dataframe(
            pk.style.format(
                {f"실측({UNIT})": "{:,.0f}", f"추세선({UNIT})": "{:,.0f}",
                 f"기준 초과({UNIT})": "{:+,.0f}"}
            ),
            hide_index=True,
        )
        st.download_button(
            "피크 목록 CSV 다운로드",
            pk.to_csv(index=False).encode("utf-8-sig"),
            file_name="peak_list.csv",
            mime="text/csv",
        )
