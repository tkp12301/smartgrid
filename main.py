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


def to_tidy(wide: pd.DataFrame) -> pd.DataFrame:
    """연도 x 1~12월 표를 [연도, 월, 전력, 날짜] 긴 표로 바꾼다. 값 0 = 집계 없음 → 제외."""
    wide = wide.copy()
    wide.columns = [str(c).strip() for c in wide.columns]
    month_cols = [c for c in wide.columns if c.endswith("월")]
    long = wide.melt(id_vars="연도", value_vars=month_cols, var_name="월", value_name="전력")
    long["월"] = long["월"].str.replace("월", "", regex=False).astype(int)
    long["전력"] = pd.to_numeric(
        long["전력"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    long = long[long["전력"] > 0].copy()
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

# 요약 지표
c1, c2, c3, c4 = st.columns(4)
c1.metric("피크 기준값", f"{threshold:,.0f} {UNIT}")
c2.metric("피크 발생 횟수", f"{len(peaks)}회 / {len(data)}개월")
c3.metric("연간 증가 추세", f"{m['slope'] * 12:+,.0f} {UNIT}/년")
c4.metric("결정계수 R²", f"{m['r2']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 회귀 그래프", "📋 회귀 결과 도표", "📊 데이터 정리 차트", "🔴 피크 목록"]
)

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
