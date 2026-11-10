"""
전력 피크 분석 대시보드 (선형회귀 전용)

- 데이터 출처: 전력통계정보시스템 EPSIS > 전력수급 > 최대/최소전력 월별평균
  https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsAepChart.do?menuId=030200
- 사용 모델: 선형회귀(LinearRegression) 하나만 사용
- 실행: streamlit run main.py
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

EPSIS_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsAepChart.do?menuId=030200"

st.set_page_config(page_title="전력 피크 분석 (선형회귀)", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> str:
    """EPSIS CSV는 보통 CP949(EUC-KR)라서 인코딩을 순서대로 시도한다."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다. (utf-8 / cp949 지원)")


def read_epsis_csv(raw: bytes) -> pd.DataFrame:
    """'(단위 : MW)' 같은 머리글 줄이 있어도 표 헤더 줄부터 읽는다."""
    text = decode_bytes(raw)
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if "1월" in line and "12월" in line), 0
    )
    return pd.read_csv(io.StringIO("\n".join(lines[start:])), on_bad_lines="skip")


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    EPSIS 표(연도 x 1~12월, 또는 월 x 연도)를
    [연도, 월, 전력, 날짜] 형태의 긴 표로 바꾼다.
    """
    first = df.columns[0]
    records = []

    month_cols = {}
    for c in df.columns[1:]:
        m = re.search(r"(\d{1,2})\s*월", str(c))
        if m:
            month_cols[c] = int(m.group(1))

    if month_cols:  # 행 = 연도, 열 = 1월~12월
        for _, row in df.iterrows():
            y = re.search(r"(\d{4})", str(row[first]))
            if not y:
                continue
            for c, m in month_cols.items():
                records.append((int(y.group(1)), m, row[c]))
    else:  # 행 = 1월~12월, 열 = 연도
        year_cols = {}
        for c in df.columns[1:]:
            y = re.search(r"(\d{4})", str(c))
            if y:
                year_cols[c] = int(y.group(1))
        for _, row in df.iterrows():
            m = re.search(r"(\d{1,2})\s*월", str(row[first]))
            if not m:
                continue
            for c, y in year_cols.items():
                records.append((y, int(m.group(1)), row[c]))

    out = pd.DataFrame(records, columns=["연도", "월", "전력"])
    out["전력"] = pd.to_numeric(
        out["전력"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    out = out.dropna()
    if out.empty:
        raise ValueError("연도/월/전력 값을 찾지 못했습니다. EPSIS에서 받은 CSV인지 확인하세요.")
    out["날짜"] = pd.to_datetime(dict(year=out["연도"], month=out["월"], day=1))
    return out.sort_values("날짜").reset_index(drop=True)


@st.cache_data
def make_demo_data() -> pd.DataFrame:
    """시연용 가상 데이터 (실제 EPSIS 값이 아님)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(dates))
    month = dates.month.values
    seasonal = (
        9000 * np.exp(-((month - 8) ** 2) / 3.0)  # 여름 피크
        + 7000 * np.exp(-((month - 1) ** 2) / 2.0)  # 겨울 피크
    )
    power = 72000 + 90 * t + seasonal + rng.normal(0, 1500, len(t))
    return pd.DataFrame(
        {"연도": dates.year, "월": dates.month, "전력": power.round(0), "날짜": dates}
    )


# ----------------------------------------------------------------------------
# 2. 선형회귀
# ----------------------------------------------------------------------------
def fit_linear_regression(df: pd.DataFrame):
    """시간(시작 월로부터 지난 개월 수)을 X, 전력(MW)을 y로 선형회귀."""
    start = df["날짜"].min()
    t = ((df["날짜"].dt.year - start.year) * 12 + (df["날짜"].dt.month - start.month)).values
    X = t.reshape(-1, 1)
    y = df["전력"].values

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    metrics = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }
    return model, start, t, pred, metrics


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
    "EPSIS 최대/최소전력 월별평균 자료를 선형회귀로 분석하고, "
    "기준값 이상인 달을 전력 피크(빨간 점)로 표시합니다."
)

with st.sidebar:
    st.header("1. 데이터")
    st.markdown(f"[EPSIS 자료 페이지 열기]({EPSIS_URL})")
    st.caption(
        "페이지 하단의 **CSV** 버튼으로 내려받은 파일을 아래에 올려주세요. "
        "(기간/데이터 항목: 최대전력 또는 최소전력 선택)"
    )
    uploaded = st.file_uploader("EPSIS CSV 업로드", type=["csv"])
    series_name = st.selectbox("데이터 종류(표시용)", ["최대전력", "최소전력"])

    st.header("2. 피크 기준")
    mode = st.radio(
        "기준값 정하는 방식",
        ["평균 + k × 표준편차", "상위 N% 분위수", "직접 입력(MW)"],
    )
    k = st.slider("k (표준편차 배수)", 0.0, 3.0, 1.0, 0.1) if mode.startswith("평균") else 1.0
    pct = st.slider("상위 N%", 1, 50, 10) if mode.startswith("상위") else 10
    manual = (
        st.number_input("기준값 (MW)", min_value=0.0, value=90000.0, step=1000.0)
        if mode.startswith("직접")
        else 0.0
    )

    st.header("3. 예측")
    horizon = st.slider("추세선 연장 개월 수", 0, 60, 12)

# 데이터 준비
using_demo = uploaded is None
try:
    data = make_demo_data() if using_demo else to_long(read_epsis_csv(uploaded.getvalue()))
except Exception as e:  # noqa: BLE001
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

if using_demo:
    st.warning(
        "지금은 **시연용 가상 데이터**입니다. 실제 분석은 왼쪽에서 EPSIS CSV를 업로드하세요.",
        icon="⚠️",
    )

if len(data) < 3:
    st.error("선형회귀에는 최소 3개 이상의 데이터가 필요합니다.")
    st.stop()

# 모델 학습 및 피크 판정
model, start, t, pred, m = fit_linear_regression(data)
threshold = calc_threshold(data["전력"], mode, k, pct, manual)

data = data.assign(
    추세선예측=pred,
    잔차=data["전력"].values - pred,
    피크=data["전력"] >= threshold,
)
peaks = data[data["피크"]]

# 요약 지표
c1, c2, c3, c4 = st.columns(4)
c1.metric("피크 기준값", f"{threshold:,.0f} MW")
c2.metric("피크 발생 횟수", f"{len(peaks)}회 / {len(data)}개월")
c3.metric("연간 증가 추세", f"{m['slope'] * 12:+,.0f} MW/년")
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
            name=f"{series_name}(일반)", marker=dict(color="#4C78A8", size=7),
            hovertemplate="%{x|%Y-%m}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=peaks["날짜"], y=peaks["전력"], mode="markers",
            name="전력 피크", marker=dict(color="red", size=10, line=dict(color="darkred", width=1)),
            hovertemplate="%{x|%Y-%m}<br>%{y:,.0f} MW (피크)<extra></extra>",
        )
    )

    # 선형회귀 추세선 (관측 구간)
    fig.add_trace(
        go.Scatter(
            x=data["날짜"], y=data["추세선예측"], mode="lines",
            name="선형회귀 추세선", line=dict(color="black", width=2),
        )
    )

    # 추세선 연장 (예측 구간)
    if horizon > 0:
        future_dates = pd.date_range(data["날짜"].max() + pd.offsets.MonthBegin(1),
                                     periods=horizon, freq="MS")
        future_t = ((future_dates.year - start.year) * 12 + (future_dates.month - start.month)).values
        future_pred = model.predict(future_t.reshape(-1, 1))
        fig.add_trace(
            go.Scatter(
                x=future_dates, y=future_pred, mode="lines",
                name="추세선 연장(예측)", line=dict(color="black", width=2, dash="dot"),
            )
        )

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"피크 기준 {threshold:,.0f} MW", annotation_position="top left",
    )
    fig.update_layout(
        title=f"{series_name} 월별 산점도와 선형회귀 추세선",
        xaxis_title="날짜", yaxis_title="전력 (MW)",
        legend=dict(orientation="h", y=-0.2), height=560,
    )
    st.plotly_chart(fig)

    sign = "+" if m["intercept"] >= 0 else "-"
    st.info(
        f"**회귀식:** 전력(MW) = {m['slope']:,.2f} × t {sign} {abs(m['intercept']):,.2f}  "
        f"(t = {start:%Y-%m}부터 지난 개월 수)"
    )

# --- 탭 2: 회귀 결과 도표 ---------------------------------------------------
with tab2:
    st.subheader("회귀 계수 및 성능")
    summary = pd.DataFrame(
        {
            "항목": [
                "기울기 (MW/월)", "기울기 (MW/년)", "절편 (MW)", "R²", "RMSE (MW)",
                "데이터 개수", "피크 기준값 (MW)", "피크 횟수",
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
    table = table.rename(columns={"전력": "실측(MW)", "추세선예측": "추세선(MW)", "잔차": "잔차(MW)"})
    st.dataframe(
        table.style.format({"실측(MW)": "{:,.0f}", "추세선(MW)": "{:,.0f}", "잔차(MW)": "{:+,.0f}"}),
        hide_index=True,
    )
    st.download_button(
        "회귀 결과 CSV 다운로드",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="regression_result.csv",
        mime="text/csv",
    )

    if horizon > 0:
        st.subheader(f"향후 {horizon}개월 추세선 예측값")
        fut = pd.DataFrame({"날짜": future_dates.strftime("%Y-%m"), "예측(MW)": future_pred})
        st.dataframe(fut.style.format({"예측(MW)": "{:,.0f}"}), hide_index=True)

# --- 탭 3: 데이터 정리 차트 -------------------------------------------------
with tab3:
    left, right = st.columns(2)

    with left:
        fig_line = px.line(
            data, x="월", y="전력", color=data["연도"].astype(str), markers=True,
            labels={"color": "연도", "전력": "전력 (MW)"},
            title="연도별 월간 추이",
        )
        fig_line.update_xaxes(dtick=1)
        st.plotly_chart(fig_line)

    with right:
        cnt = (
            data.groupby("연도")["피크"].sum().reset_index().rename(columns={"피크": "피크 횟수"})
        )
        fig_bar = px.bar(cnt, x="연도", y="피크 횟수", title="연도별 피크 발생 횟수",
                         color_discrete_sequence=["red"])
        fig_bar.update_xaxes(dtick=1)
        st.plotly_chart(fig_bar)

    pivot = data.pivot_table(index="연도", columns="월", values="전력")
    fig_heat = px.imshow(
        pivot, aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="월", y="연도", color="MW"), title="연도 × 월 히트맵",
    )
    st.plotly_chart(fig_heat)

    left2, right2 = st.columns(2)
    with left2:
        monthly_avg = data.groupby("월")["전력"].mean().reset_index()
        fig_m = px.bar(monthly_avg, x="월", y="전력", title="월별 평균 (전체 기간)",
                       labels={"전력": "전력 (MW)"})
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
        pk["기준 초과(MW)"] = peaks["전력"].values - threshold
        pk = pk.rename(columns={"전력": "실측(MW)", "추세선예측": "추세선(MW)"})
        st.dataframe(
            pk.style.format({"실측(MW)": "{:,.0f}", "추세선(MW)": "{:,.0f}", "기준 초과(MW)": "{:+,.0f}"}),
            hide_index=True,
        )
        st.download_button(
            "피크 목록 CSV 다운로드",
            pk.to_csv(index=False).encode("utf-8-sig"),
            file_name="peak_list.csv",
            mime="text/csv",
        )
