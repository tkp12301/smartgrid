"""
전력 피크 분석 대시보드 (선형회귀 전용)

- 데이터: EPSIS 전력수급 > 최대/최소전력 월별평균 (단위: MW)
  https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsAepChart.do?menuId=030200
  (내려받은 CSV 내용을 아래 RAW_CSV에 그대로 넣어 두었습니다. 별도 업로드 불필요)
- 사용 모델: 선형회귀(LinearRegression) 하나만 사용
- 실행: streamlit run main.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="전력 피크 분석 (선형회귀)", page_icon="⚡", layout="wide")

# ----------------------------------------------------------------------------
# 0. 데이터 (HOME_전력수급_최대_최소전력월별평균.csv 내용)
#    값 0 = 해당 월 집계 자료 없음 (1993~2004년 대부분, 2026년 12월)
# ----------------------------------------------------------------------------
RAW_CSV = """연도,1월,2월,3월,4월,5월,6월,7월,8월,9월,10월,11월,12월
2026,80230,73911,68515,65977,66070,71508,83943,85608,72628,68126,68812,0
2025,75680,77517,68985,65207,64992,71643,85033,85492,77581,66272,68249,76285
2024,79136,73263,70091,64455,64361,70480,80515,87799,78052,66907,67831,75537
2023,79463,76183,68151,64932,65016,70645,79213,82736,73483,65754,71252,76530
2022,79797,77278,71145,66096,66243,71805,82007,80375,70674,66443,68926,82176
2021,77620,72010,67268,63917,63388,68857,81158,77300,69327,67052,70028,78180
2020,72980,71955,65529,61320,59098,66800,68877,76952,67542,63088,67184,75851
2019,77943,72266,67921,65096,63716,66376,74481,77699,68403,64446,66970,73705
2018,78964,76149,68813,64055,64337,68127,79038,80710,66413,65070,68469,75613
2017,73183,73898,68249,61887,62302,67071,75838,74083,67696,61631,69196,76423
2016,73545,69910,66582,60977,61390,66208,71773,74996,64635,62347,66712,72279
2015,71533,68475,66488,61826,59491,64649,68341,69103,63130,61068,63772,70136
2014,69421,68346,64621,60480,59656,62792,68873,64257,61390,60616,63332,72785
2013,70541,66893,63018,60315,58211,62688,67212,68080,59517,60147,63887,69024
2012,66689,68102,63290,57832,57275,61947,65745,66748,59444,57400,63100,69267
2011,69165,63520,62182,57323,55592,59893,63897,63944,59368,57828,60392,66612
2010,63588,60211,58483,55034,52061,57693,61887,63356,57283,55271,59731,65736
2009,55944,55135,53185,50054,48891,53241,55835,56194,54840,50822,56378,61387
2008,57236,55752,52031,49233,48296,50567,56698,54473,51694,50197,51634,54511
2007,52635,50477,50391,47417,46429,49785,51237,55009,48333,48831,52458,54900
2006,49974,50536,48505,46339,44177,46558,47083,52611,46736,44316,48852,52071
2005,47379,46021,46421,43339,42371,44958,47443,47279,44485,43553,46863,51533
2004,0,0,0,0,0,45038,45114,45377,41455,41745,43943,45863
2003,0,0,0,0,0,40949,40092,41153,41626,0,0,0
2002,0,0,0,0,0,0,40991,39302,40420,0,0,0
2001,0,0,0,0,0,0,43125,0,0,0,0,0
2000,0,0,0,0,0,0,0,41007,0,0,0,0
1999,0,0,0,0,0,0,0,37293,0,0,0,0
1998,0,0,0,0,0,0,0,0,32996,0,0,0
1997,0,0,0,0,0,0,0,35851,0,0,0,0
1996,0,0,0,0,0,0,0,32282,0,0,0,0
1995,0,0,0,0,0,0,0,29878,0,0,0,0
1994,0,0,0,0,0,0,26696,0,0,0,0,0
1993,0,0,0,0,0,0,0,0,0,0,0,22112"""


# ----------------------------------------------------------------------------
# 1. 데이터 준비
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """연도 x 1~12월 표를 [연도, 월, 전력, 날짜] 긴 표로 바꾸고 0(자료 없음)은 제외."""
    wide = pd.read_csv(io.StringIO(RAW_CSV))
    long = wide.melt(id_vars="연도", var_name="월", value_name="전력")
    long["월"] = long["월"].str.replace("월", "", regex=False).astype(int)
    long["전력"] = pd.to_numeric(long["전력"], errors="coerce")
    long = long[long["전력"] > 0].copy()
    long["날짜"] = pd.to_datetime(dict(year=long["연도"], month=long["월"], day=1))
    return long.sort_values("날짜").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. 선형회귀
# ----------------------------------------------------------------------------
def months_since(dates, start) -> np.ndarray:
    return ((dates.year - start.year) * 12 + (dates.month - start.month)).values


def fit_linear_regression(df: pd.DataFrame):
    """시간(시작 월로부터 지난 개월 수)을 X, 전력(MW)을 y로 선형회귀."""
    start = df["날짜"].min()
    X = months_since(df["날짜"].dt, start).reshape(-1, 1)
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
    "EPSIS 월별 최대전력(MW) 자료를 선형회귀로 분석하고, "
    "기준값 이상인 달을 전력 피크(빨간 점)로 표시합니다."
)

all_data = load_data()
min_year, max_year = int(all_data["연도"].min()), int(all_data["연도"].max())

with st.sidebar:
    st.header("1. 분석 기간")
    start_year = st.slider("분석 시작 연도", min_year, max_year - 1, 2005)
    st.caption("2005년 이전에는 일부 월만 집계되어 있어 기본값을 2005년으로 두었습니다.")

    st.header("2. 피크 기준")
    mode = st.radio(
        "기준값 정하는 방식",
        ["평균 + k × 표준편차", "상위 N% 분위수", "직접 입력(MW)"],
    )
    k = st.slider("k (표준편차 배수)", 0.0, 3.0, 1.0, 0.1) if mode.startswith("평균") else 1.0
    pct = st.slider("상위 N%", 1, 50, 10) if mode.startswith("상위") else 10
    manual = (
        st.number_input("기준값 (MW)", min_value=0.0, value=80000.0, step=1000.0)
        if mode.startswith("직접")
        else 0.0
    )

    st.header("3. 예측")
    horizon = st.slider("추세선 연장 개월 수", 0, 60, 12)

data = all_data[all_data["연도"] >= start_year].reset_index(drop=True)
if len(data) < 3:
    st.error("선형회귀에는 최소 3개 이상의 데이터가 필요합니다. 시작 연도를 낮춰 주세요.")
    st.stop()

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
            name="최대전력(일반)", marker=dict(color="#4C78A8", size=7),
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
    future_dates, future_pred = None, None
    if horizon > 0:
        future_dates = pd.date_range(
            data["날짜"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        future_t = months_since(future_dates, start)
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
        title="월별 최대전력 산점도와 선형회귀 추세선",
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

    if future_dates is not None:
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
        cnt = data.groupby("연도")["피크"].sum().reset_index().rename(columns={"피크": "피크 횟수"})
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
        fig_m = px.bar(monthly_avg, x="월", y="전력", title="월별 평균 (분석 기간 전체)",
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
