"""
전력 피크 분석 대시보드 (선형회귀 전용)

- 데이터: GitHub에 올려 둔 전력수급 자료를 실행할 때마다 직접 읽어옵니다.
  (파일 업로드 없음)
- 사용 모델: 선형회귀(LinearRegression) 하나만 사용
- 실행: streamlit run main.py
"""

import io
import re
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="전력 피크 분석 (선형회귀)", page_icon="⚡", layout="wide")

GITHUB_URL = (
    "https://raw.githubusercontent.com/tkp12301/smartgrid/refs/heads/main/"
    "%EC%A0%84%EB%A0%A5%EC%88%98%EA%B8%89%EC%9E%90%EB%A3%8C"
)
UNIT = "MW"  # 전력 단위 (자료의 단위가 다르면 여기만 바꾸세요)
DAYS_PER_MONTH = 30.4375


# ----------------------------------------------------------------------------
# 1. GitHub 에서 데이터 읽기
# ----------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다. (utf-8 / cp949 지원)")


def read_text_table(text: str) -> pd.DataFrame:
    """CSV/TSV 텍스트를 표로 읽는다. '1월 ... 12월' 헤더가 있으면 그 줄부터 읽는다."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "1월" in ln and "12월" in ln), 0)
    lines = lines[start:]
    sep = max([",", "\t", ";", "|"], key=lines[0].count)
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(axis=1, how="all").dropna(axis=0, how="all")


@st.cache_data(ttl=3600, show_spinner="GitHub에서 데이터를 불러오는 중...")
def fetch_table(url: str):
    """주소에서 파일을 내려받아 표로 반환. 확장자가 없으면 .csv 도 함께 시도한다."""
    known = (".csv", ".txt", ".tsv", ".xlsx", ".xls")
    urls = [url] if url.lower().endswith(known) else [url, url + ".csv"]
    errors = []
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{u} → {e}")
            continue
        if raw[:2] == b"PK":  # 엑셀 파일
            return pd.read_excel(io.BytesIO(raw)), u
        return read_text_table(decode_bytes(raw)), u
    raise RuntimeError("\n".join(errors))


# ----------------------------------------------------------------------------
# 2. 표 구조 자동 인식 → [날짜, 전력, 연도, 월] 로 정리
# ----------------------------------------------------------------------------
MONTH_RE = re.compile(r"\s*(\d{1,2})\s*월\s*")


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace(r"\s", "", regex=True),
        errors="coerce",
    )


def parse_dates(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    digits = s2.str.replace(r"\D", "", regex=True)
    n = digits.str.len()
    if (n == 6).mean() > 0.8:
        return pd.to_datetime(digits, format="%Y%m", errors="coerce")
    if (n == 8).mean() > 0.8:
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s2, errors="coerce")


def detect_layout(df: pd.DataFrame) -> str:
    month_cols = [c for c in df.columns[1:] if MONTH_RE.fullmatch(str(c))]
    if len(month_cols) >= 6:
        return "wide"  # 행 = 연도, 열 = 1월~12월
    first = df.columns[0]
    month_rows = df[first].astype(str).str.fullmatch(MONTH_RE.pattern).sum()
    year_cols = [c for c in df.columns[1:] if re.search(r"\d{4}", str(c))]
    if month_rows >= 6 and year_cols:
        return "transposed"  # 행 = 1월~12월, 열 = 연도
    return "long"  # 행 = 날짜(또는 연도+월), 열 = 값


def wide_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    first = df.columns[0]
    month_cols = [c for c in df.columns[1:] if MONTH_RE.fullmatch(str(c))]
    long = df.melt(id_vars=first, value_vars=month_cols, var_name="월", value_name="전력")
    long["연도"] = long[first].astype(str).str.extract(r"(\d{4})")[0]
    long["월"] = long["월"].str.extract(r"(\d{1,2})")[0]
    return finish(long)


def transposed_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    first = df.columns[0]
    year_cols = [c for c in df.columns[1:] if re.search(r"\d{4}", str(c))]
    long = df.melt(id_vars=first, value_vars=year_cols, var_name="연도", value_name="전력")
    long["연도"] = long["연도"].str.extract(r"(\d{4})")[0]
    long["월"] = long[first].astype(str).str.extract(r"(\d{1,2})")[0]
    return finish(long)


def finish(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    long["전력"] = to_num(long["전력"])
    long["연도"] = pd.to_numeric(long["연도"], errors="coerce")
    long["월"] = pd.to_numeric(long["월"], errors="coerce")
    long = long.dropna(subset=["전력", "연도", "월"])
    long = long[(long["전력"] > 0) & long["월"].between(1, 12)]  # 0 = 집계 없음
    long["날짜"] = pd.to_datetime(
        dict(year=long["연도"].astype(int), month=long["월"].astype(int), day=1)
    )
    return long[["날짜", "전력"]].sort_values("날짜").reset_index(drop=True)


def long_candidates(df: pd.DataFrame):
    """long 형태일 때 날짜 컬럼 후보와 값 컬럼 후보를 찾는다."""
    df = df.copy()
    cols = {str(c).strip(): c for c in df.columns}
    year_c = next((cols[k] for k in ("연도", "년도", "년", "year", "Year") if k in cols), None)
    month_c = next((cols[k] for k in ("월", "month", "Month") if k in cols), None)
    if year_c is not None and month_c is not None:
        y = pd.to_numeric(df[year_c], errors="coerce")
        mth = pd.to_numeric(df[month_c], errors="coerce")
        ok = y.notna() & mth.between(1, 12)
        df["날짜(연도+월)"] = pd.NaT
        df.loc[ok, "날짜(연도+월)"] = pd.to_datetime(
            dict(year=y[ok].astype(int), month=mth[ok].astype(int), day=1)
        )

    key = re.compile(r"일자|날짜|일시|기준|년월|기간|date|time", re.I)
    date_cols = []
    for c in df.columns:
        if str(c) == "날짜(연도+월)":
            date_cols.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            continue
        sample = df[c].dropna().head(200)
        if sample.empty:
            continue
        if key.search(str(c)) or not pd.api.types.is_numeric_dtype(df[c]):
            if parse_dates(sample).notna().mean() > 0.8:
                date_cols.append(c)
    date_cols.sort(key=lambda c: (0 if str(c) == "날짜(연도+월)" else 1 if key.search(str(c)) else 2))

    num_cols = [
        c for c in df.columns
        if c not in date_cols and c not in (year_c, month_c) and to_num(df[c]).notna().mean() > 0.8
    ]
    vkey = re.compile(r"최대|전력|수요|공급|부하|발전|mw|kw", re.I)
    num_cols.sort(key=lambda c: 0 if vkey.search(str(c)) else 1)
    return df, date_cols, num_cols


# ----------------------------------------------------------------------------
# 3. 선형회귀
# ----------------------------------------------------------------------------
def elapsed_months(dates: pd.Series, start: pd.Timestamp) -> np.ndarray:
    return ((pd.to_datetime(dates) - start).dt.total_seconds() / 86400 / DAYS_PER_MONTH).values


def fit_linear_regression(df: pd.DataFrame):
    """시간(시작일로부터 지난 개월 수)을 X, 전력을 y로 선형회귀."""
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
# 4. 화면
# ----------------------------------------------------------------------------
st.title("⚡ 전력 피크 분석 대시보드 (선형회귀)")
st.caption(
    "GitHub에 올려 둔 전력수급 자료를 선형회귀로 분석하고, "
    "기준값 이상인 시점을 전력 피크(빨간 점)로 표시합니다."
)

with st.sidebar:
    st.header("데이터 주소")
    data_url = st.text_input("GitHub raw 주소", GITHUB_URL)

try:
    raw_df, used_url = fetch_table(data_url)
except Exception as e:  # noqa: BLE001
    st.error(f"GitHub에서 데이터를 불러오지 못했습니다.\n\n{e}")
    st.stop()

layout = detect_layout(raw_df)
agg_choice = "원본 그대로"
value_label = "전력"

with st.sidebar:
    if layout == "wide":
        data = wide_to_tidy(raw_df)
    elif layout == "transposed":
        data = transposed_to_tidy(raw_df)
    else:
        st.header("컬럼 선택")
        tmp, date_cols, num_cols = long_candidates(raw_df)
        if not date_cols or not num_cols:
            st.error("날짜 컬럼 또는 숫자(전력) 컬럼을 찾지 못했습니다. 아래 원본 표를 확인하세요.")
            st.stop()
        date_col = st.selectbox("날짜 컬럼", date_cols)
        value_col = st.selectbox("전력 컬럼", num_cols)
        value_label = str(value_col)
        d = pd.DataFrame(
            {"날짜": pd.to_datetime(tmp[date_col]) if pd.api.types.is_datetime64_any_dtype(tmp[date_col])
             else parse_dates(tmp[date_col]),
             "전력": to_num(tmp[value_col])}
        ).dropna()
        d = d[d["전력"] > 0].sort_values("날짜").reset_index(drop=True)
        if len(d) > 1:
            step_days = d["날짜"].diff().dt.total_seconds().median() / 86400
        else:
            step_days = 30
        options = ["원본 그대로", "월별 평균", "월별 최대"]
        agg_choice = st.selectbox("집계 단위", options, index=0 if step_days >= 27 else 1)
        if agg_choice != "원본 그대로":
            how = "mean" if agg_choice == "월별 평균" else "max"
            d = (
                d.set_index("날짜")["전력"].resample("MS").agg(how).dropna().reset_index()
            )
        data = d

if data.empty:
    st.error("분석할 수 있는 데이터가 없습니다. 아래 원본 표를 확인하세요.")
    with st.expander("불러온 원본 데이터"):
        st.dataframe(raw_df)
    st.stop()

data["연도"] = data["날짜"].dt.year
data["월"] = data["날짜"].dt.month
if len(data) > 1:
    step_days = data["날짜"].diff().dt.total_seconds().median() / 86400
else:
    step_days = 30
date_fmt = "%Y-%m" if step_days >= 27 else "%Y-%m-%d"

all_data = data
min_year, max_year = int(all_data["연도"].min()), int(all_data["연도"].max())

with st.sidebar:
    st.header("1. 분석 기간")
    if min_year < max_year:
        # 자료가 거의 다 채워진 첫 연도를 기본 시작 연도로 (초기에 몇 달만 있는 연도 제외)
        per_year = all_data.groupby("연도").size()
        dense = per_year[per_year >= 0.75 * per_year.max()]
        default_start = int(dense.index.min()) if len(dense) else min_year
        default_start = min(max(default_start, min_year), max_year - 1)
        start_year = st.slider("분석 시작 연도", min_year, max_year - 1, default_start)
    else:
        start_year = min_year
    st.caption(
        "값이 0 이하인 칸은 집계 자료가 없는 것으로 보고 제외했습니다. "
        "초기 연도에 일부 월만 있으면 기본 시작 연도를 자료가 채워진 해로 잡습니다."
    )

    st.header("2. 피크 기준")
    mode = st.radio(
        "기준값 정하는 방식",
        ["평균 + k × 표준편차", "상위 N% 분위수", "직접 입력(MW)"],
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

with st.expander("불러온 원본 데이터 (상위 20행)"):
    st.caption(f"출처: {used_url}  |  인식된 형태: {layout}  |  집계: {agg_choice}")
    st.dataframe(raw_df.head(20))

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
c2.metric("피크 발생 횟수", f"{len(peaks)}회 / {len(data)}개")
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
            name=f"{value_label}(일반)", marker=dict(color="#4C78A8", size=7),
            hovertemplate=f"%{{x|{date_fmt}}}<br>%{{y:,.0f}} {UNIT}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=peaks["날짜"], y=peaks["전력"], mode="markers",
            name="전력 피크", marker=dict(color="red", size=10, line=dict(color="darkred", width=1)),
            hovertemplate=f"%{{x|{date_fmt}}}<br>%{{y:,.0f}} {UNIT} (피크)<extra></extra>",
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
        title=f"{value_label} 산점도와 선형회귀 추세선",
        xaxis_title="날짜", yaxis_title=f"전력 ({UNIT})",
        legend=dict(orientation="h", y=-0.2), height=560,
    )
    st.plotly_chart(fig)

    sign = "+" if m["intercept"] >= 0 else "-"
    st.info(
        f"**회귀식:** 전력({UNIT}) = {m['slope']:,.2f} × t {sign} {abs(m['intercept']):,.2f}  "
        f"(t = {start:%Y-%m-%d}부터 지난 개월 수)"
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
    table["날짜"] = table["날짜"].dt.strftime(date_fmt)
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
    monthly = data.groupby(["연도", "월"], as_index=False)["전력"].mean()
    left, right = st.columns(2)

    with left:
        fig_line = px.line(
            monthly, x="월", y="전력", color=monthly["연도"].astype(str), markers=True,
            labels={"color": "연도", "전력": f"전력 ({UNIT})"},
            title="연도별 월간 추이 (월 평균)",
        )
        fig_line.update_xaxes(dtick=1)
        st.plotly_chart(fig_line)

    with right:
        cnt = data.groupby("연도")["피크"].sum().reset_index().rename(columns={"피크": "피크 횟수"})
        fig_bar = px.bar(cnt, x="연도", y="피크 횟수", title="연도별 피크 발생 횟수",
                         color_discrete_sequence=["red"])
        fig_bar.update_xaxes(dtick=1)
        st.plotly_chart(fig_bar)

    pivot = monthly.pivot_table(index="연도", columns="월", values="전력")
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
    st.subheader(f"전력 피크 발생 시점 ({len(peaks)}건)")
    if peaks.empty:
        st.write("기준값 이상인 시점이 없습니다. 왼쪽에서 기준을 낮춰 보세요.")
    else:
        pk = peaks[["날짜", "전력", "추세선예측"]].copy()
        pk["날짜"] = pk["날짜"].dt.strftime(date_fmt)
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
