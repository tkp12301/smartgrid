# -*- coding: utf-8 -*-
"""
스마트 그리드 전력망 안정성 판별 앱 (Streamlit)

- 데이터: 공공데이터포털(data.go.kr) 한국전력거래소 5분 단위 전력수급 현황
- 모델  : 선형회귀(LinearRegression / Ridge / Lasso)로 '공급예비율'을 예측하고,
          예측값이 임계값보다 낮으면 '불안정(주의)'으로 판별
- 배포  : GitHub에 main.py, requirements.txt 두 파일만 올려 Streamlit Community Cloud에서 실행
          (디스크 저장 없음, API 키는 Secrets로만 관리)
"""

# ===== 1. 설정 =====
import time
import urllib.parse
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 반드시 첫 번째 Streamlit 명령이어야 합니다.
st.set_page_config(page_title="스마트 그리드 안정성 판별", page_icon="⚡", layout="wide")

# 학습과 예측에서 똑같이 쓰는 입력 변수 목록 (순서 중요!)
# [데이터 누수 방지]
# 현재부하, 공급예비력, 운영예비력, 운영예비율은 입력에서 제외합니다.
# 공급예비력 = 공급능력 - 현재부하 이고, 공급예비율 = 공급예비력 / 현재부하 × 100 이라서
# 이 값들을 넣으면 정답(공급예비율)이 입력으로 새어 들어가 성능이 비정상적으로 높게 나옵니다.
FEATURES = [
    "공급능력", "예측부하",
    "hour", "weekday", "month", "is_weekend", "hour_sin", "hour_cos",
]
TARGET = "공급예비율"
REQUIRED_COLS = ["기준일시", "공급능력", "예측부하", "공급예비율"]
NUM_COLS = ["공급능력", "현재부하", "예측부하", "공급예비력", "공급예비율"]
OUTLIER_COLS = ["공급능력", "예측부하", "공급예비율"]
WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
MODEL_NAMES = ["LinearRegression", "Ridge", "Lasso"]
MIN_ROWS = 200  # 학습에 필요한 최소 행 수

FEATURE_LABELS = {
    "공급능력": "공급능력",
    "예측부하": "예측부하",
    "hour": "시각(hour)",
    "weekday": "요일(0=월~6=일)",
    "month": "월(month)",
    "is_weekend": "주말 여부",
    "hour_sin": "시각 주기(sin)",
    "hour_cos": "시각 주기(cos)",
}

# [확인 필요] 실제 API 응답 필드명 -> 표준 컬럼명 변환표
# 응답 필드명이 다르면 이 딕셔너리에 한 줄만 추가/수정하면 됩니다. (대소문자 구분 없음)
COLUMN_MAP = {
    "baseDatetime": "기준일시", "base_datetime": "기준일시", "baseDate": "기준일시",
    "기준일시": "기준일시", "기준일시(5분)": "기준일시", "일시": "기준일시", "datetime": "기준일시",
    "supplyCapacity": "공급능력", "suppAbility": "공급능력", "supply_capacity": "공급능력",
    "공급능력": "공급능력", "공급능력(MW)": "공급능력",
    "currPwrTot": "현재부하", "currentLoad": "현재부하", "current_load": "현재부하",
    "현재부하": "현재부하", "현재수요": "현재부하", "현재부하(MW)": "현재부하",
    "forecastLoad": "예측부하", "forecast_load": "예측부하",
    "예측부하": "예측부하", "예측수요": "예측부하", "예측부하(MW)": "예측부하",
    "suppReservePwr": "공급예비력", "supplyReserve": "공급예비력", "supply_reserve": "공급예비력",
    "공급예비력": "공급예비력", "공급예비력(MW)": "공급예비력",
    "suppReserveRate": "공급예비율", "supplyReserveRate": "공급예비율", "supply_reserve_rate": "공급예비율",
    "공급예비율": "공급예비율", "공급예비율(%)": "공급예비율",
}

# [확인 필요] 공공데이터포털 활용신청 상세페이지의 '요청주소(End Point)'로 교체하세요.
DEFAULT_API_URL = "https://apis.data.go.kr/(활용신청-상세페이지의-요청주소로-교체)"

SRC_DEMO = "데모 모드 (모의 데이터)"
SRC_API = "API 호출 (공공데이터포털)"
SRC_CSV = "CSV 업로드"


# ===== 2. 데이터 수집 (데모 / API / CSV) =====
@st.cache_data(show_spinner=False)
def generate_demo_data(n_rows: int = 4032, seed: int = 42) -> pd.DataFrame:
    """5분 단위 모의 전력수급 데이터를 만든다. (약 2주, 4032행)

    - 낮(오전 11시)과 저녁(오후 7시)에 부하가 높고 새벽에 낮은 일중 패턴
    - 주말에는 부하 감소
    - 폭염일처럼 부하가 치솟는 날을 넣어 '불안정(예비율 10% 미만)' 구간이 일부 생기게 함
    ※ 실제 데이터가 아니므로 분석 결과를 실제 결론처럼 쓰면 안 됩니다.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-20 00:00", periods=n_rows, freq="5min")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    weekday = ts.weekday.to_numpy()
    day_idx = np.arange(n_rows) // 288

    daily = (
        0.45 * np.exp(-((hour - 11) / 4.0) ** 2)
        + 0.55 * np.exp(-((hour - 18.5) / 3.0) ** 2)
        + 0.15
    )
    daily = (daily - daily.min()) / (daily.max() - daily.min())

    week_factor = np.where(weekday == 5, 0.93, np.where(weekday == 6, 0.88, 1.0))
    heat_days = np.isin(day_idx, [2, 3, 10, 11])          # 부하가 치솟는 날
    heat_factor = np.where(heat_days, 1.055, 1.0)

    load = (62000 + 28000 * daily) * week_factor * heat_factor
    load = load + rng.normal(0, 600, n_rows)

    # 공급능력: 일 단위로 조금씩 변동 + 5분 단위 작은 잡음
    cap_day = 99500 + rng.normal(0, 500, day_idx.max() + 1)
    capacity = cap_day[day_idx] + rng.normal(0, 150, n_rows)

    forecast = load * (1 + rng.normal(0, 0.008, n_rows))
    reserve = capacity - load
    rate = reserve / load * 100

    return pd.DataFrame({
        "기준일시": ts,
        "공급능력": capacity.round(1),
        "현재부하": load.round(1),
        "예측부하": forecast.round(1),
        "공급예비력": reserve.round(1),
        "공급예비율": rate.round(2),
    })


def get_service_key():
    """Secrets에서 서비스키를 안전하게 읽는다. 없으면 None (앱이 죽지 않게 처리).

    [중요] 공공데이터포털의 '일반 인증키(Decoding)'를 Secrets에 넣으세요.
    requests가 params를 자동으로 인코딩하므로, Encoding 키를 넣으면 이중 인코딩 오류가 납니다.
    (Encoding 키를 넣은 경우를 대비해 아래 fetch_api_data에서 자동으로 한 번 디코딩합니다.)
    """
    try:
        key = st.secrets["SERVICE_KEY"]
        return str(key).strip() or None
    except Exception:
        return None


def _get_with_retry(url: str, params: dict):
    """timeout 15초, 최대 3회 재시도. 오류 메시지에 서비스키가 노출되지 않게 처리."""
    last = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(
        f"API 호출이 3회 모두 실패했습니다 ({type(last).__name__}). "
        "요청주소/서비스키/활용신청 승인 여부를 확인하세요. "
        "일부 공공 API는 해외 서버(Streamlit Cloud) 접속을 막을 수 있습니다."
    )


def _parse_xml_response(text: str):
    """XML 응답에서 item 목록과 totalCount를 꺼낸다."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise RuntimeError("API 응답을 해석하지 못했습니다. 요청주소와 서비스키를 확인하세요.")
    code = root.findtext(".//resultCode")
    if code is not None and code not in ("00", "0", "0000"):
        raise RuntimeError(f"API 오류: {code} {root.findtext('.//resultMsg') or ''}")
    items = [{c.tag: c.text for c in it} for it in root.iter("item")]
    if not items:
        msg = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg") or "데이터가 없습니다"
        raise RuntimeError(f"API 응답 오류: {msg}")
    total = root.findtext(".//totalCount")
    return items, (int(total) if total and total.isdigit() else None)


def parse_api_response(resp):
    """JSON(우선) 또는 XML 응답을 (item 리스트, 전체 건수)로 변환한다. [확인 필요: 응답 구조]"""
    try:
        data = resp.json()
    except ValueError:
        return _parse_xml_response(resp.text)

    response = data.get("response", data) if isinstance(data, dict) else {}
    header = response.get("header", {}) if isinstance(response, dict) else {}
    code = str(header.get("resultCode", "00"))
    if code not in ("00", "0", "0000", "INFO-000"):
        raise RuntimeError(f"API 오류: {code} {header.get('resultMsg', '')}")
    body = response.get("body", {}) if isinstance(response, dict) else {}
    items = body.get("items", []) if isinstance(body, dict) else []
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        items = []
    total = body.get("totalCount") if isinstance(body, dict) else None
    try:
        total = int(total) if total not in (None, "") else None
    except (TypeError, ValueError):
        total = None
    return items, total


@st.cache_data(ttl=600, show_spinner=False)
def fetch_api_data(service_key: str, api_url: str, n_rows: int) -> pd.DataFrame:
    """공공데이터포털 API를 페이지 단위로 호출해 DataFrame으로 반환한다."""
    key = urllib.parse.unquote(service_key) if "%" in service_key else service_key
    per_page = int(min(1000, n_rows))
    rows, page = [], 1
    while len(rows) < n_rows and page <= 50:
        # [확인 필요] 파라미터 이름은 API마다 다를 수 있습니다.
        params = {"serviceKey": key, "pageNo": page, "numOfRows": per_page,
                  "dataType": "json", "_type": "json"}
        items, total = parse_api_response(_get_with_retry(api_url, params))
        if not items:
            break
        rows.extend(items)
        if total is not None and len(rows) >= total:
            break
        page += 1
    if not rows:
        raise RuntimeError("API 응답에 데이터가 없습니다.")
    return pd.DataFrame(rows[:n_rows])


@st.cache_data(show_spinner=False)
def parse_csv(data: bytes) -> pd.DataFrame:
    """업로드된 CSV를 읽는다. 한글 인코딩(utf-8, cp949)을 순서대로 시도."""
    last = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(BytesIO(data), encoding=enc)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"CSV를 읽지 못했습니다: {type(last).__name__}")


def parse_datetime(s: pd.Series) -> pd.Series:
    """다양한 형식(20250120123000, 2025-01-20 12:30 등)의 기준일시를 datetime으로 변환."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    t = s.astype(str).str.strip()
    if t.str.fullmatch(r"\d{14}").mean() > 0.9:
        return pd.to_datetime(t, format="%Y%m%d%H%M%S", errors="coerce")
    if t.str.fullmatch(r"\d{12}").mean() > 0.9:
        return pd.to_datetime(t, format="%Y%m%d%H%M", errors="coerce")
    return pd.to_datetime(t, errors="coerce")


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 COLUMN_MAP으로 표준화하고 기준일시를 datetime으로 바꾼다."""
    df = df.copy()
    lower_map = {k.lower(): v for k, v in COLUMN_MAP.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    if "기준일시" in df.columns:
        df["기준일시"] = parse_datetime(df["기준일시"])
    return df


# ===== 3. 전처리 =====
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 파생변수 생성. hour_sin/cos는 23시와 0시가 가깝다는 것을 모델에 알려주는 변수."""
    df = df.copy()
    ts = df["기준일시"]
    frac_hour = ts.dt.hour + ts.dt.minute / 60
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24)
    return df


def detect_outliers(df: pd.DataFrame, cols: list):
    """IQR(1.5배) 기준 이상치 표와 이상치 행 마스크를 반환."""
    rows = []
    mask = pd.Series(False, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m = (df[c] < lo) | (df[c] > hi)
        mask |= m
        rows.append({"컬럼": c, "하한": round(lo, 2), "상한": round(hi, 2),
                     "이상치 수": int(m.sum()), "비율(%)": round(m.mean() * 100, 2)})
    return pd.DataFrame(rows), mask


def preprocess(raw: pd.DataFrame, fill_method: str, remove_out: bool):
    """전처리 전체 과정. (처리된 df, 단계별 표, 이상치 표, 오류 메시지) 반환."""
    steps = [("원본", len(raw))]
    df = raw.copy()

    # 숫자 변환 (콤마 제거 포함)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

    # 기준일시 없는 행 제거 + 시간순 정렬 + 중복 제거
    df = df.dropna(subset=["기준일시"]).sort_values("기준일시")
    steps.append(("기준일시 오류 행 제거", len(df)))
    df = df.drop_duplicates(subset=["기준일시"], keep="first")
    steps.append(("중복 제거", len(df)))

    # 공급예비율이 없으면 계산 (공급예비율 = (공급능력 - 현재부하) / 현재부하 × 100)
    if TARGET not in df.columns and {"공급능력", "현재부하"} <= set(df.columns):
        df[TARGET] = (df["공급능력"] - df["현재부하"]) / df["현재부하"] * 100

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        return None, None, None, (
            f"필수 컬럼이 없습니다: {', '.join(missing_cols)}. "
            "코드 상단 COLUMN_MAP에 실제 컬럼명을 추가하세요."
        )

    # 결측치 처리 (시계열이므로 보간 또는 앞값 채우기)
    present = [c for c in NUM_COLS if c in df.columns]
    miss_before = int(df[present].isna().sum().sum())
    if fill_method == "선형보간":
        df[present] = df[present].interpolate(method="linear", limit_direction="both")
    else:
        df[present] = df[present].ffill().bfill()
    df = df.dropna(subset=REQUIRED_COLS)
    steps.append((f"결측치 처리({miss_before}개 채움)", len(df)))

    if len(df) == 0:
        return None, None, None, "전처리 후 남은 데이터가 없습니다."

    # 이상치 탐지 (기본은 제거하지 않음: 선형회귀는 이상치에 민감하므로 확인만 한다)
    outlier_df, mask = detect_outliers(df, OUTLIER_COLS)
    if remove_out:
        df = df[~mask]
        steps.append(("이상치 제거", len(df)))

    df = add_time_features(df).reset_index(drop=True)
    steps.append(("최종(파생변수 포함)", len(df)))
    step_df = pd.DataFrame(steps, columns=["단계", "행 수"])
    return df, step_df, outlier_df, None


# ===== 4. 탐구(EDA) 도구 =====
def show(fig, key: str):
    """plotly 그래프를 표시 (Streamlit 버전 차이를 피하려고 옵션 없이 호출)."""
    st.plotly_chart(fig, key=key)


def insight(hint: str = ""):
    """각 그래프 아래 '알 수 있는 점' 메모 자리."""
    tail = f" (힌트: {hint})" if hint else ""
    st.caption(f"📝 이 그래프에서 알 수 있는 점: ________{tail}")


def calc_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """다중공선성 진단(VIF). VIF가 10 이상이면 다른 변수와 겹치는 정도가 큼."""
    use = [c for c in cols if df[c].std() > 0]
    rows = []
    for c in use:
        others = [o for o in use if o != c]
        if not others:
            continue
        r2 = LinearRegression().fit(df[others], df[c]).score(df[others], df[c])
        vif = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
        rows.append({"변수": c, "VIF": vif})
    out = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["판단"] = np.where(out["VIF"] >= 10, "겹침 큼(주의)", "양호")
    return out


# ===== 5. 모델 학습 =====
def build_pipeline(name: str, alpha: float) -> Pipeline:
    """StandardScaler + 선형 모델을 하나로 묶는다. (스케일러는 train에만 fit됨)"""
    if name == "Ridge":
        model = Ridge(alpha=alpha)
    elif name == "Lasso":
        model = Lasso(alpha=alpha, max_iter=20000)
    else:
        model = LinearRegression()
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def reg_metrics(y_true, y_pred) -> dict:
    """회귀 성능 지표 R², RMSE, MAE."""
    return {
        "R²": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def data_signature(df: pd.DataFrame) -> tuple:
    """데이터가 바뀌었는지 확인하기 위한 서명."""
    return (len(df), str(df["기준일시"].min()), str(df["기준일시"].max()))


def train_models(df: pd.DataFrame, model_name: str, alpha: float, test_ratio: float) -> dict:
    """시간순 분할(앞 80% 학습 / 뒤 20% 테스트) 후 3개 모델을 학습·비교한다.
    시계열은 미래를 맞히는 문제라서 무작위 분할을 쓰지 않습니다."""
    split = int(len(df) * (1 - test_ratio))
    train, test = df.iloc[:split], df.iloc[split:]
    x_tr, y_tr = train[FEATURES], train[TARGET]
    x_te, y_te = test[FEATURES], test[TARGET]

    rows, pipes = [], {}
    for nm in MODEL_NAMES:
        pipe = build_pipeline(nm, alpha).fit(x_tr, y_tr)
        met = reg_metrics(y_te, pipe.predict(x_te))
        try:
            cv = cross_val_score(pipe, x_tr, y_tr, cv=TimeSeriesSplit(n_splits=5), scoring="r2")
            cv_mean, cv_std = float(cv.mean()), float(cv.std())
        except Exception:  # noqa: BLE001
            cv_mean, cv_std = float("nan"), float("nan")
        rows.append({"모델": nm, **met, "CV R² 평균": cv_mean, "CV R² 표준편차": cv_std})
        pipes[nm] = pipe

    pipe = pipes[model_name]
    pred = pipe.predict(x_te)
    table = pd.DataFrame(rows)
    sel = table[table["모델"] == model_name].iloc[0]
    return {
        "name": model_name, "alpha": alpha, "pipe": pipe, "pred": pred,
        "test": test[["기준일시", TARGET]].reset_index(drop=True),
        "table": table, "metrics": {k: float(sel[k]) for k in ["R²", "RMSE", "MAE"]},
        "cv_mean": float(sel["CV R² 평균"]), "cv_std": float(sel["CV R² 표준편차"]),
        "train_size": len(train), "test_size": len(test), "sig": data_signature(df),
    }


def get_valid_result(df: pd.DataFrame):
    """저장된 학습 결과가 현재 데이터와 맞으면 반환, 아니면 None."""
    res = st.session_state.get("result")
    if res is not None and res["sig"] == data_signature(df):
        return res
    return None


def class_metrics(y_true_rate, y_pred_rate, thr: float) -> dict:
    """임계값 기준으로 안정(0)/불안정(1) 라벨을 만들어 분류 지표를 계산."""
    yt = (np.asarray(y_true_rate) < thr).astype(int)
    yp = (np.asarray(y_pred_rate) < thr).astype(int)
    return {
        "Accuracy": accuracy_score(yt, yp),
        "Precision": precision_score(yt, yp, zero_division=0),
        "Recall": recall_score(yt, yp, zero_division=0),
        "F1": f1_score(yt, yp, zero_division=0),
        "yt": yt, "yp": yp,
    }


# ===== 6. 화면(탭) 구성 =====
def render_sidebar():
    """사이드바 입력값을 모아서 반환."""
    st.sidebar.header("⚙️ 설정")
    source = st.sidebar.radio("데이터 소스", [SRC_DEMO, SRC_API, SRC_CSV])
    api_url, n_rows = DEFAULT_API_URL, 2000
    if source == SRC_API:
        api_url = st.sidebar.text_input("API 요청주소 (End Point)", value=DEFAULT_API_URL)
        n_rows = st.sidebar.slider("조회 건수", 100, 10000, 2000, step=100)

    st.sidebar.subheader("전처리")
    fill_method = st.sidebar.radio("결측치 처리", ["선형보간", "앞값 채우기(ffill)"])
    remove_out = st.sidebar.checkbox("이상치 제거 (IQR 1.5배)", value=False)

    st.sidebar.subheader("모델")
    model_name = st.sidebar.selectbox("모델 선택", MODEL_NAMES)
    alpha = st.sidebar.select_slider("alpha (Ridge/Lasso)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=0.1)
    test_ratio = st.sidebar.slider("테스트 비율 (뒤쪽 구간)", 0.1, 0.4, 0.2, step=0.05)

    st.sidebar.subheader("안정성 판별")
    thr = st.sidebar.slider("불안정 임계값 (공급예비율 %)", 0.0, 30.0, 10.0, step=0.5,
                            help="공급예비율이 이 값보다 낮으면 '불안정(주의)'으로 판정합니다.")
    st.sidebar.caption("출처: 공공데이터포털 한국전력거래소 5분 전력수급")
    return source, api_url, n_rows, fill_method, remove_out, model_name, alpha, test_ratio, thr


def render_collect(source: str, api_url: str, n_rows: int):
    """탭 1. 데이터 수집. 불러온 원본 DataFrame(없으면 None)을 반환."""
    st.subheader("1. 데이터 수집")
    raw = None

    if source == SRC_DEMO:
        st.info("데모 모드: 코드 안에서 만든 모의 데이터로 앱 전체를 체험할 수 있습니다.")
        raw = generate_demo_data()

    elif source == SRC_CSV:
        up = st.file_uploader("공급예비율이 포함된 5분 전력수급 CSV 업로드", type=["csv"])
        if up is None:
            st.info("CSV를 올려주세요. 컬럼명이 다르면 코드 상단 COLUMN_MAP을 수정하세요.")
        else:
            try:
                raw = parse_csv(up.getvalue())
            except Exception as e:  # noqa: BLE001
                st.error(f"CSV 읽기 실패: {e}")

    else:
        key = get_service_key()
        if key is None:
            st.warning("API 키(SERVICE_KEY)가 없어 API를 쓸 수 없습니다. "
                       "Secrets에 키를 넣거나, CSV 업로드/데모 모드를 사용하세요.")
        if "확인" in api_url or "교체" in api_url:
            st.warning("사이드바의 'API 요청주소'를 활용신청 상세페이지의 주소로 바꿔주세요.")
        st.caption("※ 일부 공공 API는 해외 서버 접속을 막아 로컬에서는 되고 배포 환경에서는 실패할 수 있습니다.")
        if st.button("API에서 불러오기", type="primary"):
            if key is None:
                st.error("서비스키가 없어 호출하지 않았습니다.")
            else:
                with st.spinner("API 호출 중..."):
                    try:
                        st.session_state["api_df"] = fetch_api_data(key, api_url, n_rows)
                        st.success("불러오기 완료")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{e}  → CSV 업로드나 데모 모드로 대체하세요.")
        raw = st.session_state.get("api_df")
        if raw is None:
            st.info("'API에서 불러오기' 버튼을 눌러주세요.")

    if raw is None:
        return None

    raw = normalize_raw(raw)
    if "기준일시" not in raw.columns or raw["기준일시"].notna().sum() == 0:
        st.error("기준일시 컬럼을 찾지 못했습니다. 코드 상단 COLUMN_MAP을 확인하세요.")
        return None

    dmin, dmax = raw["기준일시"].min().date(), raw["기준일시"].max().date()
    sel = st.date_input("조회 기간", value=(dmin, dmax), min_value=dmin, max_value=dmax,
                        key=f"daterange_{source}_{dmin}_{dmax}")
    if isinstance(sel, (tuple, list)) and len(sel) == 2:
        start, end = sel
    else:
        start, end = dmin, dmax
    raw = raw[(raw["기준일시"] >= pd.Timestamp(start))
              & (raw["기준일시"] < pd.Timestamp(end) + pd.Timedelta(days=1))]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", f"{len(raw):,}")
    if len(raw) > 0:
        c2.metric("시작", str(raw["기준일시"].min())[:16])
        c3.metric("종료", str(raw["기준일시"].max())[:16])
    c4.metric("컬럼 수", raw.shape[1])
    st.dataframe(raw.head(10))
    miss = raw.isna().sum()
    st.caption("컬럼별 결측치: " + ", ".join(f"{k}={int(v)}" for k, v in miss.items()))
    return raw


def render_preprocess(step_df, outlier_df, df, fill_method, remove_out):
    """탭 2. 전처리 결과."""
    st.subheader("2. 전처리 결과")
    st.write(f"결측치: **{fill_method}**, 이상치 제거: **{'적용' if remove_out else '미적용'}**  (사이드바에서 변경)")
    st.markdown("**처리 단계별 행 수**")
    st.dataframe(step_df)
    st.markdown("**이상치 탐지 (IQR 1.5배)**")
    st.dataframe(outlier_df)
    st.caption("시뮬레이션이 아닌 실제 전력 데이터의 피크 구간은 정상 값일 수 있어 기본은 제거하지 않습니다. "
               "다만 선형회귀는 이상치에 민감하므로 확인합니다.")
    st.markdown("**파생변수가 추가된 데이터 (상위 10행)**")
    st.dataframe(df.head(10))


def render_eda(df: pd.DataFrame):
    """탭 3. 데이터 탐구."""
    st.subheader("3. 데이터 탐구 (EDA)")
    st.markdown("**기초통계**")
    num_cols = [c for c in NUM_COLS if c in df.columns]
    st.dataframe(df[num_cols].describe().T)
    insight("평균과 최소·최대의 차이, 공급예비율의 범위")

    st.markdown("**시계열 추이**")
    cols = [c for c in ["공급능력", "현재부하", "예측부하"] if c in df.columns]
    fig = px.line(df, x="기준일시", y=cols)
    fig.update_layout(legend_title_text="", yaxis_title="전력(MW 등)")
    show(fig, "eda_ts")
    insight("낮·저녁에 부하가 올라가고 주말에 내려가는지")

    c1, c2 = st.columns(2)
    with c1:
        by_hour = df.groupby("hour")[TARGET].mean().reset_index()
        fig = px.bar(by_hour, x="hour", y=TARGET, labels={"hour": "시각", TARGET: "평균 공급예비율(%)"})
        fig.update_layout(title="시간대별 평균 공급예비율")
        show(fig, "eda_hour")
        insight("예비율이 가장 낮은 시간대")
    with c2:
        by_wd = df.groupby("weekday")[TARGET].mean().reset_index()
        by_wd["요일"] = by_wd["weekday"].map(lambda i: WEEKDAY_NAMES[int(i)])
        fig = px.bar(by_wd, x="요일", y=TARGET, labels={TARGET: "평균 공급예비율(%)"})
        fig.update_layout(title="요일별 평균 공급예비율")
        show(fig, "eda_weekday")
        insight("평일과 주말의 차이")

    st.markdown("**상관관계**")
    corr_cols = [c for c in FEATURES + [TARGET] if df[c].std() > 0]
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    show(fig, "eda_corr")
    insight("공급예비율과 가장 강하게 움직이는 변수")
    rank = (corr[TARGET].drop(TARGET).rename("상관계수").to_frame()
            .assign(절댓값=lambda d: d["상관계수"].abs()).sort_values("절댓값", ascending=False))
    st.dataframe(rank.round(3))

    st.markdown("**변수 vs 공급예비율 산점도 + 추세선(OLS)**")
    feat = st.selectbox("변수 선택", FEATURES, key="eda_feat")
    sample = df.sample(min(len(df), 3000), random_state=0)
    fig = px.scatter(sample, x=feat, y=TARGET, opacity=0.4)
    if df[feat].std() > 0:
        slope, icpt = np.polyfit(df[feat], df[TARGET], 1)
        xs = np.array([df[feat].min(), df[feat].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines",
                                 line=dict(color="red", width=3), name="OLS 추세선"))
    show(fig, "eda_scatter")
    insight("점들이 직선 모양인지, 곡선 모양인지(곡선이면 선형회귀의 한계)")

    st.markdown("**다중공선성(VIF)**")
    st.dataframe(calc_vif(df, FEATURES))
    st.caption("VIF가 10 이상이면 다른 변수와 정보가 많이 겹칩니다. "
               "hour와 hour_sin/cos, weekday와 is_weekend는 의미가 겹쳐 높게 나올 수 있습니다.")
    insight("VIF가 큰 변수와 그 이유")


def render_train(df: pd.DataFrame, model_name: str, alpha: float, test_ratio: float):
    """탭 4. 모델 학습."""
    st.subheader("4. 모델 학습 (선형회귀)")
    if len(df) < MIN_ROWS:
        st.warning(f"학습에는 최소 {MIN_ROWS}행이 필요합니다. 현재 {len(df)}행입니다.")
        return
    st.info("타깃은 **공급예비율**입니다. 현재부하·공급예비력은 정답과 계산식으로 연결되어 "
            "입력에서 제외했습니다(데이터 누수 방지). 시간순으로 앞 구간으로 학습하고 뒤 구간으로 평가합니다.")
    st.write(f"선택 모델: **{model_name}**" + ("" if model_name == "LinearRegression" else f" (alpha={alpha})")
             + f" / 테스트 비율: {test_ratio:.0%}")

    if st.button("🚀 학습 시작", type="primary"):
        with st.spinner("학습 중..."):
            try:
                st.session_state["result"] = train_models(df, model_name, alpha, test_ratio)
            except Exception as e:  # noqa: BLE001
                st.error(f"학습 중 오류: {e}")

    res = get_valid_result(df)
    if res is None:
        st.info("아직 학습 결과가 없습니다. '학습 시작'을 눌러주세요. (데이터를 바꿨다면 다시 학습해야 합니다)")
        return

    st.success(f"학습된 모델: {res['name']} (학습 {res['train_size']:,}행 / 테스트 {res['test_size']:,}행)")
    m = res["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m['R²']:.3f}")
    c2.metric("RMSE", f"{m['RMSE']:.3f}")
    c3.metric("MAE", f"{m['MAE']:.3f}")
    c4.metric("시계열 CV R² 평균", "-" if np.isnan(res["cv_mean"]) else f"{res['cv_mean']:.3f}",
              None if np.isnan(res["cv_std"]) else f"±{res['cv_std']:.3f}", delta_color="off")

    st.markdown("**3개 모델 비교 (테스트 구간)**")
    st.dataframe(res["table"].round(4))
    insight("가장 성능이 좋은 모델과 차이가 큰지")

    test, pred = res["test"], res["pred"]
    actual = test[TARGET].to_numpy()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=actual, y=pred, opacity=0.4, labels={"x": "실제 공급예비율", "y": "예측 공급예비율"})
        lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="red", dash="dash"),
                                 name="y=x"))
        fig.update_layout(title="실제 vs 예측")
        show(fig, "tr_scatter")
        insight("점이 빨간 선 가까이에 모여 있는지")
    with c2:
        fig = px.histogram(x=actual - pred, nbins=40, labels={"x": "잔차(실제-예측)"})
        fig.update_layout(title="잔차 분포")
        show(fig, "tr_resid")
        insight("0 주변에 대칭으로 모여 있는지")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["기준일시"], y=actual, name="실제", mode="lines"))
    fig.add_trace(go.Scatter(x=test["기준일시"], y=pred, name="예측", mode="lines"))
    fig.update_layout(title="테스트 구간 실제 vs 예측 (시간순)", yaxis_title="공급예비율(%)")
    show(fig, "tr_line")
    insight("피크 시간대를 잘 따라가는지")

    st.markdown("**계수 해석** (표준화된 변수 기준: 다른 변수를 고정했을 때 1 표준편차 증가 시 예비율 변화)")
    coef = res["pipe"].named_steps["model"].coef_
    coef_df = pd.DataFrame({"변수": [FEATURE_LABELS[f] for f in FEATURES], "계수": coef})
    coef_df = coef_df.reindex(coef_df["계수"].abs().sort_values(ascending=False).index)
    coef_df["방향"] = np.where(coef_df["계수"] >= 0, "예비율 ↑", "예비율 ↓")
    fig = px.bar(coef_df, x="계수", y="변수", orientation="h", color="방향",
                 color_discrete_map={"예비율 ↑": "#2a9d8f", "예비율 ↓": "#e63946"})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    show(fig, "tr_coef")
    pos = coef_df[coef_df["계수"] > 1e-9].head(2)["변수"].tolist()
    neg = coef_df[coef_df["계수"] < -1e-9].head(2)["변수"].tolist()
    st.write(f"공급예비율을 **높이는** 변수: {', '.join(pos) if pos else '뚜렷한 변수 없음'} / "
             f"**낮추는** 변수: {', '.join(neg) if neg else '뚜렷한 변수 없음'}")
    insight("부하가 늘면 예비율이 줄어드는 방향인지")


def render_classify(df: pd.DataFrame, thr: float):
    """탭 5. 안정성 판별(분류 평가)."""
    st.subheader("5. 안정성 판별")
    res = get_valid_result(df)
    if res is None:
        st.info("먼저 '모델 학습' 탭에서 학습을 실행하세요.")
        return
    actual, pred = res["test"][TARGET].to_numpy(), res["pred"]
    cm = class_metrics(actual, pred, thr)
    n_unstable = int(cm["yt"].sum())
    ratio = n_unstable / len(actual) * 100
    st.write(f"임계값 **{thr}%** 미만을 불안정으로 판정 · 테스트 구간 실제 불안정: **{n_unstable}건 ({ratio:.2f}%)**")
    if n_unstable == 0:
        st.warning("테스트 구간에 불안정 샘플이 없습니다. 임계값을 높이거나 데이터 기간을 늘려보세요. "
                   "Precision/Recall/F1이 모두 0으로 나옵니다.")
    elif ratio < 1:
        st.warning("불안정 샘플이 1% 미만입니다. 클래스 불균형이라 Accuracy만 보면 안 됩니다. "
                   "Recall과 F1을 함께 보세요.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cm['Accuracy']:.3f}")
    c2.metric("Precision", f"{cm['Precision']:.3f}")
    c3.metric("Recall", f"{cm['Recall']:.3f}")
    c4.metric("F1", f"{cm['F1']:.3f}")

    mat = confusion_matrix(cm["yt"], cm["yp"], labels=[0, 1])
    fig = go.Figure(go.Heatmap(z=mat, x=["예측: 안정", "예측: 불안정"], y=["실제: 안정", "실제: 불안정"],
                               text=mat, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="혼동행렬")
    show(fig, "cls_cm")
    insight("불안정을 안정으로 놓친 건수(왼쪽 아래 칸)")

    ths = np.linspace(float(actual.min()), float(actual.max()), 50)
    rows = []
    for t in ths:
        mt = class_metrics(actual, pred, t)
        rows.append({"임계값": t, "Recall": mt["Recall"], "F1": mt["F1"]})
    curve = pd.DataFrame(rows).melt(id_vars="임계값", var_name="지표", value_name="값")
    fig = px.line(curve, x="임계값", y="값", color="지표")
    fig.add_vline(x=thr, line_dash="dash", line_color="red")
    fig.update_layout(title="임계값별 Recall / F1 (빨간 점선: 현재 임계값)")
    show(fig, "cls_curve")
    insight("임계값을 높이면 Recall이 어떻게 변하는지")

    st.info("불안정을 안정으로 잘못 판정하면(놓치면) 실제 전력 수급 위험을 알아채지 못합니다. "
            "그래서 이 프로젝트에서는 **Recall(불안정을 빠뜨리지 않는 비율)**을 특히 중요하게 봅니다.")


def load_random_sample(df: pd.DataFrame):
    """랜덤 시점을 슬라이더에 채우고 비교용 행을 저장 (버튼 콜백)."""
    row = df.sample(1).iloc[0]
    st.session_state["in_cap"] = float(row["공급능력"])
    st.session_state["in_load"] = float(row["예측부하"])
    st.session_state["in_hour"] = int(row["hour"])
    st.session_state["in_wd"] = WEEKDAY_NAMES[int(row["weekday"])]
    st.session_state["in_month"] = int(row["month"])
    st.session_state["rand_row"] = row[FEATURES + [TARGET, "기준일시"]].to_dict()


def make_gauge(pred: float, thr: float, data_max: float) -> go.Figure:
    """예측 공급예비율 게이지 (빨간 선 = 임계값)."""
    lo = min(0.0, pred - 2, thr - 2)
    hi = max(thr * 2, pred + 5, thr + 5, data_max)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pred, number={"suffix": " %"},
        title={"text": "예측 공급예비율"},
        gauge={"axis": {"range": [lo, hi]}, "bar": {"color": "#333333"},
               "steps": [{"range": [lo, thr], "color": "#f8b4b4"}, {"range": [thr, hi], "color": "#b7e4c7"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.8, "value": thr}},
    ))
    fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def judge(pred: float, thr: float):
    """예측값을 안정/불안정 메시지로 출력."""
    if pred < thr:
        st.error(f"🚨 불안정(주의): 예측 공급예비율 {pred:.2f}% < 임계값 {thr}%")
    else:
        st.success(f"✅ 안정: 예측 공급예비율 {pred:.2f}% ≥ 임계값 {thr}%")
    if abs(pred - thr) <= 2:
        st.warning("임계값과 2%p 이내로 가까운 경계 판정입니다. 결과를 조심해서 해석하세요.")


def render_predict(df: pd.DataFrame, thr: float):
    """탭 6. 예측."""
    st.subheader("6. 안정성 예측")
    res = get_valid_result(df)
    if res is None:
        st.info("먼저 '모델 학습' 탭에서 학습을 실행하세요.")
        return
    pipe = res["pipe"]

    def clamp(key, lo, hi):
        if key in st.session_state:
            st.session_state[key] = float(np.clip(st.session_state[key], lo, hi))

    cap_lo, cap_hi = float(df["공급능력"].min()), float(df["공급능력"].max())
    ld_lo, ld_hi = float(df["예측부하"].min()), float(df["예측부하"].max())
    cap_hi, ld_hi = max(cap_hi, cap_lo + 1.0), max(ld_hi, ld_lo + 1.0)
    clamp("in_cap", cap_lo, cap_hi)
    clamp("in_load", ld_lo, ld_hi)
    st.session_state.setdefault("in_cap", float(df["공급능력"].median()))
    st.session_state.setdefault("in_load", float(df["예측부하"].median()))
    st.session_state.setdefault("in_hour", 14)
    st.session_state.setdefault("in_wd", WEEKDAY_NAMES[0])
    st.session_state.setdefault("in_month", int(df["month"].mode().iloc[0]))

    st.button("🎲 랜덤 시점 불러오기", on_click=load_random_sample, args=(df,))
    c1, c2 = st.columns(2)
    with c1:
        cap = st.slider("공급능력", cap_lo, cap_hi, step=max((cap_hi - cap_lo) / 500, 0.01), key="in_cap",
                        help="발전소가 공급할 수 있는 전력 용량")
        load = st.slider("예측부하", ld_lo, ld_hi, step=max((ld_hi - ld_lo) / 500, 0.01), key="in_load",
                         help="전력 수요 예측값")
    with c2:
        hour = st.slider("시각(hour)", 0, 23, key="in_hour", help="하루 중 몇 시인지")
        wd_name = st.selectbox("요일", WEEKDAY_NAMES, key="in_wd")
        month = st.slider("월(month)", 1, 12, key="in_month", help="학습 데이터에 없는 월은 예측이 부정확할 수 있습니다.")

    wd = WEEKDAY_NAMES.index(wd_name)
    row = pd.DataFrame([{
        "공급능력": cap, "예측부하": load, "hour": hour, "weekday": wd, "month": month,
        "is_weekend": int(wd >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
    }])[FEATURES]
    pred = float(pipe.predict(row)[0])

    show(make_gauge(pred, thr, float(df[TARGET].max())), "pred_gauge")
    judge(pred, thr)

    rr = st.session_state.get("rand_row")
    if rr is not None:
        st.markdown("**랜덤 시점: 실제 vs 예측**")
        r = pd.DataFrame([{f: rr[f] for f in FEATURES}])[FEATURES]
        p = float(pipe.predict(r)[0])
        a = float(rr[TARGET])
        st.write(f"시점: {str(rr['기준일시'])[:16]}")
        k1, k2, k3 = st.columns(3)
        k1.metric("실제 공급예비율", f"{a:.2f}%")
        k2.metric("예측 공급예비율", f"{p:.2f}%", f"{p - a:+.2f}%p", delta_color="off")
        k3.metric("실제 / 예측 판정", f"{'불안정' if a < thr else '안정'} / {'불안정' if p < thr else '안정'}")


# ===== 7. 앱 실행 =====
def main():
    st.title("⚡ 스마트 그리드 전력망 안정성 판별")
    st.caption("한국전력거래소 5분 전력수급 데이터로 공급예비율을 선형회귀로 예측하고, 임계값 기준으로 안정/불안정을 판별합니다.")

    (source, api_url, n_rows, fill_method, remove_out,
     model_name, alpha, test_ratio, thr) = render_sidebar()

    if source == SRC_DEMO:
        st.warning("⚠️ **모의 데이터입니다. 실제 분석이 아닙니다.** (발표·제출용 결과는 실제 데이터로 다시 실행하세요)")

    tabs = st.tabs(["1 데이터 수집", "2 전처리", "3 데이터 탐구", "4 모델 학습", "5 안정성 판별", "6 실시간 예측"])
    t1, t2, t3, t4, t5, t6 = tabs

    with t1:
        raw = render_collect(source, api_url, n_rows)

    df, step_df, outlier_df, err = None, None, None, None
    if raw is not None and len(raw) > 0:
        df, step_df, outlier_df, err = preprocess(raw, fill_method, remove_out)
    elif raw is not None:
        err = "선택한 기간에 데이터가 없습니다. 조회 기간을 바꿔주세요."

    ready = df is not None
    wait_msg = err or "먼저 '데이터 수집' 탭에서 데이터를 불러오세요."

    with t2:
        if ready:
            render_preprocess(step_df, outlier_df, df, fill_method, remove_out)
        else:
            st.info(wait_msg)
    with t3:
        if ready:
            render_eda(df)
        else:
            st.info(wait_msg)
    with t4:
        if ready:
            render_train(df, model_name, alpha, test_ratio)
        else:
            st.info(wait_msg)
    with t5:
        if ready:
            render_classify(df, thr)
        else:
            st.info(wait_msg)
    with t6:
        if ready:
            render_predict(df, thr)
        else:
            st.info(wait_msg)


main()
