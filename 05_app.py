"""銅箔製程品質分析 Streamlit 應用程式。"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "copper_foil_prepared.csv"
MODEL_PATH = APP_DIR / "random_forest_anomaly.joblib"

PROCESS_LABELS = {
    "Line_Speed_mpm": "線速 (m/min)",
    "Current_ASD": "電流密度 (A/dm²)",
    "Bath_Temp_C": "槽液溫度 (°C)",
    "Cu_Conc_gL": "銅離子濃度 (g/L)",
    "Additive_ppm": "添加劑濃度 (ppm)",
    "Jet_Pressure_bar": "噴流壓力 (bar)",
    "Edge_Current_Bias": "邊緣電流偏差",
}


st.set_page_config(
    page_title="銅箔製程品質分析",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_data(ttl="15m", max_entries=2)
def load_data(csv_path: str) -> pd.DataFrame:
    """讀取並整理用於視覺化和表單範圍的參考資料。"""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Timestamp" in data:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    return data


@st.cache_resource
def load_model_bundle(model_path: str) -> dict:
    """載入已訓練模型及其輸入欄位、分類門檻。"""
    return joblib.load(model_path)


def get_data() -> pd.DataFrame:
    """載入資料，並在檔案不可用時顯示清楚的錯誤訊息。"""
    if not DATA_PATH.exists():
        st.error(f"找不到資料檔：{DATA_PATH}", icon=":material/error:")
        st.stop()
    try:
        return load_data(str(DATA_PATH))
    except Exception as exc:
        st.error(f"讀取資料失敗：{exc}", icon=":material/error:")
        st.stop()


def data_overview_page() -> None:
    """第一頁：檢視資料概況與主要製程特徵。"""
    st.title("資料視覺化")
    st.caption("銅箔厚度、製程參數與品質標籤概況。來源資料為教學用合成資料。")

    data = get_data()
    required = {"Timestamp", "Thick_Mean_um", "Is_Anomaly", "Defect_Type"}
    missing = sorted(required.difference(data.columns))
    if missing:
        st.error(f"資料檔缺少必要欄位：{', '.join(missing)}")
        st.stop()

    row_count = len(data)
    anomaly_count = int(data["Is_Anomaly"].fillna(0).sum())
    anomaly_rate = anomaly_count / row_count if row_count else 0.0
    mean_thickness = data["Thick_Mean_um"].mean()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("樣本數", f"{row_count:,}", border=True)
    kpi_cols[1].metric("異常樣本", f"{anomaly_count:,}", border=True)
    kpi_cols[2].metric("異常比例", f"{anomaly_rate:.1%}", border=True)
    kpi_cols[3].metric("平均厚度", f"{mean_thickness:.2f} µm", border=True)

    st.subheader("厚度與製程參數趨勢")
    process_features = [column for column in PROCESS_LABELS if column in data.columns]
    if not process_features:
        st.warning("找不到製程參數欄位。")
    else:
        selected_feature = st.selectbox(
            "選擇製程參數",
            process_features,
            format_func=lambda value: PROCESS_LABELS[value],
        )
        chart_cols = st.columns(2)
        trend_data = data.sort_values("Timestamp").set_index("Timestamp")
        with chart_cols[0]:
            st.line_chart(
                trend_data[["Thick_Mean_um"]],
                y_label="平均厚度 (µm)",
                alt="銅箔平均厚度隨量測時間的變化",
            )
        with chart_cols[1]:
            st.line_chart(
                trend_data[[selected_feature]],
                y_label=PROCESS_LABELS[selected_feature],
                alt=f"{PROCESS_LABELS[selected_feature]}隨量測時間的變化",
            )

    st.subheader("品質類別與製程關係")
    chart_cols = st.columns(2)
    defect_counts = (
        data["Defect_Type"]
        .fillna("未標記")
        .value_counts()
        .rename_axis("缺陷類別")
        .reset_index(name="樣本數")
    )
    with chart_cols[0]:
        st.bar_chart(
            defect_counts,
            x="缺陷類別",
            y="樣本數",
            alt="各缺陷類別的樣本數比較",
        )
    with chart_cols[1]:
        if {"Current_ASD", "Thick_Mean_um"}.issubset(data.columns):
            scatter_data = data[["Current_ASD", "Thick_Mean_um", "Defect_Type"]].dropna()
            st.scatter_chart(
                scatter_data,
                x="Current_ASD",
                y="Thick_Mean_um",
                color="Defect_Type",
                x_label="電流密度 (A/dm²)",
                y_label="平均厚度 (µm)",
                alt="電流密度與平均厚度的關係，依缺陷類別著色",
            )

    st.caption("資料中的異常與缺陷標籤用於探索性分析；模型預測頁只會使用製程輸入特徵。")


def anomaly_prediction_page() -> None:
    """第二頁：輸入製程條件並使用已訓練模型預測異常。"""
    st.title("異常預測")
    st.caption("輸入目前製程條件，使用 Random Forest 模型預測異常機率。")

    data = get_data()
    if not MODEL_PATH.exists():
        st.error(f"找不到模型檔：{MODEL_PATH}", icon=":material/error:")
        st.stop()

    try:
        bundle = load_model_bundle(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"載入模型失敗：{exc}", icon=":material/error:")
        st.stop()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = float(bundle["decision_threshold"])
    numeric_features = [column for column in PROCESS_LABELS if column in feature_columns]
    missing_features = [column for column in feature_columns if column not in data.columns]
    if missing_features:
        st.error(f"參考資料缺少模型輸入欄位：{', '.join(missing_features)}")
        st.stop()
    if "Shift" not in data.columns:
        st.error("資料檔缺少 Shift 班別欄位，無法建立模型輸入。")
        st.stop()

    st.info(
        f"模型分類門檻為 {threshold:.0%}。異常機率是模型分數，並非校準後的實際風險百分比。",
        icon=":material/info:",
    )

    input_values: dict[str, float] = {}
    with st.form("process_input_form", border=True):
        st.subheader("製程輸入")
        columns = st.columns(2)
        for index, feature in enumerate(numeric_features):
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            if values.empty:
                st.error(f"欄位 {feature} 沒有可用的參考數值。")
                st.stop()

            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())
            step = max((maximum - minimum) / 100, 0.001)
            precision = 3 if feature == "Edge_Current_Bias" else 2
            label = PROCESS_LABELS.get(feature, feature)
            with columns[index % 2]:
                input_values[feature] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    format=f"%.{precision}f",
                    help=f"參考範圍：{minimum:.3f} 至 {maximum:.3f}",
                    key=f"input_{feature}",
                )

        shifts = sorted(data["Shift"].dropna().astype(str).unique().tolist())
        if not shifts:
            st.error("找不到可用班別。")
            st.stop()
        selected_shift = st.selectbox("班別", shifts, key="input_shift")
        submitted = st.form_submit_button(
            "執行異常預測",
            type="primary",
            icon=":material/search:",
            width="stretch",
        )

    if submitted:
        for feature in feature_columns:
            if feature.startswith("Shift_"):
                shift_name = feature.removeprefix("Shift_")
                input_values[feature] = float(selected_shift == shift_name)

        model_input = pd.DataFrame([input_values], columns=feature_columns).astype(float)
        anomaly_probability = float(model.predict_proba(model_input)[0, 1])
        is_anomaly = anomaly_probability >= threshold

        if is_anomaly:
            st.error("模型預測結果：異常", icon=":material/warning:")
        else:
            st.success("模型預測結果：正常", icon=":material/check_circle:")

        result_cols = st.columns(2)
        result_cols[0].metric("異常機率（模型分數）", f"{anomaly_probability:.1%}", border=True)
        result_cols[1].metric("分類門檻", f"{threshold:.1%}", border=True)
        st.progress(anomaly_probability, text="模型異常分數")
        with st.expander("查看送入模型的特徵"):
            st.dataframe(model_input, hide_index=True, alt="本次異常預測的模型輸入特徵")


with st.sidebar:
    st.markdown("### 銅箔製程品質分析")
    st.caption("資料探索與 Random Forest 異常預測")

current_page = st.navigation(
    [
        st.Page(data_overview_page, title="資料視覺化", icon=":material/analytics:", default=True),
        st.Page(anomaly_prediction_page, title="異常預測", icon=":material/model_training:"),
    ],
    position="sidebar",
)
current_page.run()
