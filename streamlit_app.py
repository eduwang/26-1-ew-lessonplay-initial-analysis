import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="TMSSR 데이터 비교 대시보드",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("data")

DATE_CONFIG = {
    "2026년 4월 15일": {
        "main": "260415_main.csv",
        "other": "260415_other.csv",
    },
    "2026년 4월 22일": {
        "main": "260422_main.csv",
        "other": "260422_other.csv",
    },
    "2026년 4월 29일": {
        "main": "260429_main.csv",
        "other": "260429_other.csv",
    },
}

TOPIC_LABELS = {
    "2026년 4월 15일": "등차수열 일반항 구하기",
    "2026년 4월 22일": "수열을 함수로 이해하기",
    "2026년 4월 29일": "이차부등식",
}

REQUIRED_COLUMNS = ["날짜/시간", "화자", "메시지", "TMSSR", "Potential"]
CATEGORY_DISPLAY_ORDER = ["Eliciting", "Responding", "Facilitating", "Extending"]
POTENTIAL_DISPLAY_ORDER = ["High", "Low"]


def get_category_order(column: str) -> list[str]:
    """
    컬럼별 범주 표시 순서 반환.
    """
    if column == "Potential":
        return POTENTIAL_DISPLAY_ORDER
    return CATEGORY_DISPLAY_ORDER


def get_date_topic_label(date_text: str) -> str:
    """
    날짜와 주제를 함께 표시할 라벨 생성.
    """
    topic = TOPIC_LABELS.get(date_text)
    if topic:
        return f"{date_text} | {topic}"
    return date_text


# =========================
# 데이터 불러오기 함수
# =========================

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    """
    CSV 파일을 불러오는 함수.
    UTF-8-SIG를 기본으로 사용하되, 실패하면 cp949도 시도.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    df.columns = [col.strip() for col in df.columns]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_COLUMNS].copy()

    # 분석 범주 컬럼 정리
    df["TMSSR"] = df["TMSSR"].astype("string").str.strip()
    df["Potential"] = df["Potential"].astype("string").str.strip()

    df["TMSSR"] = df["TMSSR"].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    df["Potential"] = df["Potential"].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    return df


def safe_load(date_label: str, group_name: str) -> pd.DataFrame:
    """
    data 폴더에서 파일을 안전하게 불러옴.
    파일이 없으면 빈 데이터프레임 반환.
    """
    file_name = DATE_CONFIG[date_label][group_name]
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        st.warning(f"파일을 찾을 수 없습니다: `{file_path}`")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return load_csv(file_path)


# =========================
# 분석 함수
# =========================

def get_summary(df: pd.DataFrame) -> dict:
    """
    한 데이터프레임에 대한 기본 요약 정보.
    """
    if df.empty:
        return {
            "전체 행 수": 0,
            "서로 다른 날짜/시간 수": 0,
            "TMSSR 코딩 수": 0,
            "Potential 코딩 수": 0,
        }

    return {
        "전체 행 수": len(df),
        "서로 다른 날짜/시간 수": df["날짜/시간"].nunique(dropna=True),
        "TMSSR 코딩 수": df["TMSSR"].dropna().shape[0],
        "Potential 코딩 수": df["Potential"].dropna().shape[0],
    }


def get_distribution(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """
    특정 범주 컬럼의 분포를 계산.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=["범주", "빈도", "비율", "자료"])

    valid = df[column].dropna()

    if valid.empty:
        return pd.DataFrame(columns=["범주", "빈도", "비율", "자료"])

    count_df = (
        valid.value_counts()
        .reset_index()
        .rename(columns={"index": "범주", column: "빈도"})
    )

    # pandas 버전에 따라 컬럼명이 다르게 나올 수 있어 보정
    count_df.columns = ["범주", "빈도"]

    total = count_df["빈도"].sum()
    count_df["비율"] = count_df["빈도"] / total * 100
    count_df["자료"] = label

    return count_df


def make_distribution_table(main_df: pd.DataFrame, other_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    main과 other의 분포를 하나의 표로 병합.
    """
    main_dist = get_distribution(main_df, column, "main")
    other_dist = get_distribution(other_df, column, "other")

    combined = pd.concat([main_dist, other_dist], ignore_index=True)

    if combined.empty:
        return combined

    pivot = combined.pivot_table(
        index="범주",
        columns="자료",
        values="빈도",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    if "main" not in pivot.columns:
        pivot["main"] = 0
    if "other" not in pivot.columns:
        pivot["other"] = 0

    main_total = pivot["main"].sum()
    other_total = pivot["other"].sum()

    if main_total == 0:
        pivot["main 비율(%)"] = 0.0
    else:
        pivot["main 비율(%)"] = (pivot["main"] / main_total * 100).round(1)

    if other_total == 0:
        pivot["other 비율(%)"] = 0.0
    else:
        pivot["other 비율(%)"] = (pivot["other"] / other_total * 100).round(1)

    pivot["정렬값"] = pivot["main"] + pivot["other"]
    category_order = get_category_order(column)
    order_map = {name: idx for idx, name in enumerate(category_order)}
    pivot["정렬순서"] = pivot["범주"].map(order_map).fillna(len(category_order))
    pivot = pivot.sort_values(["정렬순서", "정렬값"], ascending=[True, False])

    return pivot[["범주", "main", "main 비율(%)", "other", "other 비율(%)"]]


def make_tmssr_potential_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    TMSSR(행) x Potential(열) 교차 빈도표 생성.
    """
    if df.empty:
        return pd.DataFrame()

    valid = df[["TMSSR", "Potential"]].dropna()
    if valid.empty:
        return pd.DataFrame()

    crosstab = pd.crosstab(valid["TMSSR"], valid["Potential"])

    tmssr_values = crosstab.index.tolist()
    potential_values = crosstab.columns.tolist()

    tmssr_order = CATEGORY_DISPLAY_ORDER + [
        value for value in tmssr_values if value not in CATEGORY_DISPLAY_ORDER
    ]
    potential_order = POTENTIAL_DISPLAY_ORDER + [
        value for value in potential_values if value not in POTENTIAL_DISPLAY_ORDER
    ]

    crosstab = crosstab.reindex(index=tmssr_order, columns=potential_order, fill_value=0)
    crosstab.index.name = "TMSSR"

    return crosstab.reset_index()


def make_tmssr_potential_heatmap(df: pd.DataFrame, title: str):
    """
    TMSSR(행) x Potential(열) 히트맵 생성.
    셀에는 빈도와 전체 대비 비율(%)을 함께 표시.
    """
    matrix_df = make_tmssr_potential_table(df)

    if matrix_df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    matrix_df = matrix_df.set_index("TMSSR")
    z = matrix_df.values
    total = z.sum()

    if total == 0:
        ratio = z.astype(float)
    else:
        ratio = z / total * 100

    text = [
        [f"{int(z[r][c])}<br>({ratio[r][c]:.1f}%)" for c in range(z.shape[1])]
        for r in range(z.shape[0])
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorbar={"title": "빈도"},
            hovertemplate=(
                "TMSSR: %{y}<br>"
                "Potential: %{x}<br>"
                "빈도: %{z}<br>"
                "전체 대비 비율: %{customdata:.1f}%<extra></extra>"
            ),
            customdata=ratio,
        )
    )

    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="Potential"),
        yaxis=dict(title="TMSSR"),
    )

    st.plotly_chart(fig, use_container_width=True)


def make_bar_chart(dist_df: pd.DataFrame, title: str, column: str):
    """
    main/other의 범주 비율을 Plotly 누적 막대그래프로 표시.
    """
    if dist_df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    category_order = get_category_order(column)
    existing_categories = dist_df["범주"].dropna().unique().tolist()
    ordered_categories = category_order + [
        category for category in existing_categories if category not in category_order
    ]

    percent_pivot = dist_df.pivot_table(
        index="자료",
        columns="범주",
        values="비율",
        aggfunc="sum",
        fill_value=0,
    )
    count_pivot = dist_df.pivot_table(
        index="자료",
        columns="범주",
        values="빈도",
        aggfunc="sum",
        fill_value=0,
    )

    group_order = ["main", "other"]
    group_labels = ["Main", "Other"]

    percent_pivot = percent_pivot.reindex(group_order, fill_value=0)
    count_pivot = count_pivot.reindex(group_order, fill_value=0)

    fig = go.Figure()

    for category in ordered_categories:
        y_values = [
            float(percent_pivot.at[group, category]) if category in percent_pivot.columns else 0.0
            for group in group_order
        ]
        count_values = [
            int(count_pivot.at[group, category]) if category in count_pivot.columns else 0
            for group in group_order
        ]

        fig.add_trace(
            go.Bar(
                name=category,
                x=group_labels,
                y=y_values,
                customdata=count_values,
                hovertemplate=(
                    "자료: %{x}<br>"
                    "범주: " + str(category) + "<br>"
                    "비율: %{y:.1f}%<br>"
                    "빈도: %{customdata}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        height=360,
        yaxis=dict(title="비율(%)", range=[0, 100], ticksuffix="%"),
        xaxis=dict(title="자료"),
        legend=dict(title="범주"),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_summary_metrics(main_df: pd.DataFrame, other_df: pd.DataFrame):
    """
    main과 other의 요약 지표를 가로로 비교.
    """
    main_summary = get_summary(main_df)
    other_summary = get_summary(other_df)

    st.subheader("1. 데이터 수 비교")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### main")
        main_m1, main_m2 = st.columns(2)
        with main_m1:
            st.metric("사용된 데이터 수", main_summary["서로 다른 날짜/시간 수"])
        with main_m2:
            st.metric("전체 행 수", main_summary["전체 행 수"])

    with col2:
        st.markdown("#### other")
        other_m1, other_m2 = st.columns(2)
        with other_m1:
            st.metric("사용된 데이터 수", other_summary["서로 다른 날짜/시간 수"])
        with other_m2:
            st.metric("전체 행 수", other_summary["전체 행 수"])


def render_distribution_section(main_df: pd.DataFrame, other_df: pd.DataFrame, column: str, title: str):
    """
    TMSSR 또는 Potential 분포 비교 섹션.
    """
    st.subheader(title)

    main_dist = get_distribution(main_df, column, "main")
    other_dist = get_distribution(other_df, column, "other")
    chart_df = pd.concat([main_dist, other_dist], ignore_index=True)

    table_df = make_distribution_table(main_df, other_df, column)

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("#### 빈도표")
        if table_df.empty:
            st.info("표시할 데이터가 없습니다.")
        else:
            st.dataframe(table_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 누적 비율 막대 그래프")
        make_bar_chart(chart_df, title, column)


def render_tmssr_potential_section(main_df: pd.DataFrame, other_df: pd.DataFrame):
    """
    TMSSR x Potential 교차 분포 비교 섹션.
    """
    st.subheader("4. TMSSR x Potential 분포 비교")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### main")
        make_tmssr_potential_heatmap(main_df, "main")

    with col2:
        st.markdown("#### other")
        make_tmssr_potential_heatmap(other_df, "other")


def render_raw_data(main_df: pd.DataFrame, other_df: pd.DataFrame):
    """
    원자료 확인용 섹션.
    """
    with st.expander("원자료 보기"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### main 원자료")
            st.dataframe(main_df, use_container_width=True, height=400)

        with col2:
            st.markdown("#### other 원자료")
            st.dataframe(other_df, use_container_width=True, height=400)


# =========================
# 화면 구성
# =========================

st.title("TMSSR 데이터 비교 대시보드")
st.caption("4월 15일, 4월 22일, 4월 29일의 main / other 데이터를 날짜별로 비교합니다.")

with st.sidebar:
    st.header("설정")

    selected_date = st.radio(
        "비교할 날짜 선택",
        list(DATE_CONFIG.keys()),
        index=0,
        format_func=get_date_topic_label,
    )

    st.divider()

    st.markdown("### 필요한 파일")
    st.code(
        "\n".join(
            [
                "data/260415_main.csv",
                "data/260415_other.csv",
                "data/260422_main.csv",
                "data/260422_other.csv",
                "data/260429_main.csv",
                "data/260429_other.csv",
            ]
        ),
        language="text",
    )

    st.divider()

    show_raw_data = st.checkbox("원자료 보기", value=False)


main_df = safe_load(selected_date, "main")
other_df = safe_load(selected_date, "other")

st.header(get_date_topic_label(selected_date))

render_summary_metrics(main_df, other_df)

st.divider()

render_distribution_section(
    main_df=main_df,
    other_df=other_df,
    column="TMSSR",
    title="2. TMSSR 범주 분포",
)

st.divider()

render_distribution_section(
    main_df=main_df,
    other_df=other_df,
    column="Potential",
    title="3. Potential 범주 분포",
)

st.divider()

render_tmssr_potential_section(main_df, other_df)

if show_raw_data:
    st.divider()
    render_raw_data(main_df, other_df)