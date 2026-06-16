import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


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
POTENTIAL_DISPLAY_ORDER = ["Low", "High"]


def get_category_order(column: str) -> list[str]:
	if column == "Potential":
		return POTENTIAL_DISPLAY_ORDER
	return CATEGORY_DISPLAY_ORDER


def get_date_topic_label(date_text: str) -> str:
	topic = TOPIC_LABELS.get(date_text)
	if topic:
		return f"{date_text} | {topic}"
	return date_text


@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
	try:
		df = pd.read_csv(file_path, encoding="utf-8-sig")
	except UnicodeDecodeError:
		df = pd.read_csv(file_path, encoding="cp949")

	df.columns = [col.strip() for col in df.columns]

	for col in REQUIRED_COLUMNS:
		if col not in df.columns:
			df[col] = None

	df = df[REQUIRED_COLUMNS].copy()

	df["TMSSR"] = df["TMSSR"].astype("string").str.strip()
	df["Potential"] = df["Potential"].astype("string").str.strip()

	df["TMSSR"] = df["TMSSR"].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
	df["Potential"] = df["Potential"].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

	return df


def safe_load(file_name: str) -> pd.DataFrame:
	file_path = DATA_DIR / file_name
	if not file_path.exists():
		st.warning(f"파일을 찾을 수 없습니다: `{file_path}`")
		return pd.DataFrame(columns=REQUIRED_COLUMNS)
	return load_csv(file_path)


def load_merged_by_date() -> dict[str, pd.DataFrame]:
	merged_by_date: dict[str, pd.DataFrame] = {}

	for date_label, file_info in DATE_CONFIG.items():
		main_df = safe_load(file_info["main"])
		other_df = safe_load(file_info["other"])

		merged_df = pd.concat([main_df, other_df], ignore_index=True)
		merged_by_date[date_label] = merged_df

	return merged_by_date


def load_all_merged_data(merged_by_date: dict[str, pd.DataFrame]) -> pd.DataFrame:
	if not merged_by_date:
		return pd.DataFrame(columns=REQUIRED_COLUMNS)

	return pd.concat(list(merged_by_date.values()), ignore_index=True)


def get_summary(df: pd.DataFrame) -> dict:
	if df.empty:
		return {
			"전체 행 수": 0,
			"서로 다른 날짜/시간 수": 0,
			"교사 발화수": 0,
			"사용자가 입력한 교사 발화 수": 0,
		}

	teacher_mask = df["화자"].astype("string").str.strip().eq("교사")
	tmssr_mask = df["TMSSR"].notna()

	return {
		"전체 행 수": len(df),
		"서로 다른 날짜/시간 수": df["날짜/시간"].nunique(dropna=True),
		"교사 발화수": int(teacher_mask.sum()),
		"사용자가 입력한 교사 발화 수": int((teacher_mask & tmssr_mask).sum()),
	}


def get_distribution(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
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

	count_df.columns = ["범주", "빈도"]

	total = count_df["빈도"].sum()
	count_df["비율"] = count_df["빈도"] / total * 100
	count_df["자료"] = label

	return count_df


def make_distribution_table(merged_by_date: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
	frames = []
	for date_label, df in merged_by_date.items():
		dist = get_distribution(df, column, date_label)
		if not dist.empty:
			frames.append(dist)

	if not frames:
		return pd.DataFrame()

	combined = pd.concat(frames, ignore_index=True)
	combined["빈도(비율)"] = combined.apply(
		lambda row: f"{int(row['빈도'])} ({row['비율']:.1f}%)",
		axis=1,
	)

	pivot = combined.pivot_table(
		index="범주",
		columns="자료",
		values="빈도(비율)",
		aggfunc="first",
		fill_value="0 (0.0%)",
	).reset_index()

	for date_label in DATE_CONFIG.keys():
		if date_label not in pivot.columns:
			pivot[date_label] = "0 (0.0%)"

	category_order = get_category_order(column)
	order_map = {name: idx for idx, name in enumerate(category_order)}
	pivot["정렬순서"] = pivot["범주"].map(order_map).fillna(len(category_order))
	pivot = pivot.sort_values("정렬순서")

	ordered_columns = ["범주"] + list(DATE_CONFIG.keys())
	return pivot[ordered_columns]


def make_bar_chart(dist_df: pd.DataFrame, title: str, column: str):
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

	date_order = dist_df["자료"].dropna().drop_duplicates().tolist()
	x_labels = [get_date_topic_label(date_label) if date_label in DATE_CONFIG else date_label for date_label in date_order]

	percent_pivot = percent_pivot.reindex(date_order, fill_value=0)
	count_pivot = count_pivot.reindex(date_order, fill_value=0)

	fig = go.Figure()

	for category in ordered_categories:
		y_values = [
			float(percent_pivot.at[date_label, category]) if category in percent_pivot.columns else 0.0
			for date_label in date_order
		]
		count_values = [
			int(count_pivot.at[date_label, category]) if category in count_pivot.columns else 0
			for date_label in date_order
		]

		fig.add_trace(
			go.Bar(
				name=category,
				x=x_labels,
				y=y_values,
				customdata=count_values,
				hovertemplate=(
					"날짜: %{x}<br>"
					"범주: " + str(category) + "<br>"
					"비율: %{y:.1f}%<br>"
					"빈도: %{customdata}<extra></extra>"
				),
			)
		)

	fig.update_layout(
		title=title,
		barmode="stack",
		height=420,
		yaxis=dict(title="비율(%)", range=[0, 100], ticksuffix="%"),
		xaxis=dict(title="날짜"),
		legend=dict(title="범주"),
		margin=dict(l=20, r=20, t=60, b=20),
	)

	st.plotly_chart(fig, use_container_width=True)


def render_distribution_section(df: pd.DataFrame, column: str, title: str, label: str):
	st.subheader(title)

	dist_df = get_distribution(df, column, label)
	table_df = make_distribution_table({label: df}, column)

	left, right = st.columns([1, 1.4])

	with left:
		st.markdown("#### 빈도표")
		if table_df.empty:
			st.info("표시할 데이터가 없습니다.")
		else:
			st.dataframe(table_df, use_container_width=True, hide_index=True)

	with right:
		st.markdown("#### 누적 비율 막대 그래프")
		make_bar_chart(dist_df, title, column)


def make_tmssr_potential_table(df: pd.DataFrame) -> pd.DataFrame:
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
			textfont={"size": 14},
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


def render_summary_metrics_by_date(merged_by_date: dict[str, pd.DataFrame]):
	st.subheader("1. 데이터 수 비교")

	cols = st.columns(len(merged_by_date))
	for idx, date_label in enumerate(DATE_CONFIG.keys()):
		summary = get_summary(merged_by_date[date_label])
		with cols[idx]:
			st.markdown(f"#### {get_date_topic_label(date_label)}")
			m1, m2 = st.columns(2)
			with m1:
				st.metric("사용된 데이터 수", summary["서로 다른 날짜/시간 수"])
			with m2:
				st.metric("교사 발화 수", summary["사용자가 입력한 교사 발화 수"])


def render_summary_metrics(df: pd.DataFrame, title: str):
	st.subheader("1. 데이터 수 비교")
	st.markdown(f"#### {title}")
	summary = get_summary(df)
	col1, col2 = st.columns(2)
	with col1:
		st.metric("사용된 데이터 수", summary["서로 다른 날짜/시간 수"])
	with col2:
		st.metric("교사 발화 수", summary["사용자가 입력한 교사 발화 수"])


def render_distribution_section_by_date(merged_by_date: dict[str, pd.DataFrame], column: str, title: str):
	st.subheader(title)

	dist_frames = []
	for date_label, df in merged_by_date.items():
		dist = get_distribution(df, column, date_label)
		if not dist.empty:
			dist_frames.append(dist)

	chart_df = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
	table_df = make_distribution_table(merged_by_date, column)

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


def render_tmssr_potential_section_by_date(merged_by_date: dict[str, pd.DataFrame]):
	st.subheader("4. TMSSR x Potential 분포 비교")

	cols = st.columns(len(merged_by_date))
	for idx, date_label in enumerate(DATE_CONFIG.keys()):
		with cols[idx]:
			st.markdown(f"#### {get_date_topic_label(date_label)}")
			make_tmssr_potential_heatmap(merged_by_date[date_label], date_label)


def render_tmssr_potential_section(df: pd.DataFrame, title: str):
	st.subheader("4. TMSSR x Potential 분포 비교")
	st.markdown(f"#### {title}")
	make_tmssr_potential_heatmap(df, title)


def render_raw_data_by_date(merged_by_date: dict[str, pd.DataFrame]):
	with st.expander("원자료 보기"):
		tabs = st.tabs([get_date_topic_label(date_label) for date_label in DATE_CONFIG.keys()])

		for idx, date_label in enumerate(DATE_CONFIG.keys()):
			with tabs[idx]:
				st.dataframe(merged_by_date[date_label], use_container_width=True, height=420)


def render_overall_page(df: pd.DataFrame, show_raw_data: bool):
	render_summary_metrics(df, "전체 | 3개 날짜 합계")

	st.divider()
	render_distribution_section(df, column="TMSSR", title="2. TMSSR 범주 분포", label="전체 | 3개 날짜 합계")

	st.divider()
	render_distribution_section(df, column="Potential", title="3. Potential 범주 분포", label="전체 | 3개 날짜 합계")

	st.divider()
	render_tmssr_potential_section(df, "전체 | 3개 날짜 합계")

	if show_raw_data:
		st.divider()
		with st.expander("원자료 보기"):
			st.dataframe(df, use_container_width=True, height=420)


st.title("날짜별 TMSSR 데이터 비교 대시보드")
st.caption("main / other를 합쳐 날짜별로 데이터 수와 분포를 비교합니다.")

with st.sidebar:
	st.header("설정")
	show_raw_data = st.checkbox("원자료 보기", value=False)

	st.divider()

	st.markdown("### 병합 방식")
	st.info("각 날짜에서 main + other를 합친 뒤 날짜 간 비교를 수행합니다.")


merged_data = load_merged_by_date()
all_merged_data = load_all_merged_data(merged_data)

tab_labels = ["기존 내용(날짜별 비교)", "전체(3개 날짜 합계)"]
tabs = st.tabs(tab_labels)


with tabs[0]:
	render_summary_metrics_by_date(merged_data)

	st.divider()
	render_distribution_section_by_date(
		merged_by_date=merged_data,
		column="TMSSR",
		title="2. TMSSR 범주 분포",
	)

	st.divider()
	render_distribution_section_by_date(
		merged_by_date=merged_data,
		column="Potential",
		title="3. Potential 범주 분포",
	)

	st.divider()
	render_tmssr_potential_section_by_date(merged_data)

	if show_raw_data:
		st.divider()
		render_raw_data_by_date(merged_data)

with tabs[1]:
	render_overall_page(all_merged_data, show_raw_data)
