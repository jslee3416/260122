import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="서울 기온 역대 비교기", layout="wide")

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='cp949', skiprows=7)
    df.columns = [col.strip() for col in df.columns]
    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())
    for col in ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

st.title("🌡️ 서울 기온 역대 비교 분석기")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader("추가 기온 데이터 업로드 (CSV)", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("ta_20260122174530.csv")
    except:
        st.error("데이터 파일을 찾을 수 없습니다.")
        st.stop()

# 날짜 선택 (기본값: 최신 데이터)
max_date = df['날짜'].max()
target_date = st.sidebar.date_input("비교하고 싶은 날짜 선택", max_date)

# 분석 로직
current_data = df[df['날짜'] == pd.Timestamp(target_date)]

if current_data.empty:
    st.warning("해당 날짜의 데이터가 없습니다.")
else:
    # 데이터 추출
    avg_temp = current_data['평균기온(℃)'].values[0]
    month, day = target_date.month, target_date.day
    historical = df[(df['날짜'].dt.month == month) & (df['날짜'].dt.day == day)].dropna()
    
    # 통계량
    hist_avg = historical['평균기온(℃)'].mean()
    rank = historical['평균기온(℃)'].rank(ascending=False).loc[current_data.index[0]]
    total = len(historical)

    # 대시보드 상단 지표
    col1, col2, col3 = st.columns(3)
    col1.metric("선택한 날 기온", f"{avg_temp}℃")
    col2.metric("역대 평균(평년)", f"{hist_avg:.1f}℃", f"{avg_temp - hist_avg:.1f}℃")
    col3.metric("역대 순위", f"{int(rank)}위", f"전체 {total}년 중")

    # 시각화 1: 역대 같은 날짜 기온 분포 (히스토그램)
    st.subheader(f"📊 역대 {month}월 {day}일 기온 분포")
    fig_dist = px.histogram(historical, x='평균기온(℃)', nbins=20, 
                            title=f"역대 {month}/{day} 평균 기온 분포",
                            color_discrete_sequence=['skyblue'])
    fig_dist.add_vline(x=avg_temp, line_dash="dash", line_color="red", 
                       annotation_text=f"{target_date.year}년({avg_temp}℃)")
    st.plotly_chart(fig_dist, use_container_width=True)

    # 시각화 2: 연도별 해당 날짜 기온 변화 (선 그래프)
    st.subheader(f"📈 역대 {month}월 {day}일 기온 변화 추이")
    fig_line = px.line(historical, x='날짜', y='평균기온(℃)', 
                       title=f"연도별 {month}/{day} 기온 추이")
    fig_line.add_hline(y=hist_avg, line_dash="dot", line_color="green", 
                       annotation_text="평균치")
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("상세 데이터 보기"):
        st.write(historical.sort_values(by='날짜', ascending=False))

# --- 연도별 추이 분석 섹션 ---
st.markdown("---")
st.subheader("🗓️ 서울 기온 연도별 장기 추이")
st.write("마우스를 그래프 위에 올리면 해당 연도의 상세 기온(평균/최저/최고)을 확인할 수 있습니다.")

# 1. 연도별 데이터 그룹화
df['연도'] = df['날짜'].dt.year
yearly_data = df.groupby('연도').agg({
    '평균기온(℃)': 'mean',
    '최저기온(℃)': 'mean',
    '최고기온(℃)': 'mean'
}).reset_index()

# 2. Plotly를 이용한 멀티 라인 그래프 생성
fig_yearly = go.Figure()

# 평균 기온 선
fig_yearly.add_trace(go.Scatter(
    x=yearly_data['연도'], y=yearly_data['평균기온(℃)'],
    mode='lines', name='연평균 기온',
    line=dict(color='orange', width=3),
    hovertemplate='<b>%{x}년</b><br>평균: %{y:.2f}℃'
))

# 최고 기온 선
fig_yearly.add_trace(go.Scatter(
    x=yearly_data['연도'], y=yearly_data['최고기온(℃)'],
    mode='lines', name='연평균 최고기온',
    line=dict(color='red', width=1, dash='dot'),
    hovertemplate='최고: %{y:.2f}℃'
))

# 최저 기온 선
fig_yearly.add_trace(go.Scatter(
    x=yearly_data['연도'], y=yearly_data['최저기온(℃)'],
    mode='lines', name='연평균 최저기온',
    line=dict(color='blue', width=1, dash='dot'),
    hovertemplate='최저: %{y:.2f}℃'
))

# 3. 레이아웃 설정 (커서 위치 시 수직선 표시 등)
fig_yearly.update_layout(
    hovermode="x unified",  # 커서 위치의 모든 데이터를 한 번에 표시
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_yearly, use_container_width=True)
