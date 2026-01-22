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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. 페이지 설정 (코드의 가장 처음에 딱 한 번만 와야 함)
st.set_page_config(page_title="서울 기온 분석기", layout="wide")

# 2. 데이터 로드 함수
@st.cache_data
def load_data(file):
    try:
        # 헤더 7행 스킵 (기상청 데이터 형식 대응)
        df = pd.read_csv(file, encoding='cp949', skiprows=7)
        df.columns = [col.strip() for col in df.columns]
        
        # 날짜 및 기온 데이터 정제 (\t 제거 및 수치화)
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.replace('\t', ''))
        for col in ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('\t', ''), errors='coerce')
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

st.title("🌡️ 서울 기온 역대 비교 분석기")

# 3. 데이터 로드 로직 (중복 방지를 위해 사이드바에 한 번만 선언)
uploaded_file = st.sidebar.file_uploader("추가 기온 데이터 업로드 (CSV)", type="csv", key="main_uploader")

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # 기본 파일명 (리포지토리에 이 이름으로 파일이 있어야 함)
    default_filename = "ta_20260122174530.csv"
    df = load_data(default_filename)

if df is None:
    st.warning("데이터를 불러올 수 없습니다. 파일을 업로드해 주세요.")
    st.stop()

# --- [상단 섹션] 특정 날짜 분석 ---
st.sidebar.header("📅 분석 날짜 설정")
max_date = df['날짜'].max()
target_date = st.sidebar.date_input("비교할 날짜 선택", max_date, key="main_date_picker")

current_data = df[df['날짜'] == pd.Timestamp(target_date)]

if not current_data.empty:
    avg_temp = current_data['평균기온(℃)'].values[0]
    month, day = target_date.month, target_date.day
    
    # 역대 같은 날짜 필터링
    historical = df[(df['날짜'].dt.month == month) & (df['날짜'].dt.day == day)].dropna()
    hist_avg = historical['평균기온(℃)'].mean()
    rank = historical['평균기온(℃)'].rank(ascending=False).loc[current_data.index[0]]
    total_y = len(historical)

    st.subheader(f"📊 {target_date.strftime('%Y-%m-%d')} 기온 분석")
    col1, col2, col3 = st.columns(3)
    col1.metric("선택한 날 평균", f"{avg_temp}℃")
    col2.metric("평년(역대평균)", f"{hist_avg:.1f}℃", f"{avg_temp - hist_avg:.1f}℃")
    col3.metric("기온 순위", f"{int(rank)}위", f"전체 {total_y}개년 중")

# --- [하단 섹션] 연도별 장기 추이 분석 (필터링 로직 포함) ---
st.markdown("---")
st.subheader("🗓️ 서울 기온 연도별 장기 추이")
st.info("💡 데이터 정확성을 위해 1년 데이터가 360일 미만인 해(전쟁 기간, 첫해/마지막해 등)는 자동으로 제외했습니다.")

# 연도별 통계 계산
df['연도'] = df['날짜'].dt.year
yearly_stats = df.groupby('연도').agg({
    '평균기온(℃)': ['mean', 'count'],
    '최저기온(℃)': 'mean',
    '최고기온(℃)': 'mean'
})
yearly_stats.columns = ['평균기온', '데이터개수', '최저기온', '최고기온']
yearly_stats = yearly_stats.reset_index()

# 360일 이상의 온전한 데이터만 필터링 (첫해, 마지막해, 전쟁기간 자동 필터링)
clean_yearly = yearly_stats[yearly_stats['데이터개수'] >= 360].copy()

# 그래프 생성 (커서 통합 모드)
fig_yearly = go.Figure()

fig_yearly.add_trace(go.Scatter(
    x=clean_yearly['연도'], y=clean_yearly['평균기온'],
    mode='lines+markers', name='연평균',
    line=dict(color='orange', width=3),
    hovertemplate='평균: %{y:.2f}℃'
))

fig_yearly.add_trace(go.Scatter(
    x=clean_yearly['연도'], y=clean_yearly['최고기온'],
    mode='lines', name='최고(평균)',
    line=dict(color='red', width=1, dash='dot'),
    hovertemplate='최고: %{y:.2f}℃'
))

fig_yearly.add_trace(go.Scatter(
    x=clean_yearly['연도'], y=clean_yearly['최저기온'],
    mode='lines', name='최저(평균)',
    line=dict(color='blue', width=1, dash='dot'),
    hovertemplate='최저: %{y:.2f}℃'
))

fig_yearly.update_layout(
    hovermode="x unified",
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_yearly, use_container_width=True)

# 제외된 연도 정보 제공
with st.expander("데이터 처리 상세 내역"):
    excluded = set(yearly_stats['연도']) - set(clean_yearly['연도'])
    st.write(f"✅ **포함된 연도 수:** {len(clean_yearly)}개년")
    st.write(f"❌ **제외된 연도 (데이터 부족):** {sorted(list(excluded))}")

from sklearn.linear_model import LinearRegression
import numpy as np

# --- [신규 섹션] 머신러닝 기온 예측 분석 ---
st.markdown("---")
st.subheader("🤖 머신러닝 기반 미래 기온 예측")
st.write("선형 회귀 모델을 학습하여 향후 10년, 20년, 30년 뒤의 서울 평균 기온을 예측합니다.")

# 1. 모델 학습 데이터 준비 (결측치 없는 연도별 데이터 사용)
X = clean_yearly['연도'].values.reshape(-1, 1)
y = clean_yearly['평균기온'].values

# 2. 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 3. 미래 연도 설정 및 예측
future_years = np.array([2035, 2045, 2055]).reshape(-1, 1)
predictions = model.predict(future_years)

# 4. 결과 시각화 및 지표 출력
p1, p2, p3 = st.columns(3)
p1.metric("2035년 예상 평균기온", f"{predictions[0]:.2f}℃")
p2.metric("2045년 예상 평균기온", f"{predictions[1]:.2f}℃")
p3.metric("2055년 예상 평균기온", f"{predictions[2]:.2f}℃")

# 5. 회귀선 그래프 추가
# 전체 기간에 대한 회귀선 계산
trend_line = model.predict(X)

fig_predict = go.Figure()

# 실제 데이터
fig_predict.add_trace(go.Scatter(x=clean_yearly['연도'], y=y, mode='markers', name='실제 연평균', marker=dict(color='gray', opacity=0.5)))
# 학습된 회귀선
fig_predict.add_trace(go.Scatter(x=clean_yearly['연도'], y=trend_line, mode='lines', name='상승 추세선', line=dict(color='red', width=2)))
# 미래 예측 지점
fig_predict.add_trace(go.Scatter(x=[2035, 2045, 2055], y=predictions, mode='markers+text', 
                                 name='미래 예측값', text=[f"{p:.2f}℃" for p in predictions],
                                 textposition="top center", marker=dict(color='black', size=10, symbol='diamond')))

fig_predict.update_layout(
    title="서울 연평균 기온 상승 추세 및 미래 예측",
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    showlegend=True
)

st.plotly_chart(fig_predict, use_container_width=True)

with st.expander("🎓 선형 회귀 분석 결과 요약"):
    slope = model.coef_[0]
    st.write(f"📈 **기온 상승 속도:** 서울의 기온은 매년 약 **{slope:.4f}℃**씩 상승하고 있습니다.")
    st.write(f"🌡️ **100년 환산:** 이 추세라면 100년 뒤 서울의 평균 기온은 현재보다 약 **{slope*100:.2f}℃** 더 높아질 것으로 예측됩니다.")


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# 페이지 설정
st.set_page_config(page_title="서울 기온 & 열대야 분석기", layout="wide")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='cp949', skiprows=7)
    df.columns = [col.strip() for col in df.columns]
    df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.replace('\t', ''))
    for col in ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('\t', ''), errors='coerce')
    return df

st.title("🌡️ 서울 기온 추이 및 열대야 분석 리포트")

# 데이터 로드
uploaded_file = st.sidebar.file_uploader("추가 데이터 업로드", type="csv", key="ml_uploader")
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("ta_20260122174530.csv")

# 연도별 기본 통계 계산
df['연도'] = df['날짜'].dt.year
yearly_stats = df.groupby('연도').agg({
    '평균기온(℃)': ['mean', 'count'],
    '최저기온(℃)': 'mean',
    '최고기온(℃)': 'mean'
})
yearly_stats.columns = ['평균기온', '데이터개수', '최저기온', '최고기온']
yearly_stats = yearly_stats.reset_index()

# 360일 미만 데이터 제외 (전쟁 및 불완전한 해)
clean_yearly = yearly_stats[yearly_stats['데이터개수'] >= 360].copy()

# --- [섹션 1] 열대야 분석 ---
st.header("🌙 연도별 열대야 발생 일수 변화")
st.info("열대야 기준: 일 최저기온이 **25°C 이상**인 날")

# 일별 데이터에서 최저기온 25도 이상인 날 카운트
tropical_nights = df[df['최저기온(℃)'] >= 25].groupby('연도').size().reset_index(name='열대야일수')

# 데이터가 부족한 해는 열대야 통계에서도 제외
clean_tropical = tropical_nights[tropical_nights['연도'].isin(clean_yearly['연도'])]

fig_tropical = px.bar(clean_tropical, x='연도', y='열대야일수',
                      title="연도별 열대야 발생 일수 추이",
                      color='열대야일수', color_continuous_scale='Reds')

fig_tropical.update_layout(xaxis_title="연도", yaxis_title="발생 일수 (일)")
st.plotly_chart(fig_tropical, use_container_width=True)

# --- [섹션 2] 머신러닝 기온 예측 ---
st.markdown("---")
st.header("🤖 머신러닝 기반 미래 기온 예측")

# 모델 학습
X = clean_yearly['연도'].values.reshape(-1, 1)
y = clean_yearly['평균기온'].values
model = LinearRegression().fit(X, y)

# 미래 예측 (10, 20, 30년 뒤)
current_year = 2025
future_years = np.array([current_year + 10, current_year + 20, current_year + 30]).reshape(-1, 1)
future_preds = model.predict(future_years)

c1, c2, c3 = st.columns(3)
c1.metric(f"{future_years[0][0]}년 예상", f"{future_preds[0]:.2f}℃")
c2.metric(f"{future_years[1][0]}년 예상", f"{future_preds[1]:.2f}℃")
c3.metric(f"{future_years[2][0]}년 예상", f"{future_preds[2]:.2f}℃")

# 시각화 (회귀선 포함)
fig_ml = go.Figure()
fig_ml.add_trace(go.Scatter(x=clean_yearly['연도'], y=y, mode='markers', name='실제 평균기온', marker=dict(color='gray', opacity=0.4)))
fig_ml.add_trace(go.Scatter(x=clean_yearly['연도'], y=model.predict(X), mode='lines', name='기온 상승 추세선', line=dict(color='red', width=2)))
fig_ml.add_trace(go.Scatter(x=future_years.flatten(), y=future_preds, mode='markers+text', 
                            text=[f"{p:.1f}℃" for p in future_preds], textposition="top center",
                            name='미래 예측값', marker=dict(color='black', size=10, symbol='diamond')))

fig_ml.update_layout(title="서울 연평균 기온 장기 추세 및 미래 예측", xaxis_title="연도", yaxis_title="기온 (℃)", hovermode="x")
st.plotly_chart(fig_ml, use_container_width=True)

# 결론 출력
slope = model.coef_[0]
st.success(f"📈 분석 결과: 서울의 연평균 기온은 매년 약 **{slope:.4f}℃**씩 상승하고 있습니다. (100년 기준 약 **{slope*100:.2f}℃** 상승)")
