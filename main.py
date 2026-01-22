import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="서울 기온 빅데이터 리포트", layout="wide")

# 2. 데이터 로드 함수 (탭 문자 제거 및 수치 변환 통합)
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, encoding='cp949', skiprows=7)
        df.columns = [col.strip() for col in df.columns]
        # 데이터 정제: 탭 문자(\t) 제거 및 날짜/숫자 변환
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.replace('\t', ''))
        for col in ['평균기온(℃)', '최저기온(℃)', '최고기온(℃)']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('\t', ''), errors='coerce')
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

# 3. 사이드바 통합 (중복 방지)
st.sidebar.header("📂 데이터 및 설정")
uploaded_file = st.sidebar.file_uploader("추가 기온 데이터 업로드 (CSV)", type="csv")

# 데이터 우선순위 결정
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("ta_20260122174530.csv")

if df is not None:
    # 전처리: 연도별 통계 및 필터링 (360일 기준)
    df['연도'] = df['날짜'].dt.year
    yearly_stats = df.groupby('연도').agg({
        '평균기온(℃)': ['mean', 'count'],
        '최저기온(℃)': 'mean',
        '최고기온(℃)': 'mean'
    })
    yearly_stats.columns = ['평균기온', '데이터개수', '최저기온', '최고기온']
    yearly_stats = yearly_stats.reset_index()
    clean_yearly = yearly_stats[yearly_stats['데이터개수'] >= 360].copy()

    # 사이드바 날짜 선택
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 분석 날짜 선택")
    max_date = df['날짜'].max()
    target_date = st.sidebar.date_input("비교 기준일", max_date)

    # 메인 화면 제목
    st.title("🌡️ 서울 기온 빅데이터 분석 리포트")
    
    # 4. 주제별 탭 구성 (좋은 아이디어 반영)
    tab1, tab2, tab3, tab4 = st.tabs(["📌 일별 비교", "📈 장기 추세", "🌙 열대야 분석", "🤖 머신러닝 예측"])

    # --- Tab 1: 일별 비교 ---
    with tab1:
        st.header(f"📊 {target_date.strftime('%Y년 %m월 %d일')} 기온 분석")
        current_data = df[df['날짜'] == pd.Timestamp(target_date)]
        
        if not current_data.empty:
            avg_temp = current_data['평균기온(℃)'].values[0]
            month, day = target_date.month, target_date.day
            historical = df[(df['날짜'].dt.month == month) & (df['날짜'].dt.day == day)].dropna()
            hist_avg = historical['평균기온(℃)'].mean()
            rank = historical['평균기온(℃)'].rank(ascending=False).loc[current_data.index[0]]

            c1, c2, c3 = st.columns(3)
            c1.metric("선택한 날 기온", f"{avg_temp}℃")
            c2.metric("평년 평균", f"{hist_avg:.1f}℃", f"{avg_temp - hist_avg:.1f}℃")
            c3.metric("역대 순위", f"{int(rank)}위", f"전체 {len(historical)}년 중")

            fig_dist = px.histogram(historical, x='평균기온(℃)', title=f"역대 {month}/{day} 기온 분포")
            fig_dist.add_vline(x=avg_temp, line_color="red", annotation_text="오늘")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("선택한 날짜의 데이터가 없습니다.")

    # --- Tab 2: 장기 추세 ---
    with tab2:
        st.header("🗓️ 연도별 기온 장기 변화")
        st.write("1907년부터 현재까지의 연평균 기온 변화입니다. (결측 연도 제외)")
        
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(x=clean_yearly['연도'], y=clean_yearly['평균기온'], mode='lines+markers', name='연평균', line=dict(color='orange')))
        fig_yearly.add_trace(go.Scatter(x=clean_yearly['연도'], y=clean_yearly['최고기온'], mode='lines', name='최고(평균)', line=dict(color='red', dash='dot')))
        fig_yearly.add_trace(go.Scatter(x=clean_yearly['연도'], y=clean_yearly['최저기온'], mode='lines', name='최저(평균)', line=dict(color='blue', dash='dot')))
        fig_yearly.update_layout(hovermode="x unified")
        st.plotly_chart(fig_yearly, use_container_width=True)

    # --- Tab 3: 열대야 분석 ---
    with tab3:
        st.header("🌙 연도별 열대야 일수")
        st.info("최저기온이 25℃ 이상인 밤의 횟수 변화를 확인합니다.")
        
        tropical = df[df['최저기온(℃)'] >= 25].groupby('연도').size().reset_index(name='일수')
        clean_tropical = tropical[tropical['연도'].isin(clean_yearly['연도'])]
        
        fig_trop = px.bar(clean_tropical, x='연도', y='일수', color='일수', color_continuous_scale='Reds')
        st.plotly_chart(fig_trop, use_container_width=True)

    # --- Tab 4: 머신러닝 예측 ---
    with tab4:
        st.header("🤖 미래 기온 예측 (선형 회귀)")
        X = clean_yearly['연도'].values.reshape(-1, 1)
        y = clean_yearly['평균기온'].values
        model = LinearRegression().fit(X, y)
        
        future_years = np.array([2035, 2045, 2055]).reshape(-1, 1)
        preds = model.predict(future_years)
        
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("2035년 예측", f"{preds[0]:.2f}℃")
        pc2.metric("2045년 예측", f"{preds[1]:.2f}℃")
        pc3.metric("2055년 예측", f"{preds[2]:.2f}℃")
        
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=clean_yearly['연도'], y=y, mode='markers', name='실제 기온', marker=dict(color='gray', opacity=0.3)))
        fig_ml.add_trace(go.Scatter(x=clean_yearly['연도'], y=model.predict(X), mode='lines', name='상승 추세선', line=dict(color='red')))
        fig_ml.add_trace(go.Scatter(x=future_years.flatten(), y=preds, mode='markers+text', text=[f"{p:.1f}℃" for p in preds], textposition="top center", name='예측 지점', marker=dict(size=12, symbol='diamond', color='black')))
        st.plotly_chart(fig_ml, use_container_width=True)
        
        slope = model.coef_[0]
        st.success(f"📈 분석 결과: 서울은 매년 평균 약 **{slope:.4f}℃**씩 기온이 상승하고 있습니다.")

else:
    st.error("데이터 파일을 찾을 수 없습니다. CSV 파일을 확인해주세요.")
