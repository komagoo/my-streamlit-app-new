import os
import sys
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import base64
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 환경 변수 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🚀 HERO - 반도체 정비 도우미",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로고 이미지 base64 인코딩
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "Hero_logo(final).png"
logo_base64 = get_base64_of_bin_file(logo_path)

# 상단 로고 및 헤더
st.markdown(
    f"""
    <div style="background-color:#ff4b4b; height:30px; width:100%;"></div>
    <div style="display:flex; align-items:center; padding:20px 30px;">
        <img src="data:image/png;base64,{logo_base64}" alt="logo" style="height:80px; margin-right:20px;">
        <div>
            <h1 style="margin:0; font-size:2.5rem; color:#222;">HERO</h1>
            <p style="margin:0; font-size:1.1rem; color:#555;">반도체 장비 문제 해결 도우미</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 로그인 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# 로그인 화면
if not st.session_state.logged_in:
    st.subheader("🔑 로그인")
    st.markdown("HERO를 이용하려면 로그인하세요.")

    username = st.text_input("아이디", placeholder="아이디를 입력하세요")
    password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")

    valid_users = {
        "mySUNI250728!@": "mySUNI250728!@",
    }

    if st.button("로그인"):
        if username in valid_users and password == valid_users[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
            st.success(f"✅ {username}님, 환영합니다!")
            st.experimental_rerun()
        else:
            st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# 메인 타이틀
st.markdown("<h2 style='color:#000000; font-weight:bold;'>반도체 장비 문제, HERO와 함께 해결해요!</h2>", unsafe_allow_html=True)

# 엑셀 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if uploaded_file:
    st.success(f"📂 파일 업로드 완료: {uploaded_file.name}")
    df = pd.read_excel(uploaded_file)
    st.dataframe(df.head(10))  # 파일 미리보기
else:
    st.info("엑셀 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

# 데이터 처리
if '정비일자' in df.columns:
    df['정비일자'] = pd.to_datetime(df['정비일자'], errors='coerce')

df = df.dropna(subset=['정비노트'])
st.success(f"업로드 완료: 총 {len(df)} 행")

# 사이드바 메뉴
menu = st.sidebar.radio(
    "📂 메뉴 선택",
    ["🔹 정비 검색 & 추천", "📈 정비 통계 자료"],
    index=0
)

# 검색 및 추천 페이지
if menu == "🔹 정비 검색 & 추천":
    st.subheader("🤖 HERO 챗봇 – 정비 문제 해결 도우미")
    query = st.text_input("문제를 입력하세요", placeholder="예: slot valve 동작 불량")
    if query.strip():
        st.markdown(f"🔍 검색어: {query}")
        st.spinner("🔄 검색 중입니다...")
        # 검색 결과 표시 (간단한 카드 형태)
        st.markdown("""
        <div style="border:1px solid #ccc; padding:10px; border-radius:8px;">
            <b>조치명:</b> RF Auto Match 조정<br>
            <b>성공률:</b> 85%<br>
            <b>장비:</b> Model X123<br>
            <b>정비노트:</b> RF Auto Match 불량 발생. 추가 조치로 정상 확인.
        </div>
        """, unsafe_allow_html=True)

# 통계 페이지
elif menu == "📈 정비 통계 자료":
    st.subheader("📈 정비 통계 자료")
    tab1, tab2 = st.tabs(["🏆 Top5 요약", "📊 전체 요약"])
    with tab1:
        st.markdown("### TOP5 문제 원인")
        fig = px.bar(x=[10, 20, 30], y=["원인1", "원인2", "원인3"], orientation="h")
        st.plotly_chart(fig)

    with tab2:
        st.markdown("### 전체 문제 원인 분포")
        fig = px.pie(values=[40, 30, 30], names=["원인1", "원인2", "원인3"])
        st.plotly_chart(fig)