import os
import sys
import streamlit as st
from dotenv import load_dotenv
import faiss

# 환경 변수 로드 (.env)
load_dotenv()

# 일반 라이브러리
import pandas as pd
import re
from collections import defaultdict, Counter
import plotly.express as px
import base64

# Langchain 관련
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 로고 이미지 base64 인코딩
def get_base64_of_bin_file(bin_file_path):
    try:
        with open(bin_file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

logo_path = "Hero_logo(final).png"
logo_base64 = get_base64_of_bin_file(logo_path)

# ----------------------------
# 0. Streamlit 설정
# ----------------------------
st.set_page_config(
    page_title="🚀 HERO - Hynix Equipment Response Operator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .bot-message {
        background-color: #F1F0F0;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 20%;
    }
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    .stTab {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 헤더 섹션
if logo_base64:
    st.markdown(f"""
    <div class="main-header">
        <div style="display:flex; align-items:center; justify-content:center;">
            <img src="data:image/png;base64,{logo_base64}" alt="HERO Logo" style="height:80px; margin-right:20px;">
            <div>
                <h1 style="margin:0; font-size:2.5rem;">HERO</h1>
                <p style="margin:0; font-size:1.2rem; opacity:0.9;">Hynix Equipment Response Operator</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🚀 HERO</h1>
        <p style="margin:0; font-size:1.2rem; opacity:0.9;">Hynix Equipment Response Operator</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# 로그인 세션 상태 초기화
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# ----------------------------
# 1. 로그인 단계
# ----------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="text-align: center; color: #333; margin-bottom: 30px;">🔐 로그인</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 아이디", placeholder="아이디를 입력하세요")
            password = st.text_input("🔒 비밀번호", type="password", placeholder="비밀번호를 입력하세요")
            
            submitted = st.form_submit_button("🚀 로그인", use_container_width=True)
            
            if submitted:
                valid_users = {"mySUNI250728!@": "mySUNI250728!@"}
                
                if username in valid_users and password == valid_users[username]:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
                    st.success(f"✅ {username}님, 환영합니다!")
                    st.rerun()
                else:
                    st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# ----------------------------
# OpenAI API 키 환경변수 세팅
# ----------------------------
if st.session_state.api_key and isinstance(st.session_state.api_key, str):
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
else:
    api_key_env = os.getenv("OPENAI_API_KEY")
    if api_key_env:
        os.environ["OPENAI_API_KEY"] = api_key_env
    else:
        st.error("⚠️ OpenAI API 키가 설정되어 있지 않습니다.")
        st.stop()

# 사이드바 - 사용자 정보 및 로그아웃
with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #333;">👋 환영합니다!</h4>
        <p style="margin: 5px 0 0 0; color: #666;">{st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 로그아웃", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# ----------------------------
# 2. 파일 업로드 섹션
# ----------------------------
st.markdown("### 📁 데이터 업로드")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "엑셀 파일을 업로드하세요 (.xlsx)", 
        type=["xlsx"],
        help="정비노트가 포함된 엑셀 파일을 업로드하세요"
    )

with col2:
    if uploaded_file:
        st.success("✅ 파일 업로드 완료")
    else:
        st.info("📤 파일을 선택해주세요")

if uploaded_file is None:
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0;">
        <h3 style="color: #1976d2; margin-bottom: 15px;">📋 사용 방법</h3>
        <p style="color: #424242; font-size: 16px; line-height: 1.6;">
            1. 정비 데이터가 포함된 엑셀 파일을 업로드하세요<br>
            2. HERO가 데이터를 분석하여 인사이트를 제공합니다<br>
            3. 챗봇을 통해 정비 문제 해결책을 찾아보세요
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# 데이터 로딩 및 전처리
with st.spinner("📊 데이터를 분석하고 있습니다..."):
    df = pd.read_excel(uploaded_file)
    if '정비일자' in df.columns:
        df['정비일자'] = pd.to_datetime(df['정비일자'], errors='coerce')
    
    df = df.dropna(subset=['정비노트'])
    
    # 성공 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1976d2; margin: 0;">{len(df):,}</h2>
            <p style="margin: 5px 0 0 0; color: #666;">총 정비 기록</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_equip = df['모델'].nunique() if '모델' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #388e3c; margin: 0;">{unique_equip}</h2>
            <p style="margin: 5px 0 0 0; color: #666;">장비 종류</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        date_range = "N/A"
        if '정비일자' in df.columns and not df['정비일자'].isna().all():
            start_date = df['정비일자'].min().strftime('%Y-%m')
            end_date = df['정비일자'].max().strftime('%Y-%m')
            date_range = f"{start_date} ~ {end_date}"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f57c00; margin: 0; font-size: 14px;">{date_range}</h3>
            <p style="margin: 5px 0 0 0; color: #666;">데이터 기간</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_person = df['정비자'].nunique() if '정비자' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #7b1fa2; margin: 0;">{unique_person}</h2>
            <p style="margin: 5px 0 0 0; color: #666;">정비 담당자</p>
        </div>
        """, unsafe_allow_html=True)

# 문제 원인 분석
problem_keywords = [
    "wafer not", "plasma ignition failure", "pumpdown 시간 지연",
    "mass flow controller 이상", "etch residue over spec",
    "temperature drift", "slot valve 동작 불량",
    "chamber pressure fluctuation", "he flow deviation", "RF auto match 불량"
]

alias_map = {
    "wafer not 감지됨": "wafer not",
    "wafer not 발생": "wafer not",
    "rf auto match fail": "RF auto match 불량",
    "slot valve 불량": "slot valve 동작 불량",
    "he flow dev": "he flow deviation",
}

def normalize_note(note: str) -> str:
    note = str(note).lower()
    note = re.sub(r'\s+', ' ', note)
    return note

def extract_cause(note: str):
    note_low = normalize_note(note)
    for alias, norm in alias_map.items():
        if alias in note_low:
            return norm
    for keyword in problem_keywords:
        if keyword.lower() in note_low:
            return keyword
    return "기타"

df['문제원인'] = df['정비노트'].apply(extract_cause)

# 성공률 계산
all_texts = [str(note).strip() for note in df['정비노트']]
cause_pattern = re.compile(r'LOT 진행 중 (.+) 발생')
first_action_pattern = re.compile(r'1차 조치: (.+) → 여전히 이상 발생')
second_action_pattern = re.compile(r'정비 시작\. (.+) 진행')
third_action_pattern = re.compile(r'추가 조치: (.+)')

cause_aliases = {
    "wafer not 발생": "wafer not",
    "wafer not 감지됨": "wafer not",
    "wafer not 발생 확인": "wafer not",
}

def normalize_cause(cause):
    for alias, norm in cause_aliases.items():
        if alias in cause:
            return norm
    return cause

cause_action_counts = defaultdict(lambda: defaultdict(Counter))
note_map = defaultdict(list)

for idx, note in enumerate(all_texts):
    lines = [line.strip() for line in note.split('\n') if line.strip()]
    cause = None
    for line in lines:
        cause_match = cause_pattern.search(line)
        if cause_match:
            cause = normalize_cause(cause_match.group(1).strip())
            continue
        if cause is None:
            continue

        action = None
        m1 = first_action_pattern.search(line)
        m2 = second_action_pattern.search(line)
        m3 = third_action_pattern.search(line)

        if m1:
            action = m1.group(1).strip()
            cause_action_counts[cause][action]['first'] += 1
        elif m2:
            action = m2.group(1).strip()
            cause_action_counts[cause][action]['second'] += 1
        elif m3:
            action = m3.group(1).strip()
            cause_action_counts[cause][action]['third'] += 1

        if action:
            note_map[(cause, action)].append(note)

rows = []
for cause, actions in cause_action_counts.items():
    for action, counts in actions.items():
        first_count = counts['first']
        second_count = counts['second']
        third_count = counts['third']
        total = first_count + second_count + third_count
        success = second_count + third_count
        success_rate = round(success / total * 100, 2) if total > 0 else 0

        rows.append({
            "대표원인": cause,
            "조치": action,
            "총횟수": total,
            "실패횟수": first_count,
            "성공횟수": success,
            "성공률(%)": success_rate,
            "정비노트": note_map[(cause, action)][0] if note_map[(cause, action)] else ""
        })

df_success = pd.DataFrame(rows)

# RAG 시스템 초기화
documents = [
    Document(page_content=str(row['정비노트']), metadata={'row': idx})
    for idx, row in df.iterrows()
]

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

INDEX_PATH = "faiss_index.index"

from langchain.docstore import InMemoryDocstore

def load_or_create_vectordb(documents, embedding_model):
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            index_to_docstore_id = {i: str(i) for i in range(len(documents))}
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
            vectordb = FAISS(
                embedding_function=embedding_model.embed_query,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
        except:
            vectordb = FAISS.from_documents(documents, embedding_model)
            faiss.write_index(vectordb.index, INDEX_PATH)
    else:
        vectordb = FAISS.from_documents(documents, embedding_model)
        faiss.write_index(vectordb.index, INDEX_PATH)
    return vectordb

if "embedding_model" not in st.session_state or "vectordb" not in st.session_state:
    with st.spinner("🤖 AI 모델을 준비하고 있습니다..."):
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        vectordb = load_or_create_vectordb(split_docs, embedding_model)
        st.session_state["embedding_model"] = embedding_model
        st.session_state["vectordb"] = vectordb
else:
    embedding_model = st.session_state["embedding_model"]
    vectordb = st.session_state["vectordb"]

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 20}),
    return_source_documents=True
)

# ----------------------------
# 메인 인터페이스
# ----------------------------
tab1, tab2 = st.tabs(["🤖 AI 정비 상담", "📊 정비 데이터 분석"])

# ----------------------------
# Tab1: AI 정비 상담
# ----------------------------
with tab1:
    st.markdown("### 🤖 HERO AI 상담사")
    
    # 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    chat_container = st.container()
    
    with chat_container:
        # 초기 인사말
        if not st.session_state.messages:
            st.markdown("""
            <div class="bot-message">
                <strong>🤖 HERO</strong><br>
                안녕하세요! 반도체 장비 정비 전문 AI HERO입니다 👋<br><br>
                정비 문제를 입력하시면, 유사 사례를 분석해서 최적의 해결책을 제안해드려요!<br><br>
                💡 <strong>예시:</strong> wafer not | plasma ignition failure | slot valve 동작 불량
            </div>
            """, unsafe_allow_html=True)
        
        # 이전 대화 기록 표시
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # 사용자 입력
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "정비 문제를 입력하세요", 
            key="user_input",
            placeholder="예: slot valve 동작이 안돼요...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 전송", use_container_width=True)
    
    if send_button and user_input.strip():
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🔍 유사 사례를 분석하고 있습니다..."):
            # RAG 검색
            output = qa_chain({"query": user_input})
            docs = output['source_documents']
            
            recommended = []
            seen_pairs = set()
            
            for doc in docs:
                note = doc.page_content.strip()
                for _, row in df_success.iterrows():
                    if row["조치"] in note:
                        key = (row["조치"], note)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        
                        matched_row = df[df['정비노트'].astype(str).str.strip() == note]
                        equip_id = matched_row['장비ID'].iloc[0] if '장비ID' in df.columns and not matched_row.empty else 'N/A'
                        model = matched_row['모델'].iloc[0] if '모델' in df.columns and not matched_row.empty else 'N/A'
                        
                        recommended.append({
                            "조치": row["조치"],
                            "성공률": row["성공률(%)"],
                            "정비노트": note,
                            "장비ID": equip_id,
                            "모델": model
                        })
            
            if not recommended:
                bot_response = "죄송합니다. 관련된 정비 사례를 찾을 수 없습니다. 다른 키워드로 다시 시도해보세요."
            else:
                # Top3 선정
                def is_final_action(note: str):
                    return ("추가 조치" in note) or ("정상 확인" in note)
                
                final_candidates = [r for r in recommended if is_final_action(r["정비노트"])]
                candidates_sorted = sorted(final_candidates, key=lambda x: x["성공률"], reverse=True)
                
                top3 = []
                used_actions = set()
                used_notes = set()
                for r in candidates_sorted:
                    note_key = r["정비노트"]
                    if r["조치"] not in used_actions and note_key not in used_notes:
                        top3.append(r)
                        used_actions.add(r["조치"])
                        used_notes.add(r["정비노트"])
                    if len(top3) == 3:
                        break
                
                if len(top3) < 3:
                    for r in sorted(recommended, key=lambda x: x["성공률"], reverse=True):
                        if r["조치"] not in used_actions and r["정비노트"] not in used_notes:
                            top3.append(r)
                            used_actions.add(r["조치"])
                            used_notes.add(r["정비노트"])
                        if len(top3) == 3:
                            break
                
                # 응답 생성
                top3_desc = "\n".join([f"{i+1}. {r['조치']} (성공률 {r['성공률']}%)"
                                     for i, r in enumerate(top3)])
                
                prompt = f"""
다음은 반도체 장비 정비 이슈에 대한 Top3 성공률 높은 조치입니다.
각 조치의 의미와 특징을 자연스럽게 설명해주세요.

{top3_desc}
                """
                explanation = llm.predict(prompt)
                
                bot_response = f"""
<strong>✅ 추천 해결책 Top 3</strong><br><br>
{top3_desc.replace(chr(10), '<br>')}<br><br>
<strong>💡 상세 설명:</strong><br>
{explanation}
                """
        
        # 봇 응답 추가
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # 페이지 새로고침
        st.rerun()

# ----------------------------
# Tab2: 정비 데이터 분석
# ----------------------------
with tab2:
    st.markdown("### 📊 정비 데이터 분석")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["🏆 핵심 지표", "📈 전체 현황", "🔧 장비별 분석"])
    
    with analysis_tab1:
        st.markdown("#### 🏆 핵심 지표 TOP 5")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔧 고장 빈발 장비")
            top5_equip = df['모델'].value_counts().head(5)
            
            fig1 = px.bar(
                x=top5_equip.values,
                y=top5_equip.index,
                orientation='h',
                text=[f"{v}건" for v in top5_equip.values],
                color=top5_equip.values,
                color_continuous_scale='Blues',
                height=400
            )
            fig1.update_traces(textposition='outside')
            fig1.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("##### ⚠️ 주요 문제 원인")
            top5_cause = df['문제원인'].value_counts().head(5)
            
            fig2 = px.bar(
                x=top5_cause.values,
                y=top5_cause.index,
                orientation='h',
                text=[f"{v}건" for v in top5_cause.values],
                color=top5_cause.values,
                color_continuous_scale='Reds',
                height=400
            )
            fig2.update_traces(textposition='outside')
            fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        
        # AI 인사이트
        st.markdown("#### 🧠 AI 분석 인사이트")
        
        with st.spinner("AI가 데이터를 분석하고 있습니다..."):
            prompt_insight = f"""
다음 데이터를 바탕으로 반도체 장비 정비에 대한 핵심 인사이트를 제공해주세요:

고장 빈발 장비 TOP5: {', '.join(top5_equip.index)}
주요 문제 원인 TOP5: {', '.join(top5_cause.index)}

예방 정비와 운영 효율성 관점에서 3-4문장으로 요약해주세요.
            """
            
            insight = llm.predict(prompt_insight)
            
            st.markdown(f"""
            <div class="success-card">
                <h4 style="margin-top: 0;">💡 핵심 인사이트</h4>
                <p style="margin-bottom: 0; line-height: 1.6;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with analysis_tab2:
        st.markdown("#### 📈 전체 현황 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🏭 장비별 고장 분포")
            total_equip = df['모델'].value_counts()
            
            fig_pie1 = px.pie(
                names=total_equip.index.tolist(),
                values=total_equip.values,
                hole=0.4,
                height=400
            )
            fig_pie1.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie1, use_container_width=True)
        
        with col2:
            st.markdown("##### 🔍 문제 원인 분포")
            total_cause = df['문제원인'].value_counts()
            
            fig_pie2 = px.pie(
                names=total_cause.index.tolist(),
                values=total_cause.values,
                hole=0.4,
                height=400
            )
            fig_pie2.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie2, use_container_width=True)
        
        # 전체 현황 AI 인사이트
        st.markdown("#### 🧠 전체 현황 AI 분석")
        
        with st.spinner("전체 데이터를 분석하고 있습니다..."):
            prompt_total = f"""
다음은 전체 장비 고장 및 문제 원인 분포 데이터입니다:

장비별 분포: {dict(total_equip.head(3))}
문제 원인별 분포: {dict(total_cause.head(3))}

전체적인 패턴과 운영 개선 방향을 3-4문장으로 제시해주세요.
            """
            
            total_insight = llm.predict(prompt_total)
            
            st.markdown(f"""
            <div class="success-card">
                <h4 style="margin-top: 0;">📊 전체 현황 분석</h4>
                <p style="margin-bottom: 0; line-height: 1.6;">{total_insight}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with analysis_tab3:
        st.markdown("#### 🔧 장비별 상세 분석")
        
        # 장비 선택
        equip_list = df['모델'].dropna().unique().tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_equip = st.selectbox(
                "분석할 장비를 선택하세요",
                ["전체 장비"] + equip_list,
                help="특정 장비를 선택하면 해당 장비의 상세 분석을 확인할 수 있습니다"
            )
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1976d2; margin: 0;">{len(equip_list)}</h3>
                <p style="margin: 5px 0 0 0; color: #666;">총 장비 종류</p>
            </div>
            """, unsafe_allow_html=True)
        
        if selected_equip != "전체 장비":
            # 선택된 장비 데이터 필터링
            df_filtered = df[df['모델'] == selected_equip]
            
            if df_filtered.empty:
                st.warning(f"⚠️ 선택한 장비({selected_equip})의 데이터가 없습니다.")
            else:
                # 장비 정보 요약
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #d32f2f; margin: 0;">{len(df_filtered)}</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">총 고장 건수</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    unique_causes = df_filtered['문제원인'].nunique()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #f57c00; margin: 0;">{unique_causes}</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">문제 원인 종류</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if '정비자' in df_filtered.columns:
                        unique_maintainers = df_filtered['정비자'].nunique()
                    else:
                        unique_maintainers = 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #388e3c; margin: 0;">{unique_maintainers}</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">담당 정비자</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 장비별 문제 원인 분석
                st.markdown(f"##### 🔍 {selected_equip} 문제 원인 분석")
                
                cause_counts = df_filtered['문제원인'].value_counts()
                
                # 문제 원인 차트
                fig_equip = px.bar(
                    x=cause_counts.values,
                    y=cause_counts.index,
                    orientation='h',
                    text=[f"{v}건" for v in cause_counts.values],
                    color=cause_counts.values,
                    color_continuous_scale='Viridis',
                    height=400
                )
                fig_equip.update_traces(textposition='outside')
                fig_equip.update_layout(
                    title=f"{selected_equip} 문제 원인별 발생 빈도",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_equip, use_container_width=True)
                
                # 선택된 장비의 추천 조치
                if len(cause_counts) > 0:
                    st.markdown("##### 🛠️ 추천 정비 조치")
                    
                    top_cause = cause_counts.index[0]
                    
                    # 해당 원인에 대한 성공률 높은 조치 찾기
                    cause_actions = df_success[df_success['대표원인'] == top_cause]
                    
                    if not cause_actions.empty:
                        top_actions = cause_actions.nlargest(3, '성공률(%)')
                        
                        st.markdown(f"**'{top_cause}' 문제에 대한 추천 조치 TOP 3:**")
                        
                        for idx, (_, action_row) in enumerate(top_actions.iterrows(), 1):
                            success_rate = action_row['성공률(%)']
                            action_name = action_row['조치']
                            
                            # 성공률에 따른 색상 결정
                            if success_rate >= 80:
                                color = "#4caf50"  # 초록
                                icon = "🟢"
                            elif success_rate >= 60:
                                color = "#ff9800"  # 주황
                                icon = "🟡"
                            else:
                                color = "#f44336"  # 빨강
                                icon = "🔴"
                            
                            st.markdown(f"""
                            <div style="background-color: white; border-left: 4px solid {color}; 
                                        padding: 15px; margin: 10px 0; border-radius: 5px; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <strong>{icon} {idx}. {action_name}</strong><br>
                                <span style="color: {color}; font-weight: bold;">성공률: {success_rate}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("해당 문제 원인에 대한 조치 데이터가 부족합니다.")
                
                # 장비별 AI 인사이트
                st.markdown("#### 🤖 장비별 AI 분석")
                
                with st.spinner(f"{selected_equip} 데이터를 분석하고 있습니다..."):
                    prompt_equip = f"""
다음은 {selected_equip} 장비의 정비 데이터 분석 결과입니다:

- 총 고장 건수: {len(df_filtered)}건
- 주요 문제 원인: {', '.join(cause_counts.head(3).index)}
- 문제 발생 빈도: {dict(cause_counts.head(3))}

이 장비의 특성과 문제 패턴을 바탕으로 예방 정비 방안을 3-4문장으로 제안해주세요.
                    """
                    
                    equip_insight = llm.predict(prompt_equip)
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h4 style="margin-top: 0;">🔧 {selected_equip} 맞춤 분석</h4>
                        <p style="margin-bottom: 0; line-height: 1.6;">{equip_insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # 전체 장비 선택 시
            st.markdown("##### 📋 전체 장비 현황")
            
            # 장비별 고장 건수 테이블
            equip_summary = df.groupby('모델').agg({
                '정비노트': 'count',
                '문제원인': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).rename(columns={
                '정비노트': '고장건수',
                '문제원인': '주요문제원인'
            }).sort_values('고장건수', ascending=False)
            
            st.dataframe(
                equip_summary,
                use_container_width=True,
                height=400
            )
            
            st.info("👆 특정 장비를 선택하시면 더 상세한 분석을 확인할 수 있습니다.")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 <strong>HERO</strong> - Hynix Equipment Response Operator</p>
    <p>AI 기반 반도체 장비 정비 솔루션 | Powered by OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)
