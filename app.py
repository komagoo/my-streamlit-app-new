import os
import sys
import streamlit as st
from dotenv import load_dotenv
import faiss
import json
from datetime import datetime, date, time
import sqlite3

# 환경 변수 로드 (.env)
load_dotenv()

# 일반 라이브러리
import pandas as pd
import re
from collections import defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go
import base64

# Langchain 관련
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 데이터베이스 초기화 함수
def init_database():
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    
    # 사용자 정보 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            contact TEXT,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 기본 계정들 생성 (관리자, 정비자)
    default_accounts = [
        ('admin123', 'admin123!@#', 'admin', '김철수', '010-1234-5678', 'IT부서'),
        ('maintainer123', 'maintainer123!@#', 'maintainer', '이영희', '010-9876-5432', '정비팀')
    ]
    
    for username, password, role, name, contact, department in default_accounts:
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password, role, name, contact, department) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, password, role, name, contact, department))
    
    # 정비노트 자동생성 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maintenance_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            lot_id TEXT NOT NULL,
            equipment_model TEXT NOT NULL,
            problem_cause TEXT NOT NULL,
            actions TEXT NOT NULL,
            generated_note TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# 로고 이미지 base64 인코딩
def get_base64_of_bin_file(bin_file_path):
    try:
        with open(bin_file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# 사용자 정보 저장/로드 함수
def save_user_profile(username, name, contact, department):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET name=?, contact=?, department=? WHERE username=?
    ''', (name, contact, department, username))
    conn.commit()
    conn.close()

def get_user_profile(username):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, contact, department, role FROM users WHERE username=?', (username,))
    result = cursor.fetchone()
    conn.close()
    return result if result else (None, None, None, None)

def authenticate_user(username, password):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT role FROM users WHERE username=? AND password=?', (username, password))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_maintenance_note(username, lot_id, equipment_model, problem_cause, actions_json, generated_note):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO maintenance_notes (username, lot_id, equipment_model, problem_cause, actions, generated_note)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, lot_id, equipment_model, problem_cause, actions_json, generated_note))
    conn.commit()
    conn.close()

# CSV 자동 저장 함수 개선
def save_to_csv(maintenance_data):
    """정비노트를 CSV 파일에 자동 저장"""
    csv_file = 'maintenance_notes.csv'
    
    # 새 데이터 준비
    new_row = {
        '정비일자': maintenance_data['date'],
        '정비시각': maintenance_data['time'], 
        'LOT_ID': maintenance_data['lot_id'],
        '장비모델': maintenance_data['equipment_model'],
        '정비자': maintenance_data['username'],
        '문제원인': maintenance_data['problem_cause'],
        '정비노트': maintenance_data['generated_note']
    }
    
    # CSV 파일이 존재하면 기존 데이터에 추가, 없으면 새로 생성
    try:
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file, encoding='utf-8')
            # 중복 체크 (같은 LOT_ID + 시간 조합)
            duplicate_check = existing_df[
                (existing_df['LOT_ID'] == new_row['LOT_ID']) & 
                (existing_df['정비일자'] == new_row['정비일자']) & 
                (existing_df['정비시각'] == new_row['정비시각'])
            ]
            
            if not duplicate_check.empty:
                return False, "동일한 LOT_ID와 시간의 정비노트가 이미 존재합니다."
            
            # 새 행 추가
            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([new_row])
        
        # CSV 저장
        updated_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 파일 경로 반환 개선
        abs_path = os.path.abspath(csv_file)
        return True, abs_path
        
    except Exception as e:
        return False, f"CSV 저장 중 오류 발생: {str(e)}"

# 데이터베이스 초기화
init_database()

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

# 커스텀 CSS (개선된 스타일)
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
    .hero-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .profile-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .case-example-card {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
    .file-location-info {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .maintenance-stats {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        color: #333;
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
if "user_role" not in st.session_state:
    st.session_state.user_role = None

# ----------------------------
# 1. 로그인 단계
# ----------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="text-align: center; color: #333; margin-bottom: 30px;">🔐 로그인</h2>
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #1976d2; margin: 0;">💡 테스트 계정 안내</h4>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>관리자:</strong> admin123 / admin123!@#</p>
                <p style="margin: 0; color: #424242;"><strong>정비자:</strong> maintainer123 / maintainer123!@#</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            # 역할 구분 선택
            role_option = st.selectbox("👥 역할 선택", ["관리자", "정비자"], key="role_select")
            username = st.text_input("👤 아이디", placeholder="아이디를 입력하세요")
            password = st.text_input("🔒 비밀번호", type="password", placeholder="비밀번호를 입력하세요")
            
            submitted = st.form_submit_button("🚀 로그인", use_container_width=True)
            
            if submitted:
                user_role = authenticate_user(username, password)
                
                if user_role:
                    # 선택한 역할과 DB의 역할이 일치하는지 확인
                    role_mapping = {"관리자": "admin", "정비자": "maintainer"}
                    if user_role == role_mapping[role_option] or user_role == "admin":  # 관리자는 모든 역할로 로그인 가능
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = user_role
                        st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
                        
                        # 사용자 이름 가져오기
                        user_profile = get_user_profile(username)
                        user_name = user_profile[0] if user_profile[0] else username
                        
                        st.success(f"✅ {user_name}님, 환영합니다! ({role_option})")
                        st.rerun()
                    else:
                        st.error("❌ 선택한 역할과 계정 권한이 일치하지 않습니다.")
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

# 사이드바 - 개선된 사용자 정보 표시
with st.sidebar:
    # 사용자 기본 정보 가져오기
    user_profile = get_user_profile(st.session_state.username)
    user_name, user_contact, user_dept, user_role = user_profile
    
    # 역할 표시 개선
    role_display = {
        'admin': '시스템 관리자',
        'maintainer': '정비 담당자'
    }.get(st.session_state.user_role, st.session_state.user_role)
    
    # 개선된 사용자 정보 표시
    st.markdown(f"""
    <div class="profile-section">
        <h4 style="margin: 0; color: white;">👋 환영합니다!</h4>
        <div style="margin-top: 15px;">
            <p style="margin: 5px 0; color: white; font-size: 1.1em;">
                <strong>{user_name if user_name else st.session_state.username}</strong>
            </p>
            <p style="margin: 5px 0; color: rgba(255,255,255,0.9); font-size: 0.9em;">
                📋 {role_display}
            </p>
            {f'<p style="margin: 5px 0; color: rgba(255,255,255,0.9); font-size: 0.9em;">🏢 {user_dept}</p>' if user_dept else ''}
            {f'<p style="margin: 5px 0; color: rgba(255,255,255,0.9); font-size: 0.9em;">📱 {user_contact}</p>' if user_contact else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 개인정보 입력 및 수정
    with st.expander("⚙️ 내 정보 수정", expanded=False):
        with st.form("profile_form"):
            st.markdown("**개인정보 입력/수정**")
            
            profile_name = st.text_input("이름", value=user_name or "", placeholder="이름을 입력하세요")
            profile_contact = st.text_input("연락처", value=user_contact or "", placeholder="010-0000-0000")
            profile_dept = st.text_input("소속", value=user_dept or "", placeholder="예: 정비1팀")
            
            if st.form_submit_button("💾 저장", use_container_width=True):
                save_user_profile(st.session_state.username, profile_name, profile_contact, profile_dept)
                st.success("✅ 개인정보가 저장되었습니다!")
                st.rerun()
    
    # 시스템 정보 추가
    st.markdown("---")
    st.markdown("### 📊 시스템 현황")
    
    # CSV 파일 위치 정보 표시
    csv_path = os.path.abspath('maintenance_notes.csv')
    if os.path.exists('maintenance_notes.csv'):
        csv_df = pd.read_csv('maintenance_notes.csv', encoding='utf-8')
        st.metric("저장된 정비노트", f"{len(csv_df)}건")
        st.caption(f"📁 CSV 위치: `{csv_path}`")
    else:
        st.info("아직 저장된 정비노트가 없습니다")
    
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
            3. 챗봇을 통해 정비 문제 해결책을 찾아보세요<br>
            4. 정비노트 작성 도우미를 활용해보세요
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

st.divider()

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
success_examples = defaultdict(list)

for cause, actions in cause_action_counts.items():
    for action, counts in actions.items():
        first_count = counts['first']
        second_count = counts['second']
        third_count = counts['third']
        total = first_count + second_count + third_count
        success = second_count + third_count
        success_rate = round(success / total * 100, 2) if total > 0 else 0

        # 성공 사례 저장
        if success > 0:
            example_notes = [note for note in note_map[(cause, action)] 
                           if any(pattern in note for pattern in ['정상 확인', '장비 업', '생산 재개'])]
            if example_notes:
                success_examples[action] = example_notes[:3]

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

# 장비별 정비 HERO 계산
equipment_hero = {}
if '모델' in df.columns and '정비자' in df.columns:
    for model in df['모델'].unique():
        model_data = df[df['모델'] == model]
        if not model_data.empty:
            top_maintainer = model_data['정비자'].value_counts()
            if len(top_maintainer) > 0:
                hero_name = top_maintainer.index[0]
                hero_count = top_maintainer.iloc[0]
                
                # 연락처 정보 가져오기
                hero_profile = get_user_profile(hero_name)
                hero_contact = hero_profile[1] if hero_profile[1] else "연락처 없음"
                
                equipment_hero[model] = {
                    'name': hero_name,
                    'count': hero_count,
                    'contact': hero_contact
                }

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
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI 정비 상담", "📊 정비 데이터 분석", "📝 정비노트 작성 도우미", "📂 저장된 노트 조회"])

# ----------------------------
# Tab1: AI 정비 상담 (개선됨)
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
                
                # 응답 생성 (개선: 유사 사례 간소화)
                top3_desc = "\n".join([f"{i+1}. {r['조치']} (성공률 {r['성공률']}%)"
                                     for i, r in enumerate(top3)])
                
                # 상세 사례 HTML (개선: 상세보기만 제공)
                top3_cases_html = ""
                for i, action in enumerate(top3, 1):
                    action_name = action["조치"]
                    if action_name in success_examples and success_examples[action_name]:
                        top3_cases_html += f"""
                        <details style="margin: 10px 0;">
                            <summary style="cursor: pointer; color: #007bff; font-weight: bold;">
                                📖 {i}번 조치 상세 사례 보기
                            </summary>
                            <div style="margin-top: 10px; font-size: 0.9em; background-color: #f0f0f0; padding: 15px; border-radius: 5px;">
                                {success_examples[action_name][0]}
                            </div>
                        </details>
                        """
                
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
{explanation}<br><br>
{top3_cases_html if top3_cases_html else ''}
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
            fig1.update_layout(
                showlegend=False, 
                margin=dict(l=10, r=50, t=10, b=10)
            )
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
            fig2.update_layout(
                showlegend=False, 
                margin=dict(l=10, r=50, t=10, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        
        # AI 인사이트
        st.markdown("#### 💡 핵심 인사이트")
        
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
        
        # 장비별 정비 HERO 표시
        if equipment_hero:
            st.divider()
            st.markdown("#### 🦸‍♂️ 이 장비는 내가 HERO!")
            
            hero_cols = st.columns(min(3, len(equipment_hero)))
            
            for idx, (model, hero_info) in enumerate(list(equipment_hero.items())[:3]):
                with hero_cols[idx % 3]:
                    st.markdown(f"""
                    <div class="hero-card">
                        <h4 style="margin: 0; color: #333;">🏆 {model}</h4>
                        <p style="margin: 5px 0; font-size: 1.1em;"><strong>{hero_info['name']}</strong></p>
                        <p style="margin: 0; font-size: 0.9em;">정비 횟수: {hero_info['count']}회</p>
                        <p style="margin: 0; font-size: 0.8em;">📞 {hero_info['contact']}</p>
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
                height=450
            )
            fig_pie1.update_traces(
                textposition='outside',
                textinfo='percent+label',
                textfont_size=11
            )
            fig_pie1.update_layout(
                showlegend=False,
                margin=dict(l=50, r=50, t=30, b=30)
            )
            st.plotly_chart(fig_pie1, use_container_width=True)
        
        with col2:
            st.markdown("##### 🔍 문제 원인 분포")
            total_cause = df['문제원인'].value_counts()
            
            fig_pie2 = px.pie(
                names=total_cause.index.tolist(),
                values=total_cause.values,
                hole=0.4,
                height=450
            )
            fig_pie2.update_traces(
                textposition='outside',
                textinfo='percent+label',
                textfont_size=11
            )
            fig_pie2.update_layout(
                showlegend=False,
                margin=dict(l=50, r=50, t=30, b=30)
            )
            st.plotly_chart(fig_pie2, use_container_width=True)
        
        st.divider()
        
        # 전체 현황 AI 인사이트
        st.markdown("#### 💡 전체 현황 분석")
        
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
                
                st.divider()
                
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
                    margin=dict(l=10, r=50, t=60, b=10)
                )
                st.plotly_chart(fig_equip, use_container_width=True)
                
                st.divider()
                
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
                                color = "#4caf50"
                                icon = "🟢"
                            elif success_rate >= 60:
                                color = "#ff9800"
                                icon = "🟡"
                            else:
                                color = "#f44336"
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
                
                st.divider()
                
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

# ----------------------------
# Tab3: 정비노트 작성 도우미 (개선됨)
# ----------------------------
with tab3:
    st.markdown("### 📝 정비노트 작성 도우미")
    
    st.markdown("""
    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: #2e7d32; margin: 0;">🎯 사용법</h4>
        <p style="margin: 10px 0 0 0; color: #1b5e20;">
            정비 정보를 입력하면 자동으로 표준화된 정비노트를 생성해드립니다!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("maintenance_note_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # 날짜 & 시각
            maintenance_date = st.date_input(
                "📅 정비 날짜",
                value=date.today(),
                help="정비를 수행한 날짜를 선택하세요"
            )
            
            maintenance_time = st.time_input(
                "⏰ 정비 시작 시각",
                value=time(9, 0),
                help="정비를 시작한 시간을 선택하세요"
            )
            
            # LOT ID
            lot_id = st.text_input(
                "🔖 LOT ID",
                placeholder="예: M1133097",
                help="정비 대상 LOT ID를 입력하세요"
            )
            
            # 장비 모델
            if '모델' in df.columns:
                equipment_models = ['선택하세요'] + sorted(df['모델'].unique().tolist())
            else:
                equipment_models = ['선택하세요', 'Etch Chamber A', 'Etch Chamber B', 'Deposition Tool', 'CVD System']
            
            equipment_model = st.selectbox(
                "🔧 장비 모델",
                equipment_models,
                help="정비 대상 장비 모델을 선택하세요"
            )
        
        with col2:
            # 문제 원인
            common_causes = ['직접 입력'] + problem_keywords
            cause_option = st.selectbox(
                "⚠️ 문제 원인 (일반적인 원인)",
                common_causes,
                help="일반적인 문제 원인을 선택하거나 직접 입력하세요"
            )
            
            if cause_option == '직접 입력':
                problem_cause = st.text_input(
                    "문제 원인 직접 입력",
                    placeholder="예: plasma ignition failure",
                    help="발생한 문제의 원인을 입력하세요"
                )
            else:
                problem_cause = cause_option
            
            # 1차 조치 (필수)
            first_action = st.text_input(
                "🛠️ 1차 조치 *",
                placeholder="예: RF generator 리셋 및 점검",
                help="첫 번째 시도한 조치를 입력하세요"
            )
            
            first_success = st.selectbox(
                "1차 조치 결과 *",
                ["성공", "실패"],
                index=1,
                help="1차 조치의 성공/실패 여부를 선택하세요"
            )
            
            # 동적으로 2차, 3차 조치 표시
            show_second = first_success == "실패"
            
            if show_second:
                st.markdown("---")
                second_action = st.text_input(
                    "🔧 2차 조치",
                    placeholder="예: matching unit 교체 진행",
                    help="2차 조치를 입력하세요 (1차 실패 시)"
                )
                
                if second_action:
                    second_success = st.selectbox(
                        "2차 조치 결과",
                        ["성공", "실패"],
                        help="2차 조치의 성공/실패 여부를 선택하세요"
                    )
                    
                    show_third = second_success == "실패"
                    
                    if show_third:
                        st.markdown("---")
                        third_action = st.text_input(
                            "🔩 3차 조치",
                            placeholder="예: plasma source 점검 및 connector 재연결",
                            help="3차 조치를 입력하세요 (2차 실패 시)"
                        )
                        
                        if third_action:
                            third_success = st.selectbox(
                                "3차 조치 결과",
                                ["성공", "실패"],
                                help="3차 조치의 성공/실패 여부를 선택하세요"
                            )
                else:
                    second_success = None
                    show_third = False
            else:
                second_action = None
                second_success = None
                show_third = False
                
            if show_third:
                pass
            else:
                third_action = None
                third_success = None
        
        # 기타 상황 설명
        additional_info = st.text_area(
            "📝 기타 상황 설명",
            placeholder="예: kit 수급 대기 중, 부품 입고 예정, 추가 점검 필요 등",
            help="기타 특별한 상황이나 대기 사항을 입력하세요",
            height=100
        )
        
        # 생성 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.form_submit_button(
                "📋 정비노트 자동 생성",
                use_container_width=True,
                help="입력한 정보를 바탕으로 정비노트를 생성합니다"
            )
    
    # 입력 유효성 검사
    def validate_inputs():
        """입력값 유효성 검사"""
        errors = []
        
        if not lot_id:
            errors.append("LOT ID를 입력해주세요")
        elif len(lot_id) < 3:
            errors.append("LOT ID는 3자리 이상 입력해주세요")
            
        if equipment_model == '선택하세요':
            errors.append("장비 모델을 선택해주세요")
            
        if not problem_cause:
            errors.append("문제 원인을 입력해주세요")
            
        if not first_action:
            errors.append("1차 조치를 입력해주세요")
            
        # 시간 검증
        current_time = datetime.now().time()
        if maintenance_date == date.today() and maintenance_time > current_time:
            errors.append("미래 시간은 선택할 수 없습니다")
            
        return errors
    
    # 정비노트 생성 로직
    if generate_button:
        validation_errors = validate_inputs()
        if validation_errors:
            st.error("❌ 입력 오류:\n" + "\n".join([f"• {error}" for error in validation_errors]))
        else:
            # 액션 정보 정리
            actions_data = {
                "first": {"action": first_action, "success": first_success},
                "second": {"action": second_action, "success": second_success} if second_action else None,
                "third": {"action": third_action, "success": third_success} if third_action else None
            }
            
            with st.spinner("🤖 AI가 정비노트를 생성하고 있습니다..."):
                # 시간 포맷팅
                datetime_str = f"{maintenance_date.strftime('%m월%d일')} {maintenance_time.strftime('%H:%M')}"
                
                # 프롬프트 생성
                prompt = f"""
다음 정보를 바탕으로 반도체 장비 정비노트를 자연스럽게 작성해주세요:

날짜/시간: {datetime_str}
LOT ID: {lot_id}
장비 모델: {equipment_model}
문제 원인: {problem_cause}

1차 조치: {first_action} → {'정상' if first_success == '성공' else '여전히 이상 발생'}
"""

                if second_action:
                    prompt += f"2차 조치: {second_action} → {'정상' if second_success == '성공' else '여전히 이상 발생'}\n"
                
                if third_action:
                    prompt += f"3차 조치: {third_action} → {'정상' if third_success == '성공' else '여전히 이상 발생'}\n"
                
                if additional_info:
                    prompt += f"추가 상황: {additional_info}\n"

                prompt += """
다음 형식으로 자연스러운 정비노트를 작성해주세요:
- 각 시점별 상황을 시간 순서대로 서술
- 한 문단당 1개 이벤트
- 조치 이후 결과를 자연스럽게 포함
- 최종 결과 (정상/장비업/생산재개 등)를 명시

예시 형식:
08월07일 09:15 M1133097 LOT 진행 중 plasma ignition failure 발생 → 장비 멈춤
08월07일 10:00 1차 조치: RF generator 리셋 및 점검 → 여전히 이상 발생
...
"""
                
                try:
                    generated_note = llm.predict(prompt)
                    
                    # DB에 저장
                    save_maintenance_note(
                        st.session_state.username,
                        lot_id,
                        equipment_model,
                        problem_cause,
                        json.dumps(actions_data),
                        generated_note
                    )
                    
                    # CSV 자동 저장
                    maintenance_data = {
                        'date': maintenance_date.strftime('%Y-%m-%d'),
                        'time': maintenance_time.strftime('%H:%M'),
                        'lot_id': lot_id,
                        'equipment_model': equipment_model,
                        'username': st.session_state.username,
                        'problem_cause': problem_cause,
                        'generated_note': generated_note
                    }
                    
                    csv_success, csv_path = save_to_csv(maintenance_data)
                    
                    # 결과 표시
                    st.markdown("#### ✅ 생성된 정비노트")
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745;">
                        <h5 style="color: #28a745; margin-top: 0;">📋 {lot_id} 정비노트</h5>
                        <div style="white-space: pre-line; line-height: 1.6; color: #333;">
{generated_note}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 저장 위치 안내 (개선됨)
                    if csv_success:
                        st.success("✅ 정비노트가 성공적으로 생성되었습니다!")
                        st.markdown(f"""
                        <div class="file-location-info">
                            <h5 style="color: #2196F3; margin: 0;">📄 파일 저장 위치</h5>
                            <p style="margin: 10px 0 0 0; font-family: monospace; background: white; padding: 8px; border-radius: 4px;">
                                {csv_path}
                            </p>
                            <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">
                                위 경로에서 CSV 파일을 확인하실 수 있습니다.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ 정비노트가 성공적으로 생성되었습니다!")
                        st.warning(f"⚠️ CSV 저장 실패: {csv_path}")
                    
                    st.divider()
                    
                    # 복사 버튼
                    st.markdown("##### 📋 복사용 텍스트")
                    st.text_area(
                        "아래 텍스트를 복사해서 사용하세요",
                        value=generated_note,
                        height=200,
                        help="Ctrl+A로 전체 선택 후 Ctrl+C로 복사하세요"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 정비노트 생성 중 오류가 발생했습니다: {str(e)}")

# ----------------------------
# Tab4: 저장된 노트 조회 (새로 추가)
# ----------------------------
with tab4:
    st.markdown("### 📂 저장된 정비노트 조회")
    
    # CSV 파일 확인
    csv_file = 'maintenance_notes.csv'
    
    if os.path.exists(csv_file):
        csv_df = pd.read_csv(csv_file, encoding='utf-8')
        
        # 통계 표시
        st.markdown("""
        <div class="maintenance-stats">
            <h4 style="margin: 0;">📊 저장된 정비노트 현황</h4>
            <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                <div style="text-align: center;">
                    <h2 style="margin: 0;">{}</h2>
                    <p style="margin: 5px 0 0 0;">총 노트 수</p>
                </div>
                <div style="text-align: center;">
                    <h2 style="margin: 0;">{}</h2>
                    <p style="margin: 5px 0 0 0;">등록 장비</p>
                </div>
                <div style="text-align: center;">
                    <h2 style="margin: 0;">{}</h2>
                    <p style="margin: 5px 0 0 0;">정비 담당자</p>
                </div>
            </div>
        </div>
        """.format(
            len(csv_df),
            csv_df['장비모델'].nunique() if '장비모델' in csv_df.columns else 0,
            csv_df['정비자'].nunique() if '정비자' in csv_df.columns else 0
        ), unsafe_allow_html=True)
        
        st.divider()
        
        # 필터링 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if '장비모델' in csv_df.columns:
                filter_equipment = st.selectbox(
                    "장비 필터",
                    ["전체"] + sorted(csv_df['장비모델'].unique().tolist()),
                    key="filter_equip"
                )
            else:
                filter_equipment = "전체"
        
        with col2:
            if '정비자' in csv_df.columns:
                filter_maintainer = st.selectbox(
                    "정비자 필터",
                    ["전체"] + sorted(csv_df['정비자'].unique().tolist()),
                    key="filter_main"
                )
            else:
                filter_maintainer = "전체"
        
        with col3:
            if '정비일자' in csv_df.columns:
                csv_df['정비일자'] = pd.to_datetime(csv_df['정비일자'], errors='coerce')
                date_filter = st.date_input(
                    "날짜 필터",
                    value=None,
                    key="filter_date"
                )
            else:
                date_filter = None
        
        # 필터 적용
        filtered_df = csv_df.copy()
        
        if filter_equipment != "전체" and '장비모델' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['장비모델'] == filter_equipment]
        
        if filter_maintainer != "전체" and '정비자' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['정비자'] == filter_maintainer]
        
        if date_filter and '정비일자' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['정비일자'].dt.date == date_filter]
        
        # 결과 표시
        st.markdown(f"#### 📋 검색 결과 ({len(filtered_df)}건)")
        
        if not filtered_df.empty:
            # 미리보기 컬럼 추가
            display_df = filtered_df.copy()
            if '정비노트' in display_df.columns:
                display_df['노트 미리보기'] = display_df['정비노트'].apply(
                    lambda x: str(x)[:100] + '...' if len(str(x)) > 100 else str(x)
                )
            
            # 표시할 컬럼 선택
            display_columns = ['정비일자', '정비시각', 'LOT_ID', '장비모델', '정비자', '문제원인', '노트 미리보기']
            available_columns = [col for col in display_columns if col in display_df.columns]
            
            st.dataframe(
                display_df[available_columns],
                use_container_width=True,
                height=400
            )
            
            # 상세 보기
            st.divider()
            st.markdown("##### 🔍 상세 보기")
            
            selected_index = st.selectbox(
                "상세히 볼 정비노트 선택",
                filtered_df.index,
                format_func=lambda x: f"{filtered_df.loc[x, 'LOT_ID']} - {filtered_df.loc[x, '정비일자']}"
            )
            
            if selected_index is not None:
                selected_row = filtered_df.loc[selected_index]
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <h5 style="color: #333; margin-top: 0;">📋 정비노트 상세</h5>
                    <div style="margin: 10px 0;">
                        <strong>LOT ID:</strong> {selected_row.get('LOT_ID', 'N/A')}<br>
                        <strong>장비:</strong> {selected_row.get('장비모델', 'N/A')}<br>
                        <strong>정비자:</strong> {selected_row.get('정비자', 'N/A')}<br>
                        <strong>일시:</strong> {selected_row.get('정비일자', 'N/A')} {selected_row.get('정비시각', 'N/A')}<br>
                        <strong>문제원인:</strong> {selected_row.get('문제원인', 'N/A')}
                    </div>
                    <hr>
                    <div style="white-space: pre-line; line-height: 1.6;">
                        {selected_row.get('정비노트', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("검색 조건에 맞는 정비노트가 없습니다.")
        
        # CSV 다운로드 버튼
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            csv_data = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv_data,
                file_name=f"maintenance_notes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # 파일 경로 정보
        abs_path = os.path.abspath(csv_file)
        st.caption(f"📁 원본 파일 위치: `{abs_path}`")
        
    else:
        st.info("아직 저장된 정비노트가 없습니다. 정비노트 작성 도우미를 사용해 첫 번째 노트를 생성해보세요!")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 <strong>HERO</strong> - Hynix Equipment Response Operator</p>
</div>
""", unsafe_allow_html=True)