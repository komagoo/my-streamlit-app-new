import os
import sys
import json
import base64
import re
import faiss
import sqlite3
from datetime import datetime, date, time
from collections import defaultdict, Counter

# --- Streamlit & Data ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- LangChain / RAG (community) ---
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore import InMemoryDocstore

# =========================
# 0) 공통 설정
# =========================
st.set_page_config(
    page_title="🚀 HERO - Hynix Equipment Response Operator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 1) DB 초기화/유틸
# =========================
def init_database():
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()

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

    default_accounts = [
        ('admin123', 'admin123!@#', 'admin', '김철수', '010-1234-5678', 'IT부서'),
        ('etch123', 'etch123!@#', 'maintainer', '이영희', '010-2222-2222', 'Etch팀'),
        ('photo123', 'photo123!@#', 'maintainer', '삼영희', '010-3333-3333', 'Photo팀'),
        ('diff123', 'diff123!@#', 'maintainer', '사영희', '010-4444-4444', 'Diffusion팀'),
        ('thin123', 'thin123!@#', 'maintainer', '오영희', '010-5555-5555', 'Thin Film팀'),
        ('cc123', 'cc123!@#', 'maintainer', '육영희', '010-6666-6666', 'C&C팀'),
        ('yield123', 'yield123!@#', 'maintainer', '칠영희', '010-7777-7777', '수율팀')
    ]
    for username, password, role, name, contact, department in default_accounts:
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password, role, name, contact, department)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, password, role, name, contact, department))

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

def get_user_profile(username):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, contact, department, role FROM users WHERE username=?', (username,))
    result = cursor.fetchone()
    conn.close()
    return result if result else (None, None, None, None)

def save_user_profile(username, name, contact, department):
    conn = sqlite3.connect('hero_data.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET name=?, contact=?, department=? WHERE username=?',
                   (name, contact, department, username))
    conn.commit()
    conn.close()

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

def save_to_csv(maintenance_data):
    csv_file = 'maintenance_notes.csv'
    new_row = {
        '정비일자': maintenance_data['date'],
        '정비시각': maintenance_data['time'],
        'LOT_ID': maintenance_data['lot_id'],
        '장비모델': maintenance_data['equipment_model'],
        '정비자': maintenance_data['username'],
        '문제원인': maintenance_data['problem_cause'],
        '정비노트': maintenance_data['generated_note']
    }
    try:
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file, encoding='utf-8')
            duplicate_check = existing_df[
                (existing_df['LOT_ID'] == new_row['LOT_ID']) &
                (existing_df['정비일자'] == new_row['정비일자']) &
                (existing_df['정비시각'] == new_row['정비시각'])
            ]
            if not duplicate_check.empty:
                return False, "동일한 LOT_ID와 시간의 정비노트가 이미 존재합니다."
            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([new_row])
        updated_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        return True, os.path.abspath(csv_file)
    except Exception as e:
        return False, f"CSV 저장 중 오류 발생: {str(e)}"

def get_base64_of_bin_file(path):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

init_database()

# =========================
# 2) 로그인
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="text-align: center; color: #333; margin-bottom: 30px;">🔐 로그인</h2>
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #1976d2; margin: 0;">💡 테스트 계정 안내</h4>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>관리자:</strong> admin123 / admin123!@#</p>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>Etch:</strong> etch123 / etch123!@#</p>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>Photo:</strong> photo123 / photo123!@#</p>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>Diffusion:</strong> diff123 / diff123!@#</p>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>Thin Film:</strong> thin123 / thin123!@#</p>
                <p style="margin: 10px 0 5px 0; color: #424242;"><strong>C&C:</strong> cc123 / cc123!@#</p>
                <p style="margin: 0; color: #424242;"><strong>수율:</strong> yield123 / yield123!@#</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            role_option = st.selectbox("👥 부서 선택", ["관리자", "Etch", "Photo", "Diffusion", "Thin Film", "C&C", "수율"], key="role_select")
            username = st.text_input("👤 아이디", placeholder="아이디를 입력하세요")
            password = st.text_input("🔒 비밀번호", type="password", placeholder="비밀번호를 입력하세요")
            submitted = st.form_submit_button("🚀 로그인", use_container_width=True)

            if submitted:
                user_role = authenticate_user(username, password)
                if user_role:
                    # 부서별 계정 매핑 확인
                    department_mapping = {
                        "관리자": ["admin123"],
                        "Etch": ["etch123"],
                        "Photo": ["photo123"],
                        "Diffusion": ["diff123"],
                        "Thin Film": ["thin123"],
                        "C&C": ["cc123"],
                        "수율": ["yield123"]
                    }
                    
                    # 관리자는 모든 부서 선택 가능, 그 외는 해당 부서만 가능
                    if user_role == "admin" or (role_option in department_mapping and username in department_mapping[role_option]):
                        # OPENAI KEY: secrets 우선, 없으면 env
                        st.session_state.api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
                        if not st.session_state.api_key:
                            st.error("⚠️ OpenAI API 키가 설정되어 있지 않습니다. st.secrets 또는 환경변수에 등록하세요.")
                            st.stop()

                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = user_role
                        st.session_state.selected_department = role_option
                        st.success(f"✅ {username}님, 환영합니다! ({role_option})")
                        st.rerun()
                    else:
                        st.error("❌ 선택한 부서와 계정 권한이 일치하지 않습니다.")
                else:
                    st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# OPENAI KEY 적용
os.environ["OPENAI_API_KEY"] = st.session_state.api_key


# ========================= 
# 3) 헤더
# =========================

# Streamlit 기본 여백 제거
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        .main .block-container {
            max-width: 100%;
            padding-top: 1rem;
        }
        /* 상단 여백 완전 제거 */
        .stApp > header {
            background-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

# 로고를 헤더에 포함시키기
logo_path = "Hero_logo.png"
logo_b64 = get_base64_of_bin_file(logo_path)
if logo_b64:
    # 그림자 제거로 자연스러운 하나의 이미지로
    st.markdown(f"""
        <div style="
            margin-top: 30px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 0px;
            min-height: auto;
        ">
            <img src="data:image/png;base64,{logo_b64}" style="
                max-width: 100%; 
                height: auto;
                max-height: 234px;
                display: block;
                object-fit: contain;
            ">
        </div>
    """, unsafe_allow_html=True)
else:
    # 로고가 없는 경우 - 기본 헤더
    st.markdown("""
        <div style="background: linear-gradient(90deg, #ff4b4b, #ff6b6b); color:white; padding:20px; border-radius:10px; text-align:center; margin-top: -8px; margin-bottom:30px;">
            <h1 style="margin:0; font-size:2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🚀 HERO</h1>
            <p style="margin:0; font-size:1.2rem; opacity:0.9; font-weight: 300;">Hynix Equipment Response Operator</p>
        </div>
    """, unsafe_allow_html=True)

# =========================
with st.sidebar:
    from datetime import datetime
    now = datetime.now()

    name, contact, dept, role = get_user_profile(st.session_state.username)
    role_display = {
        'admin': '시스템 관리자', 
        'maintainer': '정비 담당자'
    }.get(st.session_state.user_role, st.session_state.user_role)
    
    selected_dept = st.session_state.get('selected_department', dept)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%); 
                color:white; padding:20px; border-radius:15px; margin-bottom:25px;
                box-shadow: 0 4px 15px rgba(255,75,75,0.25);
                position:relative; overflow:hidden;">
        <div style="position:absolute; top:-20px; right:-20px; width:60px; height:60px; 
                    background:rgba(255,255,255,0.1); border-radius:50%; opacity:0.7;"></div>
        <div style="text-align:center; font-size:0.9rem; opacity:0.9; margin-bottom:8px;">
            📅 {now.strftime('%Y년 %m월 %d일')}
        </div>
        <h4 style="margin:0 0 12px 0; font-size:1.1rem; text-align:center;">
            <span style="margin-right:8px;">👋</span> 환영합니다!
        </h4>
        <div style="background:rgba(255,255,255,0.15); padding:12px; border-radius:10px; margin-bottom:10px;">
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:6px;">{name if name else st.session_state.username}</div>
            <div style="font-size:0.85rem; opacity:0.95; margin-bottom:3px;">📋 {role_display}</div>
            <div style="font-size:0.85rem; opacity:0.95; margin-bottom:3px;">🏢 {selected_dept if selected_dept else dept}</div>
            {f'<div style="font-size:0.85rem; opacity:0.95;">📱 {contact}</div>' if contact else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)



    # 내 정보 수정 - 더 직관적이고 예쁘게
    with st.expander("⚙️ 내 정보 수정", expanded=False):
        st.markdown("""
        <div style="background:#f8f9fa; padding:10px; border-radius:8px; margin-bottom:15px; border-left:3px solid #ff4b4b;">
            <small style="color:#666;">💡 <b>개인정보를 업데이트</b>하여 더 나은 서비스를 받으세요</small>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("profile_form"):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("👤")
            with col2:
                in_name = st.text_input("이름", value=name or "", placeholder="홍길동", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("📱")
            with col2:
                in_contact = st.text_input("연락처", value=contact or "", placeholder="010-1234-5678", label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("🏢")
            with col2:
                in_dept = st.text_input("소속", value=dept or "", placeholder="상세 부서명", label_visibility="collapsed")
            
            if st.form_submit_button("💾 저장하기", use_container_width=True, type="primary"):
                save_user_profile(st.session_state.username, in_name, in_contact, in_dept)
                st.success("✅ 저장완료!")
                st.balloons()  # 재미있는 효과 추가
                st.rerun()

    # 구분선을 더 세련되게
    st.markdown("""
    <div style="margin:30px 0; text-align:center;">
        <div style="height:1px; background:linear-gradient(to right, transparent, #ddd, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 시스템 현황 - 더 시각적으로
    st.markdown("### 📊 시스템 현황")
    csv_path = os.path.abspath('maintenance_notes.csv')
    if os.path.exists('maintenance_notes.csv'):
        csv_df_side = pd.read_csv('maintenance_notes.csv', encoding='utf-8')
        record_count = len(csv_df_side)
        
        # 멋진 통계 카드
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
                    padding:20px; border-radius:15px; 
                    border:2px solid rgba(255,75,75,0.1);
                    box-shadow:0 4px 15px rgba(0,0,0,0.05); 
                    margin-bottom:20px; text-align:center;">
            <div style="color:#ff4b4b; font-size:2.2rem; font-weight:bold; margin-bottom:5px;">
                {record_count}
            </div>
            <div style="color:#666; font-size:0.9rem; font-weight:500;">
                저장된 정비노트
            </div>
            <div style="margin-top:10px; padding:8px; background:rgba(255,75,75,0.1); 
                        border-radius:20px; color:#ff4b4b; font-size:0.8rem; font-weight:600;">
                ✅ 시스템 정상 운영중
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 파일 정보는 접을 수 있게
        with st.expander("🔍 상세 정보 보기", expanded=False):
            st.caption("📁 **파일 위치**")
            st.code(csv_path, language=None)
            
            # 추가 통계 정보
            try:
                import os
                from datetime import datetime
                file_size = os.path.getsize('maintenance_notes.csv')
                mtime = os.path.getmtime('maintenance_notes.csv')
                last_modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("파일 크기", f"{file_size:,} bytes")
                with col2:
                    st.metric("최근 업데이트", last_modified)
            except:
                pass
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                    padding:25px; border-radius:15px; 
                    border:2px dashed #ddd;
                    text-align:center; margin-bottom:20px;">
            <div style="color:#999; font-size:3rem; margin-bottom:10px;">📝</div>
            <div style="color:#666; font-size:0.9rem; font-weight:500;">
                아직 저장된 정비노트가 없습니다
            </div>
            <div style="color:#999; font-size:0.8rem; margin-top:5px;">
                첫 번째 정비 작업을 시작해보세요!
            </div>
        </div>
        """, unsafe_allow_html=True)



    # 로그아웃 버튼 - 눈에 띄면서도 위험하지 않게
    st.markdown("""
    <div style="margin:30px 0; text-align:center;">
        <div style="height:1px; background:linear-gradient(to right, transparent, #ddd, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 로그아웃", use_container_width=True, help="현재 세션을 종료합니다"):
        st.warning("로그아웃 중...")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# =========================
# 5) 파일 업로드 및 데이터 로드
# =========================

# 파일 저장을 위한 디렉토리 설정 및 생성
UPLOAD_DIR = "uploaded_data"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 역할과 사용자에 따라 불러올 수 있는 파일 목록을 가져오는 함수
def get_available_files(role, username):
    """
    사용자의 역할에 따라 접근 가능한 파일 목록을 반환합니다.
    - admin: 'admin_'으로 시작하는 모든 파일을 볼 수 있습니다.
    - maintainer: 'maintainer_[본인username]_'으로 시작하는 파일만 볼 수 있습니다.
    """
    files = []
    # 디렉토리 내의 모든 파일 목록을 가져옴
    all_files = os.listdir(UPLOAD_DIR)
    
    if role == 'admin':
        # 관리자는 'admin_'으로 시작하는 모든 파일을 필터링
        for filename in all_files:
            if filename.startswith("admin_") and filename.endswith(".xlsx"):
                files.append(filename)
    else:  # maintainer
        # 정비자는 'maintainer_[본인username]_'으로 시작하는 파일을 필터링
        prefix = f"maintainer_{username}_"
        for filename in all_files:
            if filename.startswith(prefix) and filename.endswith(".xlsx"):
                files.append(filename)
    
    return files

st.markdown("### 📁 데이터 로드 및 업로드")

# 현재 로그인한 사용자 정보 가져오기
username = st.session_state.username
role = st.session_state.user_role

# 파일 객체를 담을 변수 초기화
loaded_file_data = None 

# UI를 두 개의 열로 나눔
col1, col2 = st.columns(2)

# --- 왼쪽 열: 파일 업로드 ---
with col1:
    st.subheader("새 파일 업로드")
    uploaded_file = st.file_uploader(
        "엑셀 파일을 업로드하세요 (.xlsx)",
        type=["xlsx"],
        key="file_uploader_widget"
    )
    if uploaded_file is not None:
        # 파일 저장 로직
        safe_filename = re.sub(r'[\\/*?:"<>|]', "", uploaded_file.name)
        save_filename = f"{role}_{username}_{safe_filename}"
        save_path = os.path.join(UPLOAD_DIR, save_filename)
        
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 파일 '{save_filename}'이(가) 서버에 저장되었습니다. 페이지가 새로고침됩니다.")
            st.rerun()
        else:
            st.warning("이미 같은 이름의 파일이 존재합니다. 다른 이름을 사용해주세요.")
            loaded_file_data = uploaded_file

# --- 오른쪽 열: 저장된 파일 불러오기 ---
with col2:
    st.subheader("저장된 파일 불러오기")
    available_files = get_available_files(role, username)
    options = ["--- 파일을 선택하세요 ---"] + sorted(available_files)
    selected_option = st.selectbox(
        "불러올 파일을 선택하세요.",
        options,
        index=0,
        help="관리자는 모든 관리자 파일을, 정비자는 본인의 파일만 볼 수 있습니다.",
        key="file_selector_widget"
    )

    if selected_option != "--- 파일을 선택하세요 ---":
        file_path = os.path.join(UPLOAD_DIR, selected_option)
        try:
            loaded_file_data = open(file_path, "rb")
            st.info(f"💾 저장된 파일 '{selected_option}'을(를) 불러왔습니다.")
        except FileNotFoundError:
            st.error("파일을 찾을 수 없습니다. 삭제되었을 수 있습니다.")
            loaded_file_data = None

# 데이터 처리 로직은 loaded_file_data를 사용하도록 통일
if loaded_file_data is None:
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0;">
        <h3 style="color: #1976d2; margin-bottom: 15px;">📋 사용 방법</h3>
        <p style="color: #424242; font-size: 16px; line-height: 1.6;">
            1. 왼쪽에 있는 <b>'새 파일 업로드'</b>를 통해 정비 데이터를 업로드 하거나,<br>
            2. 오른쪽에 있는 목록에서 이전에 업로드한 파일을 선택하세요.<br>
            3. HERO가 데이터를 분석하여 인사이트를 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# 데이터 로딩 및 전처리
with st.spinner("📊 데이터를 분석하고 있습니다..."):
    df = pd.read_excel(loaded_file_data)
    
    if hasattr(loaded_file_data, 'close'):
        loaded_file_data.close()

    if '정비일자' in df.columns:
        df['정비일자'] = pd.to_datetime(df['정비일자'], errors='coerce')
    
    df = df.dropna(subset=['정비노트'])
    
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

# =========================
# 6) 공용: 문제원인 추출(단일화, 전역 1회만 계산)

# =========================
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROBLEM_KEYWORDS = [
    "wafer not",
    "plasma ignition failure",
    "pumpdown 시간 지연",
    "mass flow controller 이상",
    "etch residue over spec",
    "temperature abnormal",
    "slot valve 동작 불량",
    "chamber pressure fluctuation",
    "he flow deviation",
    "RF auto match 불량",
]

# 모든 별칭을 한 군데에서 표준화
ALIAS_MAP = {
    # wafer not
    "웨이퍼": "wafer not",
    "웨이퍼 낫": "wafer not",
    "wafer not 감지됨": "wafer not",
    "wafer not 발생": "wafer not",
    "wafer not 발생 확인": "wafer not",

    # plasma ignition
    "플라즈마": "plasma ignition failure",
    "플라즈마 점화": "plasma ignition failure",
    "점화 불량": "plasma ignition failure",
    "ignition fail": "plasma ignition failure",
    "이그니션": "plasma ignition failure",

    # pumpdown
    "pump down delay": "pumpdown 시간 지연",
    "펌프다운 지연": "pumpdown 시간 지연",

    # MFC
    "mfc 이상": "mass flow controller 이상",
    "mass flow 이상": "mass flow controller 이상",

    # etch residue
    "etch residue": "etch residue over spec",
    "에칭 레지듀": "etch residue over spec",

    # temperature abnormal (핵심!)
    "temperature drift": "temperature abnormal",
    "temp drift": "temperature abnormal",
    "온도 드리프트": "temperature abnormal",

    # slot valve
    "slot valve 불량": "slot valve 동작 불량",
    "슬롯 밸브 불량": "slot valve 동작 불량",

    # chamber pressure
    "챔버 압력 변동": "chamber pressure fluctuation",
    "chamber pressure": "chamber pressure fluctuation",

    # HE flow
    "he flow dev": "he flow deviation",
    "he 유량 편차": "he flow deviation",

    # RF auto match
    "rf auto match fail": "RF auto match 불량",
    "오토매치": "RF auto match 불량",
}

SURFACES = PROBLEM_KEYWORDS + list(ALIAS_MAP.keys())

def _normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s

@st.cache_resource
def _get_kw_vectorizer():
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    vec.fit(SURFACES)
    return vec

_KW_VEC = _get_kw_vectorizer()

def _canon(surface: str) -> str:
    return ALIAS_MAP.get(surface, surface)

def predict_cause_unified(text: str) -> str:
    """전역 단일 라벨러: alias → 표준키워드 → 퍼지매칭(백업)"""
    low = _normalize_text(text)

    # 1) alias 매핑
    for alias, norm in ALIAS_MAP.items():
        if alias in low:
            return norm

    # 2) 표준 키워드 직접 포함
    for kw in PROBLEM_KEYWORDS:
        if kw.lower() in low:
            return kw

    # 3) 퍼지 백업
    qv = _KW_VEC.transform([low])
    kv = _KW_VEC.transform(SURFACES)
    idx = cosine_similarity(qv, kv)[0].argmax()
    return _canon(SURFACES[idx])

# ---- 전역 1회만 계산 (여기서 만든 '문제원인'을 이후에 절대 덮어쓰지 말 것!)
df["문제원인"] = df["정비노트"].apply(predict_cause_unified)

# =========================
# 7) 성공률 계산 
# =========================
from collections import defaultdict, Counter
import re

all_texts = [str(note).strip() for note in df["정비노트"]]

# 원인/액션 패턴 (조금 더 유연하게)
cause_pattern         = re.compile(r"(?:LOT\s*진행\s*중\s*)?(.+?)\s*(?:발생|감지|현상|알람)", re.IGNORECASE)
first_action_pattern  = re.compile(r"1차\s*조치[:：]?\s*(.+?)\s*→\s*여전히 이상 발생", re.IGNORECASE)
second_action_pattern = re.compile(r"정비\s*시작\.?\s*(.+?)\s*(?:진행)?$", re.IGNORECASE)
third_action_pattern  = re.compile(r"추가\s*조치[:：]?\s*(.+)$", re.IGNORECASE)

cause_action_counts = defaultdict(lambda: defaultdict(Counter))
note_map = defaultdict(list)

for note in all_texts:
    lines = [line.strip() for line in note.split("\n") if line.strip()]
    cause = None

    # 1) 줄 단위에서 원인 후보를 찾되, 반드시 전역 통일 함수로 표준화
    for line in lines:
        m = cause_pattern.search(line)
        if m:
            cause = predict_cause_unified(m.group(1).strip())
            break

    # 2) fallback: 패턴이 없으면 노트 전체를 넣어서 표준화
    if cause is None or not str(cause).strip():
        cause = predict_cause_unified(note)

    if not cause:
        continue

    # 3) 액션 카운트
    for line in lines:
        action = None
        m1 = first_action_pattern.search(line)
        m2 = second_action_pattern.search(line)
        m3 = third_action_pattern.search(line)

        if m1:
            action = m1.group(1).strip()
            cause_action_counts[cause][action]["first"] += 1
        elif m2:
            action = m2.group(1).strip()
            cause_action_counts[cause][action]["second"] += 1
        elif m3:
            action = m3.group(1).strip()
            cause_action_counts[cause][action]["third"] += 1

        if action:
            note_map[(cause, action)].append(note)

# 4) 집계: 2·3차를 성공으로 간주
rows = []
for cause, actions in cause_action_counts.items():
    for action, counts in actions.items():
        first_count  = counts["first"]
        second_count = counts["second"]
        third_count  = counts["third"]
        total   = first_count + second_count + third_count
        success = second_count + third_count
        success_rate = round(success / total * 100, 2) if total > 0 else 0

        rows.append({
            "대표원인": cause,   # ← df['문제원인']과 동일한 표준 라벨
            "조치": action,
            "총횟수": total,
            "실패횟수": first_count,
            "성공횟수": success,
            "성공률(%)": success_rate,
            "정비노트": note_map[(cause, action)][0] if note_map[(cause, action)] else "",
        })

df_success = pd.DataFrame(rows)


# =========================
# 8) 타임라인 파싱 → 2차/3차 작업시간 & 총 리드타임
# =========================
time_pattern = re.compile(r"(\d{2})월(\d{2})일\s(\d{2}):(\d{2})\s+(.*)")
WAIT_KEYWORDS = ["대기", "입고 예정"]
VERIFY_KEYWORDS = ["seasoning", "검사 정상", "장비 업", "정상"]
ACTION_HINTS = [
    "tuning", "점검", "교체", "재결합", "리셋", "조정", 
    "leak test", "재설정", "분리", "재조립"
]

def _parse_events(note: str, base_year: int):
    evs = []
    for m in time_pattern.findall(str(note)):
        mm, dd, HH, MM, tail = m
        evs.append((int(mm), int(dd), int(HH), int(MM), tail.strip()))
    if not evs: return []
    year = base_year
    norm = []
    prev_m = None
    for mm, dd, HH, MM, tail in evs:
        if prev_m is not None and mm < prev_m:
            year += 1
        ts = datetime(year, mm, dd, HH, MM)
        norm.append((ts, tail))
        prev_m = mm
    return norm

def _is_issue(t): return "LOT 진행 중" in t and "발생" in t
def _is_wait(t):  return any(k in t.lower() for k in [w.lower() for w in WAIT_KEYWORDS])
def _is_second(t): return t.startswith("정비 시작")
def _is_third(t):  return t.startswith("추가 조치")
#def _is_verify(t): return any(k in t.lower() for k in [w.lower() for w in VERIFY_KEYWORDS])
# 수정(문맥 기반): '정상'이 있어도 액션이면 verify로 보지 않음
def _has_action_verb(t: str) -> bool:
    low = t.lower()
    return any(h in low for h in [w.lower() for w in ACTION_HINTS])

def _is_verify(t: str) -> bool:
    low = t.lower()
    hit = any(k in low for k in [w.lower() for w in VERIFY_KEYWORDS])

    # 1) 액션 라인(정비 시작/추가 조치) 또는 액션 동사가 있으면 -> 검증 아님(=시간 카운트)
    if _is_second(t) or _is_third(t) or _has_action_verb(t):
        return False

    # 2) 진짜 검증 맥락(시즈닝/샘플 검사/결과/장비 업/생산 진행)만 True
    verify_context = any(s in low for s in [
        "seasoning", "sample lot", "검사", "결과", "장비 업", "생산 진행"
    ])
    return hit and verify_context

def _compute_times(note: str, base_year: int):
    events = _parse_events(note, base_year)
    if not events: return 0.0, 0.0, 0.0
    # lead time
    lead_start = None
    for ts, text in events:
        if _is_issue(text):
            lead_start = ts; break
    if lead_start is None: lead_start = events[0][0]
    lead_end = events[-1][0]
    lead_h = (lead_end - lead_start).total_seconds()/3600.0

    t2 = 0.0; t3 = 0.0
    for i in range(len(events)-1):
        ts, text = events[i]
        ts_next, _ = events[i+1]
        dt_h = (ts_next - ts).total_seconds()/3600.0
        if _is_wait(text) or _is_verify(text): continue
        if _is_second(text): t2 += dt_h
        elif _is_third(text): t3 += dt_h
    return round(t2,2), round(t3,2), round(lead_h,2)

second_list, third_list, lead_list = [], [], []
for _, row in df.iterrows():
    note = row["정비노트"]
    base_year = int(row["정비일자"].year) if "정비일자" in df.columns and pd.notna(row["정비일자"]) else 2025
    t2, t3, lead = _compute_times(note, base_year)
    second_list.append(t2); third_list.append(t3); lead_list.append(lead)
df["2차작업시간(h)"] = second_list
df["3차작업시간(h)"] = third_list
df["총리드타임(h)"] = lead_list

# =========================
# 9) 심각도(높음/중간/낮음)
# =========================
def _safe_minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.max()==s.min(): return pd.Series([0.0]*len(s), index=s.index)
    return (s - s.min())/(s.max()-s.min())

@st.cache_data
def build_severity_map(_df: pd.DataFrame) -> dict:
    if "문제원인" not in _df.columns:
        return {}
    agg = (
        _df.groupby("문제원인", dropna=False)
          .agg(
              건수=("문제원인", "size"),
              평균리드타임_h=("총리드타임(h)", "mean"),
              P75_리드타임_h=("총리드타임(h)", lambda x: x.quantile(0.75)),
              평균2차_h=("2차작업시간(h)", "mean"),
              평균3차_h=("3차작업시간(h)", "mean"),
          )
          .fillna(0.0)
    )
    agg["평균운영시간_h"] = agg["평균2차_h"] + agg["평균3차_h"]
    agg["N_평균리드"] = _safe_minmax(agg["평균리드타임_h"])
    agg["N_P75"]      = _safe_minmax(agg["P75_리드타임_h"])
    agg["N_건수"]     = _safe_minmax(agg["건수"])
    agg["N_운영시간"] = _safe_minmax(agg["평균운영시간_h"])
    agg["sev_score"] = 0.4*agg["N_평균리드"] + 0.3*agg["N_P75"] + 0.2*agg["N_건수"] + 0.1*agg["N_운영시간"]
    q33, q66 = agg["sev_score"].quantile([0.33, 0.66])
    def _label(x):
        if x <= q33: return "낮음"
        if x <= q66: return "중간"
        return "높음"
    return {cause: _label(s) for cause, s in agg["sev_score"].items()}

SEVERITY_LABEL_BY_CAUSE = build_severity_map(df)

# 조치명 → 평균작업시간(h)
sec_name_re = re.compile(r"정비\s*시작\.?\s*(.+?)(?:\s*진행|$)", re.IGNORECASE)
thr_name_re = re.compile(r"추가\s*조치[:：]?\s*(.+)", re.IGNORECASE)

def _get_second_action(text: str):
    m = sec_name_re.search(str(text)); return m.group(1).strip() if m else ""

def _get_third_action(text: str):
    m = thr_name_re.search(str(text)); return m.group(1).strip() if m else ""

_tmp = df.copy()
_tmp["2차조치명"] = _tmp["정비노트"].apply(_get_second_action)
_tmp["3차조치명"] = _tmp["정비노트"].apply(_get_third_action)

act2 = _tmp.loc[_tmp["2차조치명"]!="", ["2차조치명","2차작업시간(h)"]].rename(columns={"2차조치명":"조치명","2차작업시간(h)":"작업시간(h)"})
act3 = _tmp.loc[_tmp["3차조치명"]!="", ["3차조치명","3차작업시간(h)"]].rename(columns={"3차조치명":"조치명","3차작업시간(h)":"작업시간(h)"})
_actions_long_ = pd.concat([act2, act3], ignore_index=True)
_actions_long_["작업시간(h)"] = pd.to_numeric(_actions_long_["작업시간(h)"], errors="coerce")
_actions_long_ = _actions_long_.dropna(subset=["작업시간(h)"])
#_actions_long_ = _actions_long_[_actions_long_["작업시간(h)"]>0]
ACTION_AVG_H = _actions_long_.groupby("조치명")["작업시간(h)"].mean().round(2).to_dict()

# =========================
# 10) RAG with FAISS (B안: 파일 해시 같을 때만 재사용)
# =========================
INDEX_PATH = "faiss_index.index"
META_PATH  = "faiss_meta.json"

def _current_file_hash(uploaded_file) -> str:
    # 파일명 + 바이트 크기 기준 단순 해시
    name = getattr(uploaded_file, "name", "unknown")
    try:
        size = uploaded_file.getbuffer().nbytes
    except:
        size = 0
    return f"{name}:{size}"

def load_or_create_vectordb(documents, embedding_model, file_hash):
    # 캐시 메타 읽기
    cached = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except:
            cached = None

    # 캐시 재사용 조건: 해시 동일 && 인덱스 파일 존재
    if cached and cached.get("file_hash")==file_hash and os.path.exists(INDEX_PATH):
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
            return vectordb
        except Exception:
            pass  # 실패 시 새로 생성

    # 새로 생성
    vectordb = FAISS.from_documents(documents, embedding_model)
    faiss.write_index(vectordb.index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"file_hash": file_hash, "n_docs": len(documents)}, f, ensure_ascii=False)
    return vectordb

documents = [Document(page_content=str(row["정비노트"]), metadata={"row": idx}) for idx, row in df.iterrows()]
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

if "embedding_model" not in st.session_state or "vectordb" not in st.session_state:
    with st.spinner("🤖 AI 모델을 준비하고 있습니다..."):
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        file_hash = _current_file_hash(uploaded_file)
        vectordb = load_or_create_vectordb(split_docs, embedding_model, file_hash)
        st.session_state["embedding_model"] = embedding_model
        st.session_state["vectordb"] = vectordb
else:
    embedding_model = st.session_state["embedding_model"]
    vectordb = st.session_state["vectordb"]

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k":20}),
    return_source_documents=True
)

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

# =========================
# 11) 메인 탭 (탭 튐 방지: 라디오 기반 네비게이션)
# =========================
# =========================
# 11) 메인 탭 (튐 완전 방지: 위젯 key = 상태)
# =========================
# 1) 기본값 1회만 세팅
if "active_main" not in st.session_state:
    st.session_state.active_main = "🤖 AI 정비 상담"
if "active_analysis" not in st.session_state:
    st.session_state.active_analysis = "🏆 핵심 지표"

_main_options = ["🤖 AI 정비 상담", "📊 정비 데이터 분석", "📝 정비노트 작성 도우미", "📂 저장된 노트 조회"]

# 2) 라디오 값은 'active_main' key에 직접 저장 (index 사용 X)
st.radio(
    "메뉴",
    _main_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_main",          # ← 이 키가 값의 단일 출처
)
main = st.session_state.active_main

# ---- Tab1: 챗봇 ----


# ---- Tab1: 챗봇 ----
if main == "🤖 AI 정비 상담":
    st.markdown("### 🤖 HERO AI 상담사")

    # ===== 고정 인사말 =====
    GREET_HTML = """
    <div class="bot-message bubble">
        <strong>🤖 HERO</strong><br>
        안녕하세요! 반도체 장비 정비 전문 AI HERO입니다 👋<br><br>
        정비 문제를 입력하시면, 유사 사례를 분석해서 최적의 해결책을 제안해드려요!<br><br>
        💡 <strong>예시:</strong> wafer not | plasma ignition failure | slot valve 동작 불량
    </div>
    """

    # ===== CSS (말풍선/줄간격/리스트) =====
    st.markdown("""
    <style>
      .chat-wrap { display:flex; flex-direction:column; gap:8px; }
      .bubble {
        display:inline-block; width:fit-content; max-width:72%;
        padding:10px 12px; border-radius:12px;
        line-height:1.42; font-size:16.5px;
        box-shadow:0 1px 2px rgba(0,0,0,0.08);
        word-break:break-word; white-space:pre-wrap;
      }
      .bot-message  { background:#f5f7fb; color:#333; border:1px solid #e6e9f2; }
      .user-message { background:#e7f6ec; color:#1b5e20; border:1px solid #cdebd6; }
      .row { display:flex; }
      .row.bot  { justify-content:flex-start; }
      .row.user { justify-content:flex-end;  }
      .loading    { font-size:14px; padding:6px 10px; opacity:.9; }
      .bubble img { margin:4px 0; display:block; max-width:100%; height:auto; }
      ol.top3 { margin:4px 0 0 0; padding-left:20px; }
      ol.top3 li { margin:2px 0; line-height:1.35; }
      .meta { opacity:.8; font-size:14px; }
      ol.explain { margin:6px 0 0 18px; padding-left:18px; }
      ol.explain li { margin:6px 0; line-height:1.45; }
      ol.explain b { font-weight:600; }
      ol.explain .desc { display:block; margin-top:2px; }
    </style>
    """, unsafe_allow_html=True)

    # ===== 세션 =====
    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_top3" not in st.session_state: st.session_state.last_top3 = None
    if "last_sev_line_html" not in st.session_state: st.session_state.last_sev_line_html = ""

    # ===== 겹침 방지 placeholder =====
    chat_ph = st.empty()
    details_ph = st.empty()

    # ===== 입력창(아래) =====
    prompt = st.chat_input("정비 문제를 입력하세요")

    # ── 유틸: 조치 라벨 축약
    import re
    def short_label(s: str) -> str:
        if not isinstance(s, str): return ""
        s = re.sub(r"\s*\(.*?\)\s*$", "", s)
        s = s.replace("정비 시작.", "").replace("추가 조치:", "")
        s = s.replace("진행", "").replace("  ", " ").strip(" -•")
        return s.strip()[:60]

    # ── 유틸: LLM 설명이 코드블록으로 올 때 정리(백틱 제거)
    def strip_code_fences(text: str) -> str:
        if not isinstance(text, str): return ""
        x = text.strip()
        x = re.sub(r"^```[a-zA-Z0-9_-]*\\s*", "", x)
        x = re.sub(r"\\s*```$", "", x)
        x = x.replace("&lt;", "<").replace("&gt;", ">")
        return x.strip()

    # ===== 입력 처리 =====
    if prompt:
        st.session_state.messages = []
        st.session_state.last_top3 = None
        st.session_state.last_sev_line_html = ""
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 유저 말풍선 + 로딩(유저 아래 작은 봇 말풍선)
        with chat_ph.container():
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            st.markdown(GREET_HTML, unsafe_allow_html=True)
            st.markdown(f'<div class="row user"><div class="bubble user-message">👤 {prompt}</div></div>', unsafe_allow_html=True)
            status_ph = st.empty()
            with status_ph.container():
                st.markdown('<div class="row bot"><div class="bubble bot-message loading">🔍 검색중입니다...</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 검색/생성
        try:
            out = qa_chain.invoke({"query": prompt})
            docs = out.get("source_documents", [])

            # 다수결 원인 추정
            doc_causes = [predict_cause_unified(getattr(d, "page_content", "")) for d in docs if getattr(d, "page_content", "").strip()]
            if not doc_causes:
                doc_causes = [predict_cause_unified(prompt)]
            display_cause = Counter(doc_causes).most_common(1)[0][0]

            # 심각도 라인
            sev = SEVERITY_LABEL_BY_CAUSE.get(display_cause)
            sev_line_html = f"⚠️ 추정 문제원인: <b>{display_cause}</b>" + (f" — 심각도 <b>{sev}</b>" if sev else "")

            # 추천 수집
            recommended, seen = [], set()
            def _append_reco(note: str, row_s):
                key = (row_s["조치"], note)
                if key in seen: return
                seen.add(key)
                matched = df[df["정비노트"].astype(str).str.strip() == note]
                equip_id = matched["장비ID"].iloc[0] if "장비ID" in df.columns and not matched.empty else "N/A"
                model    = matched["모델"].iloc[0]    if "모델"    in df.columns and not matched.empty else "N/A"
                maint_tp = matched["정비종류"].iloc[0] if "정비종류" in df.columns and not matched.empty else "N/A"
                maint_ps = matched["정비자"].iloc[0]  if "정비자"  in df.columns and not matched.empty else "N/A"
                recommended.append({
                    "조치": row_s["조치"],
                    "성공률": row_s["성공률(%)"],
                    "평균작업시간(h)": ACTION_AVG_H.get(row_s["조치"]),
                    "정비노트": note,
                    "장비ID": equip_id, "모델": model,
                    "정비종류": maint_tp, "정비자": maint_ps
                })

            # 1차: 원인 일치
            for d in docs:
                note = getattr(d, "page_content", "").strip()
                if not note: continue
                if predict_cause_unified(note) != display_cause: continue
                for _, row_s in df_success.iterrows():
                    if row_s["조치"] in note: _append_reco(note, row_s)
            # 2차: 완화
            if not recommended:
                for d in docs:
                    note = getattr(d, "page_content", "").strip()
                    if not note: continue
                    for _, row_s in df_success.iterrows():
                        if row_s["조치"] in note: _append_reco(note, row_s)

            # Top3 선정 (최종 조치 우선)
            def is_final_action(note: str): return ("추가 조치" in note) or ("정상 확인" in note)
            finals = [r for r in recommended if is_final_action(r["정비노트"])]
            finals_sorted = sorted(finals, key=lambda x: x["성공률"], reverse=True)

            top3, used_actions, used_notes = [], set(), set()
            for r in finals_sorted:
                if r["조치"] not in used_actions and r["정비노트"] not in used_notes:
                    top3.append(r); used_actions.add(r["조치"]); used_notes.add(r["정비노트"])
                if len(top3) == 3: break
            if len(top3) < 3:
                for r in sorted(recommended, key=lambda x: x["성공률"], reverse=True):
                    if r["조치"] not in used_actions and r["정비노트"] not in used_notes:
                        top3.append(r); used_actions.add(r["조치"]); used_notes.add(r["정비노트"])
                    if len(top3) == 3: break

            if not top3:
                bot_resp = "죄송합니다. 관련된 정비 사례를 찾을 수 없습니다."
                st.session_state.last_top3 = None
            else:
                # 메인 Top3 요약
                items_html = []
                top3_desc_lines = []
                for i, r in enumerate(top3, 1):
                    label = short_label(r["조치"])
                    t = ACTION_AVG_H.get(r["조치"])
                    time_badge = f" <span class='meta'>| ⏱ {t:.2f}h</span>" if isinstance(t, (int, float)) else ""
                    items_html.append(f"<li>{label}{time_badge} <span class='meta'>| 성공률 {r['성공률']}%</span></li>")
                    top3_desc_lines.append(f"{i}. {label}")
                top3_html = "<ol class='top3'>" + "".join(items_html) + "</ol>"
                top3_desc = "\n".join(top3_desc_lines)

                # LLM 설명
                explain_prompt = f"""
다음은 반도체 장비 정비 이슈에 대한 Top3 성공률 높은 조치입니다.
각 조치의 의미와 특징을 자연스럽게 설명해주세요.
결과는 '번호 + 굵은 제목 + 줄바꿈 + 설명' 형식의 HTML <ol class='explain'> 리스트로 출력하세요.
불필요한 말(서두/말미, 코드블록/백틱)은 절대 넣지 마세요.

{top3_desc}
""".strip()
                explanation_raw = llm.predict(explain_prompt)
                explanation = strip_code_fences(explanation_raw)

                bot_resp = f"""
<strong>{sev_line_html}</strong><br><br>
<strong>✅ 추천 해결책 Top 3</strong>
{top3_html}
<br>
<strong>💡 상세 설명:</strong><br>
{explanation}
""".strip()

                st.session_state.last_top3 = top3
                st.session_state.last_sev_line_html = sev_line_html

        except Exception:
            bot_resp = "❌ 처리 중 오류가 발생했습니다."

        # 로딩 말풍선 제거 + 메시지 저장
        status_ph.empty()
        st.session_state.messages.append({"role": "assistant", "content": bot_resp})

    # ===== 최종 채팅 렌더 =====
    with chat_ph.container():
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        st.markdown(GREET_HTML, unsafe_allow_html=True)
        for m in st.session_state.messages:
            cls = "user" if m["role"] == "user" else "bot"
            bubble = "user-message" if m["role"] == "user" else "bot-message"
            icon = "👤 " if m["role"] == "user" else "🤖 "
            st.markdown(f'<div class="row {cls}"><div class="bubble {bubble}">{icon}{m["content"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== 상세 조치(접힘, 3개) =====
    if st.session_state.last_top3:
        with details_ph.container():
            st.markdown("#### 🔎 상세 보기")
            ph = st.empty()
            with ph.container():
                st.info("🔍 검색중입니다...")
            import time as _t; _t.sleep(0.6)
            ph.empty()

            for idx_, r in enumerate(st.session_state.last_top3, 1):
                with st.expander(f"🔹 Top{idx_} 상세보기 — {short_label(r['조치'])} (성공률 {r['성공률']}%)", expanded=False):
                    note_html = str(r["정비노트"]).replace("\n", "<br>")
                    t = r.get("평균작업시간(h)")
                    time_str = f"{t:.2f} h" if isinstance(t, (int, float)) else "정보 없음"
                    st.markdown(f"""
                    <div style="border:2px solid #B0C4DE; border-radius:8px; padding:10px; margin-bottom:6px; background:#F3F7FF; line-height:1.42;">
                        <b>조치명:</b> {short_label(r['조치'])}<br>
                        <b>예상 작업시간:</b> {time_str}<br>
                        <b>장비종류:</b> {r.get('장비ID','N/A')} / {r.get('모델','N/A')}<br>
                        <b>정비종류:</b> {r.get('정비종류','N/A')}<br>
                        <b>정비자:</b> {r.get('정비자','N/A')}<br><br>
                        <b>정비노트:</b><br>
                        <div style="border:1px solid #ccc; padding:8px; margin-top:4px; background:#fff;">{note_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---- Tab2: 분석 ----
elif main == "📊 정비 데이터 분석":
    # 세그먼티드도 동일하게 key만 사용 (index 사용 X)
    sub_options = ["🏆 핵심 지표", "📈 전체 현황", "🔧 장비별 분석"]
    try:
        st.segmented_control("분석 보기", sub_options, key="active_analysis")
    except AttributeError:
        # Streamlit 버전이 낮아 segmented_control이 없으면 라디오로 대체
        st.radio("분석 보기", sub_options, horizontal=True, key="active_analysis")
    sub = st.session_state.get("active_analysis", "🏆 핵심 지표")

    # ========== 분석 탭 1: 핵심 지표 ==========
    if sub == "🏆 핵심 지표":
        st.markdown("#### 🏆 핵심 지표 TOP 5")
        col1, col2 = st.columns(2)

        # --- 데이터 가드 ---
        if "모델" not in df.columns or df["모델"].dropna().empty:
            st.info("⚠️ '모델' 컬럼이 없거나 값이 비어 있습니다. 업로드 데이터를 확인해주세요.")
        if "문제원인" not in df.columns or df["문제원인"].dropna().empty:
            st.info("⚠️ '문제원인' 컬럼이 비어 있습니다. 문제원인 산출 로직이 실행됐는지 확인해주세요.")

        # 좌: 고장 빈발 장비
        with col1:
            st.markdown("##### 🔧 고장 빈발 장비")
            top5_equip = df["모델"].value_counts().head(5)
            if top5_equip.empty:
                st.info("고장 장비 데이터가 없습니다.")
            else:
                fig1 = px.bar(
                    x=top5_equip.values,
                    y=top5_equip.index.tolist(),
                    orientation="h",
                    text=[f"{v}건" for v in top5_equip.values],
                    color=top5_equip.values,
                    color_continuous_scale="Blues",
                    height=400,
                )
                fig1.update_traces(textposition="outside")
                fig1.update_layout(showlegend=False, margin=dict(l=10, r=50, t=10, b=10))
                st.plotly_chart(fig1, use_container_width=True)

        # 우: 주요 문제 원인
        with col2:
            st.markdown("##### ⚠️ 주요 문제 원인")
            top5_cause = df["문제원인"].value_counts().head(5)
            if top5_cause.empty:
                st.info("문제 원인 데이터가 없습니다.")
            else:
                fig2 = px.bar(
                    x=top5_cause.values,
                    y=top5_cause.index,
                    orientation="h",
                    text=[f"{v}건" for v in top5_cause.values],
                    color=top5_cause.values,
                    color_continuous_scale="Reds",
                    height=400,
                )
                fig2.update_traces(textposition="outside")
                fig2.update_layout(showlegend=False, margin=dict(l=10, r=50, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # AI 핵심 인사이트
        if ("top5_equip" in locals() and not top5_equip.empty) and ("top5_cause" in locals() and not top5_cause.empty):
            st.markdown("#### 💡 핵심 인사이트")
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                prompt_insight = f"""
다음 데이터를 바탕으로 반도체 장비 정비에 대한 핵심 인사이트를 제공해주세요:

고장 빈발 장비 TOP5: {', '.join(top5_equip.index)}
주요 문제 원인 TOP5: {', '.join(top5_cause.index)}

예방 정비와 운영 효율성 관점에서 3-4문장으로 요약해주세요.
"""
                insight = llm.predict(prompt_insight)
                st.markdown(
                    f"""
                    <div class="success-card">
                        <h4 style="margin-top: 0;">💡 핵심 인사이트</h4>
                        <p style="margin-bottom: 0; line-height: 1.6;">{insight}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # 조치명별 평균 작업시간 TOP5
        st.markdown("#### 🛠 조치별 평균 작업시간 TOP5")
        need_cols_top5 = {"정비노트", "2차작업시간(h)", "3차작업시간(h)"}
        if need_cols_top5.issubset(df.columns):
            _tmp2 = df.copy()
            _tmp2["2차조치명"] = _tmp2["정비노트"].apply(_get_second_action)
            _tmp2["3차조치명"] = _tmp2["정비노트"].apply(_get_third_action)

            act2_ = _tmp2.loc[_tmp2["2차조치명"] != "", ["2차조치명", "2차작업시간(h)"]].rename(columns={"2차조치명": "조치명", "2차작업시간(h)": "작업시간(h)"})
            act3_ = _tmp2.loc[_tmp2["3차조치명"] != "", ["3차조치명", "3차작업시간(h)"]].rename(columns={"3차조치명": "조치명", "3차작업시간(h)": "작업시간(h)"})
            actions_all = pd.concat([act2_, act3_], ignore_index=True)
            actions_all["작업시간(h)"] = pd.to_numeric(actions_all["작업시간(h)"], errors="coerce")
            actions_all = actions_all.dropna(subset=["작업시간(h)"])
            actions_all = actions_all[actions_all["작업시간(h)"] > 0]

            action_stats = (
                actions_all.groupby("조치명", as_index=False)
                .agg(건수=("작업시간(h)", "count"), 평균_작업시간_h=("작업시간(h)", "mean"))
                .round({"평균_작업시간_h": 2})
                .sort_values(["평균_작업시간_h", "건수"], ascending=[False, False])
            )

            top5_actions = action_stats.head(5)
            fig_top5 = px.bar(
                top5_actions,
                x="평균_작업시간_h",
                y="조치명",
                orientation="h",
                text="평균_작업시간_h",
                color="평균_작업시간_h",
                color_continuous_scale="Blues",
                height=420,
            )
            fig_top5.update_traces(textposition="outside")
            fig_top5.update_layout(showlegend=False, margin=dict(l=10, r=50, t=10, b=10))
            st.plotly_chart(fig_top5, use_container_width=True)

            if not top5_actions.empty:
                _summ = [f"{r['조치명']}({r['평균_작업시간_h']:.2f}h, {int(r['건수'])}건)" for _, r in top5_actions.iterrows()]
                prompt_act = (
                    "다음은 평균 작업시간이 가장 긴 조치 Top5입니다.\n"
                    f"{'; '.join(_summ)}\n"
                    "운영상 시사점을 2~3문장으로 간결한 줄글로만 요약해 주세요. "
                    "숫자 나열이나 1,2,3 형식은 금지합니다. 가장 시간이 긴 조치명은 문장 안에 자연스럽게 언급하세요."
                )
                insight_act = llm.predict(prompt_act)
                st.markdown(
                    f"""
                    <div class="success-card">
                        <h4 style="margin-top: 0;">🛠 작업시간 인사이트</h4>
                        <p style="margin-bottom: 0; line-height: 1.6;">{insight_act}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("시간 컬럼이 없어 TOP5를 계산할 수 없습니다.")

        st.divider()

        # 문제원인 치명도 TOP5
        st.markdown("#### 🔥 문제별 심각도 TOP5 (종합 점수)")
        _agg = (
            df.groupby("문제원인", dropna=False)
            .agg(
                건수=("문제원인", "size"),
                평균리드타임_h=("총리드타임(h)", "mean"),
                P75_리드타임_h=("총리드타임(h)", lambda x: x.quantile(0.75)),
                평균2차_h=("2차작업시간(h)", "mean"),
                평균3차_h=("3차작업시간(h)", "mean"),
            )
            .fillna(0.0)
        )
        _agg["평균운영시간_h"] = _agg["평균2차_h"].fillna(0) + _agg["평균3차_h"].fillna(0)
        def _safe_minmax(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").fillna(0.0)
            if s.max()==s.min(): return pd.Series([0.0]*len(s), index=s.index)
            return (s - s.min())/(s.max()-s.min())
        _agg["N_평균리드"] = _safe_minmax(_agg["평균리드타임_h"])
        _agg["N_P75"] = _safe_minmax(_agg["P75_리드타임_h"])
        _agg["N_건수"] = _safe_minmax(_agg["건수"])
        _agg["N_운영시간"] = _safe_minmax(_agg["평균운영시간_h"])
        _agg["치명도점수"] = 0.4*_agg["N_평균리드"] + 0.3*_agg["N_P75"] + 0.2*_agg["N_건수"] + 0.1*_agg["N_운영시간"]

        _score_top5 = _agg.sort_values("치명도점수", ascending=False).head(5).reset_index()
        fig_sev_top5 = px.bar(
            _score_top5,
            x="치명도점수",
            y="문제원인",
            orientation="h",
            text=_score_top5["치명도점수"].round(2),
            color="치명도점수",
            color_continuous_scale="Reds",
            height=420,
        )
        fig_sev_top5.update_traces(textposition="outside")
        fig_sev_top5.update_layout(showlegend=False, margin=dict(l=10, r=50, t=10, b=10))
        st.plotly_chart(fig_sev_top5, use_container_width=True)

        if not _score_top5.empty:
            _m = (
                _score_top5[["문제원인", "치명도점수"]]
                .merge(_agg.reset_index()[["문제원인", "평균리드타임_h", "P75_리드타임_h"]], on="문제원인", how="left")
            )
            _summ2 = [
                f"{r['문제원인']}(점수 {float(r['치명도점수']):.2f}, 평균리드타임 {float(r.get('평균리드타임_h', 0) or 0):.2f}h, P75 {float(r.get('P75_리드타임_h', 0) or 0):.2f}h)"
                for _, r in _m.iterrows()
            ]
            prompt_sev = (
                "아래는 치명도 점수가 높은 문제원인 Top5입니다.\n"
                f"{'; '.join(_summ2)}\n"
                "예방 관점에서의 우선순위와 리스크 포인트를 2~3문장 줄글로 요약해 주세요. "
                "번호 매기기, 불릿, 과도한 숫자 나열은 금지합니다."
            )
            insight_sev = llm.predict(prompt_sev)
            st.markdown(
                f"""
                <div class="success-card">
                    <h4 style="margin-top: 0;">🔥 치명도 인사이트</h4>
                    <p style="margin-bottom: 0; line-height: 1.6;">{insight_sev}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # HERO 카드
        if "equipment_hero" in globals() and equipment_hero:
            st.divider()
            st.markdown("#### 🦸‍♂️ 이 장비는 내가 HERO!")
            hero_cols = st.columns(min(3, len(equipment_hero)))
            for idx, (model, hero_info) in enumerate(list(equipment_hero.items())[:3]):
                with hero_cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div class="hero-card">
                            <h4 style="margin: 0; color: #333;">🏆 {model}</h4>
                            <p style="margin: 5px 0; font-size: 1.1em;"><strong>{hero_info['name']}</strong></p>
                            <p style="margin: 0; font-size: 0.9em;">정비 횟수: {hero_info['count']}회</p>
                            <p style="margin: 0; font-size: 0.8em;">📞 {hero_info['contact']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ========== 분석 탭 2: 전체 현황 ==========
    elif sub == "📈 전체 현황":
        st.markdown("#### 📈 전체 현황 분석")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### 🏭 장비별 고장 분포")
            total_equip = df['모델'].value_counts()
            fig_pie1 = px.pie(names=total_equip.index.tolist(), values=total_equip.values, hole=0.4, height=450)
            fig_pie1.update_traces(textposition='outside', textinfo='percent+label', textfont_size=11)
            fig_pie1.update_layout(showlegend=False, margin=dict(l=50, r=50, t=30, b=30))
            st.plotly_chart(fig_pie1, use_container_width=True)

        with c2:
            st.markdown("##### 🔍 문제 원인 분포")
            total_cause = df['문제원인'].value_counts()
            fig_pie2 = px.pie(names=total_cause.index.tolist(), values=total_cause.values, hole=0.4, height=450)
            fig_pie2.update_traces(textposition='outside', textinfo='percent+label', textfont_size=11)
            fig_pie2.update_layout(showlegend=False, margin=dict(l=50, r=50, t=30, b=30))
            st.plotly_chart(fig_pie2, use_container_width=True)

        st.divider()
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
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color:white; padding:20px; border-radius:10px; margin:10px 0;">
                <h4 style="margin:0;">📊 전체 현황 분석</h4>
                <p style="margin:10px 0 0 0; line-height:1.6;">{total_insight}</p>
            </div>
            """, unsafe_allow_html=True)

        # 전체 조치 평균시간 도넛
        st.divider()
        st.subheader("⏱ 전체 조치 평균시간 도넛차트")
        tmp = df.copy()
        tmp["2차조치명"] = tmp["정비노트"].apply(_get_second_action)
        tmp["3차조치명"] = tmp["정비노트"].apply(_get_third_action)
        act2 = tmp.loc[tmp["2차조치명"] != "", ["2차조치명", "2차작업시간(h)"]].rename(columns={"2차조치명": "조치명", "2차작업시간(h)": "작업시간(h)"})
        act3 = tmp.loc[tmp["3차조치명"] != "", ["3차조치명", "3차작업시간(h)"]].rename(columns={"3차조치명": "조치명", "3차작업시간(h)": "작업시간(h)"})
        actions = pd.concat([act2, act3], ignore_index=True)
        actions["작업시간(h)"] = pd.to_numeric(actions["작업시간(h)"], errors="coerce")
        actions = actions.dropna(subset=["작업시간(h)"])
        actions = actions[actions["작업시간(h)"] > 0]
        actions["조치명"] = actions["조치명"].str.replace(r"\s+", " ", regex=True).str.strip()

        stats_avg = (
            actions.groupby("조치명", as_index=False)
            .agg(건수=("작업시간(h)", "count"), 평균_작업시간_h=("작업시간(h)", "mean"))
            .round({"평균_작업시간_h": 2})
            .sort_values("평균_작업시간_h", ascending=False)
        )

        def _wrap_label(s: str, width: int = 12) -> str:
            s = str(s)
            return "<br>".join([s[i:i+width] for i in range(0, len(s), width)])

        stats_avg_plot = stats_avg.copy()
        stats_avg_plot["조치명_wrapped"] = stats_avg_plot["조치명"].apply(lambda x: _wrap_label(x, 12))

        fig_pie_avg = px.pie(
            stats_avg_plot,
            names="조치명_wrapped",
            values="평균_작업시간_h",
            hole=0.35,
            title="전체 조치 (평균 작업시간 기준)",
        )
        fig_pie_avg.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>평균 작업시간: %{value:.2f} h<br>%{percent}<extra></extra>",
        )
        fig_pie_avg.update_layout(
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=11)),
            margin=dict(l=10, r=10, t=60, b=10),
            height=520,
        )
        st.plotly_chart(fig_pie_avg, use_container_width=True)

        if not stats_avg.empty:
            _long = stats_avg.head(5)
            _summ3 = [f"{r['조치명']}({r['평균_작업시간_h']:.2f}h, {int(r['건수'])}건)" for _, r in _long.iterrows()]
            prompt_avg_actions = (
                "전체 조치의 평균 작업시간 상위 항목 요약입니다.\n"
                f"{'; '.join(_summ3)}\n"
                "병목 가능성과 일정/자원 계획 포인트를 2~3문장 줄글로만 제시해 주세요. 번호/불릿 금지."
            )
            insight_avg_actions = llm.predict(prompt_avg_actions)
            st.markdown(f"💡 **조치 평균시간 인사이트:** {insight_avg_actions}")

        # 전체 문제원인 치명도
        st.divider()
        st.subheader("🔥 전체 문제원인 치명도(종합 점수)")

        _agg2 = (
            df.groupby("문제원인", dropna=False)
            .agg(
                건수=("문제원인", "size"),
                평균리드타임_h=("총리드타임(h)", "mean"),
                P75_리드타임_h=("총리드타임(h)", lambda x: x.quantile(0.75)),
                평균2차_h=("2차작업시간(h)", "mean"),
                평균3차_h=("3차작업시간(h)", "mean"),
            )
            .fillna(0.0)
        )
        _agg2["평균운영시간_h"] = _agg2["평균2차_h"].fillna(0) + _agg2["평균3차_h"].fillna(0)
        _agg2["N_평균리드"] = _safe_minmax(_agg2["평균리드타임_h"])
        _agg2["N_P75"] = _safe_minmax(_agg2["P75_리드타임_h"])
        _agg2["N_건수"] = _safe_minmax(_agg2["건수"])
        _agg2["N_운영시간"] = _safe_minmax(_agg2["평균운영시간_h"])
        _agg2["치명도점수"] = 0.4*_agg2["N_평균리드"] + 0.3*_agg2["N_P75"] + 0.2*_agg2["N_건수"] + 0.1*_agg2["N_운영시간"]

        _score_all = _agg2.sort_values("치명도점수", ascending=False).reset_index()
        fig_pie_sev = px.pie(_score_all, names="문제원인", values="치명도점수", hole=0.35, title="전체 문제원인 치명도 (도넛차트)")
        fig_pie_sev.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>치명도 점수: %{value:.2f}<br>%{percent}<extra></extra>")
        fig_pie_sev.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=500)
        st.plotly_chart(fig_pie_sev, use_container_width=True)

        if not _score_all.empty:
            _top_all = _score_all.head(5)[["문제원인", "치명도점수"]]
            _summ4 = [f"{r['문제원인']}(점수 {r['치명도점수']:.2f})" for _, r in _top_all.iterrows()]
            prompt_sev_all = (
                "전체 문제원인 치명도 도넛차트 상위 항목입니다.\n"
                f"{'; '.join(_summ4)}\n"
                "장기 리스크 관리와 모니터링 우선순위를 2~3문장 줄글로만 요약해 주세요. 숫자 나열/불릿/번호 금지."
            )
            insight_sev_all = llm.predict(prompt_sev_all)
            st.markdown(f"💡 **전체 치명도 인사이트:** {insight_sev_all}")

    # ========== 분석 탭 3: 장비별 상세 ==========
    elif sub == "🔧 장비별 분석":
        st.markdown("#### 🔧 장비별 상세 분석")

        equip_list = df['모델'].dropna().unique().tolist()
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_equip = st.selectbox(
                "분석할 장비를 선택하세요",
                ["전체 장비"] + equip_list,
                help="특정 장비를 선택하면 해당 장비의 상세 분석을 확인할 수 있습니다",
                key="equip_select_box"
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="color: #1976d2; margin: 0;">{len(equip_list)}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">총 장비 종류</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        if selected_equip != "전체 장비":
            df_filtered = df[df['모델'] == selected_equip]

            if df_filtered.empty:
                st.warning(f"⚠️ 선택한 장비({selected_equip})의 데이터가 없습니다.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h2 style="color: #d32f2f; margin: 0;">{len(df_filtered)}</h2>
                            <p style="margin: 5px 0 0 0; color: #666;">총 고장 건수</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with c2:
                    unique_causes = df_filtered['문제원인'].nunique()
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h2 style="color: #f57c00; margin: 0;">{unique_causes}</h2>
                            <p style="margin: 5px 0 0 0; color: #666;">문제 원인 종류</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with c3:
                    unique_maintainers = df_filtered['정비자'].nunique() if '정비자' in df_filtered.columns else 0
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h2 style="color: #388e3c; margin: 0;">{unique_maintainers}</h2>
                            <p style="margin: 5px 0 0 0; color: #666;">담당 정비자</p>
                        </div>
                        """, unsafe_allow_html=True
                    )

                st.divider()

                st.markdown(f"##### 🔍 {selected_equip} 문제 원인 분석")
                cause_counts = df_filtered['문제원인'].value_counts()
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

                if len(cause_counts) > 0:
                    st.markdown("##### 🛠️ 추천 정비 조치")
                    picked_cause = st.selectbox(
                        "문제 원인을 선택하세요",
                        options=cause_counts.index.tolist(),
                        index=0,
                        key="picked_cause_select"
                    )

                    if 'SEVERITY_LABEL_BY_CAUSE' in globals():
                        sev_label_equ = SEVERITY_LABEL_BY_CAUSE.get(picked_cause)
                        if sev_label_equ:
                            st.markdown(f"**심각도:** **{sev_label_equ}**")

                    if 'df_success' in globals() and isinstance(df_success, pd.DataFrame) and not df_success.empty:
                        df_cause = df_success[df_success["대표원인"] == picked_cause]
                        top3_actions = (
                            df_cause.sort_values("성공률(%)", ascending=False)
                            .head(3)[["조치", "성공률(%)"]]
                            .copy()
                        )

                        if top3_actions.empty:
                            st.info(f"'{picked_cause}'에 대한 추천 조치 데이터가 없습니다.")
                        else:
                            def _avg_time_lookup(act):
                                if 'ACTION_AVG_H' in globals():
                                    val = ACTION_AVG_H.get(act)
                                    if isinstance(val, (int, float)):
                                        return round(float(val), 2)
                                return None

                            top3_actions["평균작업시간(h)"] = top3_actions["조치"].apply(_avg_time_lookup)

                            st.markdown(f"**'{picked_cause}' 문제에 대한 추천 조치 TOP 3:**")
                            for idx, (_, r) in enumerate(top3_actions.iterrows(), 1):
                                action_name = r["조치"]
                                success_rate = float(r["성공률(%)"])
                                avg_h = r["평균작업시간(h)"]

                                if success_rate >= 80:
                                    color, icon = "#4caf50", "🟢"
                                elif success_rate >= 60:
                                    color, icon = "#ff9800", "🟡"
                                else:
                                    color, icon = "#f44336", "🔴"

                                time_text = f" · 평균작업시간 {avg_h:.2f}h" if avg_h is not None else ""
                                st.markdown(
                                    f"""
                                    <div style="background-color: white; border-left: 4px solid {color};
                                                padding: 15px; margin: 10px 0; border-radius: 5px;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <strong>{icon} {idx}. {action_name}</strong><br>
                                        <span style="color: {color}; font-weight: bold;">성공률: {success_rate:.1f}%{time_text}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("성공률/조치 데이터(df_success)가 없습니다.")

# ---- Tab3: 정비노트 작성 도우미 ----
elif main == "📝 정비노트 작성 도우미":
    st.markdown("### 📝 정비노트 작성 도우미")
    st.markdown("""
    <div style="background-color:#e8f5e8; padding:15px; border-radius:10px; margin-bottom:20px;">
        <h4 style="color:#2e7d32; margin:0;">🎯 사용법</h4>
        <p style="margin:10px 0 0 0; color:#1b5e20;">정비 정보를 입력하면 자동으로 표준화된 정비노트를 생성해드립니다!</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("maintenance_note_form"):
        c1, c2 = st.columns(2)
        with c1:
            maintenance_date = st.date_input("📅 정비 날짜", value=date.today())
            maintenance_time = st.time_input("⏰ 정비 시작 시각", value=time(9,0))
            lot_id = st.text_input("🔖 LOT ID", placeholder="예: M1133097")
            equipment_models = ['선택하세요'] + sorted(df['모델'].unique().tolist()) if '모델' in df.columns else ['선택하세요']
            equipment_model = st.selectbox("🔧 장비 모델", equipment_models)
        with c2:
            common_causes = ['직접 입력'] + PROBLEM_KEYWORDS
            cause_opt = st.selectbox("⚠️ 문제 원인 (일반적인 원인)", common_causes)
            problem_cause = st.text_input("문제 원인 직접 입력", placeholder="예: plasma ignition failure") if cause_opt=='직접 입력' else cause_opt
            first_action = st.text_input("🛠️ 1차 조치 *", placeholder="예: RF generator 리셋 및 점검")
            first_success = st.selectbox("1차 조치 결과 *", ["성공","실패"], index=1)
            show_second = first_success=="실패"
            if show_second:
                st.markdown("---")
                second_action = st.text_input("🔧 2차 조치", placeholder="예: matching unit 교체 진행")
                if second_action:
                    second_success = st.selectbox("2차 조치 결과", ["성공","실패"])
                    show_third = second_success=="실패"
                    if show_third:
                        st.markdown("---")
                        third_action = st.text_input("🔩 3차 조치", placeholder="예: plasma source 점검 및 connector 재연결")
                        third_success = st.selectbox("3차 조치 결과", ["성공","실패"]) if third_action else None
                    else:
                        third_action=None; third_success=None
                else:
                    second_success=None; third_action=None; third_success=None
            else:
                second_action=None; second_success=None; third_action=None; third_success=None
        additional_info = st.text_area("📝 기타 상황 설명", placeholder="예: kit 수급 대기 중, 부품 입고 예정, 추가 점검 필요 등", height=100)
        gen_btn = st.form_submit_button("📋 정비노트 자동 생성", use_container_width=True)

    def validate_inputs():
        errs=[]
        if not lot_id: errs.append("LOT ID를 입력해주세요")
        elif len(lot_id)<3: errs.append("LOT ID는 3자리 이상 입력해주세요")
        if equipment_model=='선택하세요': errs.append("장비 모델을 선택해주세요")
        if not problem_cause: errs.append("문제 원인을 입력해주세요")
        if not first_action: errs.append("1차 조치를 입력해주세요")
        if maintenance_date==date.today() and maintenance_time>datetime.now().time():
            errs.append("미래 시간은 선택할 수 없습니다")
        return errs

    if gen_btn:
        errs = validate_inputs()
        if errs:
            st.error("❌ 입력 오류:\n" + "\n".join([f"• {e}" for e in errs]))
        else:
            actions_data = {
                "first":{"action":first_action, "success":first_success},
                "second":{"action":second_action, "success":second_success} if second_action else None,
                "third":{"action":third_action, "success":third_success} if third_action else None
            }
            with st.spinner("🤖 AI가 정비노트를 생성하고 있습니다..."):
                dt_str = f"{maintenance_date.strftime('%m월%d일')} {maintenance_time.strftime('%H:%M')}"
                prompt_note = f"""
다음 정보를 바탕으로 반도체 장비 정비노트를 자연스럽게 작성해주세요:
날짜/시간: {dt_str}
LOT ID: {lot_id}
장비 모델: {equipment_model}
문제 원인: {problem_cause}
1차 조치: {first_action} → {'정상' if first_success=='성공' else '여전히 이상 발생'}
"""
                if second_action:
                    prompt_note += f"2차 조치: {second_action} → {'정상' if second_success=='성공' else '여전히 이상 발생'}\n"
                if third_action:
                    prompt_note += f"3차 조치: {third_action} → {'정상' if third_success=='성공' else '여전히 이상 발생'}\n"
                if additional_info:
                    prompt_note += f"추가 상황: {additional_info}\n"
                prompt_note += """
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
                    note_text = llm.predict(prompt_note)
                    save_maintenance_note(st.session_state.username, lot_id, equipment_model, problem_cause, json.dumps(actions_data), note_text)
                    csv_ok, csv_info = save_to_csv({
                        'date': maintenance_date.strftime('%Y-%m-%d'),
                        'time': maintenance_time.strftime('%H:%M'),
                        'lot_id': lot_id,
                        'equipment_model': equipment_model,
                        'username': st.session_state.username,
                        'problem_cause': problem_cause,
                        'generated_note': note_text
                    })
                    st.markdown("#### ✅ 생성된 정비노트")
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left:4px solid #28a745;">
                        <h5 style="color:#28a745; margin-top:0;">📋 {lot_id} 정비노트</h5>
                        <div style="white-space: pre-line; line-height:1.6; color:#333;">{note_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if csv_ok:
                        st.success("✅ 정비노트가 성공적으로 생성되었습니다!")
                        st.markdown(f"""
                        <div style="background-color:#e7f3ff; border-left:4px solid #2196F3; padding:15px; margin:15px 0; border-radius:5px;">
                            <h5 style="color:#2196F3; margin:0;">📄 파일 저장 위치</h5>
                            <p style="margin:10px 0 0 0; font-family: monospace; background:white; padding:8px; border-radius:4px;">{csv_info}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ CSV 저장 실패: {csv_info}")
                    st.divider()
                    st.markdown("##### 📋 복사용 텍스트")
                    st.text_area("아래 텍스트를 복사해서 사용하세요", value=note_text, height=200)
                except Exception as e:
                    st.error(f"❌ 정비노트 생성 중 오류가 발생했습니다: {str(e)}")

# ---- Tab4: 저장된 노트 조회 ----
elif main == "📂 저장된 노트 조회":
    st.markdown("### 📂 저장된 정비노트 조회")
    csv_file = 'maintenance_notes.csv'
    if os.path.exists(csv_file):
        csv_df = pd.read_csv(csv_file, encoding='utf-8')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding:20px; border-radius:10px; margin:20px 0; color:#333;">
            <h4 style="margin:0;">📊 저장된 정비노트 현황</h4>
            <div style="display:flex; justify-content:space-around; margin-top:15px;">
                <div style="text-align:center;"><h2 style="margin:0;">{len(csv_df)}</h2><p style="margin:5px 0 0 0;">총 노트 수</p></div>
                <div style="text-align:center;"><h2 style="margin:0;">{csv_df['장비모델'].nunique() if '장비모델' in csv_df.columns else 0}</h2><p style="margin:5px 0 0 0;">등록 장비</p></div>
                <div style="text-align:center;"><h2 style="margin:0;">{csv_df['정비자'].nunique() if '정비자' in csv_df.columns else 0}</h2><p style="margin:5px 0 0 0;">정비 담당자</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        f1, f2, f3 = st.columns(3)
        with f1:
            filter_equipment = st.selectbox("장비 필터",
                ["전체"] + (sorted(csv_df['장비모델'].unique().tolist()) if '장비모델' in csv_df.columns else []),
                key="csv_filter_equipment")
        with f2:
            filter_maintainer = st.selectbox("정비자 필터",
                ["전체"] + (sorted(csv_df['정비자'].unique().tolist()) if '정비자' in csv_df.columns else []),
                key="csv_filter_maint")
        with f3:
            if '정비일자' in csv_df.columns:
                csv_df['정비일자'] = pd.to_datetime(csv_df['정비일자'], errors='coerce')
                date_filter = st.date_input("날짜 필터", value=None, key="csv_date_filter")
            else:
                date_filter = None

        filtered = csv_df.copy()
        if filter_equipment!="전체" and '장비모델' in filtered.columns:
            filtered = filtered[filtered['장비모델']==filter_equipment]
        if filter_maintainer!="전체" and '정비자' in filtered.columns:
            filtered = filtered[filtered['정비자']==filter_maintainer]
        if date_filter and '정비일자' in filtered.columns:
            filtered = filtered[filtered['정비일자'].dt.date==date_filter]

        st.markdown(f"#### 📋 검색 결과 ({len(filtered)}건)")
        if not filtered.empty:
            disp = filtered.copy()
            if '정비노트' in disp.columns:
                disp['노트 미리보기'] = disp['정비노트'].apply(lambda x: (str(x)[:100] + '...') if len(str(x))>100 else str(x))
            disp_cols = ['정비일자','정비시각','LOT_ID','장비모델','정비자','문제원인','노트 미리보기']
            available = [c for c in disp_cols if c in disp.columns]
            st.dataframe(disp[available], use_container_width=True, height=400)

            st.divider()
            st.markdown("##### 🔍 상세 보기")
            sel_idx = st.selectbox("상세히 볼 정비노트 선택", filtered.index,
                                   format_func=lambda x: f"{filtered.loc[x,'LOT_ID']} - {filtered.loc[x,'정비일자']}",
                                   key="csv_detail_select")
            if sel_idx is not None:
                row = filtered.loc[sel_idx]
                st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:20px; border-radius:10px;">
                    <h5 style="color:#333; margin-top:0;">📋 정비노트 상세</h5>
                    <div style="margin:10px 0;">
                        <strong>LOT ID:</strong> {row.get('LOT_ID','N/A')}<br>
                        <strong>장비:</strong> {row.get('장비모델','N/A')}<br>
                        <strong>정비자:</strong> {row.get('정비자','N/A')}<br>
                        <strong>일시:</strong> {row.get('정비일자','N/A')} {row.get('정비시각','N/A')}<br>
                        <strong>문제원인:</strong> {row.get('문제원인','N/A')}
                    </div>
                    <hr>
                    <div style="white-space: pre-line; line-height:1.6;">{row.get('정비노트','N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("검색 조건에 맞는 정비노트가 없습니다.")

        st.divider()
        dl_col, _ = st.columns([1,3])
        with dl_col:
            st.download_button(
                label="📥 CSV 다운로드",
                data=filtered.to_csv(index=False, encoding='utf-8-sig'),
                file_name=f"maintenance_notes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.caption(f"📁 원본 파일 위치: `{os.path.abspath(csv_file)}`")
    else:
        st.info("아직 저장된 정비노트가 없습니다. 정비노트 작성 도우미를 사용해 첫 번째 노트를 생성해보세요!")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:20px;">
    <p>🚀 <strong>HERO</strong> - Hynix Equipment Response Operator</p>
</div>
""", unsafe_allow_html=True)