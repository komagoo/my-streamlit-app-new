import os
import sys
import streamlit as st
from dotenv import load_dotenv

# LangChain VectorStore용 Pinecone: 이름 충돌 방지를 위해 별칭 사용
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# pinecone 최신 라이브러리 (v3.x)—클래스 직접 import
from pinecone import Pinecone

# 기타 라이브러리들
import pandas as pd
import re
from collections import defaultdict, Counter
import plotly.express as px
import base64

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 환경변수와 .env 로드
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    st.error("❌ PINECONE_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# Pinecone 객체 생성 (v3 방식)
pc = Pinecone(api_key=pinecone_api_key)
index_name = "maintenance-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536)
index = pc.Index(index_name)

# 로고 이미지 인코딩
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "Hero_logo(final).png"
logo_base64 = get_base64_of_bin_file(logo_path)

# Streamlit 화면 설정
st.set_page_config(page_title="🚀mySUNI X SK 프로젝트", layout="wide")
st.markdown(f"""
<div style="background-color:#ff4b4b; height:30px; width:100%;"></div>
<div style="display:flex; align-items:center; padding:20px 30px;">
    <img src="data:image/png;base64,{logo_base64}" alt="logo" style="height:100px; margin-right:30px;">
    <div>
        <h1 style="margin:0; font-size:2.5rem; color:#222;">HERO</h1>
        <p style="margin:0; font-size:1.1rem; color:#555;">Hynix Equipment Response Operator</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 로그인 세션
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# 로그인 단계
if not st.session_state.logged_in:
    st.subheader("🔑 로그인")
    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")

    valid_users = {
        "mySUNI250728!@": "mySUNI250728!@"
    }
    if st.button("로그인"):
        if username in valid_users and password == valid_users[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
            st.success(f"✅ {username}님, 환영합니다!")
            st.rerun()
        else:
            st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# OpenAI 키 불러오기
api_key = (
    st.session_state.get("api_key")
    or os.getenv("OPENAI_API_KEY")
    or (st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None)
)
if not api_key:
    st.error("❌ OpenAI API 키가 설정되어 있지 않습니다.")
    st.stop()

# ✅ 데이터 로드 및 벡터DB 생성
# 예시: CSV 파일에서 데이터 로드 (파일 경로와 컬럼명은 실제 데이터에 맞게 수정)
df = pd.read_csv("maintenance_data.csv")  # 실제 파일명으로 변경
df_success = pd.read_csv("maintenance_success.csv")  # 실제 파일명으로 변경

# 문서 리스트 생성
documents = [
    Document(page_content=str(row["정비노트"]), metadata=dict(row))
    for _, row in df.iterrows()
]

# 텍스트 분할 및 임베딩 생성
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# LangChain용 Pinecone VectorDB 생성
vectordb = LangchainPinecone.from_documents(
    docs_split, embeddings, index_name=index_name, pinecone_api_key=pinecone_api_key
)

# ✅ LLM + 벡터 검색 기반 QA 체인 생성
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 20}),  # 🔍 유사도 기반 top 20
    return_source_documents=True
)


# ----------------------
# 6. 사이드바 메뉴
# ----------------------
menu = st.sidebar.radio(
    "📂 메뉴 선택",
    ["🔹 정비 검색 & 추천", "📈 정비 통계 자료"]
)


# 이후 데이터 업로드, 분석, 챗봇 등 기존 코드 이어서...
# 7. 정비 검색 & 추천 페이지
# ----------------------
if menu == "🔹 정비 검색 & 추천":
    st.subheader("🤖 HERO 챗봇 – 정비 문제, 제가 다 알고있어요!")


    example_keywords = [
        "wafer not", "plasma ignition failure",
        "pumpdown 시간 지연", "slot valve 동작 불량",
        "RF auto match 불량"
    ]
 # ------------------ 항상 보이는 초반 인사말 ------------------
    example_keywords = [
        "wafer not", "plasma ignition failure",
        "pumpdown 시간 지연", "slot valve 동작 불량",
        "RF auto match 불량"
    ]
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-start; margin-bottom:10px;">
        <div style="font-size:30px; margin-right:8px;">🤖</div>
        <div style="background-color:#F1F0F0; color:black; padding:10px 15px;
                    border-radius:15px; max-width:80%;">
            안녕하세요👋<br>
            반도체 장비 정비 이슈 해결 도우미 HERO입니다!<br>
            정비 이슈를 입력하시면, HERO가 유사 사례를 찾아 해결책을 제안해드려요.<br><br>
            💡 {' | '.join(example_keywords)}
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ------------------ 답변 생성 로딩 스피너 ------------------
    query = st.text_input(
        "메시지를 입력하세요", key="search_query",
        placeholder="예: slot valve 동작이 안돼ㅠㅠ"
    )


    # ✅ 입력이 있을 때만 실행
    if query.strip():
        with st.spinner("🔄 유사 정비 사례와 답변을 준비 중입니다. 잠시만 기다려주세요..."):
            # 사용자 말풍선
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin-bottom:10px;">
                <div style="background-color:#DCF8C6; color:black;
                            padding:10px 15px; border-radius:15px;
                            max-width:70%; text-align:right;">
                    {query}
                </div>
                <div style="font-size:30px; margin-left:8px;">👤</div>
            </div>
            """, unsafe_allow_html=True)


            # ----------------------
            # 1) RAG 검색
            # ----------------------
            output = qa_chain({"query": query})
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
                        maint_type = matched_row['정비종류'].iloc[0] if '정비종류' in df.columns and not matched_row.empty else 'N/A'
                        maint_person = matched_row['정비자'].iloc[0] if '정비자' in df.columns and not matched_row.empty else 'N/A'


                        recommended.append({
                            "조치": row["조치"],
                            "성공률": row["성공률(%)"],
                            "정비노트": note,
                            "장비ID": equip_id,
                            "모델": model,
                            "정비종류": maint_type,
                            "정비자": maint_person
                        })


            if not recommended:
                st.warning("검색된 사례가 없습니다. 다시 입력해주세요.")
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
                        used_notes.add(note_key)
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


                # ----------------------
                # 3) LLM 설명 생성
                # ----------------------
                top3_desc = "\n".join([f"{i+1}. {r['조치']} – 성공률 {r['성공률']}%"
                                       for i, r in enumerate(top3)])
                prompt = f"""
다음은 반도체 장비 정비 이슈에 대한 Top3 성공률 높은 조치 목록입니다.
각 조치의 의미와 특징을 자연스럽게 한 단락으로 설명해줘.
성공률 수치는 말하지 마.


{top3_desc}
                """
                explanation = llm.predict(prompt)
                top3_html = top3_desc.replace("\n", "<br>")


                # 챗봇 말풍선
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-start; margin-bottom:10px;">
                    <div style="font-size:30px; margin-right:8px;">🤖</div>
                    <div style="background-color:#F1F0F0; color:black; padding:10px 15px;
                                border-radius:15px; max-width:80%;">
                        ✅ Top3 성공률 높은 조치<br><br>
                        {top3_html}<br><br>
                        💡 {explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)


                # ----------------------
                # 4) 상세보기
                # ----------------------
                for idx, r in enumerate(top3, 1):
                    with st.expander(f"🔹 Top{idx} 상세보기 ({r['조치']}, 성공률 {r['성공률']}%)"):
                        note_html = r['정비노트'].replace("\n", "<br>")
                        st.markdown(f"""
                        <div style="border:2px solid #B0C4DE; border-radius:8px;
                                    padding:15px; margin-bottom:10px;
                                    background-color:#F3F7FF;">
                            <b>조치명:</b> {r['조치']}<br>
                            <b>장비:</b> {r['장비ID']} / {r['모델']}<br>
                            <b>정비종류:</b> {r['정비종류']}<br>
                            <b>정비자:</b> {r['정비자']}<br><br>
                            <b>정비노트:</b><br>
                            <div style="border:1px solid #ccc; padding:10px; margin-top:5px; background:#fff;">
                                {note_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ----------------------
# 8. 정비 통계 페이지
# ----------------------
elif menu == "📈 정비 통계 자료":
    st.subheader("📈 정비 통계 자료")


    tab1, tab2, tab3 = st.tabs(["🏆 Top5 요약", "📊 전체 요약", "🔹 장비별 상세"])


    # ----------------------
    # Tab1: Top5
    # ----------------------
    with tab1:
        with st.spinner("📊 Top5 요약 데이터를 준비 중입니다..."):
            st.subheader("🔧 가장 많이 고장난 장비 TOP5")
            top5_equip = df['모델'].value_counts().head(5)


            fig1 = px.pie(
                names=top5_equip.index.tolist(),
                values=top5_equip.values,
                hole=0.4
            )
            fig1.update_traces(textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)


            prompt_cause = f"문제 원인: {', '.join(top5_equip.index)}\n각 장비의 고장 패턴과 발생 경향을 바탕으로, 예방 정비와 공정 운영 측면에서 얻을 수 있는 핵심 인사이트를 2~3문장으로 요약해 주세요.숫자는 언급하지 마세요. 1~3위 정도는 장비도 자연스럽게 언급해 주세요."
            insight_cause = llm.predict(prompt_cause)
            st.markdown(f"💡 **문제 원인 인사이트:** {insight_cause}")


            # 문제원인 10개 정의
            problem_keywords = [
                "wafer not", "plasma ignition failure", "pumpdown 시간 지연",
                "mass flow controller 이상", "etch residue over spec",
                "temperature abnormal", "slot valve 동작 불량",
                "chamber leak", "sensor error", "RF auto match 불량"
            ]


            # TF-IDF 기반 문제원인 분류 (기타 제거)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity


            notes = df['정비노트'].astype(str).str.lower().tolist()
            corpus = notes + [kw.lower() for kw in problem_keywords]


            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)


            note_vecs = tfidf_matrix[:-len(problem_keywords)]
            keyword_vecs = tfidf_matrix[-len(problem_keywords):]


            similarity_matrix = cosine_similarity(note_vecs, keyword_vecs)
            best_match_indices = similarity_matrix.argmax(axis=1)
            df['문제원인'] = [problem_keywords[i] for i in best_match_indices]


            st.subheader("⚠️ 문제 원인 TOP5")
            top5_cause = df['문제원인'].value_counts().head(5)


            fig2 = px.bar(
                x=top5_cause.values,
                y=top5_cause.index,
                orientation='h',
                text=top5_cause.values,
                color=top5_cause.values,
                color_continuous_scale='OrRd'
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)


            prompt_cause = f"문제 원인: {', '.join(top5_cause.index)}\n각 문제 원인의 영향과 인사이트를 2~3문장으로 줄글 요약해줘."
            insight_cause = llm.predict(prompt_cause)
            st.markdown(f"💡 **문제 원인 인사이트:** {insight_cause}")


    # ----------------------
    # Tab2: 전체 요약
    # ----------------------
    with tab2:
        with st.spinner("📊 전체 요약 데이터를 준비 중입니다..."):
            st.markdown("### 📊 전체 요약")
            st.subheader("🧭 전체 장비 고장 분포")


            total_equip = df['모델'].value_counts()


            fig_all = px.pie(
                names=total_equip.index.tolist(),
                values=total_equip.values,
                title="전체 장비별 고장 비율",
                hole=0.4
            )
            fig_all.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_all, use_container_width=True)


            prompt_cause = f"문제 원인: {', '.join(top5_equip.index)}\n모든 장비의 문제 발생을 전체적인 비율을 나타낸 그래프입니다. 해당 비율을 분석해봤을 때, 얻을 수 있는 장비의 문제 발생 비율에 대한 인사이트를 2~3줄로 제시해주세요. 숫자는 언급하지 마세요. 1~3위 정도는 장비도 자연스럽게 언급해 주세요."
            insight_cause = llm.predict(prompt_cause)
            st.markdown(f"💡 **문제 원인 인사이트:** {insight_cause}")


            st.subheader("🧠 전체 문제 원인 분포")


            total_cause = df['문제원인'].value_counts()


            fig_cause = px.pie(
                names=total_cause.index.tolist(),
                values=total_cause.values,
                title="전체 문제 원인별 고장 비율",
                hole=0.4
            )
            fig_cause.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_cause, use_container_width=True)


            prompt_cause = f"문제 원인: {', '.join(top5_equip.index)}\n모든 문제의 원인을 전체적인 비율을 나타낸 그래프입니다. 해당 비율을 분석해봤을 때, 얻을 수 있는 문제 발생원인에 대한 인사이트를 2~3줄로 제시해주세요. 숫자는 언급하지 마세요. 1~3위 정도는 문제 원인도 자연스럽게 언급해 주세요."
            insight_cause = llm.predict(prompt_cause)
            st.markdown(f"💡 **문제 원인 인사이트:** {insight_cause}")


    # ----------------------
    # Tab3: 장비별 상세
    # ----------------------
    with tab3:
        with st.spinner("📊 장비별 상세 데이터를 준비 중입니다..."):
            st.markdown("### 🔹 장비별 상세")


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


            equip_list = df['모델'].dropna().unique().tolist()
            selected_equip = st.selectbox("장비를 선택하세요", ["전체"] + equip_list)


            if selected_equip != "전체":
                df_filtered = df[df['모델'] == selected_equip]


                if df_filtered.empty:
                    st.warning(f"선택한 장비({selected_equip})의 데이터가 없습니다.")
                else:
                    top5_cause_equip = (
                        df_filtered.groupby('문제원인')
                        .size()
                        .reset_index(name='빈도')
                        .sort_values('빈도', ascending=False)
                        .head(5)
                    )


                    st.subheader(f"⚡ {selected_equip} 문제 원인 TOP5")
                    fig3 = px.bar(
                        x=top5_cause_equip['빈도'],
                        y=top5_cause_equip['문제원인'],
                        orientation='h',
                        text=top5_cause_equip['빈도'],
                        color=top5_cause_equip['빈도'],
                        color_continuous_scale='Purples'
                    )
                    fig3.update_traces(textposition='outside')
                    st.plotly_chart(fig3, use_container_width=True)


                    selected_cause = st.selectbox(
                        "문제 원인을 선택하세요",
                        top5_cause_equip['문제원인'].tolist()
                    )


                    if selected_cause:
                        df_cause = df_success[df_success['대표원인'] == selected_cause]
                        top3_actions = (
                            df_cause.sort_values('성공률(%)', ascending=False)
                            .head(3)[['조치']]
                        )


                        if top3_actions.empty:
                            st.info(f"'{selected_cause}'에 대한 추천 조치 데이터가 없습니다.")
                        else:
                            st.markdown(f"### 🤖 '{selected_cause}' 추천 조치 TOP3")
                            for idx, row in top3_actions.iterrows():
                                st.markdown(f"- {row['조치']}")
            else:
                st.info("장비를 선택하면 해당 장비의 문제 원인과 추천 조치를 확인할 수 있습니다.") 