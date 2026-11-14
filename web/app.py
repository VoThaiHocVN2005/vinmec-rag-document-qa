# web/app.py
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import unicodedata
import logging
from utils.performance import PerformanceTracker
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ======================================================
# LOGGING: HIỂN THỊ CẢ TERMINAL + WEB
# ======================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    logging.debug(msg)


# ======================================================
# LOAD ENV
# ======================================================

load_dotenv()


# ======================================================
# LOAD FAISS VECTORSTORE (ĐÚNG PATH)
# ======================================================

@st.cache_resource
def load_vectorstore():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

    log(f"Loading FAISS index from: {INDEX_PATH}")

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            f"❌ Không tìm thấy FAISS index tại {INDEX_PATH}\n"
            "👉 Hãy chạy: python scripts/build_index.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
llm = Ollama(model="qwen2.5:7b-instruct-q4_K_M")

# ======================================================
# NORMALIZE TIẾNG VIỆT (CÓ DẤU / KHÔNG DẤU)
# ======================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return set(normalize(text).split())

# ======================================================
# SPLIT MULTI QUESTION
# ======================================================

def split_questions(text: str):
    parts = re.split(r"[?\.!\n]+", text)
    return [p.strip() for p in parts if p.strip()]


# ======================================================
# SECTION DETECTOR
# ======================================================

SECTION_MAP = {
    "tong_quan": [
        "la gi",
        "tong quan",
        "gioi thieu",
    ],
    "muc_do_nguy_hiem": [
        "co nguy hiem",
        "nguy hiem khong",
        "nguy hiem",
        "co chet khong",
        "tu vong",
    ],
    "nguyen_nhan": [
        "nguyen nhan",
        "vi sao mac",
        "tai sao mac"
    ],
    "trieu_chung": [
        "trieu chung",
        "dau hieu",
        "bieu hien",
    ],
    "lay_truyen": [
        "lay truyen",
        "duong lay",
        "lien quan lay"
    ],
    "doi_tuong_nguy_co": [
        "nguy co",
        "doi tuong",
        "ai de mac"
    ],
    "phong_ngua": [
        "phong ngua",
        "phong tranh",
        "phong benh",
        "co phong duoc"
    ],
    "chan_doan": [
        "chan doan",
        "xet nghiem"
    ],
    "dieu_tri": [
        "dieu tri",
        "chua tri",
        "thuoc",
        "cach dieu tri"
    ],
}

def detect_section(query_norm: str):
    for section, keywords in SECTION_MAP.items():
        for kw in keywords:
            if kw in query_norm:
                return section
    return None

def classify_query(question: str, disease_key: str, docs_found: bool):
    if disease_key and docs_found:
        return "system"
    return "ollama"

def build_answer(docs, question):
    bullets = []
    for d in docs:
        content = d.page_content
        if "Nội dung:" in content:
            content = content.split("Nội dung:")[-1]

        bullets.append(content.strip())

    bullets = bullets[:3]

    answer = f"### ❓ {question}\n"
    for b in bullets:
        answer += f"- {b[:300]}\n"

    return answer

def generate_with_ollama(question, docs):
    context = "\n\n".join(d.page_content for d in docs[:3])

    prompt = f"""
Bạn là trợ lý y tế. 
Chỉ được trả lời dựa trên thông tin sau:

{context}

Câu hỏi: {question}

Trả lời ngắn gọn, dạng gạch đầu dòng.
Nếu không đủ thông tin, nói rõ không đủ dữ liệu.
"""

    return llm.invoke(prompt)

# ======================================================
# DISEASE RESOLVER (CỐT LÕI RAG Y TẾ)
# ======================================================

@st.cache_resource
def build_disease_index():
    disease_map = defaultdict(set)

    for doc in vectorstore.docstore._dict.values():
        meta = doc.metadata
        name = meta.get("disease", "")
        key = meta.get("disease_key")

        if not name or not key:
            continue

        pure = name.split(":")[0].strip()

        aliases = set()

        # Có dấu
        aliases.add(pure.lower())

        # Không dấu
        aliases.add(normalize(pure))

        # Không chữ "bệnh"
        no_prefix = pure.lower().replace("bệnh ", "")
        aliases.add(no_prefix)
        aliases.add(normalize(no_prefix))

        for a in aliases:
            disease_map[a].add(key)

    logging.debug(f"Disease aliases loaded: {len(disease_map)}")
    return disease_map

DISEASE_INDEX = build_disease_index()


def detect_disease(query: str):
    q = normalize(query)

    for alias, keys in DISEASE_INDEX.items():
        alias_norm = normalize(alias)
        q_tokens = set(q.split())
        alias_tokens = set(alias_norm.split())

        if alias_tokens.issubset(q_tokens):
            return list(keys)[0]

    return None

def detect_multiple_diseases(query):
    q = normalize(query)
    found = []

    for alias, keys in DISEASE_INDEX.items():
        alias_norm = normalize(alias)
        q_tokens = set(q.split())
        alias_tokens = set(alias_norm.split())

        if alias_tokens.issubset(q_tokens):
            found.extend(list(keys))

    return list(dict.fromkeys(found))

QUESTION_TEMPLATES = {
    "tong_quan": "Tổng quan",
    "muc_do_nguy_hiem": "Mức độ nguy hiểm",
    "nguyen_nhan": "Nguyên nhân",
    "trieu_chung": "Triệu chứng",
    "lay_truyen": "Đường lây truyền",
    "doi_tuong_nguy_co": "Đối tượng nguy cơ",
    "phong_ngua": "Phòng ngừa",
    "chan_doan": "Chẩn đoán",
    "dieu_tri": "Điều trị",
}

def render_answer(disease_name, section, docs, question):
    title = f"### ❓ {question}"
    header = f"**🦠 Tên bệnh:** {disease_name}"

    section_title = QUESTION_TEMPLATES.get(section, "Thông tin liên quan")

    body = []
    seen_texts = set()

    for d in docs[:5]:  # lấy tối đa 5 nhưng sẽ giới hạn 3 bullet hợp lệ
        raw_text = d.page_content

        # Tách phần sau "Nội dung:"
        if "Nội dung:" in raw_text:
            text = raw_text.split("Nội dung:")[-1].strip()
        else:
            text = raw_text.strip()

        # Loại bỏ trùng lặp
        text_norm = normalize(text)
        if text_norm in seen_texts:
            continue
        seen_texts.add(text_norm)

        # Giới hạn độ dài mỗi bullet
        if len(text) > 500:
            text = text[:500].rstrip() + "..."

        body.append(f"- {text}")

        # Chỉ hiển thị tối đa 3 bullet thực sự khác nhau
        if len(body) == 3:
            break

    # Fallback nếu không có nội dung hợp lệ
    if not body:
        body.append("- Không có dữ liệu chi tiết trong hệ thống.")

    return "\n".join([
        title,
        header,
        f"**📌 {section_title}:**",
        "",
        *body
    ])

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("🩺 Trợ lý tra cứu bệnh Vinmec (RAG MVP)")

query_input = st.text_area("Nhập câu hỏi (có thể nhiều câu):")

if st.button("Hỏi") and query_input:
    tracker = PerformanceTracker()
    tracker.mark("start")

    log(f"RAW INPUT: {query_input}")

    questions = split_questions(query_input)
    log(f"Detected {len(questions)} question(s)")

    all_answers = []
    all_sources = set()
    last_disease_key = None
    for question in questions:
        log(f"--- QUESTION: {question}")

        q_norm = normalize(question)
        log(f"Normalized: {q_norm}")

        detected_key = detect_disease(q_norm)

        if detected_key:
            disease_key = detected_key
            last_disease_key = detected_key
        else:
            disease_key = last_disease_key

        section = detect_section(q_norm)

        log(f"Disease key: {disease_key}")
        log(f"Section: {section}")
        COMPARE_KEYWORDS = ["va", "so sanh", "khac nhau", "phan biet"]

        is_compare = any(kw in q_norm for kw in COMPARE_KEYWORDS)
        # ROUTE DECISION
        route = "SYSTEM"

        if not disease_key:
            route = "OLLAMA_ONLY"
        elif is_compare:
            route = "SYSTEM+OLLAMA"
        # ==============================
        # DECIDE IF NEEDS LLM
        # ==============================
        needs_llm = False

        # Chỉ dùng LLM khi:
        # 1. So sánh
        # 2. Tổng hợp nhiều bệnh
        # 3. Không có section rõ ràng
        if is_compare:
            needs_llm = True
        elif section is None:
            needs_llm = True

        log(f"Is comparison question: {is_compare}")

        # ==========================================
        # CASE 1: CÓ BỆNH → SYSTEM RETRIEVAL
        # ==========================================

        if disease_key:
            docs = [
                d for d in vectorstore.docstore._dict.values()
                if d.metadata.get("disease_key") == disease_key
            ]

            log(f"Retrieved {len(docs)} raw docs for disease {disease_key}")

            for i, d in enumerate(docs[:3]):
                log(f"Doc {i+1} section: {d.metadata.get('section')}")

            log(f"Docs for disease: {len(docs)}")
            tracker.mark("retrieval_done")

            if section:
                filtered_docs = [
                    d for d in docs
                    if d.metadata.get("section") == section
                ]

                # chỉ áp dụng filter nếu có kết quả
                if filtered_docs:
                    docs = filtered_docs

            if not docs:
                all_answers.append(
                    f"❌ Không tìm thấy dữ liệu phù hợp cho câu hỏi: **{question}**"
                )
                continue

            # ==============================
            # CASE 1A: SO SÁNH
            # ==============================
            if is_compare:
                disease_keys = detect_multiple_diseases(q_norm)

                if len(disease_keys) >= 2:
                    bullets = ["🔎 So sánh mức độ nguy hiểm giữa các bệnh:"]

                    for dk in disease_keys[:2]:
                        docs_compare = [
                            d for d in vectorstore.docstore._dict.values()
                            if d.metadata.get("disease_key") == dk
                        ]

                        if docs_compare:
                            d = docs_compare[0]
                            clean = d.page_content.split("Nội dung:")[-1].strip()
                            bullets.append("- " + clean)
                            if d.metadata.get("url"):
                                all_sources.add(d.metadata["url"])
                    summary_prompt = f"""
                    Bạn là trợ lý y tế.

                    Dựa trên các thông tin sau, hãy đưa ra MỘT đoạn kết luận ngắn gọn,
                    trung lập, KHÔNG phán đoán tuyệt đối về mức độ nguy hiểm của bệnh.

                    Nếu dữ liệu chưa đủ để kết luận rõ ràng, hãy nói rõ điều đó.

                    Thông tin:
                    {chr(10).join(bullets)}
                    """

                    conclusion = llm.invoke(summary_prompt)

                    bullets.append(f"➡️ **Kết luận:** {conclusion}")
                    all_answers.append(
                        f"### ❓ {question}\n" + "\n".join(bullets)
                    )
                    continue

            # ==============================
            # CASE 1B: HỎI THÔNG THƯỜNG
            # ==============================
            if needs_llm:
                answer_text = generate_with_ollama(question, docs)
                answer = f"### ❓ {question}\n{answer_text}"
            else:
                disease_name = docs[0].metadata.get("disease", "").split(":")[0]
                answer = render_answer(
                    disease_name=disease_name,
                    section=section,
                    docs=docs,
                    question=question
                )

            all_answers.append(answer)

            for d in docs:
                if d.metadata.get("url"):
                    all_sources.add(d.metadata["url"])

            continue

        # ==========================================
        # CASE 2: KHÔNG CÓ BỆNH → FALLBACK FAISS
        # ==========================================

        docs_scores = vectorstore.similarity_search_with_score(question, k=5)

        for i, (doc, score) in enumerate(docs_scores):
            log(f"FAISS result {i+1} score={score} | section={doc.metadata.get('section')}")

        good = [d for d, s in docs_scores if s < 0.45]

        if not good:
            all_answers.append(
                f"❌ Không tìm thấy dữ liệu phù hợp cho câu hỏi: **{question}**"
            )
            continue

        if good:
            answer_text = generate_with_ollama(question, good)
            answer = f"### ❓ {question}\n{answer_text}"
            all_answers.append(answer)

            for d in good:
                if d.metadata.get("url"):
                    all_sources.add(d.metadata["url"])
        else:
            all_answers.append(
                f"❌ Không tìm thấy dữ liệu phù hợp cho câu hỏi: **{question}**"
            )

    # ======================================================
    # OUTPUT
    # ======================================================
    tracker.mark("answer_done")
    stats = tracker.summary()

    st.markdown("---")
    st.markdown("## ⚙️ Đánh giá hiệu năng")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⏱️ Tổng thời gian (s)", stats["total_time_sec"])
        st.metric("❓ Số câu hỏi", len(questions))

    with col2:
        st.metric("🧠 RAM sử dụng (MB)", stats["memory_mb"])
        st.metric("📄 Tổng câu trả lời", len(all_answers))

    st.markdown("### 🔍 Chi tiết")
    st.json(stats)

    st.markdown("---")
    st.markdown("## 📋 Kết quả")

    for ans in all_answers:
        st.markdown(ans)

    if all_sources:
        st.markdown("### 📚 Nguồn tham khảo:")
        for url in list(all_sources)[:3]:
            st.markdown(f"- {url}")

    st.markdown(
        "\n---\n**Thông tin tham khảo, không thay thế tư vấn y tế.**"
    )
