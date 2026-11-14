# scripts/build_index.py
import re
import json
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

RAW_PATH = Path("data/raw/vinmec_all_diseases.json")
VECTOR_DIR = Path("vectorstore/faiss_index")

# ================= UTILS =================

def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= MAIN =================
SECTION_NORMALIZER = {
    "tong_quan": [
        "tong quan", "gioi thieu", "dinh nghia"
    ],
    "nguyen_nhan": [
        "nguyen nhan"
    ],
    "trieu_chung": [
        "trieu chung", "dau hieu", "bieu hien"
    ],
    "lay_truyen": [
        "lay truyen", "duong lay"
    ],
    "doi_tuong_nguy_co": [
        "nguy co", "doi tuong"
    ],
    "phong_ngua": [
        "phong ngua", "phong tranh"
    ],
    "chan_doan": [
        "chan doan", "xet nghiem"
    ],
    "dieu_tri": [
        "dieu tri", "chua tri"
    ],
    "muc_do_nguy_hiem": [
        "nguy hiem",
        "bien chung",
        "tu vong",
        "tien luong",
        "hau qua",
        "dien tien",
        "co chet",
        "nguy co",
        "anh huong tinh mang"
    ]
}

def map_section(title: str):
    t = normalize(title)
    for key, keywords in SECTION_NORMALIZER.items():
        for kw in keywords:
            if kw in t:
                return key
    return "tong_quan"

def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    metadatas = []

    for disease in raw["diseases"]:
        # ===== FIX QUAN TRỌNG =====
        # disease_name trong data thực tế là FULL TITLE
        # VD: "Bệnh dại: Nguyên nhân, triệu chứng, chẩn đoán và điều trị"
        # → chỉ lấy "Bệnh dại"
        full_title = disease.get("disease_name", "").strip()
        pure_name = full_title.split(":")[0].strip()

        disease_key = "benh_" + normalize(pure_name).replace(" ", "_")

        for sec in disease.get("sections", []):
            content = sec.get("content", "").strip()
            title = sec.get("title", "").strip()

            if len(content) < 50:
                continue

            text = f"""
Tên bệnh: {pure_name}
Mục: {title}
Nội dung: {content}
""".strip()

            docs.append(text)
            metadatas.append({
                "disease": full_title,
                "disease_key": disease_key,
                "section": map_section(title),  # ✅ FIX
                "url": disease.get("url", "")
            })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ";"]
    )

    split_docs = splitter.create_documents(docs, metadatas=metadatas)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = FAISS.from_documents(split_docs, embedder)
    db.save_local(VECTOR_DIR)

    print(f"✅ Indexed {len(split_docs)} chunks")
    print("🔑 Example disease_key:", split_docs[0].metadata["disease_key"])

if __name__ == "__main__":
    main()
