# scripts/rag_cli.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.performance import PerformanceTracker

VECTOR_DIR = "vectorstore/faiss_index"
TOP_K = 3
SCORE_THRESHOLD = 0.3

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = FAISS.load_local(VECTOR_DIR, embedder, allow_dangerous_deserialization=True)

def ask(question: str):
    tracker = PerformanceTracker()
    tracker.mark("start")

    docs = db.similarity_search_with_score(question, k=TOP_K)

    tracker.mark("retrieval_done")

    good_docs = [d for d, score in docs if score < SCORE_THRESHOLD]

    if not good_docs:
        print("❌ Chưa tìm thấy thông tin phù hợp trong dữ liệu hiện có.")
        print("👉 Vui lòng hỏi cụ thể hơn.")
        return

    tracker.mark("answer_ready")
    stats = tracker.summary()

    print("\n⚙️ ĐÁNH GIÁ HIỆU NĂNG:")
    print(f"- Thời gian truy vấn FAISS: {tracker.elapsed('start', 'retrieval_done'):.3f}s")
    print(f"- Tổng thời gian xử lý: {stats['total_time_sec']}s")
    print(f"- RAM sử dụng: {stats['memory_mb']} MB")
    print(f"- Số chunk phù hợp: {len(good_docs)}")

    print("\n📌 Trả lời (tóm tắt):")
    for d in good_docs:
        print("-", d.page_content[:200], "...")

    print("\n🔗 Nguồn tham khảo:")
    shown = set()
    for d in good_docs:
        url = d.metadata["url"]
        if url not in shown:
            print("-", url)
            shown.add(url)

    print("\n⚠️ Thông tin tham khảo, không thay thế tư vấn y tế.")

if __name__ == "__main__":
    while True:
        q = input("\n❓ Câu hỏi (enter để thoát): ")
        if not q:
            break
        ask(q)
