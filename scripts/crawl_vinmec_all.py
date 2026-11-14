# =====================================================
# VINMEC FULL DISEASE CRAWLER (A–Z, MULTI-PAGE)
# PRINT FULL JSON TO TERMINAL
# =====================================================

import requests
import json
import time
import re
import string
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# ================= CONFIG =================
BASE_URL = "https://www.vinmec.com"
LETTERS = list(string.ascii_lowercase)  # a -> z
SLEEP_TIME = 1.0

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (VinmecCrawler/1.0; +https://vinmec.com)"
}

# ================= UTILS =================
def fetch_html(url, retry=3):
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(2)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def is_noise(text: str) -> bool:
    noise_patterns = [
        r"^Xem thêm",
        r"^Bài viết liên quan",
        r"^Tìm hiểu thêm",
        r"^Nguồn:",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in noise_patterns)

# ================= STEP 1: GET ALL DISEASE LINKS (MULTI PAGE) =================
def get_all_diseases_of_letter(letter):
    all_diseases = []
    page = 1

    while True:
        if page == 1:
            url = f"{BASE_URL}/vie/tra-cuu-benh/{letter}"
        else:
            url = f"{BASE_URL}/vie/tra-cuu-benh/{letter}/page_{page}"

        print(f"      📄 Page {page}")

        try:
            soup = BeautifulSoup(fetch_html(url), "lxml")
        except Exception:
            break

        found = []
        for a in soup.select('a[href^="/vie/benh/"]'):
            href = a.get("href", "")
            if re.search(r"-\d+$", href):
                found.append({
                    "letter": letter,
                    "disease_name": clean_text(a.get_text()),
                    "url": BASE_URL + href
                })

        if not found:
            break

        uniq = {d["url"]: d for d in found}
        all_diseases.extend(uniq.values())

        page += 1
        time.sleep(0.5)

    # deduplicate toàn chữ
    return list({d["url"]: d for d in all_diseases}.values())

# ================= STEP 2: FIND MAIN CONTENT =================
def find_main_content(soup):
    candidates = []

    for div in soup.find_all("div"):
        p_count = len(div.find_all("p"))
        h_count = len(div.find_all(["h2", "h3"]))
        text_len = len(div.get_text(strip=True))

        if p_count >= 3 and text_len > 500:
            score = p_count * 2 + h_count
            candidates.append((score, div))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ================= STEP 3: PARSE DISEASE DETAIL =================
def parse_disease(disease):
    soup = BeautifulSoup(fetch_html(disease["url"]), "lxml")
    content = find_main_content(soup)

    if not content:
        return {
            **disease,
            "sections": [],
            "error": "NO_MAIN_CONTENT"
        }

    sections = []
    current_title = "Giới thiệu"
    buffer = []
    seen = set()

    def flush():
        if buffer:
            sections.append({
                "title": current_title,
                "content": " ".join(buffer)
            })

    for tag in content.find_all(["h2", "h3", "p", "li"]):
        if tag.name in ["h2", "h3"]:
            flush()
            current_title = clean_text(tag.get_text())
            buffer.clear()
            continue

        if tag.find("a"):
            continue

        text = clean_text(tag.get_text())
        if not text or is_noise(text) or text in seen:
            continue

        seen.add(text)
        buffer.append(text)

    flush()

    return {
        **disease,
        "sections": sections
    }

# ================= RUN =================
def main():
    all_diseases = []

    for letter in LETTERS:
        print(f"\n🔠 Crawling letter: {letter.upper()}")
        diseases = get_all_diseases_of_letter(letter)
        print(f"   → Found {len(diseases)} diseases")

        for d in diseases:
            try:
                detail = parse_disease(d)
                all_diseases.append(detail)
                print(f"   ✔ {detail['disease_name']}")
            except Exception:
                print(f"   ❌ Error: {d['disease_name']}")

            time.sleep(SLEEP_TIME)

    final_data = {
        "source": "Vinmec",
        "total_diseases": len(all_diseases),
        "crawled_at": datetime.utcnow().isoformat(),
        "diseases": all_diseases
    }

    output_file = OUTPUT_DIR / "vinmec_all_diseases.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # ======= PRINT FULL JSON TO TERMINAL =======
    print("\n================ FULL JSON OUTPUT ================\n")
    print(json.dumps(final_data, ensure_ascii=False, indent=2))

    print(f"\n✅ Saved to: {output_file.resolve()}")

if __name__ == "__main__":
    main()