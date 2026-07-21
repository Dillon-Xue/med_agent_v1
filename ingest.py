import os
import sys
import json
import re
import base64
import requests
from dotenv import load_dotenv
import dashscope
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.embeddings import DashscopeEmbeddings

load_dotenv()

# Auto-detect Windows D drive project path in WSL
if os.path.exists("/mnt/d/A_Study/Agent/Med_Agent"):
    PROJECT_ROOT = "/mnt/d/A_Study/Agent/Med_Agent"
else:
    PROJECT_ROOT = os.getenv("MED_AGENT_ROOT", os.getcwd())

BASE_DATA = os.path.join(PROJECT_ROOT, "data")
BASE_VECTOR = os.path.join(PROJECT_ROOT, "vector_db")
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

embeddings = DashscopeEmbeddings()

CATEGORIES = ["rag", "literature", "drug", "guideline", "risk"]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Minimum text length threshold to consider a page as image-based
MIN_TEXT_LENGTH = 50

STATE_FILE = ".index_state.json"

# Patterns to extract figure/table IDs from text
FIGURE_PATTERNS = [
    r'图\s*(\d+(?:[\.-]\d+)?)',
    r'表\s*(\d+(?:[\.-]\d+)?)',
    r'Figure\s+(\d+(?:[\.-]\d+)?)',
    r'Fig\.\s*(\d+(?:[\.-]\d+)?)',
    r'Table\s+(\d+(?:[\.-]\d+)?)',
    r'Tbl\.\s*(\d+(?:[\.-]\d+)?)',
]


def analyze_page_image(pixmap_bytes, page_num, source_file):
    """Call Qwen-VL to analyze a rendered PDF page image."""
    img_b64 = base64.b64encode(pixmap_bytes).decode("utf-8")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print(f"    [Page {page_num}] No DASHSCOPE_API_KEY, skipping image analysis")
        return ""

    payload = {
        "model": "qwen-vl-plus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "请详细描述这张图片中的所有内容，包括文字、图表、产品信息、技术参数等。如果有表格，请尽量还原表格内容。直接输出描述，不要添加前缀。"}
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            print(f"    [Page {page_num}] Image analyzed, desc length={len(content)}")
            return content
        else:
            print(f"    [Page {page_num}] API error: {data}")
            return ""
    except Exception as e:
        print(f"    [Page {page_num}] Analysis failed: {e}")
        return ""


def extract_figure_ids(text):
    """Extract figure/table IDs like 图1, 表2, Figure 3, Table 1 from text."""
    ids = set()
    for p in FIGURE_PATTERNS:
        for m in re.finditer(p, text, re.IGNORECASE):
            ids.add(m.group(0))
    # Stable sort: Chinese first, then by number
    def sort_key(x):
        num = re.search(r'\d+(?:[\.-]\d+)?', x)
        n = float(num.group(0).replace('-', '.').replace(' ', '')) if num else 0
        prefix_order = 0 if x.startswith('图') else (1 if x.startswith('表') else 2)
        return (prefix_order, n)
    return sorted(list(ids), key=sort_key)


def get_file_signature(file_path):
    """Return a simple signature for change detection."""
    stat = os.stat(file_path)
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def load_state(vector_path):
    state_file = os.path.join(vector_path, STATE_FILE)
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(vector_path, state):
    state_file = os.path.join(vector_path, STATE_FILE)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_single_pdf(file_path):
    """Load and analyze a single PDF file using PyMuPDF (fitz)."""
    file = os.path.basename(file_path)
    print(f"  Loading: {file}")

    import fitz
    doc = fitz.open(file_path)
    pages = []
    for idx in range(len(doc)):
        page_num = idx + 1
        page = doc[idx]
        text = page.get_text()
        text_len = len(text.strip())

        if text_len < MIN_TEXT_LENGTH:
            print(f"    [Page {page_num}] Text too short ({text_len} chars), rendering image...")
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            desc = analyze_page_image(img_bytes, page_num, file)
            if desc:
                text = desc
        else:
            print(f"    [Page {page_num}] Text OK ({text_len} chars)")

        figure_ids = extract_figure_ids(text)
        pages.append(Document(
            page_content=text,
            metadata={
                "source": file,
                "page": page_num,
                "figure_ids": ",".join(figure_ids) if figure_ids else ""
            }
        ))

    doc.close()
    return pages


def delete_by_source(db, source_name):
    """Delete all chunks belonging to a given source file."""
    try:
        results = db.get(where={"source": source_name})
        if results and "ids" in results and results["ids"]:
            db.delete(ids=results["ids"])
            print(f"    Deleted old vectors for {source_name}")
    except Exception as e:
        print(f"    Warning: could not delete old vectors for {source_name}: {e}")


def add_chunk_index(chunks):
    page_chunks = {}
    for chunk in chunks:
        page_key = (chunk.metadata.get("source"), chunk.metadata.get("page", 1))
        if page_key not in page_chunks:
            page_chunks[page_key] = []
        page_chunks[page_key].append(chunk)

    for page_key, chunk_list in page_chunks.items():
        for idx, chunk in enumerate(chunk_list, 1):
            chunk.metadata["chunk_index"] = idx
    return chunks


def build_index(category):
    print(f"\n===== BUILD {category} =====")
    data_path = os.path.join(BASE_DATA, category)
    vector_path = os.path.join(BASE_VECTOR, category)

    if not os.path.exists(data_path):
        print(f"跳过 {category}: 数据路径不存在")
        return

    os.makedirs(vector_path, exist_ok=True)

    # Open or create vector DB
    has_chroma = os.path.exists(os.path.join(vector_path, "chroma.sqlite3"))
    if has_chroma:
        db = Chroma(
            persist_directory=vector_path,
            embedding_function=embeddings,
            collection_name="langchain"
        )
        print(f"  Loaded existing vector DB")
    else:
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_path,
            collection_name="langchain"
        )
        print(f"  Created new vector DB")

    state = load_state(vector_path)

    # Migration: if an old DB exists but has no state file, rebuild from scratch
    if has_chroma and not state:
        print("  检测到旧版向量库，执行全量重建...")
        try:
            all_data = db.get()
            if all_data and "ids" in all_data and all_data["ids"]:
                db.delete(ids=all_data["ids"])
                print("  已清空旧数据")
        except Exception as e:
            print(f"  清空旧数据时出错: {e}")
        state = {}

    # Scan current PDFs
    current_files = {}
    for file in os.listdir(data_path):
        if not file.endswith(".pdf"):
            continue
        file_path = os.path.join(data_path, file)
        current_files[file] = get_file_signature(file_path)

    known_files = set(state.keys())
    current_set = set(current_files.keys())

    to_remove = known_files - current_set
    to_add = []
    to_update = []

    for file, sig in current_files.items():
        if file not in state:
            to_add.append(file)
        elif state[file] != sig:
            to_update.append(file)

    if not to_remove and not to_add and not to_update:
        print(f"  {category}: 所有PDF均为最新，无需更新")
        return

    print(f"  Remove: {len(to_remove)}, Add: {len(to_add)}, Update: {len(to_update)}")

    # Process removals
    for file in to_remove:
        delete_by_source(db, file)
        del state[file]

    # Process additions and updates
    for file in to_add + to_update:
        file_path = os.path.join(data_path, file)
        if file in state:
            delete_by_source(db, file)

        docs = load_single_pdf(file_path)
        if not docs:
            continue
        chunks = splitter.split_documents(docs)
        chunks = add_chunk_index(chunks)
        print(f"    Adding {len(chunks)} chunks for {file}")
        db.add_documents(chunks)
        state[file] = current_files[file]

    save_state(vector_path, state)
    print(f"{category} DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = CATEGORIES
    for c in targets:
        build_index(c)
    print("\nALL DONE")
