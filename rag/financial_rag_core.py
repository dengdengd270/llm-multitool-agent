import os
import json
import pickle
import textwrap
from typing import List, Dict, Tuple

import numpy as np
import faiss
import jieba
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()   # 让 .env 生效


# =========================
# 路径与基础配置
# =========================

# LLM 项目根目录：.../Desktop/LLM
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 索引目录：.../Desktop/LLM/storage
STORE_DIR = os.path.join(BASE_DIR, "storage")

EMBEDDER_NAME = "BAAI/bge-small-zh-v1.5"

TOP_K_VEC      = 12     # 语义召回候选数
TOP_K_BM25     = 12     # 关键词召回候选数
ALPHA          = 0.6    # 混合打分：语义占比
TOP_FINAL      = 8      # 合并后的候选进候选池
TOP_CONTEXT    = 4      # 送入 LLM 的段数（自动缩放）
CTX_MAX_CHARS  = 1600   # 单段最大字符
TOTAL_CTX_CAP  = 3800   # 所有段合计上限（避免超长）
STRICT_MODE    = True   # 缺证据时强制拒答


# =========================
# 工具函数
# =========================

def short_name(doc_name: str) -> str:
    """
    FIN__2024__PBOC__China_Financial_Stability_Report__ZH.pdf
    -> PBOC-2024
    """
    parts = doc_name.split("__")
    return f"{parts[2]}-{parts[1]}" if len(parts) >= 3 else os.path.splitext(doc_name)[0]


def wrap(s: str, width: int = 120) -> str:
    out = []
    for line in s.splitlines():
        if len(line) > width:
            out.extend(textwrap.wrap(line, width=width, replace_whitespace=False))
        else:
            out.append(line)
    return "\n".join(out)


def is_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


# =========================
# 索引 & 向量模型（懒加载，避免重复初始化）
# =========================

_index = None
_metas = None
_bm25 = None
_bm25_metas = None
_embedder = None

def load_indexes():
    """加载 FAISS / BM25 / embedder，使用全局缓存避免重复加载。"""
    global _index, _metas, _bm25, _bm25_metas, _embedder

    if _index is None or _metas is None:
        faiss_path = os.path.join(STORE_DIR, "index.faiss")
        meta_path  = os.path.join(STORE_DIR, "meta.json")
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"索引文件不存在，请先在 LLM 项目下运行 build_index.py 构建索引：{STORE_DIR}"
            )
        _index = faiss.read_index(faiss_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            _metas = json.load(f)

    if _bm25 is None or _bm25_metas is None:
        bm25_path = os.path.join(STORE_DIR, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 索引不存在，请先构建：{bm25_path}")
        with open(bm25_path, "rb") as f:
            pack = pickle.load(f)
        _bm25, _bm25_metas = pack["bm25"], pack["metas"]

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_NAME)

    return _index, _metas, _bm25, _bm25_metas, _embedder


# =========================
# 检索（混合 + 去重 + 相邻页合并）
# =========================

def hybrid_search(
    query: str,
    k_vec: int = TOP_K_VEC,
    k_bm25: int = TOP_K_BM25,
    alpha: float = ALPHA,
    top_final: int = TOP_FINAL,
) -> List[Dict]:
    """
    基于你原来的混合检索 + 去重 + 相邻页合并逻辑，
    返回若干包含 text / doc_name / page / score_mix 的片段。
    """
    index, metas, bm25, bm25_metas, embedder = load_indexes()

    # 语义检索 (FAISS)
    qv = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(qv, dtype="float32"), k_vec)
    vec_hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m = metas[idx]
        vec_hits.append({"score_vec": float(score), **m})

    # BM25
    q_tok = list(jieba.cut(query))
    scores = bm25.get_scores(q_tok)
    order = np.argsort(scores)[::-1][:k_bm25]
    bm25_hits = [{"score_bm25": float(scores[i]), **bm25_metas[i]} for i in order]

    # 归一化 & 合并
    v_max = max([h["score_vec"] for h in vec_hits], default=1.0)
    b_max = max([h["score_bm25"] for h in bm25_hits], default=1.0)
    pool: Dict[Tuple[str, int, int], Dict] = {}

    for h in vec_hits:
        key = (h["doc_name"], h["page"], h["chunk_id"])
        pool[key] = {**h, "score_mix": alpha * (h["score_vec"] / v_max)}

    for h in bm25_hits:
        key = (h["doc_name"], h["page"], h["chunk_id"])
        add = (1 - alpha) * (h["score_bm25"] / b_max)
        if key in pool:
            pool[key]["score_mix"] += add
        else:
            pool[key] = {**h, "score_mix": add}

    merged = sorted(pool.values(), key=lambda x: x["score_mix"], reverse=True)[:top_final]

    # 同一文档同一页只保留分数更高的
    uniq_page: Dict[Tuple[str, int], Dict] = {}
    for h in merged:
        k = (h["doc_name"], h["page"])
        if k not in uniq_page or h["score_mix"] > uniq_page[k]["score_mix"]:
            uniq_page[k] = h
    merged = list(uniq_page.values())
    merged = sorted(merged, key=lambda x: x["score_mix"], reverse=True)

    # 相邻页合并
    merged_sorted = sorted(merged, key=lambda x: (x["doc_name"], x["page"], -x["score_mix"]))
    combined = []
    i = 0
    while i < len(merged_sorted):
        base = merged_sorted[i]
        buf_text = base["text"]
        base_pages = [base["page"]]
        j = i + 1
        while j < len(merged_sorted):
            nxt = merged_sorted[j]
            same_doc = (nxt["doc_name"] == base["doc_name"])
            close_pg = abs(nxt["page"] - base_pages[-1]) == 1 and len(buf_text) < CTX_MAX_CHARS * 0.8
            if same_doc and close_pg:
                buf_text = (buf_text + "\n" + nxt["text"])[: CTX_MAX_CHARS * 2]
                base_pages.append(nxt["page"])
                j += 1
            else:
                break
        base2 = dict(base)
        base2["text"] = buf_text
        base2["pages_merged"] = sorted(base_pages)
        combined.append(base2)
        i = j

    return combined


# =========================
# Prompt 构造
# =========================

SYSTEM_PROMPT = """你是金融行业的严谨分析助理。必须遵守：
1) 仅依据“证据片段”作答，不可编造。
2) 每个关键结论后附带 [来源:页码] 的引用，例如 [PBOC-2024:p.79]；可多条并列引用。
3) 若证据不足或无关，请明确回答“依据不足，建议补充资料”，不得想象补全。
输出要层次清晰、要点化，尽量简洁。"""

def select_contexts(cands: List[Dict], top_context: int = TOP_CONTEXT, total_cap: int = TOTAL_CTX_CAP) -> List[Dict]:
    """按混合分数排序，控制总字数上限，尽量保证 coverage。"""
    cands = sorted(cands, key=lambda x: x.get("score_mix", 0), reverse=True)
    chosen, used = [], 0
    for c in cands:
        text = c["text"].strip()
        allow = min(CTX_MAX_CHARS, max(600, total_cap - used))
        if allow <= 0:
            break
        c2 = dict(c)
        c2["text"] = text[:allow]
        chosen.append(c2)
        used += len(c2["text"])
        if len(chosen) >= top_context:
            break
    return chosen

def build_user_prompt(question: str, contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        tag = short_name(c["doc_name"])
        pages = c.get("pages_merged") or [c["page"]]
        page_str = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}–{pages[-1]}"
        txt = c["text"].strip()
        blocks.append(f"[{i}] ({tag} {page_str})\n{txt}")
    ctx = "\n\n".join(blocks)
    return f"问题：{question}\n\n证据片段：\n{ctx}\n\n请严格基于证据作答，并在关键结论处用 [来源:页码] 标注。"

def build_sources_footer(contexts: List[Dict]) -> str:
    """
    根据检索到的 contexts，构造一个简短的“数据来源”尾注。
    格式示例：
    数据来源：
    - PBOC-2024（p.79–80）
    - IMF-2023（p.12）
    """
    if not contexts:
        return ""

    lines = ["数据来源："]
    seen = set()

    for c in contexts:
        tag = short_name(c["doc_name"])
        pages = c.get("pages_merged") or [c["page"]]
        key = (tag, tuple(pages))
        if key in seen:
            continue
        seen.add(key)

        if len(pages) == 1:
            page_str = f"p.{pages[0]}"
        else:
            page_str = f"p.{pages[0]}–{pages[-1]}"

        lines.append(f"- {tag}（{page_str}）")

    return "\n".join(lines)

# =========================
# DeepSeek Chat（使用 OpenAI 兼容 SDK）
# =========================

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL_NAME = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("rag_core.py: 未找到 OPENAI_API_KEY，请确保环境变量或 .env 已配置好。")

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def deepseek_chat(messages, temperature: float = 0.2) -> str:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# =========================
# 对外主入口：answer_question
# =========================

def answer_question(question: str) -> str:
    """
    给 Agent 调用的主入口：
    - 输入：自然语言问题
    - 输出：基于报告检索 + LLM 总结的回答，并附带数据来源
    """
    hits = hybrid_search(
        question,
        k_vec=TOP_K_VEC,
        k_bm25=TOP_K_BM25,
        alpha=ALPHA,
        top_final=TOP_FINAL,
    )
    contexts = select_contexts(hits, top_context=TOP_CONTEXT, total_cap=TOTAL_CTX_CAP)

    # 先构造一个简单的“命中情况”描述，方便你调试 / 解释给面试官
    debug_info = f"\n\n【调试提示】命中片段数：{len(contexts)}"

    # 如果严格模式下证据明显不足，仍然走“依据不足”分支，
    # 但我们也会在末尾附上一个“数据来源：无有效片段”的提示。
    if STRICT_MODE and (len(contexts) == 0 or sum(len(c["text"]) for c in contexts) < 400):
        lang_hint = "用中文回答。" if is_chinese(question) else "Answer in English."
        base_msg = (
            "依据不足，建议补充资料。\n\n"
            + lang_hint
            + " 可以尝试：更换提问表述，或确认索引包含目标页码（例如 IMF WEO 2024 的“预测摘要”页）。"
        )
        footer = build_sources_footer(contexts)
        # 如果 contexts 为空，这里 footer 会是空字符串，我们给一个明确说明
        if not footer:
            footer = "数据来源：未检索到足够相关的报告片段。"
        full_text = base_msg + "\n\n" + footer + debug_info
        return wrap(full_text)

    # 证据充足的正常路径
    lang_hint = "用中文回答。" if is_chinese(question) else "Answer in English."
    user_content = build_user_prompt(question, contexts) + "\n\n" + lang_hint
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        ans = deepseek_chat(messages, temperature=0.15)
        footer = build_sources_footer(contexts)
        # 正常情况下 footer 一定非空，但这里还是兜个底
        if not footer:
            footer = "数据来源：命中了报告片段，但未能解析出文档名称或页码。"
        full_text = ans + "\n\n" + footer + debug_info
        return wrap(full_text)
    except Exception as e:
        return f"[ERR] 调用 DeepSeek 失败：{e}"
