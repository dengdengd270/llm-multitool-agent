# agent/tools_otto.py
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.tools import StructuredTool
from db.mysql_client import query_otto_session_raw


# 项目根目录：.../llm-multitool-agent
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 约定：推荐解释文件放在 data/ 下
_DEFAULT_RECO_PATH = os.path.join(_PROJECT_ROOT, "data", "reco_explained.jsonl")


# -----------------------------
# 推荐解释文件读取（带缓存）
# -----------------------------
_RECO_CACHE: Optional[Dict[int, Dict[str, Any]]] = None
_RECO_CACHE_PATH: Optional[str] = None


def _maybe_load_reco_cache(reco_path: str) -> None:
    """
    把 reco_explained.jsonl 全量载入内存（适合 explain_n 不大、文件不大的情况）。
    如果你未来把 explain_n 开很大，也能用；只是会多占内存。
    """
    global _RECO_CACHE, _RECO_CACHE_PATH

    reco_path = os.path.abspath(reco_path)

    # 已加载且路径一致
    if _RECO_CACHE is not None and _RECO_CACHE_PATH == reco_path:
        return

    if not os.path.exists(reco_path):
        # 不抛异常，交给上层友好返回
        _RECO_CACHE = None
        _RECO_CACHE_PATH = reco_path
        return

    cache: Dict[int, Dict[str, Any]] = {}
    with open(reco_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("session")
            if sid is None:
                continue
            try:
                sid_int = int(sid)
            except Exception:
                continue
            cache[sid_int] = obj

    _RECO_CACHE = cache
    _RECO_CACHE_PATH = reco_path


def _get_reco_by_session(session_id: int, reco_path: str) -> Optional[Dict[str, Any]]:
    _maybe_load_reco_cache(reco_path)
    if _RECO_CACHE is None:
        return None
    return _RECO_CACHE.get(int(session_id))


# -----------------------------
# 行为数据解析（稳健版）
# -----------------------------
def _extract_events(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    query_otto_session_raw 可能返回：
    - 已经是事件列表：[{ts, aid, type}, ...]
    - 或者每行是 {"raw": {...}} / {"raw": "...json string..."} 之类
    这里做一个尽可能稳健的提取。
    """
    events: List[Dict[str, Any]] = []

    # 情况1：直接就是事件 dict 列表
    if raw_rows and isinstance(raw_rows[0], dict) and ("ts" in raw_rows[0] or "aid" in raw_rows[0] or "type" in raw_rows[0]):
        for e in raw_rows:
            if isinstance(e, dict):
                events.append(e)
        return events

    # 情况2：每行包一层 raw
    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        payload = None
        if "raw" in row:
            payload = row["raw"]
        else:
            payload = row

        # raw 可能是字符串 JSON
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        if isinstance(payload, dict):
            # 常见：{"events": [...]} 或 {"session":..., "events":[...]}
            if "events" in payload and isinstance(payload["events"], list):
                for e in payload["events"]:
                    if isinstance(e, dict):
                        events.append(e)
            # 也可能是 {"ts":..., "aid":..., "type":...}
            elif any(k in payload for k in ["ts", "aid", "type"]):
                events.append(payload)

        elif isinstance(payload, list):
            for e in payload:
                if isinstance(e, dict):
                    events.append(e)

    return events


def _summarize_session(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    只做“可验证”的统计，不做任何商品属性臆测。
    """
    clicks, carts, orders = [], [], []
    ts_list: List[int] = []

    def _norm_type(t: Any) -> str:
        if t is None:
            return ""
        s = str(t).lower().strip()
        # OTTO 常见是 clicks/carts/orders 或 0/1/2
        if s in ["0", "click", "clicks"]:
            return "clicks"
        if s in ["1", "cart", "carts"]:
            return "carts"
        if s in ["2", "order", "orders"]:
            return "orders"
        return s

    def _get_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    cleaned_events: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        aid = _get_int(e.get("aid"))
        ts = _get_int(e.get("ts"))
        typ = _norm_type(e.get("type"))

        if aid is None and ts is None and not typ:
            continue

        if ts is not None:
            ts_list.append(ts)

        if typ == "clicks" and aid is not None:
            clicks.append(aid)
        elif typ == "carts" and aid is not None:
            carts.append(aid)
        elif typ == "orders" and aid is not None:
            orders.append(aid)

        cleaned_events.append({"ts": ts, "aid": aid, "type": typ})

    ts_list_sorted = sorted([t for t in ts_list if isinstance(t, int)])
    duration_ms = None
    if len(ts_list_sorted) >= 2:
        duration_ms = ts_list_sorted[-1] - ts_list_sorted[0]

    return {
        "counts": {
            "events": len(cleaned_events),
            "clicks": len(clicks),
            "carts": len(carts),
            "orders": len(orders),
        },
        "unique_aids": {
            "clicks": sorted(list(set(clicks)))[:50],
            "carts": sorted(list(set(carts)))[:50],
            "orders": sorted(list(set(orders)))[:50],
        },
        "time": {
            "start_ts": ts_list_sorted[0] if ts_list_sorted else None,
            "end_ts": ts_list_sorted[-1] if ts_list_sorted else None,
            "duration_ms": duration_ms,
        },
    }


# -----------------------------
# 工具 1：查会话原始行为（MySQL）
# -----------------------------
def otto_get_session_core(session_id: int) -> str:
    raw_rows: List[Dict[str, Any]] = query_otto_session_raw(session_id, limit=50)
    if not raw_rows:
        return f"[Otto] 未找到 session_id={session_id} 的记录。"

    events = _extract_events(raw_rows)
    if not events:
        return f"[Otto] session_id={session_id} 有记录，但无法解析出 events。"

    # 控制返回长度
    max_events_to_show = 30
    events_trimmed = events[:max_events_to_show]

    payload = {
        "session_id": int(session_id),
        "session_summary": _summarize_session(events),
        "num_events_total": len(events),
        "num_events_returned": len(events_trimmed),
        "events": events_trimmed,
        "rules": [
            "仅基于 events 做复盘与统计，不要推断商品属性（品类/品牌/价格等）。",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def otto_get_session_tool_fn(session_id: int) -> str:
    return otto_get_session_core(session_id)


otto_get_session_tool = StructuredTool.from_function(
    name="otto_get_session",
    description="按 session_id 查询 otto_test 中的原始用户行为（click/cart/order），并返回结构化 JSON 供分析。",
    func=otto_get_session_tool_fn,
)


# -----------------------------
# 工具 2：取离线推荐解释（reco_explained.jsonl）
# -----------------------------
def otto_reco_explain_core(session_id: int, topn: int = 10) -> str:
    reco_path = _DEFAULT_RECO_PATH

    if not os.path.exists(reco_path):
        return "[OttoReco] reco_explained.jsonl 不存在或未放到 data/ 下。"

    reco_obj = _get_reco_by_session(int(session_id), reco_path)
    if not reco_obj:
        return f"[OttoReco] reco_explained.jsonl 中未找到 session={session_id} 的推荐解释记录。"

    top20 = reco_obj.get("top20", [])
    if not isinstance(top20, list) or not top20:
        return f"[OttoReco] session={session_id} 的 top20 为空或格式不正确。"

    topn = int(topn)
    if topn <= 0:
        topn = 10
    topn = min(topn, len(top20))

    payload = {
        "session_id": int(session_id),
        "topn": topn,
        "items": top20[:topn],
        "notes": [
            "items[i].aid 是商品 ID；我们没有任何商品属性信息，不要臆测品类/品牌/价格。",
            "解释理由必须引用 features（例如 feat_covis、feat_last_gap_min、feat_seen_cnt 等）。",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def otto_reco_explain_tool_fn(session_id: int, topn: int = 10) -> str:
    return otto_reco_explain_core(session_id=session_id, topn=topn)


otto_reco_explain_tool = StructuredTool.from_function(
    name="otto_reco_explain",
    description="按 session_id 返回离线 LGBM 排序器的 topN 推荐结果及特征，用于生成可解释推荐理由。",
    func=otto_reco_explain_tool_fn,
)


# -----------------------------
# 工具 3：一键“行为复盘 + 推荐输出 + 解释输入”整包
# -----------------------------
def otto_reco_story_core(session_id: int, topn: int = 10) -> str:
    # 1) 行为（来自 MySQL）
    raw_rows: List[Dict[str, Any]] = query_otto_session_raw(session_id, limit=50)
    if not raw_rows:
        return f"[OttoStory] 未找到 session_id={session_id} 的会话记录。"

    events = _extract_events(raw_rows)
    if not events:
        return f"[OttoStory] session_id={session_id} 有记录，但无法解析出 events。"

    max_events_to_show = 30
    events_trimmed = events[:max_events_to_show]
    sess_summary = _summarize_session(events)

    # 2) 推荐（来自 reco_explained.jsonl）
    reco_path = _DEFAULT_RECO_PATH
    reco_pack: Dict[str, Any]
    if os.path.exists(reco_path):
        reco_obj = _get_reco_by_session(int(session_id), reco_path)
    else:
        reco_obj = None

    if reco_obj and isinstance(reco_obj.get("top20"), list) and reco_obj["top20"]:
        top20 = reco_obj["top20"]
        topn = int(topn)
        if topn <= 0:
            topn = 10
        topn = min(topn, len(top20))
        reco_pack = {
            "topn": topn,
            "items": top20[:topn],
        }
    else:
        reco_pack = {
            "warning": "未找到该 session 的推荐解释记录（top20）。",
            "topn": int(topn),
            "items": [],
        }

    payload = {
        "session_id": int(session_id),
        "session_summary": sess_summary,
        "events": events_trimmed,
        "recommendation": reco_pack,
        "output_requirements": {
            "structure": [
                "1) 会话复盘：按时间解释用户做了什么（click/cart/order），列出涉及的 aid，避免臆测商品属性。",
                "2) 推荐 TopN：逐条给出 aid + 一句理由，理由必须引用 features 或会话行为统计（例如会话很短、最近性、共现等）。",
                "3) 总体画像/策略总结：说明这次推荐主要依赖哪些信号（共现/最近性/热门补全/会话长度等），并给出一句总体结论。",
            ],
            "constraints": [
                "不要出现任何商品品类/价格/品牌等描述（我们只有 aid）。",
                "不要把 features 解释成‘用户喜欢某某类’，只能说‘共现强/最近性强/出现次数/热门得分’这类可验证表述。",
                "如果 recommendation.items 为空，只输出会话复盘，并说明缺少推荐解释数据。",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def otto_reco_story_tool_fn(session_id: int, topn: int = 10) -> str:
    return otto_reco_story_core(session_id=session_id, topn=topn)


otto_reco_story_tool = StructuredTool.from_function(
    name="otto_reco_story",
    description="返回一个完整数据包：MySQL 会话行为复盘 + 离线 LGBM 推荐 topN 与特征。用于生成更真实的‘行为->推荐->理由’输出。",
    func=otto_reco_story_tool_fn,
)
