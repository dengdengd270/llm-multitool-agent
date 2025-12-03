# agent/tools_otto.py
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from db.mysql_client import query_otto_session_raw
import json


def otto_get_session_core(session_id: int) -> str:
    """
    调用 MySQL 查询 otto_test.raw，并把结果用 JSON 字符串返回给 LLM。
    让 LLM 自己去总结用户行为。
    """
    events: List[Dict[str, Any]] = query_otto_session_raw(session_id, limit=50)

    if not events:
        return f"[Otto] 未在 otto_test 表中找到 session_id={session_id} 的记录。"

    # 为了防止太长，只保留前 N 条事件
    max_events_to_show = 30
    events_trimmed = events[:max_events_to_show]

    payload = {
        "session_id": session_id,
        "num_events": len(events),
        "num_events_returned": len(events_trimmed),
        "events": events_trimmed,
    }

    # 返回 JSON 字符串，方便 LLM 解析
    return json.dumps(payload, ensure_ascii=False)


def otto_get_session_tool_fn(session_id: int) -> str:
    """
    给 LangChain 调用的封装函数。
    """
    return otto_get_session_core(session_id)


otto_get_session_tool = StructuredTool.from_function(
    name="otto_get_session",
    description=(
        "从 Otto MySQL 数据库的 otto_test 表中，按 session_id 查询用户行为原始 JSON 数据。"
        "通常用于分析某个用户会话里点击/加购/下单了哪些商品。"
    ),
    func=otto_get_session_tool_fn,
)
