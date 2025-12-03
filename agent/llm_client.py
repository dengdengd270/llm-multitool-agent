# agent/llm_client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 读取 .env
load_dotenv()

def get_llm():
    """
    统一创建一个 ChatOpenAI（DeepSeek 兼容）实例，后面 Agent / RAG 都用它。
    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("未找到 OPENAI_API_KEY，请在项目根目录配置 .env")

    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.2,
    )
    return llm
