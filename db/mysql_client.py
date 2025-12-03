# db/mysql_client.py
import os
import json
from typing import List, Dict, Any

import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

load_dotenv()

# 从环境变量读取数据库配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "otto")

# 创建一个简单的连接池，避免每次都新建连接
_connection_pool = None


def get_connection():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pooling.MySQLConnectionPool(
            pool_name="otto_pool",
            pool_size=5,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset="utf8mb4",
        )
    return _connection_pool.get_connection()


def query_otto_session_raw(session_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    按 session_id 查询 otto_test 表中的 raw JSON 列。
    返回一个列表，每个元素是一条事件记录（已经从 JSON 解析成 dict）。
    """
    conn = None
    cursor = None
    rows: List[Dict[str, Any]] = []

    try:
        conn = get_connection()
        cursor = conn.cursor()

        sql = """
        SELECT raw
        FROM otto_test
        WHERE session_id = %s
        LIMIT %s
        """
        cursor.execute(sql, (session_id, limit))

        for (raw_str,) in cursor.fetchall():
            try:
                if isinstance(raw_str, (bytes, bytearray)):
                    raw_str = raw_str.decode("utf-8")
                event = json.loads(raw_str)
                rows.append(event)
            except Exception:
                # 如果某一行 JSON 解析失败，跳过
                continue

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return rows
