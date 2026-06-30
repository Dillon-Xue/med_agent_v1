"""
数据库连接池模块
提供统一的 MySQL 连接池管理，避免每次操作新建连接
"""
import os
import logging
import pymysql
from DBUtils.PooledDB import PooledDB

logger = logging.getLogger(__name__)

# 数据库连接池（全局单例）
_pool = None


def get_pool():
    """
    获取数据库连接池（单例模式）
    如果连接池不存在，则创建
    """
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=10,      # 最大连接数
            mincached=2,            # 初始空闲连接数
            maxcached=5,            # 最大空闲连接数
            blocking=True,          # 连接耗尽时阻塞等待
            maxusage=None,          # 单个连接最大复用次数
            setsession=[],          # SQL 执行前执行的命令
            ping=1,                # 检查连接可用性
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "yourpassword"),
            database=os.getenv("DB_NAME", "patient_db"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("[DBPool] 数据库连接池初始化成功")
    return _pool


def get_connection():
    """
    从连接池获取一个连接
    使用完毕后不需要手动关闭，连接会自动归还到池中
    """
    return get_pool().connection()


def execute_query(sql: str, params: tuple = None):
    """
    执行查询语句，返回所有结果
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()


def execute_update(sql: str, params: tuple = None):
    """
    执行更新/插入/删除语句，返回影响的行数
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def close_pool():
    """
    关闭连接池（应用退出时调用）
    """
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("[DBPool] 数据库连接池已关闭")
