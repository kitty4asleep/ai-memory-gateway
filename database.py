"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================

包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
- 旧记忆结构化回填
- 主动浮现记忆
"""

import os
import re
from typing import Optional, List

import asyncpg
import jieba
import jieba.analyse


DATABASE_URL = os.getenv("DATABASE_URL", "")

# 搜索权重
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))
WEIGHT_AROUSAL = float(os.getenv("WEIGHT_AROUSAL", "0.12"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.15"))

# 主动浮现阈值
SURFACE_MIN_SCORE = float(os.getenv("SURFACE_MIN_SCORE", "1.15"))


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def ensure_backfilled_at_column():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS backfilled_at TIMESTAMPTZ;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_backfilled_at
            ON memories(backfilled_at);
        """)


async def init_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                source_session TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_accessed TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS memory_type TEXT DEFAULT 'event';
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS resolved BOOLEAN DEFAULT FALSE;
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS activation_count INTEGER DEFAULT 0;
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT FALSE;
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS valence REAL;
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS arousal REAL;
        """)

        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS project TEXT;
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts
            ON memories USING gin(to_tsvector('simple', content));
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations (session_id, created_at);
        """)

    await ensure_backfilled_at_column()
    print("✅ 数据库表结构已就绪")


# ============================================================
# 中文分词工具（基于 jieba）
# ============================================================

jieba.setLogLevel(jieba.logging.INFO)

EN_WORD_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9_./-]*')
NUM_PATTERN = re.compile(r'\d{2,}')

_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "和", "与", "也", "都", "又", "就", "但", "而", "或", "到", "被",
    "把", "让", "从", "对", "为", "以", "及", "等", "个", "不", "没", "很",
    "太", "吗", "呢", "吧", "啊", "嗯", "哦", "哈", "呀", "嘛", "么", "啦",
    "哇", "喔", "会", "能", "要", "想", "去", "来", "说", "做", "看", "给",
    "上", "下", "里", "中", "大", "小", "多", "少", "好", "可以", "什么",
    "怎么", "如何", "哪里", "哪个", "为什么", "还是", "然后", "因为", "所以",
    "虽然", "但是", "已经", "一个", "一些", "一下", "一点", "一起", "一样",
    "比较", "应该", "可能", "如果", "这个", "那个", "自己", "知道", "觉得",
    "感觉", "时候", "现在",
    # 新增弱词 / 时间词 / 语气词
    "今天", "昨天", "明天", "刚刚", "刚才", "最近", "刚", "这次", "那次",
    "有点", "一点点", "一点儿", "一下子", "真的", "就是", "然后呢", "然后捏",
    "累累", "嘟", "累累嘟", "呜呜", "嘿嘿", "宝宝", "宝贝", "老公",
})


def extract_search_keywords(query: str) -> List[str]:
    keywords = set()

    for match in EN_WORD_PATTERN.finditer(query):
        word = match.group().strip()
        if len(word) >= 2:
            keywords.add(word)

    for match in NUM_PATTERN.finditer(query):
        keywords.add(match.group())

    words = jieba.cut(query, cut_all=False)
    for word in words:
        word = word.strip()
        if not word:
            continue

        if EN_WORD_PATTERN.fullmatch(word) or NUM_PATTERN.fullmatch(word):
            continue

        if len(word) < 2 or word in _STOP_WORDS:
            continue

        keywords.add(word)

    return list(keywords)


# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
            session_id, role, content, model
        )


async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM conversations
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id, limit
        )
        return list(reversed(rows))


# ============================================================
# 记忆操作
# ============================================================

async def save_memory(
    content: str,
    importance: int = 5,
    source_session: str = "",
    memory_type: str = "event",
    resolved: bool = False,
    activation_count: int = 0,
    pinned: bool = False,
    valence: float = None,
    arousal: float = None,
    project: str = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memories
            (
                content, importance, source_session,
                memory_type, resolved, activation_count,
                pinned, valence, arousal, project
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            content, importance, source_session,
            memory_type, resolved, activation_count,
            pinned, valence, arousal, project
        )


async def search_memories(query: str, limit: int = 10):
    keywords = extract_search_keywords(query)
    if not keywords:
        print(f"🔍 搜索 '{query}' → 提取不到有效关键词，跳过检索")
        return []

    pool = await get_pool()
    async with pool.acquire() as conn:
        case_parts = []
        params = []

        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)

        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)

        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = " OR ".join(where_parts)

        limit_idx = len(keywords) + 1
        params.append(limit)

        sql = f"""
            SELECT
                id,
                content,
                importance,
                memory_type,
                resolved,
                activation_count,
                pinned,
                valence,
                arousal,
                project,
                created_at,
                ({hit_count_expr}) AS hit_count,
                (
                    {WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0
                    + {WEIGHT_IMPORTANCE} * importance::float / 10.0
                    + {WEIGHT_RECENCY} * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0))
                    + 0.10 * CASE WHEN resolved = FALSE THEN 1 ELSE 0 END
                    + 0.15 * CASE WHEN pinned = TRUE THEN 1 ELSE 0 END
                    + {WEIGHT_AROUSAL} * COALESCE(arousal, 0)
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, created_at DESC
            LIMIT ${limit_idx}
        """

        results = await conn.fetch(sql, *params)

        if MIN_SCORE_THRESHOLD > 0:
            before_count = len(results)
            results = [r for r in results if r["score"] >= MIN_SCORE_THRESHOLD]
            filtered = before_count - len(results)
        else:
            filtered = 0

        if results:
            print(
                f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords) > 8 else ''} "
                f"→ 命中 {len(results)} 条"
                + (f"（过滤 {filtered} 条低分）" if filtered else "")
            )
            for r in results[:3]:
                print(
                    f"   📌 [score={r['score']:.3f}] "
                    f"(hits={r['hit_count']}, imp={r['importance']}, "
                    f"type={r.get('memory_type')}, resolved={r.get('resolved')}, "
                    f"pinned={r.get('pinned')}, arousal={r.get('arousal')}) "
                    f"{r['content'][:60]}..."
                )

            ids = [r["id"] for r in results]
            await conn.execute(
                """
                UPDATE memories
                SET last_accessed = NOW(),
                    activation_count = COALESCE(activation_count, 0) + 1
                WHERE id = ANY($1::int[])
                """,
                ids,
            )
        else:
            print(
                f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果"
                + (f"（{filtered} 条被分数阈值过滤）" if filtered else "")
            )

        return results


async def get_surface_memories(limit: int = 3, exclude_ids: List[int] = None):
    """
    主动浮现记忆：
    不依赖当前 query，优先选择值得被“自然想起”的记忆。
    现在加入最小 surface_score 阈值，避免每次机械拉满。
    """
    exclude_ids = exclude_ids or []

    pool = await get_pool()
    async with pool.acquire() as conn:
        if exclude_ids:
            rows = await conn.fetch(
                """
                SELECT *
                FROM (
                    SELECT
                        id, content, importance, memory_type, resolved, activation_count,
                        pinned, valence, arousal, project, created_at, last_accessed,
                        (
                            0.35 * CASE WHEN pinned = TRUE THEN 1 ELSE 0 END
                            + 0.30 * CASE WHEN resolved = FALSE THEN 1 ELSE 0 END
                            + 0.20 * COALESCE(arousal, 0)
                            + 0.10 * (importance::float / 10.0)
                            + 0.05 * LEAST(COALESCE(activation_count, 0), 10)::float / 10.0
                        ) AS surface_score
                    FROM memories
                    WHERE id != ALL($1::int[])
                ) t
                WHERE surface_score >= $2
                ORDER BY
                    surface_score DESC,
                    COALESCE(last_accessed, created_at) DESC
                LIMIT $3
                """,
                exclude_ids,
                SURFACE_MIN_SCORE,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT *
                FROM (
                    SELECT
                        id, content, importance, memory_type, resolved, activation_count,
                        pinned, valence, arousal, project, created_at, last_accessed,
                        (
                            0.35 * CASE WHEN pinned = TRUE THEN 1 ELSE 0 END
                            + 0.30 * CASE WHEN resolved = FALSE THEN 1 ELSE 0 END
                            + 0.20 * COALESCE(arousal, 0)
                            + 0.10 * (importance::float / 10.0)
                            + 0.05 * LEAST(COALESCE(activation_count, 0), 10)::float / 10.0
                        ) AS surface_score
                    FROM memories
                ) t
                WHERE surface_score >= $1
                ORDER BY
                    surface_score DESC,
                    COALESCE(last_accessed, created_at) DESC
                LIMIT $2
                """,
                SURFACE_MIN_SCORE,
                limit,
            )

        results = [dict(r) for r in rows]
        if results:
            print(f"🌫️ 主动浮现候选命中 {len(results)} 条（阈值={SURFACE_MIN_SCORE}）")
            for r in results:
                print(
                    f"   🌫️ [surface={r['surface_score']:.3f}] "
                    f"(imp={r['importance']}, resolved={r['resolved']}, pinned={r['pinned']}, "
                    f"arousal={r['arousal']}) {r['content'][:70]}..."
                )
        else:
            print(f"🌫️ 主动浮现候选 0 条（阈值={SURFACE_MIN_SCORE}）")

        return results


async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT
                id, content, importance, memory_type, resolved, activation_count,
                pinned, valence, arousal, project, created_at, backfilled_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )


async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]


async def get_all_memories():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                content, importance, source_session,
                memory_type, resolved, activation_count,
                pinned, valence, arousal, project, created_at, backfilled_at
            FROM memories
            ORDER BY id
        """)
        return [dict(r) for r in rows]


async def get_all_memories_detail():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                id, content, importance, source_session,
                memory_type, resolved, activation_count,
                pinned, valence, arousal, project, created_at, backfilled_at
            FROM memories
            ORDER BY id
        """)
        return [dict(r) for r in rows]


async def update_memory(
    memory_id: int,
    content: str = None,
    importance: int = None,
    memory_type: str = None,
    resolved: bool = None,
    pinned: bool = None,
    valence: float = None,
    arousal: float = None,
    project: str = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        fields = []
        values = []
        idx = 1

        if content is not None:
            fields.append(f"content = ${idx}")
            values.append(content)
            idx += 1

        if importance is not None:
            fields.append(f"importance = ${idx}")
            values.append(importance)
            idx += 1

        if memory_type is not None:
            fields.append(f"memory_type = ${idx}")
            values.append(memory_type)
            idx += 1

        if resolved is not None:
            fields.append(f"resolved = ${idx}")
            values.append(resolved)
            idx += 1

        if pinned is not None:
            fields.append(f"pinned = ${idx}")
            values.append(pinned)
            idx += 1

        if valence is not None:
            fields.append(f"valence = ${idx}")
            values.append(valence)
            idx += 1

        if arousal is not None:
            fields.append(f"arousal = ${idx}")
            values.append(arousal)
            idx += 1

        if project is not None:
            fields.append(f"project = ${idx}")
            values.append(project)
            idx += 1

        if not fields:
            return

        values.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(fields)} WHERE id = ${idx}"
        await conn.execute(sql, *values)


async def delete_memory(memory_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def delete_memories_batch(memory_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memories WHERE id = ANY($1::int[])",
            memory_ids
        )


# ============================================================
# 旧记忆回填
# ============================================================

async def get_memories_for_backfill(limit: int = 50, only_unclassified: bool = True):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if only_unclassified:
            rows = await conn.fetch(
                """
                SELECT
                    id, content, importance, source_session,
                    memory_type, resolved, activation_count,
                    pinned, valence, arousal, project, created_at, backfilled_at
                FROM memories
                WHERE backfilled_at IS NULL
                ORDER BY id
                LIMIT $1
                """,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    id, content, importance, source_session,
                    memory_type, resolved, activation_count,
                    pinned, valence, arousal, project, created_at, backfilled_at
                FROM memories
                ORDER BY id
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]


async def update_memory_structured(
    memory_id: int,
    importance: int,
    memory_type: str,
    resolved: bool,
    valence: float = None,
    arousal: float = None,
    project: str = None,
    mark_backfilled: bool = True,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if mark_backfilled:
            await conn.execute(
                """
                UPDATE memories
                SET
                    importance = $1,
                    memory_type = $2,
                    resolved = $3,
                    valence = $4,
                    arousal = $5,
                    project = $6,
                    backfilled_at = NOW()
                WHERE id = $7
                """,
                importance,
                memory_type,
                resolved,
                valence,
                arousal,
                project,
                memory_id,
            )
        else:
            await conn.execute(
                """
                UPDATE memories
                SET
                    importance = $1,
                    memory_type = $2,
                    resolved = $3,
                    valence = $4,
                    arousal = $5,
                    project = $6
                WHERE id = $7
                """,
                importance,
                memory_type,
                resolved,
                valence,
                arousal,
                project,
                memory_id,
            )
