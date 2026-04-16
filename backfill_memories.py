"""
backfill_memories.py

用途：
- 对数据库里已有的旧记忆做结构化回填
- 不删库、不重导
- 自动补齐：
  importance / memory_type / resolved / valence / arousal / project

用法示例：
python backfill_memories.py
python backfill_memories.py --limit 20
python backfill_memories.py --batch-size 10
python backfill_memories.py --all
python backfill_memories.py --dry-run
"""

import os
import sys
import json
import asyncio
import argparse

from database import (
    get_memories_for_backfill,
    update_memory_structured,
    close_pool,
)
from memory_extractor import classify_memory_texts


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill structured fields for old memories")
    parser.add_argument("--limit", type=int, default=50, help="最多处理多少条记忆")
    parser.add_argument("--batch-size", type=int, default=10, help="每批送给模型多少条")
    parser.add_argument("--all", action="store_true", help="忽略 only_unclassified，直接处理前 limit 条")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写回数据库")
    return parser.parse_args()


async def main():
    args = parse_args()

    limit = max(1, args.limit)
    batch_size = max(1, args.batch_size)
    only_unclassified = not args.all
    dry_run = args.dry_run

    print(
        f"🧠 开始旧记忆回填：limit={limit}, batch_size={batch_size}, "
        f"only_unclassified={only_unclassified}, dry_run={dry_run}",
        flush=True
    )

    memories = await get_memories_for_backfill(limit=limit, only_unclassified=only_unclassified)

    if not memories:
        print("✅ 没有需要回填的旧记忆", flush=True)
        await close_pool()
        return

    print(f"📦 共取到 {len(memories)} 条待处理记忆", flush=True)

    total_updated = 0

    for start in range(0, len(memories), batch_size):
        batch = memories[start:start + batch_size]
        texts = [m["content"] for m in batch]

        print(
            f"\n🔄 处理批次 {start // batch_size + 1}："
            f"{start + 1}-{start + len(batch)} / {len(memories)}",
            flush=True
        )

        try:
            classified = await classify_memory_texts(texts)

            # 建一个 content -> result 的映射
            result_map = {}
            for item in classified:
                content = item.get("content", "").strip()
                if content:
                    result_map[content] = item

            for mem in batch:
                original_content = mem["content"].strip()

                # 优先精确匹配；找不到就用兜底默认值
                item = result_map.get(original_content)

                if not item:
                    print(f"⚠️ 未匹配到分类结果，跳过 id={mem['id']} content={original_content[:80]}", flush=True)
                    continue

                structured = {
                    "id": mem["id"],
                    "content": original_content,
                    "importance": item.get("importance", 5),
                    "memory_type": item.get("memory_type", "event"),
                    "resolved": item.get("resolved", False),
                    "valence": item.get("valence"),
                    "arousal": item.get("arousal"),
                    "project": item.get("project"),
                }

                print(
                    f"📝 id={structured['id']} | "
                    f"type={structured['memory_type']} | "
                    f"importance={structured['importance']} | "
                    f"resolved={structured['resolved']} | "
                    f"project={structured['project']} | "
                    f"{structured['content'][:100]}",
                    flush=True
                )

                if not dry_run:
                    await update_memory_structured(
                        memory_id=structured["id"],
                        importance=structured["importance"],
                        memory_type=structured["memory_type"],
                        resolved=structured["resolved"],
                        valence=structured["valence"],
                        arousal=structured["arousal"],
                        project=structured["project"],
                    )
                    total_updated += 1

        except Exception as e:
            print(f"⚠️ 这一批处理失败: {type(e).__name__}: {e}", flush=True)

    print(
        f"\n✅ 回填完成："
        f"{'dry-run 未写入数据库' if dry_run else f'共更新 {total_updated} 条'}",
        flush=True
    )

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
