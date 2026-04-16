"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
修复版：
1) 工具链安全裁剪（避免 tool_call / tool output 断链）
2) 自动清理孤儿 tool message
3) debug 快照保留 tool 相关字段
4) 上游流式报错把 body 细节透给前端
"""
import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from collections import deque
from typing import Any, Dict, List, Set

import httpx
import certifi
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (
    init_tables,
    close_pool,
    save_message,
    search_memories,
    save_memory,
    get_all_memories_count,
    get_recent_memories,
    get_all_memories,
    get_pool,
    get_all_memories_detail,
    update_memory,
    delete_memory,
    delete_memories_batch,
)
from memory_extractor import extract_memories, score_memories, classify_memory_texts
from routing_config import PROVIDERS, MODEL_ROUTING, MODEL_ALIASES


def resolve_model_alias(raw_model: str):
    if raw_model in MODEL_ALIASES:
        raw_model = MODEL_ALIASES[raw_model]
    if raw_model in MODEL_ROUTING:
        prefix, real_model = MODEL_ROUTING[raw_model]
        return f"{prefix}/{real_model}"
    return raw_model


def resolve_provider(model_name: str):
    if "/" in model_name:
        prefix, real_model = model_name.split("/", 1)
    else:
        prefix, real_model = None, model_name

    if prefix and prefix in PROVIDERS:
        base_env, key_env = PROVIDERS[prefix]
        return {
            "base_url": os.environ.get(base_env, ""),
            "api_key": os.environ.get(key_env, ""),
            "model": real_model,
        }

    return {
        "base_url": os.environ.get("API_BASE_URL", ""),
        "api_key": os.environ.get("API_KEY", ""),
        "model": model_name,
    }


# =========================
# 消息裁剪（工具链安全版）
# =========================
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))


def _json_dump_safe(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _message_size(msg: Dict[str, Any]) -> int:
    parts: List[str] = []
    content = msg.get("content")
    if content is not None:
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(_json_dump_safe(content))

    if msg.get("tool_calls"):
        parts.append(_json_dump_safe(msg.get("tool_calls")))

    if msg.get("tool_call_id"):
        parts.append(str(msg.get("tool_call_id")))

    if msg.get("function_call"):
        parts.append(_json_dump_safe(msg.get("function_call")))

    return len("".join(parts))


def _assistant_tool_call_ids(msg: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for tc in (msg.get("tool_calls") or []):
        tcid = tc.get("id")
        if tcid:
            ids.add(str(tcid))
    return ids


def _drop_orphan_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除没有对应 assistant.tool_calls 的 tool message，避免上游 400:
    No tool call found for function call output with call_id ...
    """
    seen_tool_call_ids: Set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            seen_tool_call_ids |= _assistant_tool_call_ids(m)

    cleaned: List[Dict[str, Any]] = []
    dropped = 0
    for m in messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid and str(tcid) not in seen_tool_call_ids:
                dropped += 1
                continue
        cleaned.append(m)

    if dropped:
        print(f"⚠️ 清理孤儿 tool 消息 {dropped} 条", flush=True)
    return cleaned


def trim_messages_tool_aware(messages: List[Dict[str, Any]], limit: int = MAX_PROMPT_CHARS):
    """
    从尾部裁剪消息，优先保证 tool 链完整：
    - 若保留了 role=tool，尽量保留它对应的 assistant.tool_calls
    - 最后再做一次孤儿 tool 清理
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    kept_reversed: List[Dict[str, Any]] = []
    total = 0

    # 若尾部保留了 tool 输出，则它依赖的 tool_call id 必须保留
    required_tool_call_ids: Set[str] = set()

    for msg in reversed(other_msgs):
        role = msg.get("role")
        size = _message_size(msg)

        must_keep = False
        if role == "tool":
            tcid = msg.get("tool_call_id")
            if tcid:
                required_tool_call_ids.add(str(tcid))
                must_keep = True

        elif role == "assistant" and msg.get("tool_calls"):
            ids = _assistant_tool_call_ids(msg)
            if ids & required_tool_call_ids:
                must_keep = True
                required_tool_call_ids -= ids

        if not must_keep and total + size > limit:
            break

        kept_reversed.append(msg)
        total += size

    kept = list(reversed(kept_reversed))
    kept = _drop_orphan_tool_messages(kept)
    return system_msgs + kept


# =========================
# 环境变量
# =========================
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
PORT = int(os.getenv("PORT", "8080"))

MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))
_round_counter = 0

FORCE_STREAM = os.getenv("FORCE_STREAM", "false").lower() == "true"
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")

EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")
DEBUG_MAX_HISTORY = int(os.getenv("DEBUG_MAX_HISTORY", "20"))
DEBUG_MAX_CONTENT_LEN = int(os.getenv("DEBUG_MAX_CONTENT_LEN", "3000"))

BACKFILL_DEFAULT_LIMIT = int(os.getenv("BACKFILL_DEFAULT_LIMIT", "500"))
BACKFILL_DEFAULT_BATCH_SIZE = int(os.getenv("BACKFILL_DEFAULT_BATCH_SIZE", "5"))
BACKFILL_BATCH_SLEEP = float(os.getenv("BACKFILL_BATCH_SLEEP", "1.5"))


def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️ 未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️ 无人设，纯转发模式")

LATEST_DEBUG_SNAPSHOT = {
    "updated_at": None,
    "session_id": None,
    "model_requested": None,
    "model_upstream": None,
    "upstream_url": None,
    "is_stream": None,
    "headers": {},
    "messages": [],
}
DEBUG_HISTORY = deque(maxlen=DEBUG_MAX_HISTORY)

BACKFILL_STATE = {
    "running": False,
    "stop_requested": False,
    "started_at": None,
    "finished_at": None,
    "limit": 0,
    "batch_size": 0,
    "only_unclassified": True,
    "dry_run": False,
    "fetched": 0,
    "processed": 0,
    "updated": 0,
    "failed": 0,
    "current_batch": 0,
    "total_batches": 0,
    "last_error": None,
    "preview": [],
}
BACKFILL_TASK = None


def _mask_headers_for_debug(headers: dict):
    safe = dict(headers or {})
    if "Authorization" in safe:
        safe["Authorization"] = "Bearer ***"
    return safe


def _truncate_debug_content(value: Any):
    if isinstance(value, str) and len(value) > DEBUG_MAX_CONTENT_LEN:
        return value[:DEBUG_MAX_CONTENT_LEN] + "...[TRUNCATED]"
    if isinstance(value, list):
        out = []
        for x in value:
            out.append(_truncate_debug_content(x))
        return out
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k] = _truncate_debug_content(v)
        return out
    return value


def _sanitize_messages_for_debug(messages: list):
    sanitized = []
    for m in messages or []:
        item = {
            "role": m.get("role"),
            "name": m.get("name"),
            "content": m.get("content", ""),
            "tool_call_id": m.get("tool_call_id"),
            "tool_calls": m.get("tool_calls"),
            "function_call": m.get("function_call"),
        }
        item = _truncate_debug_content(item)
        sanitized.append(item)
    return sanitized


def _record_debug_snapshot(
    session_id: str,
    model_requested: str,
    model_upstream: str,
    upstream_url: str,
    is_stream: bool,
    headers: dict,
    final_messages: list,
):
    snap = {
        "updated_at": datetime.now().isoformat(),
        "session_id": session_id,
        "model_requested": model_requested,
        "model_upstream": model_upstream,
        "upstream_url": upstream_url,
        "is_stream": is_stream,
        "headers": _mask_headers_for_debug(headers),
        "messages": _sanitize_messages_for_debug(final_messages),
    }
    LATEST_DEBUG_SNAPSHOT.update(snap)
    DEBUG_HISTORY.appendleft(snap)


def _check_debug_auth(request: Request):
    token = request.headers.get("X-Debug-Token", "") or request.query_params.get("token", "")
    if not DEBUG_TOKEN or token != DEBUG_TOKEN:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return None


def _reset_backfill_state():
    BACKFILL_STATE.update(
        {
            "running": False,
            "stop_requested": False,
            "started_at": None,
            "finished_at": None,
            "limit": 0,
            "batch_size": 0,
            "only_unclassified": True,
            "dry_run": False,
            "fetched": 0,
            "processed": 0,
            "updated": 0,
            "failed": 0,
            "current_batch": 0,
            "total_batches": 0,
            "last_error": None,
            "preview": [],
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MEMORY_ENABLED:
        try:
            await init_tables()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        except Exception as e:
            print(f"⚠️ 数据库初始化失败: {e}")
            print("⚠️ 记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️ 记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")

    yield

    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="3.1.1-tool-safe", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def build_system_prompt_with_memories(user_message: str) -> str:
    if not MEMORY_ENABLED:
        return SYSTEM_PROMPT

    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        if not memories:
            return SYSTEM_PROMPT

        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem["created_at"])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except Exception:
                    date_str = f"[{str(mem['created_at'])[:10]}] "

            extra = []
            if mem.get("memory_type"):
                extra.append(f"type={mem['memory_type']}")
            if mem.get("project"):
                extra.append(f"project={mem['project']}")
            if mem.get("resolved") is True:
                extra.append("resolved=true")
            if mem.get("pinned") is True:
                extra.append("pinned=true")
            meta = f" ({', '.join(extra)})" if extra else ""

            memory_lines.append(f"- {date_str}{mem['content']}{meta}")

        memory_text = "\n".join(memory_lines)
        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息保持一致性；冲突以新信息为准
- 模糊记忆可表达不确定性
- 如果是未解决问题或持续中的项目，优先维持上下文连续性

记忆是丰富对话的工具，而非对话焦点。
"""
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt

    except Exception as e:
        print(f"⚠️ 记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT


async def process_memories_background(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    model: str,
    context_messages: list = None,
):
    global _round_counter
    try:
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)

        if MEMORY_EXTRACT_INTERVAL == 0:
            print("⏭️ 记忆自动提取已禁用，跳过")
            return

        _round_counter += 1
        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            print(f"⏭️ 轮次 {_round_counter}，跳过记忆提取（每 {MEMORY_EXTRACT_INTERVAL} 轮）")
            return

        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]

        if context_messages:
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [{"role": "assistant", "content": assistant_msg}]
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)

        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取", "记忆遗漏", "尚未被记录",
            "写入不完整", "检索功能", "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署", "debug", "端口", "网关",
        ]

        filtered = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            filtered.append(mem)

        for mem in filtered:
            await save_memory(
                content=mem["content"],
                importance=mem.get("importance", 5),
                source_session=session_id,
                memory_type=mem.get("memory_type", "event"),
                resolved=mem.get("resolved", False),
                valence=mem.get("valence"),
                arousal=mem.get("arousal"),
                project=mem.get("project"),
            )

        if filtered:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered)} 条新记忆，总计 {total} 条")

    except Exception as e:
        print(f"⚠️ 后台记忆处理失败: {type(e).__name__}: {e}")


async def run_backfill_job(limit: int, batch_size: int, only_unclassified: bool, dry_run: bool):
    global BACKFILL_TASK
    try:
        from database import get_memories_for_backfill, update_memory_structured

        _reset_backfill_state()
        BACKFILL_STATE["running"] = True
        BACKFILL_STATE["started_at"] = datetime.now().isoformat()
        BACKFILL_STATE["limit"] = limit
        BACKFILL_STATE["batch_size"] = batch_size
        BACKFILL_STATE["only_unclassified"] = only_unclassified
        BACKFILL_STATE["dry_run"] = dry_run

        memories = await get_memories_for_backfill(limit=limit, only_unclassified=only_unclassified)
        BACKFILL_STATE["fetched"] = len(memories)

        if not memories:
            BACKFILL_STATE["finished_at"] = datetime.now().isoformat()
            BACKFILL_STATE["running"] = False
            return

        total_batches = (len(memories) + batch_size - 1) // batch_size
        BACKFILL_STATE["total_batches"] = total_batches

        for start in range(0, len(memories), batch_size):
            if BACKFILL_STATE["stop_requested"]:
                print("🛑 backfill 收到停止请求", flush=True)
                break

            batch = memories[start:start + batch_size]
            texts = [m["content"] for m in batch]
            BACKFILL_STATE["current_batch"] = start // batch_size + 1
            print(f"🔄 backfill 批次 {BACKFILL_STATE['current_batch']}/{total_batches} size={len(batch)}", flush=True)

            try:
                classified = await classify_memory_texts(texts)

                for idx, mem in enumerate(batch):
                    if BACKFILL_STATE["stop_requested"]:
                        break

                    item = classified[idx] if idx < len(classified) else None
                    original_content = str(mem["content"]).strip()

                    if not item:
                        BACKFILL_STATE["failed"] += 1
                        if len(BACKFILL_STATE["preview"]) < 200:
                            BACKFILL_STATE["preview"].append({
                                "id": mem["id"],
                                "content": original_content[:120],
                                "status": "skipped_no_match",
                            })
                        continue

                    row = {
                        "id": mem["id"],
                        "content": original_content,
                        "importance": item.get("importance", 5),
                        "memory_type": item.get("memory_type", "event"),
                        "resolved": item.get("resolved", False),
                        "valence": item.get("valence"),
                        "arousal": item.get("arousal"),
                        "project": item.get("project"),
                    }

                    if len(BACKFILL_STATE["preview"]) < 200:
                        BACKFILL_STATE["preview"].append({
                            "id": row["id"],
                            "content": row["content"][:120],
                            "importance": row["importance"],
                            "memory_type": row["memory_type"],
                            "resolved": row["resolved"],
                            "valence": row["valence"],
                            "arousal": row["arousal"],
                            "project": row["project"],
                            "status": "dry_run" if dry_run else "updated_pending",
                        })

                    if not dry_run:
                        await update_memory_structured(
                            memory_id=row["id"],
                            importance=row["importance"],
                            memory_type=row["memory_type"],
                            resolved=row["resolved"],
                            valence=row["valence"],
                            arousal=row["arousal"],
                            project=row["project"],
                            mark_backfilled=True,
                        )
                        BACKFILL_STATE["updated"] += 1

                    BACKFILL_STATE["processed"] += 1

            except Exception as e:
                BACKFILL_STATE["failed"] += len(batch)
                BACKFILL_STATE["last_error"] = f"{type(e).__name__}: {e}"
                print(f"⚠️ backfill 批次失败: {type(e).__name__}: {e}", flush=True)

            await asyncio.sleep(BACKFILL_BATCH_SLEEP)

    except Exception as e:
        BACKFILL_STATE["last_error"] = f"{type(e).__name__}: {e}"
        print(f"⚠️ backfill 总任务失败: {type(e).__name__}: {e}", flush=True)

    finally:
        BACKFILL_STATE["running"] = False
        BACKFILL_STATE["finished_at"] = datetime.now().isoformat()
        BACKFILL_TASK = None


@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except Exception:
            pass

    return {
        "status": "running",
        "gateway": "AI Memory Gateway v3.1.1-tool-safe",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "system_prompt_length": len(SYSTEM_PROMPT),
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": MEMORY_EXTRACT_INTERVAL,
        "debug_mode": DEBUG_MODE,
        "backfill_running": BACKFILL_STATE["running"],
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ai-memory-gateway",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break

    original_messages = [msg for msg in messages if msg.get("role") != "system"]

    enhanced = ""
    if MEMORY_ENABLED and user_message:
        enhanced = await build_system_prompt_with_memories(user_message)
    elif SYSTEM_PROMPT:
        enhanced = SYSTEM_PROMPT

    if enhanced and enhanced.strip():
        messages.insert(0, {"role": "system", "content": enhanced})

    # 关键修复：工具链安全裁剪 + 清理孤儿 tool message
    body["messages"] = trim_messages_tool_aware(messages, MAX_PROMPT_CHARS)

    model = body.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
    model = resolve_model_alias(model)
    body["model"] = model

    session_id = str(uuid.uuid4())[:8]
    provider = resolve_provider(model)
    upstream_url = provider["base_url"]
    upstream_key = provider["api_key"]
    upstream_model = provider["model"]

    if not upstream_url:
        return JSONResponse(status_code=500, content={"error": f"上游 BASE_URL 未设置：model={model}"})
    if not upstream_key:
        return JSONResponse(
            status_code=500,
            content={"error": f"上游 API Key 未设置：model={model}, upstream={upstream_url}"},
        )

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }

    if "openrouter" in upstream_url:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE

    is_stream = body.get("stream", False)
    if FORCE_STREAM and not is_stream:
        is_stream = True
        body["stream"] = True
        print("⚡ 强制开启流式传输（FORCE_STREAM=true）")

    if REASONING_EFFORT:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = REASONING_EFFORT
        print(f"🧠 注入推理参数: reasoning_effort={REASONING_EFFORT}")

    print(
        f"📡 请求: model={model}, upstream_model={upstream_model}, "
        f"stream={is_stream}, memory={'on' if MEMORY_ENABLED else 'off'}, url={upstream_url}",
        flush=True,
    )

    body["model"] = upstream_model

    if DEBUG_MODE:
        _record_debug_snapshot(
            session_id=session_id,
            model_requested=model,
            model_upstream=upstream_model,
            upstream_url=upstream_url,
            is_stream=is_stream,
            headers=headers,
            final_messages=body.get("messages", []),
        )

    if is_stream:
        return StreamingResponse(
            stream_and_capture(
                headers,
                body,
                session_id,
                user_message,
                model,
                original_messages,
                upstream_url,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # 非流式
    try:
        async with httpx.AsyncClient(timeout=300, verify=certifi.where()) as client:
            response = await client.post(upstream_url, headers=headers, json=body)
            print(
                f"📨 非流式上游响应: status={response.status_code}, url={upstream_url}, model={upstream_model}",
                flush=True,
            )

            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                try:
                    assistant_msg = resp_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    pass

                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(
                        process_memories_background(
                            session_id, user_message, assistant_msg, model, context_messages=original_messages
                        )
                    )

                return JSONResponse(status_code=200, content=resp_data)

            print(f"⚠️ 非流式上游失败: status={response.status_code}, body={response.text[:1000]}", flush=True)
            try:
                err = response.json()
            except Exception:
                err = {"error": response.text}
            return JSONResponse(status_code=response.status_code, content=err)

    except Exception as e:
        print(f"⚠️ 非流式上游请求异常: {type(e).__name__}: {e}, url={upstream_url}", flush=True)
        return JSONResponse(status_code=502, content={"error": f"上游连接失败: {type(e).__name__}: {e}"})


async def stream_and_capture(
    headers: dict,
    body: dict,
    session_id: str,
    user_message: str,
    model: str,
    original_messages: list,
    upstream_url: str,
):
    full_response = []
    line_buffer = ""

    try:
        async with httpx.AsyncClient(timeout=300, verify=certifi.where()) as client:
            async with client.stream("POST", upstream_url, headers=headers, json=body) as response:
                upstream_ct = response.headers.get("content-type", "")
                print(
                    f"📨 上游响应: status={response.status_code}, content-type={upstream_ct}, url={upstream_url}",
                    flush=True,
                )

                if response.status_code >= 400:
                    error_bytes = await response.aread()
                    preview = error_bytes[:2000].decode("utf-8", errors="ignore")
                    print(
                        f"⚠️ 上游流式请求失败: status={response.status_code}, body={preview}",
                        flush=True,
                    )
                    yield (
                        f"data: {json.dumps({'error': {'message': preview or f'upstream error {response.status_code}', 'type': 'upstream_error'}}, ensure_ascii=False)}\n\n"
                    ).encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    return

                async for chunk in response.aiter_bytes():
                    yield chunk

                    text = chunk.decode("utf-8", errors="ignore")
                    line_buffer += text

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response.append(content)
                            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                pass

    except Exception as e:
        print(f"⚠️ 流式上游请求异常: {type(e).__name__}: {e}, url={upstream_url}", flush=True)
        yield (
            f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_connection_error'}}, ensure_ascii=False)}\n\n"
        ).encode("utf-8")
        yield b"data: [DONE]\n\n"
        return

    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(
            process_memories_background(
                session_id, user_message, assistant_msg, model, context_messages=original_messages
            )
        )


@app.get("/debug/last-request")
async def debug_last_request(request: Request):
    auth_err = _check_debug_auth(request)
    if auth_err:
        return auth_err
    return {"status": "ok", "snapshot": LATEST_DEBUG_SNAPSHOT}


@app.get("/debug/recent-requests")
async def debug_recent_requests(request: Request):
    auth_err = _check_debug_auth(request)
    if auth_err:
        return auth_err
    return {"status": "ok", "count": len(DEBUG_HISTORY), "items": list(DEBUG_HISTORY)}


@app.get("/import/seed-memories")
async def import_seed_memories():
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        memories = await get_all_memories()
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        return {"total": len(memories), "exported_at": str(datetime.now()), "memories": memories}
    except Exception as e:
        return {"error": str(e)}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not MEMORY_ENABLED:
        return HTMLResponse("记忆系统未启用（设置 MEMORY_ENABLED=true 开启）")
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/memories")
async def api_get_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    memories = await get_all_memories_detail()
    tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))

    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")

    return {"memories": memories}


@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory(
        memory_id,
        content=data.get("content"),
        importance=data.get("importance"),
        memory_type=data.get("memory_type"),
        resolved=data.get("resolved"),
        pinned=data.get("pinned"),
        valence=data.get("valence"),
        arousal=data.get("arousal"),
        project=data.get("project"),
    )
    return {"status": "ok", "id": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}


@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}

    for item in updates:
        await update_memory(
            item["id"],
            content=item.get("content"),
            importance=item.get("importance"),
            memory_type=item.get("memory_type"),
            resolved=item.get("resolved"),
            pinned=item.get("pinned"),
            valence=item.get("valence"),
            arousal=item.get("arousal"),
            project=item.get("project"),
        )

    return {"status": "ok", "updated": len(updates)}


@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


@app.post("/import/text")
async def import_text_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)
        full_classify = data.get("full_classify", True)

        if not lines:
            return {"error": "没有找到记忆条目"}

        if full_classify:
            classified = await classify_memory_texts(lines)
        elif skip_scoring:
            classified = [
                {
                    "content": t,
                    "importance": 5,
                    "memory_type": "event",
                    "resolved": False,
                    "valence": 0.0,
                    "arousal": 0.2,
                    "project": None,
                }
                for t in lines
            ]
        else:
            scored = await score_memories(lines)
            classified = [
                {
                    "content": mem["content"],
                    "importance": mem.get("importance", 5),
                    "memory_type": "event",
                    "resolved": False,
                    "valence": 0.0,
                    "arousal": 0.2,
                    "project": None,
                }
                for mem in scored
            ]

        imported = 0
        skipped = 0
        for mem in classified:
            content = mem.get("content", "")
            if not content:
                continue

            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
                if existing > 0:
                    skipped += 1
                    continue

            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session="text-import",
                memory_type=mem.get("memory_type", "event"),
                resolved=mem.get("resolved", False),
                valence=mem.get("valence"),
                arousal=mem.get("arousal"),
                project=mem.get("project"),
            )
            imported += 1

        total = await get_all_memories_count()
        return {
            "status": "done",
            "mode": "full_classify" if full_classify else ("skip_scoring" if skip_scoring else "score_only"),
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/import/memories")
async def import_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}

    try:
        data = await request.json()
        memories = data.get("memories", [])
        reclassify = data.get("reclassify", True)

        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}

        raw_texts = []
        raw_items = []
        for mem in memories:
            content = str(mem.get("content", "")).strip()
            if not content:
                continue
            raw_texts.append(content)
            raw_items.append(mem)

        if not raw_texts:
            return {"error": "导入数据里没有有效 content"}

        if reclassify:
            final_memories = await classify_memory_texts(raw_texts)
        else:
            final_memories = []
            for mem in raw_items:
                final_memories.append(
                    {
                        "content": str(mem.get("content", "")).strip(),
                        "importance": int(mem.get("importance", 5)),
                        "memory_type": mem.get("memory_type", "event"),
                        "resolved": bool(mem.get("resolved", False)),
                        "valence": mem.get("valence"),
                        "arousal": mem.get("arousal"),
                        "project": mem.get("project"),
                    }
                )

        imported = 0
        skipped = 0
        for mem in final_memories:
            content = mem.get("content", "")
            if not content:
                continue

            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
                if existing > 0:
                    skipped += 1
                    continue

            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session="json-import",
                memory_type=mem.get("memory_type", "event"),
                resolved=mem.get("resolved", False),
                valence=mem.get("valence"),
                arousal=mem.get("arousal"),
                project=mem.get("project"),
            )
            imported += 1

        total = await get_all_memories_count()
        return {
            "status": "done",
            "mode": "reclassify" if reclassify else "raw_restore",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/backfill/start")
async def admin_backfill_start(
    request: Request,
    limit: int = BACKFILL_DEFAULT_LIMIT,
    batch_size: int = BACKFILL_DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    all_memories: bool = False,
):
    global BACKFILL_TASK

    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}

    auth_err = _check_debug_auth(request)
    if auth_err:
        return auth_err

    if BACKFILL_STATE["running"]:
        return {"status": "already_running", "message": "已有 backfill 任务在运行", "state": BACKFILL_STATE}

    only_unclassified = not all_memories
    BACKFILL_TASK = asyncio.create_task(
        run_backfill_job(
            limit=max(1, limit),
            batch_size=max(1, batch_size),
            only_unclassified=only_unclassified,
            dry_run=dry_run,
        )
    )

    return {
        "status": "started",
        "message": "backfill 已在后台启动",
        "limit": limit,
        "batch_size": batch_size,
        "dry_run": dry_run,
        "only_unclassified": only_unclassified,
    }


@app.get("/admin/backfill/status")
async def admin_backfill_status(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    auth_err = _check_debug_auth(request)
    if auth_err:
        return auth_err
    return {"status": "ok", "state": BACKFILL_STATE}


@app.get("/admin/backfill/stop")
async def admin_backfill_stop(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    auth_err = _check_debug_auth(request)
    if auth_err:
        return auth_err

    if not BACKFILL_STATE["running"]:
        return {"status": "idle", "message": "当前没有运行中的 backfill 任务"}

    BACKFILL_STATE["stop_requested"] = True
    return {"status": "stopping", "message": "已请求停止 backfill，当前批次结束后会停下"}


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"📝 人设长度：{len(SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{DEFAULT_MODEL}")
    print(f"🔗 API 地址：{API_BASE_URL}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    print(
        "🔄 记忆提取间隔："
        + (
            "禁用"
            if MEMORY_EXTRACT_INTERVAL == 0
            else "每轮提取"
            if MEMORY_EXTRACT_INTERVAL == 1
            else f"每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次"
        )
    )
    if FORCE_STREAM:
        print("⚡ 强制流式传输：开启")
    if REASONING_EFFORT:
        print(f"🧠 推理参数注入：{REASONING_EFFORT}")
    if DEBUG_MODE:
        print(f"🛠️ DEBUG_MODE：开启，历史条数上限={DEBUG_MAX_HISTORY}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
