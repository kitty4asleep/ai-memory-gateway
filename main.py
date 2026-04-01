"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
"""
import os
import json
import uuid
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (
    init_tables, close_pool, save_message, search_memories, save_memory,
    get_all_memories_count, get_recent_memories, get_all_memories,
    get_pool, get_all_memories_detail, update_memory, delete_memory,
    delete_memories_batch
)
from memory_extractor import extract_memories, score_memories

# 路由配置（以后只改 routing_config.py）
from routing_config import PROVIDERS, MODEL_ROUTING, MODEL_ALIASES

# ==== 工具：别名解析 / 上游路由 ====
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
            "base_url": os.environ[base_env],
            "api_key": os.environ[key_env],
            "model": real_model,
        }
    # 默认上游
    return {
        "base_url": os.environ["API_BASE_URL"],
        "api_key": os.environ["API_KEY"],
        "model": model_name,
    }

# ==== 工具：上下文截断防巨长账单 ====
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))
def trim_messages_by_chars(messages, limit=MAX_PROMPT_CHARS):
    """从后往前保留消息，直到字符数不超过 limit"""
    total = 0
    trimmed = []
    for msg in reversed(messages):
        content = msg.get("content", "")
        text = content if isinstance(content, str) else str(content)
        total += len(text)
        if total > limit:
            break
        trimmed.append(msg)
    return list(reversed(trimmed))

# ============================================================
# 配置项
# ============================================================
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

app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)
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
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")
        memory_text = "\n".join(memory_lines)
        enhanced_prompt = f"""{SYSTEM_PROMPT}
【从过往对话中检索到的相关记忆】
{memory_text}
# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息保持一致性；冲突以新信息为准
- 模糊记忆可表达不确定性
记忆是丰富对话的工具，而非对话焦点。"""
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt
    except Exception as e:
        print(f"⚠️ 记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT

async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str, context_messages: list = None):
    global _round_counter
    try:
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)
        if MEMORY_EXTRACT_INTERVAL == 0:
            print(f"⏭️ 记忆自动提取已禁用，跳过")
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
            "阈值", "数据库", "seed", "导入", "部署", "bug", "debug", "端口", "网关",
        ]
        filtered = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            filtered.append(mem)
        for mem in filtered:
            await save_memory(content=mem["content"], importance=mem["importance"], source_session=session_id)
        if filtered:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered)} 条新记忆，总计 {total} 条")
    except Exception as e:
        print(f"⚠️ 后台记忆处理失败: {e}")

@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v2.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "system_prompt_length": len(SYSTEM_PROMPT),
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": MEMORY_EXTRACT_INTERVAL,
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
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY 未设置，请在环境变量中配置"})
    body = await request.json()
    messages = body.get("messages", [])

    # 提取用户最新消息
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

    # 构建 system prompt
    original_messages = [msg for msg in messages if msg.get("role") != "system"]
    if SYSTEM_PROMPT or (MEMORY_ENABLED and user_message):
        enhanced = await build_system_prompt_with_memories(user_message) if (MEMORY_ENABLED and user_message) else SYSTEM_PROMPT
        if enhanced:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = enhanced + "\n\n" + msg["content"]
                        break
            else:
                messages.insert(0, {"role": "system", "content": enhanced})
    body["messages"] = messages

    # 截断上下文，防止超长账单
    body["messages"] = trim_messages_by_chars(body["messages"], MAX_PROMPT_CHARS)

    # 模型解析
    model = body.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
    model = resolve_model_alias(model)
    body["model"] = model

    session_id = str(uuid.uuid4())[:8]

    # 上游路由
    provider = resolve_provider(model)
    upstream_url = provider["base_url"]
    upstream_key = provider["api_key"]
    upstream_model = provider["model"]

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
        print(f"⚡ 强制开启流式传输（FORCE_STREAM=true）")

    if REASONING_EFFORT:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = REASONING_EFFORT
        print(f"🧠 注入推理参数: reasoning_effort={REASONING_EFFORT}")

    print(f"📡 请求: model={model}, upstream_model={upstream_model}, stream={is_stream}, memory={'on' if MEMORY_ENABLED else 'off'}", flush=True)
    body["model"] = upstream_model

    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages, upstream_url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(upstream_url, headers=headers, json=body)
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                try:
                    assistant_msg = resp_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(
                        process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages)
                    )
                return JSONResponse(status_code=200, content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())

async def stream_and_capture(headers: dict, body: dict, session_id: str, user_message: str, model: str, original_messages: list, upstream_url: str):
    full_response = []
    line_buffer = ""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", upstream_url, headers=headers, json=body) as response:
            upstream_ct = response.headers.get("content-type", "")
            print(f"📨 上游响应: status={response.status_code}, content-type={upstream_ct}", flush=True)
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
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(
            process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages)
        )

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
        return {
            "total": len(memories),
            "exported_at": str(__import__("datetime").datetime.now()),
            "memories": memories,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
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
    await update_memory(memory_id, content=data.get("content"), importance=data.get("importance"))
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
        await update_memory(item["id"], content=item.get("content"), importance=item.get("importance"))
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
        if not lines:
            return {"error": "没有找到记忆条目"}
        if skip_scoring:
            scored = [{"content": t, "importance": 5} for t in lines]
        else:
            scored = await score_memories(lines)
        imported = 0
        skipped = 0
        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
                if existing > 0:
                    skipped += 1
                    continue
            await save_memory(content=content, importance=mem.get("importance", 5), source_session="text-import")
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}

@app.post("/import/memories")
async def import_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    try:
        data = await request.json()
        memories = data.get("memories", [])
        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}
        imported = 0
        skipped = 0
        for mem in memories:
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
                source_session=mem.get("source_session", "json-import"),
            )
            imported += 1
        total = await get_all_memories_count()
        return {"status": "done", "imported": imported, "skipped": skipped, "total": total}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"📝 人设长度：{len(SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{DEFAULT_MODEL}")
    print(f"🔗 API 地址：{API_BASE_URL}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    print(f"🔄 记忆提取间隔：{'禁用' if MEMORY_EXTRACT_INTERVAL == 0 else '每轮提取' if MEMORY_EXTRACT_INTERVAL == 1 else f'每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次'}")
    if FORCE_STREAM:
        print(f"⚡ 强制流式传输：开启")
    if REASONING_EFFORT:
        print(f"🧠 推理参数注入：{REASONING_EFFORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
