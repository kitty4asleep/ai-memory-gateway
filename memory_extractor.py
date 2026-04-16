"""
记忆提取模块 —— 用 LLM 从对话中提炼关键记忆
=============================================

每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得长期记住的信息，存到数据库里。

v2.6 融合版：
1. 提取时注入已有记忆，让模型对比后只提取全新信息
2. 强化 importance 评分规则，减少分数漂移
3. 修复 .format() 与 JSON 花括号冲突的问题
4. 支持结构化记忆字段：memory_type / resolved / valence / arousal / project
5. 增强 404/非200 排错日志
6. 显式使用 certifi 证书包
"""

import os
import json
import re
from typing import List, Dict

import httpx
import certifi


API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 用来提取记忆的模型（便宜的就行）
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "anthropic/claude-haiku-4")


EXTRACTION_PROMPT = """你是信息提取专家，负责从对话中识别并提取值得长期记住的关键信息。

你的目标不是记录所有内容，而是只保留未来对话中真正有复用价值的信息。

# 提取重点
- 关键信息：提取用户的重要信息和值得回忆的生活细节
- 重要事件：记忆深刻的互动，尽量包含人物、时间、地点（如有）
- 长期线索：未来可能持续影响对话质量、关系连续性、个性化回应的信息

# 提取范围
- 个人：年龄、生日、职业、学历、居住地
- 偏好：明确表达的喜好或厌恶
- 健康：身体状况、过敏史、饮食禁忌
- 事件：与AI的重要互动、约定、里程碑
- 关系：家人、朋友、重要同事
- 价值观：表达的信念或长期目标
- 情感：重要的情感时刻或关系里程碑
- 生活：用户当天的活动、饮食、出行、日常经历等生活细节
- 项目：用户长期推进的项目、持续性的计划、重要技术路线与阶段进展
- 规则：用户明确提出的长期沟通偏好、写作要求、禁忌、使用习惯
- 问题：尚未解决、未来大概率还会继续影响对话的持续性问题或卡点
- 待办：明确约定的下一步行动、后续计划、继续事项

# 不要提取
- 日常寒暄（例如“你好”“在吗”）
- AI助手自己的回复内容
- 关于记忆系统本身的讨论（例如“某条记忆没有被记录”“记忆遗漏”“没有被提取”等）
- 纯技术调试、bug修复的过程性讨论（除非它反映了用户的长期技能、项目里程碑、固定工作流或长期使用习惯）
- AI的思考过程、思维链内容
- 纯一次性、低价值、未来几乎不会再用到的碎片信息
- 仅对当前一轮回复有用、但对后续关系和长期理解没有价值的信息

# 已知信息处理【最重要】
<已知信息>
{existing_memories}

- 新信息必须与已知信息逐条比对
- 相同、相似或语义重复的信息必须忽略
- 已知信息的补充或更新可以提取
- 与已知信息矛盾的新信息可以提取，并视为更新信息
- 仅提取完全新增且不与已知信息重复的内容
- 如果对话中没有任何新信息，返回空数组 []

# importance 评分规则（1-10）
请严格按照以下标准打分：

- 9-10：核心且长期稳定的信息
  例如：身份信息、年龄、生日、重要关系、长期目标、明确禁忌、长期沟通规则、核心项目主线、持续性关系定位

- 7-8：重要且未来高概率复用的信息
  例如：重要偏好、重大事件、持续性的情感需求、重要项目进展、关系中的关键约定、稳定使用习惯、长期未解决问题

- 5-6：中等重要、未来可能会用到的信息
  例如：一般生活习惯、普通偏好、阶段性安排、近期状态、常规日常信息、阶段性问题

- 3-4：短期状态、一次性提及、低复用信息
  例如：临时情绪、偶然提及的小事、短期计划、轻量生活碎片

- 1-2：琐碎、低价值、几乎不影响未来对话的信息
  这类信息通常不应提取；除非有特殊意义，否则宁可不返回

# memory_type 取值规则
只能使用以下枚举值之一：
- identity
- preference
- relationship
- rule
- project
- event
- issue
- todo
- emotion_event

说明：
- identity：稳定身份事实
- preference：长期偏好
- relationship：关系动态、称呼、互动定位
- rule：长期沟通规则、禁忌、使用规则
- project：持续中的项目、长期计划
- event：普通事件
- issue：未解决问题、长期卡点
- todo：明确下一步、待办、后续行动
- emotion_event：高情绪强度事件

# resolved 规则
- 对 issue / todo / project，如果明显已经解决，可以设为 true
- 如果仍未解决或未来还要继续跟进，设为 false
- 其他类型默认 false 即可

# valence / arousal 规则
- valence：情绪效价，范围 -1 到 1
  - 负向：-1 接近强烈负面
  - 中性：0
  - 正向：1 接近强烈正面

- arousal：情绪强度，范围 0 到 1
  - 0 接近平静、低波动
  - 1 接近强烈、激烈、难忘

如果不是明显情绪性记忆，可给较保守数值，例如：
- valence = 0
- arousal = 0.2

# project 字段
- 如果该记忆明显属于某个长期项目，填项目名，例如：
  "memory-gateway"
  "sillytavern"
  "termux"
- 否则填空字符串 ""

# 输出格式
请用以下 JSON 格式返回（不要包含其他内容）：

[
  {{
    "content": "记忆内容",
    "importance": 8,
    "memory_type": "event",
    "resolved": false,
    "valence": 0.0,
    "arousal": 0.3,
    "project": ""
  }}
]

要求：
- 只返回 JSON 数组
- 不要返回解释
- 不要使用 markdown 代码块
- 如果没有值得记住的新信息，返回空数组：[]
"""


SCORING_PROMPT = """你是记忆重要性评分专家。请对以下记忆条目逐条评分。

# 评分规则（1-10）
- 9-10：核心身份信息（名字、生日、职业、重要关系、长期规则、明确禁忌）
- 7-8：重要偏好、重大事件、深层情感、重要项目进展、稳定使用习惯、长期未解决问题
- 5-6：日常习惯、一般偏好、阶段性安排、近期状态
- 3-4：临时状态、偶然提及、低复用信息
- 1-2：琐碎信息、几乎不会影响未来对话的信息

# 评分原则
优先考虑长期价值、未来复用价值、关系重要性、一致性需求。

# 输入记忆
{memories_text}

# 输出格式
返回 JSON 数组，每条包含原文和评分：
[{{"content": "原文", "importance": 评分数字}}]

只返回 JSON，不要其他文字。
"""


def _extract_json_array(text: str):
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List[str] = None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        existing_memories: 已有记忆内容列表，用于去重对比

    返回：
        结构化记忆列表
    """
    if not API_KEY:
        print("⚠️ API_KEY 未设置，跳过记忆提取", flush=True)
        return []

    if not messages:
        return []

    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            conversation_text += f"用户: {content}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content}\n"

    if not conversation_text.strip():
        return []

    if existing_memories:
        memories_text = "\n".join(f"- {m}" for m in existing_memories)
    else:
        memories_text = "（暂无已知信息）"

    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

    print(
        f"🧠 开始记忆提取: model={MEMORY_MODEL}, url={API_BASE_URL}, "
        f"messages={len(messages)}, existing={len(existing_memories or [])}",
        flush=True
    )

    try:
        async with httpx.AsyncClient(timeout=60, verify=certifi.where()) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://midsummer-gateway.local",
                    "X-Title": "Midsummer Memory Extraction",
                },
                json={
                    "model": MEMORY_MODEL,
                    "max_tokens": 1600,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
                    ],
                },
            )

            if response.status_code != 200:
                print(
                    f"⚠️ 记忆提取请求失败: status={response.status_code}, "
                    f"url={API_BASE_URL}, model={MEMORY_MODEL}, "
                    f"body={response.text[:1000]}",
                    flush=True
                )
                return []

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            print(f"📝 记忆模型原始返回:\n{text[:1000]}", flush=True)

            try:
                memories = _extract_json_array(text)
            except Exception as e:
                print(f"⚠️ 记忆提取结果解析失败: {e}", flush=True)
                return []

            if not isinstance(memories, list):
                return []

            valid_memories = []
            allowed_types = {
                "identity", "preference", "relationship", "rule",
                "project", "event", "issue", "todo", "emotion_event"
            }

            for mem in memories:
                if not isinstance(mem, dict) or "content" not in mem:
                    continue

                content = str(mem["content"]).strip()
                if not content:
                    continue

                try:
                    importance = int(mem.get("importance", 5))
                except Exception:
                    importance = 5
                importance = max(1, min(10, importance))

                memory_type = str(mem.get("memory_type", "event")).strip() or "event"
                if memory_type not in allowed_types:
                    memory_type = "event"

                resolved = bool(mem.get("resolved", False))

                try:
                    valence = float(mem.get("valence")) if mem.get("valence") is not None else None
                    if valence is not None:
                        valence = max(-1.0, min(1.0, valence))
                except Exception:
                    valence = None

                try:
                    arousal = float(mem.get("arousal")) if mem.get("arousal") is not None else None
                    if arousal is not None:
                        arousal = max(0.0, min(1.0, arousal))
                except Exception:
                    arousal = None

                project = str(mem.get("project", "")).strip() or None

                valid_memories.append({
                    "content": content,
                    "importance": importance,
                    "memory_type": memory_type,
                    "resolved": resolved,
                    "valence": valence,
                    "arousal": arousal,
                    "project": project,
                })

            print(
                f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）",
                flush=True
            )
            return valid_memories

    except Exception as e:
        print(f"⚠️ 记忆提取出错: {type(e).__name__}: {e}", flush=True)
        return []


async def score_memories(texts: List[str]) -> List[Dict]:
    """对纯文本记忆条目批量评分"""
    if not texts:
        return []

    memories_text = "\n".join(f"- {t}" for t in texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    print(
        f"🧠 开始记忆评分: model={MEMORY_MODEL}, url={API_BASE_URL}, count={len(texts)}",
        flush=True
    )

    try:
        async with httpx.AsyncClient(timeout=60, verify=certifi.where()) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MEMORY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 4000,
                },
            )

            if response.status_code != 200:
                print(
                    f"⚠️ 记忆评分请求失败: status={response.status_code}, "
                    f"url={API_BASE_URL}, model={MEMORY_MODEL}, "
                    f"body={response.text[:1000]}",
                    flush=True
                )
                return [{"content": t, "importance": 5} for t in texts]

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            try:
                memories = _extract_json_array(text)
            except Exception:
                return [{"content": t, "importance": 5} for t in texts]

            if not isinstance(memories, list):
                return [{"content": t, "importance": 5} for t in texts]

            valid = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    content = str(mem["content"]).strip()
                    if not content:
                        continue

                    try:
                        importance = int(mem.get("importance", 5))
                    except Exception:
                        importance = 5
                    importance = max(1, min(10, importance))

                    valid.append({
                        "content": content,
                        "importance": importance,
                    })

            print(f"📝 为 {len(valid)} 条记忆完成自动评分", flush=True)
            return valid

    except Exception as e:
        print(f"⚠️ 记忆评分出错: {type(e).__name__}: {e}", flush=True)
        return [{"content": t, "importance": 5} for t in texts]
