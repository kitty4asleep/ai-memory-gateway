"""
记忆提取模块 —— 用 LLM 从对话或现有文本中提炼结构化记忆
======================================================

能力：
1. 从对话中提取新记忆
2. 对纯文本记忆进行自动评分
3. 对导入/旧记忆做完整结构化分类（importance + memory_type + resolved + valence + arousal + project）

v2.8 分类强化版：
- 支持导入时完整 AI 结构化
- 支持旧记忆批量重分类
- 显式使用 certifi
- 增强非200错误日志
- 增强文本清洗
- 强化 preference / relationship / rule / project / issue / todo 的分类边界
"""

import os
import json
import re
from typing import List, Dict

import httpx
import certifi


API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "anthropic/claude-haiku-4")


ALLOWED_MEMORY_TYPES = {
    "identity", "preference", "relationship", "rule",
    "project", "event", "issue", "todo", "emotion_event"
}


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
- 9-10：核心且长期稳定的信息
  例如：身份信息、重要关系、长期规则、明确禁忌、长期稳定偏好、长期项目主线
- 7-8：重要且未来高概率复用的信息
  例如：重要偏好、重大事件、持续性情感需求、重要项目进展、长期未解决问题
- 5-6：中等重要、未来可能会用到的信息
- 3-4：短期状态、一次性提及、低复用信息
- 1-2：琐碎、低价值信息

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

# memory_type 判定优先规则【很重要】
- 如果内容是在描述用户“喜欢什么 / 不喜欢什么 / 偏好什么 / 讨厌什么 / 倾向于什么 / 希望如何沟通”，优先判为 `preference`
- 如果内容是在描述双方“关系定位 / 称呼 / 互动模式 / 边界 / 主导-顺从动态”，优先判为 `relationship`
- 如果内容是在描述“长期规则 / 明确要求 / 禁忌 / 使用习惯 / 固定写作要求”，优先判为 `rule`
- 如果内容是在描述“长期项目 / 持续推进的任务 / 技术路线 / 阶段进展”，优先判为 `project`
- 如果内容是在描述“未解决的问题 / 卡住的点 / 仍待处理的故障”，优先判为 `issue`
- 如果内容是在描述“下一步要做什么 / 待办 / 约定后续动作”，优先判为 `todo`
- 如果内容是在描述“强情绪、强烈体验、关系中的情绪峰值事件”，优先判为 `emotion_event`
- 只有在内容主要是在描述“一次具体发生过的事情”，且不明显属于以上类型时，才判为 `event`

# 特别禁止
- 不要把稳定偏好判成 `event`
- 不要把长期规则判成 `event`
- 不要把关系设定判成 `event`
- 不要把长期项目主线判成 `event`

# resolved 规则
- 对 issue / todo / project，如果明显已经解决，可以设为 true
- 如果仍未解决或未来还要继续跟进，设为 false
- 其他类型默认 false 即可

# valence / arousal 规则
- valence：情绪效价，范围 -1 到 1
- arousal：情绪强度，范围 0 到 1
- 对纯规则、纯偏好、纯身份信息，如果没有明显情绪色彩，可用较保守值
  例如 valence=0, arousal=0.2

# project 字段
- 如果该记忆明显属于某个长期项目，填项目名
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
- 9-10：核心身份信息、重要关系、长期规则、明确禁忌、长期稳定偏好
- 7-8：重要偏好、重大事件、深层情感、重要项目进展、长期未解决问题
- 5-6：日常习惯、一般偏好、阶段性安排、近期状态
- 3-4：临时状态、偶然提及、低复用信息
- 1-2：琐碎信息、几乎不会影响未来对话的信息

# 输出格式
返回 JSON 数组，每条包含原文和评分：
[{{"content": "原文", "importance": 评分数字}}]

只返回 JSON，不要其他文字。

# 输入记忆
{memories_text}
"""


CLASSIFY_IMPORT_PROMPT = """你是长期记忆结构化整理专家。

我会给你一批已经存在的记忆文本。你的任务不是提取新记忆，而是对这些已有记忆逐条做结构化标注。

你必须为每条记忆判断以下字段：
- importance（1-10）
- memory_type
- resolved
- valence（-1 到 1）
- arousal（0 到 1）
- project（无则空字符串）

# memory_type 只能使用以下值之一
- identity
- preference
- relationship
- rule
- project
- event
- issue
- todo
- emotion_event

# 判断原则

## memory_type 判定优先规则【最重要】
- 如果内容是在描述用户“喜欢什么 / 不喜欢什么 / 偏好什么 / 厌恶什么 / 倾向于什么 / 希望如何沟通”，优先判为 `preference`
- 如果内容是在描述双方“关系定位 / 称呼 / 互动模式 / 边界 / 主导-顺从动态”，优先判为 `relationship`
- 如果内容是在描述“长期规则 / 明确要求 / 禁忌 / 使用习惯 / 固定写作要求”，优先判为 `rule`
- 如果内容是在描述“长期项目 / 持续推进的任务 / 技术路线 / 阶段进展”，优先判为 `project`
- 如果内容是在描述“未解决的问题 / 卡住的点 / 仍待处理的故障”，优先判为 `issue`
- 如果内容是在描述“下一步要做什么 / 待办 / 约定后续动作”，优先判为 `todo`
- 如果内容是在描述“强情绪、强烈体验、关系中的情绪峰值事件”，优先判为 `emotion_event`
- 只有在内容主要是在描述“一次具体发生过的事情”，且不明显属于以上类型时，才判为 `event`

## 特别禁止
- 不要把稳定偏好判成 `event`
- 不要把长期规则判成 `event`
- 不要把关系设定判成 `event`
- 不要把长期项目主线判成 `event`

## 补充例子
- “用户喜欢沉稳、有主见、能主导的沟通方式，不喜欢 emoji。” → `preference`
- “用户希望始终保留前端 system，不要被网关覆盖。” → `rule`
- “用户和AI之间存在稳定的亲密关系设定。” → `relationship`
- “用户最近在搭建和调试新的记忆库 memory-gateway。” → `project`
- “用户当前正在处理 Render 上游 SSL 和记忆提取 404 的问题。” → `issue`
- “用户下一步要继续调试记忆提取。” → `todo`

## importance
- 9-10：核心且长期稳定
- 7-8：重要且高复用
- 5-6：中等重要
- 3-4：短期/低复用
- 1-2：琐碎

## resolved
- 如果内容描述的是已完成、已解决、已过去且不再悬而未决的事项，可设 true
- 如果是仍在持续、待继续、未解决的问题或计划，设 false
- 拿不准时默认 false

## valence
- -1 到 1
- 负面更低，正面更高，中性为 0

## arousal
- 0 到 1
- 越强烈、越重要、越情绪化，越高

## project
- 如果明显属于长期项目，填项目名，例如：
  - memory-gateway
  - sillytavern
  - termux
- 否则填空字符串 ""

# 输出要求
- 输出 JSON 数组
- 数组长度必须与输入条目数量一致
- 每条都保留原始 content
- 不要输出解释
- 不要 markdown 代码块

输出格式示例：
[
  {{
    "content": "原文A",
    "importance": 8,
    "memory_type": "project",
    "resolved": false,
    "valence": 0.1,
    "arousal": 0.4,
    "project": "memory-gateway"
  }},
  {{
    "content": "原文B",
    "importance": 8,
    "memory_type": "preference",
    "resolved": false,
    "valence": 0.2,
    "arousal": 0.3,
    "project": ""
  }}
]

# 输入记忆
{memories_text}
"""


def _clean_memory_text(text: str) -> str:
    text = str(text).strip()

    # 去掉首尾空白后再剥外层引号
    text = text.strip().strip('"').strip("'").strip()

    # 去掉末尾多余逗号 / 中文逗号
    text = re.sub(r'[,\uFF0C]+\s*$', '', text)

    # 去掉再次包裹的引号
    text = text.strip().strip('"').strip("'").strip()

    # 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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


def _normalize_memory_item(mem: dict) -> dict | None:
    if not isinstance(mem, dict) or "content" not in mem:
        return None

    content = _clean_memory_text(mem.get("content", ""))
    if not content:
        return None

    try:
        importance = int(mem.get("importance", 5))
    except Exception:
        importance = 5
    importance = max(1, min(10, importance))

    memory_type = str(mem.get("memory_type", "event")).strip() or "event"
    if memory_type not in ALLOWED_MEMORY_TYPES:
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

    return {
        "content": content,
        "importance": importance,
        "memory_type": memory_type,
        "resolved": resolved,
        "valence": valence,
        "arousal": arousal,
        "project": project,
    }


async def _post_to_memory_model(messages: list, max_tokens: int = 1600):
    if not API_KEY:
        raise RuntimeError("API_KEY 未设置")

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
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": messages,
            },
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"status={response.status_code}, url={API_BASE_URL}, model={MEMORY_MODEL}, body={response.text[:1000]}"
        )

    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List[str] = None) -> List[Dict]:
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
        cleaned_existing = [_clean_memory_text(m) for m in existing_memories if str(m).strip()]
        memories_text = "\n".join(f"- {m}" for m in cleaned_existing)
    else:
        memories_text = "（暂无已知信息）"

    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

    print(
        f"🧠 开始记忆提取: model={MEMORY_MODEL}, url={API_BASE_URL}, "
        f"messages={len(messages)}, existing={len(existing_memories or [])}",
        flush=True
    )

    try:
        text = await _post_to_memory_model(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
            ],
            max_tokens=1800,
        )

        print(f"📝 记忆模型原始返回:\n{text[:1200]}", flush=True)

        memories = _extract_json_array(text)
        if not isinstance(memories, list):
            return []

        valid_memories = []
        for mem in memories:
            normalized = _normalize_memory_item(mem)
            if normalized:
                valid_memories.append(normalized)

        print(
            f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）",
            flush=True
        )
        return valid_memories

    except Exception as e:
        print(f"⚠️ 记忆提取出错: {type(e).__name__}: {e}", flush=True)
        return []


async def score_memories(texts: List[str]) -> List[Dict]:
    if not texts:
        return []

    cleaned_texts = [_clean_memory_text(t) for t in texts if str(t).strip()]
    memories_text = "\n".join(f"- {t}" for t in cleaned_texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    print(
        f"🧠 开始记忆评分: model={MEMORY_MODEL}, url={API_BASE_URL}, count={len(cleaned_texts)}",
        flush=True
    )

    try:
        text = await _post_to_memory_model(
            [{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        memories = _extract_json_array(text)
        if not isinstance(memories, list):
            return [{"content": t, "importance": 5} for t in cleaned_texts]

        valid = []
        for mem in memories:
            if isinstance(mem, dict) and "content" in mem:
                content = _clean_memory_text(mem["content"])
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
        return [{"content": t, "importance": 5} for t in cleaned_texts]


async def classify_memory_texts(texts: List[str]) -> List[Dict]:
    """
    对已有记忆文本做完整结构化分类。
    用于：
    - 导入时自动重判
    - 旧记忆重建
    """
    if not texts:
        return []

    cleaned_texts = [_clean_memory_text(t) for t in texts if str(t).strip()]
    cleaned_texts = [t for t in cleaned_texts if t]

    if not cleaned_texts:
        return []

    if not API_KEY:
        print("⚠️ API_KEY 未设置，无法进行结构化分类，退回默认值", flush=True)
        return [
            {
                "content": t,
                "importance": 5,
                "memory_type": "event",
                "resolved": False,
                "valence": 0.0,
                "arousal": 0.2,
                "project": None,
            }
            for t in cleaned_texts
        ]

    memories_text = "\n".join(f"- {t}" for t in cleaned_texts)

    print(
        f"🧠 开始结构化分类: model={MEMORY_MODEL}, url={API_BASE_URL}, count={len(cleaned_texts)}",
        flush=True
    )

    try:
        prompt = CLASSIFY_IMPORT_PROMPT.format(memories_text=memories_text)

        text = await _post_to_memory_model(
            [{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        print(f"📝 结构化分类原始返回:\n{text[:1600]}", flush=True)

        memories = _extract_json_array(text)
        if not isinstance(memories, list):
            raise ValueError("返回结果不是 JSON 数组")

        valid = []
        for mem in memories:
            normalized = _normalize_memory_item(mem)
            if normalized:
                valid.append(normalized)

        # 条数不一致时兜底补齐
        if len(valid) < len(cleaned_texts):
            existing_contents = {m["content"] for m in valid}
            for t in cleaned_texts:
                if t not in existing_contents:
                    valid.append({
                        "content": t,
                        "importance": 5,
                        "memory_type": "event",
                        "resolved": False,
                        "valence": 0.0,
                        "arousal": 0.2,
                        "project": None,
                    })

        print(f"📝 完成结构化分类 {len(valid)} 条", flush=True)
        return valid

    except Exception as e:
        print(f"⚠️ 结构化分类失败: {type(e).__name__}: {e}", flush=True)
        return [
            {
                "content": t,
                "importance": 5,
                "memory_type": "event",
                "resolved": False,
                "valence": 0.0,
                "arousal": 0.2,
                "project": None,
            }
            for t in cleaned_texts
        ]
