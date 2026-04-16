"""
记忆提取模块 —— 用 LLM 从对话或现有文本中提炼结构化记忆
======================================================

能力：
1. 从对话中提取新记忆
2. 对纯文本记忆进行自动评分
3. 对导入/旧记忆做完整结构化分类（importance + memory_type + resolved + valence + arousal + project）

v2.9 长期锚点优化版：
- 强化“长期价值优先”
- 严格减少低价值碎片入库
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


EXTRACTION_PROMPT = """
你是长期记忆筛选专家，负责从对话中识别真正值得长期保留的信息。

你的目标不是记录聊天内容本身，而是只保留未来对话中高复用、高价值、能明显提升连续性和理解深度的锚点记忆。

# 你的核心原则
宁可少提取，也不要把低价值碎片写进长期记忆库。
如果一条信息未来几乎不会再次用到，或者只对当下这一轮互动有意义，就不要提取。

# 优先提取的内容（高价值）
1. 身份信息
- 年龄、教育背景、职业、长期身份特征
- 稳定的人格倾向、长期自我认知

2. 长期偏好
- 明确、稳定、未来会反复影响对话质量的喜好或厌恶
- 沟通风格偏好、写作偏好、禁忌、雷点
- 审美偏好、文学/作品偏好、长期兴趣方向

3. 关系锚点
- 对关系连续性有长期意义的信息
- 关系中的核心需求、重要不安、重要确认点
- 对亲密关系中“被怎样对待”的稳定期待
- 长期有效的称呼、角色定位、互动模式

4. 长期规则
- 用户明确提出的长期要求、习惯、禁忌、工作流
- 稳定的表达规则、格式要求、行为边界

5. 项目与长期事务
- 用户在持续推进的项目
- 项目当前阶段、技术路线、关键进展、固定工作习惯
- 会在未来继续影响对话的卡点、路线选择、下一阶段方向

6. 未解决问题 / 待办
- 明显还会继续影响未来对话的问题
- 仍待处理的故障、卡点、长期困扰
- 明确的下一步计划、待继续事项

7. 重要情绪议题
- 长期反复出现、会持续影响关系和回应方式的情绪主题
- 重大失落、重大依恋、重大创伤式节点
- 不是一时情绪，而是持续性议题

8. 高层总结型记忆
- 如果多轮对话反复表现出同一种模式，优先提炼成高层概括
- 例如：与其记某次撒娇时说了什么，不如记“用户在感到被冷处理时会通过撒娇和肢体黏人来确认爱意”

# 明确不要提取的内容（低价值）
1. 一次性动作碎片
- 某次亲了一下、扑了一下、摸了一下、看了一眼、唱了一句、说了一句俏皮话
- 某次具体撒娇动作、具体表情、具体语气词
- 某次单独的暧昧桥段，如果它不构成长期模式

2. 重复的亲密碎片
- 大量相似的撒娇、害羞、嘴硬、亲亲、扑过去、讨抱抱场景
- 如果这些内容只是同一模式的不同变体，不要逐条记，最多提炼成一条长期模式

3. 纯过程性调情或情色细节
- 只服务于当次互动氛围的细节
- 低复用的具体动作、微场景、单句挑逗
- 没有长期价值的局部桥段

4. 一次性短期状态
- 某天困、某次脚没力气、某次被喇叭吵醒、某次想再躺一会儿
- 除非它反映长期健康状态、稳定习惯或反复出现的问题

5. 关于记忆系统本身的讨论
- 某条记忆有没有记住
- 导入、数据库、提取失败、阈值等元讨论
- 单纯的调试过程，除非它反映长期项目主线或固定工作流

6. AI自己的内容
- AI的思考过程、思维链、解释性废话
- AI一时的安慰或调情内容

# 提取策略（非常重要）
- 优先抽象，不要琐碎
- 优先总结模式，不要记录单次桥段
- 优先长期锚点，不要一次性情境
- 如果多条内容本质上在表达同一个稳定偏好或模式，只提炼成一条更高层的记忆

# 已知信息处理（最重要）
<已知信息>
{existing_memories}

- 新信息必须与已知信息逐条比对
- 相同、相似或语义重复的信息必须忽略
- 已知信息的补充或更新可以提取
- 与已知信息矛盾的新信息可以提取，并视为更新
- 如果只是旧模式的又一次重复出现，忽略
- 如果只是旧记忆的低层细节展开，忽略
- 如果对话中没有任何真正新的高价值信息，返回空数组 []

# importance 评分规则（1-10）
- 9-10：核心且长期稳定，未来高度关键
  例如：身份核心、关系核心、长期规则、关键项目主线、关键创伤或依恋议题
- 7-8：重要且高复用，未来大概率会反复用到
  例如：重要偏好、稳定模式、重要项目进展、长期问题、关键关系需求
- 5-6：中等重要，未来可能用到，但不是核心
  例如：一般偏好、阶段性安排、次级项目状态
- 3-4：低复用、短期、边缘价值
  这类通常不建议提取
- 1-2：琐碎、低价值、几乎不应进入长期记忆
  基本不要返回

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

# memory_type 判定优先规则
- 稳定喜好、厌恶、沟通偏好 -> preference
- 关系需求、关系定位、长期互动模式 -> relationship
- 长期要求、规则、禁忌、固定使用习惯 -> rule
- 长期项目、持续任务、路线进展 -> project
- 未解决问题、卡点、故障、悬而未决事项 -> issue
- 下一步计划、待继续动作 -> todo
- 持续性重大情绪议题、强烈情绪节点 -> emotion_event
- 只有在确实是值得保留的一次重大事件，且不属于以上类型时，才用 event

# 特别禁止
- 不要把稳定偏好判成 event
- 不要把关系模式判成 event
- 不要把长期规则判成 event
- 不要把项目主线判成 event
- 不要把低价值亲密碎片强行塞成 event

# resolved 规则
- issue / todo / project 中，如果明显已经解决，可设为 true
- 如果仍在持续，设为 false
- 其他类型默认 false

# valence / arousal 规则
- valence：情绪效价，范围 -1 到 1
- arousal：情绪强度，范围 0 到 1
- 如果内容主要是规则、偏好、身份信息，没有强情绪，可用保守值（如 valence=0, arousal=0.2）

# project 字段
- 如果明显属于某个长期项目，填项目名
- 否则填空字符串 ""

# 输出格式
请用以下 JSON 格式返回（不要包含其他内容）：

[
  {{
    "content": "提炼后的长期记忆内容",
    "importance": 8,
    "memory_type": "preference",
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
- 如果没有值得长期保留的新信息，返回空数组：[]
"""


SCORING_PROMPT = """
你是记忆重要性评分专家。请对以下记忆条目逐条评分。

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


CLASSIFY_IMPORT_PROMPT = """
你是长期记忆结构化整理专家。

我会给你一批已经存在的记忆文本。你的任务不是提取新记忆，而是判断：这些文本到底值不值得作为长期记忆存在，以及它们属于什么类型。

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

# 核心原则
不要被文本表面迷惑。你要判断的是这条信息是不是长期锚点，而不是这句话是不是出现过。

如果一条记忆只是：
- 一次性的动作
- 单次暧昧场景
- 低复用的互动碎片
- 单独一句撒娇或调情
那么它的长期价值通常很低，应给予较低 importance，并尽量不要夸大为重要事件。

# memory_type 判定优先规则（最重要）
- 稳定喜好、厌恶、审美、沟通偏好、长期兴趣 -> preference
- 关系定位、情感需求、被怎样对待的长期期待、核心关系模式 -> relationship
- 长期规则、明确要求、禁忌、固定工作流、写作要求 -> rule
- 长期项目、持续推进事务、技术路线、项目进展 -> project
- 未解决问题、卡点、故障、长期困扰 -> issue
- 下一步计划、待办、要继续做的动作 -> todo
- 持续性重大情绪议题、高情绪强度节点 -> emotion_event
- 只有在确实是重大且值得长期记住的一次事件时，才用 event

# 特别禁止
- 不要把稳定偏好判成 event
- 不要把关系模式判成 event
- 不要把长期规则判成 event
- 不要把项目主线判成 event
- 不要把普通亲密碎片判成高价值 event

# 重要性判断
## 9-10
核心锚点，未来非常关键
例如：
- 身份核心
- 关系核心需求
- 长期规则
- 项目主线
- 长期重大情绪议题

## 7-8
高复用、高价值
例如：
- 重要偏好
- 重要关系模式
- 重要项目进展
- 重要未解决问题

## 5-6
中等价值
例如：
- 一般偏好
- 一般习惯
- 阶段性状态
- 有一点复用价值的信息

## 3-4
低复用、短期、碎片化
例如：
- 一次性小事
- 小片段互动
- 边缘信息
这类通常不应作为优质长期记忆

## 1-2
琐碎、低价值、垃圾碎片
例如：
- 单句暧昧桥段
- 低复用动作
- 一次性表情或玩笑
- 对未来几乎无帮助的信息

# resolved
- 如果内容描述的是已经结束、不再影响未来的问题，可设 true
- 如果仍持续存在，设 false
- 拿不准时默认 false

# valence
- -1 到 1
- 负面更低，正面更高，中性为 0

# arousal
- 0 到 1
- 越强烈、越难忘、越持续影响关系，越高
- 普通规则、偏好、身份信息可给较低值

# project
- 如果明显属于长期项目，填项目名，例如：
  - memory-gateway
  - sillytavern
  - termux
- 否则填空字符串 ""

# 判断倾向
如果一条内容可以被理解为稳定模式而不是单次事件，优先按稳定模式分类。
例如：
- 用户喜欢沉稳、有主见、能主导的沟通方式 -> preference
- 用户很在意被哄、把哄和爱直接联系在一起 -> relationship
- 用户要求某种固定写法 -> rule

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
    "memory_type": "preference",
    "resolved": false,
    "valence": 0.1,
    "arousal": 0.3,
    "project": ""
  }},
  {{
    "content": "原文B",
    "importance": 3,
    "memory_type": "event",
    "resolved": false,
    "valence": 0.0,
    "arousal": 0.2,
    "project": ""
  }}
]

# 输入记忆
{memories_text}
"""


def _clean_memory_text(text: str) -> str:
    text = str(text).strip()
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r'[,\uFF0C]+\s*$', '', text)
    text = text.strip().strip('"').strip("'").strip()
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
