"""
记忆提取模块
—— 用 LLM 从对话中提炼关键记忆
=============================================

每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得长期记住的信息，存到数据库里。

v2.4 改进：
1. 提取时注入已有记忆，让模型对比后只提取全新信息
2. 强化 importance 评分规则，减少分数漂移
3. 更强调长期价值、未来复用价值、关系重要性
"""

import os
import json
import httpx
from typing import List, Dict


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
  例如：重要偏好、重大事件、持续性的情感需求、重要项目进展、关系中的关键约定、稳定使用习惯

- 5-6：中等重要、未来可能会用到的信息  
  例如：一般生活习惯、普通偏好、阶段性安排、近期状态、常规日常信息

- 3-4：短期状态、一次性提及、低复用信息  
  例如：临时情绪、偶然提及的小事、短期计划、轻量生活碎片

- 1-2：琐碎、低价值、几乎不影响未来对话的信息  
  这类信息通常不应提取；除非有特殊意义，否则宁可不返回

# 评分原则
评分时优先考虑以下维度：
1. 长期价值：这条信息在未来是否仍然重要
2. 复用价值：这条信息是否会明显提升后续对话质量
3. 关系重要性：这条信息是否影响用户关系体验、沟通方式、情感连续性
4. 一致性需求：这条信息是否需要长期保持稳定，避免前后矛盾
5. 稀缺性：这条信息是否难以从别处重新获得

# 输出格式
请用以下 JSON 格式返回（不要包含其他内容）：
[
  {"content": "记忆内容", "importance": 分数},
  {"content": "记忆内容", "importance": 分数}
]

要求：
- 只返回 JSON 数组
- 不要返回解释
- 不要使用 markdown 代码块
- 如果没有值得记住的新信息，返回空数组：[]
"""


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List[str] = None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        existing_memories: 已有记忆内容列表，用于去重对比

    返回：
        记忆列表，格式 [{"content": "...", "importance": N}, ...]
    """
    if not API_KEY:
        print("⚠️ API_KEY 未设置，跳过记忆提取")
        return []

    if not messages:
        return []

    # 把对话格式化成文本
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

    # 格式化已有记忆
    if existing_memories:
        memories_text = "\n".join(f"- {m}" for m in existing_memories)
    else:
        memories_text = "（暂无已知信息）"

    # 把已有记忆填入 prompt
    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

    # 调用 LLM 提取记忆
    try:
        async with httpx.AsyncClient(timeout=60) as client:
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
                    "max_tokens": 1200,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
                    ],
                },
            )

            if response.status_code != 200:
                print(f"⚠️ 记忆提取请求失败: {response.status_code}")
                return []

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 打印模型原始返回（截断防刷屏）
            print(f"📝 记忆模型原始返回:\n{text[:500]}", flush=True)

            # 清理可能的 markdown 格式
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # 强力 JSON 提取：如果上面清理后仍然解析失败，用正则兜底
            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                        print("📝 JSON正则兜底提取成功")
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 记忆提取结果解析失败: {e}")
                        return []
                else:
                    print("⚠️ 记忆提取结果中未找到JSON数组")
                    return []

            if not isinstance(memories, list):
                return []

            # 验证格式 + importance 夹紧
            valid_memories = []
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

                    valid_memories.append({
                        "content": content,
                        "importance": importance,
                    })

            print(f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）")
            return valid_memories

    except json.JSONDecodeError as e:
        print(f"⚠️ 记忆提取结果解析失败: {e}")
        return []
    except Exception as e:
        print(f"⚠️ 记忆提取出错: {e}")
        return []


SCORING_PROMPT = """你是记忆重要性评分专家。请对以下记忆条目逐条评分。

# 评分规则（1-10）
- 9-10：核心身份信息（名字、生日、职业、重要关系、长期规则、明确禁忌）
- 7-8：重要偏好、重大事件、深层情感、重要项目进展、稳定使用习惯
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


async def score_memories(texts: List[str]) -> List[Dict]:
    """对纯文本记忆条目批量评分"""
    if not texts:
        return []

    memories_text = "\n".join(f"- {t}" for t in texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
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
                print(f"⚠️ 记忆评分请求失败: {response.status_code}")
                # 失败时返回默认分数
                return [{"content": t, "importance": 5} for t in texts]

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                    except json.JSONDecodeError:
                        return [{"content": t, "importance": 5} for t in texts]
                else:
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

            print(f"📝 为 {len(valid)} 条记忆完成自动评分")
            return valid

    except Exception as e:
        print(f"⚠️ 记忆评分出错: {e}")
        return [{"content": t, "importance": 5} for t in texts]
