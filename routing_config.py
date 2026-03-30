# routing_config.py
# 真实模型名 → (站子前缀, 真实模型名)
MODEL_ROUTING = {
    "claude-opus-4-6-thinking": ("zhenhaoji", "claude-opus-4-6-thinking"),
    "claude-sonnet-4-5-20250929": ("zhenhaoji", "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4.5[假流式]": ("sunlea", "claude-sonnet-4.5[假流式]"),
    "[车厘子]claude-4.6-opus-thinking④": ("fuka", "[车厘子]claude-4.6-opus-thinking④"),
    "claude-sonnet-4-6": ("qmbabyy", "claude-sonnet-4-6"),
}

# 别名表：给同名模型起两个代称，便于共存
MODEL_ALIASES = {
    "sonnet-main": "zhenhaoji/claude-sonnet-4-6",   # 主用
    "sonnet-backup": "qmbabyy/claude-sonnet-4-6",   # 备用
    # 想加别名就按这个格式往下加："你的短名": "前缀/真实模型名"
}
