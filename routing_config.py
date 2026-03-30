# routing_config.py

# 站子前缀 → (BASE_URL 环境变量名, API_KEY 环境变量名)
PROVIDERS = {
    "zhenhaoji": ("ZHENHAOJI_BASE_URL", "ZHENHAOJI_API_KEY"),
    "sunlea": ("SUNLEA_BASE_URL", "SUNLEA_API_KEY"),
    "fuka": ("FUKA_BASE_URL", "FUKA_API_KEY"),
    "qmbabyy": ("QMBABYY_BASE_URL", "QMBABYY_API_KEY"),
    "newop": ("NEWOP_BASE_URL", "NEWOP_API_KEY"),  # 例子：你的新站子
    # 想加新站子就在这里加："前缀": ("ENV_BASE", "ENV_KEY"),
}

# 真实模型名 → (站子前缀, 真实模型名)
MODEL_ROUTING = {
    "claude-opus-4-6-thinking": ("zhenhaoji", "claude-opus-4-6-thinking"),
    "claude-sonnet-4-5-20250929": ("zhenhaoji", "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4.5[假流式]": ("sunlea", "claude-sonnet-4.5[假流式]"),
    "[车厘子]claude-4.6-opus-thinking④": ("fuka", "[车厘子]claude-4.6-opus-thinking④"),
    "claude-sonnet-4-6": ("qmbabyy", "claude-sonnet-4-6"),
    # 想加模型就在这里加，格式同上
}

# 别名表：给同名模型/好记短名用
MODEL_ALIASES = {
    "sonnet-main": "zhenhaoji/claude-sonnet-4-6",
    "sonnet-backup": "qmbabyy/claude-sonnet-4-6",
    "opus-main": "zhenhaoji/claude-opus-4-6-thinking",
    "opus-new":  "newop/claude-opus-4-6-thinking",
    # 想加别名就在这里加："短名": "前缀/真实模型名"
}
