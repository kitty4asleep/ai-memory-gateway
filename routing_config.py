# routing_config.py

# 站子前缀 → (BASE_URL 环境变量名, API_KEY 环境变量名)
PROVIDERS = {
    "zhenhaoji": ("ZHENHAOJI_BASE_URL", "ZHENHAOJI_API_KEY"),  # https://api.zhenhaoji.qzz.io/v1/chat/completions
    "sunlea":    ("SUNLEA_BASE_URL",    "SUNLEA_API_KEY"),     # https://sunlea.de/v1/chat/completions
    "fuka":      ("FUKA_BASE_URL",      "FUKA_API_KEY"),       # https://api.fuka.win/v1/chat/completions
    "kongbeiqie":("KBQ_BASE_URL",       "KBQ_API_KEY"),        # https://api.空悲切.cn/v1/chat/completions
    "ciallo":    ("CIALLO_BASE_URL",    "CIALLO_API_KEY"),     # https://ioll.pp.ua/v1/chat/completions

   "ling":    ("LING_BASE_URL",    "LING_API_KEY"),     # https://api.ldsx.asia/v1/chat/completions
   "tree":    ("TREE_BASE_URL",    "TREE_API_KEY"),     # https://api.treegpt.top/v1/chat/completions
    "run":       ("RUN_BASE_URL",       "RUN_API_KEY"),        # https://runanytime.hxi.me/v1/chat/completions
    "zyra":      ("ZYRA_BASE_URL",      "ZYRA_API_KEY"),       # https://zyraonline.org/v1/chat/completions
    "ciwei":     ("CIWEI_BASE_URL",     "CIWEI_API_KEY"),      # https://cf-cc.cwapi.vip/v1/chat/completions
    "sakura":    ("SAKURA_BASE_URL",    "SAKURA_API_KEY"),     # https://codex.sakurapy.de/v1/chat/completions
    "pond":      ("POND_BASE_URL",      "POND_API_KEY"),       # https://code.claudex.us.ci/v1/chat/completions
    "ggboom":    ("GGBOOM_BASE_URL",    "GGBOOM_API_KEY"),     # https://ai.qaq.al/v1/chat/completions
    "cups":      ("CUPS_BASE_URL",      "CUPS_API_KEY"),       # https://free-llm.cups.moe/v1/chat/completions
    "yizi":      ("YIZI_BASE_URL",      "YIZI_API_KEY"),       # https://api.cetaceang.qzz.io/v1/chat/completions
    "paolu":     ("PAOLU_BASE_URL",     "PAOLU_API_KEY"),      # https://api.sillytaverns.com/v1/chat/completions
    "heabl":     ("HEABL_BASE_URL",     "HEABL_API_KEY"),      # https://api.heabl.top/v1/chat/completions
    "ice":       ("ICE_BASE_URL",       "ICE_API_KEY"),        # https://ice.v.ua/v1/chat/completions
    "ekan":      ("EKAN_BASE_URL",      "EKAN_API_KEY"),       # https://api.ekan8.com/v1/chat/completions
    "wong":      ("WONG_BASE_URL",      "WONG_API_KEY"),       # https://wzw.pp.ua/v1/chat/completions

    "qmbabyy":   ("QMBABYY_BASE_URL",   "QMBABYY_API_KEY"),    # https://qmbabyy.cn/v1/chat/completions
    "qwqtao":    ("QWQTAO_BASE_URL",    "QWQTAO_API_KEY"),     # https://newapi.qwqtao.one/v1/chat/completions
    "elysiver":  ("ELYSIVER_BASE_URL",  "ELYSIVER_API_KEY"),   # https://elysiver.h-e.top/v1/chat/completions
    "hotaru":    ("HOTARU_BASE_URL",    "HOTARU_API_KEY"),     # https://hotaruapi.com/v1/chat/completions

    # 新站子继续往下加："prefix": ("PREFIX_BASE_URL", "PREFIX_API_KEY"),
}

# 可选：无前缀时默认路由（常用老模型）
MODEL_ROUTING = {
    "claude-opus-4-6-thinking": ("pond", "claude-opus-4-6-thinking"),
    "claude-sonnet-4-5-20250929": ("zhenhaoji", "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4.5[假流式]": ("sunlea", "claude-sonnet-4.5[假流式]"),
    "[车厘子]claude-4.6-opus-thinking④": ("fuka", "[车厘子]claude-4.6-opus-thinking④"),
    "claude-sonnet-4-6": ("qmbabyy", "claude-sonnet-4-6"),
}

# 别名表：好记短名 / 同名共存
MODEL_ALIASES = {
    "sonnet-main":   "zhenhaoji/claude-sonnet-4-6",
    "sonnet-backup": "qmbabyy/claude-sonnet-4-6",
    "opus-main":     "zhenhaoji/claude-opus-4-6-thinking",
    # 需要短名再加："short": "prefix/real-model"
}
