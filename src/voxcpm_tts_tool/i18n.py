"""Minimal in-app translation dictionary with zh→key fallback.

Keys are stable string IDs (e.g. "tab.generation"); per spec §UI Design,
Chinese must cover every key; English may have gaps.
"""
from __future__ import annotations

Lang = str  # "zh" or "en"

STRINGS: dict[str, dict[Lang, str]] = {
    # tabs
    "tab.generation": {"zh": "语音生成", "en": "Speech Generation"},
    "tab.voice_library": {"zh": "音色管理", "en": "Voice Library"},
    "tab.usage": {"zh": "使用说明", "en": "Usage"},
    # mode labels
    "mode.design": {"zh": "声音设计", "en": "Voice Design"},
    "mode.clone": {"zh": "可控克隆", "en": "Controllable Cloning"},
    "mode.hifi": {"zh": "极致克隆", "en": "Ultimate Cloning"},
    # field labels
    "field.voice_name": {"zh": "音色名称", "en": "Voice Name"},
    "field.control": {"zh": "风格描述", "en": "Control Instruction"},
    "field.reference_audio": {"zh": "参考音频", "en": "Reference Audio"},
    "field.prompt_text": {"zh": "转写文本", "en": "Transcript"},
    "field.denoise": {"zh": "启用降噪", "en": "Enable Denoise"},
    "field.default_voice": {"zh": "默认音色", "en": "Default Voice"},
    "field.script": {"zh": "脚本文本", "en": "Script Text"},
    # buttons
    "btn.transcribe": {"zh": "识别转写", "en": "Transcribe"},
    "btn.generate": {"zh": "开始生成", "en": "Generate"},
    "btn.preview": {"zh": "生成预览", "en": "Generate Preview"},
    "btn.save": {"zh": "保存", "en": "Save"},
    "btn.delete": {"zh": "删除", "en": "Delete"},
    "btn.refresh": {"zh": "刷新", "en": "Refresh"},
    "btn.insert_voice": {"zh": "插入音色标签", "en": "Insert Voice Tag"},
    "btn.insert_tag": {"zh": "插入非语言标签", "en": "Insert Tag"},
    # status / messages
    "status.asr_unavailable": {
        "zh": "ASR 模型不可用，识别按钮已禁用",
        "en": "ASR model unavailable; Transcribe button disabled",
    },
    "status.denoise_unavailable": {
        "zh": "ZipEnhancer 不可用，降噪开关无效",
        "en": "ZipEnhancer unavailable; denoise toggle has no effect",
    },
    "err.empty_script": {"zh": "脚本为空", "en": "Script is empty"},
    "err.missing_reference": {
        "zh": "找不到参考音频文件",
        "en": "Reference audio file is missing",
    },
    "err.unsupported_audio": {
        "zh": "仅支持 .wav 文件",
        "en": "Only .wav files are supported",
    },
    "err.duplicate_name": {"zh": "音色名重复", "en": "Voice name already exists"},
    "err.invalid_voice_name": {
        "zh": "音色名不能包含 < 或 >",
        "en": "Voice name must not contain < or >",
    },
    # default voice display name
    "voice.default": {"zh": "默认", "en": "Default"},
    # advanced generation knobs
    "field.advanced": {"zh": "高级设置", "en": "Advanced settings"},
    "field.cfg_value": {"zh": "CFG（引导强度）", "en": "CFG (guidance scale)"},
    "field.normalize": {"zh": "文本规范化", "en": "Text normalization"},
    "field.seed_text": {
        "zh": "样本朗读文本（用于固化音色）",
        "en": "Sample text (used to seed and fix the voice)",
    },
    "field.preview_audio": {"zh": "音色预览", "en": "Voice preview"},
    "field.inference_timesteps": {
        "zh": "推理步数",
        "en": "Inference timesteps",
    },
}


def t(key: str, lang: Lang) -> str:
    """Return localized string. Fallback order: lang → zh → key."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    if lang in entry:
        return entry[lang]
    if "zh" in entry:
        return entry["zh"]
    return key
