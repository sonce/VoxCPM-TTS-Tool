"""Line-scoped script parser for `<voice name>(control)?` switches.

See spec §Script Semantics for line-splitting rules, voice-name matching,
and unknown-tag handling.

Syntax:
    <voice>text                  -> hifi-style at generation (no control)
    <voice>(control)text         -> clone-style at generation (control = parens content)

Each parsed segment carries `voice_name`, optional `control`, and the text run
that follows. The same voice can appear multiple times in a line; each switch
starts a new segment with its own (possibly empty/None) control.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Match `<voice>` optionally followed by `(control)` immediately (no whitespace
# between `>` and `(`). The control body excludes `)` so it stays one-line.
_SWITCH_RE = re.compile(r"<([^<>\n\r]+)>(?:\(([^)\n\r]*)\))?")


# UI label (zh) → SDK token. The dropdown shows the keys; the script gets the
# key inserted as `[zh_label]`; `localize_non_lang_tags` rewrites those to
# `[en_token]` immediately before the script is parsed for generation. English
# tokens already in the script are passed through unchanged.
NON_LANG_TAG_MAP_ZH: dict[str, str] = {
    "笑声": "laughing",
    "叹息": "sigh",
    "嗯": "Uhm",
    "嘘": "Shh",
    "疑问-啊": "Question-ah",
    "疑问-诶": "Question-ei",
    "疑问-嗯": "Question-en",
    "疑问-哦": "Question-oh",
    "惊讶-哇": "Surprise-wa",
    "惊讶-哟": "Surprise-yo",
    "不满-哼": "Dissatisfaction-hnn",
}


def localize_non_lang_tags(script: str, mapping: dict[str, str] | None = None) -> str:
    """Replace `[zh_label]` occurrences with `[en_token]` for known mappings.

    Idempotent: if the script already contains `[en_token]`, it is left alone.
    Tags not in the mapping (whether localized or unknown) are passed through
    verbatim — the model receives whatever the user wrote.

    Longer keys are substituted first so that overlapping prefixes (e.g.
    `[疑问-嗯]` vs `[嗯]`) don't collide. With the default mapping no key is
    a substring of another inside the brackets, but ordering by length keeps
    callers safe if they extend the mapping later.
    """
    m = NON_LANG_TAG_MAP_ZH if mapping is None else mapping
    for zh_label in sorted(m, key=len, reverse=True):
        script = script.replace(f"[{zh_label}]", f"[{m[zh_label]}]")
    return script


@dataclass
class ParsedSegment:
    line_no: int                 # 1-based
    voice_name: str              # always resolved to a real (or ephemeral default) voice name
    text: str
    control: str | None = None   # set when script wrote `<voice>(control)`; None otherwise


def parse_script(
    script: str,
    *,
    default_voice: str,
    known_names: Iterable[str],
) -> tuple[list[ParsedSegment], list[str]]:
    """Tokenize script into [(line_no, voice_name, control, text), ...] segments + warnings.

    `default_voice` and `known_names` are matched case-insensitively after trim.
    The default voice always starts each line with `control=None`.
    """
    name_lookup = {n.strip().lower(): n for n in known_names}
    default_norm = default_voice.strip().lower()
    if default_norm not in name_lookup:
        # Allow the default to be used even if not in known_names (ephemeral default voice).
        # Exception: names starting with `__` are internal/ephemeral and must NOT be
        # addressable via <voice name> script switches — preserve them as unknown text.
        if not default_voice.startswith("__"):
            name_lookup[default_norm] = default_voice

    segments: list[ParsedSegment] = []
    warnings: list[str] = []

    # Split on \r\n, \r, or \n in any order.
    lines = re.split(r"\r\n|\r|\n", script)

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        active_name = name_lookup.get(default_norm, default_voice)
        active_control: str | None = None
        cursor = 0
        accumulator = ""

        for match in _SWITCH_RE.finditer(line):
            # text before the tag goes to the active voice
            accumulator += line[cursor:match.start()]
            raw_name = match.group(1).strip()
            raw_control = match.group(2)  # None if `(...)` absent
            norm = raw_name.lower()
            if norm in name_lookup:
                # Flush accumulator and switch voice.
                if accumulator:
                    segments.append(ParsedSegment(idx, active_name, accumulator, active_control))
                    accumulator = ""
                active_name = name_lookup[norm]
                active_control = raw_control  # may be None or "" or "control text"
            else:
                # Unknown voice: preserve the entire matched span verbatim (including
                # any trailing `(control)`) and warn. Active voice/control unchanged.
                accumulator += match.group(0)
                warnings.append(
                    f"line {idx}: unknown voice tag {match.group(0)!r} preserved as text"
                )
            cursor = match.end()

        accumulator += line[cursor:]
        if accumulator:
            segments.append(ParsedSegment(idx, active_name, accumulator, active_control))

    return segments, warnings
