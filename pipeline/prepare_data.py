"""Build clean, source-aware datasets for Qwen fine-tuning.

Outputs JSONL records instead of mixing unrelated text formats:
- data/processed/knowledge.jsonl: clean Amharic knowledge text
- data/processed/chat.jsonl: validated user/assistant conversations
- data/processed/review.jsonl: factual claims held out for human review
- data/processed/manifest.json: counts, rejection reasons, and warnings

This is quality control, not fact verification. Numeric, political, and
historical claims are flagged in the manifest for human review.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from soul.safety_gate import SafetyGate


GEEZ_RE = re.compile(r"[\u1200-\u137f]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"\s+")
HEADING_RE = re.compile(
    r"^(ምዕራፍ|ክፍል|ማውጫ|ይዘት|ማጣቀሻ|የታሪክ|የታሪካዊ|"
    r"የሚከተለው|አገናኞች|የውጭ አገናኞች)\b"
)


class AmharicDataBuilder:
    """Normalize, filter, deduplicate, and serialize training records."""

    def __init__(self, config):
        self.cfg = config
        self.data_raw = Path(config.paths.data_raw)
        self.data_processed = Path(config.paths.data_processed)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.min_length = config.data.min_sentence_length
        self.max_length = config.data.max_sentence_length
        self.min_ratio = config.data.min_amharic_ratio
        self.max_duplicate_similarity = config.data.dedup_threshold
        self.stats: Dict[str, int] = {
            "raw_text_candidates": 0,
            "kept_knowledge": 0,
            "kept_chat": 0,
            "removed_empty": 0,
            "removed_short": 0,
            "removed_low_amharic": 0,
            "removed_heading": 0,
            "removed_fragment": 0,
            "removed_url": 0,
            "removed_duplicate": 0,
            "removed_repeated": 0,
            "removed_unsafe": 0,
            "chat_invalid": 0,
            "review_numeric_or_historical": 0,
            "held_for_review": 0,
        }
        self._seen_normalized: set[str] = set()
        self._recent_knowledge: List[str] = []
        self._safety = SafetyGate()

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\u00a0", " ")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("。", "።").replace("，", "፣")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*([።፣፤፥፦፧፨!?])\s*", r"\1 ", text)
        return SPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(bool(GEEZ_RE.fullmatch(c)) for c in letters) / len(letters)

    @staticmethod
    def _normalized_key(text: str) -> str:
        return re.sub(r"[^\u1200-\u137f\w]+", "", text.casefold())

    def _quality_reason(self, text: str) -> str | None:
        if not text:
            return "empty"
        if URL_RE.search(text):
            return "url"
        if len(text) < self.min_length:
            return "short"
        if self._ratio(text) < self.min_ratio:
            return "low_amharic"
        if HEADING_RE.match(text) or text.endswith(":") and len(text) < 100:
            return "heading"
        if len(set(text)) <= 5 or re.search(r"(.)\1{5,}", text):
            return "repeated"
        # Wikipedia table/list fragments and unfinished quotations are poor LM data.
        if text.startswith(("|", "*", "#")) or text.count("«") > text.count("»"):
            return "fragment"
        return None

    def _is_duplicate(self, text: str) -> bool:
        key = self._normalized_key(text)
        if not key or key in self._seen_normalized:
            return True
        self._seen_normalized.add(key)
        # Compare only nearby accepted records to keep the builder fast.
        for previous in self._recent_knowledge[-2000:]:
            if len(text) >= 60 and SequenceMatcher(None, text, previous).ratio() >= self.max_duplicate_similarity:
                return True
        self._recent_knowledge.append(text)
        return False

    def _split_sentences(self, raw: str) -> Iterable[str]:
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        # Preserve sentence boundaries while preventing article headings from
        # becoming records. Newlines are also boundaries for scraped lists.
        for part in re.split(r"(?<=[።!?])\s+|\n+", raw):
            part = self.normalize(part)
            if part:
                yield part

    def _read_text_candidates(self) -> Iterable[Tuple[str, str]]:
        for path in sorted(self.data_raw.glob("*.txt")):
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for text in self._split_sentences(raw):
                yield text, path.name

    def _load_chat_candidates(self) -> List[dict]:
        records: List[dict] = []
        for path in sorted(self.data_raw.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        self.stats["chat_invalid"] += 1
                        continue
                    messages = item.get("messages") if isinstance(item, dict) else None
                    if not isinstance(messages, list) or len(messages) < 2:
                        self.stats["chat_invalid"] += 1
                        continue
                    cleaned = []
                    valid = True
                    for message in messages:
                        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}:
                            valid = False
                            break
                        content = self.normalize(str(message.get("content", "")))
                        if not content or len(content) < 2:
                            valid = False
                            break
                        safe, _, _ = self._safety.check_input(content)
                        if not safe:
                            valid = False
                            break
                        cleaned.append({"role": message["role"], "content": content})
                    if valid and cleaned[0]["role"] == "user" and any(m["role"] == "assistant" for m in cleaned[1:]):
                        records.append({"messages": cleaned, "source": path.name})
                    else:
                        self.stats["chat_invalid"] += 1
        return records

    def _review_flag(self, text: str) -> List[str]:
        flags = []
        if re.search(r"\d", text) or any(word in text for word in ("እ.ኤ.አ", "ዓ.ም", "መንግሥት", "ፓርቲ", "ጦርነት")):
            flags.append("human_review_numeric_historical_or_political")
        return flags

    def build(self) -> Dict[str, Path]:
        knowledge = []
        review = []
        for candidate, source in self._read_text_candidates():
            self.stats["raw_text_candidates"] += 1
            candidate = candidate[: self.max_length]
            reason = self._quality_reason(candidate)
            if reason:
                self.stats[f"removed_{reason}"] += 1
                continue
            safe, _, _ = self._safety.check_input(candidate)
            if not safe:
                self.stats["removed_unsafe"] += 1
                continue
            if self._is_duplicate(candidate):
                self.stats["removed_duplicate"] += 1
                continue
            flags = self._review_flag(candidate)
            if flags:
                self.stats["review_numeric_or_historical"] += 1
                self.stats["held_for_review"] += 1
                review.append({
                    "type": "text",
                    "text": candidate,
                    "source": source,
                    "review_flags": flags,
                })
                continue
            knowledge.append({
                "type": "text",
                "text": candidate,
                "source": source,
                "review_flags": flags,
            })

        chat = self._load_chat_candidates()
        # Deduplicate chat records by normalized user/assistant content.
        seen_chat = set()
        clean_chat = []
        for item in chat:
            key = json.dumps(item["messages"], ensure_ascii=False, sort_keys=True)
            if key in seen_chat:
                continue
            seen_chat.add(key)
            item["type"] = "chat"
            item["review_flags"] = []
            clean_chat.append(item)

        self.stats["kept_knowledge"] = len(knowledge)
        self.stats["kept_chat"] = len(clean_chat)
        outputs = {
            "knowledge": self.data_processed / "knowledge.jsonl",
            "chat": self.data_processed / "chat.jsonl",
            "review": self.data_processed / "review.jsonl",
            "manifest": self.data_processed / "manifest.json",
        }
        self._write_jsonl(outputs["knowledge"], knowledge)
        self._write_jsonl(outputs["chat"], clean_chat)
        self._write_jsonl(outputs["review"], review)
        manifest = {
            "schema_version": 1,
            "quality_note": "Flagged numeric, historical, and political claims are held out until human review.",
            "stats": self.stats,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "mix_recommendation": {"knowledge": 0.70, "chat": 0.30},
        }
        outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return outputs

    @staticmethod
    def _write_jsonl(path: Path, records: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
