"""
soul/culture_injector.py — Injects Amharic cultural knowledge into training data.
Upweights cultural examples, adds proverbs, history, and Ethiopian context.
"""

import re
import random
from typing import List, Dict, Optional
from pathlib import Path


# ── Amharic proverbs (ምሳሌ) with meanings ─────────────────────────────────────
AMHARIC_PROVERBS = [
    {"am": "እንቁላል ሲፈለፈል ያምራል።", "meaning": "Patience brings beautiful results."},
    {"am": "ተስፋ የቆረጠ ሰው ከሞተ አይቆጠርም።", "meaning": "One who loses hope is as good as dead."},
    {"am": "አንድ እጅ ጭብጨባ አያሰማም።", "meaning": "One hand cannot clap alone — unity is needed."},
    {"am": "ዝናብ ሲዘንብ ማዕበል ጓደኛው ነው።", "meaning": "In tough times, challenges come together."},
    {"am": "ቤት የሌለው ሰው ልቡ የሌለው ነው።", "meaning": "A person without a home has no heart."},
    {"am": "ፈጣን ምላሽ ከጠቢቡ ይወጣል።", "meaning": "A quick wise answer comes from the learned."},
    {"am": "ሁሉም ነገር ጊዜው አለው።", "meaning": "Everything has its time."},
    {"am": "ትምህርት ሁሉም ነገር ነው።", "meaning": "Education is everything."},
    {"am": "ብዙ ጓደኛ ካለህ ሀብታም ነህ።", "meaning": "If you have many friends, you are rich."},
    {"am": "ዕድሜ ካለ ሁሉ ይሆናል።", "meaning": "If there is life, everything is possible."},
]

# ── Ethiopian cultural QA pairs for training ─────────────────────────────────
CULTURAL_QA_PAIRS = [
    {
        "q": "የቡና ሥነ ሥርዓት ምንድን ነው?",
        "a": "የቡና ሥነ ሥርዓት (Coffee Ceremony) ኢትዮጵያዊ ባህል ሲሆን ቡና ሶስት ጊዜ ይሰራል፦ አቦል፣ ቶና፣ እና በርካ። ቡና ጠጥ ማለት ከጎረቤቶች፣ ቤተሰብ እና ጓደኞች ጋር ጊዜ ማሳለፍ ማለት ነው።"
    },
    {
        "q": "ጥምቀት ምን ዓይነት በዓል ነው?",
        "a": "ጥምቀት የኢትዮጵያ ኦርቶዶክስ ቤተ ክርስቲያን ዋና በዓሎች አንዱ ሲሆን የጌታን ጥምቀት ያስታውሳል። ታቦቶቹ ወደ ውሃ ወርደው ህዝቡ ይባረካሉ።"
    },
    {
        "q": "ኢትዮጵያ ስንት ክልሎች አሏት?",
        "a": "ኢትዮጵያ በአሁኑ ጊዜ 12 ክልሎች እና 2 ከተማ አስተዳደሮች (አዲስ አበባ እና ድሬ ዳዋ) አሏት።"
    },
    {
        "q": "አማርኛ ቋንቋ ስንት ፊደሎች አሉት?",
        "a": "አማርኛ የፊደሎቹ ቁጥር 33 ቤተሰቦች ሲሆን እያንዳንዱ 7 ቅርጾች አሉት። በጠቅላላ 231 ፊደሎች አሉ።"
    },
    {
        "q": "ዓድዋ ጦርነት መቼ ነበር?",
        "a": "የዓድዋ ጦርነት በ1896 (እ.ኤ.አ.) ማርች 1 ቀን ሲሆን ኢትዮጵያ ጣሊያንን አሸንፋ ነጻነቷን ጠብቃለች። ይህ ቀን አሁንም ብሔራዊ ቀን ሆኖ ይከበራል።"
    },
    {
        "q": "የኢትዮጵያ አዲስ ዓመት መቼ ነው?",
        "a": "የኢትዮጵያ አዲስ ዓመት (እንቁጣጣሽ) በሴፕቴምበር 11 ወይም 12 (ጎርጎሮሳዊ) ይከበራል። ኢትዮጵያ የራሷ ዘመን አቆጣጠር ሲኖራት 13 ወሮች አሏት።"
    },
    {
        "q": "ሰላምታ እንዴት ይሰጣል?",
        "a": "አማርኛ ሰላምታ፦ ሰላም (hello)፣ እንደምን አደሩ (good morning - formal)፣ እንደምን ዋሉ (good afternoon - formal)፣ እንደምን አመሹ (good evening)። ለሽማግሌ ሰዎች ሁሌ ፎርማል ሰላምታ ይጠቀሙ።"
    },
]

# ── Ethiopian calendar months ─────────────────────────────────────────────────
EC_MONTHS = [
    "መስከረም", "ጥቅምት", "ህዳር", "ታህሳስ",
    "ጥር", "የካቲት", "መጋቢት", "ሚያዚያ",
    "ግንቦት", "ሰኔ", "ሐምሌ", "ነሃሴ", "ጳጉሜ"
]


class CultureInjector:
    """
    Injects Ethiopian cultural knowledge into training batches.
    Upweights cultural data by repeating it during training.
    """

    def __init__(self, upweight_factor: int = 3, seed: int = 42):
        self.upweight_factor = upweight_factor
        self.rng = random.Random(seed)
        self.proverbs    = AMHARIC_PROVERBS
        self.qa_pairs    = CULTURAL_QA_PAIRS
        self.ec_months   = EC_MONTHS

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_into_dataset(self, texts: List[str]) -> List[str]:
        """Adds cultural examples to a text list, upweighted."""
        cultural = self._build_cultural_texts()
        augmented = texts + cultural * int(self.upweight_factor)
        self.rng.shuffle(augmented)
        return augmented

    def build_chat_examples(self) -> List[Dict]:
        """Returns chat-format training examples from cultural QA pairs."""
        examples = []
        for pair in self.qa_pairs:
            examples.append({
                "messages": [
                    {"role": "user",      "content": pair["q"]},
                    {"role": "assistant", "content": pair["a"]},
                ]
            })
        return examples * self.upweight_factor

    def get_random_proverb(self) -> str:
        p = self.rng.choice(self.proverbs)
        return f"{p['am']} — {p['meaning']}"

    def amharic_date_context(self, gc_month: int, gc_day: int) -> str:
        """Returns an Amharic cultural date context string."""
        ec_idx = (gc_month - 9) % 13
        ec_month = self.ec_months[ec_idx]
        return f"የኢትዮጵያ ቀን: {ec_month} {gc_day}"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_cultural_texts(self) -> List[str]:
        texts = []
        # Proverbs as plain text
        for p in self.proverbs:
            texts.append(f"ምሳሌ: {p['am']}\nትርጉም: {p['meaning']}")
        # QA pairs as plain text
        for qa in self.qa_pairs:
            texts.append(f"ጥያቄ: {qa['q']}\nመልስ: {qa['a']}")
        # Calendar
        for i, month in enumerate(self.ec_months, 1):
            texts.append(f"የኢትዮጵያ {i}ኛ ወር {month} ይባላል።")
        return texts
