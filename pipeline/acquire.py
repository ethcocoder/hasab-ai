"""
pipeline/acquire.py — Data Acquisition
Downloads Amharic text from Wikipedia, OSCAR corpus, and local files.
"""

import os
import re
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Optional


class DataAcquirer:
    """Downloads and saves raw Amharic text corpus."""

    def __init__(self, output_dir: Path, max_mb: int = 500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes  = max_mb * 1024 * 1024
        self.collected  = 0

    def acquire_wikipedia(self) -> Path:
        """
        Download Amharic Wikipedia dump using wikipediaapi.
        Falls back to a small starter set if unavailable.
        """
        out_path = self.output_dir / "wikipedia_am.txt"

        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia(
                language    = "am",
                user_agent  = "AmharicAGI/1.0 (educational project)",
            )

            starter_pages = [
                "ኢትዮጵያ", "አዲስ አበባ", "አማርኛ", "ታሪክ", "ሳይንስ",
                "ጤና", "ትምህርት", "ቴክኖሎጂ", "ሙዚቃ", "ስፖርት",
                "Africa", "Ethiopia", "Addis_Ababa",
            ]

            texts = []
            for title in starter_pages:
                try:
                    page = wiki.page(title)
                    if page.exists():
                        texts.append(page.text)
                        self.collected += len(page.text.encode())
                        print(f"  ✓ {title} ({len(page.text)} chars)")
                    if self.collected >= self.max_bytes:
                        break
                    time.sleep(0.5)     # polite crawling
                except Exception as e:
                    print(f"  ✗ {title}: {e}")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts))
            print(f"✅ Wikipedia: {len(texts)} pages → {out_path}")

        except ImportError:
            print("⚠️  wikipediaapi not installed. Creating sample corpus.")
            self._write_sample_corpus(out_path)

        return out_path

    def acquire_local(self, local_files: List[Path]) -> Path:
        """Incorporate user-provided Amharic text files."""
        out_path = self.output_dir / "local_corpus.txt"
        texts = []
        for f in local_files:
            f = Path(f)
            if f.exists():
                with open(f, "r", encoding="utf-8") as fp:
                    texts.append(fp.read())
                print(f"  ✓ {f.name}")
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("\n\n".join(texts))
        return out_path

    def _write_sample_corpus(self, path: Path):
        """Write a minimal sample corpus for testing the pipeline."""
        sample = """
ኢትዮጵያ በምስራቅ አፍሪካ የምትገኝ አገር ናት። ዋና ከተማዋ አዲስ አበባ ስትሆን ህዝቧ ከ120 ሚሊዮን በላይ ነው።
ኢትዮጵያ ጥንታዊ ታሪክ ያላት አገር ናት። አክሱም፣ ላሊበላ፣ እና ጎንደር ታሪካዊ ከተሞቿ ናቸው።

አማርኛ የኢትዮጵያ ዋና ቋንቋ ሲሆን ከ25 ሚሊዮን በላይ ሰዎች ይናገሩታል።
ቋንቋው በግዕዝ ፊደል ይጻፋል። ፊደሉ ከቀኝ ወደ ግራ ሳይሆን ከግራ ወደ ቀኝ ነው የሚነበበው።

ቡና የኢትዮጵያ ዋና ምርት አንዱ ነው። ኢትዮጵያ የቡናው መነሻ አገር ናት ተብሎ ይታመናል።
የቡና ሥነ ሥርዓቱ ኢትዮጵያዊ ባህል አካል ሲሆን ማህበራዊ ትስስርን ያጠናክራል።

ኢትዮጵያ 13 ወሮች አሏት። ዘጠኝ ወሮች 30 ቀናት ሲኖሯቸው ሶስቱ 30 ቀን አሏቸው፣ ጳጉሜ ደግሞ 5 ወይም 6 ቀናት አሉት።
የኢትዮጵያ አዲስ ዓመት (እንቁጣጣሽ) በሴፕቴምበር ወር ይከበራል።

ትምህርት ለእያንዳንዱ ሰው ምን ያህል አስፈላጊ እንደሆነ ሁሉም ያውቃሉ።
ዕውቀት ሀብት ነው። ትምህርት ቤቶችና ዩኒቨርሲቲዎች ለሀገር ልማት ወሳኝ ናቸው።

ጤና ሕይወት ነው። ጤናማ ሕይወት ለመኖር ምግብ፣ ውሃ፣ እና እንቅስቃሴ አስፈላጊ ናቸው።
ዶክተሮችና ነርሶች ለሕዝብ ጤና ወሳኝ ሚና ይጫወታሉ።
""" * 50  # repeat to have enough data
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"  📝 Sample corpus written: {path}")
