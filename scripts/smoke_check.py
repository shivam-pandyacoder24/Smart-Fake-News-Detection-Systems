"""Quickly check the core project imports and preprocessing behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_text

checks = {
    "lowercase": clean_text("BREAKING News") == "breaking news",
    "html_removed": "script" not in clean_text("<b>News</b>"),
    "url_normalized": "url" in clean_text("Read https://example.com/story"),
    "empty_safe": clean_text(None) == "",
}
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL") + " - " + name)
if not all(checks.values()):
    raise SystemExit(1)
print("Core preprocessing smoke checks passed.")
