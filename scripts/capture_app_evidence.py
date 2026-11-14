"""Capture genuine Streamlit screenshots and a short demo video with Playwright."""
from __future__ import annotations
import shutil
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "screenshots"
DEMO = ROOT / "demo"
SCREEN.mkdir(exist_ok=True)
DEMO.mkdir(exist_ok=True)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            record_video_dir=str(DEMO),
            record_video_size={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SCREEN / "01_news_detector.png"), full_page=True)

        # Exercise validation for empty input.
        page.get_by_role("button", name="Analyze news").click()
        page.wait_for_timeout(900)
        page.screenshot(path=str(SCREEN / "02_empty_input_validation.png"), full_page=True)

        # Exercise prediction with a non-empty sample. The output is a model prediction,
        # not an externally fact-checked verdict.
        sample = (
            "The national weather agency published its weekly forecast on Tuesday, "
            "detailing expected rainfall and temperature changes across several regions. "
            "The report includes regional observations and the agency's normal safety guidance."
        )
        page.locator("textarea").first.fill(sample)
        page.get_by_role("button", name="Analyze news").click()
        page.wait_for_timeout(1800)
        page.screenshot(path=str(SCREEN / "03_sample_prediction.png"), full_page=True)

        # Capture the metrics dashboard and methodology pages.
        try:
            page.get_by_text("Model dashboard", exact=True).click()
            page.wait_for_timeout(1400)
            page.screenshot(path=str(SCREEN / "04_model_dashboard.png"), full_page=True)
            page.get_by_text("Methodology & limits", exact=True).click()
            page.wait_for_timeout(800)
            page.screenshot(path=str(SCREEN / "05_methodology.png"), full_page=True)
        except Exception as exc:
            print(f"Some optional dashboard navigation screenshots were skipped: {exc}")

        video_path = page.video.path()
        context.close()
        browser.close()
        final_video = DEMO / "NewsGuard_Demo.webm"
        if video_path and Path(video_path).exists():
            shutil.move(video_path, final_video)
            print(f"Demo video saved: {final_video}")
        else:
            raise RuntimeError("Playwright did not produce a demo video.")
        print(f"Screenshots saved in {SCREEN}")


if __name__ == "__main__":
    main()
