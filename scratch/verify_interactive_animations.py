import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1366, 'height': 900})
        page = await context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        print("Navigating to http://localhost:8080/index.html...")
        await page.goto("http://localhost:8080/index.html", wait_until="networkidle")

        # Check console errors
        print(f"Console errors: {console_errors}")

        # Wait for paradigm canvases to render
        await page.wait_for_selector("#paradigm canvas")
        await asyncio.sleep(1.0)
        await page.screenshot(path="scratch/shot_01_paradigm_live.png", full_page=False)
        print("Captured scratch/shot_01_paradigm_live.png")

        # Scroll to repair demo
        repair_demo = page.locator("#repair-demo")
        await repair_demo.scroll_into_view_if_needed()
        await asyncio.sleep(1.2)
        await page.screenshot(path="scratch/shot_02_repair_demo_clean.png", full_page=False)
        print("Captured scratch/shot_02_repair_demo_clean.png")

        # Click "Damage it"
        damage_btn = page.locator('button[data-act="damage"]')
        await damage_btn.click()
        print("Clicked Damage it...")
        await asyncio.sleep(1.2)
        await page.screenshot(path="scratch/shot_03_repair_demo_damaged.png", full_page=False)
        print("Captured scratch/shot_03_repair_demo_damaged.png")

        # Click "Let SP-SV-WSP Repair It"
        repair_btn = page.locator('button[data-act="repair"]')
        await repair_btn.click()
        print("Clicked Let SP-SV-WSP Repair It...")
        # Capture midway during scanning sweep
        await asyncio.sleep(0.7)
        await page.screenshot(path="scratch/shot_04_repair_demo_scanning.png", full_page=False)
        print("Captured scratch/shot_04_repair_demo_scanning.png")

        # Wait for repair to finish
        await asyncio.sleep(2.0)
        await page.screenshot(path="scratch/shot_05_repair_demo_repaired.png", full_page=False)
        print("Captured scratch/shot_05_repair_demo_repaired.png")

        # Toggle Dark Mode
        theme_btn = page.locator("#themeToggleBtn")
        await theme_btn.click()
        print("Toggled Dark Mode...")
        await asyncio.sleep(0.8)
        await page.screenshot(path="scratch/shot_06_repair_demo_dark_repaired.png", full_page=False)
        print("Captured scratch/shot_06_repair_demo_dark_repaired.png")

        # Scroll back up to paradigm in dark mode
        paradigm = page.locator("#paradigm")
        await paradigm.scroll_into_view_if_needed()
        await asyncio.sleep(1.0)
        await page.screenshot(path="scratch/shot_07_paradigm_dark.png", full_page=False)
        print("Captured scratch/shot_07_paradigm_dark.png")

        await browser.close()
        print("Verification script finished successfully!")

if __name__ == '__main__':
    asyncio.run(main())
