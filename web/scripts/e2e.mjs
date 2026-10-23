import { chromium } from "playwright-core";

const BASE = process.env.BASE_URL || "http://127.0.0.1:4321";
const executablePath = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const browser = await chromium.launch({ executablePath, headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("response", (r) => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

let step = 0;
function ok(msg) { console.log(`  ✓ ${++step}. ${msg}`); }

try {
  await page.goto(`${BASE}/learn/m01/dot-product/`, { waitUntil: "networkidle" });
  await page.waitForSelector(".quiz", { timeout: 15000 });
  await page.locator(".quiz").scrollIntoViewIfNeeded();
  await page.waitForTimeout(1200);
  ok("concept page loads quiz island");

  await page.locator(".numeric-row input").fill("11");
  await page.waitForFunction(() => {
    const b = document.querySelector(".quiz-actions button");
    return b && !b.disabled;
  }, { timeout: 5000 });
  await page.getByRole("button", { name: "提交答案" }).click();
  await page.waitForSelector(".explain.good", { timeout: 5000 });
  ok("numeric answer judged correct with explanation");

  for (let k = 0; k < 2; k++) {
    await page.getByRole("button", { name: /下一题/ }).click();
    await page.waitForTimeout(200);
    if (await page.locator(".numeric-row input").count()) {
      await page.locator(".numeric-row input").fill(k === 0 ? "0.707" : "5");
    } else {
      await page.locator(".option").first().click();
    }
    await page.getByRole("button", { name: "提交答案" }).click();
    await page.waitForSelector(".explain", { timeout: 5000 });
  }
  ok("choice/numeric questions submit and show explanations");

  await page.getByRole("button", { name: /查看结果/ }).click();
  await page.waitForSelector(".quiz-result", { timeout: 5000 });
  const sourceHref = await page.locator(".quiz-result a").first().getAttribute("href");
  const sourceBase = new URL(BASE).pathname.replace(/\/$/, "");
  if (!sourceHref || !sourceHref.startsWith(sourceBase + "/learn/")) {
    throw new Error(`quiz source link is outside base: ${sourceHref}`);
  }
  ok("quiz reaches result screen and its source link respects base");

  const mark = page.getByRole("button", { name: "标记为已读" });
  if (await mark.count()) {
    await mark.click();
    await page.waitForTimeout(400);
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForSelector("button:has-text('已读')", { timeout: 8000 });
    ok("read state persists across reload (IndexedDB)");
  }

  await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
  await page.locator("#search-open").click();
  await page.locator("#search-input").fill("点积");
  await page.waitForSelector(".search-item", { timeout: 5000 });
  const first = await page.locator(".search-item").first().getAttribute("href");
  if (!first) throw new Error("search item has no href");
  await page.locator(".search-item").first().click();
  await page.waitForURL(/what-is-a-vector/);
  const pathname = new URL(page.url()).pathname;
  const basePath = new URL(BASE).pathname.replace(/\/$/, "");
  if (!pathname.startsWith(basePath + "/learn/")) {
    throw new Error(`search navigated outside base: ${pathname}`);
  }
  ok(`search returns and navigates inside base: ${pathname}`);

  await page.goto(`${BASE}/review/`, { waitUntil: "networkidle" });
  await page.waitForSelector(".flashcard", { timeout: 10000 });
  await page.locator(".flashcard").click();
  const flipped = await page.locator(".flashcard.flipped").count();
  if (!flipped) throw new Error("flashcard did not flip");
  await page.getByRole("button", { name: /良好/ }).click();
  await page.waitForTimeout(300);
  ok("flashcard flips and schedules");

  await page.goto(`${BASE}/practice/m03/`, { waitUntil: "networkidle" });
  await page.waitForSelector(".quiz", { timeout: 10000 });
  ok("module practice page loads");

  await page.goto(`${BASE}/progress/`, { waitUntil: "networkidle" });
  await page.waitForSelector(".progress-grid", { timeout: 10000 });
  ok("progress dashboard reads local data");

  await page.goto(`${BASE}/learn/m01/orthogonal-projection/`, { waitUntil: "networkidle" });
  const imgOk = await page.locator(".figure-card img").first().evaluate((el) => el.complete && el.naturalWidth > 0);
  if (!imgOk) throw new Error("generated figure did not load");
  ok("generated SVG figure loads");

  await page.goto(`${BASE}/modules/m01/`, { waitUntil: "networkidle" });
  await page.waitForSelector(".mermaid svg", { timeout: 10000 });
  ok("Mermaid diagram renders to SVG");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${BASE}/learn/m01/dot-product/`, { waitUntil: "networkidle" });
  await page.locator("#menu-btn").click();
  const open = await page.evaluate(() => document.body.classList.contains("sidebar-open"));
  if (!open) throw new Error("mobile sidebar did not open");
  ok("mobile sidebar opens");

  if (errors.length) {
    console.log("\nBrowser errors:");
    for (const e of errors.slice(0, 10)) console.log("  -", e);
    process.exitCode = 1;
  } else {
    console.log(`\n✓ E2E ${step} checks passed, no console/page errors`);
  }
} finally {
  await browser.close();
}
