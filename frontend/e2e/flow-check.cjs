// Browser flow check. Needs: backend on :8000 (API_CORS_ORIGINS=http://127.0.0.1:5173), 'pnpm preview --port 5173', and playwright with Chromium.
// Run from frontend/: NODE_PATH=$(npm root -g) node e2e/flow-check.cjs
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const reqs = [];
  page.on('request', r => { if (r.url().includes('/predict')) reqs.push(r.method() + ' ' + r.url()); });
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: /Start Assessment/ }).first().click();
  // Submit without answers first: should show validation, not call the API
  await page.getByRole('button', { name: 'Continue' }).click();
  console.log('validation shown:', await page.locator('.validation').count() > 0);
  for (let step = 0; step < 3; step++) {
    const labels = page.locator('.fields label');
    const n = await labels.count();
    for (let i = 0; i < n; i++) {
      const sel = labels.nth(i).locator('select');
      if (await sel.count()) { await sel.selectOption({ index: 2 }); }
      else { const v = step === 0 ? '27.5' : '3'; await labels.nth(i).locator('input').fill(v); }
    }
    await page.getByRole('button', { name: 'Continue' }).click();
  }
  await page.screenshot({ path: '/home/claude/work/scratch/review.png', fullPage: true });
  await page.getByRole('button', { name: 'Submit Assessment' }).click();
  await page.waitForSelector('.results-page', { timeout: 30000 });
  console.log('results pill:', await page.locator('.result-pill').innerText());
  console.log('gauge:', await page.locator('.score-ring strong').innerText());
  await page.screenshot({ path: '/home/claude/work/scratch/results.png', fullPage: true });
  await page.getByRole('button', { name: /Explore What-If/ }).first().click();
  await page.locator('.sim-fields select').nth(1).selectOption({ value: '4' });
  await page.getByRole('button', { name: /Compare Scenarios/ }).click();
  await page.waitForSelector('.difference b', { timeout: 30000 });
  await page.waitForTimeout(500);
  console.log('difference:', await page.locator('.difference b').innerText());
  await page.screenshot({ path: '/home/claude/work/scratch/simulator.png', fullPage: true });
  console.log('predict calls:', reqs);
  console.log('errors:', errors);
  await browser.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
