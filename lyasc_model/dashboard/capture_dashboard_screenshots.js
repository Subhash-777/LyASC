const puppeteer = require('puppeteer');
const path = require('path');
const http = require('http');
const fs = require('fs');

const PORT = 8999;

const server = http.createServer((req, res) => {
  let cleanUrl = req.url.split('?')[0];
  let filePath = path.join(__dirname, cleanUrl === '/' ? 'index.html' : cleanUrl);
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + filePath);
    } else {
      let contentType = 'text/html';
      if (filePath.endsWith('.js')) contentType = 'application/javascript';
      if (filePath.endsWith('.json')) contentType = 'application/json';
      if (filePath.endsWith('.png')) contentType = 'image/png';
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    }
  });
});

server.listen(PORT, async () => {
  try {
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });

    await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'networkidle0' });
    
    // Wait for CDN Chart.js to load and init() to finish rendering
    await page.waitForFunction(() => typeof window.Chart !== 'undefined', { timeout: 10000 });
    await page.evaluate(() => {
      if (typeof init === 'function') init();
    });
    await new Promise(r => setTimeout(r, 2000));

    // 1. Hero & KPI Section
    const heroElement = await page.$('#overview');
    if (heroElement) {
      await heroElement.screenshot({ path: path.join(__dirname, 'hero_kpi_screenshot.png') });
      console.log('Saved hero_kpi_screenshot.png');
    }

    // 2. Headline Finding & KPI Strip
    const hfElement = await page.$('#headline-finding');
    if (hfElement) {
      await hfElement.screenshot({ path: path.join(__dirname, 'headline_finding_screenshot.png') });
      console.log('Saved headline_finding_screenshot.png');
    }

    // 3. Phase E1 Section
    const phaseEElement = await page.$('#phase-e');
    if (phaseEElement) {
      await phaseEElement.screenshot({ path: path.join(__dirname, 'phase_e_screenshot.png') });
      console.log('Saved phase_e_screenshot.png');
    }

    // 4. Decisions Section
    const decisionsElement = await page.$('#decisions');
    if (decisionsElement) {
      await decisionsElement.screenshot({ path: path.join(__dirname, 'decisions_screenshot.png') });
      console.log('Saved decisions_screenshot.png');
    }

    await browser.close();
    console.log('All screenshots captured successfully.');
  } catch (e) {
    console.error(e);
  } finally {
    server.close();
  }
});
