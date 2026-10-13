const puppeteer = require('puppeteer');
const path = require('path');
const http = require('http');
const fs = require('fs');

const PORT = 8998;
const dashDir = '/home/subhash/projects/LyASC/lyasc_model/dashboard';
const outDir = '/home/subhash/projects/LyASC/report_materials/04_ui_screenshots';

const server = http.createServer((req, res) => {
  let cleanUrl = req.url.split('?')[0];
  let filePath = path.join(dashDir, cleanUrl === '/' ? 'index.html' : cleanUrl);
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
    
    await new Promise(r => setTimeout(r, 2500));

    // Capture hero KPI section
    const heroElement = await page.$('#overview');
    if (heroElement) {
      await heroElement.screenshot({ path: path.join(outDir, 'dashboard_hero.png') });
      console.log('Captured dashboard_hero.png');
    }

    // Capture headline finding section
    const headlineElement = await page.$('#headline-finding');
    if (headlineElement) {
      await headlineElement.screenshot({ path: path.join(outDir, 'dashboard_headline_finding.png') });
      console.log('Captured dashboard_headline_finding.png');
    }

    // Capture Phase E section
    const phaseEElement = await page.$('#phase-e');
    if (phaseEElement) {
      await phaseEElement.screenshot({ path: path.join(outDir, 'dashboard_phase_e.png') });
      console.log('Captured dashboard_phase_e.png');
    }

    // Capture Decisions section
    const decisionsElement = await page.$('#decisions');
    if (decisionsElement) {
      await decisionsElement.screenshot({ path: path.join(outDir, 'dashboard_decisions.png') });
      console.log('Captured dashboard_decisions.png');
    }

    // Capture Full Dashboard view
    await page.screenshot({ path: path.join(outDir, 'dashboard_full_page.png'), fullPage: true });
    console.log('Captured dashboard_full_page.png');

    await browser.close();
    console.log('Dashboard screenshots successfully captured!');
  } catch (e) {
    console.error('Puppeteer capture error:', e);
  } finally {
    server.close();
  }
});
