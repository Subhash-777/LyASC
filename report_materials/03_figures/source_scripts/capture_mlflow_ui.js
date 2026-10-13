const { spawn } = require('child_process');
const puppeteer = require('puppeteer');
const path = require('path');

const PORT = 5050;
const outDir = '/home/subhash/projects/LyASC/report_materials/04_ui_screenshots';

// Spawn MLflow UI server against the copied database
const mlflowProc = spawn('/home/subhash/miniconda3/envs/lyasc-experiments/bin/mlflow', [
  'ui',
  '--backend-store-uri', 'sqlite:////tmp/lyasc_mlflow_copy/mlflow.db',
  '--port', String(PORT),
  '--host', '127.0.0.1'
], {
  cwd: '/home/subhash/projects/LyASC'
});

mlflowProc.stdout.on('data', data => console.log(`[MLflow UI]: ${data}`));
mlflowProc.stderr.on('data', data => console.log(`[MLflow UI err]: ${data}`));

setTimeout(async () => {
  try {
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });

    // 1. Experiments list view
    await page.goto(`http://127.0.0.1:${PORT}/#/experiments/0`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: path.join(outDir, 'mlflow_experiments_list.png') });
    console.log('Captured mlflow_experiments_list.png');

    // 2. Core Experiment Runs Table (Experiment ID 7)
    await page.goto(`http://127.0.0.1:${PORT}/#/experiments/7`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: path.join(outDir, 'mlflow_runs_table.png') });
    console.log('Captured mlflow_runs_table.png');

    // 3. Stress 3.0 metric compare view
    await page.goto(`http://127.0.0.1:${PORT}/#/experiments/0`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: path.join(outDir, 'mlflow_stress3_compare.png') });
    console.log('Captured mlflow_stress3_compare.png');

    await browser.close();
    console.log('MLflow UI screenshots successfully captured!');
  } catch (e) {
    console.error('Puppeteer MLflow error:', e);
  } finally {
    mlflowProc.kill();
    process.exit(0);
  }
}, 8000);
