#!/usr/bin/env python3
"""
LyASC Dashboard Data Exporter
==============================
Reads all experimental results from disk and produces a single
dashboard_data.json for the static showcase dashboard.

CPU-only. Read-only from results/. No torch/CUDA dependency.
Re-run this script right before the review to pick up any newly
completed Phase E results.

Usage:
    python lyasc_model/scripts/export_dashboard_data.py
"""
import json
import os
import re
import csv
import sys
import statistics
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LYASC_RESULTS = PROJECT_ROOT / "lyasc_model" / "results"
BASE_RESULTS = PROJECT_ROOT / "base_model" / "results"
LOG_DIR = LYASC_RESULTS / "logs"
OUTPUT_PATH = PROJECT_ROOT / "lyasc_model" / "dashboard" / "dashboard_data.json"


# ── Helpers ────────────────────────────────────────────────────────────

def read_json(path):
    """Read a JSON result file, return dict or None."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_csv_table(path):
    """Read a reproduced CSV table into a list of dicts."""
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def parse_training_log(path, sample_interval=5):
    """Parse a training .log file for iteration-level metrics.
    
    Returns every `sample_interval`-th data point (plus first/last)
    to keep JSON size manageable.
    """
    points = []
    try:
        with open(path) as f:
            for line in f:
                # Match iter lines
                m = re.search(
                    r'iter=\s*(\d+)\s.*?loss=([0-9.]+)\s+'
                    r'train_acc=([0-9.]+)%\s+grad_norm=([0-9.]+)',
                    line
                )
                if m:
                    point = {
                        'iter': int(m.group(1)),
                        'loss': float(m.group(2)),
                        'train_acc': float(m.group(3)),
                        'grad_norm': float(m.group(4))
                    }
                    points.append(point)
                # Match alpha diagnostic lines (always follow an iter line)
                m2 = re.search(
                    r'alpha: mean=([0-9.]+)\s+std=([0-9.]+)\s+'
                    r'min=([0-9.]+)\s+max=([0-9.]+)',
                    line
                )
                if m2 and points:
                    points[-1]['alpha_mean'] = float(m2.group(1))
                    points[-1]['alpha_std'] = float(m2.group(2))
                    points[-1]['alpha_min'] = float(m2.group(3))
                    points[-1]['alpha_max'] = float(m2.group(4))
    except FileNotFoundError:
        return []

    if not points:
        return []

    # Subsample but always keep first and last
    sampled = []
    for i, p in enumerate(points):
        if i == 0 or i == len(points) - 1 or i % sample_interval == 0:
            sampled.append(p)
    return sampled


def aggregate_seeds(results):
    """Given a list of result dicts, compute mean/std of final_test_error."""
    errors = [r['final_test_error'] for r in results if r.get('final_test_error') is not None]
    if not errors:
        return None, None, 0
    mean = statistics.mean(errors)
    std = statistics.pstdev(errors) if len(errors) > 1 else 0.0
    return round(mean, 3), round(std, 3), len(errors)


# ── Collect base_model reproduction tables ─────────────────────────────

def collect_base_model_tables():
    tables = {}
    for name, filename in [
        ('table1', 'reproduced_table1_FINAL.csv'),
        ('table3', 'reproduced_table3_FINAL.csv'),
        ('table4', 'reproduced_table4_FINAL.csv'),
    ]:
        path = BASE_RESULTS / "tables" / filename
        rows = read_csv_table(path)
        if rows:
            tables[name] = []
            for r in rows:
                tables[name].append({
                    'variant': r.get('variant_name', ''),
                    'depth': int(r.get('depth', 110)),
                    'n_seeds': int(r.get('n_seeds_found', 0)),
                    'mean_error': float(r.get('mean_error', 0)),
                    'std_error': float(r.get('std_error', 0)),
                    'paper_error': float(r.get('paper_error', 0)),
                    'abs_diff': float(r.get('abs_diff', 0)),
                    'pass_fail': r.get('pass_fail', ''),
                })
    return tables


# ── Collect LyASC experiment results ───────────────────────────────────

def collect_lyasc_results():
    """Collect all LyASC experimental results, grouped by phase."""
    phases = {}

    # ── Phase C: Table 1 Equivalence (LyASC without LN) ──
    phase_c = {
        'name': 'Phase C: Table 1 Equivalence',
        'description': 'LyASC (no normalization) vs base_model 1xSkip on Table 1',
        'status': 'complete',
        'runs': [],
    }
    for seed in [0, 1, 2]:
        path = LOG_DIR / f"lyasc_table1_equiv__LyASC__depth110__seed{seed}.json"
        r = read_json(path)
        if r:
            log_path = LOG_DIR / f"lyasc_table1_equiv__LyASC__seed{seed}.log"
            phase_c['runs'].append({
                'variant': r.get('variant_name', 'LyASC'),
                'seed': seed,
                'final_test_error': r['final_test_error'],
                'total_iters': r['total_iters'],
                'wall_clock_hours': round(r.get('wall_clock_seconds', 0) / 3600, 2),
                'lyasc_diagnostics': r.get('lyasc_diagnostics', {}),
                'training_curve': parse_training_log(str(log_path)),
            })
    errors = [run['final_test_error'] for run in phase_c['runs']]
    if errors:
        phase_c['aggregate'] = {
            'mean': round(statistics.mean(errors), 3),
            'std': round(statistics.pstdev(errors), 3),
            'n_seeds': len(errors),
            'paper_baseline': 6.31,
            'paper_baseline_label': '1xSkip (Table 1)',
        }
    phases['phase_c'] = phase_c

    # ── LyASC+LN Ablation ──
    abl_path = LOG_DIR / "lyasc_ablation_ln__LyASCplusLN__depth110__seed0.json"
    abl = read_json(abl_path)
    if abl:
        abl_log = LOG_DIR / "lyasc_ablation_ln__LyASCplusLN__seed0.log"
        phases['lyasc_ln_ablation'] = {
            'name': 'LyASC+LN Ablation',
            'description': 'Single-seed LyASC+LayerNorm2d test to verify LN integration',
            'status': 'complete',
            'runs': [{
                'variant': 'LyASC+LN',
                'seed': 0,
                'final_test_error': abl['final_test_error'],
                'total_iters': abl['total_iters'],
                'wall_clock_hours': round(abl.get('wall_clock_seconds', 0) / 3600, 2),
                'lyasc_diagnostics': abl.get('lyasc_diagnostics', {}),
                'training_curve': parse_training_log(str(abl_log)),
            }],
        }

    # ── Phase D: Cold Start Strategy ──
    phase_d = {
        'name': 'Phase D: Cold Start Strategy',
        'description': 'Evaluate cold_start_strategy (none vs damped vs warmup) for LyASC+LN',
        'status': 'complete',
        'decision': 'cold_start_strategy = "none" (pragmatic baseline — none and damped statistically indistinguishable)',
        'strategies': {},
    }
    for strat in ['none', 'damped', 'warmup']:
        strat_runs = []
        for seed in [0, 1, 2]:
            path = LOG_DIR / "phase_d_coldstart" / f"lyasc_phase_d_coldstart__LyASCplusLN_{strat}__depth110__seed{seed}.json"
            r = read_json(path)
            if r:
                log_path = LOG_DIR / "phase_d_coldstart" / f"lyasc_phase_d_coldstart__LyASCplusLN_{strat}__seed{seed}.log"
                strat_runs.append({
                    'variant': r.get('variant_name', f'LyASC+LN_{strat}'),
                    'seed': seed,
                    'final_test_error': r['final_test_error'],
                    'total_iters': r['total_iters'],
                    'wall_clock_hours': round(r.get('wall_clock_seconds', 0) / 3600, 2),
                    'lyasc_diagnostics': r.get('lyasc_diagnostics', {}),
                    'training_curve': parse_training_log(str(log_path)),
                })
        if strat_runs:
            errors = [run['final_test_error'] for run in strat_runs]
            phase_d['strategies'][strat] = {
                'runs': strat_runs,
                'aggregate': {
                    'mean': round(statistics.mean(errors), 3),
                    'std': round(statistics.pstdev(errors), 3),
                    'n_seeds': len(errors),
                }
            }
    phases['phase_d'] = phase_d

    # ── Phase E1: Table 3 Stress Tests ──
    phase_e1 = {
        'name': 'Phase E1: Table 3 Stress Tests',
        'description': 'LyASC under catastrophic residual scaling (skip_scale/residual_scale variations)',
        'status': 'in_progress',
        'variants': {},
    }
    stress_configs = [
        ('LyASC_stress_0.5', 0.5, 1.0, 37.14, 'LN(0.5x+F)'),
        ('LyASC_stress_2.0', 1.0, 2.0, 37.59, 'LN(x+2F)'),
        ('LyASC_stress_3.0', 1.0, 3.0, 49.27, 'LN(x+3F)'),
    ]
    all_e1_complete = True
    for variant_name, skip_s, res_s, paper_target, paper_label in stress_configs:
        variant_runs = []
        for seed in [0, 1, 2]:
            json_path = LOG_DIR / "phase_e_stresstest" / f"lyasc_phase_e_stresstest__{variant_name}__depth110__seed{seed}.json"
            log_path = LOG_DIR / "phase_e_stresstest" / f"lyasc_phase_e_stresstest__{variant_name}__seed{seed}.log"
            r = read_json(json_path)
            if r:
                variant_runs.append({
                    'variant': r.get('variant_name', variant_name),
                    'seed': seed,
                    'final_test_error': r['final_test_error'],
                    'total_iters': r['total_iters'],
                    'wall_clock_hours': round(r.get('wall_clock_seconds', 0) / 3600, 2),
                    'lyasc_diagnostics': r.get('lyasc_diagnostics', {}),
                    'training_curve': parse_training_log(str(log_path)),
                })
            else:
                # Check if in-progress (log file exists but no JSON)
                if log_path.exists():
                    in_progress_curve = parse_training_log(str(log_path))
                    if in_progress_curve:
                        last = in_progress_curve[-1]
                        variant_runs.append({
                            'variant': variant_name,
                            'seed': seed,
                            'final_test_error': None,
                            'current_iter': last['iter'],
                            'total_iters': 64000,
                            'status': 'in_progress',
                            'training_curve': in_progress_curve,
                        })
                all_e1_complete = False

        if variant_runs:
            completed = [r for r in variant_runs if r.get('final_test_error') is not None]
            in_progress = [r for r in variant_runs if r.get('final_test_error') is None]

            errors = [r['final_test_error'] for r in completed]
            agg = None
            if errors:
                agg = {
                    'mean': round(statistics.mean(errors), 3),
                    'std': round(statistics.pstdev(errors) if len(errors) > 1 else 0, 3),
                    'n_seeds_complete': len(errors),
                    'n_seeds_in_progress': len(in_progress),
                    'n_seeds_total': 3,
                }

            phase_e1['variants'][variant_name] = {
                'skip_scale': skip_s,
                'residual_scale': res_s,
                'paper_comparison': {
                    'label': paper_label,
                    'error': paper_target,
                },
                'runs': variant_runs,
                'aggregate': agg,
                'status': 'complete' if len(completed) == 3 else ('partial' if completed else 'in_progress'),
            }

    if all_e1_complete:
        phase_e1['status'] = 'complete'
    phases['phase_e1'] = phase_e1

    return phases


# ── Collect decisions log ──────────────────────────────────────────────

def collect_decisions():
    """Parse key decisions from the main decisions.log."""
    decisions = []
    main_log = LOG_DIR / "phase_d_coldstart" / "decisions.log"
    if not main_log.exists():
        return decisions

    with open(main_log) as f:
        content = f.read()

    # Split on separator lines
    blocks = re.split(r'-{20,}', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        dec_m = re.search(r'DECISION:\s*(.+)', block)
        val_m = re.search(r'VALUE:\s*(.+?)(?:\nRATIONALE:)', block, re.DOTALL)
        rat_m = re.search(r'RATIONALE:\s*(.+)', block, re.DOTALL)
        if dec_m:
            decisions.append({
                'decision': dec_m.group(1).strip(),
                'value': val_m.group(1).strip() if val_m else '',
                'rationale': rat_m.group(1).strip() if rat_m else '',
            })

    # Deduplicate: keep only unique decision names (last occurrence wins)
    seen = {}
    for d in decisions:
        seen[d['decision']] = d
    return list(seen.values())


# ── Timing benchmarks ─────────────────────────────────────────────────

def collect_benchmarks():
    """Collect Phase E pre-flight timing benchmarks."""
    benchmarks = {}
    bench_dir = LOG_DIR / "phase_e_benchmark"
    for name, filename in [
        ('noG', 'lyasc_phase_e_stresstest__LyASC_noG__depth110__seed99.json'),
        ('LayerNorm2d', 'lyasc_phase_e_stresstest__LyASC_stress_2.0__depth110__seed99.json'),
        ('BatchNorm2d', 'lyasc_phase_e_stresstest__LyASCplusBN__depth110__seed99.json'),
    ]:
        r = read_json(bench_dir / filename)
        if r:
            iters = r.get('total_iters', 2000)
            wall = r.get('wall_clock_seconds', 0)
            s_per_iter = wall / iters if iters > 0 else 0
            benchmarks[name] = {
                'variant': r.get('variant_name', name),
                'iters': iters,
                'wall_seconds': round(wall, 1),
                's_per_iter': round(s_per_iter, 4),
                'projected_64k_hours': round(64000 * s_per_iter / 3600, 2),
            }
    return benchmarks


# ── Build the full export ──────────────────────────────────────────────

def build_export():
    export = {
        'meta': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'generator': 'export_dashboard_data.py',
            'project': 'LyASC — Lyapunov-Adaptive Skip Connections',
            'note': 'Auto-generated. Re-run before presentation for latest data.',
        },
        'base_model_tables': collect_base_model_tables(),
        'lyasc_phases': collect_lyasc_results(),
        'decisions': collect_decisions(),
        'timing_benchmarks': collect_benchmarks(),
        'formula': {
            'base': 'y_l = G(α_l · (skip_scale · x_l) + (1 - α_l) · (residual_scale · F_l(x_l)))',
            'alpha': 'α_l = σ(β_l) where β_l is a learned per-layer parameter',
            'lambda': 'λ_l = Lyapunov exponent estimated from residual norms via StabilityEstimator',
            'cold_start': 'none (Phase D conclusion: no cold-start strategy needed)',
        },
    }
    return export


def main():
    data = build_export()

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    # Print summary
    phases = data['lyasc_phases']
    total_runs = 0
    for phase_key, phase_data in phases.items():
        if 'runs' in phase_data:
            total_runs += len(phase_data['runs'])
        if 'strategies' in phase_data:
            for strat, sd in phase_data['strategies'].items():
                total_runs += len(sd.get('runs', []))
        if 'variants' in phase_data:
            for vname, vd in phase_data['variants'].items():
                total_runs += len(vd.get('runs', []))

    file_size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"✓ Exported dashboard_data.json")
    print(f"  Path: {OUTPUT_PATH}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"  Total runs exported: {total_runs}")
    print(f"  Base model tables: {len(data['base_model_tables'])}")
    print(f"  Decisions logged: {len(data['decisions'])}")
    print(f"  Timing benchmarks: {len(data['timing_benchmarks'])}")
    print(f"  Generated at: {data['meta']['generated_at']}")

    # Warn about in-progress runs
    e1 = phases.get('phase_e1', {})
    for vname, vd in e1.get('variants', {}).items():
        in_progress = [r for r in vd.get('runs', []) if r.get('status') == 'in_progress']
        if in_progress:
            for r in in_progress:
                print(f"  ⚠ {vname} seed {r['seed']}: IN PROGRESS (iter {r.get('current_iter', '?')}/64000)")
        not_started = 3 - len(vd.get('runs', []))
        if not_started > 0:
            print(f"  ⚠ {vname}: {not_started} seed(s) not yet started")


if __name__ == '__main__':
    main()
