import json
import glob
import os
import numpy as np
from collections import defaultdict

def main():
    log_dir = "/home/subhash/projects/LyASC/base_model/results/logs"
    json_files = glob.glob(os.path.join(log_dir, "*_table6_*.json"))
    
    # Structure: results[depth][variant] = [errors]
    results = defaultdict(lambda: defaultdict(list))
    
    for f in json_files:
        with open(f, 'r') as file:
            data = json.load(file)
            depth = data.get('depth')
            variant = data.get('variant_name')
            test_error = data.get('final_test_error')
            if test_error is not None:
                results[depth][variant].append(test_error)
                
    print("Table 6 Results:")
    print("Depth | Baseline (%) | 2rSkip+LN (%)")
    print("------|-------------|--------------")
    
    depths = sorted(results.keys())
    for depth in depths:
        base_errs = results[depth].get('baseline', [])
        r2_errs = results[depth].get('2rSkip+LN', [])
        
        base_str = "N/A"
        if base_errs:
            mean = np.mean(base_errs)
            std = np.std(base_errs, ddof=1) if len(base_errs) > 1 else 0.0
            base_str = f"{mean:.2f} ± {std:.2f}"
            
        r2_str = "N/A"
        if r2_errs:
            mean = np.mean(r2_errs)
            std = np.std(r2_errs, ddof=1) if len(r2_errs) > 1 else 0.0
            r2_str = f"{mean:.2f} ± {std:.2f}"
            
        print(f"{depth:5d} | {base_str:11s} | {r2_str:12s}")

if __name__ == "__main__":
    main()
