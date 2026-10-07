import json
import glob
import os
import numpy as np
from collections import defaultdict

def main():
    log_dir = "/home/subhash/projects/LyASC/base_model/results/logs"
    json_files = glob.glob(os.path.join(log_dir, "*.json"))
    
    # Structure: results[table][config][variant][depth] = [errors]
    tables = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for f in json_files:
        with open(f, 'r') as file:
            data = json.load(file)
            config = data.get('config_name', '')
            if 'table1' in config:
                table = 'Table 1'
            elif 'table4' in config:
                table = 'Table 4'
            elif 'table6' in config:
                table = 'Table 6'
            else:
                table = 'Other'
            
            variant = data.get('variant_name', data.get('variant', 'baseline'))
            depth = data.get('depth', 110)
            test_error = data.get('final_test_error')
            
            if test_error is not None:
                tables[table][config][variant][depth].append(test_error)
                
    for table_name in sorted(tables.keys()):
        print(f"=== {table_name} ===")
        for config in sorted(tables[table_name].keys()):
            for variant in sorted(tables[table_name][config].keys()):
                for depth in sorted(tables[table_name][config][variant].keys()):
                    errs = tables[table_name][config][variant][depth]
                    mean = np.mean(errs)
                    std = np.std(errs, ddof=1) if len(errs) > 1 else 0.0
                    print(f"{config} | {variant} | depth {depth} : {mean:.2f} ± {std:.2f} (n={len(errs)})")
        print()

if __name__ == "__main__":
    main()
