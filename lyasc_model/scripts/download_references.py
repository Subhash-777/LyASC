#!/usr/bin/env python3
"""
download_references.py
Downloads PDF documents for all 23 reference papers cited in LyASC IEEE paper
into paper/Reference_pdfs/.
"""

import os
import shutil
import time
import urllib.request
import urllib.error

OUTPUT_DIR = os.path.abspath("paper/Reference_pdfs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCES = [
    # [1-5] Residual Learning, Identity Mappings, Scaling, Normalization, Optimization
    ("he2016deep", "https://arxiv.org/pdf/1512.03385.pdf", "Deep Residual Learning for Image Recognition"),
    ("he2016identity", "https://arxiv.org/pdf/1603.05027.pdf", "Identity Mappings in Deep Residual Networks"),
    ("liu2021rethinking", "https://arxiv.org/pdf/2105.07205.pdf", "Rethinking Skip Connections with Layer Normalization in Transformers and ResNets"),
    ("zaeemzadeh2020norm", "https://arxiv.org/pdf/1805.07477.pdf", "Norm-Preservation - Why Residual Networks Can Become Extremely Deep"),
    ("macdonald2023skip", "https://arxiv.org/pdf/2308.10515.pdf", "On Skip Connections and Normalisation Layers in Deep Optimisation"),
    
    # [6-11] Gating, Dynamic Routing, Stochastic Depth, Skip Controls
    ("srivastava2015highway", "https://arxiv.org/pdf/1505.00387.pdf", "Highway Networks"),
    ("wang2018skipnet", "https://arxiv.org/pdf/1711.09485.pdf", "SkipNet - Learning Dynamic Routing in Convolutional Networks"),
    ("veit2020adaptive", "https://arxiv.org/pdf/1711.11586.pdf", "Convolutional Networks with Adaptive Inference Graphs"),
    ("huang2016deep", "https://arxiv.org/pdf/1603.09382.pdf", "Deep Networks with Stochastic Depth"),
    ("joseph2024lambda", "https://arxiv.org/pdf/2410.10609.pdf", "Lambda-Skip Connections - The Architectural Component That Prevents Rank Collapse"),
    ("thota2025cosinegate", "https://arxiv.org/pdf/2512.22206.pdf", "CosineGate - Semantic Dynamic Routing via Cosine Incompatibility in Residual Networks"),
    
    # [12-18] Stability Analysis, Edge of Stability, Lyapunov Dynamics
    ("nar2018residual", "https://arxiv.org/pdf/1803.08203.pdf", "Residual Networks - Lyapunov Stability and Convex Decomposition"),
    ("cohen2021gradient", "https://arxiv.org/pdf/2103.00065.pdf", "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"),
    ("arora2022understanding", "https://arxiv.org/pdf/2205.09745.pdf", "Understanding Gradient Descent on the Edge of Stability in Deep Learning"),
    ("damian2022self", "https://arxiv.org/pdf/2209.15594.pdf", "Self-Stabilization - The Implicit Bias of Gradient Descent at the Edge of Stability"),
    ("andreyev2024edge", "https://arxiv.org/pdf/2412.20553.pdf", "Edge of Stochastic Stability - Revisiting the Edge of Stability for SGD"),
    ("mittra2024utilizing", "https://arxiv.org/pdf/2410.05988.pdf", "Utilizing Lyapunov Exponents in Designing Deep Neural Networks"),
    ("sharafi2024weight", "https://arxiv.org/pdf/2405.00743.pdf", "On the Weight Dynamics of Learning Networks"),
    
    # Foundational Architecture & Backbone Citations
    ("hu2018squeeze", "https://arxiv.org/pdf/1709.01507.pdf", "Squeeze-and-Excitation Networks"),
    ("veit2016residual", "https://arxiv.org/pdf/1605.06431.pdf", "Residual Networks Behave Like Ensembles of Relatively Shallow Networks"),
    ("ba2016layer", "https://arxiv.org/pdf/1607.06450.pdf", "Layer Normalization"),
    ("krizhevsky2009learning", "https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf", "Learning Multiple Layers of Features from Tiny Images"),
    ("ioffe2015batch", "https://arxiv.org/pdf/1502.03167.pdf", "Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

def is_valid_pdf(filepath):
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 1000:
        return False
    with open(filepath, "rb") as f:
        header = f.read(5)
        return header == b"%PDF-"

def main():
    print(f"Target directory: {OUTPUT_DIR}")
    
    # If Base_paper.pdf exists and liu2021rethinking.pdf doesn't, copy it
    base_paper_path = os.path.join(OUTPUT_DIR, "Base_paper.pdf")
    liu_path = os.path.join(OUTPUT_DIR, "liu2021rethinking.pdf")
    if is_valid_pdf(base_paper_path) and not is_valid_pdf(liu_path):
        print("Copying existing Base_paper.pdf to liu2021rethinking.pdf...")
        shutil.copyfile(base_paper_path, liu_path)

    success_count = 0
    total = len(REFERENCES)

    for idx, (key, url, title) in enumerate(REFERENCES, 1):
        clean_title = title.replace(":", " -").replace("?", "").replace("/", "-")
        target_name = f"{idx:02d} - {clean_title}.pdf"
        target_file = os.path.join(OUTPUT_DIR, target_name)
        if is_valid_pdf(target_file):
            size_mb = os.path.getsize(target_file) / (1024 * 1024)
            print(f"[{idx:02d}/{total:02d}] ALREADY EXISTS: {target_name} ({size_mb:.2f} MB)")
            success_count += 1
            continue

        print(f"[{idx:02d}/{total:02d}] Downloading {target_name} from {url}...")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp, open(target_file, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)

            if is_valid_pdf(target_file):
                size_mb = os.path.getsize(target_file) / (1024 * 1024)
                print(f"  -> SUCCESS: {target_name} ({size_mb:.2f} MB)")
                success_count += 1
            else:
                print(f"  -> ERROR: Downloaded file is not a valid PDF! Removing {target_file}")
                if os.path.exists(target_file):
                    os.remove(target_file)
        except Exception as e:
            print(f"  -> FAILED to download {target_name}: {e}")

        # Politeness delay to prevent rate limiting
        time.sleep(1.0)

    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {success_count}/{total} reference papers verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
