import numpy as np
from typing import List, Dict
def otsu_threshold(data_dict):
    if not data_dict:
        return {}
    
    counts = np.array(list(data_dict.values()))
    if len(counts) < 2:
        return data_dict

    # Histogram
    min_val, max_val = counts.min(), counts.max()
    if min_val == max_val:
        return data_dict
        
    bins = np.arange(min_val, max_val + 2)
    hist, bin_edges = np.histogram(counts, bins=bins)
    
    bin_mids = bin_edges[:-1]
    total = hist.sum()
    
    current_max = -1
    threshold = min_val
    
    for i in range(1, len(bin_mids)):
        # Background (weights, means)
        w0 = hist[:i].sum() / total
        w1 = hist[i:].sum() / total
        if w0 == 0 or w1 == 0: continue
        
        m0 = (bin_mids[:i] * hist[:i]).sum() / hist[:i].sum()
        m1 = (bin_mids[i:] * hist[i:]).sum() / hist[i:].sum()
        
        # Between-class variance
        var_b = w0 * w1 * ((m0 - m1) ** 2)
        
        if var_b > current_max:
            current_max = var_b
            threshold = bin_mids[i]
            
    return {k: v for k, v in data_dict.items() if v >= threshold}

def render_text(token_indices: List[int], index_map: Dict[int, Dict], keep_underscores: bool = False) -> str:
    """Helper method to format token texts."""
    words = [index_map[i]["wordForm"] for i in token_indices]
    if not keep_underscores:
        words = [w.replace("_", " ") for w in words]
    return " ".join(words)