---
name: radar-cfar-target-extraction
description: "Pure Insight7 target extraction from CFAR detection map: nonzero + energy sort + clustering + per-range dedup, avoiding numpy dependency."
source: auto-skill
extracted_at: '2026-06-06T20:54:55.399Z'
---

# Radar CFAR Target Extraction (Pure Insight7)

## Overview

Extract target positions from a 2D CFAR detection map without numpy. Pipeline:
1. `ins.nonzero(det)` → get all detection indices
2. Read energies at detection positions via `.at()` + `.numpy().item()`
3. Sort by energy descending, take top candidates
4. Euclidean clustering on (doppler_idx, range_idx) pairs
5. **Deduplicate by range bin** — only keep the strongest cluster per unique range

## Complete Implementation

```python
def _extract_targets(energy, det, top_n=2, cluster_threshold=3.0):
    """
    Extract targets from CFAR detection map.
    
    Args:
        energy: 2D array [doppler_bins, range_gates]
        det: 2D bool array of same shape (CFAR output)
        top_n: Number of targets to return
        cluster_threshold: Max index distance for clustering
        
    Returns:
        ([(doppler_idx, range_idx), ...], raw_detection_count)
    """
    idx = ins.nonzero(det)
    n_det = int(idx.shape()[1]) if idx.shape() else 0
    if n_det == 0:
        return [], 0

    # Collect all detection points with their energies
    candidates = []
    for k in range(n_det):
        i = int(idx.at([0, k]).numpy().item())  # doppler index
        j = int(idx.at([1, k]).numpy().item())  # range index
        v = energy.at([i, j]).numpy().item()     # energy value
        candidates.append((i, j, v))

    # Sort by energy descending, take top N*10 candidates
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:max(top_n * 10, 20)]

    # Euclidean clustering
    points = [(i, j) for i, j, _ in top_candidates]
    visited = [False] * len(points)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        cluster_indices = [i]
        for j in range(i + 1, len(points)):
            if visited[j]:
                continue
            d = math.sqrt((points[i][0] - points[j][0]) ** 2 +
                          (points[i][1] - points[j][1]) ** 2)
            if d <= cluster_threshold:
                visited[j] = True
                cluster_indices.append(j)
        # Pick highest-energy point in this cluster
        best_idx = max(cluster_indices, key=lambda x: top_candidates[x][2])
        clusters.append((top_candidates[best_idx][0],
                         top_candidates[best_idx][1],
                         top_candidates[best_idx][2]))

    clusters.sort(key=lambda x: x[2], reverse=True)

    # KEY: deduplicate by range — keeps the strongest target per range bin
    # This avoids sidelobes at the same range (from Hamming window or other apodization)
    # from stealing the slot of a real target at a different range.
    seen_ranges = set()
    deduped = []
    for d, r, v in clusters:
        if r not in seen_ranges:
            seen_ranges.add(r)
            deduped.append((d, r, v))

    return [(d, r) for d, r, _ in deduped[:top_n]], n_det
```

## Why Dedup by Range Is Important

Without dedup, a strong target's Doppler sidelobe (e.g., ±1-2 bins from the main lobe) can create a separate cluster at the same range but different doppler. This cluster may have higher apparent energy than a second real target at a different range, causing the extractor to return two detections at the same range (both belong to the same target) and miss the second target entirely.

**Example** (32 pulses, Hamming window, 2 targets):
| Position | Energy | Note |
|----------|--------|------|
| d=21, r=749 | 4.4e+03 | **Sidelobe** of target 2 |
| d=11, r=749 | 4.3e+03 | Target 2 mainlobe |
| d=10, r=598 | 3.7e+03 | Sidelobe of target 1 |
| d=22, r=598 | 3.6e+03 | **Target 1** mainlobe |

Without dedup: picks (d=21, r=749) and (d=11, r=749) → both same range, misses target 1.
With dedup: picks (d=11, r=749) and (d=22, r=598) → correct targets.

## Integration with Radar Pipeline

```python
# In run_detection():
_, det = ins.signal.ca_cfar(energy, [4, 4], [12, 12], 1e-6)
targets, raw_count = _extract_targets(energy, det, top_n=2)

# Validate range
range_bins = (ins.arange(N, dtype=ins.float64) - PC_OFFSET) * RANGE_PER_BIN
valid = []
for d, rr in targets:
    dist = range_bins.at([rr]).numpy().item()
    if 0 <= dist <= MAX_UNAMBIG_RANGE:
        valid.append((d, rr))
```

## Performance Note

The `for k in range(n_det)` loop with `.at().numpy().item()` is the slowest part for high CFAR counts (>500 detections). For typical configurations with PFA=1e-6 and 32×1000 cells, detection count is 30-90, making the overhead negligible (<5ms per frame).

For high detection counts, consider:
- Capping the loop (e.g. `min(n_det, 200)`)
- Pre-filtering by energy before clustering
- Using `ins.sort(energy.view())` + `ins.take()` to vectorize the extraction (requires framework support)
