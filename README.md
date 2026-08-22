# Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for the paper *"Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems"*.

A framework for analyzing road geometry, segmenting and clustering similar road sections using DTW-based matching and hybrid agglomerative clustering, performing coverage-based test suite reduction via greedy road selection, and prioritizing roads using multi-metric scoring with APFD-based evaluation. The extended version of the paper is under review.

The code has been updated to improve the clustering technique with a comprehensive distance metric based on DTW, max curvature, curvature entropy, and section length.

## Quick Start

```bash
# Run the full analysis pipeline
python Code/main.py
```

The pipeline automatically loads road data, performs section matching, clusters similar sections, selects a minimal representative road set, and prioritizes them for testing.

---

## Architecture

The framework is organized into modular components:

```
main.py                      # Pipeline orchestration
├── path_config.py          # Path management
├── data_classes.py         # Data structures
├── io_operations.py        # File I/O
├── section_analysis.py     # Road section analysis
├── dynamic_analysis.py    # Vehicle behavior analysis
├── gpu_operations.py      # GPU acceleration
├── clustering.py          # Hybrid agglomerative clustering
├── coverage_reduction.py  # Test reduction
├── prioritization.py      # Road prioritization
└── comparison_analysis.py # Approach comparison
```

### Core Infrastructure

**`main.py`** — Orchestrates the pipeline: data loading, section analysis, clustering, reduction, prioritization, and result comparison. Supports GPU acceleration with automatic fallback.

**`path_config.py`** — Centralized path management with support for custom dataset configurations and automatic directory creation.

**`data_classes.py`** — Defines `DynamicMetrics` (vehicle behavior metrics) and `RoadSegment` (classified road sections with geometric properties).

**`io_operations.py`** — Manages all file I/O: JSON/CSV reading, section registry persistence, failed road ID loading, and comprehensive result saving (coverage reduction, cluster analysis, dynamic analysis, prioritization comparison).

### Road Analysis

**`section_analysis.py`** — Classifies road segments into straight/curved sections using hysteresis-based thresholding. Performs DTW-based section matching with curvature profiles, including full-section and subsection matching.

**`dynamic_analysis.py`** — Calculates dynamic behavior metrics (steering complexity, speed variation, cross-track error, yaw rate) from vehicle simulation data. Computes dynamic similarity between road sections for hybrid clustering.

**`gpu_operations.py`** — Provides GPU acceleration for CUDA (NVIDIA) and Apple Silicon (MPS via PyTorch). Implements batch DTW computation, GPU-accelerated agglomerative clustering via cuML, and CPU fallback with performance benchmarking.

### Clustering & Reduction

**`clustering.py`** — Implements hybrid agglomerative clustering with a multi-metric geometric distance combining DTW profile similarity, peak curvature (difficulty), shape entropy (complexity), and section length (size) via weighted Euclidean (L2) norm. Supports dynamic behavioral features for hybrid geometric + dynamic clustering. Adaptive threshold calculation uses CV-based percentile selection from pairwise distance distributions. Clustering is applied separately to left-curve and right-curve sections.

**`coverage_reduction.py`** — Applies coverage-based test reduction using greedy road selection with priority scoring. Handles unique cluster prioritization (clusters covered by only one road). Priority scores combine curvature variation, critical scenarios, unique patterns, road length, and dynamic behavior, with a bonus for historically failed roads.

### Prioritization & Evaluation

**`prioritization.py`** — Implements hybrid (multi-metric) and random prioritization approaches. Includes F*K/N probability benchmarking for expected random-selection performance and APFD calculation.

**`comparison_analysis.py`** — Compares prioritization approaches using APFD (whole suite and selected subset), top-k fault detection (k = 5–25), overlap analysis, and prints a formatted summary with recommendations.

---

## Pipeline

### 1. Data Preparation

Road geometry (JSON) and vehicle simulation data (CSV) are organized by campaign:

```
SensoDat/
├── roads/
│   └── a2/
│       ├── 0.json, 1.json, ...
├── roads_dynamic_data/
│   └── a2/
│       ├── 0.csv, 1.csv, ...
└── failed_data/
    └── a2.csv
```

### 2. Configuration

Default paths can be customized:

```python
config = PathConfig.create_custom_config(
    base_output_dir="./my_analysis",
    road_data_dir="./custom_data/roads",
    dynamic_data_dir="./custom_data/dynamic"
)
```

### 3. Execution

```bash
python Code/main.py
```

The pipeline automatically:
1. Checks GPU availability and benchmarks GPU vs CPU performance
2. Loads or computes section registry and road metadata
3. Performs or loads DTW-based section matching
4. Applies hybrid agglomerative clustering (with optional GPU acceleration)
5. Runs coverage-based road reduction with greedy selection
6. Prioritizes selected roads using hybrid and random approaches
7. Compares prioritization approaches (APFD, fault detection, top-k analysis)
8. Saves comprehensive results to the output directory

---

## Output

```
output/
├── a2_sections/
│   ├── section_registry.json
│   └── road_metadata.json
├── a2_matching_info/
│   ├── matched_sections.json
│   ├── unmatched_sections.json
│   └── section_matches.json
├── a2_dp_tsp_coverage_based_reduction/
│   ├── coverage_reduction_summary.json
│   ├── cluster_analysis.json
│   ├── prioritized_roads_for_testing.json
│   ├── dynamic_analysis.json
│   └── prioritization_comparison/
│       ├── hybrid_prioritization.json
│       ├── random_prioritization.json
│       └── prioritization_comparison_analysis.json
```

---

## Configuration

### RoadAnalyzer Parameters

```python
analyzer = RoadAnalyzer(
    curvature_threshold=0.015,      # Curvature threshold for segment classification
    min_segment_length=5,           # Minimum points per segment
    hysteresis_window=3,            # Hysteresis for stable classification
    min_subsection_length=5,        # Minimum points for subsection matching
    dtw_similarity_threshold=0.95,  # DTW similarity threshold
    use_gpu=True                    # Enable GPU acceleration
)
```

### Clustering Parameters

```python
reduction_results = analyzer.coverage_based_road_reduction(
    section_matches,
    coverage_threshold=1.0,                    # 100% cluster coverage required
    curvature_similarity_threshold=None,       # Auto-calculated from pairwise distance distribution (CV-based percentile)
    enable_dynamic_clustering=True,            # Use hybrid geometric + dynamic features
    dynamic_weight=0.5,                        # 50% dynamic, 50% geometric weighting
    include_dynamic_analysis=True,             # Include vehicle behavior data in road selection
    prioritize_unique_clusters=False           # Prioritize clusters covered by only one road
)
```

### Clustering Validation Metrics

The clustering step computes the **Silhouette Score** as an internal validation metric after clustering. This measures how well each section fits within its assigned cluster relative to other clusters, using the precomputed distance matrix.

- **Range**: -1 to 1 (higher is better; > 0.5 indicates reasonable structure, < 0 indicates potential misassignment)
- **Scope**: Computed over matched curved sections only (left-curve and right-curve groups combined), excluding straight sections (single cluster) and unmatched sections (individual clusters)
- **Effect**: Purely diagnostic — does not influence cluster assignments or road selection


---

## Citation

If you use this framework in academic work, please cite:

```bibtex
@article{ali2026coverage,
  title={Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems},
  author={Ali, Qurban and Stocco, Andrea and Mariani, Leonardo and Riganelli, Oliviero},
  journal={arXiv preprint arXiv:2601.08609},
  year={2026}
}
```

## License

MIT License — see the [LICENSE](LICENSE) file for details.
