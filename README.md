# Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems

This repository contains the implementation for the paper *"Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems"*.

A comprehensive framework for analyzing road geometry, segmenting and clustering similar road sections using DTW-based matching and hybrid agglomerative clustering, performing coverage-based test suite reduction via greedy road selection, and prioritizing roads using multi-metric scoring with APFD-based evaluation. The extended version of the paper is under review.

## Note: 

The code is updated in order to improve the clustering technique with comprehensive distance metric based on DTW, max curvature, curvature entropy, and section length.  

## Architecture Overview

The framework is organized into modular components that handle specific responsibilities:

```
main.py                      # Main orchestration
├── path_config.py          # Path management
├── data_classes.py         # Data structures
├── io_operations.py        # File I/O operations
├── section_analysis.py     # Road section analysis
├── dynamic_analysis.py    # Vehicle behavior analysis
├── gpu_operations.py      # GPU acceleration (CUDA/MPS)
├── clustering.py          # Hybrid agglomerative clustering
├── coverage_reduction.py  # Test reduction logic
├── prioritization.py      # Road prioritization
└── comparison_analysis.py # Approach comparison
```

## Core Files

### `main.py` (Main Entry Point)
- Orchestrates the entire analysis pipeline
- Manages workflow: data loading → section analysis → clustering → reduction → prioritization
- Contains the `main()` function for execution

### `path_config.py`
- Centralized path management
- Supports custom configurations for different datasets
- Manages input/output directory structures

### `data_classes.py`
- `DynamicMetrics`: Stores vehicle behavior metrics (speed, steering, errors, etc.)
- `RoadSegment`: Represents classified road sections with geometric properties

## Analysis Components

### `section_analysis.py`
- Classifies road segments into straight/curved sections using hysteresis
- Performs DTW-based section matching with curvature profiles

### `dynamic_analysis.py`
- Calculates dynamic behavior metrics (steering complexity, speed variation, etc.)
- Computes dynamic similarity between road sections

### `clustering.py`
- Implements hybrid agglomerative clustering with multi-metric geometric features
- Combines DTW curvature profile similarity, peak difficulty, shape entropy, and section length into a weighted Euclidean (L2) distance
- Supports dynamic behavioral features for hybrid geometric + dynamic clustering
- Adaptive threshold calculation based on data distribution characteristics (CV-based percentile selection)
- Uses agglomerative clustering with complete linkage and precomputed distance matrices
- Groups sections by type (left_curve/right_curve) and applies clustering separately

### `gpu_operations.py`
- Provides GPU acceleration for CUDA (NVIDIA) and Apple Silicon (MPS via PyTorch)
- Implements batch DTW computation on GPU for large distance matrices
- Supports GPU-accelerated agglomerative clustering via cuML when available
- Falls back to CPU (scikit-learn with optimized BLAS/LAPACK) when GPU unavailable
- Includes GPU vs CPU performance benchmarking

## Reduction & Prioritization

### `coverage_reduction.py`
- Applies coverage-based test reduction using hybrid agglomerative clustering results
- Uses greedy coverage-based road selection with priority scoring
- Handles unique cluster prioritization (clusters covered by only one road)
- Priority scoring combines curvature variation, critical scenarios, unique patterns, road length, and dynamic behavior metrics
- Applies failed-road bonus for roads with historical simulation failures

### `prioritization.py`
- Implements multiple prioritization approaches (hybrid, random)
- Includes F*K/N probability benchmarking

## Infrastructure

### `io_operations.py`
- Handles file I/O operations (JSON, CSV)
- Manages section registry persistence and loading
- Saves comprehensive analysis results (coverage reduction, cluster analysis, dynamic analysis, prioritization comparison)
- Loads failed road IDs for priority weighting from CSV files
- Saves prioritized roads with priority distribution and failed road statistics

### `comparison_analysis.py`
- Compares multiple prioritization approaches (hybrid, random)
- Calculates APFD (Average Percentage of Fault Detection) for both whole test suite and selected subset
- Performs top-k fault detection analysis (k = 5, 10, 15, 20, 25)
- Provides overlap analysis between approaches' top 10 roads
- Saves comparison results to JSON files
- Prints formatted comparison summary with recommendations

## Step-by-Step Execution Pipeline

### 1. Data Preparation

The data is organized in the following structure:

```
Dataset/
├── roads/
│   └── a1/
│       ├── 0.json
│       ├── 1.json
│       └── ... (road geometry files)
├── roads_dynamic_data/
│   └── a1/
│       ├── 0.csv
│       ├── 1.csv
│       └── ... (vehicle simulation data)
└── failed_data/
    └── a1.csv          # List of failed road IDs
```

### 2. Configuration Setup

The system uses default paths but can be customized:

```python
# Custom configuration
config = PathConfig.create_custom_config(
    base_output_dir="./my_analysis",
    road_data_dir="./custom_data/roads",
    dynamic_data_dir="./custom_data/dynamic"
)
```

### 3. Pipeline Execution

Run the complete analysis:
```bash
python main.py
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

## 📊 Output Structure

After execution, the system creates:

```
output/
├── a1_sections/
│   ├── section_registry.json
│   └── road_metadata.json
├── a1_matching_info/
│   ├── matched_sections.json
│   ├── unmatched_sections.json
│   └── section_matches.json
├── a1_coverage_based_reduction/
│   ├── coverage_reduction_summary.json
│   ├── cluster_analysis.json
│   ├── prioritized_selected_roads_for_testing.json
│   ├── dynamic_analysis.json
│   └── prioritization_comparison/
│       ├── hybrid_prioritization.json
│       ├── random_prioritization.json
│       └── prioritization_comparison_analysis.json
```

## ⚙️ Key Configuration Parameters

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

## 📄 License & Citation

This project is released under the MIT License — see the LICENSE file for details.

This framework is intended for research on test suite reduction and autonomous vehicle validation. 

If you use this framework, or any part of it, in academic work, please cite the following paper:
```
@article{ali2026coverage,
  title={Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems},
  author={Ali, Qurban and Stocco, Andrea and Mariani, Leonardo and Riganelli, Oliviero},
  journal={arXiv preprint arXiv:2601.08609},
  year={2026}
}
```
