# Road Analysis Framework

This is the repository for the paper 'Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems'

A comprehensive framework for analyzing road geometry, segmenting and clustering similar road sections, and performing test suite reduction using hybrid agglomerative clustering.

## Architecture Overview

The framework is organized into modular components that handle specific responsibilities:

```
main.py                      # Main orchestration
├── path_config.py          # Path management
├── data_classes.py         # Data structures
├── io_operations.py        # File I/O operations
├── section_analysis.py     # Road section analysis
├── dynamic_analysis.py    # Vehicle behavior analysis
├── clustering.py          # Hybrid clustering algorithms
├── coverage_reduction.py  # Test reduction logic
├── prioritization.py      # Road prioritization
├── comparison_analysis.py # Approach comparison
└── visualization.py       # Results visualization
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
- Implements hybrid agglomerative clustering combining geometric + dynamic features
- Manages cluster formation with adaptive thresholding

## Reduction & Prioritization

### `coverage_reduction.py`
- Applies coverage-based test reduction
- Maps roads to clusters and selects optimal representatives
- Handles unique cluster prioritization

### `prioritization.py`
- Implements multiple prioritization approaches (hybrid, random)
- Includes F*K/N probability benchmarking

### `comparison_analysis.py`
- Compares different prioritization approaches
- Calculates APFD (Average Percentage of Fault Detection)
- Provides fault detection analysis

## Infrastructure

### `io_operations.py`
- Handles file I/O operations (JSON, CSV)
- Manages section registry persistence
- Saves analysis results and metadata
- Loads failed road IDs for priority weighting

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

MIT License - see LICENSE file for details.

This framework is designed for research in test suite reduction and autonomous vehicle validation. Please cite appropriately if used in academic publications.

For questions or contributions, please open an issue or pull request on the repository.
