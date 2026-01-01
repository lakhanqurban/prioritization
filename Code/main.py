import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import json, os, sys, traceback
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import time
import datetime
import csv
import math
import random

# Import refactored modules
from path_config import PathConfig
from data_classes import DynamicMetrics, RoadSegment
from clustering import ClusteringMixin
from section_analysis import SectionAnalysisMixin
from prioritization import PrioritizationMixin
from comparison_analysis import ComparisonAnalysisMixin
from coverage_reduction import CoverageReductionMixin
from io_operations import IOMixin
from dynamic_analysis import DynamicAnalysisMixin
from visualization import VisualizationMixin
from typing import List, Dict, Any
# Handle both relative and absolute imports
try:
    from utils.road_utils import calculate_curvature, calculate_road_length
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path to find utils
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.road_utils import calculate_curvature, calculate_road_length

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadAnalyzer(
    IOMixin,
    SectionAnalysisMixin,
    DynamicAnalysisMixin,
    ClusteringMixin,
    CoverageReductionMixin,
    PrioritizationMixin,
    ComparisonAnalysisMixin,
    VisualizationMixin
):
    """Main class for road analysis and comparison"""

    def __init__(self, curvature_threshold: float = 0.01,
                 dtw_similarity_threshold: float = 3.0,
                 path_config: Optional[PathConfig] = None,
                 min_segment_length: int = 5,
                 hysteresis_window: int = 3,
                 min_subsection_length: int = 10):

        self.curvature_threshold = curvature_threshold
        self.min_segment_length = min_segment_length
        self.hysteresis_window = hysteresis_window
        self.min_subsection_length = min_subsection_length
        self.dtw_similarity_threshold = dtw_similarity_threshold
        self.section_registry = {}
        self.next_section_id = 1
        self.road_metadata = {}
        
        # Path configuration
        self.paths = path_config if path_config is not None else PathConfig()
        self.paths.create_directories()
        
        # DEBUG: Log the actual configuration being used
        logger.info(f"🔧 PATH CONFIGURATION DEBUG:")
        logger.info(f"   • dynamic_data_dir: {self.paths.dynamic_data_dir}")
        logger.info(f"   • dynamic_data_path: {self.paths.dynamic_data_path}")
        logger.info(f"   • failed_roads_dir: {self.paths.failed_roads_dir}")
        logger.info(f"   • failed_roads_path: {self.paths.failed_roads_path}")
        
        # Dynamic data configuration
        self.dynamic_data_available = self.paths.dynamic_data_path.exists()
        
        if self.dynamic_data_available:
            logger.info(f"Dynamic data directory found: {self.paths.dynamic_data_path}")
        else:
            logger.warning(f"Dynamic data directory not found: {self.paths.dynamic_data_path}")
        
        # Load failed roads for priority weighting
        self.failed_road_ids = self.load_failed_road_ids()

def main():
    """Main function focused on all-section comparison and coverage-based reduction"""
    start_time = time.time()
    try:
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger(__name__)

        # Initialize analyzer with optimized parameters and dynamic data support
        analyzer = RoadAnalyzer()
        
        # System will automatically use optimized backends (OpenBLAS, SIMD, etc.)
        print("🔧 Using optimized computational backends")
        print("="*50)
        
        # Load and process roads using centralized paths
        road_files = [analyzer.paths.road_data_path / f"{i}.json" for i in range(1000) if (analyzer.paths.road_data_path / f"{i}.json").exists()]
        road_files = [f for f in road_files if f.exists()]
        if not road_files:
            logger.error("No valid road files found in the directory.")
            return

        # Load section_registry and road_metadata using centralized paths
        if analyzer.paths.section_registry_path.exists():
            with open(analyzer.paths.section_registry_path, "r") as f:
                analyzer.section_registry = json.load(f)
            if analyzer.paths.road_metadata_path.exists():
                with open(analyzer.paths.road_metadata_path, "r") as f:
                    analyzer.road_metadata = json.load(f)
        else:
            # Process all roads and save registry
            all_segments = analyzer.process_roads_batch(road_files)
            analyzer.save_section_registry(analyzer.paths.sections_output_path)

        # Load section_matches from disk if available using centralized paths
        if analyzer.paths.matched_sections_path.exists():
            with open(analyzer.paths.matched_sections_path, "r") as f:
                section_matches = json.load(f)
        else:
            print("🔍 Performing section matching analysis...")
            section_matches = analyzer.compare_all_sections()
            analyzer.save_matching_info(section_matches, analyzer.paths.matching_output_path)
        
        use_saved_match_info = analyzer.paths.matched_sections_path.exists() and analyzer.paths.unmatched_sections_path.exists()
        if use_saved_match_info:
            with open(analyzer.paths.matched_sections_path, "r") as f:
                matched_sections_data = json.load(f)
            with open(analyzer.paths.unmatched_sections_path, "r") as f:
                unmatched_sections_data = json.load(f)
            all_unmatched_sections = [s['section_id'] for s in unmatched_sections_data]
            # Accept only explicit curved types: 'curved', 'left_curve', 'right_curve'
            unmatched_curved_sections = [s['section_id'] for s in unmatched_sections_data if s.get('type', None) in ['curved', 'left_curve', 'right_curve']]
            unmatched_straight_sections = [s['section_id'] for s in unmatched_sections_data if s.get('type') == 'straight']
            straight_sections = [s['section_id'] for s in matched_sections_data if s.get('type') == 'straight'] + unmatched_straight_sections
            section_source_note = "(from saved matched/unmatched files)"
        else:
            # Use the same definition as _agglomerative_clustering for unmatched curved sections
            all_unmatched_sections = [sid for sid, data in analyzer.section_registry.items() if not data.get('matched', False)]
            unmatched_curved_sections = [sid for sid, data in analyzer.section_registry.items()
                                        if not data.get('matched', False) and data.get('type') in ['curved', 'left_curve', 'right_curve']]
            unmatched_straight_sections = [sid for sid, data in analyzer.section_registry.items()
                                           if not data.get('matched', False) and data.get('type') == 'straight']
            straight_sections = [sid for sid, data in analyzer.section_registry.items() if data.get('type') == 'straight']
            section_source_note = "(computed from section_registry)"

        # Coverage-based reduction with hybrid Agglomerative Clustering and dynamic analysis
        print("⚡ Applying coverage-based road reduction with hybrid clustering (geometric + dynamic)...")
        reduction_results = analyzer.coverage_based_road_reduction(
            section_matches,
            coverage_threshold=1.0,
            include_dynamic_analysis=True,
            curvature_similarity_threshold=None,
            enable_dynamic_clustering=True,
            dynamic_weight=0.4  # 40% dynamic, 60% geometric (matches method default)
        )
        analyzer.print_detailed_cluster_info(section_matches, reduction_results['section_to_cluster'])
        
        # Get all available roads for comparison
        all_available_roads = list(analyzer.road_metadata.keys())
        target_selection_size = len(reduction_results['selected_roads'])
        
        print(f"\n🎯 PRIORITIZATION ANALYSIS")
        print("="*50)
        print(f"Comparing prioritization approaches on {target_selection_size} selected roads...")
        
        # Calculate F*K/N benchmark once
        random_roads = set(random.sample(all_available_roads, target_selection_size))
        fkn_benchmark = analyzer._calculate_fkn_probability_score(
            list(random_roads),
            all_available_roads
        )
        
        # Run comparison (this will do the prioritization internally)
        comparison_results = analyzer.compare_prioritization_approaches(
            reduction_results['selected_roads'],
            approaches=["hybrid", "random"],
            fkn_benchmark=fkn_benchmark
        )
        
        # Extract prioritized roads from hybrid approach results
        prioritization_results = comparison_results.get('prioritization_results', {})
        prioritized_roads = prioritization_results.get('hybrid', [])
        
        # Print comparison summary
        print("\n📊 PRIORITIZATION COMPARISON RESULTS")
        print("="*50)
        analyzer.print_prioritization_comparison_summary(comparison_results)
        
        print("\n💾 Saving comprehensive analysis results...")
        current_time = time.time()
        execution_time = current_time - start_time
        analyzer.save_coverage_reduction_results(reduction_results, prioritized_roads, execution_time=execution_time)

        print(f"F*K/N Benchmark Score: {fkn_benchmark['expected_failing_tests']:.2f} expected failing tests in selection of {target_selection_size} roads")
        print(f"   F*K/N Parameters: F={fkn_benchmark['failed_roads']}, K={fkn_benchmark['selected_roads']}, N={fkn_benchmark['total_roads']}")
        print(f"   Expected failures: {fkn_benchmark['expected_failing_tests']}")
        print(f"   Formula: {fkn_benchmark['formula']}")
        print(f"   Note: This benchmark shows expected performance of random selection")
        

        print(f"Top Priority Roads (HYBRID Approach):")
        for i, road in enumerate(prioritized_roads[:10], 1):  # Show top 10
            failed_indicator = " 🔴" if road.get('is_failed_road', False) else ""
            print(f"  {i}. Road {road['road_id']}: {road.get('priority_class', 'N/A')} priority (score: {road.get('priority_score', 'N/A'):.3f}){failed_indicator}")
        
        # Show failed roads summary - filter to only selected roads
        selected_roads_set = set(reduction_results['selected_roads'])
        failed_road_ids_set = set(analyzer.failed_road_ids)
        
        # Find failed roads that are in the selected set (direct comparison)
        failed_roads_selected = selected_roads_set.intersection(failed_road_ids_set)
        
        print(f"\nFailed Roads Analysis:")
        print(f"  - Total failed roads loaded from CSV files: {len(analyzer.failed_road_ids)}")
        print(f"  - Failed roads in selected set: {len(failed_roads_selected)}")
        if failed_roads_selected:
            print(f"  - Failed road IDs selected: {sorted(list(failed_roads_selected))}")
            
            # Calculate average priority score for failed roads that were selected
            failed_roads_with_scores = [r for r in prioritized_roads if r['road_id'] in failed_roads_selected]
            if failed_roads_with_scores:
                avg_score = sum(r.get('priority_score', 0) for r in failed_roads_with_scores) / len(failed_roads_with_scores)
                print(f"  - Average priority score for failed roads: {avg_score:.3f}")
        
        # Show percentage of failed roads captured
        if len(analyzer.failed_road_ids) > 0:
            capture_rate = (len(failed_roads_selected) / len(analyzer.failed_road_ids)) * 100
            print(f"  - Failed road capture rate: {capture_rate:.1f}% ({len(failed_roads_selected)}/{len(analyzer.failed_road_ids)})")
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n⏱️  Total execution time before error: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nError occurred: {str(e)}", file=sys.stderr)
        traceback.print_exc()

if __name__ == '__main__':
    main()