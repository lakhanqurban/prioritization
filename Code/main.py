import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import List, Dict, Tuple, Optional, Union, Any
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import refactored modules
from path_config import PathConfig
from data_classes import DynamicMetrics, RoadSegment
from gpu_operations import GPUOperationsMixin, GPU_AVAILABLE, CUDA_GPU_AVAILABLE, APPLE_GPU_AVAILABLE
from clustering import ClusteringMixin
from section_analysis import SectionAnalysisMixin
from prioritization import PrioritizationMixin
from comparison_analysis import ComparisonAnalysisMixin
from coverage_reduction import CoverageReductionMixin
from io_operations import IOMixin
from dynamic_analysis import DynamicAnalysisMixin

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
    GPUOperationsMixin
):
    """Main class for road analysis and comparison"""

    def __init__(self,
                curvature_threshold: float = 0.015,
                min_segment_length: int = 5,
                hysteresis_window: int = 3,
                min_subsection_length: int = 5,
                dtw_similarity_threshold: float = 0.95,
                path_config: PathConfig = None,
                use_gpu: bool = True):

        self.curvature_threshold = curvature_threshold
        self.min_segment_length = min_segment_length
        self.hysteresis_window = hysteresis_window
        self.min_subsection_length = min_subsection_length
        self.dtw_similarity_threshold = dtw_similarity_threshold
        self.section_registry = {}
        self.next_section_id = 1
        self.road_metadata = {}
        
        # GPU acceleration setup
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            if CUDA_GPU_AVAILABLE:
                logger.info("🚀 Enabling CUDA GPU acceleration")
                self.gpu_type = "CUDA"
            elif APPLE_GPU_AVAILABLE:
                logger.info("🍎 Enabling Apple Silicon GPU acceleration")
                self.gpu_type = "MPS"
        else:
            logger.info("💻 Using CPU computation")
            self.gpu_type = "CPU"
        
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

    def visualize_curved_clusters(self, section_to_cluster: Dict[str, int],
                            method: str = 'pca',
                            title: str = "Curved Section Clusters",
                            ):
        curved_sections = []
        cluster_ids = []

        for section_id, cluster_id in section_to_cluster.items():
            features = self._get_section_features(section_id)
            if features['type'] in ['left_curve', 'right_curve']:
                curved_sections.append(section_id)
                cluster_ids.append(cluster_id)

        if len(curved_sections) < 2:
            logger.info("Not enough curved sections for visualization")
            return

        cluster_counts = defaultdict(int)
        for cluster_id in cluster_ids:
            cluster_counts[cluster_id] += 1

        meaningful_clusters = {cid for cid, count in cluster_counts.items() if count > 1}
        filtered_data = [(sid, cid) for sid, cid in zip(curved_sections, cluster_ids)
                        if cid in meaningful_clusters]

        if not filtered_data:
            logger.info("No meaningful curved clusters found to visualize")
            return

        section_ids, cluster_ids = zip(*filtered_data)
        profiles = [np.array(self.section_registry[sid]['curvature_profile'])
                   for sid in section_ids]

        features = []
        for profile in profiles:
            features.append([
                np.mean(profile),
                np.std(profile),
                np.max(profile),
                np.min(profile),
                len(profile)
            ])

        features = np.array(features)

        if method.lower() == 'tsne':
            perplexity = min(30, max(5, len(profiles) // 3))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            reducer = PCA(n_components=2)

        embeddings = reducer.fit_transform(features)

        plt.figure(figsize=(12, 8))

        unique_clusters = sorted(set(cluster_ids))
        colors = plt.colormaps.get_cmap('tab20')

        logger.info(f"Visualizing {len(unique_clusters)} clusters with {len(section_ids)} total sections")
        logger.info(f"method: {method}")

        for i, cluster_id in enumerate(unique_clusters):
            mask = np.array(cluster_ids) == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_sections = np.array(section_ids)[mask]

            cluster_mean_curvatures = [np.mean(self.section_registry[sid]['curvature_profile'])
                                     for sid in cluster_sections]
            cluster_mean = np.mean(cluster_mean_curvatures)
            cluster_std = np.std(cluster_mean_curvatures)
            plt.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1],
                       c=[colors(i)], label=f'Cluster {cluster_id} ({len(cluster_sections)} sections)',
                       alpha=0.7, s=60)

        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.close()

    def print_detailed_cluster_info(self, section_matches: Dict[str, Dict] = None,
                                   section_to_cluster: Dict[str, int] = None) -> None:
        print("\n" + "="*80)
        print("🔍 DETAILED CLUSTER FORMATION ANALYSIS")
        print("="*80)

        if section_matches is None:
            print("📊 Computing section matches...")
            section_matches = self.compare_all_sections()

        if section_to_cluster is None:
            print("🔗 Applying Agglomerative Clustering...")
            section_to_cluster = self._agglomerative_clustering(section_matches)

        cluster_to_sections = {}
        for section_id, cluster_id in section_to_cluster.items():
            if cluster_id not in cluster_to_sections:
                cluster_to_sections[cluster_id] = []
            cluster_to_sections[cluster_id].append(section_id)

        straight_clusters = []
        left_curve_clusters = []
        right_curve_clusters = []
        individual_clusters = []

        for cluster_id, sections in cluster_to_sections.items():
            if len(sections) == 1:
                section_id = sections[0]
                section_data = self.section_registry[section_id]
                if not section_data.get('matched', False):
                    individual_clusters.append((cluster_id, section_id))
                else:
                    features = self._get_section_features(section_id)
                    section_type = features['type']

                    if section_type == 'straight':
                        straight_clusters.append((cluster_id, sections))
                    elif section_type == 'left_curve':
                        left_curve_clusters.append((cluster_id, sections))
                    elif section_type == 'right_curve':
                        right_curve_clusters.append((cluster_id, sections))
            else:
                first_section = sections[0]
                features = self._get_section_features(first_section)
                section_type = features['type']

                if section_type == 'straight':
                    straight_clusters.append((cluster_id, sections))
                elif section_type == 'left_curve':
                    left_curve_clusters.append((cluster_id, sections))
                elif section_type == 'right_curve':
                    right_curve_clusters.append((cluster_id, sections))

        print(f"\n🎯 CLUSTERING SUMMARY:")
        print(f"   • Straight section clusters: {len(straight_clusters)}")
        print(f"   • Left curve clusters: {len(left_curve_clusters)}")
        print(f"   • Right curve clusters: {len(right_curve_clusters)}")
        print(f"   • Individual unmatched sections: {len(individual_clusters)}")
        print(f"   • Total clusters formed: {len(cluster_to_sections)}")

        if straight_clusters:
            print(f"\n🟢 STRAIGHT SECTION CLUSTERS ({len(straight_clusters)} clusters):")
            for cluster_id, sections in straight_clusters:
                print(f"\n   Cluster {cluster_id}: {len(sections)} straight sections")
                example_sections = sections[:5]
                print(f"   └─ Example sections ({len(example_sections)}/{len(sections)}):")
                for section_id in example_sections:
                    try:
                        features = self._get_section_features(section_id)
                        data = self.section_registry[section_id]
                        road_id = data['road_id']
                        print(f"      • {section_id} (Road {road_id}): length={features['length']:.1f}m")
                    except Exception as e:
                        print(f"      • {section_id}: Error getting details - {e}")
                if len(sections) > 5:
                    print(f"      ... and {len(sections) - 5} more sections")

        if left_curve_clusters:
            print(f"\n🔵 LEFT CURVE CLUSTERS ({len(left_curve_clusters)} clusters):")
            for i, (cluster_id, sections) in enumerate(left_curve_clusters):
                print(f"\n   Cluster {cluster_id}: {len(sections)} left curve sections")
                print(f"   └─ Section composition:")
                self._print_cluster_sections(sections, max_shown=15)

        if right_curve_clusters:
            print(f"\n🔴 RIGHT CURVE CLUSTERS ({len(right_curve_clusters)} clusters):")
            for i, (cluster_id, sections) in enumerate(right_curve_clusters):
                print(f"\n   Cluster {cluster_id}: {len(sections)} right curve sections")
                print(f"   └─ Section composition:")
                self._print_cluster_sections(sections, max_shown=15)

        if individual_clusters:
            print(f"\n⚪ INDIVIDUAL UNMATCHED SECTIONS ({len(individual_clusters)} clusters):")
            individual_by_type = {'left_curve': [], 'right_curve': []}
            for cluster_id, section_id in individual_clusters:
                features = self._get_section_features(section_id)
                section_type = features['type']
                if section_type in individual_by_type:
                    individual_by_type[section_type].append((cluster_id, section_id))

            for curve_type, sections_list in individual_by_type.items():
                if sections_list:
                    print(f"\n   {curve_type.replace('_', ' ').title()} ({len(sections_list)} sections):")
                    for cluster_id, section_id in sections_list[:5]:
                        print(f"   └─ Cluster {cluster_id}: {section_id}")
                        try:
                            features = self._get_section_features(section_id)
                            data = self.section_registry[section_id]
                            road_id = data['road_id']
                            print(f"      Details: Road {road_id}, length={features['length']:.1f}m, "
                                  f"curvature={features['mean_curvature']:.4f}")
                        except Exception as e:
                            print(f"      Details: Error getting section info - {e}")
                    if len(sections_list) > 5:
                        print(f"   ... and {len(sections_list) - 5} more individual {curve_type} sections")

        print(f"\n" + "="*60)
        print("🔧 HYBRID AGGLOMERATIVE CLUSTERING ALGORITHM SUMMARY")
        print("="*60)
        print(f"   └─ {len(cluster_to_sections)} total clusters formed")

        cluster_sizes = [len(sections) for sections in cluster_to_sections.values()]
        size_distribution = {}
        for size in cluster_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1

        print(f"\n📊 CLUSTER SIZE DISTRIBUTION:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"   └─ Size {size}: {count} clusters")

        print("="*80)

    def _print_cluster_sections(self, sections: List[str], max_shown: int = 50) -> None:
        shown_sections = sections[:max_shown]
        for section_id in shown_sections:
            try:
                features = self._get_section_features(section_id)
                data = self.section_registry[section_id]
                road_id = data['road_id']
                print(f"      • {section_id} (Road {road_id}): length={features['length']:.1f}m, "
                      f"curvature={features['mean_curvature']:.4f}")
            except Exception as e:
                print(f"      • {section_id}: Error getting details - {e}")

        if len(sections) > max_shown:
            print(f"      ... and {len(sections) - max_shown} more sections")

def main():
    """Main function focused on all-section comparison and coverage-based reduction"""
    start_time = time.time()
    try:
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger(__name__)

        # Initialize analyzer with optimized parameters and dynamic data support
        analyzer = RoadAnalyzer(
            use_gpu=True  # Enable GPU acceleration
        )
        
        # Check GPU performance
        print("🔧 GPU Acceleration Setup")
        print("="*50)
        gpu_info = analyzer.check_gpu_performance()
        if gpu_info['gpu_available']:
            perf = gpu_info['performance_comparison']
            print(f"GPU Type: {gpu_info['gpu_type']}")
            print(f"CPU Time: {perf['cpu_time']:.4f}s")
            print(f"GPU Time: {perf['gpu_time']:.4f}s") 
            print(f"Speedup: {perf['speedup']:.2f}x")
        else:
            print("No GPU acceleration available")
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

        # Coverage-based reduction with PURE DYNAMIC clustering and analysis
        print("⚡ Applying coverage-based road reduction with PURE DYNAMIC clustering (dynamic weight = 1.0)...")
        reduction_results = analyzer.coverage_based_road_reduction(
            section_matches,
            coverage_threshold=1.0,
            include_dynamic_analysis=True,
            curvature_similarity_threshold=None,
            enable_dynamic_clustering=True,
            dynamic_weight=0.5  # 50% dynamic, 50% geometric
        )
        analyzer.print_detailed_cluster_info(section_matches, reduction_results['section_to_cluster'])
        print("🎯 Prioritizing selected roads...")
        prioritized_roads = analyzer.prioritize_selected_roads(reduction_results['selected_roads'], include_unselected=False)
        
        # NEW: Compare different prioritization approaches
        print("📊 Comparing prioritization approaches...")
        
        # Calculate F*K/N benchmark for comparison
        fkn_benchmark = analyzer._calculate_fkn_probability_score(
            reduction_results['selected_roads'],
            list(analyzer.road_metadata.keys())
        )
        
        comparison_results = analyzer.compare_prioritization_approaches(
            reduction_results['selected_roads'],
            approaches=["hybrid", "random"],
            fkn_benchmark=fkn_benchmark
        )
        
        # Print comparison summary
        analyzer.print_prioritization_comparison_summary(comparison_results)
        print("💾 Saving comprehensive analysis results...")
        current_time = time.time()
        execution_time = current_time - start_time
        analyzer.save_coverage_reduction_results(reduction_results, prioritized_roads, execution_time=execution_time)

        print(f"\n=== FINAL ANALYSIS RESULTS ===")
        print(f"Total sections analyzed: {len(analyzer.section_registry)}")
        print(f"Sections with DTW matches: {len(section_matches)}")
        print(f"Straight sections (total): {len(straight_sections)} {section_source_note}")
        print(f"All unmatched sections: {len(all_unmatched_sections)} {section_source_note}")
        print(f"  - Unmatched straight sections: {len(unmatched_straight_sections)} {section_source_note}")
        print(f"  - Unmatched curved sections: {len(unmatched_curved_sections)} {section_source_note}")
        print(f"")
        print(f"Coverage-Based Reduction:")
        total_roads = reduction_results.get('total_roads', 0)
        selected_count = reduction_results.get('selected_count', 0)
        if total_roads > 0:
            reduction_percentage = (1 - selected_count / total_roads) * 100
            print(f"  - Original roads: {total_roads}")
            print(f"  - Selected roads: {selected_count}")
            print(f"  - Reduction: {reduction_percentage:.1f}%")
        else:
            print(f"  - No roads found in metadata. Skipping reduction percentage calculation.")
        print(f"")
        print(f"Top Priority Roads (HYBRID Approach):")
        for i, road in enumerate(prioritized_roads[:20], 1):
            failed_indicator = " 🔴" if road.get('is_failed_road', False) else ""
            print(f"  {i}. Road {road['road_id']}: {road.get('priority_class', 'N/A')} priority (score: {road.get('priority_score', 'N/A')}){failed_indicator}")
        
        # Compare different road selection approaches on the FULL dataset
        print(f"\n" + "="*80)
        print("🔄 COMPARISON: Different Road Selection Approaches")
        print("="*80)
        
        all_available_roads = list(analyzer.road_metadata.keys())
        target_selection_size = len(reduction_results['selected_roads'])
        
        print(f"Comparing selection of {target_selection_size} roads from {len(all_available_roads)} total roads:\n")
        
        # Random Selection from full dataset
        print(f"\n2️⃣ RANDOM Selection from full dataset:")
        print(f"   Method: Random sampling of {target_selection_size} roads from all {len(all_available_roads)}")
        random.seed(42)
        random_roads = set(random.sample(all_available_roads, target_selection_size))
        for i, road_id in enumerate(sorted(random_roads), 1):
            failed_indicator = " 🔴" if road_id in analyzer.failed_road_ids else ""
            print(f"     {i}. Road {road_id}{failed_indicator}")
        
        # F*K/N Probability Benchmark
        print(f"\n3️⃣ F*K/N PROBABILITY BENCHMARK:")
        fkn_benchmark = analyzer._calculate_fkn_probability_score(
            list(random_roads),
            all_available_roads
        )
        print(f"   F*K/N Parameters: F={fkn_benchmark['failed_roads']}, K={fkn_benchmark['selected_roads']}, N={fkn_benchmark['total_roads']}")
        print(f"   Expected failures: {fkn_benchmark['expected_failing_tests']}")
        print(f"   Formula: {fkn_benchmark['formula']}")
        
        # Selection Overlap Analysis
        print(f"\n📊 SELECTION OVERLAP ANALYSIS:")
        random_vs_hybrid = len(random_roads & set(reduction_results['selected_roads']))
        random_hybrid_pct = random_vs_hybrid / min(len(random_roads), len(reduction_results['selected_roads'])) * 100
        print(f"   • Random vs Hybrid overlap: {random_vs_hybrid} roads ({random_hybrid_pct:.1f}% of smaller set)")
        
        # Failed road capture analysis
        failed_in_random = len([r for r in random_roads if r in analyzer.failed_road_ids])
        failed_in_hybrid = len([r for r in reduction_results['selected_roads'] if r in analyzer.failed_road_ids])
        
        print(f"\n🔴 FAILED ROAD CAPTURE ANALYSIS:")
        print(f"   Total failed roads available: {len(analyzer.failed_road_ids)}")
        print(f"   • Random approach captured: {failed_in_random}/{len(analyzer.failed_road_ids)} failed roads")
        print(f"   • Hybrid approach captured: {failed_in_hybrid}/{len(analyzer.failed_road_ids)} failed roads")
        print(f"   • F*K/N benchmark: {fkn_benchmark['expected_failing_tests']:.2f} expected failed roads")
        
        if failed_in_hybrid > failed_in_random:
            print(f"   🏆 Hybrid approach shows better failed road capture!")
        elif failed_in_random > failed_in_hybrid:
            print(f"   🏆 Random selection shows better failed road capture!")
        else:
            print(f"   🤝 Random and Hybrid approaches show equal failed road capture!")
        
        print("="*80)
        
        # Show failed roads summary
        selected_roads_set = set(reduction_results['selected_roads'])
        failed_roads_in_selection = [r for r in prioritized_roads if r['road_id'] in selected_roads_set and r.get('is_failed_road', False)]
        print(f"\nFailed Roads Analysis:")
        print(f"  - Total failed roads loaded from CSV files: {len(analyzer.failed_road_ids)}")
        print(f"  - Failed roads in selected set: {len(failed_roads_in_selection)}")
        if failed_roads_in_selection:
            print(f"  - Failed road IDs selected: {[r['road_id'] for r in failed_roads_in_selection]}")
            print(f"  - Average priority score for failed roads: {sum(r['priority_score'] for r in failed_roads_in_selection) / len(failed_roads_in_selection):.3f}")
        
        print(f"================================")
        print(f"\nUnmatched curved sections ({len(unmatched_curved_sections)}): {section_source_note}")
        if len(unmatched_curved_sections) > 0:
            for sid in unmatched_curved_sections[:10]:
                if use_saved_match_info:
                    sec = next((s for s in unmatched_sections_data if s['section_id'] == sid), None)
                    road_id = sec['road_id'] if sec else 'N/A'
                    section_type = sec['type'] if sec else 'N/A'
                else:
                    road_id = analyzer.section_registry[sid]['road_id']
                    section_type = analyzer._get_section_features(sid)['type']
                print(f"  Sec ID: {sid}, Road ID: {road_id}, Type: {section_type}")
            if len(unmatched_curved_sections) > 10:
                print(f"  ... and {len(unmatched_curved_sections) - 10} more")
        else:
            print("  No unmatched curved sections - all unique curved patterns are covered!")
        if len(unmatched_straight_sections) > 0:
            print(f"\nNote: {len(unmatched_straight_sections)} unmatched straight sections exist but are covered")
            print(f"      through cluster representation (all straight sections form one cluster).")
        try:
            analyzer.visualize_curved_clusters(
                section_to_cluster=reduction_results['section_to_cluster'],
                method='pca',
                title="Clustering Results Visualization",
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
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