import json
import csv
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
from collections import defaultdict
from data_classes import RoadSegment

logger = logging.getLogger(__name__)

class IOMixin:
    """Mixin class for input/output operations"""
    
    def load_failed_road_ids(self) -> set:
        """
        Load road IDs from csv file specifically.
        
        Returns:
            Set of road IDs that have failed in previous simulations
        """
        failed_ids = set()
        
        if not self.paths.failed_roads_path.exists():
            logger.warning(f"Failed roads directory not found: {self.paths.failed_roads_path}")
            return failed_ids
        
        try:
            # Extract the dataset identifier from dynamic_data_dir to match the CSV file
            # e.g., "./SensoDat/roads_dynamic_data/f" -> "f"
            dynamic_data_name = self.paths.dynamic_data_path.name
            csv_file = self.paths.failed_roads_path / f"{dynamic_data_name}.csv"
            
            # DEBUG: Log the CSV file path construction
            logger.info(f"🔧 CSV FILE DEBUG:")
            logger.info(f"   • dynamic_data_path.name: {dynamic_data_name}")
            logger.info(f"   • Constructed CSV path: {csv_file}")
            
            if not csv_file.exists():
                logger.warning(f"csv file not found: {csv_file}")
                return failed_ids
            
            logger.info(f"Loading failed road IDs from csv: {csv_file}")
            
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        road_id = str(row.get('road_id', '').strip())
                        if road_id:
                            failed_ids.add(road_id)
                
                logger.debug(f"Loaded failed road IDs from csv")
                
            except Exception as e:
                logger.warning(f"Error reading csv: {e}")
            
            logger.info(f"Loaded {len(failed_ids)} unique failed road IDs from csv: {sorted(list(failed_ids))}")
            
        except Exception as e:
            logger.error(f"Error loading failed road IDs: {e}")
        
        return failed_ids

    def load_road_points(self, filepath: Union[str, Path]) -> List[Tuple[float, float]]:
        """Load road points from JSON file with improved error handling"""
        path = Path(filepath) if isinstance(filepath, str) else filepath
        if not path.exists():
            raise FileNotFoundError(f"Road data file not found: {path}")
    
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return [(float(p[0]), float(p[1])) for p in data]
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.error(f"Error loading {path}: {e}")
            raise ValueError(f"Invalid road data format in {path}") from e

    def _register_section(self, road_id: str, section: RoadSegment) -> str:
        """Register a section and assign unique ID"""
        section_id = f"S{self.next_section_id}"
        self.next_section_id += 1
    
        self.section_registry[section_id] = {
            'road_id': road_id,
            'type': section.segment_type,
            'points': section.points,
            'curvature_profile': section.curvature_profile.tolist(),
            'length': section.length,
            'start_idx': section.start_index,
            'end_idx': section.end_index,
            'matched': section.matched
        }
    
        if road_id not in self.road_metadata:
            self.road_metadata[road_id] = []
        self.road_metadata[road_id].append(section_id)
    
        return section_id

    def process_roads_batch(self, road_files: List[Union[str, Path]]) -> Dict[str, Dict]:
        """Process multiple roads and return their data"""
        results = {}
        for filepath in road_files:
            path = Path(filepath) if isinstance(filepath, str) else filepath
            road_id = path.stem
        
            try:
                points = self.load_road_points(path)
                classification = self.classify_road_segments(points)
            
                # Register all sections
                for section in classification['segments']:
                    self._register_section(road_id, section)
            
                results[road_id] = {
                    'points': points,
                    'classification': classification,
                    'segments': classification['segments']
                }
            
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
            
        return results

    def save_section_registry(self, output_dir: Union[str, Path]) -> None:
        """Save the section registry to disk for later analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
        with open(output_path / 'section_registry.json', 'w') as f:
            json.dump(self.section_registry, f, indent=2)
    
        with open(output_path / 'road_metadata.json', 'w') as f:
            json.dump(self.road_metadata, f, indent=2)

    def save_matching_info(self, section_matches: Dict[str, Dict], output_dir: Union[str, Path]) -> None:
        """Save matched and unmatched section information to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
        # Separate matched and unmatched sections
        matched_sections = []
        unmatched_sections = []
    
        for section_id, data in self.section_registry.items():
            road_id = data['road_id']
            section_info = {
                'section_id': section_id,
                'road_id': road_id,
                'type': data['type'],
                'length': data['length'],
                'point_range': (data['start_idx'], data['end_idx']),
                'matched': data['matched']
            }
        
            if data['matched']:
                matched_sections.append(section_info)
            else:
                unmatched_sections.append(section_info)
    
        # Save to JSON files
        with open(output_path / 'matched_sections.json', 'w') as f:
            json.dump(matched_sections, f, indent=2)
    
        with open(output_path / 'unmatched_sections.json', 'w') as f:
            json.dump(unmatched_sections, f, indent=2)
    
        # Save detailed match information
        with open(output_path / 'section_matches.json', 'w') as f:
            json.dump(section_matches, f, indent=2)

    def save_coverage_reduction_results(self, results: Dict[str, Any],
                                      prioritized_roads: List[Dict[str, Any]] = None,
                                      execution_time: float = None) -> None:
        """
        Save comprehensive coverage reduction results to JSON files.
        
        Args:
            results: Coverage reduction results
            prioritized_roads: Prioritized roads list (optional)
            execution_time: Time taken for execution (optional)
        """
        # Use centralized path configuration
        output_dir = self.paths.coverage_reduction_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        summary = {
            'analysis_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'analysis_type': 'coverage_based_reduction',
                'algorithm': 'Hybrid_Agglomerative_clustering_with_greedy_selection',
                'total_execution_time_seconds': execution_time,
                'total_execution_time_minutes': round(execution_time / 60, 2) if execution_time else None,
                'clustering_cv': getattr(self, 'clustering_stats', {}).get('cv', None),
                'clustering_percentile': getattr(self, 'clustering_stats', {}).get('percentile', None),
                'clustering_threshold': getattr(self, 'clustering_stats', {}).get('threshold', None),
            },
            'reduction_statistics': {
                'total_roads_analyzed': results['total_roads'],
                'roads_selected_for_testing': results['selected_count'],
                'roads_eliminated': results['total_roads'] - results['selected_count'],
                'reduction_percentage': results['reduction_percentage'],
                'cluster_coverage_achieved': results['cluster_coverage'],
                'total_clusters_identified': results['total_clusters'],
                'coverage_percentage': results['coverage_percentage'],
                'failed_roads_loaded': len(self.failed_road_ids),
                'failed_roads_in_selection': len([r for r in results['selected_roads'] if r in self.failed_road_ids])
            },
            'selected_roads': results['selected_roads'],
            'failed_roads_detected': sorted(list(self.failed_road_ids)),
            'clustering_method': 'Hybrid Agglomerative Clustering (Geometric + Dynamic)',
            'coverage_threshold': 1.0
        }
        
        with open(output_dir / 'coverage_reduction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save cluster analysis
        cluster_analysis = {
            'analysis_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'analysis_type': 'cluster_analysis'
            },
            'cluster_statistics': {
                'total_clusters': results['total_clusters'],
                'clusters_covered': results['cluster_coverage'],
                'coverage_percentage': results['coverage_percentage']
            },
            'cluster_representatives': results['cluster_representatives'],
            'section_to_cluster_mapping': results['section_to_cluster']
        }
        
        with open(output_dir / 'cluster_analysis.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        # Save all prioritized roads (both selected and unselected) if provided
        if prioritized_roads:
            # Since prioritized_roads now only contains selected roads (due to include_unselected=False), 
            # save them directly to prioritized_roads_for_testing.json
            priority_data = {
                'analysis_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'analysis_type': 'road_prioritization',
                    'total_roads_analyzed': len(prioritized_roads)
                },
                'prioritized_roads': prioritized_roads,
                'priority_distribution': {
                    'HIGH': len([r for r in prioritized_roads if r['priority_class'] == 'HIGH']),
                    'MEDIUM': len([r for r in prioritized_roads if r['priority_class'] == 'MEDIUM']),
                    'LOW': len([r for r in prioritized_roads if r['priority_class'] == 'LOW'])
                },
                'selection_distribution': {
                    'selected': len(prioritized_roads),
                    'unselected': 0
                },
                'failed_roads_distribution': {
                    'total_failed_roads': len([r for r in prioritized_roads if r['is_failed_road']]),
                    'failed_roads_by_priority': {
                        'HIGH': len([r for r in prioritized_roads if r['priority_class'] == 'HIGH' and r['is_failed_road']]),
                        'MEDIUM': len([r for r in prioritized_roads if r['priority_class'] == 'MEDIUM' and r['is_failed_road']]),
                        'LOW': len([r for r in prioritized_roads if r['priority_class'] == 'LOW' and r['is_failed_road']])
                    },
                    'failed_roads_by_selection': {
                        'selected': len([r for r in prioritized_roads if r['is_failed_road']]),
                        'unselected': 0
                    },
                    'failed_road_ids': [r['road_id'] for r in prioritized_roads if r['is_failed_road']]
                }
            }
            
            with open(output_dir / 'prioritized_roads_for_testing.json', 'w') as f:
                json.dump(priority_data, f, indent=2)
        
        # Save dynamic analysis data if available
        if self.dynamic_data_available:
            dynamic_analysis = self.analyze_dynamic_coverage(results['selected_roads'])
            dynamic_data = {
                'analysis_metadata': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'analysis_type': 'dynamic_vehicle_analysis',
                    'dynamic_data_available': True
                },
                'dynamic_coverage_analysis': dynamic_analysis,
                'selected_roads_with_dynamic_data': [
                    {
                        'road_id': road_id,
                        'dynamic_score': dynamic_analysis['dynamic_scores'].get(road_id, 0.0)
                    }
                    for road_id in results['selected_roads']
                    if road_id in dynamic_analysis['dynamic_scores']
                ]
            }
            
            with open(output_dir / 'dynamic_analysis.json', 'w') as f:
                json.dump(dynamic_data, f, indent=2)
            
            logger.info(f"Dynamic analysis data saved to {output_dir}/dynamic_analysis.json")
        else:
            logger.info("No dynamic data available - skipping dynamic analysis")
        
        logger.info(f"Results saved to {output_dir}")