import json
import logging
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

class ComparisonAnalysisMixin:
    """Mixin class for comparison analysis operations"""
    
    def compare_prioritization_approaches(self, selected_roads: List[str], 
                                        approaches: List[str] = None,
                                        fkn_benchmark: Dict = None) -> Dict[str, Any]:
        """
        Compare prioritization approaches and analyze their differences.
        For accurate analysis, prioritizes ALL roads and analyzes top 10 performance.
        
        Args:
            selected_roads: List of selected road IDs (used for context, but all roads are prioritized)
            approaches: List of approaches to compare. If None, compares hybrid and random.
            fkn_benchmark: Dictionary containing F*K/N benchmark calculation results
            
        Returns:
            Dictionary containing comparison results and analysis
        """
        if approaches is None:
            approaches = ["hybrid", "random"]
        
        logger.info(f"Comparing {len(approaches)} prioritization approaches...")
        
        # Get all available roads for comprehensive analysis
        all_roads = list(self.road_metadata.keys())
        
        # Run each approach on ALL roads
        results = {}
        for approach in approaches:
            logger.info(f"Running {approach} approach...")
            try:
                # Prioritize ALL roads, not just selected ones, for proper top 10 analysis
                prioritized_roads = self.prioritize_selected_roads(selected_roads, method=approach, include_unselected=True)
                results[approach] = prioritized_roads
            except Exception as e:
                logger.error(f"Error running {approach} approach: {e}")
                results[approach] = []
        
        # Calculate F*K/N probability benchmark
        fkn_benchmark = self._calculate_fkn_probability_score(selected_roads, all_roads)
        
        # Analyze comparison
        comparison_analysis = self._analyze_prioritization_comparison(results, fkn_benchmark, selected_roads)
        
        # Save comparison results
        self._save_prioritization_comparison(results, comparison_analysis)
        
        return {
            'approaches_compared': approaches,
            'prioritization_results': results,
            'comparison_analysis': comparison_analysis
        }

    def _analyze_prioritization_comparison(self, results: Dict[str, List[Dict]], 
                                         fkn_benchmark: Dict[str, Any], 
                                         selected_roads: List[str]) -> Dict[str, Any]:
        """
        Analyze the differences between prioritization approaches and compare against F*K/N benchmark.
        """
        analysis = {
            'approach_statistics': {},
            'failed_road_performance': {},
            'fkn_probability_benchmark': fkn_benchmark,
            'top_10_overlap': {},
            'ranking_correlations': {},
            'priority_class_distribution': {}
        }
        
        # Basic statistics for each approach
        for approach, road_list in results.items():
            if not road_list:
                continue
                
            scores = [r['priority_score'] for r in road_list]
            failed_roads = [r for r in road_list if r.get('is_failed_road', False)]
            
            analysis['approach_statistics'][approach] = {
                'total_roads': len(road_list),
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'std_score': np.std(scores) if len(scores) > 1 else 0,
                'failed_roads_count': len(failed_roads),
                'failed_roads_avg_score': sum(r['priority_score'] for r in failed_roads) / len(failed_roads) if failed_roads else 0
            }
            
            # Priority class distribution
            high_count = len([r for r in road_list if r.get('priority_class') == 'HIGH'])
            medium_count = len([r for r in road_list if r.get('priority_class') == 'MEDIUM'])
            low_count = len([r for r in road_list if r.get('priority_class') == 'LOW'])
            
            analysis['priority_class_distribution'][approach] = {
                'HIGH': high_count,
                'MEDIUM': medium_count,
                'LOW': low_count,
                'HIGH_percentage': high_count / len(road_list) * 100 if road_list else 0
            }
        
        # Failed road performance analysis
        # Use the original failed_road_ids set instead of collecting from results to ensure consistency
        # Convert to strings to ensure type consistency
        failed_road_ids = set(str(road_id) for road_id in self.failed_road_ids)
        selected_roads_set = set(selected_roads)
        
        for approach, road_list in results.items():
            if not road_list:
                continue
                
            # Calculate failed road metrics directly without intermediate variables
            failed_roads_in_approach = [r for r in road_list if str(r['road_id']) in failed_road_ids]
            
            # Calculate failed roads in top K directly for analysis
            failed_roads_top_10_count = len([r for r in road_list[:10] if str(r['road_id']) in failed_road_ids])
            failed_roads_top_20_count = len([r for r in road_list[:20] if str(r['road_id']) in failed_road_ids])
            
            # Selected context: count failed roads in top K that are also in selected set
            failed_roads_top_10_selected_count = len([r for r in road_list[:10] if str(r['road_id']) in failed_road_ids and str(r['road_id']) in selected_roads_set])
            failed_roads_top_20_selected_count = len([r for r in road_list[:20] if str(r['road_id']) in failed_road_ids and str(r['road_id']) in selected_roads_set])
            
            # Calculate available failed roads in selected subset for percentage calculations
            available_failed_roads_in_selection = len([r_id for r_id in failed_road_ids if r_id in selected_roads_set])
            
            # Debug logging for verification
            if approach == 'random':
                top_20_road_ids = [r['road_id'] for r in road_list[:20]]
                failed_in_top_20_ids = [r['road_id'] for r in road_list[:20] if str(r['road_id']) in failed_road_ids]
                logger.info(f"Random approach debug - Top 20 roads: {top_20_road_ids}")
                logger.info(f"Random approach debug - Failed roads in top 20: {failed_in_top_20_ids}")
                logger.info(f"Random approach debug - Known failed road IDs: {list(failed_road_ids)}")
            
            analysis['failed_road_performance'][approach] = {
                # Selected roads context metrics (constrained to available roads)
                'failed_roads_in_top_10_selected_context': failed_roads_top_10_selected_count,
                'failed_roads_in_top_20_selected_context': failed_roads_top_20_selected_count,
                'available_failed_roads_in_selection': available_failed_roads_in_selection,
                'failed_roads_in_top_10_percentage_selected_context': failed_roads_top_10_selected_count / available_failed_roads_in_selection * 100 if available_failed_roads_in_selection > 0 else 0,
                
                # Additional context
                'note': f'Whole test suite analysis vs selected roads context (max possible failed roads in selection: {available_failed_roads_in_selection})'
            }
        
        # Fault detection analysis for multiple top-k values
        analysis['fault_detection_analysis'] = {}
        for approach, road_list in results.items():
            if not road_list:
                continue
            
            # Calculate fault detection for different top-k values
            total_failed = len(failed_road_ids)
            fault_detection_data = {}
            
            for k in [5, 10, 15, 20, 25]:
                if k <= len(road_list):
                    top_k_roads = road_list[:k]
                    failed_detected = len([r for r in top_k_roads if str(r['road_id']) in failed_road_ids])
                    
                    fault_detection_percentage = (failed_detected / total_failed) * 100 if total_failed > 0 else 0
                    detection_efficiency = (failed_detected / k) * 100 if k > 0 else 0
                    
                    fault_detection_data[f'top_{k}'] = {
                        'k_value': k,
                        'failed_roads_detected': failed_detected,
                        'total_failed_roads': total_failed,
                        'fault_detection_percentage': round(fault_detection_percentage, 2),
                        'detection_efficiency': round(detection_efficiency, 2),
                        'roads_analyzed': k
                    }
            
            # Add failed road positions for detailed analysis
            failed_road_positions = []
            for i, road_data in enumerate(road_list, 1):
                if str(road_data['road_id']) in failed_road_ids:
                    failed_road_positions.append({
                        'road_id': road_data['road_id'],
                        'position': i,
                        'priority_score': road_data['priority_score'],
                        'total_roads': len(road_list)
                    })
            
            analysis['fault_detection_analysis'][approach] = {
                'top_k_analysis': fault_detection_data,
                'failed_road_positions': failed_road_positions,
                'summary': {
                    'total_failed_roads': total_failed,
                    'average_failed_road_position': round(sum(pos['position'] for pos in failed_road_positions) / len(failed_road_positions), 1) if failed_road_positions else 0,
                    'best_failed_road_position': min(pos['position'] for pos in failed_road_positions) if failed_road_positions else 0,
                    'worst_failed_road_position': max(pos['position'] for pos in failed_road_positions) if failed_road_positions else 0
                }
            }
        
        # APFD (Average Percentage of Fault Detection) analysis
        analysis['apfd_analysis'] = {}
        analysis['apfd_selected_analysis'] = {}
        for approach, road_list in results.items():
            if not road_list:
                continue
            
            # Calculate APFD for this approach (whole test suite)
            apfd_result = self.calculate_apfd(road_list)
            analysis['apfd_analysis'][approach] = apfd_result
            
            # Calculate APFD for selected roads only
            apfd_selected_result = self.calculate_apfd_for_selected_roads(road_list, selected_roads)
            analysis['apfd_selected_analysis'][approach] = apfd_selected_result
        
        # Top 10 overlap analysis
        top_10_sets = {}
        for approach, road_list in results.items():
            if road_list:
                top_10_sets[approach] = set(r['road_id'] for r in road_list[:10])
        
        for approach1 in top_10_sets:
            analysis['top_10_overlap'][approach1] = {}
            for approach2 in top_10_sets:
                if approach1 != approach2:
                    overlap = len(top_10_sets[approach1] & top_10_sets[approach2])
                    analysis['top_10_overlap'][approach1][approach2] = {
                        'overlap_count': overlap,
                        'overlap_percentage': overlap / 10.0 * 100
                    }
        
        return analysis

    def _save_prioritization_comparison(self, results: Dict[str, List[Dict]], 
                                      analysis: Dict[str, Any]) -> None:
        """
        Save prioritization comparison results to files.
        """
        output_dir = self.paths.coverage_reduction_path / "prioritization_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results for each approach
        for approach, road_list in results.items():
            if road_list:
                approach_data = {
                    'analysis_metadata': {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'approach': approach,
                        'total_roads_analyzed': len(road_list)
                    },
                    'prioritized_roads': road_list
                }
                
                with open(output_dir / f'{approach}_prioritization.json', 'w') as f:
                    json.dump(approach_data, f, indent=2)
        
        # Save comparison analysis
        comparison_data = {
            'analysis_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'analysis_type': 'prioritization_comparison',
                'approaches_compared': list(results.keys())
            },
            'comparison_analysis': analysis
        }
        
        with open(output_dir / 'prioritization_comparison_analysis.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Prioritization comparison results saved to {output_dir}")

    def print_prioritization_comparison_summary(self, comparison_results: Dict[str, Any]) -> None:
        
        """
        Print a summary of the prioritization comparison results.
        """
        if not comparison_results:
            print("No comparison results available.")
            return
