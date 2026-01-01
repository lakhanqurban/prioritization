import random
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PrioritizationMixin:
    """Mixin class for road prioritization operations"""
    
    def prioritize_selected_roads(self, selected_roads: List[str], method: str = "hybrid", include_unselected: bool = True) -> List[Dict[str, Any]]:
        """
        Prioritize roads using specified method, optionally including unselected roads with lower priority.
        
        Args:
            selected_roads: List of road IDs selected for primary testing
            method: Prioritization method - "hybrid", "random"
            include_unselected: If True, include unselected roads in prioritization with lower priority
            
        Returns:
            List of prioritized roads with scores and classifications, selected roads having higher priority
        """
        # Get all available roads if including unselected
        all_roads = None
        if include_unselected:
            all_roads = list(self.road_metadata.keys())
            logger.info(f"Including all {len(all_roads)} roads in prioritization (selected roads will have higher priority)")
        
        logger.info(f"Prioritizing roads using '{method}' approach...")
        
        if method == "hybrid":
            return self._prioritize_hybrid_approach(selected_roads, all_roads)
        elif method == "random":
            return self._prioritize_random_baseline(selected_roads, all_roads)
        else:
            logger.warning(f"Unknown method '{method}', falling back to hybrid approach")
            return self._prioritize_hybrid_approach(selected_roads, all_roads)

    def _prioritize_hybrid_approach(self, selected_roads: List[str], all_roads: List[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid approach that handles both selected and unselected roads.
        
        Args:
            selected_roads: List of road IDs that were selected for primary testing
            all_roads: Optional list of all available road IDs. If provided, will prioritize
                      unselected roads as well with lower priority scores.
        
        Returns:
            List of dictionaries containing road priorities, with selected roads having
            higher priority than unselected ones.
        """
        road_priorities = []
        
        # Process all available roads if provided, otherwise just selected roads
        roads_to_process = all_roads if all_roads is not None else selected_roads
        
        for road_id in roads_to_process:
            road_sections = self.road_metadata.get(road_id, [])
            if not road_sections:
                continue
            
            # Calculate priority metrics
            metrics = self._calculate_road_priority_metrics(road_sections)
            
            # Calculate base priority score
            base_priority_score = (
                min(metrics['curvature_variation'] * 3.0, 0.3) +     # Curvature variation (capped at 0.3)
                min(metrics['critical_scenarios'] * 0.1, 0.3) +      # Critical scenarios (capped at 0.3)
                min(metrics['unique_patterns'] * 0.1, 0.3) +         # Unique patterns (capped at 0.3)
                min(metrics['total_length'] / 1000.0, 0.1)           # Length bonus (capped at 0.1)
            )
            
            # Apply selection status modifier
            # Selected roads keep their original score, unselected roads get reduced priority
            is_selected = road_id in selected_roads
            if not is_selected:
                base_priority_score *= 0.3  # Reduce priority for unselected roads
            
            # Check if this road was previously failed
            is_failed_road = road_id in self.failed_road_ids
            
            # Apply failed road bonus for hybrid approach (same as in selection process)
            failed_road_bonus = 0.25 if is_failed_road else 0.0
            priority_score = base_priority_score + failed_road_bonus
            
            # Classify priority level based on final score (including failed road bonus)
            if priority_score >= 0.6:
                priority_class = "HIGH"
            elif priority_score >= 0.3:
                priority_class = "MEDIUM"
            else:
                priority_class = "LOW"
            
            road_priorities.append({
                'road_id': road_id,
                'priority_score': round(priority_score, 3),
                'priority_class': priority_class,
                'is_failed_road': is_failed_road,
                'approach': 'hybrid',
                'failed_road_bonus_applied': is_failed_road,
                'metrics': {
                    'curvature_variation': round(metrics['curvature_variation'], 3),
                    'critical_scenarios': metrics['critical_scenarios'],
                    'unique_patterns': metrics['unique_patterns'],
                    'rare_patterns': 0,  # Could be enhanced
                    'total_length': round(metrics['total_length'], 1),
                    'section_count': len(road_sections),
                    'base_score': round(base_priority_score, 3),
                    'failed_road_bonus': round(failed_road_bonus, 3),
                    'final_score': round(priority_score, 3)
                }
            })
        
        # Sort by priority score (descending)
        road_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Count failed roads in selected set
        failed_roads_count = len([r for r in road_priorities if r['is_failed_road']])
        
        logger.info(f"Prioritized {len(road_priorities)} roads with hybrid approach")
        logger.info(f"  - Roads with failed simulation history: {failed_roads_count}")
        if failed_roads_count > 0:
            failed_road_ids = [r['road_id'] for r in road_priorities if r['is_failed_road']]
            logger.info(f"  - Failed roads in selection: {failed_road_ids}")
        
        return road_priorities

    def _prioritize_random_baseline(self, selected_roads: List[str], all_roads: List[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced random baseline that supports both selected and unselected roads.
        
        Args:
            selected_roads: List of road IDs selected for primary testing
            all_roads: Optional list of all available road IDs. If provided, will include
                      unselected roads with lower priority
        """
        import random
        
        road_priorities = []
        random.seed(42)  # For reproducible results - same seed as main function
        
        # Use all roads if provided, otherwise just selected roads
        roads_to_process = all_roads if all_roads is not None else selected_roads
        
        # Simulate the exact same logic as the main function to ensure consistency
        # Main function does: random.sample(all_available_roads, target_selection_size)
        # then sorts the selected roads, so we need to replicate this exactly
        
        if all_roads is not None and len(selected_roads) < len(roads_to_process):
            # This is the comparison case - simulate main function's random.sample() approach
            target_selection_size = len(selected_roads)  # Usually 39
            random_selected_roads = set(random.sample(roads_to_process, target_selection_size))
            
            # Give highest priority to the randomly selected roads (in sorted order)
            sorted_selected = sorted(random_selected_roads)
            
            # Process all roads, but give higher scores to the randomly selected ones
            for road_id in roads_to_process:
                road_sections = self.road_metadata.get(road_id, [])
                if not road_sections:
                    continue
                
                if road_id in random_selected_roads:
                    # This road was randomly selected - assign high priority based on sorted position
                    position_in_selection = sorted_selected.index(road_id)
                    # Higher priority for earlier positions in sorted list
                    priority_score = 1.0 - (position_in_selection / len(sorted_selected)) * 0.3  # 0.7 to 1.0 range
                else:
                    # This road was not selected - assign lower priority
                    priority_score = random.random() * 0.3  # 0.0 to 0.3 range
                
                # Classification based on score
                if priority_score >= 0.67:
                    priority_class = "HIGH"
                elif priority_score >= 0.33:
                    priority_class = "MEDIUM"
                else:
                    priority_class = "LOW"
                
                is_failed_road = road_id in self.failed_road_ids
                
                road_priorities.append({
                    'road_id': road_id,
                    'priority_score': round(priority_score, 3),
                    'priority_class': priority_class,
                    'is_failed_road': is_failed_road,
                    'approach': 'random',
                    'metrics': {
                        'random_seed': 42,
                        'randomly_selected': road_id in random_selected_roads,
                        'section_count': len(road_sections),
                        'failed_road_bonus_applied': False
                    }
                })
        else:
            # Simple case - just assign random priorities to all roads
            for road_id in roads_to_process:
                road_sections = self.road_metadata.get(road_id, [])
                if not road_sections:
                    continue
                
                priority_score = random.random()
                
                # Random classification
                if priority_score >= 0.67:
                    priority_class = "HIGH"
                elif priority_score >= 0.33:
                    priority_class = "MEDIUM"
                else:
                    priority_class = "LOW"
                
                is_failed_road = road_id in self.failed_road_ids
                
                road_priorities.append({
                    'road_id': road_id,
                    'priority_score': round(priority_score, 3),
                    'priority_class': priority_class,
                    'is_failed_road': is_failed_road,
                    'approach': 'random',
                    'metrics': {
                        'random_seed': 42,
                        'section_count': len(road_sections),
                        'failed_road_bonus_applied': False
                    }
                })
        
        # Sort by priority score (descending) - this will match the analysis expectations
        road_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Prioritized {len(road_priorities)} roads with random baseline")
        return road_priorities

    def _calculate_fkn_probability_score(self, selected_roads: List[str], all_roads: List[str] = None) -> Dict[str, Any]:
        """
        Calculate F*K/N probability score as a performance benchmark.
        
        This is NOT a road prioritization method, but a statistical benchmark calculation.
        F*K/N tells us the expected number of failed roads in a random selection.
        
        Formula: E[failing_tests] = (F * K) / N
        Where:
        - F = total number of failing tests (historically failed roads)
        - K = number of selected tests (current selection size)  
        - N = total number of available tests (total roads)
        
        Args:
            selected_roads: List of road IDs selected for primary testing
            all_roads: Optional list of all available road IDs for calculation
            
        Returns:
            Dictionary with F*K/N probability analysis and benchmark data
        """
        
        # Calculate statistical parameters for F*K/N formula
        total_roads = len(self.road_metadata)  # N - total available tests
        selected_count = len(selected_roads)   # K - selected tests count
        failed_count = len(self.failed_road_ids)  # F - historically failing tests
        
        # Calculate expected number of failing tests using F*K/N formula
        expected_failing_tests = (failed_count * selected_count) / total_roads if total_roads > 0 else 0
        
        # Calculate probability of failure for individual roads in random selection
        base_failure_probability = failed_count / total_roads if total_roads > 0 else 0
        
        # Calculate APFD percentage based on F*K/N expected performance
        # For top 10 analysis, calculate expected failed roads in top 10
        top_k_analysis = 10  # Standard top-k analysis size
        expected_failing_in_top_k = (failed_count * top_k_analysis) / total_roads if total_roads > 0 else 0
        
        # Calculate failed_roads_in_top_10_percentage based on F*K/N expectation
        failed_roads_in_top_10_percentage = (expected_failing_in_top_k / failed_count * 100) if failed_count > 0 else 0
        
        # logger.info(f"F*K/N Probability Benchmark:")
        # logger.info(f"  - Total roads (N): {total_roads}")
        # logger.info(f"  - Selected roads (K): {selected_count}")
        # logger.info(f"  - Failed roads (F): {failed_count}")
        # logger.info(f"  - Expected failing tests (F*K/N): {expected_failing_tests:.2f}")
        # logger.info(f"  - Expected failing in top 10: {expected_failing_in_top_k:.2f}")
        # logger.info(f"  - Base failure probability (F/N): {base_failure_probability:.4f}")
        # logger.info(f"  - Expected failed roads in top 10 percentage: {failed_roads_in_top_10_percentage:.1f}%")
        
        return {
            'approach': 'fkn_probability_benchmark',
            'total_roads': total_roads,
            'selected_roads': selected_count,
            'failed_roads': failed_count,
            'expected_failing_tests': round(expected_failing_tests, 3),
            'expected_failing_in_top_10': round(expected_failing_in_top_k, 3),
            'failed_roads_in_top_10_percentage': round(failed_roads_in_top_10_percentage, 2),
            'base_failure_probability': round(base_failure_probability, 4),
            'benchmark_note': 'F*K/N provides expected performance of random selection',
            'formula': f'({failed_count} * {selected_count}) / {total_roads} = {expected_failing_tests:.2f}',
            'top_10_formula': f'({failed_count} * {top_k_analysis}) / {total_roads} = {expected_failing_in_top_k:.2f}'
        }

    def calculate_fault_detection_percentage(self, prioritized_roads: List[Dict[str, Any]], 
                                           top_k: int = 10) -> Dict[str, Any]:
        """
        Calculate the average percentage of fault detection for a prioritized road list.
        
        This implements the fault detection analysis to see how many failed roads 
        are detected in the first K selected roads.
        
        Args:
            prioritized_roads: List of prioritized roads with 'is_failed_road' flag
            top_k: Number of top roads to analyze (default: 10)
            
        Returns:
            Dictionary with fault detection analysis results
        """
        
        if not prioritized_roads:
            return {
                'fault_detection_percentage': 0.0,
                'failed_roads_detected': 0,
                'total_failed_roads': 0,
                'roads_analyzed': 0,
                'analysis_note': 'No roads to analyze'
            }
        
        # Get total number of failed roads available
        total_failed_roads = len([r for r in prioritized_roads if r.get('is_failed_road', False)])
        
        # Get the first K roads (or all roads if fewer than K)
        top_roads = prioritized_roads[:min(top_k, len(prioritized_roads))]
        
        # Count failed roads in the top K selection
        failed_roads_detected = len([r for r in top_roads if r.get('is_failed_road', False)])
        
        # Calculate fault detection percentage
        if total_failed_roads > 0:
            fault_detection_percentage = (failed_roads_detected / total_failed_roads) * 100
        else:
            fault_detection_percentage = 0.0
        
        # Get the IDs of detected failed roads for detailed analysis
        detected_failed_road_ids = [r['road_id'] for r in top_roads if r.get('is_failed_road', False)]
        all_failed_road_ids = [r['road_id'] for r in prioritized_roads if r.get('is_failed_road', False)]
        
        return {
            'fault_detection_percentage': round(fault_detection_percentage, 2),
            'failed_roads_detected': failed_roads_detected,
            'total_failed_roads': total_failed_roads,
            'roads_analyzed': len(top_roads),
            'top_k': top_k,
            'detected_failed_road_ids': detected_failed_road_ids,
            'all_failed_road_ids': all_failed_road_ids,
            'detection_efficiency': round(failed_roads_detected / len(top_roads) * 100, 2) if top_roads else 0.0,
            'analysis_note': f'Analyzed first {len(top_roads)} roads out of {len(prioritized_roads)} total roads'
        }

    def calculate_apfd(self, prioritized_roads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate APFD (Average Percentage of Fault Detection) for a prioritized road list.
        
        APFD measures the weighted average of the percentage of faults detected during 
        the execution of the test suite. Higher APFD values indicate better prioritization.
        
        Formula: APFD = 1 - (TF1 + TF2 + ... + TFm) / (n * m) + 1/(2*n)
        Where:
        - TFi = position of first test that reveals fault i
        - n = total number of tests (roads)
        - m = total number of faults (failed roads)
        
        Args:
            prioritized_roads: List of prioritized roads with 'is_failed_road' flag
            
        Returns:
            Dictionary with APFD analysis results
        """
        if not prioritized_roads:
            return {
                'apfd': 0.0,
                'total_roads': 0,
                'total_failed_roads': 0,
                'failed_road_positions': [],
                'analysis_note': 'No roads to analyze'
            }
        
        # Get failed road positions
        failed_road_positions = []
        for i, road in enumerate(prioritized_roads, 1):  # 1-indexed position
            if road.get('is_failed_road', False):
                failed_road_positions.append({
                    'road_id': road['road_id'],
                    'position': i,
                    'priority_score': road.get('priority_score', 0.0)
                })
        
        n = len(prioritized_roads)  # Total number of tests
        m = len(failed_road_positions)  # Total number of faults
        
        if m == 0:
            # No failed roads found
            return {
                'apfd': 1.0,  # Perfect score when no faults exist
                'total_roads': n,
                'total_failed_roads': m,
                'failed_road_positions': [],
                'analysis_note': 'No failed roads found - perfect APFD score'
            }
        
        # Calculate sum of positions where faults are detected
        sum_positions = sum(pos['position'] for pos in failed_road_positions)
        
        # Calculate APFD using the standard formula
        apfd = 1 - (sum_positions / (n * m)) + (1 / (2 * n))
        
        return {
            'apfd': round(apfd, 4),
            'total_roads': n,
            'total_failed_roads': m,
            'failed_road_positions': failed_road_positions,
            'sum_positions': sum_positions,
            'average_failed_road_position': round(sum_positions / m, 2) if m > 0 else 0,
            'analysis_note': f'APFD calculated for {n} roads with {m} failed roads'
        }

    def calculate_apfd_for_selected_roads(self, prioritized_roads: List[Dict[str, Any]], 
                                        selected_roads: List[str]) -> Dict[str, Any]:
        """
        Calculate APFD (Average Percentage of Fault Detection) specifically for selected roads subset.
        
        This provides fault detection performance analysis within the context of the selected road subset,
        allowing comparison of how well different prioritization approaches perform on the reduced test suite.
        
        Args:
            prioritized_roads: Full list of prioritized roads with 'is_failed_road' flag
            selected_roads: List of road IDs that were selected for testing
            
        Returns:
            Dictionary with APFD analysis results for selected roads only
        """
        # Filter prioritized roads to only include selected roads, maintaining their relative order
        selected_roads_set = set(selected_roads)
        selected_prioritized_roads = [road for road in prioritized_roads 
                                    if road['road_id'] in selected_roads_set]
        
        if not selected_prioritized_roads:
            return {
                'apfd_selected': 0.0,
                'total_selected_roads': 0,
                'total_failed_roads_in_selection': 0,
                'failed_road_positions_in_selection': [],
                'analysis_note': 'No selected roads to analyze'
            }
        
        # Re-index positions for selected roads only (1-based indexing)
        failed_road_positions = []
        for i, road in enumerate(selected_prioritized_roads, 1):
            if road.get('is_failed_road', False):
                failed_road_positions.append({
                    'road_id': road['road_id'],
                    'position_in_selection': i,
                    'original_position': next(j+1 for j, r in enumerate(prioritized_roads) 
                                            if r['road_id'] == road['road_id']),
                    'priority_score': road.get('priority_score', 0.0)
                })
        
        n = len(selected_prioritized_roads)  # Total number of selected tests
        m = len(failed_road_positions)  # Total number of faults in selection
        
        if m == 0:
            # No failed roads found in selection
            return {
                'apfd_selected': 1.0,  # Perfect score when no faults exist in selection
                'total_selected_roads': n,
                'total_failed_roads_in_selection': m,
                'failed_road_positions_in_selection': [],
                'analysis_note': 'No failed roads found in selection - perfect selected APFD score'
            }
        
        # Calculate sum of positions where faults are detected (within selected roads)
        sum_positions = sum(pos['position_in_selection'] for pos in failed_road_positions)
        
        # Calculate APFD using the standard formula for selected roads
        apfd_selected = 1 - (sum_positions / (n * m)) + (1 / (2 * n))
        
        return {
            'apfd_selected': round(apfd_selected, 4),
            'total_selected_roads': n,
            'total_failed_roads_in_selection': m,
            'failed_road_positions_in_selection': failed_road_positions,
            'sum_positions_in_selection': sum_positions,
            'average_failed_road_position_in_selection': round(sum_positions / m, 2) if m > 0 else 0,
            'failed_roads_coverage_in_selection': round(m / len(self.failed_road_ids) * 100, 2) if self.failed_road_ids else 0,
            'analysis_note': f'Selected APFD calculated for {n} selected roads with {m} failed roads'
        }