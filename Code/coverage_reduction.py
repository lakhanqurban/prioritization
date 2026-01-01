import logging
import numpy as np
from typing import List, Dict, Set, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class CoverageReductionMixin:
    """Mixin class for coverage-based road reduction operations"""
    
    def coverage_based_road_reduction(self, section_matches: Dict[str, Dict] = None,
                                    coverage_threshold: float = 1.0,
                                    prioritize_unique_clusters: bool = False,
                                    include_dynamic_analysis: bool = True,
                                    **kwargs) -> Dict[str, Any]:
        """
        Apply coverage-based test reduction using hybrid agglomerative clustering on DTW-matched sections.
        
        Args:
            section_matches: Pre-computed section matches (optional)
            coverage_threshold: Minimum coverage fraction required (0.0 to 1.0)
            prioritize_unique_clusters: If True, ensures all unique clusters are covered first
            include_dynamic_analysis: Whether to include dynamic vehicle data in road selection
            **kwargs: Additional parameters for agglomerative clustering:
                     - curvature_similarity_threshold: Max curvature difference within cluster
                     - enable_dynamic_clustering: Whether to use hybrid clustering (default: True)
                     - dynamic_weight: Weight for dynamic features in clustering (default: 0.3)
            
        Returns:
            Dictionary containing reduction results and selected roads
        """
        # Get section matches if not provided
        if section_matches is None:
            section_matches = self.compare_all_sections()
        
        # Apply clustering (using agglomerative clustering)
        
        section_to_cluster = self._agglomerative_clustering(
            section_matches,
            curvature_similarity_threshold=kwargs.get('curvature_similarity_threshold', None),
            enable_dynamic_clustering=kwargs.get('enable_dynamic_clustering', True),
            dynamic_weight=kwargs.get('dynamic_weight', 0.4)
        )
        if section_to_cluster is None:
            logger.error("Clustering method returned None. No clusters formed.")
            section_to_cluster = {}
        
        # Select cluster representatives
        cluster_representatives = self.select_cluster_representatives(section_to_cluster, section_matches)
        
        # Build coverage requirements
        required_clusters = set(cluster_representatives.keys())
        required_coverage_count = max(1, int(coverage_threshold * len(required_clusters)))
        
        # Map roads to clusters they cover
        road_to_clusters = defaultdict(set)
        cluster_to_roads = defaultdict(set)
        for road_id, section_ids in self.road_metadata.items():
            for section_id in section_ids:
                if section_id in section_to_cluster:
                    cluster_id = section_to_cluster[section_id]
                    road_to_clusters[road_id].add(cluster_id)
                    cluster_to_roads[cluster_id].add(road_id)
        
        # Greedy coverage-based road selection
        selected_roads = set()
        covered_clusters = set()
        selection_details = []
        
        # Optional Phase 1: Handle unique clusters first
        if prioritize_unique_clusters:
            logger.info("Phase 1: Ensuring coverage of unique clusters...")
            
            # Find clusters that are only covered by one road (unique clusters)
            unique_clusters = {}
            for cluster_id, roads in cluster_to_roads.items():
                if len(roads) == 1:
                    unique_clusters[cluster_id] = list(roads)[0]
            
            logger.info(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")
            
            # Add roads that contain unique clusters
            for cluster_id, road_id in unique_clusters.items():
                if road_id not in selected_roads:
                    selected_roads.add(road_id)
                    new_clusters = road_to_clusters[road_id] - covered_clusters
                    covered_clusters.update(new_clusters)
                    
                    # Calculate priority score for unique cluster road
                    score = self._calculate_road_priority_score(
                        road_id, new_clusters, cluster_representatives, section_matches,
                        include_dynamic_analysis
                    )
                    
                    selection_details.append({
                        'road_id': road_id,
                        'new_clusters_covered': len(new_clusters),
                        'total_clusters_covered': len(covered_clusters),
                        'priority_score': score,
                        'coverage_percentage': len(covered_clusters) / len(required_clusters) * 100,
                        'selection_reason': f'Unique cluster {cluster_id} coverage'
                    })
                    
                    logger.info(f"Added road {road_id} for unique cluster {cluster_id} coverage")
        
        # Phase 2: Greedy selection for remaining clusters
        
        while len(covered_clusters) < required_coverage_count:
            best_road = None
            best_new_clusters = set()
            best_score = -1
            
            for road_id in self.road_metadata:
                if road_id in selected_roads:
                    continue
                
                new_clusters = road_to_clusters[road_id] - covered_clusters
                if not new_clusters:
                    continue
                
                # Calculate priority score
                score = self._calculate_road_priority_score(
                    road_id, new_clusters, cluster_representatives, section_matches,
                    include_dynamic_analysis
                )
                
                # Bonus for roads that cover multiple remaining clusters (if prioritizing unique clusters)
                if prioritize_unique_clusters and len(new_clusters) > 1:
                    score += len(new_clusters) * 0.5
                
                if score > best_score:
                    best_score = score
                    best_road = road_id
                    best_new_clusters = new_clusters
            
            if best_road is None:
                logger.warning("No more roads can improve coverage. Stopping selection.")
                break
            
            selected_roads.add(best_road)
            covered_clusters.update(best_new_clusters)
            
            selection_details.append({
                'road_id': best_road,
                'new_clusters_covered': len(best_new_clusters),
                'total_clusters_covered': len(covered_clusters),
                'priority_score': best_score,
                'coverage_percentage': len(covered_clusters) / len(required_clusters) * 100
            })
        
        # Calculate reduction metrics
        total_roads = len(self.road_metadata)
        reduction_percentage = (1 - len(selected_roads) / total_roads) * 100 if total_roads > 0 else 0
        
        logger.info(f"Coverage-based reduction complete:")
        logger.info(f"  - Original roads: {total_roads}")
        logger.info(f"  - Selected roads: {len(selected_roads)}")
        logger.info(f"  - Reduction: {reduction_percentage:.1f}%")
        logger.info(f"  - Cluster coverage: {len(covered_clusters)}/{len(required_clusters)}")
    
        return {
            'selected_roads': list(selected_roads),
            'total_roads': total_roads,
            'selected_count': len(selected_roads),
            'reduction_percentage': reduction_percentage,
            'cluster_coverage': len(covered_clusters),
            'total_clusters': len(required_clusters),
            'coverage_percentage': len(covered_clusters) / len(required_clusters) * 100,
            'selection_details': selection_details,
            'section_to_cluster': section_to_cluster,
            'cluster_representatives': cluster_representatives,
            'unique_clusters_handled': len([d for d in selection_details if 'Unique cluster' in d.get('selection_reason', '')]) if prioritize_unique_clusters else 0
        }

    def _analyze_coverage_completeness(self, selected_roads: set, road_to_clusters: dict,
                                     cluster_representatives: dict, section_to_cluster: dict) -> dict:
        """
        Analyze the completeness of coverage provided by selected roads.
        
        Args:
            selected_roads: Set of selected road IDs
            road_to_clusters: Mapping from road_id to clusters covered
            cluster_representatives: Representative sections for each cluster
            section_to_cluster: Mapping from section_id to cluster_id
            
        Returns:
            Dictionary with detailed coverage analysis
        """
        analysis = {
            'total_clusters': len(cluster_representatives),
            'covered_clusters': set(),
            'cluster_coverage_by_road': {},
            'section_type_coverage': {'straight': 0, 'left_curve': 0, 'right_curve': 0},
            'missing_section_types': []
        }
        
        # Analyze cluster coverage by each selected road
        for road_id in selected_roads:
            covered = road_to_clusters.get(road_id, set())
            analysis['covered_clusters'].update(covered)
            analysis['cluster_coverage_by_road'][road_id] = {
                'clusters': list(covered),
                'cluster_count': len(covered)
            }
        
        # Analyze section type coverage
        covered_section_types = set()
        for cluster_id in analysis['covered_clusters']:
            representatives = cluster_representatives.get(cluster_id, [])
            for section_id in representatives:
                features = self._get_section_features(section_id)
                section_type = features['type']
                covered_section_types.add(section_type)
                analysis['section_type_coverage'][section_type] += 1
        
        # Identify missing section types
        all_section_types = {'straight', 'left_curve', 'right_curve'}
        analysis['missing_section_types'] = list(all_section_types - covered_section_types)
        
        analysis['coverage_completeness'] = len(analysis['covered_clusters']) / analysis['total_clusters']
        analysis['section_type_completeness'] = len(covered_section_types) / len(all_section_types)
        
        return analysis

    def select_cluster_representatives(self, section_to_cluster: Dict[str, int],
                                     section_matches: Dict[str, Dict]) -> Dict[int, List[str]]:
        """
        Select representative sections from each cluster to ensure comprehensive pattern coverage.
        
        Args:
            section_to_cluster: Mapping from section_id to cluster_id
            section_matches: Section matching results
            
        Returns:
            Dictionary mapping cluster_id to list of representative section_ids
        """
        
        cluster_to_sections = defaultdict(list)
        for section_id, cluster_id in section_to_cluster.items():
            cluster_to_sections[cluster_id].append(section_id)
        
        cluster_representatives = {}
        
        for cluster_id, section_ids in cluster_to_sections.items():
            if len(section_ids) == 1:
                # Single section cluster - it's the representative
                cluster_representatives[cluster_id] = section_ids
            else:
                # Multiple sections - select representative(s) based on cluster characteristics
                first_section = section_ids[0]
                features = self._get_section_features(first_section)
                
                if features['type'] == 'straight':
                    # For straight sections, select one representative (they're all similar)
                    cluster_representatives[cluster_id] = [section_ids[0]]
                else:
                    # For curved sections, select sections that best represent the cluster's diversity
                    # Select up to 3 representatives to capture variation within the cluster
                    num_representatives = min(3, len(section_ids))
                    
                    if num_representatives == len(section_ids):
                        # Small cluster, use all sections
                        cluster_representatives[cluster_id] = section_ids
                    else:
                        # Large cluster, select diverse representatives
                        # Sort by curvature to get range of values
                        sections_with_curvature = [(s, self._get_section_features(s)['mean_curvature']) 
                                                 for s in section_ids]
                        sections_with_curvature.sort(key=lambda x: abs(x[1]))
                        
                        # Select from low, medium, high curvature
                        representatives = []
                        step = len(sections_with_curvature) // num_representatives
                        for i in range(0, len(sections_with_curvature), step):
                            if len(representatives) < num_representatives:
                                representatives.append(sections_with_curvature[i][0])
                        
                        cluster_representatives[cluster_id] = representatives
        
        logger.info(f"Selected representatives for {len(cluster_representatives)} clusters")
        return cluster_representatives

    def _calculate_road_priority_score(self, road_id: str, new_clusters: set,
                                     cluster_representatives: Dict[int, List[str]],
                                     section_matches: Dict[str, Dict],
                                     include_dynamic_analysis: bool = True) -> float:
        """
        Calculate priority score for a road during greedy selection.
        
        Args:
            road_id: Road identifier
            new_clusters: Set of new clusters this road would cover
            cluster_representatives: Representative sections for each cluster
            section_matches: Section matching results
            include_dynamic_analysis: Whether to include dynamic vehicle data in scoring
            
        Returns:
            Priority score for the road
        """
        if not new_clusters:
            return 0.0
        
        # Get road sections
        road_sections = self.road_metadata.get(road_id, [])
        if not road_sections:
            return 0.0
        
        # Determine if dynamic analysis will be included
        use_dynamic = include_dynamic_analysis and self.dynamic_data_available
        
        # Normalize weights to ensure all components sum to 1.0
        # Define relative weights for each component (50% geometric, 50% dynamic)
        weights = {
            # 'coverage': 0.20,      # Base coverage (clusters)
            'curvature': 0.25,     # Road curvature variation
            'critical': 0.20,      # Critical scenarios
            'unique': 0.10,        # Unique patterns
            'length': 0.05,        # Road length diversity
            'dynamic': 0.40        # Dynamic vehicle behavior
        }
        
        # Calculate road-specific priority metrics
        road_metrics = self._calculate_road_priority_metrics(road_sections)
        
        # # Base coverage contribution (normalized by cluster count)
        # max_expected_clusters = 10.0  # Normalize assuming 10 is a typical max
        # coverage_score = weights['coverage'] * (min(len(new_clusters) / max_expected_clusters, 1.0))
        
        # Curvature variation bonus (normalized)
        max_curvature_variation = 0.1  # Typical max value
        curvature_score = weights['curvature'] * min(road_metrics['curvature_variation'] / max_curvature_variation, 1.0)
        
        # Critical scenarios bonus (normalized)
        max_critical_scenarios = 5.0  # Typical max value
        critical_score = weights['critical'] * min(road_metrics['critical_scenarios'] / max_critical_scenarios, 1.0)
        
        # Unique patterns bonus (normalized)
        max_patterns = 2.0  # Max is typically 2 (left and right curves)
        unique_score = weights['unique'] * min(road_metrics['unique_patterns'] / max_patterns, 1.0)
        
        # Road length diversity bonus (normalized)
        max_length = 200.0  # Normalize for typical max length
        length_bonus = weights['length'] * min(road_metrics['total_length'] / max_length, 1.0)
        
        # Dynamic scoring bonus (normalized if enabled and data available)
        dynamic_score = 0.0
        if use_dynamic:
            # Normalize dynamic score (typical max is around 20)
            max_dynamic_score = 20.0
            dynamic_score = weights['dynamic'] * min(self.get_road_dynamic_score(road_id) / max_dynamic_score, 1.0)
        else:
            # Redistribute dynamic weight proportionally to other factors when not used
            redistribution = weights['dynamic'] / 4.0  # Distribute 50% among 4 geometric factors
            # coverage_score += redistribution
            curvature_score += redistribution
            critical_score += redistribution
            unique_score += redistribution
            length_bonus += redistribution

        total_score = curvature_score + critical_score + unique_score + length_bonus + dynamic_score

        # Apply failed road bonus if this road has failed in previous simulations
        failed_road_bonus = 0.0
        if road_id in self.failed_road_ids:
            # Add significant bonus for failed roads (25% of total possible score)
            failed_road_bonus = 0.25
            logger.debug(f"Road {road_id} gets failed road bonus: +{failed_road_bonus}")
        
        # Apply failed road bonus as a multiplier to emphasize importance
        final_score = total_score + failed_road_bonus

        return final_score
    
    def _calculate_road_priority_metrics(self, road_sections: List[str]) -> Dict[str, float]:
        """
        Calculate priority metrics for a road based on its sections.
        
        Args:
            road_sections: List of section IDs for the road
            
        Returns:
            Dictionary with priority metrics
        """
        metrics = {
            'curvature_variation': 0.0,
            'critical_scenarios': 0,
            'unique_patterns': 0,
            'total_length': 0.0
        }
        
        if not road_sections:
            return metrics
        
        # Calculate curvature variation and collect curvatures for critical scenarios
        curvatures = []
        for section_id in road_sections:
            features = self._get_section_features(section_id)
            curvatures.append(abs(features['mean_curvature']))
            metrics['total_length'] += features['length']
        
        if curvatures:
            metrics['curvature_variation'] = np.std(curvatures)
        
        # Count critical scenarios (high curvature sections)
        metrics['critical_scenarios'] = len([c for c in curvatures if c > 0.1])
        
        # Unique patterns (simplified - could be enhanced with actual pattern analysis)
        curved_sections = [s for s in road_sections if self._get_section_features(s)['type'] != 'straight']
        metrics['unique_patterns'] = len(set(self._get_section_features(s)['type'] for s in curved_sections))
        
        return metrics