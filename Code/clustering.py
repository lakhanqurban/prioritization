import json
import logging
import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

# Note: Distance computation uses optimized NumPy operations with automatic backend selection
# Clustering uses scikit-learn with optimized BLAS/LAPACK for cross-platform compatibility

logger = logging.getLogger(__name__)

class ClusteringMixin:
    """Mixin class for clustering operations"""
    
    def _agglomerative_clustering(self, section_matches: Dict[str, Dict],
                                       curvature_similarity_threshold: float = None,
                                       enable_dynamic_clustering: bool = True,
                                       dynamic_weight: float = 0.5) -> Dict[str, int]:
        """
        Create clusters using Hybrid Agglomerative Clustering based on geometric and dynamic features.
        Loads section_registry and section_matches from disk if available, otherwise uses in-memory data.
        
        HYBRID AGGLOMERATIVE CLUSTERING APPROACH:
        - Uses hybrid distance metric combining curvature differences and dynamic behavior
        - Applies Agglomerative Clustering only to DTW-matched curved sections
        - Groups sections by type (left_curve/right_curve) and applies clustering separately
        - Straight sections form one cluster
        - Unmatched curved sections form individual clusters
        - Uses optimal distance thresholds calculated from data distribution
        
        Args:
            section_matches: Section matching results from compare_all_sections()
            curvature_similarity_threshold: Maximum allowed curvature difference within a cluster
                                          If None, will be calculated automatically based on data
            enable_dynamic_clustering: Whether to include dynamic data in clustering distance
            dynamic_weight: Weight for dynamic features (0.0-1.0, remaining weight goes to geometric)
        Returns:
            Dictionary mapping section_id to cluster_id
        """
        # Load section_registry from disk if available
        if self.paths.section_registry_path.exists():
            with open(self.paths.section_registry_path, "r") as f:
                section_registry = json.load(f)
        else:
            section_registry = self.section_registry

        # Load section_matches from disk if available
        if self.paths.matched_sections_path.exists():
            with open(self.paths.matched_sections_path, "r") as f:
                section_matches = json.load(f)
        # else: use provided section_matches argument
        else:
            # Fallback: use provided argument, or call compare_all_sections if not available
            if section_matches is not None:
                pass  # use provided argument
            elif hasattr(self, 'section_matches'):
                section_matches = self.section_matches
            else:
                section_matches = self.compare_all_sections()

        # Calculate optimal threshold if not provided (will be done during clustering to avoid redundant calculations)
        calculate_threshold_from_data = curvature_similarity_threshold is None
        if curvature_similarity_threshold is None:
            curvature_similarity_threshold = 0.020  # Temporary placeholder, will be recalculated from distance matrices

        logger.info(f"Using curvature similarity threshold: {'auto-calculated from data' if calculate_threshold_from_data else f'{curvature_similarity_threshold:.4f}' }")
        
        # Initialize cluster mapping
        section_to_cluster = {}
        next_cluster_id = 0

        # 1. Handle straight sections - all belong to one global cluster
        straight_sections = []
        for section_id, data in section_registry.items():
            features = self._get_section_features(section_id) if section_registry is self.section_registry else {
                'type': 'straight' if abs(sum(data['curvature_profile'])/len(data['curvature_profile'])) < self.curvature_threshold else ('left_curve' if sum(data['curvature_profile'])/len(data['curvature_profile']) > 0 else 'right_curve')
            }
            if features['type'] == 'straight':
                straight_sections.append(section_id)
        if straight_sections:
            straight_cluster_id = next_cluster_id
            next_cluster_id += 1
            for section_id in straight_sections:
                section_to_cluster[section_id] = straight_cluster_id
            logger.info(f"Assigned {len(straight_sections)} straight sections to cluster {straight_cluster_id}")
        
        # 2. Handle unmatched curved sections - each gets its own unique cluster
        unmatched_curved_sections = []
        # Try to load unmatched sections from disk
        if self.paths.unmatched_sections_path.exists():
            with open(self.paths.unmatched_sections_path, "r") as f:
                unmatched_sections_data = json.load(f)
            unmatched_curved_sections = [s['section_id'] for s in unmatched_sections_data if s.get('type') != 'straight']
            found_unmatched = True
        else:
            found_unmatched = False
            for section_id, data in section_registry.items():
                # Use the SAME definition as save_matching_info for consistency
                # Only consider sections that are marked as unmatched in the registry
                if not data['matched']:
                    section_type = data.get('type')
                    if section_type in ['curved', 'left_curve', 'right_curve']:
                        unmatched_curved_sections.append(section_id)
        
        for section_id in unmatched_curved_sections:
            section_to_cluster[section_id] = next_cluster_id
            next_cluster_id += 1
        logger.info(f"Assigned {len(unmatched_curved_sections)} unmatched curved sections to individual clusters")
        
        # 3. Collect DTW-matched curved sections and separate by type for Agglomerative Clustering
        matched_curved_sections = []
        # Ensure section_matches is a dictionary
        if isinstance(section_matches, list):
            # Convert list to dict if possible (assume each item has 'section_id')
            section_matches_dict = {}
            for item in section_matches:
                if isinstance(item, dict) and 'section_id' in item:
                    section_matches_dict[item['section_id']] = item
            section_matches = section_matches_dict
        for section_id in section_matches.keys():
            data = section_registry[section_id]
            profile = data['curvature_profile']
            mean_curv = sum(profile)/len(profile) if profile else 0
            abs_mean_curv = abs(mean_curv)
            features = self._get_section_features(section_id) if section_registry is self.section_registry else {
                'type': 'straight' if abs_mean_curv < self.curvature_threshold else ('left_curve' if mean_curv > 0 else 'right_curve')
            }
            if abs_mean_curv >= self.curvature_threshold:
                matched_curved_sections.append(section_id)
        logger.info(f"Found {len(matched_curved_sections)} DTW-matched curved sections for Agglomerative Clustering")
        
        # 4. Separate matched curved sections by type and apply Agglomerative Clustering
        left_curve_sections = []
        right_curve_sections = []
        for section_id in matched_curved_sections:
            features = self._get_section_features(section_id) if section_registry is self.section_registry else {
                'type': 'straight' if abs(sum(section_registry[section_id]['curvature_profile'])/len(section_registry[section_id]['curvature_profile'])) < self.curvature_threshold else ('left_curve' if sum(section_registry[section_id]['curvature_profile'])/len(section_registry[section_id]['curvature_profile']) > 0 else 'right_curve')
            }
            if features['type'] == 'left_curve':
                left_curve_sections.append(section_id)
            elif features['type'] == 'right_curve':
                right_curve_sections.append(section_id)
        logger.info(f"Separated into {len(left_curve_sections)} left curves and {len(right_curve_sections)} right curves")
        
        # 5. Apply Agglomerative Clustering to each curve type separately
        cluster_stats = {'left_curve': [], 'right_curve': []}
        all_pairwise_distances = []  # Collect all distances for threshold calculation
        
        # Determine if dynamic clustering is enabled and data is available globally
        use_dynamic_clustering = enable_dynamic_clustering and self.dynamic_data_available
        
        for curve_type, curve_sections in [('left_curve', left_curve_sections), ('right_curve', right_curve_sections)]:
            if len(curve_sections) == 0:
                continue
                
            logger.info(f"Clustering {len(curve_sections)} {curve_type} sections...")
            
            if len(curve_sections) == 1:
                # Single section - assign individual cluster
                section_id = curve_sections[0]
                section_to_cluster[section_id] = next_cluster_id
                next_cluster_id += 1
                logger.info(f"Single {curve_type} section {section_id} assigned to cluster {next_cluster_id - 1}")
                continue
            
            # Build distance matrix using hybrid approach (geometric + dynamic)
            n_sections = len(curve_sections)
            distance_matrix = np.zeros((n_sections, n_sections))
            
            # Log clustering approach for this curve type
            if use_dynamic_clustering:
                logger.info(f"Using hybrid clustering for {curve_type} (geometric weight: {1-dynamic_weight:.2f}, dynamic weight: {dynamic_weight:.2f})")
                
                # Use optimized approach: augment existing sections with dynamic features
                # For large datasets, use sampling to avoid excessive computation time
                # Calculate total matched sections count properly
                if isinstance(section_matches, dict):
                    total_matched_sections = len(section_matches)
                elif isinstance(section_matches, (list, set)):
                    total_matched_sections = len(section_matches)
                else:
                    total_matched_sections = 2000  # Fallback default
                
                max_sections_for_full_dynamic = total_matched_sections  # Threshold based on total matched sections
                use_sampling = n_sections > max_sections_for_full_dynamic
                
                if use_sampling:
                    # Randomly sample sections for dynamic feature computation
                    
                    random.seed(42)  # For reproducibility
                    sample_size = min(max_sections_for_full_dynamic, n_sections)
                    sampled_sections = random.sample(curve_sections, sample_size)
                    logger.info(f"Large dataset detected ({n_sections} sections). Using sampling approach: computing dynamic features for {sample_size} representative sections.")
                    section_dynamic_features = self._precompute_section_dynamic_features(sampled_sections)
                else:
                    # logger.info(f"Computing dynamic features for all {n_sections} sections using existing section boundaries.")
                    section_dynamic_features = self._precompute_section_dynamic_features(curve_sections)
                
            else:
                logger.info(f"Using pure geometric clustering for {curve_type} (dynamic data unavailable or disabled)")
                use_sampling = False
                section_dynamic_features = {}
            
            # Collect curvature data and build distance matrix
            section_curvatures = []
            
            # Prepare curvature profiles for batch processing
            curvature_profiles = []
            for i, section_id in enumerate(curve_sections):
                section_data = section_registry[section_id]
                section_profile = np.array(section_data['curvature_profile'])
                section_mean_curv = np.mean(section_profile)
                section_abs_mean_curv = np.abs(section_mean_curv)
                section_curvatures.append((i, section_id, section_abs_mean_curv, section_mean_curv))
                curvature_profiles.append(section_profile)
            
            # Compute distance matrix with proper hybrid approach
            if n_sections >= 25:
                if use_dynamic_clustering and section_dynamic_features:
                    logger.info(f"Computing hybrid {n_sections}x{n_sections} distance matrix (geometric + dynamic)...")
                    distance_matrix = self._compute_hybrid_distance_matrix(
                        curvature_profiles, curve_sections, section_dynamic_features, dynamic_weight
                    )
                else:
                    logger.info(f"Computing geometric {n_sections}x{n_sections} distance matrix...")
                    distance_matrix = self._compute_distance_matrix_optimized(curvature_profiles)
            else:
                # Use fallback for smaller datasets
                logger.info(f"Building {n_sections}x{n_sections} distance matrix...")
                distance_matrix = np.zeros((n_sections, n_sections))
            
            for i, section_id in enumerate(curve_sections):
                if i % 100 == 0 and i > 0:  # Progress indicator
                    logger.info(f"Processing section {i}/{n_sections} ({i/n_sections*100:.1f}%)")
                
                section_data = section_registry[section_id]
                section_profile = np.array(section_data['curvature_profile'])
                section_mean_curv = np.mean(section_profile)
                section_abs_mean_curv = np.abs(section_mean_curv)
                section_curvatures.append((i, section_id, section_abs_mean_curv, section_mean_curv))
                
                for j, other_section_id in enumerate(curve_sections):
                    if i != j:
                        other_data = section_registry[other_section_id]
                        other_profile = np.array(other_data['curvature_profile'])
                        other_mean_curv = np.mean(other_profile)
                        
                        # Calculate geometric distance (curvature difference)
                        geometric_distance = abs(section_mean_curv - other_mean_curv)
                        
                        # Calculate dynamic distance if enabled (using pre-computed features)
                        if use_dynamic_clustering:
                            # For sampling approach, use dynamic features if available, otherwise use geometric-only
                            if use_sampling:
                                if section_id in section_dynamic_features and other_section_id in section_dynamic_features:
                                    features1 = section_dynamic_features[section_id]
                                    features2 = section_dynamic_features[other_section_id]
                                    dynamic_distance = self._compute_dynamic_feature_distance(features1, features2)
                                    hybrid_distance = (1 - dynamic_weight) * geometric_distance + dynamic_weight * dynamic_distance
                                else:
                                    # Fall back to geometric-only for non-sampled sections
                                    hybrid_distance = geometric_distance
                            else:
                                features1 = section_dynamic_features.get(section_id, {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0})
                                features2 = section_dynamic_features.get(other_section_id, {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0})
                                dynamic_distance = self._compute_dynamic_feature_distance(features1, features2)
                                hybrid_distance = (1 - dynamic_weight) * geometric_distance + dynamic_weight * dynamic_distance
                        else:
                            hybrid_distance = geometric_distance
                        
                        distance_matrix[i, j] = hybrid_distance
                        
                        # Collect distances for threshold calculation if needed
                        if calculate_threshold_from_data and i < j:  # Only collect upper triangle to avoid duplicates
                            all_pairwise_distances.append(hybrid_distance)
            
            logger.info(f"Distance matrix computation complete.")
            
            # Log curvature distribution for this curve type
            curvatures_only = [abs(curv_signed) for _, _, _, curv_signed in section_curvatures]
            curv_std = np.std(curvatures_only)
            curv_range = max(curvatures_only) - min(curvatures_only)
            # logger.info(f"{curve_type} curvature distribution: std={curv_std:.4f}, range={curv_range:.4f}")
            
            # Calculate threshold from collected distances if needed (do this once when we have enough data)
            if calculate_threshold_from_data and len(all_pairwise_distances) >= 10:  # Need sufficient data
                # Adaptive percentile selection based on data distribution characteristics
                distances_array = np.array(all_pairwise_distances)
                # Calculate distribution characteristics
                q25, q50, q75, q90 = np.percentile(distances_array, [25, 50, 75, 90])
                std_dev = np.std(distances_array)
                cv = std_dev / np.mean(distances_array) if np.mean(distances_array) > 0 else 0
                # Define CV-based scaling
                cv_min, cv_max = 0.1, 1.0  # tuned based on data
                percentile_min, percentile_max = 60, 90  # adjust for desired cluster tightness
                cv_clipped = np.clip(cv, cv_min, cv_max)
                percentile = percentile_max - ((cv_clipped - cv_min) / (cv_max - cv_min)) * (percentile_max - percentile_min)
                # Compute adaptive threshold
                curvature_similarity_threshold = np.percentile(distance_matrix, percentile)
                # Store clustering stats for reporting
                clustering_stats = getattr(self, 'clustering_stats', {})
                clustering_stats['cv'] = float(f"{cv:.4f}")
                clustering_stats['percentile'] = float(f"{percentile:.2f}")
                clustering_stats['threshold'] = float(f"{curvature_similarity_threshold:.4f}")
                self.clustering_stats = clustering_stats
                print(f"CV: {cv:.4f}, Percentile used: {percentile:.2f}%, Threshold distance: {curvature_similarity_threshold:.4f}")
                # logger.info(f"Distribution quartiles - Q25: {q25:.4f}, Q50: {q50:.4f}, Q75: {q75:.4f}, Q90: {q90:.4f}")
                # logger.info(f"Based on {len(all_pairwise_distances)} pairwise distances, "
                        #    f"range: {min(all_pairwise_distances):.4f} to {max(all_pairwise_distances):.4f}")
                calculate_threshold_from_data = False  # Don't recalculate again
            
            # Apply Agglomerative Clustering
            try:
                # Note: scikit-learn automatically uses optimized BLAS/LAPACK (OpenBLAS with SIMD)
                logger.info(f"Clustering {n_sections} {curve_type} sections using AgglomerativeClustering")
                
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=curvature_similarity_threshold,
                    metric='precomputed',
                    linkage='complete'  # Complete linkage ensures all pairs respect threshold
                )
                labels = clustering.fit_predict(distance_matrix)
                
                # Group sections by cluster labels
                cluster_groups = defaultdict(list)
                for i, label in enumerate(labels):
                    section_id = curve_sections[i]
                    cluster_groups[label].append(section_id)
                
                # Assign global cluster IDs directly from agglomerative clustering results
                for cluster_sections in cluster_groups.values():
                    cluster_id = next_cluster_id
                    next_cluster_id += 1
                    
                    for section_id in cluster_sections:
                        section_to_cluster[section_id] = cluster_id
                    
                    # Log cluster characteristics
                    if len(cluster_sections) > 1:
                        curvatures = []
                        for section_id in cluster_sections:
                            profile = np.array(section_registry[section_id]['curvature_profile'])
                            curvatures.append(abs(np.mean(profile)))
                        
                        curv_range = max(curvatures) - min(curvatures)
                        avg_curvature = np.mean(curvatures)
                        
                        # logger.info(f"Cluster {cluster_id} ({curve_type}): {len(cluster_sections)} sections, "
                                #    f"avg curvature: {avg_curvature:.4f}, range: {curv_range:.4f}")
                        
                        cluster_stats[curve_type].append(len(cluster_sections))
                    else:
                        logger.info(f"Cluster {cluster_id} ({curve_type}): 1 section (individual)")
                
                logger.info(f"{curve_type} clustering produced {len(cluster_groups)} clusters")
                
            except Exception as e:
                logger.warning(f"Clustering failed for {curve_type} sections: {e}")
                # Fallback: assign individual clusters
                for section_id in curve_sections:
                    section_to_cluster[section_id] = next_cluster_id
                    next_cluster_id += 1
                logger.info(f"Fallback: Assigned {len(curve_sections)} individual clusters for {curve_type}")
        
        # Log final clustering results
        total_straight = len(straight_sections)
        total_matched_curved = len(matched_curved_sections)
        total_unmatched_curved = len(unmatched_curved_sections)
        
        clustering_type = "Hybrid Agglomerative" if use_dynamic_clustering else "Pure Agglomerative"
        logger.info(f"{clustering_type} Clustering results:") 
        logger.info(f"  - Total sections analyzed: {len(section_registry)}")
        logger.info(f"  - Straight sections in single cluster: {total_straight}")
        logger.info(f"  - DTW-matched curved sections processed: {total_matched_curved}")
        logger.info(f"  - Unmatched curved sections in individual clusters: {total_unmatched_curved}")
        logger.info(f"  - Distance threshold used: {curvature_similarity_threshold:.4f}")
        if use_dynamic_clustering:
            logger.info(f"  - Dynamic weight in clustering: {dynamic_weight:.2f}")
        
        for curve_type, sizes in cluster_stats.items():
            if sizes:
                logger.info(f"  - {curve_type} clusters: {len(sizes)} clusters, "
                           f"sizes: {sizes}, total sections: {sum(sizes)}")
        
        return section_to_cluster
                
    def _precompute_section_dynamic_features(self, section_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Pre-compute dynamic features for existing sections using cached geometric data.
        This augments existing section registry with dynamic behavioral features.
        
        Args:
            section_ids: List of section IDs to compute dynamic features for
            
        Returns:
            Dictionary mapping section_id to dynamic features
        """
        if not self.dynamic_data_available:
            logger.warning("Dynamic data not available. Returning empty features.")
            return {}
        
        logger.info(f"Pre-computing dynamic features for {len(section_ids)} existing sections...")
        section_dynamic_features = {}
        
        # Initialize dynamic data cache if not exists
        if not hasattr(self, '_dynamic_data_cache'):
            self._dynamic_data_cache = {}
        
        processed_count = 0
        for section_id in section_ids:
            try:
                # Get existing section data (already computed geometric info)
                section_data = self.section_registry[section_id]
                road_id = section_data['road_id']
                
                # Load and cache dynamic data for this road if not already cached
                if road_id not in self._dynamic_data_cache:
                    self._dynamic_data_cache[road_id] = self.load_dynamic_data(road_id)
                
                dynamic_data = self._dynamic_data_cache[road_id]
                if dynamic_data is not None:
                    # Use existing section boundaries from geometric analysis
                    start_idx = section_data['start_idx']
                    end_idx = section_data['end_idx']
                    
                    # Filter dynamic data to section boundaries
                    section_dynamic = [row for row in dynamic_data 
                                     if start_idx <= row.get('idx', 0) <= end_idx]
                    
                    if section_dynamic:
                        # Extract dynamic features for this section
                        section_dynamic_features[section_id] = self._extract_section_dynamic_features(section_dynamic)
                    else:
                        # Default features if no dynamic data in range
                        section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}
                else:
                    # Default features if no dynamic data file
                    section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(section_ids)} sections ({processed_count/len(section_ids)*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"Error processing dynamic features for section {section_id}: {e}")
                section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}
        
        logger.info(f"Dynamic feature pre-computation complete. Processed {len(section_dynamic_features)} sections.")
        return section_dynamic_features
    
    def _compute_distance_matrix_optimized(self, curvature_profiles: List[np.ndarray]) -> np.ndarray:
        """
        Compute geometric distance matrix using optimized NumPy operations.
        Automatically uses the best available BLAS/LAPACK backend (OpenBLAS with SIMD).
        """
        n = len(curvature_profiles)
        
        # Pad all profiles to same length for vectorized operations
        max_len = max(len(profile) for profile in curvature_profiles)
        padded_profiles = np.zeros((n, max_len))
        
        for i, profile in enumerate(curvature_profiles):
            padded_profiles[i, :len(profile)] = profile
        
        # Vectorized distance computation using broadcasting
        # This automatically leverages optimized BLAS operations
        diff = padded_profiles[:, np.newaxis, :] - padded_profiles[np.newaxis, :, :]
        distances = np.mean(np.abs(diff), axis=2)
        
        return distances
    
    def _compute_hybrid_distance_matrix(self, curvature_profiles: List[np.ndarray], 
                                       curve_sections: List[str],
                                       section_dynamic_features: Dict[str, Dict[str, float]],
                                       dynamic_weight: float) -> np.ndarray:
        """
        Compute hybrid distance matrix combining geometric and dynamic features from the start.
        This is the correct approach - compute hybrid distances directly, not patch geometric ones.
        """
        n = len(curvature_profiles)
        
        # First compute geometric distances efficiently
        geometric_distances = self._compute_distance_matrix_optimized(curvature_profiles)
        
        # Then compute dynamic distances and combine properly
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    # Get geometric distance
                    geometric_distance = geometric_distances[i, j]
                    
                    # Get section IDs
                    section_id_i = curve_sections[i]
                    section_id_j = curve_sections[j]
                    
                    # Compute dynamic distance if both sections have features
                    if (section_id_i in section_dynamic_features and 
                        section_id_j in section_dynamic_features):
                        features_i = section_dynamic_features[section_id_i]
                        features_j = section_dynamic_features[section_id_j]
                        dynamic_distance = self._compute_dynamic_feature_distance(features_i, features_j)
                        
                        # Proper hybrid combination
                        hybrid_distance = (1 - dynamic_weight) * geometric_distance + dynamic_weight * dynamic_distance
                        distance_matrix[i, j] = hybrid_distance
                    else:
                        # Fall back to geometric-only
                        distance_matrix[i, j] = geometric_distance
        
        return distance_matrix