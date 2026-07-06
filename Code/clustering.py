import json
import logging
import random
import math
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from gpu_operations import CUDA_GPU_AVAILABLE, cp, CumlAgglomerativeClustering, GPU_AVAILABLE, APPLE_GPU_AVAILABLE

logger = logging.getLogger(__name__)

class ClusteringMixin:
    """Mixin class for clustering operations"""

    def _extract_section_geometric_features(self, section_id: str, section_registry: Dict = None) -> Dict[str, float]:
        registry = section_registry if section_registry is not None else self.section_registry
        data = registry[section_id]
        profile = np.array(data['curvature_profile'])
        abs_profile = np.abs(profile)

        max_curvature = float(np.max(abs_profile)) if len(profile) > 0 else 0.0

        if len(profile) > 2:
            n_bins = min(10, max(3, len(profile) // 5))
            hist, _ = np.histogram(profile, bins=n_bins, density=True)
            hist = hist / (hist.sum() + 1e-10)
            curvature_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10)))
        else:
            curvature_entropy = 0.0

        section_length = float(data.get('length', 0.0))

        return {
            'max_curvature': max_curvature,
            'curvature_entropy': curvature_entropy,
            'section_length': section_length,
        }

    def _compute_geometric_multi_metric_distance(self, features1: Dict[str, float],
                                                  features2: Dict[str, float],
                                                  profile1: np.ndarray = None,
                                                  profile2: np.ndarray = None,
                                                  precomputed_dtw: float = None) -> float:
        normalizers = {
            'max_curvature': 0.15,
            'curvature_entropy': 3.5,
            'section_length': 150.0,
        }

        weights = {
            'curvature_profile_dtw': 0.25,
            'max_curvature': 0.25,
            'curvature_entropy': 0.25,
            'section_length': 0.25,
        }

        distance_components = []

        if precomputed_dtw is not None:
            distance_components.append((weights['curvature_profile_dtw'], min(precomputed_dtw, 1.0)))
        elif profile1 is not None and profile2 is not None and len(profile1) > 0 and len(profile2) > 0:
            try:
                dtw_dist, _ = fastdtw(profile1.reshape(-1, 1), profile2.reshape(-1, 1), dist=euclidean)
                avg_len = (len(profile1) + len(profile2)) / 2.0
                normalized_dtw = min(dtw_dist / (avg_len + 1e-10), 1.0)
                distance_components.append((weights['curvature_profile_dtw'], normalized_dtw))
            except Exception:
                mean_diff = abs(np.mean(np.abs(profile1)) - np.mean(np.abs(profile2)))
                distance_components.append((weights['curvature_profile_dtw'], min(mean_diff / 0.1, 1.0)))
        else:
            redistribute = weights['curvature_profile_dtw'] / 3
            weights['max_curvature'] += redistribute
            weights['curvature_entropy'] += redistribute
            weights['section_length'] += redistribute

        for feature_name in ['max_curvature', 'curvature_entropy', 'section_length']:
            val1 = features1.get(feature_name, 0.0)
            val2 = features2.get(feature_name, 0.0)
            denom = normalizers.get(feature_name, 1.0)
            normalized_diff = min(abs(val1 - val2) / (denom + 1e-10), 1.0)
            distance_components.append((weights[feature_name], normalized_diff))

        total_distance = np.sqrt(sum(w * d * d for w, d in distance_components))

        return total_distance

    def _agglomerative_clustering(self, section_matches: Dict[str, Dict],
                                       curvature_similarity_threshold: float = None,
                                       enable_dynamic_clustering: bool = True,
                                       dynamic_weight: float = 0.5) -> Dict[str, int]:
        if self.paths.section_registry_path.exists():
            with open(self.paths.section_registry_path, "r") as f:
                section_registry = json.load(f)
        else:
            section_registry = self.section_registry

        if self.paths.matched_sections_path.exists():
            with open(self.paths.matched_sections_path, "r") as f:
                section_matches = json.load(f)
        else:
            if section_matches is not None:
                pass
            elif hasattr(self, 'section_matches'):
                section_matches = self.section_matches
            else:
                section_matches = self.compare_all_sections()

        calculate_threshold_from_data = curvature_similarity_threshold is None
        if curvature_similarity_threshold is None:
            curvature_similarity_threshold = 0.020

        logger.info(f"Using curvature similarity threshold: {'auto-calculated from data' if calculate_threshold_from_data else f'{curvature_similarity_threshold:.4f}' }")

        section_to_cluster = {}
        next_cluster_id = 0

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

        unmatched_curved_sections = []
        if self.paths.unmatched_sections_path.exists():
            with open(self.paths.unmatched_sections_path, "r") as f:
                unmatched_sections_data = json.load(f)
            unmatched_curved_sections = [s['section_id'] for s in unmatched_sections_data if s.get('type') != 'straight']
        else:
            for section_id, data in section_registry.items():
                if not data['matched']:
                    section_type = data.get('type')
                    if section_type in ['curved', 'left_curve', 'right_curve']:
                        unmatched_curved_sections.append(section_id)

        for section_id in unmatched_curved_sections:
            section_to_cluster[section_id] = next_cluster_id
            next_cluster_id += 1
        logger.info(f"Assigned {len(unmatched_curved_sections)} unmatched curved sections to individual clusters")

        matched_curved_sections = []
        if isinstance(section_matches, list):
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
            if abs_mean_curv >= self.curvature_threshold:
                matched_curved_sections.append(section_id)
        logger.info(f"Found {len(matched_curved_sections)} DTW-matched curved sections for Agglomerative Clustering")

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

        cluster_stats = {'left_curve': [], 'right_curve': []}
        all_pairwise_distances = []

        use_dynamic_clustering = enable_dynamic_clustering and self.dynamic_data_available

        for curve_type, curve_sections in [('left_curve', left_curve_sections), ('right_curve', right_curve_sections)]:
            if len(curve_sections) == 0:
                continue

            logger.info(f"Clustering {len(curve_sections)} {curve_type} sections...")

            if len(curve_sections) == 1:
                section_id = curve_sections[0]
                section_to_cluster[section_id] = next_cluster_id
                next_cluster_id += 1
                logger.info(f"Single {curve_type} section {section_id} assigned to cluster {next_cluster_id - 1}")
                continue

            n_sections = len(curve_sections)
            distance_matrix = np.zeros((n_sections, n_sections))

            if use_dynamic_clustering:
                clustering_mode = "PURE DYNAMIC" if dynamic_weight == 1.0 else "hybrid"
                logger.info(f"Using {clustering_mode} clustering for {curve_type} (geometric weight: {1-dynamic_weight:.2f}, dynamic weight: {dynamic_weight:.2f})")

                if isinstance(section_matches, dict):
                    total_matched_sections = len(section_matches)
                elif isinstance(section_matches, (list, set)):
                    total_matched_sections = len(section_matches)
                else:
                    total_matched_sections = 2000

                max_sections_for_full_dynamic = total_matched_sections
                use_sampling = n_sections > max_sections_for_full_dynamic

                if use_sampling:
                    random.seed(42)
                    sample_size = min(max_sections_for_full_dynamic, n_sections)
                    sampled_sections = random.sample(curve_sections, sample_size)
                    logger.info(f"Large dataset detected ({n_sections} sections). Using sampling approach: computing dynamic features for {sample_size} representative sections.")
                    section_dynamic_features = self._precompute_section_dynamic_features(sampled_sections)
                else:
                    logger.info(f"Computing dynamic features for all {n_sections} sections using existing section boundaries.")
                    section_dynamic_features = self._precompute_section_dynamic_features(curve_sections)

            else:
                logger.info(f"Using pure geometric clustering for {curve_type} (dynamic data unavailable or disabled)")
                use_sampling = False
                section_dynamic_features = {}

            section_curvatures = []

            logger.info(f"📐 Pre-computing multi-metric geometric features for {n_sections} {curve_type} sections...")
            section_geo_features = {}
            for section_id in curve_sections:
                section_geo_features[section_id] = self._extract_section_geometric_features(
                    section_id, section_registry)

            difficulty_scores = [f['max_curvature'] for f in section_geo_features.values()]
            entropy_scores = [f['curvature_entropy'] for f in section_geo_features.values()]
            failed_count = sum(1 for sid in curve_sections if section_registry[sid].get('road_id', '') in self.failed_road_ids)
            logger.info(f"   Difficulty (max_curv): mean={np.mean(difficulty_scores):.4f}, std={np.std(difficulty_scores):.4f}")
            logger.info(f"   Entropy: mean={np.mean(entropy_scores):.4f}, std={np.std(entropy_scores):.4f}")
            logger.info(f"   Sections from failed roads: {failed_count}/{n_sections}")

            curvature_profiles = []
            for i, section_id in enumerate(curve_sections):
                section_data = section_registry[section_id]
                section_profile = np.array(section_data['curvature_profile'])
                section_mean_curv = np.mean(section_profile)
                section_abs_mean_curv = np.abs(section_mean_curv)
                section_curvatures.append((i, section_id, section_abs_mean_curv, section_mean_curv))
                curvature_profiles.append(section_profile)

            if n_sections >= 25 and self.use_gpu:
                logger.info(f"🚀 Computing DTW batch on GPU for {n_sections}x{n_sections} matrix (one-time)...")
                dtw_distance_matrix = self._gpu_dtw_batch(curvature_profiles)
            else:
                logger.info(f"💻 Computing DTW batch on CPU for {n_sections}x{n_sections} matrix (one-time)...")
                dtw_distance_matrix = self._cpu_dtw_batch(curvature_profiles)

            avg_profile_lengths = np.array([(len(curvature_profiles[i]) + len(curvature_profiles[j])) / 2.0
                                           for i in range(n_sections) for j in range(n_sections)]).reshape(n_sections, n_sections)
            normalized_dtw_matrix = np.minimum(dtw_distance_matrix / (avg_profile_lengths + 1e-10), 1.0)
            logger.info(f"DTW batch computation complete.")

            logger.info(f"⚡ Building multi-metric distances (using precomputed DTW + difficulty + length + entropy + failure)...")
            for i, section_id in enumerate(curve_sections):
                if i % 200 == 0 and i > 0:
                    logger.info(f"Processing section {i}/{n_sections} ({i/n_sections*100:.1f}%)")

                section_data = section_registry[section_id]
                section_profile = np.array(section_data['curvature_profile'])
                section_mean_curv = np.mean(section_profile)

                geo_feat1 = section_geo_features[section_id]

                for j, other_section_id in enumerate(curve_sections):
                    if i != j:
                        geo_feat2 = section_geo_features[other_section_id]

                        geometric_distance = self._compute_geometric_multi_metric_distance(
                            geo_feat1, geo_feat2,
                            precomputed_dtw=normalized_dtw_matrix[i, j])

                        if use_dynamic_clustering:
                            if use_sampling:
                                if section_id in section_dynamic_features and other_section_id in section_dynamic_features:
                                    features1 = section_dynamic_features[section_id]
                                    features2 = section_dynamic_features[other_section_id]
                                    dynamic_distance = self._compute_dynamic_feature_distance(features1, features2)
                                    hybrid_distance = (1 - dynamic_weight) * geometric_distance + dynamic_weight * dynamic_distance
                                else:
                                    hybrid_distance = geometric_distance
                            else:
                                features1 = section_dynamic_features.get(section_id, {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0})
                                features2 = section_dynamic_features.get(other_section_id, {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0})
                                dynamic_distance = self._compute_dynamic_feature_distance(features1, features2)
                                hybrid_distance = (1 - dynamic_weight) * geometric_distance + dynamic_weight * dynamic_distance
                        else:
                            hybrid_distance = geometric_distance

                        distance_matrix[i, j] = hybrid_distance

                        if calculate_threshold_from_data and i < j:
                            all_pairwise_distances.append(hybrid_distance)

            logger.info(f"Distance matrix computation complete.")

            non_zero_distances = distance_matrix[distance_matrix > 0]
            if len(non_zero_distances) > 0:
                logger.info(f"Distance matrix statistics:")
                logger.info(f"   Min distance: {non_zero_distances.min():.4f}")
                logger.info(f"   Max distance: {non_zero_distances.max():.4f}")
                logger.info(f"   Mean distance: {non_zero_distances.mean():.4f}")
                logger.info(f"   Median distance: {np.median(non_zero_distances):.4f}")

            curvatures_only = [abs(curv_signed) for _, _, _, curv_signed in section_curvatures]
            curv_std = np.std(curvatures_only)
            curv_range = max(curvatures_only) - min(curvatures_only)
            logger.info(f"{curve_type} curvature distribution: std={curv_std:.4f}, range={curv_range:.4f}")

            if calculate_threshold_from_data and len(all_pairwise_distances) >= 10:
                distances_array = np.array(all_pairwise_distances)
                q25, q50, q75, q90 = np.percentile(distances_array, [25, 50, 75, 90])
                std_dev = np.std(distances_array)
                cv = std_dev / np.mean(distances_array) if np.mean(distances_array) > 0 else 0
                cv_min, cv_max = 0.1, 1.0
                percentile_min, percentile_max = 60, 90
                cv_clipped = np.clip(cv, cv_min, cv_max)
                percentile = percentile_max - ((cv_clipped - cv_min) / (cv_max - cv_min)) * (percentile_max - percentile_min)
                curvature_similarity_threshold = np.percentile(distance_matrix, percentile)
                clustering_stats = getattr(self, 'clustering_stats', {})
                clustering_stats['cv'] = float(f"{cv:.4f}")
                clustering_stats['percentile'] = float(f"{percentile:.2f}")
                clustering_stats['threshold'] = float(f"{curvature_similarity_threshold:.4f}")
                self.clustering_stats = clustering_stats
                print(f"CV: {cv:.4f}, Percentile used: {percentile:.2f}%, Threshold distance: {curvature_similarity_threshold:.4f}")
                logger.info(f"Distribution quartiles - Q25: {q25:.4f}, Q50: {q50:.4f}, Q75: {q75:.4f}, Q90: {q90:.4f}")
                logger.info(f"Based on {len(all_pairwise_distances)} pairwise distances, "
                           f"range: {min(all_pairwise_distances):.4f} to {max(all_pairwise_distances):.4f}")
                calculate_threshold_from_data = False

            try:
                if self.use_gpu and self.gpu_type == "CUDA" and CUDA_GPU_AVAILABLE and n_sections >= 25:
                    logger.info(f"🚀 Using GPU-accelerated clustering for {n_sections} {curve_type} sections")
                    gpu_distance_matrix = cp.asarray(distance_matrix)

                    clustering = CumlAgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=curvature_similarity_threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    labels = clustering.fit_predict(gpu_distance_matrix)
                    labels = cp.asnumpy(labels)
                else:
                    if n_sections >= 25:
                        logger.info(f"💻 Using CPU clustering for {n_sections} {curve_type} sections")
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=curvature_similarity_threshold,
                        metric='precomputed',
                        linkage='complete'
                    )
                    labels = clustering.fit_predict(distance_matrix)

                cluster_groups = defaultdict(list)
                for i, label in enumerate(labels):
                    section_id = curve_sections[i]
                    cluster_groups[label].append(section_id)

                for cluster_sections in cluster_groups.values():
                    cluster_id = next_cluster_id
                    next_cluster_id += 1

                    for section_id in cluster_sections:
                        section_to_cluster[section_id] = cluster_id

                    if len(cluster_sections) > 1:
                        curvatures = []
                        max_curvatures = []
                        entropies = []
                        failed_count = 0
                        for section_id in cluster_sections:
                            profile = np.array(section_registry[section_id]['curvature_profile'])
                            curvatures.append(abs(np.mean(profile)))
                            max_curvatures.append(float(np.max(np.abs(profile))))
                            geo_feat = section_geo_features.get(section_id, {})
                            entropies.append(geo_feat.get('curvature_entropy', 0.0))
                            road_id = section_registry[section_id].get('road_id', '')
                            if road_id in self.failed_road_ids:
                                failed_count += 1

                        curv_range = max(curvatures) - min(curvatures)
                        avg_curvature = np.mean(curvatures)
                        avg_max_curv = np.mean(max_curvatures)
                        avg_entropy = np.mean(entropies)

                        logger.info(f"Cluster {cluster_id} ({curve_type}): {len(cluster_sections)} sections, "
                                   f"avg curvature: {avg_curvature:.4f}, range: {curv_range:.4f}, "
                                   f"avg_difficulty(max_curv): {avg_max_curv:.4f}, "
                                   f"avg_entropy: {avg_entropy:.2f}, "
                                   f"failed_roads: {failed_count}/{len(cluster_sections)}")

                        cluster_stats[curve_type].append(len(cluster_sections))
                    else:
                        logger.info(f"Cluster {cluster_id} ({curve_type}): 1 section (individual)")

                logger.info(f"{curve_type} clustering produced {len(cluster_groups)} clusters")

            except Exception as e:
                logger.warning(f"Clustering failed for {curve_type} sections: {e}")
                for section_id in curve_sections:
                    section_to_cluster[section_id] = next_cluster_id
                    next_cluster_id += 1
                logger.info(f"Fallback: Assigned {len(curve_sections)} individual clusters for {curve_type}")

        total_straight = len(straight_sections)
        total_matched_curved = len(matched_curved_sections)
        total_unmatched_curved = len(unmatched_curved_sections)

        clustering_mode = "PURE DYNAMIC" if (use_dynamic_clustering and dynamic_weight == 0.5) else ("Hybrid Agglomerative" if use_dynamic_clustering else "Multi-Metric Geometric")
        logger.info(f"\n{'='*60}")
        logger.info(f"{clustering_mode} Clustering results:")
        logger.info(f"  - Distance metrics: DTW profile + max_curvature + entropy + section_length")
        logger.info(f"  - Clustering mode: {'100% Dynamic, 0% Geometric' if dynamic_weight == 0.0 else f'{dynamic_weight*100:.0f}% Dynamic, {(1-dynamic_weight)*100:.0f}% Geometric'}")
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
        if not self.dynamic_data_available:
            logger.warning("Dynamic data not available. Returning empty features.")
            return {}

        logger.info(f"Pre-computing dynamic features for {len(section_ids)} existing sections...")
        section_dynamic_features = {}

        if not hasattr(self, '_dynamic_data_cache'):
            self._dynamic_data_cache = {}

        processed_count = 0
        for section_id in section_ids:
            try:
                section_data = self.section_registry[section_id]
                road_id = section_data['road_id']

                if road_id not in self._dynamic_data_cache:
                    self._dynamic_data_cache[road_id] = self.load_dynamic_data(road_id)

                dynamic_data = self._dynamic_data_cache[road_id]
                if dynamic_data is not None:
                    start_idx = section_data['start_idx']
                    end_idx = section_data['end_idx']

                    section_dynamic = [row for row in dynamic_data
                                     if start_idx <= row.get('idx', 0) <= end_idx]

                    if section_dynamic:
                        section_dynamic_features[section_id] = self._extract_section_dynamic_features(section_dynamic)
                    else:
                        section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}
                else:
                    section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(section_ids)} sections ({processed_count/len(section_ids)*100:.1f}%)")

            except Exception as e:
                logger.warning(f"Error processing dynamic features for section {section_id}: {e}")
                section_dynamic_features[section_id] = {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}

        logger.info(f"Dynamic feature pre-computation complete. Processed {len(section_dynamic_features)} sections.")
        return section_dynamic_features
