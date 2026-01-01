import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Handle both relative and absolute imports
try:
    from utils.road_utils import calculate_curvature, calculate_road_length
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path to find utils
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.road_utils import calculate_curvature, calculate_road_length

try:
    from data_classes import RoadSegment
except ImportError:
    from .data_classes import RoadSegment

logger = logging.getLogger(__name__)

class SectionAnalysisMixin:
    """Mixin class for section analysis operations"""
    
    def classify_road_segments(self, road_points: List[Tuple[float, float]]) -> Dict:
        """Classify road segments into straight and curved sections with hysteresis"""
        if len(road_points) < 3:
            return {"segments": [], "straight": [], "curved": [], "total_length": 0, "straight_ratio": 0}
    
        curvatures = calculate_curvature(road_points)
        if len(curvatures) == 0:
            return {"segments": [], "straight": [], "curved": [], "total_length": 0, "straight_ratio": 0}
    
        # Pad curvatures to match road points length
        curvatures = np.pad(curvatures, (1, 1), 'constant')
    
        segments = []
        current_segment = []
        current_type = None
        hysteresis_buffer = []
    
        for i, (point, curvature) in enumerate(zip(road_points, curvatures)):
            abs_curv = abs(curvature)
            hysteresis_buffer.append(abs_curv)
        
            if len(hysteresis_buffer) > self.hysteresis_window:
                hysteresis_buffer.pop(0)
        
            # Determine segment type using hysteresis
            if current_type == 'curved':
                segment_type = 'straight' if all(c < self.curvature_threshold for c in hysteresis_buffer) else 'curved'
            else:
                segment_type = 'curved' if all(c >= self.curvature_threshold for c in hysteresis_buffer) else 'straight'
        
            if current_type is None:
                current_type = segment_type
            
            if segment_type == current_type:
                current_segment.append(point)
            else:
                # Finalize current segment
                if len(current_segment) >= self.min_segment_length and current_type is not None:
                    start_idx = i - len(current_segment)
                    segments.append(RoadSegment(
                        segment_type=current_type,
                        points=current_segment.copy(),
                        start_index=start_idx,
                        end_index=i - 1,
                        length=calculate_road_length(current_segment),
                        curvature_profile=curvatures[start_idx:i]
                    ))
                current_segment = [point]
                current_type = segment_type
    
        # Add the final segment if valid
        if len(current_segment) >= self.min_segment_length and current_type is not None:
            start_idx = len(road_points) - len(current_segment)
            segments.append(RoadSegment(
                segment_type=current_type,
                points=current_segment.copy(),
                start_index=start_idx,
                end_index=len(road_points) - 1,
                length=calculate_road_length(current_segment),
                curvature_profile=curvatures[start_idx:len(road_points)]
            ))
    
        # Separate straight and curved segments
        straight_segments = [s for s in segments if s.segment_type == "straight"]
        curved_segments = [s for s in segments if s.segment_type == "curved"]
    
        return {
            "segments": segments,
            "straight": straight_segments,
            "curved": curved_segments,
            "total_length": calculate_road_length(road_points),
            "straight_ratio": len(straight_segments)/len(segments) if segments else 0
        }

    def compare_all_sections(self) -> Dict[str, Dict]:
        """Directly compare all sections in the registry with each other with subsection matching"""
        # Create a list of all sections with their metadata
        all_sections = []
        for section_id, data in self.section_registry.items():
            # Skip very short sections to reduce computation
            if len(data['curvature_profile']) < self.min_subsection_length:
                continue
            
            section = {
                'section_id': section_id,
                'road_id': data['road_id'],
                'segment': RoadSegment(
                    segment_type=data['type'],
                    points=data['points'],
                    start_index=data['start_idx'],
                    end_index=data['end_idx'],
                    length=data['length'],
                    curvature_profile=np.array(data['curvature_profile']),
                    matched=data['matched']
                )
            }
            all_sections.append(section)

        # Pre-calculate mean curvatures for all sections (will be used for enhanced type checking)
        section_means = {}
        for section in all_sections:
            section_means[section['section_id']] = np.mean(section['segment'].curvature_profile)

        # Calculate the expected number of type-filtered comparisons
        distribution = self.analyze_section_distribution()
        expected_comparisons = distribution['total_potential_comparisons']

        # Print sections summary before starting comparisons
        logger.info("SECTIONS SUMMARY:")
        logger.info(f"   • Total sections: {distribution['total_sections']}")
        logger.info(f"   • Straight sections: {distribution['type_counts']['straight']}")
        logger.info(f"   • Left curve sections: {distribution['type_counts']['left_curve']}")
        logger.info(f"   • Right curve sections: {distribution['type_counts']['right_curve']}")
        logger.info(f"   • Potential comparisons (excluding straight sections): {expected_comparisons}")
        logger.info("Performing section matching analysis...")

        # Compare all sections pairwise
        section_matches = {}
        actual_comparisons = 0  # Track only comparisons that are actually performed
        last_percentage = -1
    
        for i in range(len(all_sections)):
            section1 = all_sections[i]
            s1_key = section1['section_id']
        
            for j in range(i+1, len(all_sections)):
                section2 = all_sections[j]
                s2_key = section2['section_id']
            
                # Skip if same road
                if section1['road_id'] == section2['road_id']:
                    continue
            
                # Get enhanced section types for better classification
                features1 = self._get_section_features(s1_key)
                features2 = self._get_section_features(s2_key)
            
                # Skip if different enhanced types (straight, left_curve, right_curve)
                if features1['type'] != features2['type']:
                    continue

                # OPTIMIZATION: Skip straight section comparisons as they lack discriminative patterns
                # Straight sections are important for coverage but not for finding unique matches
                if features1['type'] == 'straight':
                    continue

                # Count only comparisons that actually get processed
                actual_comparisons += 1
                if actual_comparisons % 100 == 0:  # Update less frequently
                    current_percentage = (actual_comparisons * 100) // expected_comparisons
                    if current_percentage > last_percentage:
                        logger.info(f"Progress: {current_percentage}% ({actual_comparisons}/{expected_comparisons} curve-only comparisons)")
                        last_percentage = current_percentage

                # Try full section match first
                norm_dtw = self._calculate_normalized_dtw(
                    section1['segment'].curvature_profile,
                    section2['segment'].curvature_profile
                )
            
                if norm_dtw is None:
                    continue
            
                similarity = 1 - norm_dtw
                matches_found = []
            
                # If full section match is good enough, use it
                if similarity >= self.dtw_similarity_threshold:
                    matches_found.append({
                        'matched_section_id': s2_key,
                        'matched_road_id': section2['road_id'],
                        'similarity_score': similarity,
                        'match_type': 'full_section',
                        'point_range': (section1['segment'].start_index, section1['segment'].end_index),
                        'matched_point_range': (section2['segment'].start_index, section2['segment'].end_index)
                    })
                else:
                    # Try subsection matching: find if one complete section can be found within another
                    # (This follows the cleaned road analysis logic)
                    curv1 = section1['segment'].curvature_profile
                    curv2 = section2['segment'].curvature_profile
                
                    # Determine which is shorter and which is longer
                    if len(curv1) < len(curv2):
                        shorter_curv, longer_curv = curv1, curv2
                        shorter_section, longer_section = section1, section2
                        shorter_key, longer_key = s1_key, s2_key
                        is_section1_shorter = True
                    elif len(curv2) < len(curv1):
                        shorter_curv, longer_curv = curv2, curv1
                        shorter_section, longer_section = section2, section1
                        shorter_key, longer_key = s2_key, s1_key
                        is_section1_shorter = False
                    else:
                        # Same length, skip subsection matching
                        continue
                
                    # Skip if sections are too similar in length (should be full matches)
                    length_ratio = len(shorter_curv) / len(longer_curv)
                    if length_ratio > 0.8:
                        continue
                
                    # Skip if shorter section is too small
                    if len(shorter_curv) < self.min_subsection_length:
                        continue
                
                    # Find best position for the shorter section within the longer section
                    best_match = None
                    best_similarity = 0
                    shorter_len = len(shorter_curv)
                
                    # Slide the complete shorter section along the longer section
                    for start_pos in range(len(longer_curv) - shorter_len + 1):
                        longer_segment = longer_curv[start_pos:start_pos + shorter_len]
                    
                        # Validate subsection match for curved sections
                        if shorter_section['segment'].segment_type == 'curved':
                            if not (np.all(np.abs(shorter_curv) >= self.curvature_threshold) and
                                np.all(np.abs(longer_segment) >= self.curvature_threshold)):
                                continue
                    
                        # Check sign compatibility for curved sections
                        if shorter_section['segment'].segment_type == 'curved':
                            mean_short, mean_long = np.mean(shorter_curv), np.mean(longer_segment)
                            if (abs(mean_short) >= self.curvature_threshold and
                                abs(mean_long) >= self.curvature_threshold and
                                np.sign(mean_short) != np.sign(mean_long)):
                                continue
                    
                        # Calculate DTW similarity
                        norm_dtw = self._calculate_normalized_dtw(shorter_curv, longer_segment)
                        if norm_dtw is None or norm_dtw > 0.02:  # Use subsection threshold
                            continue
                    
                        similarity = 1 - norm_dtw
                        if similarity >= self.dtw_similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                        
                            # Create the match based on which section is shorter
                            if is_section1_shorter:
                                # section1 is shorter, found within section2
                                best_match = {
                                    'matched_section_id': s2_key,
                                    'matched_road_id': section2['road_id'],
                                    'similarity_score': similarity,
                                    'match_type': 'subsection',
                                    'point_range': (section1['segment'].start_index, section1['segment'].end_index),
                                    'matched_point_range': (
                                        section2['segment'].start_index + start_pos,
                                        section2['segment'].start_index + start_pos + shorter_len - 1
                                    ),
                                    'subsection_info': {
                                        'complete_section': s1_key,
                                        'within_section': s2_key,
                                        'position_in_longer': start_pos
                                    }
                                }
                            else:
                                # section2 is shorter, found within section1
                                best_match = {
                                    'matched_section_id': s2_key,
                                    'matched_road_id': section2['road_id'],
                                    'similarity_score': similarity,
                                    'match_type': 'subsection',
                                    'point_range': (
                                        section1['segment'].start_index + start_pos,
                                        section1['segment'].start_index + start_pos + shorter_len - 1
                                    ),
                                    'matched_point_range': (section2['segment'].start_index, section2['segment'].end_index),
                                    'subsection_info': {
                                        'complete_section': s1_key,
                                        'within_section': s2_key,
                                        'position_in_longer': start_pos
                                    }
                                }
                
                    # Add the best subsection match if found
                    if best_match is not None:
                        matches_found.append(best_match)
            
                # If we found any matches, record them
                if matches_found:
                    # Sort matches by similarity score and keep the best non-overlapping ones
                    matches_found.sort(key=lambda x: x['similarity_score'], reverse=True)
                    best_matches = []
                    used_ranges1 = set()
                    used_ranges2 = set()
                
                    for match in matches_found:
                        range1 = set(range(match['point_range'][0], match['point_range'][1] + 1))
                        range2 = set(range(match['matched_point_range'][0], match['matched_point_range'][1] + 1))
                    
                        if not (range1 & used_ranges1 or range2 & used_ranges2):  # No overlap
                            best_matches.append(match)
                            used_ranges1.update(range1)
                            used_ranges2.update(range2)
                
                    # Record the best matches
                    if best_matches:  # Only create entries if we have matches
                        if s1_key not in section_matches:
                            section_matches[s1_key] = {
                                'road_id': section1['road_id'],
                                'section_idx': i,
                                'matches': []
                            }
                        if s2_key not in section_matches:
                            section_matches[s2_key] = {
                                'road_id': section2['road_id'],
                                'section_idx': j,
                                'matches': []
                            }
                    
                        # Record matches in both directions
                        for match in best_matches:
                            section_matches[s1_key]['matches'].append(match)
                            # Create reverse match
                            reverse_match = match.copy()
                            reverse_match['matched_section_id'] = s1_key
                            reverse_match['matched_road_id'] = section1['road_id']
                            reverse_match['point_range'], reverse_match['matched_point_range'] = \
                                match['matched_point_range'], match['point_range']
                            section_matches[s2_key]['matches'].append(reverse_match)
                    
                        # Update matched status in registry
                        self.section_registry[s1_key]['matched'] = True
                        self.section_registry[s2_key]['matched'] = True
    
        return section_matches

    def _calculate_normalized_dtw(self, curv1: np.ndarray, curv2: np.ndarray) -> Optional[float]:
        """Calculate normalized DTW distance with sign awareness"""
        # Ensure inputs are numpy arrays
        curv1 = np.array(curv1) if not isinstance(curv1, np.ndarray) else curv1
        curv2 = np.array(curv2) if not isinstance(curv2, np.ndarray) else curv2
        
        if len(curv1) < self.min_subsection_length or len(curv2) < self.min_subsection_length:
            return None
    
        # Check sign compatibility
        mean1, mean2 = np.mean(curv1), np.mean(curv2)
        is_straight1 = abs(mean1) < self.curvature_threshold
        is_straight2 = abs(mean2) < self.curvature_threshold
    
        if not (is_straight1 and is_straight2) and not is_straight1 and not is_straight2:
            if np.sign(mean1) != np.sign(mean2):
                return float('inf')
    
        try:
            dtw_dist, _ = fastdtw(curv1.reshape(-1, 1), curv2.reshape(-1, 1), dist=euclidean)
            avg_length = (len(curv1) + len(curv2)) / 2
            return dtw_dist / avg_length if avg_length > 0 else None
        except Exception as e:
            logger.warning(f"DTW calculation failed: {e}")
            return None

    def analyze_section_distribution(self) -> dict:
        """Analyze the distribution of section types and calculate potential comparison counts"""
        type_counts = {'straight': 0, 'left_curve': 0, 'right_curve': 0}
    
        # Count sections by enhanced type
        for section_id in self.section_registry:
            features = self._get_section_features(section_id)
            section_type = features['type']
            type_counts[section_type] += 1
    
        # Calculate potential comparison counts for each type
        # NOTE: Straight sections are skipped in comparison for optimization
        comparison_counts = {}
        for section_type, count in type_counts.items():
            if section_type == 'straight':
                # Straight sections are skipped in comparison but counted for coverage
                potential_comparisons = 0
            else:
                # Only curves are compared: n * (n-1) / 2 for unique pairs
                potential_comparisons = count * (count - 1) // 2 if count > 1 else 0
            comparison_counts[section_type] = potential_comparisons
    
        total_sections = sum(type_counts.values())
        total_potential_comparisons = sum(comparison_counts.values())
    
        return {
            'total_sections': total_sections,
            'type_counts': type_counts,
            'potential_comparisons': comparison_counts,
            'total_potential_comparisons': total_potential_comparisons
        }

    def _get_section_features(self, section_id):
        """Get enhanced features for a section with optimized type detection"""
        data = self.section_registry[section_id]
        profile = np.array(data['curvature_profile'])
    
        # Enhanced curvature direction detection
        mean_curv = np.mean(profile)
        abs_mean_curv = np.abs(mean_curv)
        
        # Optimized type detection
        if abs_mean_curv < self.curvature_threshold:
            curvature_type = 'straight'
        else:
            curvature_type = 'left_curve' if mean_curv > 0 else 'right_curve'
        
        return {
            'type': curvature_type,
            'length': round(data['length'], 3),
            'mean_curvature': mean_curv,
            'std_curvature': round(float(np.std(profile)), 5),
            'max_curvature': round(float(np.max(np.abs(profile))), 5),
            'sign_changes': np.sum(np.diff(np.sign(profile[profile != 0])) != 0)
        }