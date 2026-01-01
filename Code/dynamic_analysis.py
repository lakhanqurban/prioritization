import math
import logging
import csv
from typing import List, Dict, Optional, Any
import numpy as np
from data_classes import DynamicMetrics

logger = logging.getLogger(__name__)

class DynamicAnalysisMixin:
    """Mixin class for dynamic vehicle data analysis"""
    
    def load_dynamic_data(self, road_id: str) -> Optional[List[Dict]]:
        """
        Load dynamic vehicle data for a specific road.
        
        Args:
            road_id: Road identifier (e.g., "0", "1", etc.)
            
        Returns:
            List of dictionaries with dynamic data or None if file doesn't exist
        """
        csv_path = self.paths.dynamic_data_path / f"{road_id}.csv"
        
        if not csv_path.exists():
            logger.warning(f"Dynamic data file not found: {csv_path}")
            return None
        
        try:
            data = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values to float
                    row_float = {}
                    for key, value in row.items():
                        try:
                            row_float[key] = float(value)
                        except ValueError:
                            row_float[key] = value
                    data.append(row_float)
            
            logger.debug(f"Loaded dynamic data for road {road_id}: {len(data)} records")
            return data
        except Exception as e:
            logger.error(f"Error loading dynamic data for road {road_id}: {e}")
            return None

    def calculate_statistics(self, values: List[float]) -> Dict:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        
        mean = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        
        return {
            'mean': mean,
            'max': max_val,
            'min': min_val,
            'std': std
        }

    def calculate_dynamic_metrics(self, road_id: str) -> Optional[DynamicMetrics]:
        """
        Calculate dynamic metrics for a road from its vehicle simulation data.
        
        Args:
            road_id: Road identifier
            
        Returns:
            DynamicMetrics object or None if data unavailable
        """
        # if road_id in self.dynamic_metrics_cache:
        #     return self.dynamic_metrics_cache[road_id]
        
        data = self.load_dynamic_data(road_id)
        if data is None:
            return None
        
        try:
            # Extract columns
            speeds = [row['speed'] for row in data]
            steering_angles = [row['str_angle'] for row in data]
            ctes = [row['cte'] for row in data]
            heading_errors = [row['hdg_err'] for row in data]
            yaws = [row['yaw'] for row in data]
            
            # Calculate statistics
            speed_stats = self.calculate_statistics(speeds)
            steering_stats = self.calculate_statistics(steering_angles)
            cte_stats = self.calculate_statistics([abs(x) for x in ctes])
            heading_stats = self.calculate_statistics([abs(x) for x in heading_errors])
            
            # Calculate yaw rate (derivative of yaw)
            yaw_rates = []
            for i in range(1, len(yaws)):
                # Calculate difference between consecutive yaw values
                diff = yaws[i] - yaws[i-1]
                # Adjust for circular nature of angles (crossing from 2π to 0)
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                yaw_rates.append(abs(diff))
                
            yaw_rate_stats = self.calculate_statistics(yaw_rates) if yaw_rates else {'mean': 0.0, 'max': 0.0, 'std': 0.0}
            
            # Count critical events
            critical_steering_events = len([x for x in steering_angles if abs(x) > 0.8])
            critical_speed_events = len([x for x in speeds if x > 30.0])
            critical_cte_events = len([x for x in ctes if abs(x) > 2.5])
            
            metrics = DynamicMetrics(
                avg_speed=speed_stats['mean'],
                max_speed=speed_stats['max'],
                avg_steering_angle=steering_stats['mean'],
                max_steering_angle=steering_stats['max'],
                avg_cross_track_error=cte_stats['mean'],
                max_cross_track_error=cte_stats['max'],
                avg_heading_error=heading_stats['mean'],
                max_heading_error=heading_stats['max'],
                steering_variation=steering_stats['std'],
                speed_variation=speed_stats['std'],
                cte_variation=cte_stats['std'],
                critical_steering_events=critical_steering_events,
                critical_speed_events=critical_speed_events,
                critical_cte_events=critical_cte_events,
                avg_yaw_rate=yaw_rate_stats['mean'],
                max_yaw_rate=yaw_rate_stats['max'],
                yaw_rate_variation=yaw_rate_stats['std']
            )
            
            # self.dynamic_metrics_cache[road_id] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating dynamic metrics for road {road_id}: {e}")
            return None

    def get_road_dynamic_score(self, road_id: str) -> float:
        """
        Calculate a dynamic score for a road based on vehicle behavior complexity.
        
        Args:
            road_id: Road identifier
            
        Returns:
            Dynamic score (higher = more challenging/complex driving)
        """
        metrics = self.calculate_dynamic_metrics(road_id)
        if metrics is None:
            return 0.0
        
        # Weighted scoring based on dynamic behavior
        score = 0.0
        
        # Steering complexity (weight: 1.0)
        steering_score = (metrics.steering_variation * 1.0 + 
                         abs(metrics.max_steering_angle) * 1.0 +
                         metrics.critical_steering_events * 1.0)
        score += steering_score
        
        # Speed complexity (weight: 1.0)
        speed_score = (metrics.speed_variation * 1.0 + 
                      metrics.max_speed * 1.0 +
                      metrics.critical_speed_events * 1.0)
        score += speed_score
        
        # Control difficulty (weight: 1.0)
        control_score = (metrics.avg_cross_track_error * 1.0 + 
                        metrics.max_cross_track_error * 1.0 +
                        metrics.critical_cte_events * 1.0)
        score += control_score
        
        # Heading control difficulty (weight: 1.0)
        heading_score = (metrics.avg_heading_error * 1.0 + 
                        metrics.max_heading_error * 1.0)
        score += heading_score
        
        # Yaw dynamics (weight: 1.0)
        yaw_score = (metrics.avg_yaw_rate * 1.0 + 
                    metrics.max_yaw_rate * 1.0 +
                    metrics.yaw_rate_variation * 1.0)
        score += yaw_score
        
        return score

    def analyze_dynamic_coverage(self, selected_roads: List[str]) -> Dict[str, Any]:
        """
        Analyze the dynamic behavior coverage of selected roads.
        
        Args:
            selected_roads: List of selected road IDs
            
        Returns:
            Dictionary with dynamic coverage analysis
        """
        if not self.dynamic_data_available:
            return {'dynamic_data_available': False}
        
        analysis = {
            'dynamic_data_available': True,
            'total_roads': len(selected_roads),
            'roads_with_dynamic_data': 0,
            'dynamic_scores': {},
            'avg_dynamic_score': 0.0,
            'max_dynamic_score': 0.0,
            'dynamic_score_distribution': {}
        }
        
        dynamic_scores = []
        
        for road_id in selected_roads:
            dynamic_score = self.get_road_dynamic_score(road_id)
            if dynamic_score > 0:
                analysis['roads_with_dynamic_data'] += 1
                analysis['dynamic_scores'][road_id] = dynamic_score
                dynamic_scores.append(dynamic_score)
        
        if dynamic_scores:
            analysis['avg_dynamic_score'] = sum(dynamic_scores) / len(dynamic_scores)
            analysis['max_dynamic_score'] = max(dynamic_scores)
            
            # Score distribution
            score_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
            for low, high in score_ranges:
                count = len([s for s in dynamic_scores if low <= s < high])
                analysis['dynamic_score_distribution'][f"{low}-{high if high != float('inf') else '∞'}"] = count
        
        return analysis

    def _calculate_dynamic_distance(self, section_id1: str, section_id2: str) -> float:
        """
        Calculate dynamic distance between two sections based on vehicle behavior data.
        Ensures proper alignment between section boundaries and dynamic data indices.
        Uses caching to avoid repeated data loading.
        
        Args:
            section_id1: First section identifier
            section_id2: Second section identifier
            
        Returns:
            Normalized dynamic distance (0-1 scale)
        """
        try:
            # Get section data to determine road_id and index ranges
            section1_data = self.section_registry[section_id1]
            section2_data = self.section_registry[section_id2]
            
            road_id1 = section1_data['road_id']
            road_id2 = section2_data['road_id']
            
            # Use cached dynamic data if available
            if not hasattr(self, '_dynamic_data_cache'):
                self._dynamic_data_cache = {}
                logger.debug("Initialized dynamic data cache")
            # logger.debug(f"Calculating dynamic distance for sections {section_id1} and {section_id2}")
            
            # Load and cache dynamic data for road 1
            if road_id1 not in self._dynamic_data_cache:
            
                self._dynamic_data_cache[road_id1] = self.load_dynamic_data(road_id1)
            dynamic_data1 = self._dynamic_data_cache[road_id1]
            
            # Load and cache dynamic data for road 2
            if road_id2 not in self._dynamic_data_cache:
                self._dynamic_data_cache[road_id2] = self.load_dynamic_data(road_id2)
            dynamic_data2 = self._dynamic_data_cache[road_id2]
            
            if dynamic_data1 is None or dynamic_data2 is None:
                return 0.0  # No dynamic data available, return neutral distance
            
            # Extract section-specific dynamic data using start_idx and end_idx
            start_idx1 = section1_data['start_idx']
            end_idx1 = section1_data['end_idx']
            start_idx2 = section2_data['start_idx']
            end_idx2 = section2_data['end_idx']
            
            # Filter dynamic data to section boundaries (using idx column for alignment)
            section1_dynamic = [row for row in dynamic_data1 
                              if start_idx1 <= row.get('idx', 0) <= end_idx1]
            section2_dynamic = [row for row in dynamic_data2 
                              if start_idx2 <= row.get('idx', 0) <= end_idx2]
            
            if not section1_dynamic or not section2_dynamic:
                return 0.0  # No aligned dynamic data, return neutral distance
            
            # Calculate dynamic features for each section
            features1 = self._extract_section_dynamic_features(section1_dynamic)
            features2 = self._extract_section_dynamic_features(section2_dynamic)
            
            # Calculate normalized distance between dynamic features
            distance = self._compute_dynamic_feature_distance(features1, features2)
            
            return distance
            
        except Exception as e:
            logger.debug(f"Error calculating dynamic distance for {section_id1}, {section_id2}: {e}")
            return 0.0  # Return neutral distance on error

    def _extract_section_dynamic_features(self, dynamic_data: List[Dict]) -> Dict[str, float]:
        """
        Extract dynamic behavioral features from section-specific dynamic data.
        
        Args:
            dynamic_data: List of dynamic data points for the section
            
        Returns:
            Dictionary of normalized dynamic features
        """
        if not dynamic_data:
            return {'speed_var': 0.0, 'steering_var': 0.0, 'cte_severity': 0.0, 'yaw_rate_var': 0.0}
        
        # Extract relevant columns
        speeds = [row.get('speed', 0) for row in dynamic_data]
        steering_angles = [row.get('str_angle', 0) for row in dynamic_data]
        ctes = [abs(row.get('cte', 0)) for row in dynamic_data]
        yaws = [row.get('yaw', 0) for row in dynamic_data]
        
        # Calculate variability features (indicating driving difficulty)
        speed_var = np.std(speeds) if len(speeds) > 1 else 0.0
        steering_var = np.std(steering_angles) if len(steering_angles) > 1 else 0.0
        cte_severity = np.mean(ctes) if ctes else 0.0
        
        # Calculate yaw rate variability
        yaw_rates = []
        for i in range(1, len(yaws)):
            diff = yaws[i] - yaws[i-1]
            # Handle angle wraparound
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            yaw_rates.append(abs(diff))
        
        yaw_rate_var = np.std(yaw_rates) if len(yaw_rates) > 1 else 0.0
        
        return {
            'speed_var': speed_var,
            'steering_var': steering_var,
            'cte_severity': cte_severity,
            'yaw_rate_var': yaw_rate_var
        }

    def _compute_dynamic_feature_distance(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Compute normalized distance between two sets of dynamic features.
        
        Args:
            features1: Dynamic features for first section
            features2: Dynamic features for second section
            
        Returns:
            Normalized distance (0-1 scale)
        """
        # Define normalization scales for each feature (based on analysis of actual dynamic data)
        normalizers = {
            'speed_var': 8.0,       # Based on 95th percentile of speed variance across roads (7.2 + margin)
            'steering_var': 0.35,   # Based on 95th percentile of steering variance across roads (0.292 + margin)
            'cte_severity': 1.0,    # Based on 95th percentile of CTE severity across roads (≈0.85) + margin
            'yaw_rate_var': 0.05    # Based on 95th percentile of yaw rate variance across roads (0.033 + margin)
        }
        
        total_distance = 0.0
        feature_count = len(features1)
        
        for feature_name in features1:
            if feature_name in features2:
                # Calculate normalized difference
                diff = abs(features1[feature_name] - features2[feature_name])
                denom = normalizers.get(feature_name, 1.0)
                if denom <= 0:
                    # Defensive fallback to avoid division by zero; treat any non-zero diff as maximal
                    normalized_diff = 0.0 if diff == 0 else 1.0
                else:
                    normalized_diff = min(diff / denom, 1.0)
                total_distance += normalized_diff
        
        # Return average normalized distance
        return total_distance / feature_count if feature_count > 0 else 0.0