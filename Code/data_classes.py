from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DynamicMetrics:
    """Data class to store dynamic vehicle metrics"""
    avg_speed: float
    max_speed: float
    avg_steering_angle: float
    max_steering_angle: float
    avg_cross_track_error: float
    max_cross_track_error: float
    avg_heading_error: float
    max_heading_error: float
    steering_variation: float
    speed_variation: float
    cte_variation: float
    critical_steering_events: int
    critical_speed_events: int
    critical_cte_events: int
    avg_yaw_rate: float = 0.0
    max_yaw_rate: float = 0.0
    yaw_rate_variation: float = 0.0

@dataclass
class RoadSegment:
    """Data class to store road segment information"""
    segment_type: str
    points: List[Tuple[float, float]]
    start_index: int
    end_index: int
    length: float
    curvature_profile: np.ndarray
    matched: bool = False