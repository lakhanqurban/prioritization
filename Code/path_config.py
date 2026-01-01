from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathConfig:
    """Centralized path configuration for the road analysis system"""
    
    # Base directories
    base_output_dir: str = "./output"
    base_senso_dir: str = "./SensoDat"
    
    # Road data paths
    road_data_dir: str = "./SensoDat/roads/a3"
    dynamic_data_dir: str = "./SensoDat/roads_dynamic_data/a3"
    failed_roads_dir: str = "./SensoDat/failed_data"
    
    # Section-related paths
    sections_output_dir: str = "./output/a3_sections"
    section_registry_file: str = "section_registry.json"
    road_metadata_file: str = "road_metadata.json"
    
    # Matching-related paths  
    matching_output_dir: str = "./output/a3_matching_info"
    matched_sections_file: str = "matched_sections.json"
    unmatched_sections_file: str = "unmatched_sections.json"
    
    # Analysis output paths
    coverage_reduction_dir: str = "./output/a3_dp_tsp_coverage_based_reduction"
    visualization_dir: str = "./output"
    
    def __post_init__(self):
        """Convert string paths to Path objects and create property accessors"""
        # Convert to Path objects
        self.base_output_path = Path(self.base_output_dir)
        self.base_senso_path = Path(self.base_senso_dir)
        self.road_data_path = Path(self.road_data_dir)
        self.dynamic_data_path = Path(self.dynamic_data_dir)
        self.failed_roads_path = Path(self.failed_roads_dir)
        self.sections_output_path = Path(self.sections_output_dir)
        self.matching_output_path = Path(self.matching_output_dir)
        self.coverage_reduction_path = Path(self.coverage_reduction_dir)
        self.visualization_path = Path(self.visualization_dir)
    
    @property
    def section_registry_path(self) -> Path:
        """Full path to section registry file"""
        return self.sections_output_path / self.section_registry_file
    
    @property
    def road_metadata_path(self) -> Path:
        """Full path to road metadata file"""
        return self.sections_output_path / self.road_metadata_file
    
    @property
    def matched_sections_path(self) -> Path:
        """Full path to matched sections file"""
        return self.matching_output_path / self.matched_sections_file
    
    @property
    def unmatched_sections_path(self) -> Path:
        """Full path to unmatched sections file"""
        return self.matching_output_path / self.unmatched_sections_file
    
    def create_directories(self):
        """Create all necessary output directories"""
        for path in [self.base_output_path, self.sections_output_path, 
                     self.matching_output_path, self.coverage_reduction_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def create_custom_config(cls, 
                           base_output_dir: str = None,
                           road_data_dir: str = None,
                           dynamic_data_dir: str = None,
                           failed_roads_dir: str = None,
                           coverage_reduction_dir: str = None,
                           sections_subdir: str = "a3_sections",
                           matching_subdir: str = "a3_matching_info") -> "PathConfig":
        """
        Create a custom PathConfig with user-specified paths.
        
        Args:
            base_output_dir: Base output directory (default: "./output")
            road_data_dir: Road data directory (default: "./SensoDat/roads/campaign_2_frenetic_road_data")
            dynamic_data_dir: Dynamic data directory (default: "./SensoDat/roads_dynamic_data/")
            failed_roads_dir: Failed roads directory (default: "./SensoDat/failed")
            sections_subdir: Subdirectory for sections (default: "sections")
            matching_subdir: Subdirectory for matching info (default: "matching_info")
            
        Returns:
            PathConfig instance with custom paths
        """
        config = cls()
        
        if base_output_dir:
            config.base_output_dir = base_output_dir
            config.sections_output_dir = f"{base_output_dir}/{sections_subdir}"
            config.matching_output_dir = f"{base_output_dir}/{matching_subdir}"
            # config.coverage_reduction_dir = f"{base_output_dir}/a10_dp4_coverage_based_reduction" 
            config.coverage_reduction_dir = f"{base_output_dir}/{coverage_reduction_dir}" 

            config.visualization_dir = base_output_dir
            
        if road_data_dir:
            config.road_data_dir = road_data_dir
            
        if dynamic_data_dir:
            config.dynamic_data_dir = dynamic_data_dir
            
        if failed_roads_dir:
            config.failed_roads_dir = failed_roads_dir
            
        # Re-initialize paths
        config.__post_init__()
        return config