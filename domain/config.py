"""
Configuration domain models.

Validated configuration objects for the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for which tasks to run."""
    
    do_parsing: bool = False
    do_mass_calculating: bool = False
    do_post_processing: bool = False
    do_histogram_creation: bool = False
    
    def any_enabled(self) -> bool:
        """Check if any task is enabled."""
        return any([
            self.do_parsing,
            self.do_mass_calculating,
            self.do_post_processing,
            self.do_histogram_creation
        ])


@dataclass(frozen=True)
class ParsingConfig:
    """Configuration for parsing task."""
    
    # Paths
    output_path: str
    file_urls_path: str
    jobs_logs_path: str
    
    # Processing
    release_years: tuple[str, ...] = field(default_factory=tuple)
    specific_record_ids: Optional[tuple[int, ...]] = None
    parse_mc: bool = False
    
    # Performance
    threads: int = 8
    chunk_yield_threshold_bytes: int = 5_000_000_000  # 5GB default
    env_threshold_memory_mb: int = 20_000  # 20GB default
    
    # Behavior
    create_dirs: bool = False
    show_progress_bar: bool = True
    count_retries_failed_files: int = 3
    fetching_metadata_timeout: int = 60
    
    # Data selection
    possible_data_tree_names: tuple[str, ...] = ("CollectionTree",)
    max_files_to_process: Optional[int] = None  # Limit files (for testing)
    
    def __post_init__(self):
        """Validate parsing configuration."""
        if self.threads <= 0:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if self.chunk_yield_threshold_bytes <= 0:
            raise ValueError(f"chunk_yield_threshold_bytes must be positive, got {self.chunk_yield_threshold_bytes}")
        if self.env_threshold_memory_mb <= 0:
            raise ValueError(f"env_threshold_memory_mb must be positive, got {self.env_threshold_memory_mb}")
        if self.count_retries_failed_files < 0:
            raise ValueError(f"count_retries_failed_files must be non-negative, got {self.count_retries_failed_files}")
        if not self.output_path:
            raise ValueError("output_path cannot be empty")
        if not self.file_urls_path:
            raise ValueError("file_urls_path cannot be empty")
        if not self.jobs_logs_path:
            raise ValueError("jobs_logs_path cannot be empty")


@dataclass(frozen=True)
class MassCalculationConfig:
    """Configuration for mass calculation task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing
    field_to_slice_by: str = "pt"  # Which kinematic field to rank/slice particles by
    use_multiprocessing: bool = True
    parallel_processes: int = 4
    fs_chunk_threshold_bytes: int = 500_000_000  # 500MB - chunk threshold for final-state files
    
    # Combinatorics configuration
    objects_to_calculate: tuple[str, ...] = ("Electrons", "Muons", "Jets", "Photons")
    min_particles_in_combination: int = 2
    max_particles_in_combination: int = 4
    min_count_particle_in_combination: int = 2
    max_count_particle_in_combination: int = 4
    min_events_per_fs: int = 100  # Minimum events for a final state to be kept
    
    def __post_init__(self):
        """Validate mass calculation configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.parallel_processes <= 0:
            raise ValueError(f"parallel_processes must be positive, got {self.parallel_processes}")
        if self.min_particles_in_combination < 1:
            raise ValueError(f"min_particles_in_combination must be >= 1, got {self.min_particles_in_combination}")
        if self.max_particles_in_combination < self.min_particles_in_combination:
            raise ValueError(
                f"max_particles_in_combination ({self.max_particles_in_combination}) must be >= "
                f"min_particles_in_combination ({self.min_particles_in_combination})"
            )
        if self.min_count_particle_in_combination < 1:
            raise ValueError(f"min_count_particle_in_combination must be >= 1, got {self.min_count_particle_in_combination}")
        if self.max_count_particle_in_combination < self.min_count_particle_in_combination:
            raise ValueError(
                f"max_count_particle_in_combination ({self.max_count_particle_in_combination}) must be >= "
                f"min_count_particle_in_combination ({self.min_count_particle_in_combination})"
            )
        if self.fs_chunk_threshold_bytes <= 0:
            raise ValueError(f"fs_chunk_threshold_bytes must be positive, got {self.fs_chunk_threshold_bytes}")
        if not self.objects_to_calculate:
            raise ValueError("objects_to_calculate cannot be empty")
        if self.min_events_per_fs < 0:
            raise ValueError(f"min_events_per_fs must be non-negative, got {self.min_events_per_fs}")


@dataclass(frozen=True)
class PostProcessingConfig:
    """Configuration for post-processing task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing parameters
    peak_detection_bin_width_gev: float = 10.0
    
    def __post_init__(self):
        """Validate post-processing configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.peak_detection_bin_width_gev <= 0:
            raise ValueError(f"peak_detection_bin_width_gev must be positive, got {self.peak_detection_bin_width_gev}")


@dataclass(frozen=True)
class HistogramCreationConfig:
    """Configuration for histogram creation task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing parameters
    bin_width_gev: float = 10.0
    single_output_file: bool = False
    output_filename: Optional[str] = None
    
    # Outlier and naming
    exclude_outliers: bool = True  # Whether to exclude outlier histograms
    use_bumpnet_naming: bool = False  # When true, use mass_<combo>_cat_<final_state> naming
    
    def __post_init__(self):
        """Validate histogram creation configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.bin_width_gev <= 0:
            raise ValueError(f"bin_width_gev must be positive, got {self.bin_width_gev}")
        if self.single_output_file and not self.output_filename:
            raise ValueError("output_filename required when single_output_file=True")


@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Immutable configuration object validated at creation.
    """
    
    # Task configuration
    tasks: TaskConfig
    
    # Stage configurations
    parsing_config: Optional[ParsingConfig] = None
    mass_calculation_config: Optional[MassCalculationConfig] = None
    post_processing_config: Optional[PostProcessingConfig] = None
    histogram_creation_config: Optional[HistogramCreationConfig] = None
    
    # Run metadata
    run_name: str = "pipeline_run"
    batch_job_index: Optional[int] = None
    total_batch_jobs: Optional[int] = None
    
    def __post_init__(self):
        """Validate pipeline configuration."""
        if not self.tasks.any_enabled():
            raise ValueError("At least one task must be enabled")
        
        if self.tasks.do_parsing and not self.parsing_config:
            raise ValueError("parsing_config required when do_parsing=True")
        
        if self.tasks.do_mass_calculating and not self.mass_calculation_config:
            raise ValueError("mass_calculation_config required when do_mass_calculating=True")
        
        if self.tasks.do_post_processing and not self.post_processing_config:
            raise ValueError("post_processing_config required when do_post_processing=True")
        
        if self.tasks.do_histogram_creation and not self.histogram_creation_config:
            raise ValueError("histogram_creation_config required when do_histogram_creation=True")
        
        if self.batch_job_index is not None:
            if self.batch_job_index < 1:
                raise ValueError(f"batch_job_index must be >= 1 (1-indexed), got {self.batch_job_index}")
            if self.total_batch_jobs is None:
                raise ValueError("total_batch_jobs required when batch_job_index is set")
            if self.total_batch_jobs < 1:
                raise ValueError(f"total_batch_jobs must be >= 1, got {self.total_batch_jobs}")
            if self.batch_job_index > self.total_batch_jobs:
                raise ValueError(
                    f"batch_job_index ({self.batch_job_index}) must be <= "
                    f"total_batch_jobs ({self.total_batch_jobs})"
                )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """
        Create PipelineConfig from a dictionary (e.g., loaded from YAML).
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Validated PipelineConfig instance
        """
        # Parse task config
        tasks_dict = config_dict.get("tasks", {})
        tasks = TaskConfig(
            do_parsing=tasks_dict.get("do_parsing", False),
            do_mass_calculating=tasks_dict.get("do_mass_calculating", False),
            do_post_processing=tasks_dict.get("do_post_processing", False),
            do_histogram_creation=tasks_dict.get("do_histogram_creation", False),
        )
        
        # Parse parsing config if parsing is enabled
        parsing_config = None
        if tasks.do_parsing:
            parsing_dict = config_dict.get("parsing_task_config", {})
            
            # Handle release_years (can be None, list, or empty)
            release_years_raw = parsing_dict.get("release_years")
            release_years = tuple(release_years_raw) if release_years_raw else ()
            
            parsing_config = ParsingConfig(
                output_path=parsing_dict["output_path"],
                file_urls_path=parsing_dict["file_urls_path"],
                jobs_logs_path=parsing_dict["jobs_logs_path"],
                release_years=release_years,
                specific_record_ids=tuple(parsing_dict["specific_record_ids"]) if parsing_dict.get("specific_record_ids") else None,
                parse_mc=parsing_dict.get("parse_mc", False),
                threads=parsing_dict.get("threads", 8),
                chunk_yield_threshold_bytes=parsing_dict.get("chunk_yield_threshold_bytes", 5_000_000_000),
                env_threshold_memory_mb=parsing_dict.get("env_threshold_memory_mb", 20_000),
                create_dirs=parsing_dict.get("create_dirs", False),
                show_progress_bar=parsing_dict.get("show_progress_bar", True),
                count_retries_failed_files=parsing_dict.get("count_retries_failed_files", 3),
                fetching_metadata_timeout=parsing_dict.get("fetching_metadata_timeout", 60),
                possible_data_tree_names=tuple(parsing_dict.get("possible_data_tree_names", ["CollectionTree"])),
                max_files_to_process=parsing_dict.get("max_files_to_process"),
            )
        
        # Parse mass calculation config if enabled
        mass_calculation_config = None
        if tasks.do_mass_calculating:
            mass_dict = config_dict.get("mass_calculation_task_config", {})
            
            # Handle objects_to_calculate (can be None or list)
            objects_raw = mass_dict.get("objects_to_calculate")
            objects = tuple(objects_raw) if objects_raw else ("Electrons", "Muons", "Jets", "Photons")
            
            mass_calculation_config = MassCalculationConfig(
                input_dir=mass_dict["input_dir"],
                output_dir=mass_dict["output_dir"],
                field_to_slice_by=mass_dict.get("field_to_slice_by", "pt"),
                use_multiprocessing=mass_dict.get("use_multiprocessing", True),
                parallel_processes=mass_dict.get("parallel_processes", 4),
                fs_chunk_threshold_bytes=mass_dict.get("fs_chunk_threshold_bytes", 500_000_000),
                objects_to_calculate=objects,
                min_particles_in_combination=mass_dict.get("min_particles_in_combination", 2),
                max_particles_in_combination=mass_dict.get("max_particles_in_combination", 4),
                min_count_particle_in_combination=mass_dict.get("min_count_particle_in_combination", 2),
                max_count_particle_in_combination=mass_dict.get("max_count_particle_in_combination", 4),
                min_events_per_fs=mass_dict.get("min_events_per_fs", 100),
            )
        
        # Parse post-processing config if enabled
        post_processing_config = None
        if tasks.do_post_processing:
            post_dict = config_dict.get("post_processing_task_config", {})
            post_processing_config = PostProcessingConfig(
                input_dir=post_dict["input_dir"],
                output_dir=post_dict["output_dir"],
                peak_detection_bin_width_gev=post_dict.get("peak_detection_bin_width_gev", 10.0),
            )
        
        # Parse histogram creation config if enabled
        histogram_creation_config = None
        if tasks.do_histogram_creation:
            hist_dict = config_dict.get("histogram_creation_task_config", {})
            histogram_creation_config = HistogramCreationConfig(
                input_dir=hist_dict["input_dir"],
                output_dir=hist_dict["output_dir"],
                bin_width_gev=hist_dict.get("bin_width_gev", 10.0),
                single_output_file=hist_dict.get("single_output_file", False),
                output_filename=hist_dict.get("output_filename"),
                exclude_outliers=hist_dict.get("exclude_outliers", True),
                use_bumpnet_naming=hist_dict.get("use_bumpnet_naming", False),
            )
        
        # Parse run metadata
        run_metadata = config_dict.get("run_metadata", {})
        
        return cls(
            tasks=tasks,
            parsing_config=parsing_config,
            mass_calculation_config=mass_calculation_config,
            post_processing_config=post_processing_config,
            histogram_creation_config=histogram_creation_config,
            run_name=run_metadata.get("run_name", "pipeline_run"),
            batch_job_index=run_metadata.get("batch_job_index"),
            total_batch_jobs=run_metadata.get("total_batch_jobs"),
        )
