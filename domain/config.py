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
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Immutable configuration object validated at creation.
    """
    
    # Task configuration
    tasks: TaskConfig
    
    # Stage configurations
    parsing_config: Optional[ParsingConfig] = None
    
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
        
        if self.batch_job_index is not None:
            if self.batch_job_index < 0:
                raise ValueError(f"batch_job_index must be non-negative, got {self.batch_job_index}")
            if self.total_batch_jobs is None:
                raise ValueError("total_batch_jobs required when batch_job_index is set")
            if self.batch_job_index >= self.total_batch_jobs:
                raise ValueError(
                    f"batch_job_index ({self.batch_job_index}) must be less than "
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
            parsing_config = ParsingConfig(
                output_path=parsing_dict["output_path"],
                file_urls_path=parsing_dict["file_urls_path"],
                jobs_logs_path=parsing_dict["jobs_logs_path"],
                release_years=tuple(parsing_dict.get("release_years", [])),
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
            )
        
        # Parse run metadata
        run_metadata = config_dict.get("run_metadata", {})
        
        return cls(
            tasks=tasks,
            parsing_config=parsing_config,
            run_name=run_metadata.get("run_name", "pipeline_run"),
            batch_job_index=run_metadata.get("batch_job_index"),
            total_batch_jobs=run_metadata.get("total_batch_jobs"),
        )
