#!/usr/bin/env python3
"""
Main entry point for refactored ATLAS pipeline.

Clean, maintainable pipeline with state machine orchestration.
"""

import sys
import logging
import argparse
import yaml
from pathlib import Path

from domain.config import PipelineConfig
from pipeline.executor import PipelineExecutor


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dict
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ATLAS Pipeline - Refactored Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main.py
  
  # Run with custom config
  python main.py --config my_config.yaml
  
  # Run with debug logging
  python main.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("ATLAS Pipeline - Refactored Architecture")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_dict = load_config(args.config)
        
        # Validate and create PipelineConfig
        config = PipelineConfig.from_dict(config_dict)
        logger.info("Configuration loaded and validated successfully")
        
        if args.dry_run:
            logger.info("Dry run mode - configuration is valid, exiting")
            logger.info(f"Enabled tasks: {[k for k, v in vars(config.tasks).items() if v]}")
            return 0
        
        # Create executor
        executor = PipelineExecutor(config)
        
        # Run pipeline
        final_context = executor.run()
        
        # Check results
        if final_context.is_successful:
            logger.info("✓ Pipeline completed successfully")
            return 0
        else:
            logger.error(f"✗ Pipeline failed: {final_context.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
