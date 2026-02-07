#!/usr/bin/env python3
"""
Quick test script for refactored pipeline.

Tests basic functionality without running full pipeline.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from domain.config import (
    PipelineConfig, TaskConfig, ParsingConfig
)
from orchestration.states import PipelineState
from orchestration.context import PipelineContext
from services.metadata.fetcher import MetadataFetcher
from services.metadata.cache import MetadataCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_config_creation():
    """Test that we can create a valid configuration."""
    print("\n=== Testing Configuration Creation ===")
    
    config = PipelineConfig(
        run_name="test_run",
        tasks=TaskConfig(do_parsing=True),
        parsing_config=ParsingConfig(
            output_path="./output",
            file_urls_path="./cache.json",
            jobs_logs_path="./logs",
            parse_mc=False,
            count_retries_failed_files=0,
            fetching_metadata_timeout=10,
            env_threshold_memory_mb=1000,
            chunk_yield_threshold_bytes=1024 * 1024,
            threads=2,
            specific_record_ids=None,
            release_years=("2024r-pp",),
            possible_data_tree_names=("CollectionTree",),
            create_dirs=False,
            show_progress_bar=False
        )
    )
    
    print(f"✓ Created config: {config.run_name}")
    print(f"✓ Parsing enabled: {config.tasks.do_parsing}")
    print(f"✓ Release years: {list(config.parsing_config.release_years)}")
    return config


def test_context_creation(config):
    """Test that we can create a pipeline context."""
    print("\n=== Testing Context Creation ===")
    
    context = PipelineContext(
        config=config,
        current_state=PipelineState.IDLE
    )
    
    print(f"✓ Created context in state: {context.current_state}")
    print(f"✓ Context is terminal: {context.is_terminal}")
    print(f"✓ Context is successful: {context.is_successful}")
    
    # Test immutability
    new_context = context.with_state(PipelineState.FETCHING_METADATA)
    assert context.current_state == PipelineState.IDLE
    assert new_context.current_state == PipelineState.FETCHING_METADATA
    print("✓ Context immutability works correctly")
    
    return context


def test_services():
    """Test that services can be instantiated."""
    print("\n=== Testing Services ===")
    
    fetcher = MetadataFetcher(timeout=10, show_progress=False)
    print(f"✓ Created MetadataFetcher with timeout: {fetcher.timeout}s")
    
    cache = MetadataCache(cache_path="./test_cache.json", max_wait_time=10)
    print(f"✓ Created MetadataCache at: {cache.cache_path}")
    
    return fetcher, cache


def test_state_transitions():
    """Test state transition logic."""
    print("\n=== Testing State Transitions ===")
    
    from orchestration.states import is_valid_transition
    
    # Valid transitions
    assert is_valid_transition(PipelineState.IDLE, PipelineState.FETCHING_METADATA)
    assert is_valid_transition(PipelineState.FETCHING_METADATA, PipelineState.PARSING)
    assert is_valid_transition(PipelineState.PARSING, PipelineState.COMPLETED)
    print("✓ Valid transitions work")
    
    # Invalid transitions
    assert not is_valid_transition(PipelineState.COMPLETED, PipelineState.PARSING)
    assert not is_valid_transition(PipelineState.FAILED, PipelineState.FETCHING_METADATA)
    print("✓ Invalid transitions are rejected")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Refactored Pipeline - Quick Test")
    print("=" * 60)
    
    try:
        # Test 1: Config
        config = test_config_creation()
        
        # Test 2: Context
        context = test_context_creation(config)
        
        # Test 3: Services
        fetcher, cache = test_services()
        
        # Test 4: State transitions
        test_state_transitions()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
