# ATLAS Pipeline - Refactored Architecture

Clean, maintainable, state-machine-based architecture for ATLAS Open Data processing.

## Architecture

### Layer 1: Domain Models (`src/domain/`)
Pure data structures with validation. No business logic.
- `events.py` - EventBatch, EventChunk
- `metadata.py` - FileMetadata, ReleaseMetadata  
- `statistics.py` - ParsingStatistics, ChunkStatistics
- `config.py` - PipelineConfig, ParsingConfig, TaskConfig

### Layer 2: Services (`src/services/`)
Single-responsibility services with no hidden dependencies.
- `parsing/file_parser.py` - Parse individual ROOT files
- `parsing/event_accumulator.py` - Accumulate events into chunks
- `metadata/fetcher.py` - Fetch metadata from ATLAS API
- `io/` - File I/O operations
- `monitoring/` - Progress tracking, statistics

### Layer 3: State Machine (`src/orchestration/`)
Clear pipeline control flow with explicit states.
- `states.py` - PipelineState enum
- `context.py` - PipelineContext (immutable state)
- `state_machine.py` - State transition logic
- `handlers/` - One handler per state

### Layer 4: Executor (`src/pipeline/`)
High-level orchestration.
- `executor.py` - PipelineExecutor (runs state machine)
- `multiprocess.py` - Multiprocessing support

## Design Principles

1. **Single Responsibility**: Each class/module does ONE thing
2. **Immutable Data**: Domain models are frozen dataclasses
3. **Dependency Injection**: No hidden dependencies
4. **Testable**: Every component tested in isolation
5. **Observable**: Clear state transitions and logging

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Current test coverage: 100% for completed components

## Status

Phase 1: Foundation - In Progress
- ✅ Domain models (45 tests)
- ✅ FileParser service
- ✅ EventAccumulator service
- 🔄 Additional services in progress

