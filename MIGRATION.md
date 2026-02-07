# Migration Guide

Guide for migrating from original `atlas_utilization` to refactored architecture.

## Quick Comparison

### Original vs Refactored

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Entry Point** | `main_pipeline.py` | `main.py` |
| **Config** | Hardcoded in Python | `config.yaml` + dataclasses |
| **Main Class** | `AtlasOpenParser` (1200+ lines) | Multiple small services |
| **Flow Control** | Nested functions | State machine |
| **Testing** | Difficult (tight coupling) | Easy (dependency injection) |
| **Logging** | `utils/logger.py` | Standard logging (configured in main.py) |

## Key Changes

### 1. Configuration

**Original:**
```python
# Hardcoded in main_pipeline.py
config = {
    "output_path": "./output",
    "threads": 8,
    # ... scattered throughout file
}
```

**Refactored:**
```yaml
# config.yaml
parsing_config:
  output_path: "./output"
  threads: 8
  # ... centralized, validated
```

### 2. Parser Class

**Original:**
```python
# AtlasOpenParser - monolithic god class
parser = AtlasOpenParser()
parser.parse_files(...)  # Does everything
```

**Refactored:**
```python
# Separated into focused services
file_parser = FileParser()
accumulator = EventAccumulator(threshold)
processor = ThreadedFileProcessor(file_parser, threads)

# Orchestrated by state machine
executor = PipelineExecutor(config)
executor.run()
```

### 3. Data Flow

**Original:**
```
parse_files()
  ├─ fetch_metadata()
  ├─ for each file:
  │   ├─ _parse_single_file()
  │   ├─ accumulate in memory
  │   └─ yield chunks manually
  └─ save statistics
```

**Refactored:**
```
StateMachine.run()
  ├─ FetchMetadataHandler
  │   └─ MetadataFetcher + MetadataCache
  ├─ ParsingHandler
  │   ├─ ThreadedFileProcessor
  │   │   └─ FileParser (per file)
  │   └─ EventAccumulator
  └─ Statistics collection
```

## Step-by-Step Migration

### Phase 1: Understand New Structure

1. Read `README.md` for architecture overview
2. Explore `domain/` - understand data models
3. Look at `services/` - see how logic is separated
4. Review `orchestration/` - understand state machine

### Phase 2: Update Configuration

1. Create `config.yaml` based on your needs:
```yaml
run_metadata:
  run_name: "my_run_name"

tasks:
  do_parsing: true
  do_mass_calculating: false  # TODO: not implemented yet

parsing_config:
  output_path: "./output/parsed"
  file_urls_path: "./cache/metadata.json"
  jobs_logs_path: "./logs"
  release_years:
    - "2024r-pp"
  threads: 8
  chunk_yield_threshold_bytes: 2147483648
```

2. Validate config:
```bash
python main.py --config config.yaml --dry-run
```

### Phase 3: Run Parsing

```bash
# Run parsing with your config
python main.py --config config.yaml --log-level INFO

# Results will be in:
# - output_path: parsed data
# - jobs_logs_path: execution logs
```

### Phase 4: Extend for Your Use Case

If you need custom behavior:

**Add a new state handler:**
```python
# orchestration/handlers/my_custom_handler.py
from .base import StateHandler

class MyCustomHandler(StateHandler):
    def handle(self, context):
        # Your logic here
        updated_context = context.with_custom_data("key", "value")
        next_state = self._determine_next_state(context)
        return updated_context, next_state
```

**Register in executor:**
```python
# pipeline/executor.py
def _create_handlers(self, services):
    handlers = {
        # ... existing handlers
        PipelineState.MY_CUSTOM: MyCustomHandler()
    }
    return handlers
```

## Code Mapping

### Metadata Fetching

**Original:**
```python
# In AtlasOpenParser.__init__
self._fetch_file_urls()
```

**Refactored:**
```python
# services/metadata/fetcher.py
fetcher = MetadataFetcher(timeout=60)
metadata = fetcher.fetch(release_years=["2024r-pp"])

# With caching
cache = MetadataCache("./cache.json")
if cached := cache.load():
    metadata = cached
else:
    metadata = fetcher.fetch(...)
    cache.save(metadata)
```

### File Parsing

**Original:**
```python
# In AtlasOpenParser._parse_single_file
def _parse_single_file(self, file_path):
    # 200+ lines of logic
    # Reads ROOT, extracts branches, builds awkward arrays
    return events
```

**Refactored:**
```python
# services/parsing/file_parser.py
file_parser = FileParser()
batch = file_parser.parse_file(
    file_path=path,
    file_id=file_id,
    release_year=year,
    tree_names=["CollectionTree"]
)
# Returns EventBatch with metadata
```

### Chunking

**Original:**
```python
# In AtlasOpenParser.parse_files
accumulated_events = []
current_size = 0
for file_events in parsed_files:
    accumulated_events.append(file_events)
    current_size += sys.getsizeof(file_events)
    if current_size > threshold:
        yield accumulated_events  # Manual chunking
        accumulated_events = []
        current_size = 0
```

**Refactored:**
```python
# services/parsing/event_accumulator.py
accumulator = EventAccumulator(threshold_bytes=2_000_000_000)

for batch in parsed_batches:
    chunk = accumulator.add_batch(batch)
    if chunk:  # Automatic chunking
        # Process chunk
        save_chunk(chunk)

# Don't forget final flush
if final_chunk := accumulator.flush():
    save_chunk(final_chunk)
```

### Concurrent Processing

**Original:**
```python
# In multiprocessing_pipeline.py
with multiprocessing.Pool(processes=N) as pool:
    results = pool.map(parse_file, file_paths)
```

**Refactored:**
```python
# services/parsing/threaded_processor.py
processor = ThreadedFileProcessor(
    file_parser=FileParser(),
    max_threads=8,
    show_progress=True
)

for batch in processor.process_files(
    file_urls=urls,
    tree_names=["CollectionTree"],
    release_year="2024r-pp"
):
    # Process batch
    handle_batch(batch)
```

## Testing

### Unit Tests

**Original:** Difficult due to tight coupling

**Refactored:** Easy with dependency injection

```python
# tests/test_file_parser.py
def test_file_parser():
    parser = FileParser()
    
    # Mock just what you need
    with patch('uproot.open') as mock_open:
        mock_open.return_value = mock_root_file
        batch = parser.parse_file(...)
        
    assert batch.event_count == expected
```

### Integration Tests

```python
# tests/test_integration.py
def test_full_pipeline(mock_api, tmp_path):
    config = PipelineConfig(...)
    executor = PipelineExecutor(config)
    
    final_context = executor.run()
    
    assert final_context.is_successful
    assert len(final_context.parsed_files) > 0
```

## Common Pitfalls

### 1. Config Validation Errors

**Error:** `ValueError: parsing_config required when do_parsing=True`

**Fix:** Ensure `parsing_config` is provided when enabling parsing:
```yaml
tasks:
  do_parsing: true

parsing_config:  # Must be present!
  output_path: "./output"
  # ... other required fields
```

### 2. Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Fix:** Use direct imports (no `src.` prefix):
```python
# Wrong
from src.domain.events import EventBatch

# Correct
from domain.events import EventBatch
```

### 3. Immutability

**Error:** `dataclasses.FrozenInstanceError: cannot assign to field`

**Fix:** Use context helper methods:
```python
# Wrong
context.metadata = new_metadata  # Error!

# Correct
context = context.with_metadata(new_metadata)
```

## Performance Comparison

| Metric | Original | Refactored | Notes |
|--------|----------|------------|-------|
| **Code Lines** | ~1500 in parser.py | ~200 per service | More modular |
| **Test Coverage** | ~20% | Target: 80% | Easier to test |
| **Parse Time** | Baseline | Similar | Same core logic |
| **Memory Usage** | High (monolithic) | Lower (streaming) | Better chunking |
| **Extensibility** | Difficult | Easy | State machine |

## FAQ

**Q: Can I still use the old code?**
A: Yes, original code is in `/srv01/agrp/netalev/atlas_utilization`

**Q: Is feature parity complete?**
A: Parsing is complete. Mass calculation, post-processing, and histograms are TODO.

**Q: How do I debug state transitions?**
A: Use `--log-level DEBUG` to see all state transitions and handler executions.

**Q: Can I add custom states?**
A: Yes! Add to `PipelineState` enum, create handler, register in executor.

**Q: Where are the schemas?**
A: Copied from original: `services/parsing/schemas.py`

## Support

For issues or questions:
1. Check logs in `jobs_logs_path`
2. Run with `--log-level DEBUG`
3. Review `test_quick.py` for examples
4. Compare with original implementation

## Next Steps

1. ✅ Phase 1-3: Foundation, services, state machine - **Complete**
2. ✅ Phase 4: Documentation and migration guide - **In Progress**
3. ⏳ Phase 5: Mass calculation, post-processing, histogram handlers
4. ⏳ Phase 6: Full integration tests
5. ⏳ Phase 7: Performance optimization
