import sys
import gc
import psutil
import os
from collections import defaultdict
import tracemalloc

def _get_size(obj, seen=None):
    """Recursively calculate size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([_get_size(v, seen) for v in obj.values()])
        size += sum([_get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += _get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum([_get_size(i, seen) for i in obj])
        except:
            pass
    
    return size

def _format_bytes(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def print_top_memory_variables(n=10, frame=None):
    """
    Print top N variables consuming the most memory
    
    Args:
        n: Number of top variables to show
        frame: Stack frame to inspect (None = caller's frame)
    """
    if frame is None:
        frame = sys._getframe(1)
    
    print(f"\n{'='*70}")
    print(f"🔍 TOP {n} MEMORY-CONSUMING VARIABLES")
    print(f"{'='*70}")
    
    # Get all variables from local and global scope
    all_vars = {}
    all_vars.update(frame.f_locals)
    all_vars.update(frame.f_globals)
    
    # Calculate sizes
    var_sizes = []
    for name, obj in all_vars.items():
        if not name.startswith('_'):  # Skip private variables
            try:
                size = _get_size(obj)
                var_sizes.append((name, obj, size))
            except:
                pass
    
    # Sort by size
    var_sizes.sort(key=lambda x: x[2], reverse=True)
    
    # Print top N
    print(f"{'Rank':<6} {'Variable':<30} {'Type':<25} {'Size':<15}")
    print(f"{'-'*70}")
    
    for i, (name, obj, size) in enumerate(var_sizes[:n], 1):
        obj_type = type(obj).__name__
        if hasattr(obj, '__len__'):
            try:
                obj_type += f"[{len(obj)}]"
            except:
                pass
        
        print(f"{i:<6} {name[:29]:<30} {obj_type[:24]:<25} {_format_bytes(size):<15}")
    
    total_size = sum(size for _, _, size in var_sizes[:n])
    print(f"{'-'*70}")
    print(f"Total for top {n}: {_format_bytes(total_size)}")
    print(f"{'='*70}\n")
    
    return var_sizes[:n]

def print_gc_stats():
    """Print garbage collection statistics"""
    print(f"\n{'='*70}")
    print(f"🗑️  GARBAGE COLLECTION STATISTICS")
    print(f"{'='*70}")
    
    # Get GC stats
    gc_stats = gc.get_stats()
    gc_count = gc.get_count()
    
    print(f"GC Enabled: {gc.isenabled()}")
    print(f"GC Thresholds: {gc.get_threshold()}")
    print(f"GC Counts (gen0, gen1, gen2): {gc_count}")
    print(f"GC Objects tracked: {len(gc.get_objects()):,}")
    
    # Count objects by type
    type_counts = defaultdict(int)
    type_sizes = defaultdict(int)
    
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        type_counts[obj_type] += 1
        try:
            type_sizes[obj_type] += sys.getsizeof(obj)
        except:
            pass
    
    # Sort by count
    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop 10 Object Types by Count:")
    print(f"{'Type':<30} {'Count':<15} {'Est. Size':<15}")
    print(f"{'-'*70}")
    for obj_type, count in top_types:
        size = type_sizes.get(obj_type, 0)
        print(f"{obj_type[:29]:<30} {count:<15,} {_format_bytes(size):<15}")
    
    print(f"{'='*70}\n")

def force_garbage_collection():
    """Force garbage collection and return stats"""
    print(f"\n{'='*70}")
    print(f"🧹 FORCING GARBAGE COLLECTION")
    print(f"{'='*70}")
    
    before_count = len(gc.get_objects())
    
    # Collect all generations
    collected = []
    for generation in range(3):
        n = gc.collect(generation)
        collected.append(n)
        print(f"Generation {generation}: Collected {n} objects")
    
    after_count = len(gc.get_objects())
    
    print(f"\nObjects before: {before_count:,}")
    print(f"Objects after:  {after_count:,}")
    print(f"Net reduction:  {before_count - after_count:,}")
    print(f"{'='*70}\n")
    
    return collected

def get_process_mem_usage_mb():
    """Get actual process memory usage"""
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    return process_rss_bytes / (1024**2)
