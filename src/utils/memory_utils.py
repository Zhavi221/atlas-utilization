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
    Get top N variables consuming the most memory (silent).
    
    Args:
        n: Number of top variables to show
        frame: Stack frame to inspect (None = caller's frame)
    
    Returns:
        List of tuples: (variable_name, object, size_in_bytes)
    """
    if frame is None:
        frame = sys._getframe(1)
    
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
    
    return var_sizes[:n]

def print_gc_stats():
    """Get garbage collection statistics (no longer prints)."""
    pass

def force_garbage_collection():
    """Force garbage collection and return stats (silent)."""
    # Collect all generations
    collected = []
    for generation in range(3):
        n = gc.collect(generation)
        collected.append(n)
    
    return collected

def get_process_mem_usage_mb():
    """Get actual process memory usage"""
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    return process_rss_bytes / (1024**2)
