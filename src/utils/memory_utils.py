import sys
import gc
import psutil
import os
from collections import defaultdict
import tracemalloc

def get_size(obj, seen=None):
    """Recursively calculate size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum([get_size(i, seen) for i in obj])
        except:
            pass
    
    return size

def format_bytes(bytes_size):
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
                size = get_size(obj)
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
        
        print(f"{i:<6} {name[:29]:<30} {obj_type[:24]:<25} {format_bytes(size):<15}")
    
    total_size = sum(size for _, _, size in var_sizes[:n])
    print(f"{'-'*70}")
    print(f"Total for top {n}: {format_bytes(total_size)}")
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
        print(f"{obj_type[:29]:<30} {count:<15,} {format_bytes(size):<15}")
    
    print(f"{'='*70}\n")

def print_process_memory():
    """Print current process memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()
    
    print(f"\n{'='*70}")
    print(f"💾 PROCESS MEMORY USAGE")
    print(f"{'='*70}")
    print(f"RSS (Resident Set Size):  {format_bytes(mem_info.rss)}")
    print(f"VMS (Virtual Memory):     {format_bytes(mem_info.vms)}")
    print(f"Memory Percent:           {mem_percent:.2f}%")
    
    # System memory
    sys_mem = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total:      {format_bytes(sys_mem.total)}")
    print(f"  Available:  {format_bytes(sys_mem.available)}")
    print(f"  Used:       {format_bytes(sys_mem.used)} ({sys_mem.percent}%)")
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

def track_memory_growth(func):
    """Decorator to track memory growth during function execution"""
    def wrapper(*args, **kwargs):
        gc.collect()
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        print(f"\n🏁 Starting {func.__name__}")
        print(f"Memory before: {format_bytes(mem_before)}")
        
        result = func(*args, **kwargs)
        
        gc.collect()
        mem_after = process.memory_info().rss
        mem_diff = mem_after - mem_before
        
        print(f"Memory after:  {format_bytes(mem_after)}")
        print(f"Memory delta:  {format_bytes(mem_diff)}")
        
        if mem_diff > 0:
            print(f"⚠️  Memory increased by {format_bytes(mem_diff)}")
        else:
            print(f"✅ Memory decreased by {format_bytes(-mem_diff)}")
        
        return result
    
    return wrapper

def comprehensive_memory_report(top_n=10):
    """Print a comprehensive memory report"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE MEMORY REPORT")
    print("="*70 + "\n")
    
    print_process_memory()
    print_gc_stats()
    print_top_memory_variables(n=top_n)
    force_garbage_collection()
    print_process_memory()

def get_process_mem_usage_mb():
    """Get actual process memory usage"""
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    return process_rss_bytes / (1024**2)
