"""
Batch splitting utilities for distributed PBS processing.

Splits work items (files, URLs) across multiple batch jobs.
Uses 1-indexed batch jobs to match PBS $PBS_ARRAY_INDEX convention.
"""

import logging

logger = logging.getLogger(__name__)


def get_batch_slice(items: list, batch_index: int, total_batches: int) -> list:
    """
    Extract the slice of items for a specific batch job.

    Uses even distribution with the last batch absorbing remainder.
    Batch indices are 1-based (matching PBS $PBS_ARRAY_INDEX).

    Args:
        items: Full list of items to split
        batch_index: This job's index (1-based)
        total_batches: Total number of batch jobs

    Returns:
        Slice of items for this batch job
    """
    if not items:
        return []

    batch_index = int(batch_index)
    total_batches = int(total_batches)

    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be 1..{total_batches}, got {batch_index}")

    total_items = len(items)
    items_per_batch = total_items // total_batches
    start_idx = (batch_index - 1) * items_per_batch

    if batch_index == total_batches:
        end_idx = total_items  # Last batch gets remainder
    else:
        end_idx = start_idx + items_per_batch

    logger.debug(
        f"Batch {batch_index}/{total_batches}: items[{start_idx}:{end_idx}] "
        f"({end_idx - start_idx} of {total_items})"
    )
    return items[start_idx:end_idx]


def get_batch_slice_by_year(
    file_ids_by_year: dict,
    batch_index: int,
    total_batches: int,
) -> dict:
    """
    Split file URLs across batch jobs, preserving year grouping.

    Flattens all files across years, splits evenly, then reconstructs
    the {year: [urls]} dict for this batch's slice.

    Args:
        file_ids_by_year: Dict of {year: [file_urls]}
        batch_index: This job's index (1-based)
        total_batches: Total number of batch jobs

    Returns:
        Dict of {year: [file_urls]} for this batch
    """
    # Flatten with year tracking
    all_files = []
    for year, file_ids in file_ids_by_year.items():
        for file_id in file_ids:
            all_files.append((year, file_id))

    # Get batch slice
    batch_slice = get_batch_slice(all_files, batch_index, total_batches)

    # Reconstruct year dict
    result = {}
    for year, file_id in batch_slice:
        if year not in result:
            result[year] = []
        result[year].append(file_id)

    total_all = sum(len(v) for v in file_ids_by_year.values())
    total_batch = sum(len(v) for v in result.values())
    logger.info(
        f"Batch {batch_index}/{total_batches}: "
        f"{total_batch} files (of {total_all} total) across {len(result)} year(s)"
    )
    return result
