"""
Binary Search implementation for the Learned Index Visualizer.

Supports:
- Traditional whole-array binary search O(log N).
- Range-bounded local binary search O(log W).
- Comparison count tracking and step-by-step state recording.
- Physical record dereferencing via row_id pointers.
"""

import numpy as np


def binary_search(arr: np.ndarray, target: int) -> int:
    """
    Standard binary search over a sorted 1D array.

    Returns array index if found, -1 otherwise.
    """
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        val = arr[mid]
        if val == target:
            return mid
        elif val < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def binary_search_with_stats(keys: np.ndarray, target: int,
                             row_ids: np.ndarray = None,
                             bundle = None) -> dict:
    """
    Standard binary search that records comparison count, step history,
    and dereferences the original record via row_id pointer.
    """
    low, high = 0, len(keys) - 1
    comparisons = 0
    steps = []

    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        val = int(keys[mid])

        steps.append({
            "step": comparisons,
            "low": low,
            "high": high,
            "mid": mid,
            "mid_key": val,
            "action": ""
        })

        if val == target:
            steps[-1]["action"] = "FOUND"
            row_id = int(row_ids[mid]) if row_ids is not None else mid
            record = bundle.get_record(row_id) if bundle else None
            return {
                "index": mid,
                "found": True,
                "comparisons": comparisons,
                "steps": steps,
                "row_id": row_id,
                "record": record
            }
        elif val < target:
            steps[-1]["action"] = "GO RIGHT"
            low = mid + 1
        else:
            steps[-1]["action"] = "GO LEFT"
            high = mid - 1

    return {
        "index": -1,
        "found": False,
        "comparisons": comparisons,
        "steps": steps,
        "row_id": -1,
        "record": None
    }


def binary_search_in_range(arr: np.ndarray, target: int, lo: int, hi: int) -> tuple:
    """
    Local binary search restricted to index window [lo, hi].

    Returns (index, comparisons) where index is -1 if not found.
    """
    low, high = lo, hi
    comparisons = 0

    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        val = arr[mid]
        if val == target:
            return mid, comparisons
        elif val < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1, comparisons


def binary_search_in_range_with_stats(keys: np.ndarray, target: int,
                                       lo: int, hi: int,
                                       row_ids: np.ndarray = None,
                                       bundle = None) -> dict:
    """
    Local binary search within [lo, hi], recording step-by-step history
    and resolving physical record dereference.
    """
    low, high = lo, hi
    comparisons = 0
    steps = []

    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        val = int(keys[mid])

        steps.append({
            "step": comparisons,
            "low": low,
            "high": high,
            "mid": mid,
            "mid_key": val,
            "action": ""
        })

        if val == target:
            steps[-1]["action"] = "FOUND"
            row_id = int(row_ids[mid]) if row_ids is not None else mid
            record = bundle.get_record(row_id) if bundle else None
            return {
                "index": mid,
                "found": True,
                "comparisons": comparisons,
                "steps": steps,
                "search_lo": lo,
                "search_hi": hi,
                "row_id": row_id,
                "record": record
            }
        elif val < target:
            steps[-1]["action"] = "GO RIGHT"
            low = mid + 1
        else:
            steps[-1]["action"] = "GO LEFT"
            high = mid - 1

    return {
        "index": -1,
        "found": False,
        "comparisons": comparisons,
        "steps": steps,
        "search_lo": lo,
        "search_hi": hi,
        "row_id": -1,
        "record": None
    }
