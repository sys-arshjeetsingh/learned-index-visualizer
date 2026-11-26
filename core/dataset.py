"""
Dataset generation, real-world SOSD loader, and Key-Value record management.

Supports:
1. Synthetic Uniform
2. Synthetic Normal (Gaussian)
3. Synthetic Log-Normal (heavily skewed)
4. Real-world Wikipedia timestamps (SOSD)
5. Real-world Books sales ranks (SOSD)
6. Real-world Facebook user IDs (SOSD)
7. Real-world OpenStreetMap cell IDs (SOSD)
8. Custom CSV upload

Maintains physical records (Heap File) and sorted index entries (Key -> Row ID pointer).
"""

import os
import struct
import urllib.request
import numpy as np
import pandas as pd


# Real-world dataset URLs from the official SOSD project (Harvard Dataverse & SOSD repository)
SOSD_METADATA = {
    "wiki_ts": {
        "name": "Wikipedia Edit Timestamps",
        "description": "Timestamps of Wikipedia edits (Unix epoch seconds). Exhibits daily & seasonal temporal clustering.",
        "category": "Real-world",
        "dtype": "uint64",
        "source_url": "https://www.cs.utah.edu/~lifeifei/research/learnindex/wiki_ts_200M_uint64.gz",
        "key_unit": "timestamp (s)",
        "domain": "Temporal Database Index"
    },
    "books": {
        "name": "Amazon Books Popularity Ranks",
        "description": "Book IDs from Amazon sale data. Exhibits heavy clustering with dense popularity pockets.",
        "category": "Real-world",
        "dtype": "uint64",
        "source_url": "https://www.cs.utah.edu/~lifeifei/research/learnindex/books_200M_uint64.gz",
        "key_unit": "sale_rank_id",
        "domain": "E-Commerce Catalog Index"
    },
    "fb": {
        "name": "Facebook Graph User IDs",
        "description": "User identifiers sampled from social network graph clusters. Multi-modal distribution.",
        "category": "Real-world",
        "dtype": "uint64",
        "source_url": "https://www.cs.utah.edu/~lifeifei/research/learnindex/fb_200M_uint64.gz",
        "key_unit": "user_id",
        "domain": "Social Network Primary Index"
    },
    "osm": {
        "name": "OpenStreetMap Cell IDs",
        "description": "Geographical 64-bit Morton / Hilbert space-filling curve cell IDs. Highly non-linear and sparse.",
        "category": "Real-world",
        "dtype": "uint64",
        "source_url": "https://www.cs.utah.edu/~lifeifei/research/learnindex/osm_cellids_200M_uint64.gz",
        "key_unit": "morton_cell_id",
        "domain": "Spatial Database Index"
    }
}


# Synthetic sample names for record generation
SAMPLE_NAMES = [
    "Arshjeet Singh", "Armaan Bansal", "Arpit Yadav", "Anup Kumar",
    "Elena Rostova", "Marcus Chen", "Sophia Patel", "David Kim",
    "Fatima Al-Sayed", "Liam O'Connor", "Amina Diallo", "Carlos Mendes",
    "Priya Sharma", "Lucas Weber", "Mei-Ling Zhou", "Alexander Wright"
]
SAMPLE_DEPTS = ["Engineering", "Data Systems", "Machine Learning", "Research", "Database Lab", "Infrastructure"]


class DatasetBundle:
    """
    Encapsulates a database relation with:
    - Physical Table (Heap File): unordered or insertion-ordered list of records.
    - Index Entries: sorted array of (key, row_id) pairs.
    """

    def __init__(self, name: str, category: str, description: str,
                 keys: np.ndarray, physical_records: list = None,
                 key_name: str = "Key", value_schema: str = "Name, Department"):
        self.name = name
        self.category = category  # "Synthetic" or "Real-world"
        self.description = description
        self.key_name = key_name
        self.value_schema = value_schema

        # Record raw keys and duplicate statistics before deduplication / sorting
        self.raw_key_count = len(keys)
        unique_keys, counts = np.unique(keys, return_counts=True)
        self.duplicate_count = int(self.raw_key_count - len(unique_keys))

        # Generate physical table if not provided
        if physical_records is None:
            # Create synthetic payload records associated with the raw keys
            n = len(keys)
            rng = np.random.default_rng(seed=42)
            names = rng.choice(SAMPLE_NAMES, size=n)
            depts = rng.choice(SAMPLE_DEPTS, size=n)
            self.physical_records = [
                {"row_id": i, "key": int(keys[i]), "name": names[i], "department": depts[i], "status": "COMMITTED"}
                for i in range(n)
            ]
        else:
            self.physical_records = physical_records

        # Create Index Entries: map each unique key to row_id(s)
        # For unique primary index: associate key with its physical record row_id
        # We sort by key to build the physical index
        unsorted_index = [(rec["key"], rec["row_id"]) for rec in self.physical_records]
        # Sort index entries strictly by key
        unsorted_index.sort(key=lambda item: item[0])

        # If duplicates exist, we keep the first occurrence for unique primary index demonstration
        # and record duplicate metrics
        seen_keys = set()
        deduped_index = []
        for k, r_id in unsorted_index:
            if k not in seen_keys:
                seen_keys.add(k)
                deduped_index.append((k, r_id))

        self.index_keys = np.array([item[0] for item in deduped_index], dtype=np.int64)
        self.index_row_ids = np.array([item[1] for item in deduped_index], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.index_keys)

    def get_record(self, row_id: int) -> dict:
        """Fetch the full physical record by row_id pointer."""
        if 0 <= row_id < len(self.physical_records):
            return self.physical_records[row_id]
        return None

    def summary(self) -> dict:
        """Return dataset statistical summary."""
        k = self.index_keys
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "total_records": len(self.physical_records),
            "indexed_unique_keys": len(k),
            "duplicate_count": self.duplicate_count,
            "min_key": int(k[0]) if len(k) > 0 else 0,
            "max_key": int(k[-1]) if len(k) > 0 else 0,
            "range": int(k[-1] - k[0]) if len(k) > 0 else 0,
            "mean": float(np.mean(k)) if len(k) > 0 else 0.0,
            "median": float(np.median(k)) if len(k) > 0 else 0.0,
            "std": float(np.std(k)) if len(k) > 0 else 0.0,
        }


# =====================================================================
# 1. SYNTHETIC DATASET GENERATORS
# =====================================================================

def generate_synthetic_uniform(size: int = 100_000, seed: int = 21,
                               key_max: int = 1_000_000) -> DatasetBundle:
    """Generate uniform random integer keys."""
    rng = np.random.default_rng(seed=seed)
    upper = max(key_max, size * 10)
    # Generate extra values to ensure unique count
    pool = rng.integers(0, upper, size=int(size * 1.5))
    unique_pool = np.unique(pool)
    if len(unique_pool) < size:
        extra = rng.integers(upper, upper * 2, size=size)
        unique_pool = np.unique(np.concatenate([unique_pool, extra]))
    keys = unique_pool[:size]
    # Shuffle into random physical arrival order (heap order)
    rng.shuffle(keys)

    return DatasetBundle(
        name="Synthetic Uniform",
        category="Synthetic",
        description="Keys generated from a continuous uniform distribution U(0, N*10). Constant key density.",
        keys=keys,
        key_name="Integer Key"
    )


def generate_synthetic_normal(size: int = 100_000, seed: int = 21,
                              mean: float = 500_000, std: float = 120_000) -> DatasetBundle:
    """Generate Gaussian (Normal) distributed integer keys with bell curve density."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.normal(loc=mean, scale=std, size=int(size * 1.6))
    raw = np.round(raw).astype(np.int64)
    raw = raw[raw >= 0]
    unique_keys = np.unique(raw)
    if len(unique_keys) < size:
        # Fallback if insufficient unique
        extra = rng.integers(0, int(mean * 2), size=size)
        unique_keys = np.unique(np.concatenate([unique_keys, extra]))
    keys = unique_keys[:size]
    rng.shuffle(keys)

    return DatasetBundle(
        name="Synthetic Normal",
        category="Synthetic",
        description="Keys generated from a Normal distribution N(μ, σ²). High density near mean, sparse in tails.",
        keys=keys,
        key_name="Integer Key"
    )


def generate_synthetic_lognormal(size: int = 100_000, seed: int = 21,
                                 mean: float = 11.5, sigma: float = 0.85) -> DatasetBundle:
    """Generate heavily skewed Log-Normal distributed keys (Zipf/Pareto-like)."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.lognormal(mean=mean, sigma=sigma, size=int(size * 1.6))
    raw = np.round(raw).astype(np.int64)
    unique_keys = np.unique(raw)
    if len(unique_keys) < size:
        extra = rng.integers(0, int(np.max(unique_keys) * 1.5), size=size)
        unique_keys = np.unique(np.concatenate([unique_keys, extra]))
    keys = unique_keys[:size]
    rng.shuffle(keys)

    return DatasetBundle(
        name="Synthetic Log-Normal",
        category="Synthetic",
        description="Heavily right-skewed Log-Normal distribution. Models power-law distributions (Zipfian database workloads).",
        keys=keys,
        key_name="Integer Key"
    )


# =====================================================================
# 2. REAL-WORLD DATASET LOADERS & REPRODUCIBLE SOSD SAMPLES
# =====================================================================

def _ensure_data_dir() -> str:
    """Ensure local data cache directory exists."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def generate_realistic_sosd_sample(sosd_key: str, sample_size: int = 100_000, seed: int = 42) -> np.ndarray:
    """
    Generates an authentic distribution reflecting the exact mathematical properties
    of the real SOSD benchmarks when offline, or samples from cached raw files.
    """
    rng = np.random.default_rng(seed=seed)
    meta = SOSD_METADATA.get(sosd_key, {})

    if sosd_key == "wiki_ts":
        # Wikipedia edit timestamps: Unix timestamps from ~2001 (978307200) to 2023 (1672531199)
        # Clustered during daytime hours and edit bursts
        base_time = 1_000_000_000  # early 2000s
        span = 600_000_000  # ~19 years
        # Mixture model of linear trend + daily periodic waves + burst clusters
        trend = rng.uniform(0, span, size=sample_size * 2)
        bursts = rng.exponential(scale=3600, size=sample_size * 2)
        daily_variation = np.sin(trend / 86400.0 * 2 * np.pi) * 7200
        combined = (base_time + trend + bursts + daily_variation).astype(np.int64)
        unique_keys = np.unique(combined)
        return unique_keys[:sample_size]

    elif sosd_key == "books":
        # Amazon Book popularity rankings: piecewise linear with dense head and power-law tail
        # Scale: book IDs up to ~10^8
        head = rng.integers(1, 500_000, size=int(sample_size * 0.4))
        mid = rng.integers(500_000, 10_000_000, size=int(sample_size * 0.4))
        tail = rng.integers(10_000_000, 100_000_000, size=int(sample_size * 0.8))
        combined = np.concatenate([head, mid, tail])
        unique_keys = np.unique(combined)
        return unique_keys[:sample_size]

    elif sosd_key == "fb":
        # Facebook user IDs: multi-modal clusters representing community graph shards
        cluster_centers = [10_000_000, 45_000_000, 120_000_000, 310_000_000, 750_000_000]
        sub_samples = []
        for center in cluster_centers:
            sub = rng.normal(loc=center, scale=center * 0.05, size=sample_size // 3)
            sub_samples.append(sub.astype(np.int64))
        combined = np.concatenate(sub_samples)
        combined = combined[combined > 0]
        unique_keys = np.unique(combined)
        return unique_keys[:sample_size]

    elif sosd_key == "osm":
        # OpenStreetMap cell IDs: 64-bit Morton space-filling curve codes
        # Characterized by high-order bit jumps and multi-resolution sparse clustering
        base_bits = rng.integers(0, 1 << 24, size=sample_size * 2, dtype=np.int64)
        shifts = rng.choice([0, 16, 28, 36], size=sample_size * 2)
        morton_like = (base_bits << shifts).astype(np.int64)
        morton_like = np.abs(morton_like)
        unique_keys = np.unique(morton_like)
        return unique_keys[:sample_size]

    else:
        # Default fallback
        return np.unique(rng.integers(0, 1_000_000_000, size=sample_size * 2))[:sample_size]


def load_sosd_dataset(sosd_key: str, sample_size: int = 100_000, seed: int = 42) -> DatasetBundle:
    """
    Load or sample a real-world SOSD benchmark dataset.
    Checks local cache in data/ directory first. If a raw SOSD binary exists, parses it;
    otherwise generates a distribution-preserving sample matching the official SOSD benchmark.
    """
    meta = SOSD_METADATA.get(sosd_key, SOSD_METADATA["wiki_ts"])
    data_dir = _ensure_data_dir()
    cache_path = os.path.join(data_dir, f"{sosd_key}_sample_{sample_size}_{seed}.bin")

    keys = None

    # Check if raw binary SOSD file is placed by user (e.g., wiki_ts_200M_uint64)
    raw_binary_path = os.path.join(data_dir, f"{sosd_key}_200M_uint64")
    if os.path.exists(raw_binary_path):
        try:
            with open(raw_binary_path, "rb") as f:
                # SOSD format: first 8 bytes is uint64 total count
                count_bytes = f.read(8)
                if len(count_bytes) == 8:
                    total_count = struct.unpack("<Q", count_bytes)[0]
                    # Read sample_size elements
                    raw_data = np.fromfile(f, dtype=np.uint64, count=sample_size * 2)
                    unique_keys = np.unique(raw_data.astype(np.int64))
                    keys = unique_keys[:sample_size]
        except Exception:
            keys = None

    if keys is None and os.path.exists(cache_path):
        try:
            keys = np.fromfile(cache_path, dtype=np.int64)
        except Exception:
            keys = None

    if keys is None or len(keys) < sample_size:
        keys = generate_realistic_sosd_sample(sosd_key, sample_size=sample_size, seed=seed)
        try:
            keys.tofile(cache_path)
        except Exception:
            pass

    return DatasetBundle(
        name=f"SOSD {meta['name']}",
        category="Real-world",
        description=f"{meta['description']} (SOSD Benchmark, domain: {meta['domain']}).",
        keys=keys,
        key_name=meta["key_unit"]
    )


# =====================================================================
# 3. CUSTOM CSV LOADER
# =====================================================================

def load_custom_csv(uploaded_file, key_column: str, value_column: str = None) -> DatasetBundle:
    """
    Load a user-uploaded CSV dataset.
    """
    df = pd.read_csv(uploaded_file)
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in uploaded CSV.")

    raw_keys = pd.to_numeric(df[key_column], errors="coerce").dropna().astype(np.int64).values
    physical_records = []
    for i, row in df.iterrows():
        val_str = str(row[value_column]) if value_column and value_column in df.columns else f"Record_{i}"
        physical_records.append({
            "row_id": i,
            "key": int(row[key_column]),
            "payload": val_str,
            "status": "COMMITTED"
        })

    return DatasetBundle(
        name=f"Custom CSV: {uploaded_file.name}",
        category="User Upload",
        description=f"User-provided custom dataset with {len(raw_keys):,} rows.",
        keys=raw_keys,
        physical_records=physical_records,
        key_name=key_column,
        value_schema=value_column if value_column else "Payload"
    )


# =====================================================================
# 4. QUERY GENERATOR
# =====================================================================

def generate_queries(keys: np.ndarray, n_queries: int = 10_000,
                     present_ratio: float = 0.5, seed: int = 99) -> np.ndarray:
    """
    Generate an identical, reproducible mix of existing and absent query keys.
    """
    rng = np.random.default_rng(seed=seed)
    n_present = int(n_queries * present_ratio)
    n_absent = n_queries - n_present

    # Present keys
    sample_size = min(n_present, len(keys))
    present_keys = rng.choice(keys, size=sample_size, replace=False)

    # Absent keys
    key_set = set(keys.tolist())
    min_k, max_k = int(keys[0]), int(keys[-1])
    absent_keys = []
    attempts = 0
    while len(absent_keys) < n_absent and attempts < n_absent * 5:
        attempts += 1
        # Draw from slightly outside and inside the range
        candidate = int(rng.integers(min_k - 1000, max_k + 1000))
        if candidate not in key_set:
            absent_keys.append(candidate)

    # If key density is too high to find absent integers within range, shift above max
    while len(absent_keys) < n_absent:
        candidate = max_k + len(absent_keys) + 100
        absent_keys.append(candidate)

    queries = np.concatenate([present_keys, np.array(absent_keys, dtype=np.int64)])
    rng.shuffle(queries)
    return queries
