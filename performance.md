# Performance Proposal: Prefetched Data Loading for Per-Pair Evaluation

## Problem

The per-pair evaluation loops (`Processing depth pairs`, `Processing RGB pairs`,
`Processing rays pairs`) run at ~3 s/iteration on cluster workloads. Profiling
the loop body shows that CPU-side metric computation (numpy) and GPU metrics
(LPIPS) together take well under 1 s, so **the dominant cost is synchronous I/O
inside `dataset[i]`** — reading samples from zip archives on a network
filesystem (`/cluster/work/...`).

The current loop is fully synchronous:

```
for i in range(N):
    sample = dataset[i]        # ~2-2.5 s  I/O (zip read on NFS)
    metrics = compute(sample)  #  ~0.5-1 s  CPU + GPU
```

Nothing overlaps: while metrics are computed, no I/O is in progress.

## Root Cause

`evaluate_depth_samples`, `evaluate_rgb_samples`, and `evaluate_rays_samples`
all iterate with a plain `for i in range(num_samples): sample = dataset[i]`
loop. The `batch_size` / `num_workers` CLI arguments are only forwarded to the
FID/KID `DataLoader`, not to the per-pair loop.

`MultiModalDataset` is a `torch.utils.data.Dataset` subclass and already
handles per-process zip file handles (`_get_zip_handle` keys on PID), so it is
safe to use with a multi-worker `DataLoader` out of the box.

## Proposed Solution

Wrap the per-pair iteration in a `torch.utils.data.DataLoader` with
`num_workers > 0` so that worker processes read and decompress the next N
samples while the main process computes metrics on the current one.

### Design Constraints

The per-pair loop is not a pure map — several pieces of state are set on the
first sample and carried forward:

| State | Set when | Used for |
|---|---|---|
| `normalized_predictions` | `i == 0`, based on pred value range | Decides whether scale-and-shift alignment is applied for all subsequent samples |
| `gt_native_dims` / `pred_native_dims` | `i == 0` | Recorded in output metadata |
| `spatial_method` | `i == 0` | Recorded in output metadata |
| `resolved_domain` (rays only) | `i == 0`, from intrinsics | Determines ρ_A threshold for all samples |
| `logged_alignment` / `logged_stats` | First occurrence | One-time log messages |

These are all **determined from the first sample only**, then static for the
rest of the loop. This means the loop can be cleanly split into two phases.

### Architecture: Peek + Prefetched Loop

```
Phase 1 — Peek (single sample, no DataLoader)
    sample_0 = dataset[0]
    Determine: normalized_predictions, spatial_method, dims, fov_domain
    Compute metrics for sample_0
    Log one-time messages

Phase 2 — Prefetched loop (DataLoader, samples 1..N-1)
    loader = DataLoader(
        Subset(dataset, range(1, N)),
        batch_size=1,          # logical batch = 1 sample
        num_workers=K,
        prefetch_factor=2,
        collate_fn=identity,   # no batching, return sample dict as-is
        persistent_workers=True,
    )
    for sample in loader:
        Compute metrics (using state from Phase 1)
```

#### Why `batch_size=1` on the DataLoader

The samples are not collated into tensors — each sample is a dict of
variable-resolution arrays, intrinsics matrices, and hierarchical modality
dicts. The point of the DataLoader here is **I/O prefetching**, not
tensor batching. Workers call `dataset.__getitem__` in parallel, reading
from zip and decoding, so that by the time the main process finishes metric
computation, the next sample is already in memory.

A `collate_fn` that passes through the sample dict unchanged avoids any
padding or stacking.

### Worker Count

The CLI already has a `--num-workers` flag (default 4). This proposal reuses it
for the prefetched loader. Recommended values:

| Storage | Suggested `num_workers` |
|---|---|
| NFS / GPFS zip archives | 4-8 (bound by I/O concurrency) |
| Local SSD | 2-4 (diminishing returns) |
| Local uncompressed dirs | 2 (I/O already fast) |

The DataLoader's `prefetch_factor=2` means each worker prepares 2 samples
ahead, so with `num_workers=4` there are up to 8 samples buffered.

### Expected Speedup

With I/O at ~2-2.5 s and compute at ~0.5-1 s per sample:

- **Without prefetch**: `I/O + compute = ~3 s/it`
- **With prefetch (steady state)**: `max(I/O / num_workers, compute) ≈ ~1 s/it`

With 4 workers, I/O is spread across workers, so the main process sees
pre-loaded samples immediately and only pays the compute cost. This gives
an estimated **~3x throughput improvement** in the per-pair loop.

Startup cost (Phase 1 + worker fork) is a fixed ~5-10 s overhead, amortized
across hundreds or thousands of samples.

## Implementation Plan

### Step 1: Extract first-sample setup into a helper

Refactor each `evaluate_*_samples` function so that the first-iteration
logic (dimension capture, alignment mode detection, logging) is a separate
function:

```python
def _depth_first_sample_setup(
    sample: dict,
    alignment_mode: str,
    sky_mask_enabled: bool,
    is_radial: bool,
    verbose: bool,
) -> DepthSetupState:
    """Process the first sample and return all state carried forward."""
    depth_gt = to_numpy_depth(sample["gt"])
    depth_pred = to_numpy_depth(sample["pred"])
    gt_native_dims = depth_gt.shape[:2]
    pred_native_dims = depth_pred.shape[:2]
    spatial_method = classify_spatial_alignment(...)
    normalized_predictions = ...  # range check
    return DepthSetupState(
        gt_native_dims=gt_native_dims,
        pred_native_dims=pred_native_dims,
        spatial_method=spatial_method,
        normalized_predictions=normalized_predictions,
    )
```

Similarly for RGB and Rays.

### Step 2: Extract per-sample metric computation into a standalone function

Currently, the loop body is deeply nested inside `evaluate_depth_samples`
with closures over `lpips_metric`, `stores`, etc. Refactor the per-sample
work into a function that takes explicit arguments:

```python
def _process_depth_sample(
    sample: dict,
    *,
    state: DepthSetupState,
    lpips_metric: LPIPSMetric,
    is_radial: bool,
    alignment_mode: str,
    sky_mask_enabled: bool,
    benchmark_depth_range: Optional[tuple[float, float]],
) -> DepthSampleResult:
    """Compute all per-sample depth metrics. Pure function of inputs."""
    ...
```

This makes the loop body independent of iteration index (except for
writing npy files, which takes `i` for the filename — pass it explicitly).

### Step 3: Build the prefetched iteration

```python
from torch.utils.data import DataLoader, Subset

def _identity_collate(batch):
    """Pass single-element batches through without collation."""
    return batch[0]

def _prefetched_iter(dataset, num_workers, skip=0):
    """Yield samples from dataset with worker-based prefetching."""
    if num_workers <= 0 or len(dataset) <= skip:
        for i in range(skip, len(dataset)):
            yield i, dataset[i]
        return

    indices = list(range(skip, len(dataset)))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2,
        collate_fn=_identity_collate,
        persistent_workers=True,
    )
    for offset, sample in enumerate(loader):
        yield skip + offset, sample
```

### Step 4: Wire into evaluate functions

```python
def evaluate_depth_samples(dataset, ..., num_workers=4, ...):
    # Phase 1: first sample
    sample_0 = dataset[0]
    state = _depth_first_sample_setup(sample_0, ...)
    result_0 = _process_depth_sample(sample_0, state=state, ...)
    _accumulate(stores, result_0, index=0)

    # Phase 2: remaining samples with prefetch
    for i, sample in _prefetched_iter(dataset, num_workers, skip=1):
        result = _process_depth_sample(sample, state=state, ...)
        _accumulate(stores, result, index=i)

    # Phase 3: aggregation (FID/KID, quantiles, etc.)
    ...
```

### Step 5: Deferred LPIPS batching (optional, additive)

As a secondary optimization, defer LPIPS computation to after the per-pair
loop and use the existing `compute_batch` method:

```python
# During per-pair loop: skip LPIPS, save processed arrays
lpips_gt_paths = []
lpips_pred_paths = []
for i, sample in _prefetched_iter(...):
    ...
    # Save processed depth for LPIPS (already writing for FID anyway)
    lpips_gt_paths.append(gt_npy_path)
    lpips_pred_paths.append(pred_npy_path)

# After loop: batch LPIPS
print("Computing LPIPS (batched)...")
all_gt = [np.load(p) for p in lpips_gt_paths]
all_pred = [np.load(p) for p in lpips_pred_paths]
lpips_values = lpips_metric.compute_batch(all_pred, all_gt, batch_size=batch_size)
```

This turns N sequential GPU forward passes into N/batch_size batched passes.
Impact is modest (~50-200 ms saved per sample) compared to the I/O prefetch,
but it composes well and reduces total GPU time.

**Trade-off**: Per-file LPIPS values are computed after the loop, so the
per-file metrics dict must be updated retroactively, or LPIPS must be
excluded from per-file output. The npy files are already written to the temp
directory for FID/KID, so no additional disk cost.

## Scope of Changes

| File | Change |
|---|---|
| `euler_eval/evaluate.py` | Refactor loop structure; add `_prefetched_iter`, `_identity_collate`; split first-sample setup; optionally batch LPIPS |
| `euler_eval/cli.py` | Pass `num_workers` to all three `evaluate_*_samples` calls (depth/RGB already do; rays does not) |

No changes to `euler_loading`, `ds_crawler`, metrics modules, or CLI flags.
The `--num-workers` flag gains broader effect but its interface is unchanged.

## Risks and Mitigations

### Worker process memory

Each worker loads one sample into memory (depth map + RGB + segmentation +
calibration). At 1920x1080 float32, a depth pair is ~16 MB. With
`num_workers=4, prefetch_factor=2`, peak buffer is ~130 MB. Acceptable.

### Zip file handle safety

`MultiModalDataset._get_zip_handle` already keys on `(path, pid)`, opening
a fresh `ZipFile` per worker process. No changes needed.

### Non-determinism

`DataLoader` with `num_workers > 0` does not change sample order (no
shuffle). Results are identical to the sequential loop.

### Graceful fallback

When `num_workers=0` (or on systems where forking is problematic),
`_prefetched_iter` falls back to the current sequential `dataset[i]` loop.
No behavioral change.

### LPIPS per-file metrics (if deferred)

If LPIPS is deferred to a batch pass, per-file metrics are written without
LPIPS initially, then patched. Alternatively, keep per-sample LPIPS in the
loop (it's only ~100 ms) and only batch it as a follow-up optimization.
The prefetch alone delivers the major speedup.

## Phases

**Phase 1 (high impact, low risk):** DataLoader prefetching only.
Refactor loop into peek + prefetched iteration. No metric computation
changes. Expected speedup: ~3x on NFS/zip workloads.

**Phase 2 (moderate impact, low risk):** Wire `num_workers` into
`evaluate_rays_samples` (currently hardcoded sequential). Same pattern.

**Phase 3 (low impact, moderate complexity):** Deferred batched LPIPS.
Requires restructuring per-file metric accumulation. Expected additional
speedup: ~5-10% on top of Phase 1.
