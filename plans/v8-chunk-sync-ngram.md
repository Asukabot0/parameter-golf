# Consensus Plan v8: Chunk-Level GPU Sync for N-gram Eval Cache

**Date**: 2026-03-26
**Goal**: 对齐 PR #779/809 的 chunk-level cache 架构，每 GPU cache 从 4M→32M tokens
**Current**: 0.9674 BPB (PR #727, per-GPU partition)
**Target**: ~0.30 BPB (matching PR #809's architecture)
**Mode**: RALPLAN-DR Consensus (Planner v3, Architect v2 CONDITIONAL APPROVE, Critic v2 ACCEPT-WITH-RESERVATIONS)
**Status**: FINAL

---

## ADR

- **Decision**: Chunk-level GPU sync + vectorized bincount cache update + orders 2-7
- **Drivers**: per-GPU cache 4M→32M tokens 是 0.30 BPB 的关键路径 (scientist report 确认)
- **Alternatives considered**:
  - (1) 优化 pre-fill only — rejected: 无法跨 rank 共享 eval 期间的 cache 更新
  - (2) NCCL cache table all-reduce — rejected: 48MB/sync 开销，每 chunk 需同步
  - (3) Numba @njit — rejected by Critic: prefill 已删除，bincount 0.66s/chunk 足够
- **Why chosen**: 匹配 PR #779/809 验证架构，timing ~50s 在 eval 预算内
- **Consequences**: 16 次 all-reduce (negligible)，每 chunk 0.66s CPU update
- **Follow-ups**: 8xH100 验证后可叠加 kNN 补充或 per-order multipliers

---

## Implementation Tasks

### Task 1: Chunk-level eval loop (core)

Replace per-GPU window partition (lines 1063-1065) with chunk-based iteration:

```python
chunk_size = int(os.environ.get("NGRAM_CHUNK_SIZE", "4000000"))  # 4M tokens
chunk_starts = list(range(0, total_tokens, chunk_size))

eval_start_time = time.perf_counter()

for ci, cs in enumerate(chunk_starts):
    ce = min(cs + chunk_size, total_tokens)
    chunk_windows = [ws for ws in window_starts if cs <= ws < ce]
    my_chunk_windows = chunk_windows[rank::world_size]  # interleaved assignment

    for bi in range(0, len(my_chunk_windows), batch_seqs):
        batch_ws = my_chunk_windows[bi:bi + batch_seqs]
        bsz = len(batch_ws)
        # PAD to batch_seqs to avoid torch.compile recompilation
        x_batch = torch.zeros(batch_seqs, seq_len, ...)
        y_batch = torch.zeros(batch_seqs, seq_len, ...)
        # ... fill first bsz entries, score, mask loss to bsz

    # Per-chunk all-reduce (3 scalars only, NOT cache tables)
    if distributed:
        dist.all_reduce(loss_sum); dist.all_reduce(token_count); dist.all_reduce(byte_count)

    # ALL GPUs independently update cache from full chunk range (deterministic, no NCCL)
    if use_ngram:
        _bulk_cache_update(val_np, cs + 1, ce, ctx_tables, full_tables, ...)

    # Upfront timing go/no-go after 2 chunks
    if ci == 1 and use_ngram:
        elapsed = time.perf_counter() - eval_start_time
        est_total = elapsed / 2 * len(chunk_starts)
        max_eval = float(os.environ.get("MAX_EVAL_SECONDS", "550"))
        if est_total > max_eval:
            print(f"ngram_abort: est {est_total:.0f}s > {max_eval}s, disabling cache")
            use_ngram = False
```

Key properties:
- Window assigned to chunk where window_start falls
- Cache update uses `val_np[cs:ce]` (ALL tokens in chunk, not just scored ones)
- Each GPU computes identical cache update independently (no NCCL for cache)
- all-reduce covers only loss/token/byte accumulators (3 float64 scalars)

### Task 2: Bulk vectorized cache update function

```python
def _bulk_cache_update(val_np, start, end, ctx_tables, full_tables,
                        ng_primes, ng_mask, ngram_min_order, _n_orders):
    for oi in range(_n_orders):
        ctx_w = ngram_min_order + oi - 1
        pos = np.arange(max(start, ctx_w + 1), end + 1, dtype=np.int64)
        if len(pos) == 0:
            continue
        ctx_hash = np.zeros(len(pos), dtype=np.uint64)
        for k in range(ctx_w):
            ctx_hash ^= val_np[pos - ctx_w + k].astype(np.uint64) * ng_primes[k % len(ng_primes)]
        ctx_key = (ctx_hash & ng_mask).astype(np.int64)
        tgt = val_np[pos].astype(np.uint64)
        full_key = ((ctx_hash ^ (tgt * ng_primes[ctx_w % len(ng_primes)])) & ng_mask).astype(np.int64)
        ctx_tables[oi] += np.bincount(ctx_key, minlength=len(ctx_tables[oi])).astype(np.uint32)
        full_tables[oi] += np.bincount(full_key, minlength=len(full_tables[oi])).astype(np.uint32)
```

### Task 3: Cleanup + defaults

- Delete pre-fill loop (lines 1107-1130)
- `NGRAM_ORDER` default: 7 (orders 2-7, 6 hash tables)
- Trim primes array to 7 entries
- Keep per-order entropy centers (already implemented, lines 1224-1237)
- Pad last batch to batch_seqs for torch.compile stability

### Task 4: Upfront timing go/no-go

After chunk 1 (2 chunks completed), extrapolate total eval time.
If over budget → disable n-gram for remaining chunks (pure LM scoring).
Fall back to scoring remaining chunks without cache, NOT to old per-GPU partition.

---

## Timing Estimate (8xH100, 6 orders, 16 chunks of 4M)

| Component | Time |
|-----------|------|
| GPU forward (all chunks) | ~39s |
| Cache update (bincount, 16 chunks) | ~10.5s |
| All-reduce (16 × 3 scalars) | <0.5s |
| **Total** | **~50s** |

## Acceptance Criteria

1. `python -c "import py_compile; py_compile.compile('train_gpt.py')"` passes
2. `NGRAM_CACHE=0` training path unaffected
3. No Numba dependency
4. Code lines < 2100
5. All changes in eval_val_sliding only

## Review History

| Round | Reviewer | Verdict | Key Feedback |
|-------|---------|---------|-------------|
| v1 | Architect | CONDITIONAL APPROVE | Drop multipliers, add pipelining, orders 2-7 |
| v1 | Critic | REJECT | CPU timing 5x underestimated, no abort, Task 3 no-op |
| v2 | Architect | CONDITIONAL APPROVE | Upfront go/no-go, pad batch_seqs, orders 2-7 |
| v2 | Critic | ACCEPT-WITH-RESERVATIONS | Drop Numba (not needed), specify bincount path |
| v3 | Planner | FINAL | All feedback incorporated |
