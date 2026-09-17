# Metal Context page runtime

This package adds the ownership layer below the future serving executor. It is
not a scheduler and it is not enabled by `--attention-backend` yet.

`vllm_mlx.metal_context_runtime.MetalContextPageRuntime` owns a preallocated
BF16 KV page pool with one physical page layout shared across layers:

```text
[page, layer, kv_head, block_offset, head_dim]
```

The lifecycle boundary is deliberately small:

- allocate request handles and physical pages;
- append `[tokens, kv_heads, head_dim]` keys/values;
- attach and fork immutable prefix page chains;
- copy a shared page before a request writes to it;
- release references and evict only unreferenced pages;
- dispatch the phase-one native paged-decode kernel or the explicit NumPy
  correctness mode;
- maintain GPU-ready page tables/sequence lengths incrementally on lifecycle
  mutations, so steady-state decode performs no CPU page-chain traversal;
- report ownership, dispatch, preallocation/copy, metadata-byte, and bounded
  attention-validation metrics (`dispatches`/`dispatch_failures`, with
  truthful `native_*` and `oracle_*` aliases); and
- tear down deterministically.

Native execution is selected explicitly with `execution="native"`. It requires
the compiled `_metal_context` extension's `PageRuntime` type, a matching ABI,
and a usable Metal device. In this mode the Python class is only an adapter:
allocation, KV ownership, prefix references, copy-on-write, eviction, metrics,
decode, and teardown remain owned by the compiled runtime. Missing or stale
capabilities raise `MetalContextCapabilityError`. Native dispatch failures are
propagated; the runtime never silently switches to the NumPy oracle.
`execution="numpy"` is a portable test mode and is not a serving fallback.

Persistence is intentionally not part of this package. `snapshot()` and
`restore()` raise `NotImplementedError` until the separate persistence/tiering
package adds versioned identities, checksummed pages, tenant namespaces, and
atomic manifests. No Python objects are serialized here.

Prefix creation is only allowed after every supported layer has the same
non-zero populated length. If `token_count` is supplied, it must equal that
complete length in both NumPy and native modes; partial layer/page-chain
prefixes are intentionally rejected until they have an explicit immutable
ownership contract.

The page runtime is currently not wired into the scheduler or serving
defaults. Its focused lifecycle tests are in
`tests/test_metal_context_page_runtime.py` and compare decode results with the
shared NumPy/MLX oracle contract.
