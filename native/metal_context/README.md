# Metal Context Engine native foundation

This directory contains the optional phase-one native kernel package.  It is
deliberately independent of the scheduler and page-owner implementation:

* `kernels/paged_decode.metal` is a BF16, online-softmax paged decode kernel.
  It accepts non-contiguous `int32` page tables, GQA, head dimension 128,
  block sizes 16/32, and partial tail blocks.
* `src/python_module.mm` is a small CPython/Metal bridge.  Its initial ABI
  accepts contiguous host buffers containing BF16 bits as `uint16` buffers and
  returns a float32 byte buffer.  The copies are intentionally visible; this
  is a correctness/build foundation, not a serving-performance claim.
* `scripts/build_metal_context.py` compiles the checked-in shader to a
  `_metal_context.metallib`.  The optional setuptools hook packages that
  library beside the extension.

The logical page layout at this boundary is:

```text
query       [batch, query_heads, head_dim]
key/value   [page, kv_heads, block_offset, head_dim]
page_table  [batch, logical_block]
seq_lens    [batch]
```

The host validates shapes, native-endian PEP-3118 formats, page bounds for
every live token, finite BF16 data, conservative float32 dot/score bounds, and
the supported geometry before allocating any Metal buffer.  Unused page table
entries may be `-1`; live entries may not be.  The shader repeats the
page-bound guard so a malformed low-level dispatch cannot read outside the
page buffer, while the host remains responsible for surfacing the error.

## Building

Normal installs remain pure Python/MLX.  On a macOS/Xcode builder, opt in to
the native build explicitly:

```sh
VLLM_MLX_BUILD_METAL_CONTEXT=1 python -m pip install -e .
```

The strict mode fails if the host is not Apple Silicon or
`xcrun metal`/`xcrun metallib` is missing.  Normal installs leave the optional
extension disabled; the fallback module then reports the exact reason and an
explicit `metal-context` selection must fail closed.

For a standalone library build:

```sh
python scripts/build_metal_context.py --output /tmp/_metal_context.metallib
```

The extension's `capabilities()` record is the source of truth for whether the
device, ABI, and packaged library are usable.  A compiled extension is not a
serving qualification or performance result.
