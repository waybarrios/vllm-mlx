// SPDX-License-Identifier: Apache-2.0
//
// The first Metal Context Engine kernel.  The host API intentionally keeps
// the layout explicit so that the page runtime can validate every stride
// before dispatching it.
//
// Inputs:
//   query       [batch, query_heads, head_dim]       BF16 bits (ushort)
//   key_pages   [pages, kv_heads, block_size, head_dim] BF16 bits
//   value_pages [pages, kv_heads, block_size, head_dim] BF16 bits
//   page_table  [batch, max_blocks]                 int32
//   seq_lens    [batch]                             int32
//
// Output:
//   output      [batch, query_heads, head_dim]       float32
//
// A threadgroup owns one (request, query-head) output vector.  The online
// softmax state is updated token by token, so pages need not be physically
// contiguous.  Query heads are mapped to KV heads by the GQA ratio.

#include <metal_stdlib>

using namespace metal;

struct PagedDecodeParams {
  uint batch_size;
  uint query_heads;
  uint kv_heads;
  uint head_dim;
  uint block_size;
  uint max_blocks;
  uint page_count;
  uint page_table_stride;
  float scale;
  uint reserved0;
  uint reserved1;
  uint reserved2;
};

// MLX stores BF16 as the IEEE-754 upper 16 bits of a float.  Keeping the
// storage type as ushort avoids depending on the availability of the MSL
// bfloat type on the oldest supported Apple GPU family.
inline float bf16_to_float(ushort bits) {
  return as_type<float>(uint(bits) << 16);
}

kernel void metal_context_paged_decode(
    device const ushort* query [[buffer(0)]],
    device const ushort* key_pages [[buffer(1)]],
    device const ushort* value_pages [[buffer(2)]],
    device const int* page_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant PagedDecodeParams& p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  // The host dispatches exactly one 128-thread group per output vector.  A
  // guard remains here because it is cheap and prevents accidental writes if
  // a future caller uses a larger grid.
  const uint request = tg_pos.x / p.query_heads;
  const uint query_head = tg_pos.x % p.query_heads;
  if (request >= p.batch_size || query_head >= p.query_heads || tid >= 128) {
    return;
  }

  // Phase-1 qualification is head_dim=128.  Keeping this guard in the shader
  // makes an unsupported dispatch memory-safe even if host validation is
  // bypassed by a future low-level caller.
  if (p.head_dim != 128 || p.kv_heads == 0 ||
      (p.query_heads % p.kv_heads) != 0) {
    return;
  }

  threadgroup float partial_dot[128];
  threadgroup float softmax_state[4];

  const uint query_base = (request * p.query_heads + query_head) * p.head_dim;
  const uint q_per_kv = p.query_heads / p.kv_heads;
  const uint kv_head = query_head / q_per_kv;
  const int requested_length = seq_lens[request];
  const uint sequence_length = requested_length > 0
      ? min(uint(requested_length), p.max_blocks * p.block_size)
      : 0;

  // Each lane owns one output dimension for head_dim=128.  The accumulators
  // stay in float32 even though K/V are BF16.
  float accumulator = 0.0f;
  if (tid == 0) {
    softmax_state[0] = -INFINITY;  // running max
    softmax_state[1] = 0.0f;       // running denominator
    softmax_state[2] = 0.0f;       // exp(old_max - new_max)
    softmax_state[3] = 0.0f;       // exp(score - new_max)
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint token = 0; token < sequence_length; ++token) {
    const uint logical_block = token / p.block_size;
    const uint block_offset = token % p.block_size;
    const int physical_page = page_table[
        request * p.page_table_stride + logical_block];

    // The host rejects invalid page IDs.  Retaining the check in the shader
    // prevents a malformed table from turning into an out-of-bounds GPU read;
    // a skipped page is surfaced as an error by the host-side validation path.
    // Do not `continue` here: every lane must execute the same barriers.
    const bool page_valid = physical_page >= 0 &&
        uint(physical_page) < p.page_count;

    const uint key_base = page_valid
        ? ((uint(physical_page) * p.kv_heads + kv_head) * p.block_size +
           block_offset) * p.head_dim
        : 0;

    float dot_part = 0.0f;
    if (page_valid && tid < p.head_dim) {
      dot_part = bf16_to_float(query[query_base + tid]) *
          bf16_to_float(key_pages[key_base + tid]);
    }
    partial_dot[tid] = dot_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for a fixed 128-wide group.  All accesses are in-bounds and
    // every lane participates, including when a future head dimension uses
    // fewer lanes.
    for (uint stride = 64; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial_dot[tid] += partial_dot[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      const float score = partial_dot[0] * p.scale;
      const float old_max = softmax_state[0];
      const float new_max = max(old_max, score);
      // exp(-inf - -inf) is NaN on some GPU families.  The explicit branch
      // keeps the empty-prefix and first-token cases deterministic.
      const float old_weight = page_valid && isfinite(old_max)
          ? exp(old_max - new_max) * softmax_state[1]
          : (page_valid ? 0.0f : softmax_state[1]);
      const float token_weight = page_valid ? exp(score - new_max) : 0.0f;
      if (page_valid) {
        softmax_state[0] = new_max;
        softmax_state[1] = old_weight + token_weight;
        softmax_state[2] = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        softmax_state[3] = token_weight;
      } else {
        softmax_state[2] = 1.0f;
        softmax_state[3] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.head_dim) {
      const float value = page_valid ? bf16_to_float(value_pages[key_base + tid])
                                     : 0.0f;
      accumulator = accumulator * softmax_state[2] +
          value * softmax_state[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid < p.head_dim) {
    const uint output_base = (request * p.query_heads + query_head) * p.head_dim;
    const float denominator = softmax_state[1];
    output[output_base + tid] = denominator > 0.0f
        ? accumulator / denominator
        : 0.0f;
  }
}
