// SPDX-License-Identifier: Apache-2.0
//
// Native ownership runtime for the Metal Context Engine.

#include "page_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace metal_context {
namespace {

constexpr uint32_t kSupportedHeadDim = 128;
constexpr uint32_t kMinBlockSize = 16;
constexpr uint32_t kMaxBlockSize = 32;
constexpr uint32_t kThreadsPerThreadgroup = 128;
constexpr uint64_t kMaximumMetadataBytes = UINT64_C(256) * 1024 * 1024;

struct PagedDecodeParams {
  uint32_t batch_size;
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t max_blocks;
  uint32_t page_count;
  uint32_t page_table_stride;
  float scale;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(PagedDecodeParams) == 48, "Metal parameter ABI changed");

uint64_t checked_product(
    std::initializer_list<uint64_t> values,
    const char* what) {
  uint64_t result = 1;
  for (uint64_t value : values) {
    if (value != 0 &&
        result > std::numeric_limits<uint64_t>::max() / value) {
      throw std::invalid_argument(std::string(what) + " overflows uint64");
    }
    result *= value;
  }
  return result;
}

uint64_t make_handle(uint32_t generation, uint32_t slot) {
  return (static_cast<uint64_t>(generation) << 32) | slot;
}

uint32_t handle_slot(uint64_t handle) {
  return static_cast<uint32_t>(handle & UINT64_C(0xffffffff));
}

uint32_t handle_generation(uint64_t handle) {
  return static_cast<uint32_t>(handle >> 32);
}

uint32_t next_generation(uint32_t generation) {
  ++generation;
  return generation == 0 ? 1 : generation;
}

uint32_t validated_page_count(uint32_t value) {
  if (value == 0) {
    throw std::invalid_argument("max_pages must be positive");
  }
  // Page tables and the Metal shader use signed int32 page IDs.
  if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("max_pages must not exceed INT32_MAX");
  }
  return value;
}

uint32_t validated_block_capacity(uint32_t value, uint32_t block_size) {
  if (value == 0) {
    throw std::invalid_argument("max_blocks_per_request must be positive");
  }
  if (block_size == 0 ||
      value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) /
          block_size) {
    throw std::invalid_argument(
        "max_blocks_per_request * block_size exceeds the int32 sequence limit");
  }
  return value;
}

uint32_t validated_request_capacity(uint32_t value) {
  if (value == 0) {
    throw std::invalid_argument("max_requests must be positive");
  }
  return value;
}

uint64_t validated_metadata_elements(
    uint32_t request_capacity,
    uint32_t per_request_elements,
    const char* what) {
  const uint64_t elements = checked_product(
      {request_capacity, per_request_elements}, what);
  if (elements > kMaximumMetadataBytes / sizeof(int32_t)) {
    throw std::invalid_argument(
        std::string(what) + " exceeds the 256 MiB qualified metadata limit");
  }
  return elements;
}

bool finite_bf16(uint16_t bits) {
  return (bits & 0x7f80u) != 0x7f80u;
}

float bf16_to_float(uint16_t bits) {
  const uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

bool all_finite_bf16(const uint16_t* values, size_t count) {
  if (values == nullptr) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    if (!finite_bf16(values[index])) {
      return false;
    }
  }
  return true;
}

float max_abs_bf16(const uint16_t* values, size_t count) {
  float maximum = 0.0f;
  for (size_t index = 0; index < count; ++index) {
    maximum = std::max(maximum, std::fabs(bf16_to_float(values[index])));
  }
  return maximum;
}

bool validate_attention_envelope(
    const uint16_t* query,
    size_t query_elements,
    uint32_t sequence_length,
    float max_key,
    float max_value,
    uint32_t head_dim,
    float scale,
    std::string* error) {
  if (!std::isfinite(scale)) {
    *error = "scale must be finite";
    return false;
  }
  if (!all_finite_bf16(query, query_elements) ||
      !std::isfinite(max_key) || !std::isfinite(max_value)) {
    *error = "query and live KV metadata must contain only finite BF16 values";
    return false;
  }
  const float max_query = max_abs_bf16(query, query_elements);
  // Keep the same conservative half-FLT_MAX envelope as the foundation
  // bridge.  Finite BF16 products can still overflow a float32 reduction;
  // rejecting them here prevents Inf-Inf/NaN states in online softmax.
  constexpr long double kFloat32SafeMagnitude =
      static_cast<long double>(std::numeric_limits<float>::max()) * 0.5L;
  const long double max_dot =
      static_cast<long double>(max_query) *
      static_cast<long double>(max_key) *
      static_cast<long double>(head_dim);
  if (max_dot > kFloat32SafeMagnitude) {
    *error =
        "finite BF16 query/key magnitudes may overflow the float32 dot "
        "product at head_dim 128; reduce magnitudes";
    return false;
  }
  const long double max_score =
      max_dot * std::fabs(static_cast<long double>(scale));
  if (max_score > kFloat32SafeMagnitude) {
    *error =
        "finite BF16 query/key magnitudes and scale may overflow the "
        "float32 attention score; reduce scale or magnitudes";
    return false;
  }
  // Online softmax keeps the final value as a normalized weighted average,
  // but the shader still forms a running value numerator.  Bound the
  // worst-case finite accumulation so an extreme BF16 V cannot turn that
  // intermediate into Inf before normalization.  The request maxima are
  // maintained on append/prefix operations, so this check is O(1) in the
  // context length.
  const long double max_value_accumulation =
      static_cast<long double>(max_value) * sequence_length;
  if (max_value_accumulation > kFloat32SafeMagnitude) {
    *error =
        "finite BF16 value magnitudes and sequence length may overflow the "
        "float32 value accumulation; reduce values or context length";
    return false;
  }
  return true;
}

std::string metallib_path() {
  const char* override_path =
      std::getenv("VLLM_MLX_METAL_CONTEXT_METALLIB");
  if (override_path != nullptr && override_path[0] != '\0') {
    return override_path;
  }
  // The Python bridge normally supplies the package path through the same
  // environment override during tests.  Keep the fallback explicit instead
  // of compiling or loading shader source at runtime.
  return {};
}

struct KernelRuntime {
  std::mutex mutex;
  uint64_t active_runtimes = 0;
  bool attempted = false;
  bool available = false;
  std::string path;
  std::string error;
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipeline = nil;
};

KernelRuntime g_kernel;

void reset_kernel_unlocked() {
  g_kernel.device = nil;
  g_kernel.pipeline = nil;
  g_kernel.attempted = false;
  g_kernel.available = false;
  g_kernel.path.clear();
  g_kernel.error.clear();
}

void retain_kernel_runtime() {
  std::lock_guard<std::mutex> lock(g_kernel.mutex);
  ++g_kernel.active_runtimes;
}

void release_kernel_runtime() {
  std::lock_guard<std::mutex> lock(g_kernel.mutex);
  if (g_kernel.active_runtimes == 0) {
    return;
  }
  --g_kernel.active_runtimes;
  if (g_kernel.active_runtimes == 0) {
    // Each dispatch retains its pipeline locally, and a runtime drains its
    // active leases before calling this function.  The final reference can
    // therefore release the shared pipeline without a process-wide dispatch
    // mutex.
    reset_kernel_unlocked();
  }
}

void shutdown_kernel_if_unused_impl() {
  std::lock_guard<std::mutex> lock(g_kernel.mutex);
  if (g_kernel.active_runtimes == 0) {
    reset_kernel_unlocked();
  }
}

bool compiled_for_apple_silicon() {
#if defined(__arm64__) || defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

std::string metal_error(NSError* error, const char* fallback) {
  if (error != nil && error.localizedDescription != nil) {
    const char* description = [error.localizedDescription UTF8String];
    if (description != nullptr && description[0] != '\0') {
      return description;
    }
  }
  return fallback;
}

bool ensure_kernel(const std::string& path) {
  std::lock_guard<std::mutex> lock(g_kernel.mutex);
  if (g_kernel.attempted && g_kernel.path == path) {
    return g_kernel.available;
  }
  if (g_kernel.active_runtimes != 0 && g_kernel.attempted &&
      g_kernel.path != path) {
    g_kernel.error =
        "native Metal Context runtimes must share one metallib path while "
        "dispatches are active";
    return false;
  }
  g_kernel.attempted = true;
  g_kernel.available = false;
  g_kernel.path = path;
  g_kernel.error.clear();
  g_kernel.device = nil;
  g_kernel.pipeline = nil;

  if (!compiled_for_apple_silicon()) {
    g_kernel.error =
        "the Metal Context Engine requires an Apple Silicon arm64 build";
    return false;
  }
  if (path.empty()) {
    g_kernel.error =
        "the packaged _metal_context.metallib could not be located; set "
        "VLLM_MLX_METAL_CONTEXT_METALLIB for an explicit path";
    return false;
  }
  try {
    if (!std::filesystem::exists(path)) {
      g_kernel.error = "the Metal library does not exist at " + path;
      return false;
    }
  } catch (const std::filesystem::filesystem_error& exception) {
    g_kernel.error = "could not inspect the Metal library path: ";
    g_kernel.error += exception.what();
    return false;
  }

  @autoreleasepool {
    do {
      g_kernel.device = MTLCreateSystemDefaultDevice();
      if (g_kernel.device == nil) {
        g_kernel.error = "MTLCreateSystemDefaultDevice returned no GPU";
        break;
      }
      NSData* bytes = [NSData
          dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
      if (bytes == nil || bytes.length == 0) {
        g_kernel.error = "the packaged Metal library could not be read";
        break;
      }
      void* owned_bytes = std::malloc(bytes.length);
      if (owned_bytes == nullptr) {
        g_kernel.error = "could not allocate Metal library storage";
        break;
      }
      std::memcpy(owned_bytes, bytes.bytes, bytes.length);
      dispatch_data_t library_data = dispatch_data_create(
          owned_bytes,
          bytes.length,
          nullptr,
          DISPATCH_DATA_DESTRUCTOR_FREE);
      if (library_data == nullptr) {
        std::free(owned_bytes);
        g_kernel.error = "could not map the Metal library data";
        break;
      }
      NSError* error = nil;
      id<MTLLibrary> library =
          [g_kernel.device newLibraryWithData:library_data error:&error];
      if (library == nil) {
        g_kernel.error = metal_error(error, "Metal rejected the library");
        break;
      }
      id<MTLFunction> function =
          [library newFunctionWithName:@"metal_context_paged_decode"];
      if (function == nil) {
        g_kernel.error =
            "the packaged Metal library is missing "
            "metal_context_paged_decode";
        break;
      }
      g_kernel.pipeline = [g_kernel.device
          newComputePipelineStateWithFunction:function
                                          error:&error];
      if (g_kernel.pipeline == nil) {
        g_kernel.error =
            metal_error(error, "Metal could not create the kernel pipeline");
        break;
      }
      g_kernel.available = true;
    } while (false);
  }
  return g_kernel.available;
}

bool dispatch_kernel(
    id<MTLCommandQueue> queue,
    id<MTLBuffer> query_buffer,
    NSUInteger query_offset,
    id<MTLBuffer> kv_buffer,
    NSUInteger key_offset,
    NSUInteger value_offset,
    id<MTLBuffer> page_table_buffer,
    NSUInteger page_table_offset,
    id<MTLBuffer> sequence_lengths_buffer,
    NSUInteger sequence_lengths_offset,
    id<MTLBuffer> output_buffer,
    NSUInteger output_offset,
    const uint16_t* query,
    size_t query_elements,
    uint32_t batch_size,
    uint32_t query_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_blocks,
    uint32_t page_count,
    float scale,
    const std::string& path,
    std::vector<float>* output,
    std::string* error,
    uint64_t* query_copies,
    uint64_t* output_copies) {
  if (queue == nil || query_buffer == nil || kv_buffer == nil ||
      page_table_buffer == nil || sequence_lengths_buffer == nil ||
      output_buffer == nil || query == nullptr || output == nullptr ||
      error == nullptr) {
    if (error != nullptr) {
      *error = "paged decode received a null runtime buffer";
    }
    return false;
  }
  if (head_dim != kSupportedHeadDim ||
      (block_size != kMinBlockSize && block_size != kMaxBlockSize) ||
      kv_heads == 0 || query_heads == 0 || query_heads % kv_heads != 0 ||
      batch_size == 0 || max_blocks == 0 || page_count == 0 ||
      !std::isfinite(scale)) {
    *error = "paged decode geometry is outside the Metal Context v1 ABI";
    return false;
  }
  const uint64_t expected_query =
      checked_product({batch_size, query_heads, head_dim}, "query size");
  if (expected_query > std::numeric_limits<uint32_t>::max() ||
      query_elements != expected_query) {
    *error = "query element count does not match the runtime geometry";
    return false;
  }
  const uint64_t output_elements = expected_query;
  if (output_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    *error = "paged decode output exceeds native size limits";
    return false;
  }

  if (!ensure_kernel(path)) {
    std::lock_guard<std::mutex> lock(g_kernel.mutex);
    *error = "native Metal Context kernel unavailable: " + g_kernel.error;
    return false;
  }
  id<MTLComputePipelineState> pipeline = nil;
  {
    // Retain the pipeline locally, but release the global lock before command
    // encoding.  Independent PageRuntime queues must not serialize on the
    // process-wide lifecycle mutex.
    std::lock_guard<std::mutex> lock(g_kernel.mutex);
    if (!g_kernel.available || g_kernel.pipeline == nil) {
      *error = "native Metal Context kernel lost its pipeline";
      return false;
    }
    pipeline = g_kernel.pipeline;
  }

  @autoreleasepool {
    const size_t query_bytes =
        static_cast<size_t>(expected_query) * sizeof(uint16_t);
    const size_t output_bytes =
        static_cast<size_t>(output_elements) * sizeof(float);
    const size_t page_table_bytes = checked_product(
        {max_blocks, sizeof(int32_t)}, "page table dispatch range");
    const size_t sequence_bytes = sizeof(int32_t);
    const auto buffer_range_valid = [](id<MTLBuffer> buffer,
                                       NSUInteger offset,
                                       size_t bytes) {
      return buffer != nil && buffer.contents != nullptr &&
          bytes <= std::numeric_limits<NSUInteger>::max() &&
          offset <= buffer.length &&
          static_cast<NSUInteger>(bytes) <= buffer.length - offset;
    };
    const size_t kv_bytes = checked_product(
        {static_cast<uint64_t>(page_count), static_cast<uint64_t>(block_size),
         static_cast<uint64_t>(kv_heads), static_cast<uint64_t>(head_dim),
         sizeof(uint16_t)},
        "KV dispatch range");
    if (!buffer_range_valid(query_buffer, query_offset, query_bytes) ||
        !buffer_range_valid(output_buffer, output_offset, output_bytes) ||
        !buffer_range_valid(kv_buffer, key_offset, kv_bytes) ||
        !buffer_range_valid(kv_buffer, value_offset, kv_bytes) ||
        !buffer_range_valid(
            page_table_buffer, page_table_offset, page_table_bytes) ||
        !buffer_range_valid(
            sequence_lengths_buffer, sequence_lengths_offset, sequence_bytes)) {
      *error = "paged decode runtime buffer range is outside its allocation";
      return false;
    }
    std::memcpy(
        static_cast<uint8_t*>(query_buffer.contents) + query_offset,
        query,
        query_bytes);
    if (query_copies != nullptr) {
      ++*query_copies;
    }
    std::memset(
        static_cast<uint8_t*>(output_buffer.contents) + output_offset,
        0,
        output_bytes);

    PagedDecodeParams params{};
    params.batch_size = batch_size;
    params.query_heads = query_heads;
    params.kv_heads = kv_heads;
    params.head_dim = head_dim;
    params.block_size = block_size;
    params.max_blocks = max_blocks;
    params.page_count = page_count;
    params.page_table_stride = max_blocks;
    params.scale = scale;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        command == nil ? nil : [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      *error = "Metal could not create the paged decode command encoder";
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query_buffer offset:query_offset atIndex:0];
    [encoder setBuffer:kv_buffer offset:key_offset atIndex:1];
    [encoder setBuffer:kv_buffer offset:value_offset atIndex:2];
    [encoder setBuffer:page_table_buffer offset:page_table_offset atIndex:3];
    [encoder setBuffer:sequence_lengths_buffer
              offset:sequence_lengths_offset
            atIndex:4];
    [encoder setBuffer:output_buffer offset:output_offset atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    const MTLSize grid =
        MTLSizeMake(static_cast<NSUInteger>(batch_size) * query_heads, 1, 1);
    const MTLSize group = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      *error = metal_error(
          command.error, "Metal paged decode command did not complete");
      return false;
    }
    output->resize(static_cast<size_t>(output_elements));
    std::memcpy(
        output->data(),
        static_cast<const uint8_t*>(output_buffer.contents) + output_offset,
        output_bytes);
    if (output_copies != nullptr) {
      ++*output_copies;
    }
  }
  return true;
}

}  // namespace

// Export only the guarded module-shutdown hook.  The kernel object and its
// mutex remain translation-unit private; callers cannot tear down resources
// without passing through the active PageRuntime reference count.
void shutdown_kernel_if_unused() {
  shutdown_kernel_if_unused_impl();
}

uint64_t page_runtime_sequence_buffer_offset(
    uint32_t request_slot,
    uint32_t num_layers,
    uint32_t layer) {
  if (num_layers == 0 || layer >= num_layers) {
    throw std::invalid_argument(
        "sequence buffer layer is outside the configured layer count");
  }
  const uint64_t request_base = checked_product(
      {request_slot, num_layers}, "sequence buffer request offset");
  return checked_product(
      {request_base + layer, sizeof(int32_t)},
      "sequence buffer byte offset");
}

id<MTLBuffer> allocate_runtime_buffer(
    id<MTLDevice> device,
    uint64_t bytes,
    uint64_t* allocation_counter) {
  if (device == nil) {
    return nil;
  }
  if (bytes > std::numeric_limits<NSUInteger>::max()) {
    throw std::invalid_argument(
        "native Metal buffer length exceeds NSUInteger capacity");
  }
  id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>(bytes)
                   options:MTLResourceStorageModeShared];
  if (buffer != nil && allocation_counter != nullptr) {
    ++*allocation_counter;
  }
  return buffer;
}

struct MutationPageSnapshot {
  uint32_t page = 0;
  bool allocated = false;
  uint32_t generation = 0;
  uint32_t references = 0;
  uint64_t last_used = 0;
};

struct PageRuntime::MutationToken::State {
  enum class Kind : uint8_t {
    kNone,
    kRequest,
    kPages,
    kPrefix,
    kEviction,
    kDecode,
  };

  Kind kind = Kind::kNone;
  bool armed = false;
  bool prefix_was_new = false;
  bool has_request_sequence = false;
  uint32_t request_slot = 0;
  uint32_t prefix_slot = 0;
  uint32_t first_block = 0;
  uint32_t prior_request_generation = 0;
  uint32_t new_generation = 0;
  uint32_t prior_prefix_generation = 0;
  uint32_t prior_request_sequence_length = 0;
  uint32_t prior_page_table_length = 0;
  uint64_t old_clock = 0;
  uint64_t old_page_allocations = 0;
  uint64_t old_eviction_count = 0;
  uint64_t old_prefix_allocations = 0;
  uint64_t old_request_allocations = 0;
  uint64_t old_metadata_copies = 0;
  uint64_t old_metadata_bytes = 0;
  uint64_t old_dispatches = 0;
  uint64_t old_dispatch_failures = 0;
  uint64_t old_native_dispatches = 0;
  uint64_t old_native_failures = 0;
  uint64_t old_query_copies = 0;
  uint64_t old_output_copies = 0;
  uint64_t old_attention_validation_bytes = 0;
  std::vector<int32_t> prior_request_pages;
  std::vector<int32_t> prior_layer_lengths;
  std::vector<int32_t> prefix_pages;
  std::vector<MutationPageSnapshot> page_snapshots;

  void reset() noexcept {
    kind = Kind::kNone;
    armed = false;
    prefix_was_new = false;
    has_request_sequence = false;
    request_slot = 0;
    prefix_slot = 0;
    first_block = 0;
    prior_request_generation = 0;
    new_generation = 0;
    prior_prefix_generation = 0;
    prior_request_sequence_length = 0;
    prior_page_table_length = 0;
    old_clock = 0;
    old_page_allocations = 0;
    old_eviction_count = 0;
    old_prefix_allocations = 0;
    old_request_allocations = 0;
    old_metadata_copies = 0;
    old_metadata_bytes = 0;
    old_dispatches = 0;
    old_dispatch_failures = 0;
    old_native_dispatches = 0;
    old_native_failures = 0;
    old_query_copies = 0;
    old_output_copies = 0;
    old_attention_validation_bytes = 0;
    prior_request_pages.clear();
    prior_layer_lengths.clear();
    prefix_pages.clear();
    page_snapshots.clear();
  }
};

PageRuntime::MutationToken::MutationToken()
    : state_(std::make_unique<State>()) {}

PageRuntime::MutationToken::~MutationToken() = default;

PageRuntime::MutationToken::MutationToken(MutationToken&&) noexcept = default;

PageRuntime::MutationToken& PageRuntime::MutationToken::operator=(
    MutationToken&&) noexcept = default;

struct PageRuntime::Impl {
  struct LayerStorage {
    id<MTLBuffer> buffer = nil;
    std::vector<uint16_t> host;
    uint16_t* data() {
      return buffer == nil ? host.data() : static_cast<uint16_t*>(buffer.contents);
    }
    const uint16_t* data() const {
      return buffer == nil ? host.data()
                           : static_cast<const uint16_t*>(buffer.contents);
    }
  };

  struct PageMeta {
    bool allocated = false;
    uint32_t generation = 0;
    uint32_t references = 0;
    uint64_t last_used = 0;
  };

  struct RequestMeta {
    bool live = false;
    uint32_t generation = 0;
    std::string id;
    uint32_t max_tokens = 0;
    uint32_t sequence_length = 0;
    uint32_t page_table_length = 0;
    std::vector<uint32_t> layer_lengths;
    std::vector<float> layer_max_key;
    std::vector<float> layer_max_value;
    std::vector<int32_t> pages;
  };

  struct PrefixMeta {
    bool live = false;
    uint32_t generation = 0;
    uint32_t token_count = 0;
    std::vector<float> layer_max_key;
    std::vector<float> layer_max_value;
    std::vector<int32_t> pages;
  };

  struct DispatchResult {
    bool validation_failed = false;
    bool dispatch_attempted = false;
    bool dispatched = false;
    uint64_t validation_bytes = 0;
    uint64_t query_copies = 0;
    uint64_t output_copies = 0;
  };

  static_assert(
      std::is_nothrow_move_assignable<RequestMeta>::value,
      "request replacement must publish without throwing");
  static_assert(
      std::is_nothrow_move_constructible<PrefixMeta>::value,
      "prefix replacement must publish without throwing");

  const uint32_t num_layers;
  const uint32_t num_attention_heads;
  const uint32_t num_key_value_heads;
  const uint32_t head_dim;
  const uint32_t block_size;
  const uint32_t max_pages;
  const uint32_t max_blocks_per_request;
  const uint32_t max_requests;
  const uint64_t page_elements;
  const uint64_t bytes_per_layer;
  const uint64_t query_elements;
  const uint64_t query_bytes_per_slot;
  const uint64_t output_bytes_per_slot;

  mutable std::mutex mutex;
  // Lock order is strict: ``Impl::mutex`` is acquired before a
  // ``slot_mutexes[slot]`` lock.  Code that has released ``Impl::mutex`` may
  // keep a slot lock across synchronous dispatch, but it must release the
  // slot lock before reacquiring ``Impl::mutex``.  In particular, dispatch
  // completion records metrics only after the slot lock is dropped.  No
  // helper called while holding a slot lock may acquire the runtime mutex.
  bool stopped = false;
  bool stopping = false;
  uint32_t active_dispatches = 0;
  std::condition_variable dispatch_cv;
  uint64_t clock = 0;
  uint64_t page_allocations = 0;
  uint64_t request_allocations = 0;
  uint64_t prefix_allocations = 0;
  uint64_t cow_events = 0;
  uint64_t eviction_count = 0;
  uint64_t release_count = 0;
  uint64_t cancellation_count = 0;
  uint64_t append_tokens = 0;
  uint64_t dispatches = 0;
  uint64_t dispatch_failures = 0;
  uint64_t native_dispatches = 0;
  uint64_t native_failures = 0;
  // Keep process-wide kernel teardown ordered with every live PageRuntime,
  // including host-only runtimes that never dispatch a shader.
  bool kernel_reference = false;

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;
  std::vector<LayerStorage> layers;
  std::vector<int32_t> page_table;
  std::vector<int32_t> layer_sequence_lengths;
  id<MTLBuffer> page_table_buffer = nil;
  id<MTLBuffer> sequence_lengths_buffer = nil;
  id<MTLBuffer> query_scratch_buffer = nil;
  id<MTLBuffer> output_scratch_buffer = nil;
  std::vector<std::mutex> slot_mutexes;
  std::vector<PageMeta> pages;
  std::vector<RequestMeta> requests;
  std::vector<PrefixMeta> prefixes;
  std::string library_path;
  uint64_t buffer_allocations = 0;
  uint64_t decode_buffer_allocations = 0;
  uint64_t query_copies = 0;
  uint64_t output_copies = 0;
  uint64_t metadata_copies = 0;
  uint64_t metadata_bytes = 0;
  uint64_t kv_copy_bytes = 0;
  uint64_t kv_pool_copies = 0;
  uint64_t attention_validation_bytes = 0;
  uint64_t decode_page_resolution_checks = 0;
  uint64_t snapshot_failures = 0;
  uint64_t restore_failures = 0;

  Impl(
      uint32_t layers_count,
      uint32_t attention_heads,
      uint32_t kv_heads,
      uint32_t dimension,
      uint32_t block,
      uint32_t page_count,
      uint32_t block_capacity,
      uint32_t request_capacity)
      : num_layers(layers_count),
        num_attention_heads(attention_heads),
        num_key_value_heads(kv_heads),
        head_dim(dimension),
        block_size(block),
        max_pages(validated_page_count(page_count)),
        max_blocks_per_request(validated_block_capacity(block_capacity, block)),
        max_requests(validated_request_capacity(request_capacity)),
        page_elements(checked_product(
            {kv_heads, block, dimension}, "page element count")),
        // One native allocation contains a key plane followed by a value
        // plane.  ``page_elements`` is the element count of one plane.
        bytes_per_layer(checked_product(
            {2, validated_page_count(page_count), kv_heads, block, dimension,
             sizeof(uint16_t)},
            "per-layer KV allocation")),
        query_elements(checked_product(
            {attention_heads, dimension}, "query element count")),
        query_bytes_per_slot(
            checked_product({query_elements, sizeof(uint16_t)}, "query scratch")),
        output_bytes_per_slot(checked_product(
            {query_elements, sizeof(float)}, "output scratch")),
        page_table(
            validated_metadata_elements(
                validated_request_capacity(request_capacity),
                validated_block_capacity(block_capacity, block),
                "page table"),
            -1),
        layer_sequence_lengths(validated_metadata_elements(
            validated_request_capacity(request_capacity),
            layers_count,
            "layer sequence lengths"), 0),
        slot_mutexes(validated_request_capacity(request_capacity)),
        pages(validated_page_count(page_count)),
        requests(validated_request_capacity(request_capacity)),
        prefixes() {
    if (num_layers == 0 || num_attention_heads == 0 ||
        num_key_value_heads == 0) {
      throw std::invalid_argument("layer and head counts must be positive");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
      throw std::invalid_argument(
          "num_attention_heads must be divisible by num_key_value_heads for GQA");
    }
    if (head_dim != kSupportedHeadDim) {
      throw std::invalid_argument("Metal Context v1 supports head_dim=128 only");
    }
    if (block_size != kMinBlockSize && block_size != kMaxBlockSize) {
      throw std::invalid_argument("Metal Context v1 supports block_size 16 or 32 only");
    }
    if (page_elements > std::numeric_limits<size_t>::max() ||
        bytes_per_layer > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("page runtime allocation exceeds native size limits");
    }
    const uint64_t all_page_elements = checked_product(
        {max_pages, page_elements}, "all-page shader index space");
    const uint64_t page_table_elements = checked_product(
        {max_requests, max_blocks_per_request}, "page-table index space");
    if (page_elements > std::numeric_limits<uint32_t>::max() ||
        all_page_elements > std::numeric_limits<uint32_t>::max() ||
        page_table_elements > std::numeric_limits<uint32_t>::max() ||
        query_elements > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument(
          "page or query index space exceeds the uint32 shader limit");
    }
    const uint64_t query_scratch_bytes = checked_product(
        {max_requests, query_bytes_per_slot}, "query scratch allocation");
    const uint64_t output_scratch_bytes = checked_product(
        {max_requests, output_bytes_per_slot}, "output scratch allocation");
    if (query_scratch_bytes > kMaximumMetadataBytes ||
        output_scratch_bytes > kMaximumMetadataBytes ||
        query_scratch_bytes > kMaximumMetadataBytes - output_scratch_bytes) {
      throw std::invalid_argument(
          "combined decode scratch allocation exceeds the 256 MiB qualified "
          "metadata limit");
    }
    // Do not let a missing/typoed capacity silently attempt a multi-hundred
    // gigabyte allocation during process startup.  Real qualification runs
    // can choose a larger explicit capacity in a later resource policy.
    constexpr uint64_t kMaximumLayerBytes = UINT64_C(64) * 1024 * 1024 * 1024;
    if (bytes_per_layer > kMaximumLayerBytes) {
      throw std::invalid_argument(
          "per-layer KV allocation exceeds the 64 GiB native safety limit");
    }

    requests.resize(max_requests);
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      layers.resize(num_layers);
      for (LayerStorage& layer : layers) {
        if (device != nil) {
          layer.buffer = allocate_runtime_buffer(
              device, bytes_per_layer, &buffer_allocations);
          if (layer.buffer == nil) {
            throw std::runtime_error(
                "Metal could not allocate the preallocated per-layer KV buffer");
          }
          std::memset(layer.buffer.contents, 0, static_cast<size_t>(bytes_per_layer));
        } else {
          layer.host.resize(
              static_cast<size_t>(bytes_per_layer / sizeof(uint16_t)),
              static_cast<uint16_t>(0));
        }
      }
      const uint64_t page_table_bytes = checked_product(
          {max_requests, max_blocks_per_request, sizeof(int32_t)},
          "page table bytes");
      const uint64_t sequence_bytes =
          checked_product(
              {max_requests, num_layers, sizeof(int32_t)},
              "sequence length bytes");
      if (device != nil) {
        command_queue = [device newCommandQueue];
        if (command_queue == nil) {
          throw std::runtime_error("Metal could not create a runtime command queue");
        }
        page_table_buffer = allocate_runtime_buffer(
            device, page_table_bytes, &buffer_allocations);
        sequence_lengths_buffer = allocate_runtime_buffer(
            device, sequence_bytes, &buffer_allocations);
        query_scratch_buffer = allocate_runtime_buffer(
            device, query_scratch_bytes, &buffer_allocations);
        output_scratch_buffer = allocate_runtime_buffer(
            device, output_scratch_bytes, &buffer_allocations);
        if (page_table_buffer == nil || sequence_lengths_buffer == nil ||
            query_scratch_buffer == nil || output_scratch_buffer == nil) {
          throw std::runtime_error(
              "Metal could not allocate the preallocated runtime buffers");
        }
        std::memset(
            page_table_buffer.contents, 0xff, static_cast<size_t>(page_table_bytes));
        std::memset(
            sequence_lengths_buffer.contents,
            0,
            static_cast<size_t>(sequence_bytes));
        std::memset(
            query_scratch_buffer.contents,
            0,
            static_cast<size_t>(query_scratch_bytes));
        std::memset(
            output_scratch_buffer.contents,
            0,
            static_cast<size_t>(output_scratch_bytes));
      }
    }
    retain_kernel_runtime();
    kernel_reference = true;
  }

  // Destruction is also a teardown entry point (for example, a constructor
  // replacement from the Python bridge).  Drain every synchronous decode
  // lease before releasing the buffers those leases reference.  Keep the
  // kernel release outside the runtime mutex, matching PageRuntime::shutdown.
  void shutdown_and_wait() {
    bool release_kernel = false;
    {
      std::unique_lock<std::mutex> lock(mutex);
      if (stopped) {
        return;
      }
      if (stopping) {
        dispatch_cv.wait(lock, [this] { return stopped; });
        return;
      }
      stopping = true;
      dispatch_cv.wait(lock, [this] { return active_dispatches == 0; });
      release_kernel = shutdown_unlocked();
      dispatch_cv.notify_all();
    }
    if (release_kernel) {
      release_kernel_runtime();
    }
  }

  ~Impl() { shutdown_and_wait(); }

  bool shutdown_unlocked() {
    if (stopped) {
      return false;
    }
    stopped = true;
    stopping = false;
    // Clear references before releasing Objective-C resources so a future
    // destructor cannot observe a half-torn-down page table.
    for (RequestMeta& request : requests) {
      request.live = false;
      request.pages.clear();
      request.layer_lengths.clear();
      request.layer_max_key.clear();
      request.layer_max_value.clear();
      request.sequence_length = 0;
      request.page_table_length = 0;
    }
    for (PrefixMeta& prefix : prefixes) {
      prefix.live = false;
      prefix.pages.clear();
      prefix.layer_max_key.clear();
      prefix.layer_max_value.clear();
      prefix.token_count = 0;
    }
    for (PageMeta& page : pages) {
      page.references = 0;
      page.allocated = false;
    }
    page_table.clear();
    layer_sequence_lengths.clear();
    layers.clear();
    command_queue = nil;
    page_table_buffer = nil;
    sequence_lengths_buffer = nil;
    query_scratch_buffer = nil;
    output_scratch_buffer = nil;
    device = nil;
    const bool release_kernel = kernel_reference;
    kernel_reference = false;
    return release_kernel;
  }

  bool check_running(std::string* error) const {
    if (stopped || stopping) {
      if (error != nullptr) {
        *error = "page runtime is shut down";
      }
      return false;
    }
    return true;
  }

  void finish_dispatch(const DispatchResult& result) {
    std::lock_guard<std::mutex> lock(mutex);
    if (active_dispatches > 0) {
      --active_dispatches;
    }
    if (result.validation_bytes != 0) {
      attention_validation_bytes += result.validation_bytes;
    }
    if (result.dispatched) {
      ++dispatches;
      native_dispatches = dispatches;
    } else if (result.validation_failed || result.dispatch_attempted) {
      ++dispatch_failures;
      native_failures = dispatch_failures;
    }
    query_copies += result.query_copies;
    output_copies += result.output_copies;
    dispatch_cv.notify_all();
  }

  using MutationState = PageRuntime::MutationToken::State;

  static MutationPageSnapshot snapshot_page(
      uint32_t page,
      const PageMeta& meta) {
    return MutationPageSnapshot{
        page,
        meta.allocated,
        meta.generation,
        meta.references,
        meta.last_used};
  }

  void save_metadata_counters(MutationState& state) const {
    state.old_metadata_copies = metadata_copies;
    state.old_metadata_bytes = metadata_bytes;
  }

  bool prepare_request_undo_unlocked(
      uint32_t slot,
      MutationState& state,
      std::string* error) {
    (void)error;
    state.reset();
    state.kind = MutationState::Kind::kRequest;
    state.request_slot = slot;
    const RequestMeta& request = requests[slot];
    state.prior_request_generation = request.generation;
    state.new_generation = next_generation(request.generation);
    state.old_request_allocations = request_allocations;
    save_metadata_counters(state);
    state.prior_request_pages.assign(
        page_table.begin() +
            static_cast<size_t>(slot) * max_blocks_per_request,
        page_table.begin() +
            static_cast<size_t>(slot + 1) * max_blocks_per_request);
    state.prior_layer_lengths.assign(
        layer_sequence_lengths.begin() +
            static_cast<size_t>(slot) * num_layers,
        layer_sequence_lengths.begin() +
            static_cast<size_t>(slot + 1) * num_layers);
    state.armed = true;
    return true;
  }

  bool prepare_pages_undo_unlocked(
      uint32_t slot,
      uint32_t first_block,
      uint32_t count,
      MutationState& state,
      std::string* error) {
    (void)error;
    state.reset();
    state.kind = MutationState::Kind::kPages;
    state.request_slot = slot;
    state.first_block = first_block;
    const RequestMeta& request = requests[slot];
    state.prior_request_generation = request.generation;
    state.prior_page_table_length = request.page_table_length;
    state.old_clock = clock;
    state.old_page_allocations = page_allocations;
    state.old_eviction_count = eviction_count;
    save_metadata_counters(state);
    state.prior_request_pages.assign(
        request.pages.begin() + first_block,
        request.pages.begin() + first_block + count);
    state.page_snapshots.reserve(count);
    std::vector<uint32_t> reserved;
    reserved.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t page = find_allocation_candidate_unlocked(reserved);
      if (page == max_pages) {
        state.reset();
        if (error != nullptr) {
          *error =
              "page capacity exhausted: every resident page is still referenced";
        }
        return false;
      }
      reserved.push_back(page);
      state.page_snapshots.push_back(snapshot_page(page, pages[page]));
    }
    state.armed = true;
    return true;
  }

  bool prepare_prefix_undo_unlocked(
      uint32_t prefix_slot,
      bool prefix_was_new,
      uint32_t prior_generation,
      uint32_t new_generation,
      bool has_request_sequence,
      uint32_t request_slot,
      uint32_t prior_request_sequence_length,
      const std::vector<int32_t>& pages_to_reference,
      MutationState& state) {
    state.reset();
    state.kind = MutationState::Kind::kPrefix;
    state.prefix_slot = prefix_slot;
    state.prefix_was_new = prefix_was_new;
    state.prior_prefix_generation = prior_generation;
    state.new_generation = new_generation;
    state.has_request_sequence = has_request_sequence;
    state.request_slot = request_slot;
    state.prior_request_sequence_length = prior_request_sequence_length;
    state.old_clock = clock;
    state.old_prefix_allocations = prefix_allocations;
    state.prefix_pages = pages_to_reference;
    state.page_snapshots.reserve(state.prefix_pages.size());
    for (int32_t page_value : state.prefix_pages) {
      if (page_value < 0 || static_cast<uint32_t>(page_value) >= pages.size()) {
        state.reset();
        return false;
      }
      const uint32_t page = static_cast<uint32_t>(page_value);
      state.page_snapshots.push_back(snapshot_page(page, pages[page]));
    }
    state.armed = true;
    return true;
  }

  void prepare_eviction_undo_unlocked(
      uint32_t target,
      MutationState& state) {
    state.reset();
    state.kind = MutationState::Kind::kEviction;
    state.old_clock = clock;
    state.old_eviction_count = eviction_count;
    state.page_snapshots.reserve(std::min(target, max_pages));
    state.armed = true;
  }

  void prepare_decode_undo_unlocked(MutationState& state) {
    state.reset();
    state.kind = MutationState::Kind::kDecode;
    state.old_dispatches = dispatches;
    state.old_dispatch_failures = dispatch_failures;
    state.old_native_dispatches = native_dispatches;
    state.old_native_failures = native_failures;
    state.old_query_copies = query_copies;
    state.old_output_copies = output_copies;
    state.old_attention_validation_bytes = attention_validation_bytes;
    state.armed = true;
  }

  bool resolve_request(
      uint64_t handle,
      uint32_t* slot,
      RequestMeta** request,
      std::string* error) {
    if (!check_running(error)) {
      return false;
    }
    const uint32_t index = handle_slot(handle);
    if (index >= requests.size() || handle_generation(handle) == 0) {
      if (error != nullptr) {
        *error = "invalid request handle";
      }
      return false;
    }
    RequestMeta& candidate = requests[index];
    if (!candidate.live || candidate.generation != handle_generation(handle)) {
      if (error != nullptr) {
        *error = "stale or released request handle";
      }
      return false;
    }
    if (slot != nullptr) {
      *slot = index;
    }
    if (request != nullptr) {
      *request = &candidate;
    }
    return true;
  }

  bool resolve_prefix(
      uint64_t handle,
      uint32_t* slot,
      PrefixMeta** prefix,
      std::string* error) {
    if (!check_running(error)) {
      return false;
    }
    const uint32_t index = handle_slot(handle);
    if (index >= prefixes.size() || handle_generation(handle) == 0) {
      if (error != nullptr) {
        *error = "invalid prefix handle";
      }
      return false;
    }
    PrefixMeta& candidate = prefixes[index];
    if (!candidate.live || candidate.generation != handle_generation(handle)) {
      if (error != nullptr) {
        *error = "stale or released prefix handle";
      }
      return false;
    }
    if (slot != nullptr) {
      *slot = index;
    }
    if (prefix != nullptr) {
      *prefix = &candidate;
    }
    return true;
  }

  bool resolve_page(uint32_t page, std::string* error) const {
    if (page >= pages.size() || !pages[page].allocated) {
      if (error != nullptr) {
        *error = "page is not resident";
      }
      return false;
    }
    return true;
  }

  void touch(uint32_t page) {
    pages[page].last_used = ++clock;
  }

  void sync_table_range(
      uint32_t request_slot,
      uint32_t first_block,
      uint32_t block_count) {
    if (block_count == 0 || page_table_buffer == nil) {
      return;
    }
    if (first_block > max_blocks_per_request ||
        block_count > max_blocks_per_request - first_block) {
      throw std::out_of_range("page-table metadata range is outside capacity");
    }
    const size_t offset =
        (static_cast<size_t>(request_slot) * max_blocks_per_request +
         first_block) *
        sizeof(int32_t);
    const size_t bytes = static_cast<size_t>(block_count) * sizeof(int32_t);
    std::memcpy(
        static_cast<uint8_t*>(page_table_buffer.contents) + offset,
        page_table.data() +
            static_cast<size_t>(request_slot) * max_blocks_per_request +
            first_block,
        bytes);
    ++metadata_copies;
    metadata_bytes += bytes;
  }

  void sync_table(uint32_t request_slot) {
    sync_table_range(request_slot, 0, max_blocks_per_request);
  }

  void sync_length_layer(uint32_t request_slot, uint32_t layer) {
    if (layer >= num_layers) {
      throw std::out_of_range("sequence-length metadata layer is out of range");
    }
    const size_t index = static_cast<size_t>(request_slot) * num_layers + layer;
    const auto& lengths = requests[request_slot].layer_lengths;
    const int32_t length =
        layer < lengths.size() ? static_cast<int32_t>(lengths[layer]) : 0;
    layer_sequence_lengths[index] = length;
    if (sequence_lengths_buffer != nil) {
      static_cast<int32_t*>(sequence_lengths_buffer.contents)[index] = length;
      ++metadata_copies;
      metadata_bytes += sizeof(int32_t);
    }
  }

  void sync_length(uint32_t request_slot) {
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      sync_length_layer(request_slot, layer);
    }
  }

  // Every KV data movement goes through this helper so instrumentation cannot
  // silently miss a future COW or host-ingress bulk copy.  This is deliberately
  // separate from kv_pool_copies: the latter is reserved for a forbidden
  // capacity-sized snapshot of the whole preallocated pool and must remain
  // zero on the decode path.
  void copy_kv_bytes(void* destination, const void* source, size_t bytes) {
    std::memcpy(destination, source, bytes);
    kv_copy_bytes += bytes;
  }

  uint32_t find_evictable_page_unlocked() const {
    uint32_t candidate = max_pages;
    uint64_t candidate_tick = std::numeric_limits<uint64_t>::max();
    for (uint32_t page = 0; page < max_pages; ++page) {
      const PageMeta& meta = pages[page];
      if (meta.allocated && meta.references == 0 &&
          meta.last_used <= candidate_tick) {
        candidate = page;
        candidate_tick = meta.last_used;
      }
    }
    return candidate;
  }

  uint32_t find_allocation_candidate_unlocked(
      const std::vector<uint32_t>& reserved) const {
    for (uint32_t page = 0; page < max_pages; ++page) {
      if (!pages[page].allocated &&
          std::find(reserved.begin(), reserved.end(), page) == reserved.end()) {
        return page;
      }
    }
    uint32_t candidate = max_pages;
    uint64_t candidate_tick = std::numeric_limits<uint64_t>::max();
    for (uint32_t page = 0; page < max_pages; ++page) {
      const PageMeta& meta = pages[page];
      if (meta.allocated && meta.references == 0 &&
          std::find(reserved.begin(), reserved.end(), page) == reserved.end() &&
          meta.last_used <= candidate_tick) {
        candidate = page;
        candidate_tick = meta.last_used;
      }
    }
    return candidate;
  }

  uint32_t evict_one_unlocked() {
    const uint32_t candidate = find_evictable_page_unlocked();
    if (candidate == max_pages) {
      return max_pages;
    }
    // Every live request/prefix page-table entry owns one reference.  A zero
    // reference therefore proves that no GPU-ready table can still point at
    // this page; the next owner publishes its replacement entry through
    // sync_table_range before decode can observe it.
    pages[candidate].allocated = false;
    pages[candidate].references = 0;
    pages[candidate].last_used = 0;
    pages[candidate].generation = next_generation(pages[candidate].generation);
    ++eviction_count;
    return candidate;
  }

  uint32_t allocate_physical_page_unlocked(std::string* error) {
    for (;;) {
      for (uint32_t page = 0; page < max_pages; ++page) {
        if (!pages[page].allocated) {
          PageMeta& meta = pages[page];
          meta.allocated = true;
          meta.references = 0;
          meta.generation = next_generation(meta.generation);
          touch(page);
          ++page_allocations;
          return page;
        }
      }
      if (evict_one_unlocked() == max_pages) {
        if (error != nullptr) {
          *error =
              "page capacity exhausted: every resident page is still referenced";
        }
        return max_pages;
      }
    }
  }

  bool can_reserve_physical_pages_unlocked(
      uint32_t needed,
      std::string* error) const {
    uint64_t available = 0;
    for (const PageMeta& page : pages) {
      if (!page.allocated || page.references == 0) {
        ++available;
      }
    }
    if (available < needed) {
      if (error != nullptr) {
        *error =
            "page capacity exhausted: every resident page is still referenced";
      }
      return false;
    }
    return true;
  }

  void increment_page(uint32_t page) {
    ++pages[page].references;
    touch(page);
  }

  void decrement_page(uint32_t page) {
    if (pages[page].references == 0) {
      return;
    }
    --pages[page].references;
    touch(page);
  }

  bool cow_page_unlocked(
      RequestMeta& request,
      uint32_t request_slot,
      uint32_t logical_block,
      std::string* error) {
    const int32_t old_value = request.pages[logical_block];
    if (old_value < 0 || !resolve_page(static_cast<uint32_t>(old_value), error)) {
      return false;
    }
    const uint32_t old_page = static_cast<uint32_t>(old_value);
    if (pages[old_page].references <= 1) {
      touch(old_page);
      return true;
    }
    const uint32_t new_page = allocate_physical_page_unlocked(error);
    if (new_page == max_pages) {
      return false;
    }
    const size_t source_offset =
        static_cast<size_t>(old_page) * static_cast<size_t>(page_elements);
    const size_t destination_offset =
        static_cast<size_t>(new_page) * static_cast<size_t>(page_elements);
    const size_t bytes = static_cast<size_t>(page_elements) * sizeof(uint16_t);
    const size_t value_plane =
        static_cast<size_t>(max_pages) * static_cast<size_t>(page_elements);
    for (LayerStorage& layer : layers) {
      copy_kv_bytes(
          layer.data() + destination_offset,
          layer.data() + source_offset,
          bytes);
      copy_kv_bytes(
          layer.data() + value_plane + destination_offset,
          layer.data() + value_plane + source_offset,
          bytes);
    }
    decrement_page(old_page);
    increment_page(new_page);
    request.pages[logical_block] = static_cast<int32_t>(new_page);
    page_table[static_cast<size_t>(request_slot) * max_blocks_per_request +
               logical_block] = static_cast<int32_t>(new_page);
    sync_table_range(request_slot, logical_block, 1);
    ++cow_events;
    return true;
  }

  bool ensure_request_page_unlocked(
      RequestMeta& request,
      uint32_t request_slot,
      uint32_t logical_block,
      std::string* error) {
    if (logical_block >= max_blocks_per_request) {
      if (error != nullptr) {
        *error = "logical block exceeds the request page-table capacity";
      }
      return false;
    }
    int32_t& page_entry = request.pages[logical_block];
    if (page_entry < 0) {
      const uint32_t page = allocate_physical_page_unlocked(error);
      if (page == max_pages) {
        return false;
      }
      page_entry = static_cast<int32_t>(page);
      increment_page(page);
      page_table[static_cast<size_t>(request_slot) * max_blocks_per_request +
                 logical_block] = page_entry;
      request.page_table_length =
          std::max(request.page_table_length, logical_block + 1);
      sync_table_range(request_slot, logical_block, 1);
      return true;
    }
    return resolve_page(static_cast<uint32_t>(page_entry), error);
  }

};

PageRuntime::PageRuntime(
    uint32_t num_layers,
    uint32_t num_attention_heads,
    uint32_t num_key_value_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_pages,
    uint32_t max_blocks_per_request,
    uint32_t max_requests)
    : impl_(std::make_unique<Impl>(
          num_layers,
          num_attention_heads,
          num_key_value_heads,
          head_dim,
          block_size,
          max_pages,
          max_blocks_per_request,
          max_requests)) {}

PageRuntime::~PageRuntime() = default;

void PageRuntime::commit(MutationToken* mutation) noexcept {
  if (mutation != nullptr && mutation->state_ != nullptr) {
    mutation->state_->reset();
  }
}

void PageRuntime::rollback(MutationToken* mutation) noexcept {
  if (mutation == nullptr || mutation->state_ == nullptr ||
      !mutation->state_->armed) {
    return;
  }
  MutationToken::State& state = *mutation->state_;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto restore_page_snapshots = [&] {
    for (const MutationPageSnapshot& snapshot : state.page_snapshots) {
      if (snapshot.page >= impl_->pages.size()) {
        continue;
      }
      auto& page = impl_->pages[snapshot.page];
      page.allocated = snapshot.allocated;
      page.generation = snapshot.generation;
      page.references = snapshot.references;
      page.last_used = snapshot.last_used;
    }
  };
  switch (state.kind) {
    case MutationToken::State::Kind::kRequest: {
      if (state.request_slot < impl_->requests.size()) {
        auto& request = impl_->requests[state.request_slot];
        if (request.live && request.generation == state.new_generation) {
          request.live = false;
          request.generation = state.prior_request_generation;
          request.id.clear();
          request.max_tokens = 0;
          request.sequence_length = 0;
          request.page_table_length = 0;
          request.layer_lengths.clear();
          request.layer_max_key.clear();
          request.layer_max_value.clear();
          request.pages.clear();
          const size_t table_base =
              static_cast<size_t>(state.request_slot) *
              impl_->max_blocks_per_request;
          if (state.prior_request_pages.size() ==
              impl_->max_blocks_per_request) {
            std::copy(
                state.prior_request_pages.begin(),
                state.prior_request_pages.end(),
                impl_->page_table.begin() + table_base);
            if (impl_->page_table_buffer != nil) {
              std::memcpy(
                  static_cast<uint8_t*>(impl_->page_table_buffer.contents) +
                      table_base * sizeof(int32_t),
                  state.prior_request_pages.data(),
                  state.prior_request_pages.size() * sizeof(int32_t));
            }
          }
          const size_t length_base =
              static_cast<size_t>(state.request_slot) * impl_->num_layers;
          if (state.prior_layer_lengths.size() == impl_->num_layers) {
            std::copy(
                state.prior_layer_lengths.begin(),
                state.prior_layer_lengths.end(),
                impl_->layer_sequence_lengths.begin() + length_base);
            if (impl_->sequence_lengths_buffer != nil) {
              std::memcpy(
                  static_cast<uint8_t*>(
                      impl_->sequence_lengths_buffer.contents) +
                      length_base * sizeof(int32_t),
                  state.prior_layer_lengths.data(),
                  state.prior_layer_lengths.size() * sizeof(int32_t));
            }
          }
          impl_->request_allocations = state.old_request_allocations;
          impl_->metadata_copies = state.old_metadata_copies;
          impl_->metadata_bytes = state.old_metadata_bytes;
        }
      }
      break;
    }
    case MutationToken::State::Kind::kPages: {
      if (state.request_slot < impl_->requests.size()) {
        auto& request = impl_->requests[state.request_slot];
        if (request.live && request.generation == state.prior_request_generation) {
          for (size_t index = 0; index < state.prior_request_pages.size();
               ++index) {
            const size_t block = state.first_block + index;
            if (block < request.pages.size()) {
              request.pages[block] = state.prior_request_pages[index];
              impl_->page_table[
                  static_cast<size_t>(state.request_slot) *
                      impl_->max_blocks_per_request +
                  block] = state.prior_request_pages[index];
            }
          }
          request.page_table_length = state.prior_page_table_length;
          if (impl_->page_table_buffer != nil &&
              !state.prior_request_pages.empty()) {
            const size_t offset =
                (static_cast<size_t>(state.request_slot) *
                     impl_->max_blocks_per_request +
                 state.first_block) *
                sizeof(int32_t);
            std::memcpy(
                static_cast<uint8_t*>(impl_->page_table_buffer.contents) +
                    offset,
                state.prior_request_pages.data(),
                state.prior_request_pages.size() * sizeof(int32_t));
          }
        }
      }
      restore_page_snapshots();
      impl_->clock = state.old_clock;
      impl_->page_allocations = state.old_page_allocations;
      impl_->eviction_count = state.old_eviction_count;
      impl_->metadata_copies = state.old_metadata_copies;
      impl_->metadata_bytes = state.old_metadata_bytes;
      break;
    }
    case MutationToken::State::Kind::kPrefix: {
      std::unique_lock<std::mutex> slot_lock;
      if (state.has_request_sequence &&
          state.request_slot < impl_->requests.size()) {
        slot_lock = std::unique_lock<std::mutex>(
            impl_->slot_mutexes[state.request_slot]);
      }
      if (state.prefix_was_new) {
        if (state.prefix_slot + 1 == impl_->prefixes.size()) {
          impl_->prefixes.pop_back();
        }
      } else if (state.prefix_slot < impl_->prefixes.size()) {
        auto& prefix = impl_->prefixes[state.prefix_slot];
        prefix.live = false;
        prefix.generation = state.prior_prefix_generation;
        prefix.token_count = 0;
        prefix.pages.clear();
        prefix.layer_max_key.clear();
        prefix.layer_max_value.clear();
      }
      if (state.has_request_sequence &&
          state.request_slot < impl_->requests.size()) {
        impl_->requests[state.request_slot].sequence_length =
            state.prior_request_sequence_length;
      }
      restore_page_snapshots();
      impl_->clock = state.old_clock;
      impl_->prefix_allocations = state.old_prefix_allocations;
      break;
    }
    case MutationToken::State::Kind::kEviction:
      restore_page_snapshots();
      impl_->clock = state.old_clock;
      impl_->eviction_count = state.old_eviction_count;
      break;
    case MutationToken::State::Kind::kDecode:
      impl_->dispatches = state.old_dispatches;
      impl_->dispatch_failures = state.old_dispatch_failures;
      impl_->native_dispatches = state.old_native_dispatches;
      impl_->native_failures = state.old_native_failures;
      impl_->query_copies = state.old_query_copies;
      impl_->output_copies = state.old_output_copies;
      impl_->attention_validation_bytes = state.old_attention_validation_bytes;
      break;
    case MutationToken::State::Kind::kNone:
      break;
  }
  state.reset();
}

bool PageRuntime::allocate_request(
    const std::string& request_id,
    uint32_t max_tokens,
    uint64_t* handle,
    std::string* error) {
  return allocate_request(request_id, max_tokens, handle, nullptr, error);
}

bool PageRuntime::allocate_request(
    const std::string& request_id,
    uint32_t max_tokens,
    uint64_t* handle,
    MutationToken* mutation,
    std::string* error) {
  if (handle == nullptr) {
    if (error != nullptr) {
      *error = "request handle output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return false;
  }
  if (request_id.empty()) {
    if (error != nullptr) {
      *error = "request_id must be a non-empty string";
    }
    return false;
  }
  for (const auto& candidate : impl_->requests) {
    if (candidate.live && candidate.id == request_id) {
      if (error != nullptr) {
        *error = "request_id is already active";
      }
      return false;
    }
  }
  const uint64_t capacity = static_cast<uint64_t>(impl_->max_blocks_per_request) *
      impl_->block_size;
  if (max_tokens > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    if (error != nullptr) {
      *error = "max_tokens must not exceed INT32_MAX";
    }
    return false;
  }
  if (max_tokens == 0 || static_cast<uint64_t>(max_tokens) > capacity) {
    if (error != nullptr) {
      *error = "max_tokens must be in [1, max_blocks_per_request * block_size]";
    }
    return false;
  }
  for (uint32_t slot = 0; slot < impl_->max_requests; ++slot) {
    auto& request = impl_->requests[slot];
    if (request.live) {
      continue;
    }
    if (mutation != nullptr) {
      if (mutation->state_ == nullptr || mutation->state_->armed) {
        if (error != nullptr) {
          *error = "mutation token is already armed or invalid";
        }
        return false;
      }
      if (!impl_->prepare_request_undo_unlocked(
              slot, *mutation->state_, error)) {
        return false;
      }
    }
    // Build all potentially-allocating state before publishing ``live``.
    // A bad_alloc here must leave this slot indistinguishable from its prior
    // released state so a later request cannot observe a half-initialized
    // generation or page table.
    Impl::RequestMeta replacement;
    replacement.live = true;
    replacement.generation = next_generation(request.generation);
    replacement.id = request_id;
    replacement.max_tokens = max_tokens;
    replacement.sequence_length = 0;
    replacement.page_table_length = 0;
    replacement.layer_lengths.assign(impl_->num_layers, 0);
    replacement.layer_max_key.assign(impl_->num_layers, 0.0f);
    replacement.layer_max_value.assign(impl_->num_layers, 0.0f);
    replacement.pages.assign(impl_->max_blocks_per_request, -1);
    request = std::move(replacement);
    std::fill(
        impl_->page_table.begin() +
            static_cast<size_t>(slot) * impl_->max_blocks_per_request,
        impl_->page_table.begin() +
            static_cast<size_t>(slot + 1) * impl_->max_blocks_per_request,
        -1);
    impl_->sync_table(slot);
    impl_->sync_length(slot);
    *handle = make_handle(request.generation, slot);
    ++impl_->request_allocations;
    return true;
  }
  if (error != nullptr) {
    *error = "request capacity exhausted";
  }
  return false;
}

bool PageRuntime::allocate_pages(
    uint64_t request_handle,
    uint32_t count,
    std::vector<uint64_t>* handles,
    std::string* error) {
  return allocate_pages(request_handle, count, handles, nullptr, error);
}

bool PageRuntime::allocate_pages(
    uint64_t request_handle,
    uint32_t count,
    std::vector<uint64_t>* handles,
    MutationToken* mutation,
    std::string* error) {
  if (handles == nullptr) {
    if (error != nullptr) {
      *error = "page handle output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t request_slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!impl_->resolve_request(request_handle, &request_slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[request_slot]);
  if (count == 0) {
    handles->clear();
    return true;
  }
  uint32_t first_empty = impl_->max_blocks_per_request;
  for (uint32_t block = 0; block < impl_->max_blocks_per_request; ++block) {
    if (request->pages[block] < 0) {
      first_empty = block;
      break;
    }
  }
  const uint32_t request_page_capacity =
      (request->max_tokens + impl_->block_size - 1) / impl_->block_size;
  if (first_empty == impl_->max_blocks_per_request ||
      first_empty >= request_page_capacity ||
      count > request_page_capacity - first_empty) {
    if (error != nullptr) {
      *error =
          "requested pages exceed the request max_tokens capacity (" +
          std::to_string(request_page_capacity) + " pages)";
    }
    return false;
  }
  // Allocate transactionally: if capacity cannot satisfy the whole request,
  // leave both metadata and refcounts unchanged.
  uint32_t unallocated = 0;
  for (const auto& page : impl_->pages) {
    if (!page.allocated) {
      ++unallocated;
    }
  }
  if (unallocated < count) {
    // Count evictable pages separately so referenced pages can never be
    // accidentally reclaimed to satisfy an allocation.
    uint32_t evictable = 0;
    for (const auto& page : impl_->pages) {
      if (page.allocated && page.references == 0) {
        ++evictable;
      }
    }
    if (unallocated + evictable < count) {
      if (error != nullptr) {
        *error =
            "page capacity exhausted: every resident page is still referenced";
      }
      return false;
    }
  }
  handles->clear();
  handles->reserve(count);
  if (mutation != nullptr) {
    if (mutation->state_ == nullptr || mutation->state_->armed) {
      if (error != nullptr) {
        *error = "mutation token is already armed or invalid";
      }
      return false;
    }
    if (!impl_->prepare_pages_undo_unlocked(
            request_slot, first_empty, count, *mutation->state_, error)) {
      return false;
    }
  }
  for (uint32_t offset = 0; offset < count; ++offset) {
    const uint32_t block = first_empty + offset;
    const uint32_t page = impl_->allocate_physical_page_unlocked(error);
    if (page == impl_->max_pages) {
      handles->clear();
      return false;
    }
    request->pages[block] = static_cast<int32_t>(page);
    impl_->increment_page(page);
    impl_->page_table[static_cast<size_t>(request_slot) *
                          impl_->max_blocks_per_request +
                      block] = static_cast<int32_t>(page);
    request->page_table_length =
        std::max(request->page_table_length, block + 1);
    handles->push_back(make_handle(
        impl_->pages[page].generation, page));
  }
  impl_->sync_table_range(request_slot, first_empty, count);
  return true;
}

bool PageRuntime::append_kv(
    uint64_t request_handle,
    uint32_t layer,
    const uint16_t* keys,
    const uint16_t* values,
    uint32_t tokens,
    std::string* error) {
  if (keys == nullptr || values == nullptr) {
    if (error != nullptr) {
      *error = "keys and values must not be null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t request_slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!impl_->resolve_request(request_handle, &request_slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[request_slot]);
  if (layer >= impl_->num_layers) {
    if (error != nullptr) {
      *error = "layer index is outside the configured layer count";
    }
    return false;
  }
  const uint64_t elements = checked_product(
      {tokens, impl_->num_key_value_heads, impl_->head_dim}, "KV append size");
  if (elements > std::numeric_limits<size_t>::max()) {
    if (error != nullptr) {
      *error = "KV append exceeds native size limits";
    }
    return false;
  }
  const uint32_t current = request->layer_lengths[layer];
  if (tokens == 0) {
    return true;
  }
  const size_t append_elements = static_cast<size_t>(elements);
  if (!all_finite_bf16(keys, append_elements) ||
      !all_finite_bf16(values, append_elements)) {
    if (error != nullptr) {
      *error = "keys and values must contain only finite BF16 values";
    }
    return false;
  }
  const float append_max_key = max_abs_bf16(keys, append_elements);
  const float append_max_value = max_abs_bf16(values, append_elements);
  if (static_cast<uint64_t>(current) + tokens > request->max_tokens) {
    if (error != nullptr) {
      *error = "KV append exceeds the request max_tokens limit";
    }
    return false;
  }
  const uint32_t end = current + tokens;
  const uint32_t first_block = current / impl_->block_size;
  const uint32_t last_block = (end - 1) / impl_->block_size;
  uint32_t pages_needed = 0;
  for (uint32_t block = first_block; block <= last_block; ++block) {
    const int32_t page = request->pages[block];
    if (page < 0) {
      ++pages_needed;
    } else if (!impl_->resolve_page(static_cast<uint32_t>(page), error)) {
      return false;
    } else if (impl_->pages[static_cast<uint32_t>(page)].references > 1) {
      ++pages_needed;
    }
  }
  // No page table, refcount, or LRU mutation occurs before this preflight.
  // Therefore an OOM during a multi-block append cannot leave a partial COW
  // or an evicted page behind.
  if (!impl_->can_reserve_physical_pages_unlocked(pages_needed, error)) {
    return false;
  }
  for (uint32_t block = first_block; block <= last_block; ++block) {
    if (!impl_->ensure_request_page_unlocked(*request, request_slot, block, error)) {
      return false;
    }
    // A shared partial prefix page is copied before the first write.  Full
    // pages at an exact block boundary remain read-only and shared.
    if (!impl_->cow_page_unlocked(*request, request_slot, block, error)) {
      return false;
    }
  }

  const size_t page_stride = static_cast<size_t>(impl_->page_elements);
  uint32_t cursor = 0;
  while (cursor < tokens) {
    const uint32_t absolute = current + cursor;
    const uint32_t logical_block = absolute / impl_->block_size;
    const uint32_t block_offset = absolute % impl_->block_size;
    const uint32_t copy_tokens = std::min(
        tokens - cursor, impl_->block_size - block_offset);
    const int32_t page_value = request->pages[logical_block];
    if (page_value < 0 ||
        !impl_->resolve_page(static_cast<uint32_t>(page_value), error)) {
      return false;
    }
    const uint32_t page = static_cast<uint32_t>(page_value);
    impl_->touch(page);
    const size_t destination =
        static_cast<size_t>(page) * page_stride +
        static_cast<size_t>(block_offset) * impl_->num_key_value_heads *
            impl_->head_dim;
    const size_t source =
        static_cast<size_t>(cursor) * impl_->num_key_value_heads *
        impl_->head_dim;
    const size_t bytes = static_cast<size_t>(copy_tokens) *
        impl_->num_key_value_heads * impl_->head_dim * sizeof(uint16_t);
    // K and V occupy separate preallocated planes in each layer buffer.
    impl_->copy_kv_bytes(
        impl_->layers[layer].data() + destination, keys + source, bytes);
    const size_t value_plane = static_cast<size_t>(impl_->max_pages) * page_stride;
    impl_->copy_kv_bytes(
        impl_->layers[layer].data() + value_plane + destination,
        values + source,
        bytes);
    cursor += copy_tokens;
  }
  request->layer_lengths[layer] = end;
  request->layer_max_key[layer] =
      std::max(request->layer_max_key[layer], append_max_key);
  request->layer_max_value[layer] =
      std::max(request->layer_max_value[layer], append_max_value);
  request->sequence_length = *std::max_element(
      request->layer_lengths.begin(), request->layer_lengths.end());
  impl_->append_tokens += tokens;
  impl_->sync_length_layer(request_slot, layer);
  return true;
}

bool PageRuntime::create_prefix(
    uint64_t request_handle,
    uint64_t* prefix_handle,
    std::string* error) {
  return create_prefix(request_handle, prefix_handle, nullptr, error);
}

bool PageRuntime::create_prefix(
    uint64_t request_handle,
    uint64_t* prefix_handle,
    MutationToken* mutation,
    std::string* error) {
  if (prefix_handle == nullptr) {
    if (error != nullptr) {
      *error = "prefix handle output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t request_slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!impl_->resolve_request(request_handle, &request_slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[request_slot]);
  if (request->layer_lengths.empty() || request->layer_lengths[0] == 0 ||
      std::any_of(
          request->layer_lengths.begin() + 1,
          request->layer_lengths.end(),
          [&](uint32_t length) { return length != request->layer_lengths[0]; })) {
    if (error != nullptr) {
      *error =
          "prefix creation requires all layers to have equal, non-zero lengths";
    }
    return false;
  }
  const uint32_t token_count = request->layer_lengths[0];
  if (request->layer_max_key.size() != impl_->num_layers ||
      request->layer_max_value.size() != impl_->num_layers) {
    if (error != nullptr) {
      *error = "request attention range metadata is not initialized";
    }
    return false;
  }
  const uint32_t blocks =
      (token_count + impl_->block_size - 1) / impl_->block_size;
  if (blocks > request->page_table_length ||
      blocks > request->pages.size()) {
    if (error != nullptr) {
      *error = "request page table is shorter than the prefix sequence";
    }
    return false;
  }
  uint32_t prefix_slot = 0;
  for (; prefix_slot < impl_->prefixes.size(); ++prefix_slot) {
    if (!impl_->prefixes[prefix_slot].live) {
      break;
    }
  }
  const uint32_t prior_generation =
      prefix_slot < impl_->prefixes.size()
          ? impl_->prefixes[prefix_slot].generation
          : 0;

  // Prepare every vector before publishing the prefix slot.  This keeps a
  // bad_alloc from leaving a live prefix with only part of its metadata.
  Impl::PrefixMeta replacement;
  replacement.live = true;
  replacement.generation = next_generation(prior_generation);
  replacement.token_count = token_count;
  replacement.layer_max_key = request->layer_max_key;
  replacement.layer_max_value = request->layer_max_value;
  replacement.pages.assign(
      request->pages.begin(), request->pages.begin() + blocks);
  for (int32_t page : replacement.pages) {
    if (page < 0 || !impl_->resolve_page(static_cast<uint32_t>(page), error)) {
      return false;
    }
  }

  if (mutation != nullptr) {
    if (mutation->state_ == nullptr || mutation->state_->armed) {
      if (error != nullptr) {
        *error = "mutation token is already armed or invalid";
      }
      return false;
    }
    if (!impl_->prepare_prefix_undo_unlocked(
            prefix_slot,
            prefix_slot == impl_->prefixes.size(),
            prior_generation,
            replacement.generation,
            true,
            request_slot,
            request->sequence_length,
            replacement.pages,
            *mutation->state_)) {
      if (error != nullptr) {
        *error = "could not prepare prefix rollback state";
      }
      return false;
    }
  }

  if (prefix_slot == impl_->prefixes.size()) {
    impl_->prefixes.emplace_back(std::move(replacement));
  } else {
    auto& prefix = impl_->prefixes[prefix_slot];
    prefix.live = replacement.live;
    prefix.generation = replacement.generation;
    prefix.token_count = replacement.token_count;
    prefix.layer_max_key.swap(replacement.layer_max_key);
    prefix.layer_max_value.swap(replacement.layer_max_value);
    prefix.pages.swap(replacement.pages);
  }
  const auto& prefix = impl_->prefixes[prefix_slot];
  for (int32_t page : prefix.pages) {
    impl_->increment_page(static_cast<uint32_t>(page));
  }
  request->sequence_length = token_count;
  *prefix_handle = make_handle(prefix.generation, prefix_slot);
  ++impl_->prefix_allocations;
  return true;
}

bool PageRuntime::fork_prefix(
    uint64_t prefix_handle,
    uint64_t* forked_handle,
    std::string* error) {
  return fork_prefix(prefix_handle, forked_handle, nullptr, error);
}

bool PageRuntime::fork_prefix(
    uint64_t prefix_handle,
    uint64_t* forked_handle,
    MutationToken* mutation,
    std::string* error) {
  if (forked_handle == nullptr) {
    if (error != nullptr) {
      *error = "forked prefix output is null";
    }
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Impl::PrefixMeta* source = nullptr;
  if (!impl_->resolve_prefix(prefix_handle, nullptr, &source, error)) {
    return false;
  }
  if (source->layer_max_key.size() != impl_->num_layers ||
      source->layer_max_value.size() != impl_->num_layers) {
    if (error != nullptr) {
      *error = "prefix attention range metadata is not initialized";
    }
    return false;
  }
  // ``prefixes`` is a growable vector.  Copy the source payload before any
  // emplace_back so a reallocation cannot invalidate the source pointer.
  const uint32_t source_token_count = source->token_count;
  std::vector<float> source_max_key = source->layer_max_key;
  std::vector<float> source_max_value = source->layer_max_value;
  std::vector<int32_t> source_pages = source->pages;
  uint32_t slot = 0;
  for (; slot < impl_->prefixes.size(); ++slot) {
    if (!impl_->prefixes[slot].live) {
      break;
    }
  }
  const uint32_t prior_generation =
      slot < impl_->prefixes.size() ? impl_->prefixes[slot].generation : 0;
  Impl::PrefixMeta replacement;
  replacement.live = true;
  replacement.generation = next_generation(prior_generation);
  replacement.token_count = source_token_count;
  replacement.layer_max_key = std::move(source_max_key);
  replacement.layer_max_value = std::move(source_max_value);
  replacement.pages = std::move(source_pages);
  for (int32_t page : replacement.pages) {
    if (page < 0 || !impl_->resolve_page(static_cast<uint32_t>(page), error)) {
      return false;
    }
  }

  if (mutation != nullptr) {
    if (mutation->state_ == nullptr || mutation->state_->armed) {
      if (error != nullptr) {
        *error = "mutation token is already armed or invalid";
      }
      return false;
    }
    if (!impl_->prepare_prefix_undo_unlocked(
            slot,
            slot == impl_->prefixes.size(),
            prior_generation,
            replacement.generation,
            false,
            0,
            0,
            replacement.pages,
            *mutation->state_)) {
      if (error != nullptr) {
        *error = "could not prepare prefix rollback state";
      }
      return false;
    }
  }

  if (slot == impl_->prefixes.size()) {
    impl_->prefixes.emplace_back(std::move(replacement));
  } else {
    auto& destination = impl_->prefixes[slot];
    destination.live = replacement.live;
    destination.generation = replacement.generation;
    destination.token_count = replacement.token_count;
    destination.layer_max_key.swap(replacement.layer_max_key);
    destination.layer_max_value.swap(replacement.layer_max_value);
    destination.pages.swap(replacement.pages);
  }
  const auto& destination = impl_->prefixes[slot];
  for (int32_t page : destination.pages) {
    impl_->increment_page(static_cast<uint32_t>(page));
  }
  *forked_handle = make_handle(destination.generation, slot);
  ++impl_->prefix_allocations;
  return true;
}

bool PageRuntime::attach_prefix(
    uint64_t request_handle,
    uint64_t prefix_handle,
    std::string* error) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t request_slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!impl_->resolve_request(request_handle, &request_slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[request_slot]);
  Impl::PrefixMeta* prefix = nullptr;
  if (!impl_->resolve_prefix(prefix_handle, nullptr, &prefix, error)) {
    return false;
  }
  if (request->sequence_length != 0 ||
      std::any_of(
          request->pages.begin(), request->pages.end(),
          [](int32_t page) { return page >= 0; })) {
    if (error != nullptr) {
      *error = "a prefix can only attach to an empty request";
    }
    return false;
  }
  if (prefix->token_count > request->max_tokens) {
    if (error != nullptr) {
      *error = "prefix token count exceeds the request max_tokens limit";
    }
    return false;
  }
  if (prefix->pages.size() > request->pages.size()) {
    if (error != nullptr) {
      *error = "prefix page count exceeds the request page-table capacity";
    }
    return false;
  }
  if (prefix->layer_max_key.size() != impl_->num_layers ||
      prefix->layer_max_value.size() != impl_->num_layers) {
    if (error != nullptr) {
      *error = "prefix attention range metadata is not initialized";
    }
    return false;
  }
  if (request->layer_lengths.size() != impl_->num_layers ||
      request->layer_max_key.size() != impl_->num_layers ||
      request->layer_max_value.size() != impl_->num_layers) {
    if (error != nullptr) {
      *error = "request attention range metadata is not initialized";
    }
    return false;
  }
  for (size_t index = 0; index < prefix->pages.size(); ++index) {
    const int32_t page = prefix->pages[index];
    if (page < 0 || !impl_->resolve_page(static_cast<uint32_t>(page), error)) {
      return false;
    }
  }
  for (size_t index = 0; index < prefix->pages.size(); ++index) {
    const int32_t page = prefix->pages[index];
    request->pages[index] = page;
    impl_->increment_page(static_cast<uint32_t>(page));
    impl_->page_table[static_cast<size_t>(request_slot) *
                          impl_->max_blocks_per_request +
                      index] = page;
  }
  request->sequence_length = prefix->token_count;
  request->page_table_length = static_cast<uint32_t>(prefix->pages.size());
  std::fill(
      request->layer_lengths.begin(), request->layer_lengths.end(),
      request->sequence_length);
  std::copy(
      prefix->layer_max_key.begin(),
      prefix->layer_max_key.end(),
      request->layer_max_key.begin());
  std::copy(
      prefix->layer_max_value.begin(),
      prefix->layer_max_value.end(),
      request->layer_max_value.begin());
  impl_->sync_table_range(
      request_slot, 0, static_cast<uint32_t>(prefix->pages.size()));
  impl_->sync_length(request_slot);
  return true;
}

bool PageRuntime::release_prefix(uint64_t prefix_handle, std::string* error) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return false;
  }
  const uint32_t prefix_slot = handle_slot(prefix_handle);
  const uint32_t prefix_generation = handle_generation(prefix_handle);
  if (prefix_slot < impl_->prefixes.size() && prefix_generation != 0) {
    const auto& prior = impl_->prefixes[prefix_slot];
    if (!prior.live && prior.generation == prefix_generation) {
      // Releasing the exact same handle twice is deliberately idempotent.
      // A reused slot has a different generation and still fails below.
      return true;
    }
  }
  Impl::PrefixMeta* prefix = nullptr;
  if (!impl_->resolve_prefix(prefix_handle, nullptr, &prefix, error)) {
    return false;
  }
  for (int32_t page : prefix->pages) {
    if (page >= 0 && static_cast<uint32_t>(page) < impl_->pages.size()) {
      impl_->decrement_page(static_cast<uint32_t>(page));
    }
  }
  prefix->live = false;
  prefix->pages.clear();
  prefix->layer_max_key.clear();
  prefix->layer_max_value.clear();
  prefix->token_count = 0;
  return true;
}

bool PageRuntime::release_request(
    uint64_t request_handle,
    bool cancelled,
    std::string* error) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return false;
  }
  const uint32_t prior_slot = handle_slot(request_handle);
  const uint32_t request_generation = handle_generation(request_handle);
  if (prior_slot < impl_->requests.size() && request_generation != 0) {
    const auto& prior = impl_->requests[prior_slot];
    if (!prior.live && prior.generation == request_generation) {
      // Releasing/cancelling an already released handle is a safe no-op.  A
      // stale handle from a later slot reuse has a different generation and
      // is rejected by resolve_request below.
      return true;
    }
  }
  uint32_t request_slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!impl_->resolve_request(request_handle, &request_slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[request_slot]);
  for (int32_t page : request->pages) {
    if (page >= 0 && static_cast<uint32_t>(page) < impl_->pages.size()) {
      impl_->decrement_page(static_cast<uint32_t>(page));
    }
  }
  request->live = false;
  request->id.clear();
    request->max_tokens = 0;
    request->sequence_length = 0;
    request->page_table_length = 0;
  request->layer_lengths.clear();
  request->layer_max_key.clear();
  request->layer_max_value.clear();
  request->pages.clear();
  std::fill(
      impl_->page_table.begin() +
          static_cast<size_t>(request_slot) * impl_->max_blocks_per_request,
      impl_->page_table.begin() +
          static_cast<size_t>(request_slot + 1) * impl_->max_blocks_per_request,
      -1);
  impl_->sync_table(request_slot);
  impl_->sync_length(request_slot);
  if (cancelled) {
    ++impl_->cancellation_count;
  } else {
    ++impl_->release_count;
  }
  return true;
}

bool PageRuntime::snapshot(
    uint64_t prefix_handle,
    const std::string& destination,
    std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return false;
  }
  (void)prefix_handle;
  (void)destination;
  ++impl_->snapshot_failures;
  if (error != nullptr) {
    *error =
        "Metal Context page-runtime snapshots are deferred to the persistence "
        "package; no page bytes are written by this backend";
  }
  return false;
}

bool PageRuntime::restore(
    const std::string& source,
    uint64_t* prefix_handle,
    std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return false;
  }
  (void)source;
  if (prefix_handle != nullptr) {
    *prefix_handle = 0;
  }
  ++impl_->restore_failures;
  if (error != nullptr) {
    *error =
        "Metal Context page-runtime restore is deferred to the persistence "
        "package; arbitrary Python objects are never deserialized";
  }
  return false;
}

uint32_t PageRuntime::evict(
    uint32_t target_pages,
    bool has_target,
    std::string* error) {
  return evict(target_pages, has_target, nullptr, error);
}

uint32_t PageRuntime::evict(
    uint32_t target_pages,
    bool has_target,
    MutationToken* mutation,
    std::string* error) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->check_running(error)) {
    return 0;
  }
  const uint32_t target = has_target ? target_pages : impl_->max_pages;
  if (mutation != nullptr) {
    if (mutation->state_ == nullptr || mutation->state_->armed) {
      if (error != nullptr) {
        *error = "mutation token is already armed or invalid";
      }
      return 0;
    }
    impl_->prepare_eviction_undo_unlocked(target, *mutation->state_);
  }
  uint32_t evicted = 0;
  while (evicted < target) {
    const uint32_t candidate = impl_->find_evictable_page_unlocked();
    if (candidate == impl_->max_pages) {
      break;
    }
    if (mutation != nullptr) {
      mutation->state_->page_snapshots.push_back(
          Impl::snapshot_page(candidate, impl_->pages[candidate]));
    }
    if (impl_->evict_one_unlocked() == impl_->max_pages) {
      break;
    }
    ++evicted;
  }
  return evicted;
}

bool PageRuntime::page_table(
    uint64_t request_handle,
    uint32_t layer,
    std::vector<int32_t>* table,
    std::string* error) const {
  if (table == nullptr) {
    if (error != nullptr) {
      *error = "page-table output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!const_cast<Impl*>(impl_.get())->resolve_request(
          request_handle, &slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[slot]);
  if (layer >= impl_->num_layers) {
    if (error != nullptr) {
      *error = "layer index is outside the configured layer count";
    }
    return false;
  }
  const uint32_t length = request->layer_lengths[layer];
  const uint32_t blocks =
      (length + impl_->block_size - 1) / impl_->block_size;
  if (blocks > request->page_table_length ||
      blocks > request->pages.size()) {
    if (error != nullptr) {
      *error = "request page table is shorter than its layer sequence";
    }
    return false;
  }
  table->assign(
      impl_->page_table.begin() +
          static_cast<size_t>(slot) * impl_->max_blocks_per_request,
      impl_->page_table.begin() +
          static_cast<size_t>(slot) * impl_->max_blocks_per_request + blocks);
  return true;
}

bool PageRuntime::request_pages(
    uint64_t request_handle,
    std::vector<uint64_t>* pages,
    std::string* error) const {
  if (pages == nullptr) {
    if (error != nullptr) {
      *error = "request-pages output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!const_cast<Impl*>(impl_.get())->resolve_request(
          request_handle, &slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[slot]);
  pages->clear();
  for (int32_t page : request->pages) {
    if (page < 0) {
      break;
    }
    if (!impl_->resolve_page(static_cast<uint32_t>(page), error)) {
      return false;
    }
    pages->push_back(make_handle(
        impl_->pages[static_cast<uint32_t>(page)].generation,
        static_cast<uint32_t>(page)));
  }
  return true;
}

bool PageRuntime::sequence_length(
    uint64_t request_handle,
    uint32_t* length,
    std::string* error) const {
  if (length == nullptr) {
    if (error != nullptr) {
      *error = "sequence-length output is null";
    }
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint32_t slot = 0;
  Impl::RequestMeta* request = nullptr;
  if (!const_cast<Impl*>(impl_.get())->resolve_request(
          request_handle, &slot, &request, error)) {
    return false;
  }
  std::unique_lock<std::mutex> slot_lock(impl_->slot_mutexes[slot]);
  *length = request->sequence_length;
  return true;
}

bool PageRuntime::paged_decode(
    uint64_t request_handle,
    uint32_t layer,
    const uint16_t* query,
    size_t query_elements,
    float scale,
    std::vector<float>* output,
    std::string* error) {
  return paged_decode(
      request_handle,
      layer,
      query,
      query_elements,
      scale,
      output,
      nullptr,
      error);
}

bool PageRuntime::paged_decode(
    uint64_t request_handle,
    uint32_t layer,
    const uint16_t* query,
    size_t query_elements,
    float scale,
    std::vector<float>* output,
    MutationToken* mutation,
    std::string* error) {
  if (query == nullptr || output == nullptr || error == nullptr) {
    if (error != nullptr) {
      *error = "paged decode query/output must not be null";
    }
    return false;
  }
  uint32_t slot = 0;
  uint32_t length = 0;
  float max_key = 0.0f;
  float max_value = 0.0f;
  Impl::RequestMeta* request = nullptr;
  std::unique_lock<std::mutex> slot_lock;
  std::string path;
  Impl::DispatchResult dispatch_result;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (!const_cast<Impl*>(impl_.get())->resolve_request(
            request_handle, &slot, &request, error)) {
      return false;
    }
    if (layer >= impl_->num_layers) {
      *error = "layer index is outside the configured layer count";
      return false;
    }
    // A slot owns its query/output scratch ranges and its live page table.
    // Hold that slot lock across validation and dispatch so append/COW/release
    // cannot mutate the shared KV pages while the GPU is reading them.  The
    // runtime mutex is released before the slot lock is used; the dispatch
    // lease drops the slot lock before it records any result under the runtime
    // mutex, preserving the documented Impl -> slot lock order.
    slot_lock = std::unique_lock<std::mutex>(impl_->slot_mutexes[slot]);
    if (request->layer_lengths.size() != impl_->num_layers ||
        request->layer_max_key.size() != impl_->num_layers ||
        request->layer_max_value.size() != impl_->num_layers) {
      *error = "request layer metadata is not initialized";
      return false;
    }
    if (query_elements != impl_->query_elements) {
      *error = "query element count does not match the runtime geometry";
      return false;
    }
    length = request->layer_lengths[layer];
    max_key = request->layer_max_key[layer];
    max_value = request->layer_max_value[layer];
    const uint32_t blocks =
        (length + impl_->block_size - 1) / impl_->block_size;
    if (blocks > request->page_table_length ||
        blocks > request->pages.size()) {
      *error = "request page table is shorter than its sequence length";
      return false;
    }
    // Page-table and sequence-length buffers are kept GPU-ready by every
    // lifecycle mutation.  Do not walk or touch the logical page chain here:
    // steady-state decode CPU work is O(query) plus constant metadata.
    path = impl_->library_path;
    // Register the in-flight operation before releasing the runtime mutex.
    // Shutdown marks the runtime as stopping and waits for this lease, so it
    // cannot release the shared kernel while the preallocated buffers are in
    // use.
    if (mutation != nullptr) {
      if (mutation->state_ == nullptr || mutation->state_->armed) {
        *error = "mutation token is already armed or invalid";
        return false;
      }
      impl_->prepare_decode_undo_unlocked(*mutation->state_);
    }
    ++impl_->active_dispatches;
  }
  struct DispatchLease {
    Impl* impl;
    std::unique_lock<std::mutex>* slot_lock;
    Impl::DispatchResult* result;
    ~DispatchLease() noexcept {
      // Mutators take the runtime mutex before a slot mutex.  Release the
      // slot first so they cannot wait on this lease while finish_dispatch()
      // waits on the runtime mutex.
      if (slot_lock != nullptr && slot_lock->owns_lock()) {
        slot_lock->unlock();
      }
      if (impl != nullptr) {
        impl->finish_dispatch(*result);
      }
    }
  } dispatch_lease{impl_.get(), &slot_lock, &dispatch_result};
  dispatch_result.validation_bytes = query_elements * sizeof(uint16_t);
  dispatch_result.validation_failed = true;
  if (!validate_attention_envelope(
          query,
          query_elements,
          length,
          max_key,
          max_value,
          impl_->head_dim,
          scale,
          error)) {
    return false;
  }
  dispatch_result.validation_failed = false;
  dispatch_result.dispatch_attempted = true;
  if (path.empty()) {
    path = metallib_path();
  }
  dispatch_result.dispatched = dispatch_kernel(
      impl_->command_queue,
      impl_->query_scratch_buffer,
      static_cast<NSUInteger>(
          static_cast<uint64_t>(slot) * impl_->query_bytes_per_slot),
      impl_->layers[layer].buffer,
      0,
      static_cast<NSUInteger>(checked_product(
          {impl_->max_pages, impl_->page_elements, sizeof(uint16_t)},
          "value buffer offset")),
      impl_->page_table_buffer,
      static_cast<NSUInteger>(checked_product(
          {slot, impl_->max_blocks_per_request, sizeof(int32_t)},
          "page-table buffer offset")),
      impl_->sequence_lengths_buffer,
      static_cast<NSUInteger>(page_runtime_sequence_buffer_offset(
          slot, impl_->num_layers, layer)),
      impl_->output_scratch_buffer,
      static_cast<NSUInteger>(
          static_cast<uint64_t>(slot) * impl_->output_bytes_per_slot),
      query,
      query_elements,
      1,
      impl_->num_attention_heads,
      impl_->num_key_value_heads,
      impl_->head_dim,
      impl_->block_size,
      impl_->max_blocks_per_request,
      impl_->max_pages,
      scale,
      path,
      output,
      error,
      &dispatch_result.query_copies,
      &dispatch_result.output_copies);
  return dispatch_result.dispatched;
}

PageRuntimeMetrics PageRuntime::metrics(std::string* error) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  PageRuntimeMetrics result;
  (void)error;
  result.shutdown = impl_->stopped;
  result.max_pages = impl_->max_pages;
  result.max_blocks_per_request = impl_->max_blocks_per_request;
  result.block_size = impl_->block_size;
  // The configured page capacity remains available for diagnostics, but the
  // backing storage has been released after shutdown; report resident bytes,
  // rather than stale allocation capacity, in the post-shutdown snapshot.
  result.bytes_per_layer = impl_->stopped ? 0 : impl_->bytes_per_layer;
  result.kv_bytes = result.bytes_per_layer * impl_->num_layers;
  result.page_allocations = impl_->page_allocations;
  result.request_allocations = impl_->request_allocations;
  result.prefix_allocations = impl_->prefix_allocations;
  result.cow_events = impl_->cow_events;
  result.evictions = impl_->eviction_count;
  result.releases = impl_->release_count;
  result.cancellations = impl_->cancellation_count;
  result.append_tokens = impl_->append_tokens;
  result.dispatches = impl_->dispatches;
  result.dispatch_failures = impl_->dispatch_failures;
  result.native_dispatches = impl_->native_dispatches;
  result.native_failures = impl_->native_failures;
  result.buffer_allocations = impl_->buffer_allocations;
  result.decode_buffer_allocations = impl_->decode_buffer_allocations;
  result.query_copies = impl_->query_copies;
  result.output_copies = impl_->output_copies;
  result.metadata_copies = impl_->metadata_copies;
  result.metadata_bytes = impl_->metadata_bytes;
  result.kv_copy_bytes = impl_->kv_copy_bytes;
  result.kv_pool_copies = impl_->kv_pool_copies;
  result.attention_validation_bytes = impl_->attention_validation_bytes;
  result.decode_page_resolution_checks =
      impl_->decode_page_resolution_checks;
  result.snapshot_failures = impl_->snapshot_failures;
  result.restore_failures = impl_->restore_failures;
  for (const auto& page : impl_->pages) {
    if (!page.allocated) {
      ++result.free_pages;
    } else {
      ++result.resident_pages;
      if (page.references > 0) {
        ++result.referenced_pages;
      } else {
        ++result.evictable_pages;
      }
      if (page.references > 1) {
        ++result.shared_pages;
      }
    }
  }
  for (const auto& request : impl_->requests) {
    if (request.live) {
      ++result.requests;
    }
  }
  for (const auto& prefix : impl_->prefixes) {
    if (prefix.live) {
      ++result.prefixes;
    }
  }
  return result;
}

void PageRuntime::shutdown() {
  impl_->shutdown_and_wait();
}

bool PageRuntime::is_shutdown() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stopped;
}

uint32_t PageRuntime::num_layers() const { return impl_->num_layers; }
uint32_t PageRuntime::num_attention_heads() const {
  return impl_->num_attention_heads;
}
uint32_t PageRuntime::num_key_value_heads() const {
  return impl_->num_key_value_heads;
}
uint32_t PageRuntime::head_dim() const { return impl_->head_dim; }
uint32_t PageRuntime::block_size() const { return impl_->block_size; }
uint32_t PageRuntime::max_pages() const { return impl_->max_pages; }
uint32_t PageRuntime::max_blocks_per_request() const {
  return impl_->max_blocks_per_request;
}
uint32_t PageRuntime::max_requests() const { return impl_->max_requests; }

void PageRuntime::set_metallib_path(const std::string& path) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->library_path = path;
}

}  // namespace metal_context
