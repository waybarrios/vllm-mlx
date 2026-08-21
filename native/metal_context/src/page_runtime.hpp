// SPDX-License-Identifier: Apache-2.0
//
// Ownership runtime for the optional Metal Context Engine.
//
// This header intentionally contains no Python or Objective-C types.  The
// CPython bridge in page_runtime_python.mm owns the Python ABI while this
// class owns request/page lifetime, page-table state, and the preallocated
// per-layer KV storage.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metal_context {

struct PageRuntimeMetrics {
  uint64_t resident_pages = 0;
  uint64_t referenced_pages = 0;
  uint64_t shared_pages = 0;
  uint64_t evictable_pages = 0;
  uint64_t free_pages = 0;
  uint64_t requests = 0;
  uint64_t prefixes = 0;
  uint64_t page_allocations = 0;
  uint64_t request_allocations = 0;
  uint64_t prefix_allocations = 0;
  uint64_t cow_events = 0;
  uint64_t evictions = 0;
  uint64_t releases = 0;
  uint64_t cancellations = 0;
  uint64_t append_tokens = 0;
  uint64_t dispatches = 0;
  uint64_t dispatch_failures = 0;
  uint64_t native_dispatches = 0;
  uint64_t native_failures = 0;
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
  uint64_t max_pages = 0;
  uint64_t max_blocks_per_request = 0;
  uint64_t block_size = 0;
  bool shutdown = false;
  uint64_t bytes_per_layer = 0;
  uint64_t kv_bytes = 0;
};

class PageRuntime {
 public:
  // A bridge-side mutation transaction.  Native state is committed only
  // after the CPython caller has built its return object; rollback is
  // allocation-free and generation-safe.  The token is intentionally opaque
  // to Python and must not outlive its PageRuntime owner.
  class MutationToken {
   public:
    MutationToken();
    ~MutationToken();
    MutationToken(const MutationToken&) = delete;
    MutationToken& operator=(const MutationToken&) = delete;
    MutationToken(MutationToken&&) noexcept;
    MutationToken& operator=(MutationToken&&) noexcept;

   public:
    // Definition is private to the native translation unit; the public
    // declaration only lets PageRuntime's implementation carry the opaque
    // transaction state through its helpers.
    struct State;

   private:
    std::unique_ptr<State> state_;
    friend class PageRuntime;
  };

  PageRuntime(
      uint32_t num_layers,
      uint32_t num_attention_heads,
      uint32_t num_key_value_heads,
      uint32_t head_dim,
      uint32_t block_size,
      uint32_t max_pages,
      uint32_t max_blocks_per_request,
      uint32_t max_requests);
  ~PageRuntime();

  PageRuntime(const PageRuntime&) = delete;
  PageRuntime& operator=(const PageRuntime&) = delete;
  PageRuntime(PageRuntime&&) = delete;
  PageRuntime& operator=(PageRuntime&&) = delete;

  // All methods return false and populate error on stale handles, invalid
  // shapes, exhausted capacity, or a shut-down runtime.  The Python bridge
  // converts these errors to the narrow exception types documented by the
  // public backend protocol.
  bool allocate_request(
      const std::string& request_id,
      uint32_t max_tokens,
      uint64_t* handle,
      std::string* error);
  bool allocate_request(
      const std::string& request_id,
      uint32_t max_tokens,
      uint64_t* handle,
      MutationToken* mutation,
      std::string* error);
  bool allocate_pages(
      uint64_t request,
      uint32_t count,
      std::vector<uint64_t>* handles,
      std::string* error);
  bool allocate_pages(
      uint64_t request,
      uint32_t count,
      std::vector<uint64_t>* handles,
      MutationToken* mutation,
      std::string* error);
  bool append_kv(
      uint64_t request,
      uint32_t layer,
      const uint16_t* keys,
      const uint16_t* values,
      uint32_t tokens,
      std::string* error);

  bool create_prefix(uint64_t request, uint64_t* prefix, std::string* error);
  bool create_prefix(
      uint64_t request,
      uint64_t* prefix,
      MutationToken* mutation,
      std::string* error);
  bool fork_prefix(uint64_t prefix, uint64_t* forked, std::string* error);
  bool fork_prefix(
      uint64_t prefix,
      uint64_t* forked,
      MutationToken* mutation,
      std::string* error);
  bool attach_prefix(uint64_t request, uint64_t prefix, std::string* error);
  bool release_prefix(uint64_t prefix, std::string* error);
  bool release_request(uint64_t request, bool cancelled, std::string* error);

  // Persistence is intentionally deferred to the next PR-stack package.  The
  // explicit failure methods keep the contract fail-closed while exposing
  // truthful counters to status/metrics callers.
  bool snapshot(
      uint64_t prefix,
      const std::string& destination,
      std::string* error);
  bool restore(
      const std::string& source,
      uint64_t* prefix,
      std::string* error);

  // Evict at most target_pages resident pages with no request/prefix
  // references.  A null target means evict every currently evictable page.
  uint32_t evict(uint32_t target_pages, bool has_target, std::string* error);
  uint32_t evict(
      uint32_t target_pages,
      bool has_target,
      MutationToken* mutation,
      std::string* error);

  bool page_table(
      uint64_t request,
      uint32_t layer,
      std::vector<int32_t>* table,
      std::string* error) const;
  bool request_pages(
      uint64_t request,
      std::vector<uint64_t>* pages,
      std::string* error) const;
  bool sequence_length(uint64_t request, uint32_t* length, std::string* error)
      const;

  // Dispatch the phase-one kernel using a consistent snapshot of the runtime
  // owned pages and metadata.  The kernel bridge remains synchronous in this
  // package; scheduler-owned asynchronous queues are a later package.
  bool paged_decode(
      uint64_t request,
      uint32_t layer,
      const uint16_t* query,
      size_t query_elements,
      float scale,
      std::vector<float>* output,
      std::string* error);
  bool paged_decode(
      uint64_t request,
      uint32_t layer,
      const uint16_t* query,
      size_t query_elements,
      float scale,
      std::vector<float>* output,
      MutationToken* mutation,
      std::string* error);

  // Commit/rollback are noexcept by contract: rollback is used from a C API
  // failure path after CPython allocation failure and performs no allocation.
  void commit(MutationToken* mutation) noexcept;
  void rollback(MutationToken* mutation) noexcept;

  PageRuntimeMetrics metrics(std::string* error) const;
  void shutdown();
  bool is_shutdown() const;

  uint32_t num_layers() const;
  uint32_t num_attention_heads() const;
  uint32_t num_key_value_heads() const;
  uint32_t head_dim() const;
  uint32_t block_size() const;
  uint32_t max_pages() const;
  uint32_t max_blocks_per_request() const;
  uint32_t max_requests() const;

  // The Python bridge supplies the packaged metallib location after module
  // initialization.  Keeping it on the runtime avoids a process-global
  // environment mutation and makes multiple extension copies deterministic.
  void set_metallib_path(const std::string& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Shared host-side ABI helper used by the native dispatch path and its
// allocation-free offset harness.  It returns the byte offset of one layer's
// sequence length within the [request, layer] int32 metadata buffer.
uint64_t page_runtime_sequence_buffer_offset(
    uint32_t request_slot,
    uint32_t num_layers,
    uint32_t layer);

}  // namespace metal_context
