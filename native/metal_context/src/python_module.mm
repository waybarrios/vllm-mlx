// SPDX-License-Identifier: Apache-2.0
//
// Optional CPython bridge for the phase-1 Metal Context Engine kernel.
//
// This bridge deliberately accepts contiguous host buffers rather than
// reaching into MLX's private allocator.  That makes the ABI testable and
// keeps installation optional.  The page runtime can add a zero-copy MLX
// adapter once the ownership/synchronization contract is qualified; it does
// not need to change the kernel's explicit layout or validation rules.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kThreadsPerThreadgroup = 128;

bool compiled_for_apple_silicon() {
#if defined(__arm64__) || defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

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

struct BufferGuard {
  Py_buffer view{};
  bool acquired = false;

  ~BufferGuard() {
    if (acquired) {
      PyBuffer_Release(&view);
    }
  }
};

struct Runtime {
  std::mutex mutex;
  bool attempted = false;
  bool available = false;
  std::string path;
  std::string error;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
};

Runtime g_runtime;

bool set_dict_item(PyObject* dict, const char* key, PyObject* value) {
  if (value == nullptr) {
    return false;
  }
  if (PyDict_SetItemString(dict, key, value) < 0) {
    Py_DECREF(value);
    return false;
  }
  Py_DECREF(value);
  return true;
}

bool set_dict_string(PyObject* dict, const char* key, const std::string& value) {
  return set_dict_item(dict, key, PyUnicode_FromString(value.c_str()));
}

bool set_dict_bool(PyObject* dict, const char* key, bool value) {
  return set_dict_item(dict, key, PyBool_FromLong(value ? 1 : 0));
}

bool set_dict_uint(PyObject* dict, const char* key, uint32_t value) {
  return set_dict_item(dict, key, PyLong_FromUnsignedLong(value));
}

PyObject* unavailable_capabilities(const std::string& reason) {
  PyObject* result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }
  if (!set_dict_bool(result, "available", false) ||
      !set_dict_bool(result, "compiled", true) ||
      !set_dict_bool(result, "metal_device", false) ||
      !set_dict_bool(result, "apple_silicon", compiled_for_apple_silicon()) ||
      !set_dict_bool(result, "serving_ready", false) ||
      !set_dict_uint(result, "abi_version", kAbiVersion) ||
      !set_dict_string(result, "backend", "metal-context") ||
      !set_dict_string(result, "reason", reason) ||
      !set_dict_string(result, "kernel", "metal_context_paged_decode") ||
      !set_dict_string(result, "kv_dtype", "bfloat16") ||
      !set_dict_bool(result, "gqa", true) ||
      !set_dict_bool(result, "partial_blocks", true) ||
      !set_dict_bool(result, "online_softmax", true)) {
    Py_DECREF(result);
    return nullptr;
  }
  PyObject* block_sizes = Py_BuildValue("(ii)", 16, 32);
  PyObject* head_dims = Py_BuildValue("(i)", static_cast<int>(kHeadDim));
  if (block_sizes == nullptr || head_dims == nullptr ||
      PyDict_SetItemString(result, "block_sizes", block_sizes) < 0 ||
      PyDict_SetItemString(result, "head_dims", head_dims) < 0) {
    Py_XDECREF(block_sizes);
    Py_XDECREF(head_dims);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(block_sizes);
  Py_DECREF(head_dims);
  return result;
}

std::string py_object_path(PyObject* module) {
  const char* override_path = std::getenv("VLLM_MLX_METAL_CONTEXT_METALLIB");
  if (override_path != nullptr && override_path[0] != '\0') {
    return override_path;
  }

  PyObject* file_object = PyModule_GetFilenameObject(module);
  if (file_object == nullptr) {
    PyErr_Clear();
    return {};
  }
  const char* file_name = PyUnicode_AsUTF8(file_object);
  std::string result;
  if (file_name != nullptr) {
    try {
      result =
          (std::filesystem::path(file_name).parent_path() /
           "_metal_context.metallib")
              .string();
    } catch (const std::exception&) {
      result.clear();
    }
  } else {
    PyErr_Clear();
  }
  Py_DECREF(file_object);
  return result;
}

std::string ns_error_description(NSError* error, const char* fallback) {
  if (error != nil && error.localizedDescription != nil) {
    const char* description = [error.localizedDescription UTF8String];
    if (description != nullptr && description[0] != '\0') {
      return description;
    }
  }
  return fallback;
}

bool ensure_runtime(const std::string& path) {
  std::lock_guard<std::mutex> lock(g_runtime.mutex);
  if (g_runtime.attempted && g_runtime.path == path) {
    return g_runtime.available;
  }

  g_runtime.attempted = true;
  g_runtime.available = false;
  g_runtime.path = path;
  g_runtime.error.clear();
  g_runtime.device = nil;
  g_runtime.queue = nil;
  g_runtime.pipeline = nil;

  if (!compiled_for_apple_silicon()) {
    g_runtime.error = "the Metal Context Engine requires an Apple Silicon arm64 build";
    return false;
  }
  if (path.empty()) {
    g_runtime.error =
        "the packaged _metal_context.metallib could not be located";
    return false;
  }
  try {
    if (!std::filesystem::exists(path)) {
      g_runtime.error = "the Metal library does not exist at " + path;
      return false;
    }
  } catch (const std::filesystem::filesystem_error& exception) {
    g_runtime.error = "could not inspect the Metal library path: ";
    g_runtime.error += exception.what();
    return false;
  }

  @autoreleasepool {
    do {
      g_runtime.device = MTLCreateSystemDefaultDevice();
      if (g_runtime.device == nil) {
        g_runtime.error = "MTLCreateSystemDefaultDevice returned no GPU";
        break;
      }

      NSData* library_bytes = [NSData
          dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
      if (library_bytes == nil) {
        g_runtime.error = "the packaged Metal library could not be read";
        break;
      }

      if (library_bytes.length == 0) {
        g_runtime.error = "the packaged Metal library is empty";
        break;
      }
      void* owned_library_bytes = std::malloc(library_bytes.length);
      if (owned_library_bytes == nullptr) {
        g_runtime.error =
            "could not allocate owned storage for the packaged Metal library";
        break;
      }
      std::memcpy(owned_library_bytes, library_bytes.bytes, library_bytes.length);
      dispatch_data_t library_data = dispatch_data_create(
          owned_library_bytes,
          library_bytes.length,
          nullptr,
          DISPATCH_DATA_DESTRUCTOR_FREE);
      if (library_data == nullptr) {
        std::free(owned_library_bytes);
        g_runtime.error = "the packaged Metal library could not be mapped";
        break;
      }
      NSError* error = nil;
      id<MTLLibrary> library =
          [g_runtime.device newLibraryWithData:library_data error:&error];
      if (library == nil) {
        g_runtime.error = ns_error_description(error, "Metal rejected the library");
        break;
      }

      id<MTLFunction> function =
          [library newFunctionWithName:@"metal_context_paged_decode"];
      if (function == nil) {
        g_runtime.error =
            "the packaged Metal library is missing metal_context_paged_decode";
        break;
      }

      g_runtime.pipeline =
          [g_runtime.device newComputePipelineStateWithFunction:function
                                                            error:&error];
      if (g_runtime.pipeline == nil) {
        g_runtime.error = ns_error_description(
            error, "Metal could not create the kernel pipeline");
        break;
      }

      g_runtime.queue = [g_runtime.device newCommandQueue];
      if (g_runtime.queue == nil) {
        g_runtime.error = "Metal could not create a command queue";
        break;
      }
      g_runtime.available = true;
    } while (false);
  }
  return g_runtime.available;
}

bool acquire_contiguous(PyObject* object, BufferGuard* guard, const char* name) {
  if (PyObject_GetBuffer(
          object,
          &guard->view,
          PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) {
    PyErr_Format(
        PyExc_TypeError,
        "%s must expose a contiguous buffer (NumPy/MLX host arrays are "
        "accepted after conversion)",
        name);
    return false;
  }
  guard->acquired = true;
  return true;
}

bool has_shape(const Py_buffer& view, int ndim) {
  return view.ndim == ndim && view.shape != nullptr;
}

bool host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

bool is_native_format(const Py_buffer& view, char expected) {
  if (view.format == nullptr) {
    return false;
  }
  // NumPy's native scalar buffers expose a one-character PEP-3118 format.
  // Accept the native/no-prefix forms and an explicitly native ``@``/``=``
  // prefix.  An explicit ``>``/``!`` is always rejected; ``<`` is accepted
  // only on the little-endian Apple Silicon host this extension targets.
  const char* format = view.format;
  const char prefix = *format;
  if (prefix == '@' || prefix == '=') {
    ++format;
  } else if (prefix == '<') {
    if (!host_is_little_endian()) {
      return false;
    }
    ++format;
  } else if (prefix == '>' || prefix == '!') {
    return false;
  }
  return std::strlen(format) == 1 && format[0] == expected;
}

bool finite_bf16_bits(const uint16_t bits) {
  // BF16 has an eight-bit exponent.  All exponent bits set denotes either an
  // infinity or NaN; both are rejected before dispatch by this host-buffer
  // foundation API.
  return (bits & 0x7f80u) != 0x7f80u;
}

float bf16_to_float_host(const uint16_t bits) {
  const uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

bool all_finite_bf16(const Py_buffer& view) {
  const auto* values = static_cast<const uint16_t*>(view.buf);
  const size_t count = static_cast<size_t>(view.len) / sizeof(uint16_t);
  for (size_t index = 0; index < count; ++index) {
    if (!finite_bf16_bits(values[index])) {
      return false;
    }
  }
  return true;
}

float max_abs_bf16(const Py_buffer& view) {
  const auto* values = static_cast<const uint16_t*>(view.buf);
  const size_t count = static_cast<size_t>(view.len) / sizeof(uint16_t);
  float maximum = 0.0f;
  for (size_t index = 0; index < count; ++index) {
    maximum = std::max(maximum, std::fabs(bf16_to_float_host(values[index])));
  }
  return maximum;
}

bool checked_product(std::initializer_list<uint64_t> values, uint64_t* result) {
  uint64_t product = 1;
  for (uint64_t value : values) {
    if (value != 0 && product > std::numeric_limits<uint64_t>::max() / value) {
      return false;
    }
    product *= value;
  }
  *result = product;
  return true;
}

bool validate_inputs(
    const BufferGuard& query,
    const BufferGuard& key_pages,
    const BufferGuard& value_pages,
    const BufferGuard& page_table,
    const BufferGuard& sequence_lengths,
    int num_kv_heads,
    int block_size,
    float scale,
    PagedDecodeParams* params,
    std::string* error) {
  const Py_buffer& q = query.view;
  const Py_buffer& k = key_pages.view;
  const Py_buffer& v = value_pages.view;
  const Py_buffer& table = page_table.view;
  const Py_buffer& lengths = sequence_lengths.view;

  // Check dimensionality before dereferencing any shape pointer.  Malformed
  // Python buffer exporters are part of the adversarial input surface.
  if (!has_shape(q, 3) || !has_shape(k, 4) || !has_shape(v, 4) ||
      !has_shape(table, 2) ||
      !has_shape(lengths, 1)) {
    *error =
        "expected query [batch, query_heads, 128], key/value "
        "[pages, kv_heads, block_size, 128], page_table [batch, max_blocks], "
        "and sequence_lengths [batch]";
    return false;
  }

  if (q.shape[2] != kHeadDim) {
    *error = "query must have head_dim 128 for the phase-1 Metal kernel";
    return false;
  }

  if (q.itemsize != 2 || k.itemsize != 2 || v.itemsize != 2 ||
      !is_native_format(q, 'H') || !is_native_format(k, 'H') ||
      !is_native_format(v, 'H')) {
    *error =
        "query, key_pages, and value_pages must expose BF16 storage as "
        "contiguous uint16 buffers (PEP-3118 format H)";
    return false;
  }
  if (table.itemsize != sizeof(int32_t) || lengths.itemsize != sizeof(int32_t) ||
      !is_native_format(table, 'i') || !is_native_format(lengths, 'i')) {
    *error =
        "page_table and sequence_lengths must expose signed int32 buffers "
        "(PEP-3118 format i)";
    return false;
  }
  if (block_size != 16 && block_size != 32) {
    *error = "block_size must be 16 or 32 for the phase-1 Metal kernel";
    return false;
  }
  if (num_kv_heads <= 0) {
    *error = "num_kv_heads must be positive";
    return false;
  }

  const Py_ssize_t batch = q.shape[0];
  const Py_ssize_t query_heads = q.shape[1];
  const Py_ssize_t pages = k.shape[0];
  const Py_ssize_t kv_heads = k.shape[1];
  const Py_ssize_t max_blocks = table.shape[1];
  if (batch <= 0 || query_heads <= 0 || pages <= 0 || kv_heads <= 0 ||
      max_blocks <= 0) {
    *error = "batch, heads, pages, and max_blocks must be positive";
    return false;
  }
  if (table.shape[0] != batch || lengths.shape[0] != batch ||
      k.shape[1] != num_kv_heads || v.shape[0] != pages ||
      v.shape[1] != kv_heads || k.shape[2] != block_size ||
      v.shape[2] != block_size || k.shape[3] != kHeadDim ||
      v.shape[3] != kHeadDim) {
    *error = "buffer shapes do not agree with batch, heads, block_size, or 128 head_dim";
    return false;
  }
  if (query_heads % kv_heads != 0) {
    *error = "query_heads must be an integer multiple of kv_heads (GQA)";
    return false;
  }

  const uint64_t max_sequence =
      static_cast<uint64_t>(max_blocks) * static_cast<uint64_t>(block_size);
  if (max_sequence > std::numeric_limits<uint32_t>::max() ||
      batch > std::numeric_limits<uint32_t>::max() ||
      query_heads > std::numeric_limits<uint32_t>::max() ||
      kv_heads > std::numeric_limits<uint32_t>::max() ||
      pages > std::numeric_limits<uint32_t>::max() ||
      max_blocks > std::numeric_limits<uint32_t>::max()) {
    *error = "buffer dimensions exceed the native uint32 ABI";
    return false;
  }

  uint64_t query_elements = 0;
  uint64_t kv_elements = 0;
  uint64_t table_elements = 0;
  if (!checked_product(
          {static_cast<uint64_t>(batch), static_cast<uint64_t>(query_heads),
           kHeadDim},
          &query_elements) ||
      !checked_product(
          {static_cast<uint64_t>(pages), static_cast<uint64_t>(kv_heads),
           static_cast<uint64_t>(block_size), kHeadDim},
          &kv_elements) ||
      !checked_product(
          {static_cast<uint64_t>(batch), static_cast<uint64_t>(max_blocks)},
          &table_elements)) {
    *error = "buffer dimensions overflow the native size calculation";
    return false;
  }
  // Every composite offset in the MSL kernel is a uint32 expression.  Keep
  // the entire reachable buffer/index space below UINT32_MAX rather than
  // allowing a valid host allocation to wrap inside the shader.
  if (query_elements > std::numeric_limits<uint32_t>::max() ||
      kv_elements > std::numeric_limits<uint32_t>::max() ||
      table_elements > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(batch) * static_cast<uint64_t>(query_heads) >
          std::numeric_limits<uint32_t>::max()) {
    *error =
        "buffer dimensions exceed the uint32 index/grid limit of the Metal "
        "kernel";
    return false;
  }
  const uint64_t expected_q = query_elements * sizeof(uint16_t);
  const uint64_t expected_kv = kv_elements * sizeof(uint16_t);
  const uint64_t expected_table = table_elements * sizeof(int32_t);
  if (expected_q != static_cast<uint64_t>(q.len) ||
      expected_kv != static_cast<uint64_t>(k.len) ||
      expected_kv != static_cast<uint64_t>(v.len) ||
      expected_table != static_cast<uint64_t>(table.len) ||
      static_cast<uint64_t>(batch) * sizeof(int32_t) !=
          static_cast<uint64_t>(lengths.len)) {
    *error = "buffer byte lengths do not match their declared shapes";
    return false;
  }

  if (!all_finite_bf16(q) || !all_finite_bf16(k) || !all_finite_bf16(v)) {
    *error =
        "query, key_pages, and value_pages must contain only finite BF16 "
        "values";
    return false;
  }

  // The shader performs the dot reduction in float32.  Finite BF16 inputs
  // can still produce an infinity when multiplied/reduced at head_dim=128,
  // and a subsequent Inf-Inf in online softmax would become NaN.  Use a
  // conservative half-FLT_MAX envelope to leave room for float32 rounding;
  // this is intentionally a host rejection in the host-buffer foundation.
  constexpr long double kFloat32SafeMagnitude =
      static_cast<long double>(std::numeric_limits<float>::max()) * 0.5L;
  const long double max_dot =
      static_cast<long double>(max_abs_bf16(q)) *
      static_cast<long double>(max_abs_bf16(k)) * kHeadDim;
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

  const auto* table_data = static_cast<const int32_t*>(table.buf);
  const auto* length_data = static_cast<const int32_t*>(lengths.buf);
  for (Py_ssize_t request = 0; request < batch; ++request) {
    const int32_t length = length_data[request];
    if (length < 0 || static_cast<uint64_t>(length) > max_sequence) {
      *error = "sequence_lengths must be in [0, max_blocks * block_size]";
      return false;
    }
    const Py_ssize_t needed_blocks =
        (static_cast<Py_ssize_t>(length) + block_size - 1) / block_size;
    for (Py_ssize_t logical_block = 0; logical_block < needed_blocks;
         ++logical_block) {
      const int32_t physical_page =
          table_data[request * table.shape[1] + logical_block];
      if (physical_page < 0 || physical_page >= pages) {
        *error = "page_table contains an out-of-range page for a live token";
        return false;
      }
    }
  }

  params->batch_size = static_cast<uint32_t>(batch);
  params->query_heads = static_cast<uint32_t>(query_heads);
  params->kv_heads = static_cast<uint32_t>(kv_heads);
  params->head_dim = kHeadDim;
  params->block_size = static_cast<uint32_t>(block_size);
  params->max_blocks = static_cast<uint32_t>(max_blocks);
  params->page_count = static_cast<uint32_t>(pages);
  params->page_table_stride = static_cast<uint32_t>(max_blocks);
  params->reserved0 = 0;
  params->reserved1 = 0;
  params->reserved2 = 0;
  return true;
}

PyObject* py_capabilities(PyObject* self, PyObject*) {
  const std::string path = py_object_path(self);
  const bool available = ensure_runtime(path);
  std::lock_guard<std::mutex> lock(g_runtime.mutex);

  if (!available) {
    return unavailable_capabilities(g_runtime.error);
  }

  PyObject* result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }
  const char* device_name =
      g_runtime.device == nil ? "" : [g_runtime.device.name UTF8String];
  const std::string name = device_name == nullptr ? "" : device_name;
  if (!set_dict_bool(result, "available", true) ||
      !set_dict_bool(result, "compiled", true) ||
      !set_dict_bool(result, "metal_device", true) ||
      !set_dict_bool(result, "apple_silicon", true) ||
      !set_dict_bool(result, "serving_ready", false) ||
      !set_dict_uint(result, "abi_version", kAbiVersion) ||
      !set_dict_string(result, "backend", "metal-context") ||
      !set_dict_string(result, "reason", "") ||
      !set_dict_string(result, "device_name", name) ||
      !set_dict_string(result, "metallib_path", g_runtime.path) ||
      !set_dict_string(result, "kernel", "metal_context_paged_decode")) {
    Py_DECREF(result);
    return nullptr;
  }

  PyObject* block_sizes = Py_BuildValue("(ii)", 16, 32);
  PyObject* head_dims = Py_BuildValue("(i)", static_cast<int>(kHeadDim));
  if (block_sizes == nullptr || head_dims == nullptr ||
      PyDict_SetItemString(result, "block_sizes", block_sizes) < 0 ||
      PyDict_SetItemString(result, "head_dims", head_dims) < 0 ||
      !set_dict_bool(result, "gqa", true) ||
      !set_dict_bool(result, "partial_blocks", true) ||
      !set_dict_bool(result, "online_softmax", true) ||
      !set_dict_string(result, "kv_dtype", "bfloat16")) {
    Py_XDECREF(block_sizes);
    Py_XDECREF(head_dims);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(block_sizes);
  Py_DECREF(head_dims);
  return result;
}

PyObject* py_paged_decode(PyObject* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "query", "key_pages", "value_pages", "page_table",
      "sequence_lengths", "num_kv_heads", "block_size", "scale", nullptr};
  PyObject* query_object = nullptr;
  PyObject* key_object = nullptr;
  PyObject* value_object = nullptr;
  PyObject* table_object = nullptr;
  PyObject* lengths_object = nullptr;
  int num_kv_heads = 0;
  int block_size = 0;
  float scale = 0.0f;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OOOOOiif:paged_decode",
          const_cast<char**>(keywords),
          &query_object,
          &key_object,
          &value_object,
          &table_object,
          &lengths_object,
          &num_kv_heads,
          &block_size,
          &scale)) {
    return nullptr;
  }
  if (!std::isfinite(scale)) {
    PyErr_SetString(PyExc_ValueError, "scale must be finite");
    return nullptr;
  }

  BufferGuard query;
  BufferGuard key_pages;
  BufferGuard value_pages;
  BufferGuard page_table;
  BufferGuard sequence_lengths;
  if (!acquire_contiguous(query_object, &query, "query") ||
      !acquire_contiguous(key_object, &key_pages, "key_pages") ||
      !acquire_contiguous(value_object, &value_pages, "value_pages") ||
      !acquire_contiguous(table_object, &page_table, "page_table") ||
      !acquire_contiguous(
          lengths_object, &sequence_lengths, "sequence_lengths")) {
    return nullptr;
  }

  PagedDecodeParams params{};
  std::string validation_error;
  if (!validate_inputs(
          query,
          key_pages,
          value_pages,
          page_table,
          sequence_lengths,
          num_kv_heads,
          block_size,
          scale,
          &params,
          &validation_error)) {
    PyErr_SetString(PyExc_ValueError, validation_error.c_str());
    return nullptr;
  }
  params.scale = scale;

  const std::string path = py_object_path(self);
  if (!ensure_runtime(path)) {
    std::lock_guard<std::mutex> lock(g_runtime.mutex);
    PyErr_Format(
        PyExc_RuntimeError,
        "metal-context backend unavailable: %s",
        g_runtime.error.c_str());
    return nullptr;
  }

  uint64_t output_elements = 0;
  if (!checked_product(
          {params.batch_size, params.query_heads, params.head_dim},
          &output_elements) ||
      output_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    PyErr_SetString(PyExc_OverflowError, "output shape exceeds native size limits");
    return nullptr;
  }
  const size_t output_bytes = static_cast<size_t>(output_elements) * sizeof(float);

  @autoreleasepool {
    // Serialize the first low-level bridge while ownership is still host
    // backed.  The future page runtime will replace these copies with
    // allocator-owned GPU buffers and keep the command queue asynchronous.
    std::lock_guard<std::mutex> lock(g_runtime.mutex);
    id<MTLDevice> device = g_runtime.device;
    id<MTLCommandQueue> queue = g_runtime.queue;
    id<MTLComputePipelineState> pipeline = g_runtime.pipeline;
    if (device == nil || queue == nil || pipeline == nil) {
      PyErr_SetString(PyExc_RuntimeError, "metal-context runtime lost its pipeline");
      return nullptr;
    }

    id<MTLBuffer> query_buffer =
        [device newBufferWithLength:(NSUInteger)query.view.len
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer =
        [device newBufferWithLength:(NSUInteger)key_pages.view.len
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> value_buffer =
        [device newBufferWithLength:(NSUInteger)value_pages.view.len
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> table_buffer =
        [device newBufferWithLength:(NSUInteger)page_table.view.len
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> lengths_buffer =
        [device newBufferWithLength:(NSUInteger)sequence_lengths.view.len
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device newBufferWithLength:(NSUInteger)output_bytes
                             options:MTLResourceStorageModeShared];
    if (query_buffer == nil || key_buffer == nil || value_buffer == nil ||
        table_buffer == nil || lengths_buffer == nil || output_buffer == nil) {
      PyErr_SetString(PyExc_MemoryError, "Metal could not allocate kernel buffers");
      return nullptr;
    }

    std::memcpy(query_buffer.contents, query.view.buf, query.view.len);
    std::memcpy(key_buffer.contents, key_pages.view.buf, key_pages.view.len);
    std::memcpy(value_buffer.contents, value_pages.view.buf, value_pages.view.len);
    std::memcpy(table_buffer.contents, page_table.view.buf, page_table.view.len);
    std::memcpy(
        lengths_buffer.contents,
        sequence_lengths.view.buf,
        sequence_lengths.view.len);
    std::memset(output_buffer.contents, 0, output_bytes);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      PyErr_SetString(PyExc_RuntimeError, "Metal could not create a command encoder");
      return nullptr;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:key_buffer offset:0 atIndex:1];
    [encoder setBuffer:value_buffer offset:0 atIndex:2];
    [encoder setBuffer:table_buffer offset:0 atIndex:3];
    [encoder setBuffer:lengths_buffer offset:0 atIndex:4];
    [encoder setBuffer:output_buffer offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    const MTLSize grid = MTLSizeMake(
        (NSUInteger)params.batch_size * params.query_heads, 1, 1);
    const MTLSize group = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      std::string error = ns_error_description(
          command_buffer.error,
          "Metal command buffer did not complete successfully");
      PyErr_SetString(PyExc_RuntimeError, error.c_str());
      return nullptr;
    }
    return PyBytes_FromStringAndSize(
        static_cast<const char*>(output_buffer.contents),
        static_cast<Py_ssize_t>(output_bytes));
  }
}

PyObject* py_shutdown(PyObject*, PyObject*) {
  std::lock_guard<std::mutex> lock(g_runtime.mutex);
  g_runtime.device = nil;
  g_runtime.queue = nil;
  g_runtime.pipeline = nil;
  g_runtime.attempted = false;
  g_runtime.available = false;
  g_runtime.path.clear();
  g_runtime.error.clear();
  Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"capabilities", py_capabilities, METH_NOARGS,
     "Return the compiled Metal Context Engine capability record."},
    {"paged_decode", reinterpret_cast<PyCFunction>(py_paged_decode),
     METH_VARARGS | METH_KEYWORDS,
     "Run BF16 paged decode attention from contiguous host buffers."},
    {"shutdown", py_shutdown, METH_NOARGS,
     "Release the native Metal pipeline and queue."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "vllm_mlx._metal_context",
    "Optional native Metal Context Engine kernel bridge.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__metal_context() {
  return PyModule_Create(&kModule);
}
