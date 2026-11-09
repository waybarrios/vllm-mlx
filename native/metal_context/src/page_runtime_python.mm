// SPDX-License-Identifier: Apache-2.0
//
// CPython ABI for the native page ownership runtime.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "page_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

using metal_context::PageRuntime;

struct BufferGuard {
  Py_buffer view{};
  bool acquired = false;
  ~BufferGuard() {
    if (acquired) {
      PyBuffer_Release(&view);
    }
  }
};

struct PageRuntimeObject {
  PyObject_HEAD
  PageRuntime* runtime = nullptr;
  std::mutex lifecycle_mutex;
  std::condition_variable lifecycle_cv;
  uint32_t active_calls = 0;
};

bool acquire_buffer(PyObject* object, BufferGuard* guard, const char* name) {
  if (PyObject_GetBuffer(
          object,
          &guard->view,
          PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) {
    PyErr_Format(
        PyExc_TypeError,
        "%s must expose a contiguous buffer (NumPy arrays are accepted)",
        name);
    return false;
  }
  guard->acquired = true;
  return true;
}

bool host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

bool is_native_uint16_buffer(const Py_buffer& view) {
  if (view.itemsize != sizeof(uint16_t) || view.format == nullptr) {
    return false;
  }
  const char* format = view.format;
  if (*format == '@' || *format == '=') {
    ++format;
  } else if (*format == '<') {
    if (!host_is_little_endian()) {
      return false;
    }
    ++format;
  } else if (*format == '>' || *format == '!') {
    return false;
  }
  return std::strlen(format) == 1 && format[0] == 'H';
}

bool finite_bf16(const Py_buffer& view) {
  const auto* data = static_cast<const uint16_t*>(view.buf);
  const size_t count = static_cast<size_t>(view.len) / sizeof(uint16_t);
  for (size_t index = 0; index < count; ++index) {
    if ((data[index] & 0x7f80u) == 0x7f80u) {
      return false;
    }
  }
  return true;
}

bool runtime_error(const std::string& error) {
  PyErr_SetString(PyExc_RuntimeError, error.c_str());
  return false;
}

bool value_error(const std::string& error) {
  PyErr_SetString(PyExc_ValueError, error.c_str());
  return false;
}

bool lifecycle_error(const std::string& error) {
  if (error.find("capacity exhausted") != std::string::npos) {
    PyErr_SetString(PyExc_MemoryError, error.c_str());
    return false;
  }
  return value_error(error);
}

bool not_implemented_error(const std::string& error) {
  PyErr_SetString(PyExc_NotImplementedError, error.c_str());
  return false;
}

// No C++ exception may cross a CPython ``PyCFunction`` ABI boundary.  Keep
// this mapping in one place and use the scope macros below around every
// bridge method that touches the native runtime (including vector/dict
// construction that can throw bad_alloc).
void set_native_exception(const std::bad_alloc& exception) {
  PyErr_SetString(
      PyExc_MemoryError,
      exception.what()[0] == '\0' ? "native page runtime allocation failed"
                                  : exception.what());
}

void set_native_exception(const std::exception& exception) {
  PyErr_SetString(
      PyExc_RuntimeError,
      exception.what()[0] == '\0' ? "native page runtime operation failed"
                                  : exception.what());
}

void set_unknown_native_exception() {
  PyErr_SetString(
      PyExc_RuntimeError,
      "unknown native page runtime exception");
}

#define PAGE_RUNTIME_PY_TRY try {
#define PAGE_RUNTIME_PY_CATCH                                                 \
  } catch (const std::bad_alloc& exception) {                                 \
    set_native_exception(exception);                                           \
    return nullptr;                                                            \
  } catch (const std::exception& exception) {                                  \
    set_native_exception(exception);                                           \
    return nullptr;                                                            \
  } catch (...) {                                                              \
    set_unknown_native_exception();                                            \
    return nullptr;                                                            \
  }

// A decode call releases the GIL while Metal waits synchronously.  Keep a
// lifecycle lease on the Python object's native pointer so __init__ and
// deallocation cannot delete the PageRuntime until the call has reacquired
// the GIL and left this scope.
class RuntimeCallLease {
 public:
  explicit RuntimeCallLease(PageRuntimeObject* owner) : owner_(owner) {}

  bool acquire() {
    std::unique_lock<std::mutex> lock(owner_->lifecycle_mutex);
    if (owner_->runtime == nullptr || owner_->runtime->is_shutdown()) {
      PyErr_SetString(PyExc_RuntimeError, "page runtime is shut down");
      return false;
    }
    runtime_ = owner_->runtime;
    ++owner_->active_calls;
    acquired_ = true;
    return true;
  }

  PageRuntime* get() const { return runtime_; }

  ~RuntimeCallLease() noexcept {
    if (!acquired_) {
      return;
    }
    std::lock_guard<std::mutex> lock(owner_->lifecycle_mutex);
    if (owner_->active_calls > 0) {
      --owner_->active_calls;
    }
    owner_->lifecycle_cv.notify_all();
  }

 private:
  PageRuntimeObject* owner_;
  PageRuntime* runtime_ = nullptr;
  bool acquired_ = false;
};

// Native mutators are deliberately kept uncommitted until the bridge has
// successfully constructed the Python object it returns.  CPython allocators
// can fail after the native operation has changed page ownership, so the
// guard rolls the operation back on every NULL return (including a caught
// bad_alloc).  PageRuntime::rollback is allocation-free and generation-safe.
class NativeMutationGuard {
 public:
  explicit NativeMutationGuard(PageRuntime* runtime) : runtime_(runtime) {}

  PageRuntime::MutationToken* token() { return &token_; }

  void commit() noexcept {
    if (!committed_) {
      runtime_->commit(&token_);
      committed_ = true;
    }
  }

  void rollback() noexcept {
    if (!committed_) {
      runtime_->rollback(&token_);
      committed_ = true;
    }
  }

  ~NativeMutationGuard() noexcept { rollback(); }

 private:
  PageRuntime* runtime_;
  PageRuntime::MutationToken token_;
  bool committed_ = false;
};

bool require_runtime(PageRuntimeObject* self) {
  if (self->runtime == nullptr || self->runtime->is_shutdown()) {
    PyErr_SetString(PyExc_RuntimeError, "page runtime is shut down");
    return false;
  }
  return true;
}

std::string packaged_metallib_path() {
  const char* override_path =
      std::getenv("VLLM_MLX_METAL_CONTEXT_METALLIB");
  if (override_path != nullptr && override_path[0] != '\0') {
    return override_path;
  }
  PyObject* module = PyImport_AddModule("vllm_mlx._metal_context");
  if (module == nullptr) {
    PyErr_Clear();
    return {};
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
      result = (std::filesystem::path(file_name).parent_path() /
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

bool py_uint64(PyObject* object, uint64_t* result, const char* name) {
  if (!PyLong_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be an integer handle", name);
    return false;
  }
  unsigned long long value = PyLong_AsUnsignedLongLong(object);
  if (value == ULLONG_MAX && PyErr_Occurred()) {
    return false;
  }
  *result = static_cast<uint64_t>(value);
  return true;
}

bool result_fault_requested(const char* operation, const char* stage) {
  const char* configured =
      std::getenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT");
  if (configured == nullptr || configured[0] == '\0') {
    return false;
  }
  return std::strcmp(configured, "all") == 0 ||
      std::strcmp(configured, operation) == 0 ||
      std::strcmp(configured, stage) == 0 ||
      (std::strstr(configured, operation) != nullptr &&
       std::strstr(configured, stage) != nullptr);
}

bool fail_result_allocation(const char* operation, const char* stage) {
  if (!result_fault_requested(operation, stage)) {
    return false;
  }
  PyErr_NoMemory();
  return true;
}

PyObject* result_u64(const char* operation, uint64_t value) {
  if (fail_result_allocation(operation, "long")) {
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(value);
}

PyObject* result_u32(const char* operation, uint32_t value) {
  if (fail_result_allocation(operation, "long")) {
    return nullptr;
  }
  return PyLong_FromUnsignedLong(value);
}

PyObject* result_bytes(
    const char* operation,
    const char* data,
    Py_ssize_t size) {
  if (fail_result_allocation(operation, "bytes")) {
    return nullptr;
  }
  return PyBytes_FromStringAndSize(data, size);
}

PyObject* tuple_from_u64(
    const std::vector<uint64_t>& values,
    const char* operation) {
  if (fail_result_allocation(operation, "tuple")) {
    return nullptr;
  }
  PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
  if (result == nullptr) {
    return nullptr;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    if (fail_result_allocation(operation, "tuple_item")) {
      Py_DECREF(result);
      return nullptr;
    }
    PyObject* value = PyLong_FromUnsignedLongLong(values[index]);
    if (value == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index), value);
  }
  return result;
}

PyObject* tuple_from_i32(
    const std::vector<int32_t>& values,
    const char* operation) {
  if (fail_result_allocation(operation, "tuple")) {
    return nullptr;
  }
  PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
  if (result == nullptr) {
    return nullptr;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    if (fail_result_allocation(operation, "tuple_item")) {
      Py_DECREF(result);
      return nullptr;
    }
    PyObject* value = PyLong_FromLong(values[index]);
    if (value == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(index), value);
  }
  return result;
}

int page_runtime_init(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  static const char* keywords[] = {
      "num_layers",
      "num_attention_heads",
      "num_key_value_heads",
      "head_dim",
      "block_size",
      "max_pages",
      "max_blocks_per_request",
      "max_requests",
      nullptr};
  unsigned int num_layers = 0;
  unsigned int num_attention_heads = 0;
  unsigned int num_key_value_heads = 0;
  unsigned int head_dim = 128;
  unsigned int block_size = 16;
  unsigned int max_pages = 64;
  unsigned int max_blocks_per_request = 1024;
  unsigned int max_requests = 64;

  // ``num_kv_heads`` is a short alias useful to adapters that already expose
  // the attention geometry under that spelling.  Normalize it to the public
  // canonical keyword, then remove the alias: PyArg_ParseTupleAndKeywords
  // rejects unknown keys even when the canonical value has been supplied.
  PyObject* normalized_kwargs = kwargs;
  if (kwargs != nullptr &&
      PyDict_GetItemString(kwargs, "num_key_value_heads") == nullptr) {
    PyObject* alias = PyDict_GetItemString(kwargs, "num_kv_heads");
    if (alias != nullptr) {
      normalized_kwargs = PyDict_Copy(kwargs);
      if (normalized_kwargs == nullptr ||
          PyDict_SetItemString(
              normalized_kwargs, "num_key_value_heads", alias) < 0) {
        Py_XDECREF(normalized_kwargs == kwargs ? nullptr : normalized_kwargs);
        return -1;
      }
      if (PyDict_DelItemString(normalized_kwargs, "num_kv_heads") < 0) {
        Py_DECREF(normalized_kwargs);
        return -1;
      }
    }
  }
  const int parsed = PyArg_ParseTupleAndKeywords(
      args,
      normalized_kwargs,
      "III|IIIII:PageRuntime",
      const_cast<char**>(keywords),
      &num_layers,
      &num_attention_heads,
      &num_key_value_heads,
      &head_dim,
      &block_size,
      &max_pages,
      &max_blocks_per_request,
      &max_requests);
  if (normalized_kwargs != kwargs) {
    Py_DECREF(normalized_kwargs);
  }
  if (!parsed) {
    return -1;
  }
  try {
    std::unique_ptr<PageRuntime> replacement = std::make_unique<PageRuntime>(
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        block_size,
        max_pages,
        max_blocks_per_request,
        max_requests);
    replacement->set_metallib_path(packaged_metallib_path());
    PageRuntime* prior = nullptr;
    {
      std::unique_lock<std::mutex> lock(self->lifecycle_mutex);
      std::exception_ptr wait_failure;
      Py_BEGIN_ALLOW_THREADS
      try {
        self->lifecycle_cv.wait(
            lock, [self] { return self->active_calls == 0; });
      } catch (...) {
        wait_failure = std::current_exception();
      }
      Py_END_ALLOW_THREADS
      if (wait_failure) {
        std::rethrow_exception(wait_failure);
      }
      prior = self->runtime;
      self->runtime = replacement.release();
    }
    delete prior;
  } catch (const std::bad_alloc&) {
    PyErr_SetString(PyExc_MemoryError, "could not allocate the native page runtime");
    return -1;
  } catch (const std::invalid_argument& exception) {
    PyErr_SetString(PyExc_ValueError, exception.what());
    return -1;
  } catch (const std::exception& exception) {
    set_native_exception(exception);
    return -1;
  } catch (...) {
    set_unknown_native_exception();
    return -1;
  }
  return 0;
}

PyObject* page_runtime_new(PyTypeObject* type, PyObject*, PyObject*) {
  PAGE_RUNTIME_PY_TRY
  auto* self = reinterpret_cast<PageRuntimeObject*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  try {
    // Default-initialize the C++ members without value-initializing the
    // PyObject_HEAD fields that tp_alloc already populated.
    new (self) PageRuntimeObject;
  } catch (...) {
    type->tp_free(reinterpret_cast<PyObject*>(self));
    throw;
  }
  return reinterpret_cast<PyObject*>(self);
  PAGE_RUNTIME_PY_CATCH
}

void page_runtime_dealloc(PageRuntimeObject* self) {
  PageRuntime* prior = nullptr;
  try {
    {
      std::unique_lock<std::mutex> lock(self->lifecycle_mutex);
      std::exception_ptr wait_failure;
      Py_BEGIN_ALLOW_THREADS
      try {
        self->lifecycle_cv.wait(
            lock, [self] { return self->active_calls == 0; });
      } catch (...) {
        wait_failure = std::current_exception();
      }
      Py_END_ALLOW_THREADS
      if (wait_failure) {
        std::rethrow_exception(wait_failure);
      }
      prior = self->runtime;
      self->runtime = nullptr;
    }
    delete prior;
  } catch (const std::bad_alloc& exception) {
    set_native_exception(exception);
    PyErr_WriteUnraisable(reinterpret_cast<PyObject*>(self));
  } catch (const std::exception& exception) {
    PyErr_SetString(PyExc_RuntimeError, exception.what());
    PyErr_WriteUnraisable(reinterpret_cast<PyObject*>(self));
  } catch (...) {
    set_unknown_native_exception();
    PyErr_WriteUnraisable(reinterpret_cast<PyObject*>(self));
  }
  self->~PageRuntimeObject();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* page_runtime_allocate_request(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request_id", "max_tokens", nullptr};
  const char* request_id = nullptr;
  unsigned int max_tokens = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "sI:allocate_request",
          const_cast<char**>(keywords),
          &request_id,
          &max_tokens) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t handle = 0;
  std::string error;
  NativeMutationGuard mutation(self->runtime);
  if (!self->runtime->allocate_request(
          request_id, max_tokens, &handle, mutation.token(), &error)) {
    lifecycle_error(error);
    return nullptr;
  }
  PyObject* result = result_u64("allocate_request", handle);
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_allocate_pages(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request", "count", nullptr};
  PyObject* request_object = nullptr;
  unsigned int count = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OI:allocate_pages",
          const_cast<char**>(keywords),
          &request_object,
          &count) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  std::vector<uint64_t> handles;
  std::string error;
  NativeMutationGuard mutation(self->runtime);
  if (!self->runtime->allocate_pages(
          request, count, &handles, mutation.token(), &error)) {
    lifecycle_error(error);
    return nullptr;
  }
  PyObject* result = tuple_from_u64(handles, "allocate_pages");
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_append_kv(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request", "layer", "keys", "values", nullptr};
  PyObject* request_object = nullptr;
  unsigned int layer = 0;
  PyObject* keys_object = nullptr;
  PyObject* values_object = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OIOO:append_kv",
          const_cast<char**>(keywords),
          &request_object,
          &layer,
          &keys_object,
          &values_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  BufferGuard keys;
  BufferGuard values;
  if (!acquire_buffer(keys_object, &keys, "keys") ||
      !acquire_buffer(values_object, &values, "values")) {
    return nullptr;
  }
  if (keys.view.ndim != 3 || values.view.ndim != 3 ||
      keys.view.shape == nullptr || values.view.shape == nullptr ||
      !is_native_uint16_buffer(keys.view) ||
      !is_native_uint16_buffer(values.view)) {
    value_error(
        "keys and values must be contiguous uint16 BF16 buffers with shape "
        "[tokens, kv_heads, 128]");
    return nullptr;
  }
  if (keys.view.shape[0] != values.view.shape[0] ||
      keys.view.shape[1] != values.view.shape[1] ||
      keys.view.shape[1] <= 0 ||
      keys.view.shape[1] != self->runtime->num_key_value_heads() ||
      static_cast<uint64_t>(keys.view.shape[0]) >
          std::numeric_limits<uint32_t>::max() ||
      keys.view.shape[2] != 128 || values.view.shape[2] != 128 ||
      static_cast<uint64_t>(keys.view.len) !=
          static_cast<uint64_t>(keys.view.shape[0]) * keys.view.shape[1] * 128 *
              sizeof(uint16_t) ||
      static_cast<uint64_t>(values.view.len) !=
          static_cast<uint64_t>(values.view.shape[0]) * values.view.shape[1] * 128 *
              sizeof(uint16_t)) {
    value_error("keys and values shapes/byte lengths do not agree");
    return nullptr;
  }
  if (!finite_bf16(keys.view) || !finite_bf16(values.view)) {
    value_error("keys and values must contain finite BF16 values");
    return nullptr;
  }
  std::string error;
  if (!self->runtime->append_kv(
          request,
          layer,
          static_cast<const uint16_t*>(keys.view.buf),
          static_cast<const uint16_t*>(values.view.buf),
          static_cast<uint32_t>(keys.view.shape[0]),
          &error)) {
    lifecycle_error(error);
    return nullptr;
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_create_prefix(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* request_object = nullptr;
  if (!PyArg_ParseTuple(args, "O:create_prefix", &request_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  uint64_t prefix = 0;
  std::string error;
  NativeMutationGuard mutation(self->runtime);
  if (!self->runtime->create_prefix(
          request, &prefix, mutation.token(), &error)) {
    value_error(error);
    return nullptr;
  }
  PyObject* result = result_u64("create_prefix", prefix);
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_fork_prefix(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* prefix_object = nullptr;
  if (!PyArg_ParseTuple(args, "O:fork_prefix", &prefix_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t prefix = 0;
  if (!py_uint64(prefix_object, &prefix, "prefix")) {
    return nullptr;
  }
  uint64_t forked = 0;
  std::string error;
  NativeMutationGuard mutation(self->runtime);
  if (!self->runtime->fork_prefix(
          prefix, &forked, mutation.token(), &error)) {
    value_error(error);
    return nullptr;
  }
  PyObject* result = result_u64("fork_prefix", forked);
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_attach_prefix(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* request_object = nullptr;
  PyObject* prefix_object = nullptr;
  if (!PyArg_ParseTuple(args, "OO:attach_prefix", &request_object, &prefix_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  uint64_t prefix = 0;
  if (!py_uint64(request_object, &request, "request") ||
      !py_uint64(prefix_object, &prefix, "prefix")) {
    return nullptr;
  }
  std::string error;
  if (!self->runtime->attach_prefix(request, prefix, &error)) {
    value_error(error);
    return nullptr;
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_release_prefix(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* prefix_object = nullptr;
  if (!PyArg_ParseTuple(args, "O:release_prefix", &prefix_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t prefix = 0;
  if (!py_uint64(prefix_object, &prefix, "prefix")) {
    return nullptr;
  }
  std::string error;
  if (!self->runtime->release_prefix(prefix, &error)) {
    value_error(error);
    return nullptr;
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_release(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs,
    bool cancelled) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request", nullptr};
  PyObject* request_object = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O:release",
          const_cast<char**>(keywords),
          &request_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  std::string error;
  if (!self->runtime->release_request(request, cancelled, &error)) {
    value_error(error);
    return nullptr;
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_snapshot(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"prefix", "destination", nullptr};
  PyObject* prefix_object = nullptr;
  const char* destination = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "Os:snapshot",
          const_cast<char**>(keywords),
          &prefix_object,
          &destination) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t prefix = 0;
  if (!py_uint64(prefix_object, &prefix, "prefix")) {
    return nullptr;
  }
  std::string error;
  if (!self->runtime->snapshot(prefix, destination, &error)) {
    if (error.find("deferred") != std::string::npos) {
      not_implemented_error(error);
    } else {
      runtime_error(error);
    }
    return nullptr;
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_restore(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"source", nullptr};
  const char* source = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "s:restore",
          const_cast<char**>(keywords),
          &source) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t prefix = 0;
  std::string error;
  if (!self->runtime->restore(source, &prefix, &error)) {
    if (error.find("deferred") != std::string::npos) {
      not_implemented_error(error);
    } else {
      runtime_error(error);
    }
    return nullptr;
  }
  return result_u64("restore", prefix);
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_release_request(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  return page_runtime_release(self, args, kwargs, false);
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_cancel(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  return page_runtime_release(self, args, kwargs, true);
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_evict(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"target_pages", nullptr};
  PyObject* target_object = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|O:evict",
          const_cast<char**>(keywords),
          &target_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  bool has_target = target_object != Py_None;
  unsigned int target = 0;
  if (has_target) {
    if (!PyLong_Check(target_object)) {
      PyErr_SetString(PyExc_TypeError, "target_pages must be a non-negative integer");
      return nullptr;
    }
    unsigned long value = PyLong_AsUnsignedLong(target_object);
    if (value == ULONG_MAX && PyErr_Occurred()) {
      return nullptr;
    }
    if (value > std::numeric_limits<unsigned int>::max()) {
      PyErr_SetString(PyExc_OverflowError, "target_pages is too large");
      return nullptr;
    }
    target = static_cast<unsigned int>(value);
  }
  std::string error;
  NativeMutationGuard mutation(self->runtime);
  const uint32_t evicted =
      self->runtime->evict(target, has_target, mutation.token(), &error);
  if (!error.empty()) {
    runtime_error(error);
    return nullptr;
  }
  PyObject* result = result_u32("evict", evicted);
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_page_table(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request", "layer", nullptr};
  PyObject* request_object = nullptr;
  unsigned int layer = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|I:page_table",
          const_cast<char**>(keywords),
          &request_object,
          &layer) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  std::vector<int32_t> table;
  std::string error;
  if (!self->runtime->page_table(request, layer, &table, &error)) {
    value_error(error);
    return nullptr;
  }
  return tuple_from_i32(table, "page_table");
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_request_pages(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* request_object = nullptr;
  if (!PyArg_ParseTuple(args, "O:request_pages", &request_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  std::vector<uint64_t> pages;
  std::string error;
  if (!self->runtime->request_pages(request, &pages, &error)) {
    value_error(error);
    return nullptr;
  }
  return tuple_from_u64(pages, "request_pages");
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_sequence_length(
    PageRuntimeObject* self,
    PyObject* args) {
  PAGE_RUNTIME_PY_TRY
  PyObject* request_object = nullptr;
  if (!PyArg_ParseTuple(args, "O:sequence_length", &request_object) ||
      !require_runtime(self)) {
    return nullptr;
  }
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  uint32_t length = 0;
  std::string error;
  if (!self->runtime->sequence_length(request, &length, &error)) {
    value_error(error);
    return nullptr;
  }
  return result_u32("sequence_length", length);
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_paged_decode(
    PageRuntimeObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PAGE_RUNTIME_PY_TRY
  static const char* keywords[] = {"request", "layer", "query", "scale", nullptr};
  PyObject* request_object = nullptr;
  unsigned int layer = 0;
  PyObject* query_object = nullptr;
  float scale = 1.0f / std::sqrt(128.0f);
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OIO|f:paged_decode",
          const_cast<char**>(keywords),
          &request_object,
          &layer,
          &query_object,
          &scale)) {
    return nullptr;
  }
  RuntimeCallLease runtime_lease(self);
  if (!runtime_lease.acquire()) {
    return nullptr;
  }
  PageRuntime* runtime = runtime_lease.get();
  uint64_t request = 0;
  if (!py_uint64(request_object, &request, "request")) {
    return nullptr;
  }
  BufferGuard query;
  if (!acquire_buffer(query_object, &query, "query")) {
    return nullptr;
  }
  if (query.view.ndim != 2 || query.view.shape == nullptr ||
      !is_native_uint16_buffer(query.view) ||
      query.view.shape[0] != runtime->num_attention_heads() ||
      query.view.shape[1] != 128 ||
      static_cast<uint64_t>(query.view.len) !=
          static_cast<uint64_t>(query.view.shape[0]) * 128 * sizeof(uint16_t) ||
      !finite_bf16(query.view)) {
    value_error(
        "query must be a contiguous finite uint16 BF16 buffer with shape "
        "[query_heads, 128]");
    return nullptr;
  }
  if (!std::isfinite(scale)) {
    value_error("scale must be finite");
    return nullptr;
  }
  std::vector<float> output;
  std::string error;
  bool dispatched = false;
  NativeMutationGuard mutation(runtime);
  // Validation and buffer acquisition above require the GIL.  The compiled
  // runtime owns its lifecycle/dispatch locks, so release the GIL while the
  // synchronous Metal command is in flight and allow independent requests to
  // overlap at the Python adapter boundary.
  std::exception_ptr no_gil_failure;
  Py_BEGIN_ALLOW_THREADS
  try {
    dispatched = runtime->paged_decode(
        request,
        layer,
        static_cast<const uint16_t*>(query.view.buf),
        static_cast<size_t>(query.view.shape[0]) * 128,
        scale,
        &output,
        mutation.token(),
        &error);
  } catch (...) {
    // No C++ exception may cross CPython's GIL-release macros.  Translate
    // only after Py_END_ALLOW_THREADS has restored the thread state.
    no_gil_failure = std::current_exception();
  }
  Py_END_ALLOW_THREADS
  if (no_gil_failure) {
    std::rethrow_exception(no_gil_failure);
  }
  if (!dispatched) {
    // Failure counters are part of the observable runtime contract.  Keep
    // that native failure mutation even though no Python result is built.
    mutation.commit();
    if (error.find("finite BF16") != std::string::npos ||
        error.find("overflow") != std::string::npos ||
        error.find("value accumulation") != std::string::npos ||
        error == "scale must be finite") {
      value_error(error);
    } else {
      runtime_error(error);
    }
    return nullptr;
  }
  PyObject* result = result_bytes(
      "paged_decode",
      reinterpret_cast<const char*>(output.data()),
      static_cast<Py_ssize_t>(output.size() * sizeof(float)));
  if (result == nullptr) {
    return nullptr;
  }
  mutation.commit();
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_metrics(PageRuntimeObject* self, PyObject*) {
  PAGE_RUNTIME_PY_TRY
  if (self->runtime == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "page runtime is not initialized");
    return nullptr;
  }
  std::string error;
  const auto metrics = self->runtime->metrics(&error);
  if (!error.empty()) {
    runtime_error(error);
    return nullptr;
  }
  PyObject* result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }
#define SET_METRIC(name, value)                                                   \
  do {                                                                             \
    PyObject* metric_value = PyLong_FromUnsignedLongLong(value);                  \
    if (metric_value == nullptr || PyDict_SetItemString(result, name, metric_value) < 0) { \
      Py_XDECREF(metric_value);                                                   \
      Py_DECREF(result);                                                           \
      return nullptr;                                                             \
    }                                                                              \
    Py_DECREF(metric_value);                                                       \
  } while (false)
  SET_METRIC("resident_pages", metrics.resident_pages);
  SET_METRIC("referenced_pages", metrics.referenced_pages);
  SET_METRIC("shared_pages", metrics.shared_pages);
  SET_METRIC("evictable_pages", metrics.evictable_pages);
  SET_METRIC("free_pages", metrics.free_pages);
  SET_METRIC("requests", metrics.requests);
  SET_METRIC("prefixes", metrics.prefixes);
  SET_METRIC("page_allocations", metrics.page_allocations);
  SET_METRIC("request_allocations", metrics.request_allocations);
  SET_METRIC("prefix_allocations", metrics.prefix_allocations);
  SET_METRIC("cow_events", metrics.cow_events);
  SET_METRIC("evictions", metrics.evictions);
  SET_METRIC("releases", metrics.releases);
  SET_METRIC("cancellations", metrics.cancellations);
  SET_METRIC("append_tokens", metrics.append_tokens);
  SET_METRIC("dispatches", metrics.dispatches);
  SET_METRIC("dispatch_failures", metrics.dispatch_failures);
  SET_METRIC("native_dispatches", metrics.native_dispatches);
  SET_METRIC("native_failures", metrics.native_failures);
  SET_METRIC("buffer_allocations", metrics.buffer_allocations);
  SET_METRIC("decode_buffer_allocations", metrics.decode_buffer_allocations);
  SET_METRIC("query_copies", metrics.query_copies);
  SET_METRIC("output_copies", metrics.output_copies);
  SET_METRIC("metadata_copies", metrics.metadata_copies);
  SET_METRIC("metadata_bytes", metrics.metadata_bytes);
  SET_METRIC("kv_copy_bytes", metrics.kv_copy_bytes);
  SET_METRIC("kv_pool_copies", metrics.kv_pool_copies);
  SET_METRIC("attention_validation_bytes", metrics.attention_validation_bytes);
  SET_METRIC(
      "decode_page_resolution_checks",
      metrics.decode_page_resolution_checks);
  SET_METRIC("snapshot_failures", metrics.snapshot_failures);
  SET_METRIC("restore_failures", metrics.restore_failures);
  // Keep mode-qualified aliases aligned with the adapter/oracle metric
  // contract.  These are native counts, not a second dispatch counter.
  SET_METRIC("max_pages", metrics.max_pages);
  SET_METRIC("max_blocks_per_request", metrics.max_blocks_per_request);
  SET_METRIC("block_size", metrics.block_size);
  SET_METRIC("bytes_per_layer", metrics.bytes_per_layer);
  SET_METRIC("kv_bytes", metrics.kv_bytes);
#undef SET_METRIC
  PyObject* kv_dtype = PyUnicode_FromString("bfloat16");
  if (kv_dtype == nullptr || PyDict_SetItemString(result, "kv_dtype", kv_dtype) < 0) {
    Py_XDECREF(kv_dtype);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(kv_dtype);
  PyObject* shutdown = PyBool_FromLong(metrics.shutdown ? 1 : 0);
  if (shutdown == nullptr ||
      PyDict_SetItemString(result, "shutdown", shutdown) < 0) {
    Py_XDECREF(shutdown);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(shutdown);
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_shutdown(PageRuntimeObject* self, PyObject*) {
  PAGE_RUNTIME_PY_TRY
  if (self->runtime != nullptr) {
    self->runtime->shutdown();
  }
  Py_RETURN_NONE;
  PAGE_RUNTIME_PY_CATCH
}

PyObject* page_runtime_geometry(PageRuntimeObject* self, PyObject*) {
  PAGE_RUNTIME_PY_TRY
  if (!require_runtime(self)) {
    return nullptr;
  }
  PyObject* result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }
  const struct Entry {
    const char* name;
    uint32_t value;
  } entries[] = {
      {"num_layers", self->runtime->num_layers()},
      {"num_attention_heads", self->runtime->num_attention_heads()},
      {"num_key_value_heads", self->runtime->num_key_value_heads()},
      {"head_dim", self->runtime->head_dim()},
      {"block_size", self->runtime->block_size()},
      {"max_pages", self->runtime->max_pages()},
      {"max_blocks_per_request", self->runtime->max_blocks_per_request()},
      {"max_requests", self->runtime->max_requests()},
  };
  for (const auto& entry : entries) {
    PyObject* value = PyLong_FromUnsignedLong(entry.value);
    if (value == nullptr || PyDict_SetItemString(result, entry.name, value) < 0) {
      Py_XDECREF(value);
      Py_DECREF(result);
      return nullptr;
    }
    Py_DECREF(value);
  }
  return result;
  PAGE_RUNTIME_PY_CATCH
}

PyMethodDef page_runtime_methods[] = {
    {"allocate_request",
     _PyCFunction_CAST(page_runtime_allocate_request),
     METH_VARARGS | METH_KEYWORDS,
     "Allocate a request slot and return an opaque request handle."},
    {"allocate_pages",
     _PyCFunction_CAST(page_runtime_allocate_pages),
     METH_VARARGS | METH_KEYWORDS,
     "Allocate contiguous logical pages for a request."},
    {"append_kv",
     _PyCFunction_CAST(page_runtime_append_kv),
     METH_VARARGS | METH_KEYWORDS,
     "Append contiguous BF16 K/V tokens to one layer."},
    {"create_prefix",
     _PyCFunction_CAST(page_runtime_create_prefix),
     METH_VARARGS,
     "Create a shared read-only prefix handle from a request."},
    {"fork_prefix",
     _PyCFunction_CAST(page_runtime_fork_prefix),
     METH_VARARGS,
     "Fork a prefix handle while retaining page references."},
    {"attach_prefix",
     _PyCFunction_CAST(page_runtime_attach_prefix),
     METH_VARARGS,
     "Attach a shared prefix to an empty request."},
    {"release_prefix",
     _PyCFunction_CAST(page_runtime_release_prefix),
     METH_VARARGS,
     "Release a prefix handle."},
    {"snapshot",
     _PyCFunction_CAST(page_runtime_snapshot),
     METH_VARARGS | METH_KEYWORDS,
     "Fail closed until the persistent context-store package is available."},
    {"restore",
     _PyCFunction_CAST(page_runtime_restore),
     METH_VARARGS | METH_KEYWORDS,
     "Fail closed until the persistent context-store package is available."},
    {"release",
     _PyCFunction_CAST(page_runtime_release_request),
     METH_VARARGS | METH_KEYWORDS,
     "Release a request and its page references."},
    {"cancel",
     _PyCFunction_CAST(page_runtime_cancel),
     METH_VARARGS | METH_KEYWORDS,
     "Cancel and release a request."},
    {"release_request",
     _PyCFunction_CAST(page_runtime_release_request),
     METH_VARARGS | METH_KEYWORDS,
     "Release a request and its page references."},
    {"evict",
     _PyCFunction_CAST(page_runtime_evict),
     METH_VARARGS | METH_KEYWORDS,
     "Evict unreferenced pages by LRU order."},
    {"page_table",
     _PyCFunction_CAST(page_runtime_page_table),
     METH_VARARGS | METH_KEYWORDS,
     "Return the request's physical page table."},
    {"request_pages",
     _PyCFunction_CAST(page_runtime_request_pages),
     METH_VARARGS,
     "Return opaque handles for resident request pages."},
    {"sequence_length",
     _PyCFunction_CAST(page_runtime_sequence_length),
     METH_VARARGS,
     "Return the request sequence length."},
    {"paged_decode",
     _PyCFunction_CAST(page_runtime_paged_decode),
     METH_VARARGS | METH_KEYWORDS,
     "Dispatch the BF16 paged decode kernel for one request/layer."},
    {"paged_decode_attention",
     _PyCFunction_CAST(page_runtime_paged_decode),
     METH_VARARGS | METH_KEYWORDS,
     "Alias for paged_decode."},
    {"metrics",
     _PyCFunction_CAST(page_runtime_metrics),
     METH_NOARGS,
     "Return page ownership and lifecycle metrics."},
    {"geometry",
     _PyCFunction_CAST(page_runtime_geometry),
     METH_NOARGS,
     "Return the validated runtime geometry."},
    {"shutdown",
     _PyCFunction_CAST(page_runtime_shutdown),
     METH_NOARGS,
     "Deterministically release all runtime resources."},
    {nullptr, nullptr, 0, nullptr},
};

// Keep the type object zero-initialized in C++ and initialize its Python
// object header through the public setters below.  This avoids a
// ``-Wmissing-field-initializers`` warning while remaining valid across the
// supported CPython versions (3.10+).
PyTypeObject page_runtime_type = {};

}  // namespace

namespace metal_context {

int add_page_runtime_type(PyObject* module) {
  try {
    if (page_runtime_type.tp_name == nullptr) {
      PyObject* type_object = reinterpret_cast<PyObject*>(&page_runtime_type);
      Py_SET_REFCNT(type_object, 1);
      Py_SET_TYPE(type_object, &PyType_Type);
      Py_SET_SIZE(reinterpret_cast<PyVarObject*>(type_object), 0);
    }
    page_runtime_type.tp_name = "vllm_mlx._metal_context.PageRuntime";
    page_runtime_type.tp_basicsize = sizeof(PageRuntimeObject);
    page_runtime_type.tp_dealloc = reinterpret_cast<destructor>(page_runtime_dealloc);
    page_runtime_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    page_runtime_type.tp_doc = "Native Metal Context Engine page ownership runtime.";
    page_runtime_type.tp_methods = page_runtime_methods;
    page_runtime_type.tp_init = reinterpret_cast<initproc>(page_runtime_init);
    page_runtime_type.tp_new = page_runtime_new;
    if (PyType_Ready(&page_runtime_type) < 0) {
      return -1;
    }
    Py_INCREF(&page_runtime_type);
    if (PyModule_AddObject(
            module,
            "PageRuntime",
            reinterpret_cast<PyObject*>(&page_runtime_type)) < 0) {
      Py_DECREF(&page_runtime_type);
      return -1;
    }
  } catch (const std::bad_alloc& exception) {
    set_native_exception(exception);
    return -1;
  } catch (const std::exception& exception) {
    set_native_exception(exception);
    return -1;
  } catch (...) {
    set_unknown_native_exception();
    return -1;
  }
  return 0;
}

}  // namespace metal_context

#undef PAGE_RUNTIME_PY_TRY
#undef PAGE_RUNTIME_PY_CATCH
