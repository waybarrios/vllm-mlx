// SPDX-License-Identifier: Apache-2.0
// Deterministic host/ASAN regression for the PageRuntime lock contract.

#include "page_runtime.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int main() {
  using metal_context::PageRuntime;

  PageRuntime runtime(
      /*num_layers=*/4,
      /*num_attention_heads=*/1,
      /*num_key_value_heads=*/1,
      /*head_dim=*/128,
      /*block_size=*/16,
      /*max_pages=*/8,
      /*max_blocks_per_request=*/4,
      /*max_requests=*/1);
  std::string error;
  uint64_t request = 0;
  if (!runtime.allocate_request("lock-order", 64, &request, &error)) {
    std::fprintf(stderr, "allocate_request failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<uint16_t> kv(128, 0);
  std::vector<uint16_t> query(128, 0);
  constexpr uint32_t kWorkers = 4;
  constexpr uint32_t kIterations = 8;
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<uint32_t> completed{0};
  std::atomic<uint32_t> failures{0};
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  std::vector<std::thread> workers;
  workers.reserve(kWorkers);

  for (uint32_t layer = 0; layer < kWorkers; ++layer) {
    workers.emplace_back([&, layer] {
      try {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
          std::string append_error;
          if (!runtime.append_kv(
                  request,
                  layer,
                  kv.data(),
                  kv.data(),
                  1,
                  &append_error)) {
            ++failures;
          }
          std::vector<float> output;
          std::string decode_error;
          // A host-only build is expected to fail closed before dispatch;
          // the lock/lifecycle assertion is that this call always returns.
          (void)runtime.paged_decode(
              request,
              layer,
              query.data(),
              query.size(),
              1.0f,
              &output,
              &decode_error);
        }
      } catch (...) {
        ++failures;
      }
      completed.fetch_add(1, std::memory_order_release);
      completion_cv.notify_one();
    });
  }

  while (ready.load(std::memory_order_acquire) != kWorkers) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    if (!completion_cv.wait_for(
            lock,
            std::chrono::seconds(5),
            [&] {
              return completed.load(std::memory_order_acquire) == kWorkers;
            })) {
      std::fprintf(stderr, "PageRuntime lock-order regression timed out\n");
      std::_Exit(2);
    }
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  if (failures.load(std::memory_order_acquire) != 0) {
    std::fprintf(stderr, "PageRuntime concurrency worker failed\n");
    return 1;
  }

  if (!runtime.release_request(request, false, &error)) {
    std::fprintf(stderr, "release_request failed: %s\n", error.c_str());
    return 1;
  }
  runtime.shutdown();
  return 0;
}
