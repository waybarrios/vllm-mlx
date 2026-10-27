// SPDX-License-Identifier: Apache-2.0

#include "page_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
  metal_context::PageRuntime runtime(
      1, 1, 1, 128, 16, 2, 1, 1);
  std::string error;
  uint64_t request = 0;
  assert(runtime.allocate_request("fork-stress", 16, &request, &error));
  std::vector<uint64_t> pages;
  assert(runtime.allocate_pages(request, 1, &pages, &error));
  const uint16_t zero[128] = {};
  assert(runtime.append_kv(request, 0, zero, zero, 1, &error));

  uint64_t source = 0;
  assert(runtime.create_prefix(request, &source, &error));
  std::vector<uint64_t> forks;
  forks.reserve(4096);
  for (int index = 0; index < 4096; ++index) {
    uint64_t forked = 0;
    // Keeping every fork live forces repeated PrefixMeta vector growth.  The
    // source pointer must remain valid across each reallocation.
    assert(runtime.fork_prefix(source, &forked, &error));
    forks.push_back(forked);
  }
  for (uint64_t forked : forks) {
    assert(runtime.release_prefix(forked, &error));
  }
  assert(runtime.release_prefix(source, &error));
  assert(runtime.release_request(request, false, &error));
  runtime.shutdown();
  return 0;
}
