// SPDX-License-Identifier: Apache-2.0

#include "page_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

int main() {
  using metal_context::page_runtime_sequence_buffer_offset;

  // The layout is [request slot][layer], not request*layer.  These cases
  // cover both nonzero dimensions and the layer-zero rows that the old
  // multiplication formula collapsed onto offset zero.
  assert(page_runtime_sequence_buffer_offset(0, 2, 0) == 0);
  assert(page_runtime_sequence_buffer_offset(0, 2, 1) == sizeof(int32_t));
  assert(page_runtime_sequence_buffer_offset(1, 2, 0) == 2 * sizeof(int32_t));
  assert(page_runtime_sequence_buffer_offset(1, 2, 1) == 3 * sizeof(int32_t));
  assert(page_runtime_sequence_buffer_offset(7, 4, 3) == 31 * sizeof(int32_t));

  bool rejected = false;
  try {
    (void)page_runtime_sequence_buffer_offset(0, 2, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
