"""Source-only guard regression; no Metal device or MLX imports."""

import re
import unittest
from pathlib import Path


class ShaderGeometryContract(unittest.TestCase):
    def test_uniform_geometry_guard_precedes_scratch_and_barriers(self):
        root = Path(__file__).resolve().parents[1]
        shader = (root / "native/metal_context/kernels/paged_decode.metal").read_text()
        shader = re.sub(r"//[^\n]*", "", shader)
        self.assertIn("uint3 group_size [[threads_per_threadgroup]]", shader)
        guard = re.search(
            r"if\s*\(group_size\.x != 128 \|\| group_size\.y != 1 "
            r"\|\| group_size\.z != 1 \|\|\s*p\.query_heads == 0\)"
            r"\s*\{\s*return;\s*\}",
            shader,
        )
        self.assertIsNotNone(guard)
        for operation in (
            "tg_pos.x / p.query_heads",
            "threadgroup float partial_dot[128]",
            "partial_dot[tid] =",
            "threadgroup_barrier(",
        ):
            self.assertLess(guard.end(), shader.index(operation))
        self.assertNotRegex(shader, r"if\s*\([^)]*tid[^)]*\)\s*\{\s*return;")
        host = (root / "native/metal_context/src/python_module.mm").read_text()
        self.assertIn("constexpr uint32_t kThreadsPerThreadgroup = 128;", host)
        self.assertIn("MTLSizeMake(kThreadsPerThreadgroup, 1, 1)", host)


if __name__ == "__main__":
    unittest.main()
