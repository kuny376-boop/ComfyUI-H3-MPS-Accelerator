import unittest

from guard import validate_runtime


class GuardTests(unittest.TestCase):
    def test_safe_mps_h3(self):
        result = validate_runtime(
            platform="darwin",
            device="mps",
            is_h3=True,
            pytorch_attention=True,
        )
        self.assertTrue(result.ok)

    def test_rejects_missing_sdpa(self):
        result = validate_runtime(
            platform="darwin",
            device="mps",
            is_h3=True,
            pytorch_attention=False,
        )
        self.assertFalse(result.ok)
        self.assertIn("--use-pytorch-cross-attention", result.render())

    def test_rejects_forced_fp16(self):
        result = validate_runtime(
            platform="darwin",
            device="mps",
            is_h3=True,
            pytorch_attention=True,
            enabled_flags=("--force-fp16",),
        )
        self.assertFalse(result.ok)

    def test_windows_is_passthrough(self):
        result = validate_runtime(
            platform="win32",
            device="cuda",
            is_h3=True,
            pytorch_attention=False,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()

