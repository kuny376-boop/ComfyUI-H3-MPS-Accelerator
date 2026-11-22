import unittest

from profiler import make_profile_override, report_session


class FakeTensor:
    def __init__(self, shape, device="mps:0", dtype="torch.bfloat16"):
        self.shape = shape
        self.device = device
        self.dtype = dtype


class ProfilerTests(unittest.TestCase):
    def test_preserves_result_and_records_geometry(self):
        q = FakeTensor((1, 128, 64))
        k = FakeTensor((1, 128, 64))
        v = FakeTensor((1, 128, 64))

        def dense(*_args, **_kwargs):
            return "result"

        wrapped = make_profile_override(session="test", previous=None)
        options = {"sol_block": 7, "sigmas": [2.5]}
        self.assertEqual(
            wrapped(dense, q, k, v, 8, transformer_options=options),
            "result",
        )
        report = report_session("test")
        self.assertIn('"attention_calls": 1', report)
        self.assertIn('"max_tokens": 128', report)
        self.assertIn('"7": 1', report)
        self.assertIn('"2.5": 1', report)
        self.assertIn('"mps:0": 1', report)
        self.assertIn('"torch.bfloat16": 1', report)

    def test_records_memory_peak(self):
        q = FakeTensor((1, 32, 16))
        values = iter((1024, 4096))

        def dense(*_args, **_kwargs):
            return "ok"

        wrapped = make_profile_override(
            session="memory",
            previous=None,
            memory_probe=lambda: next(values),
        )
        self.assertEqual(wrapped(dense, q, q, q, 2), "ok")
        # The profiler probes after the call; this first report therefore sees
        # the first supplied value.
        self.assertIn('"peak_mps_allocated_mb": 0.0', report_session("memory"))

    def test_chains_existing_override(self):
        q = FakeTensor((1, 64, 32))

        def dense(*_args, **_kwargs):
            return "dense"

        def previous(func, *args, **kwargs):
            self.assertIs(func, dense)
            return "previous"

        wrapped = make_profile_override(session="chain", previous=previous)
        self.assertEqual(wrapped(dense, q, q, q, 4), "previous")


if __name__ == "__main__":
    unittest.main()

