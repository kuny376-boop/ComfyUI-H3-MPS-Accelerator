import unittest

from profiler import make_profile_override, report_session


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class ProfilerTests(unittest.TestCase):
    def test_preserves_result_and_records_geometry(self):
        q = FakeTensor((1, 128, 64))
        k = FakeTensor((1, 128, 64))
        v = FakeTensor((1, 128, 64))

        def dense(*_args, **_kwargs):
            return "result"

        wrapped = make_profile_override(session="test", previous=None)
        self.assertEqual(wrapped(dense, q, k, v, 8), "result")
        report = report_session("test")
        self.assertIn('"attention_calls": 1', report)
        self.assertIn('"max_tokens": 128', report)

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

