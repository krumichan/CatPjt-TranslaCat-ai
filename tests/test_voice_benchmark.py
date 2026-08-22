import unittest

from scripts.benchmark_voice_v2 import percentile, summarize


class VoiceBenchmarkStatisticsTest(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self):
        values = [100.0, 200.0, 300.0, 400.0]
        self.assertEqual(percentile(values, 0.50), 250.0)
        self.assertEqual(percentile(values, 0), 100.0)
        self.assertEqual(percentile(values, 1), 400.0)

    def test_empty_summary_is_explicit(self):
        self.assertEqual(
            summarize([]),
            {
                "count": 0,
                "p50": None,
                "p95": None,
                "p99": None,
                "mean": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
