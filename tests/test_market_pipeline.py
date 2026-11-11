import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from market_calendar import roc_to_iso, trading_status, is_closed_record
from market_cache import merge_prices, validate_and_mask_breadth, _write_cache, read_cache


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"Name": "農曆春節前最後交易日", "Date": "1150211"},
            {"Name": "市場無交易，僅辦理結算交割作業", "Date": "1150212"},
            {"Name": "農曆春節及除夕", "Date": "1150216"},
            {"Name": "農曆春節後開始交易日", "Date": "1150223"},
            {"Name": "國慶日", "Date": "1151009"},
        ]

    def test_dates(self):
        self.assertEqual(roc_to_iso("1150211"), "2026-02-11")
        for day, expected in [
            (date(2026, 2, 11), "open"),
            (date(2026, 2, 12), "closed"),
            (date(2026, 2, 16), "closed"),
            (date(2026, 2, 23), "open"),
            (date(2026, 10, 9), "closed"),
            (date(2026, 11, 14), "closed"),
            (date(2026, 11, 11), "open"),
        ]:
            with self.subTest(day=day):
                self.assertEqual(trading_status(day, self.rows)[0], expected)

    def test_missing_calendar_is_not_assumed_open(self):
        self.assertEqual(trading_status(date(2026, 11, 11), None)[0], "unknown")


class PriceTests(unittest.TestCase):
    def test_new_quotes_take_priority_but_nulls_do_not(self):
        idx = pd.to_datetime(["2026-11-09", "2026-11-10"])
        old = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=idx)
        fresh = pd.DataFrame({"a": [float("nan"), 9], "b": [7, float("nan")]}, index=idx)
        actual = merge_prices(old, fresh)
        self.assertEqual(actual.loc["2026-11-09", "a"], 1)
        self.assertEqual(actual.loc["2026-11-09", "b"], 7)
        self.assertEqual(actual.loc["2026-11-10", "a"], 9)
        self.assertEqual(actual.loc["2026-11-10", "b"], 4)

    def test_bad_1_of_640_denominator_becomes_missing(self):
        breadth = pd.Series([100.0, 45.0, 0.0])
        valid = pd.Series([1, 620, 0])
        masked, n = validate_and_mask_breadth(breadth, valid, 640)
        self.assertEqual(n, 2)
        self.assertTrue(pd.isna(masked.iloc[0]))
        self.assertEqual(masked.iloc[1], 45.0)

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quotes.csv.gz"
            frame = pd.DataFrame(
                {"2330.TW": [100.5, None], "2454.TW": [None, 200.25]},
                index=pd.to_datetime(["2026-11-10", "2026-11-11"]),
            )
            _write_cache(path, frame)
            after = read_cache(path)
            self.assertAlmostEqual(after.iloc[0, 0], 100.5)
            self.assertTrue(pd.isna(after.iloc[1, 0]))


if __name__ == "__main__":
    unittest.main()
