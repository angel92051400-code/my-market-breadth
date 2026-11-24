import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_calendar import roc_to_iso, trading_status, is_closed_record, resolve_target_date
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

    def test_scheduled_run_delayed_past_midnight_uses_prior_session(self):
        from zoneinfo import ZoneInfo
        taipei = ZoneInfo("Asia/Taipei")
        self.assertEqual(
            resolve_target_date(datetime(2026,11,21,0,14,tzinfo=taipei),event_name="schedule"),
            date(2026,11,20)
        )
        self.assertEqual(
            resolve_target_date(datetime(2026,11,20,17,30,tzinfo=taipei),event_name="schedule"),
            date(2026,11,20)
        )
        self.assertEqual(
            resolve_target_date(datetime(2026,11,21,0,14,tzinfo=taipei),event_name="workflow_dispatch"),
            date(2026,11,21)
        )

    def test_manual_recent_backfill_date_rejects_old_or_future(self):
        from zoneinfo import ZoneInfo
        taipei=ZoneInfo("Asia/Taipei")
        now=datetime(2026,11,21,22,55,tzinfo=taipei)
        self.assertEqual(resolve_target_date(now,requested_date="2026-11-20"),date(2026,11,20))
        for wrong in ("2026-11-22","2026-11-12"):
            with self.assertRaises(ValueError):
                resolve_target_date(now,requested_date=wrong)

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

    def test_newly_added_stock_receives_full_history(self):
        from unittest.mock import patch
        from market_cache import load_stock_closes
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stock_closes.csv.gz"
            dates = pd.date_range("2026-06-01", periods=120, freq="B")
            names = [f"{i:04d}.TW" for i in range(1000, 1639)]
            old = pd.DataFrame({x: [100.0] * 120 for x in names}, index=dates)
            _write_cache(path, old)
            requests = []
            def fake_download(symbols, period):
                requests.append((tuple(symbols), period))
                subset = dates[-22:] if period == "1mo" else dates
                return pd.DataFrame({x: [100.0] * len(subset) for x in symbols}, index=subset)
            with patch("market_cache.CLOSE_PATH", path):
                result = load_stock_closes(names + ["B.TW"], fake_download)
            self.assertIn((("B.TW",), "2y"), requests)
            self.assertEqual(result["B.TW"].notna().sum(), 120)

    def test_missing_target_quotes_are_retried_and_published_only_if_complete(self):
        import json
        from unittest.mock import patch
        from market_cache import load_stock_closes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cache=root/"stock_closes.csv.gz"
            dates=pd.bdate_range("2026-06-01",periods=120)
            target=dates[-1]
            symbols=[f"{i:04d}.TW" for i in range(1000,1640)]
            old=pd.DataFrame(100.0,index=dates[:-1],columns=symbols)
            _write_cache(cache,old)
            attempts=[]
            def fake_download(names,period):
                attempts.append((tuple(names),period))
                if len(attempts)==1:
                    # 80 stocks missing on the intended session, 12.5%.
                    data=pd.DataFrame(110.0,index=dates[-4:],columns=names)
                    data.loc[target,names[-80:]]=float("nan")
                    # A future date with only 1 ticker MUST NOT change the
                    # coverage date and MUST be dropped from the output.
                    extra=pd.DataFrame(float("nan"),index=[target+pd.Timedelta(days=1)],columns=names)
                    extra.loc[:,names[0]]=200.0
                    return pd.concat([data,extra])
                return pd.DataFrame(111.0,index=[target],columns=names)
            with patch("market_cache.CLOSE_PATH",cache),patch("market_cache.CACHE_DIR",root):
                result=load_stock_closes(symbols,fake_download,
                    expected_date=target.strftime("%Y-%m-%d"),
                    priority_symbols=symbols)
            self.assertEqual(result.index[-1],target)
            self.assertEqual(len(attempts),2)
            self.assertEqual(len(attempts[1][0]),80)
            self.assertTrue(result.loc[target].notna().all())
            self.assertEqual(json.loads((root/"quote_diagnostics.json").read_text())["valid"],640)

    def test_quality_gate_stays_strict_and_saves_missing_symbols(self):
        import json
        from unittest.mock import patch
        from market_cache import load_stock_closes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cache=root/"stock_closes.csv.gz"
            dates=pd.bdate_range("2026-06-01",periods=120)
            symbols=[f"{i:04d}.TW" for i in range(1000,1640)]
            old=pd.DataFrame(100.0,index=dates[:-1],columns=symbols)
            _write_cache(cache,old)
            def empty_latest(names,period):
                return pd.DataFrame(110.0,index=dates[-5:-1],columns=names)
            with patch("market_cache.CLOSE_PATH",cache),patch("market_cache.CACHE_DIR",root):
                with self.assertRaisesRegex(RuntimeError,"報價不足"):
                    load_stock_closes(symbols,empty_latest,
                        expected_date=dates[-1].strftime("%Y-%m-%d"),
                        priority_symbols=symbols)
            d=json.loads((root/"quote_diagnostics.json").read_text())
            self.assertEqual(d["valid"],0)
            self.assertEqual(d["missing_count"],640)
            self.assertEqual(len(d["attempts"]),2)
            self.assertEqual(d["attempts"][0]["requested"],80)

    def test_targets_prior_day_not_newer_partial_row(self):
        from unittest.mock import patch
        from market_cache import load_stock_closes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            dates=pd.bdate_range("2026-06-01",periods=120)
            symbols=[f"{i:04d}.TW" for i in range(1000,1640)]
            data=pd.DataFrame(100.0,index=dates,columns=symbols)
            future=pd.DataFrame(float("nan"),index=[dates[-1]+pd.Timedelta(days=1)],columns=symbols)
            future.iloc[0,0]=99.0
            with patch("market_cache.CLOSE_PATH",root/"none.csv.gz"),patch("market_cache.CACHE_DIR",root):
                result=load_stock_closes(symbols,lambda names,period:pd.concat([data,future]),
                    expected_date=str(dates[-1].date()),priority_symbols=symbols)
            self.assertEqual(result.index[-1],dates[-1])

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
