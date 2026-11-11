"""Network-free fixtures for historic TWSE / TPEx repair parsing."""
import unittest

from repair_2025 import number, parse_market, normalize_day, repair_span, fetch_day


class RepairParseTests(unittest.TestCase):
    def test_twse_price_table(self):
        body = {
            "date": "20250801",
            "tables": [
                {"fields": ["指數", "收盤指數"], "data": [["TAIEX", "22,000"]]},
                {"fields": ["證券代號", "證券名稱", "收盤價"], "data":
                 [[str(n), "stock", f"{100 + n/1000:.2f}"] for n in range(1100, 1250)]}
            ],
        }
        prices = parse_market(body, "2025-08-01", ".TW")
        self.assertEqual(len(prices), 150)
        self.assertIn("1101.TW", prices)

    def test_tpex_price_table(self):
        body = {"tables": [{"fields": ["代號", "名稱", "收盤"], "data":
            [[str(n), "stock", "20.00"] for n in range(3000, 3120)]}]}
        prices = parse_market(body, "2025-08-01", ".TWO")
        self.assertEqual(len(prices), 120)
        self.assertIn("3000.TWO", prices)

    def test_wrong_date_not_accepted(self):
        with self.assertRaises(ValueError):
            normalize_day({"date": "20250804"}, "2025-08-01")

    def test_empty_and_invalid_quote(self):
        self.assertTrue(number("--") != number("--"))
        self.assertEqual(number("1,234.50"), 1234.5)


    def test_backfill_includes_59_previous_trading_sessions(self):
        import pandas as pd
        dates = pd.bdate_range("2025-04-01", periods=170)
        target = dates[75:135]
        span = repair_span(dates, target)
        self.assertEqual(len(span),119)
        self.assertEqual(span[0], dates[16])
        self.assertEqual(span[-1], target[-1])

    def test_tpex_request_requires_EW(self):
        from unittest.mock import patch
        class Session:
            pass
        def mock_fetch(session, url, params):
            if "tpex" in url:
                self.assertEqual(params["type"], "EW")
                return {"date":"114/08/01","tables":[{"fields":["代號","收盤"],"data":[[str(n),"20.50"] for n in range(3000,3120)]}]}
            return {"date":"20250801","tables":[{"fields":["證券代號","收盤價"],"data":[[str(n),"100.00"] for n in range(1100,1800)]}]}
        with patch("repair_2025.fetch_json",side_effect=mock_fetch), patch("repair_2025.time.sleep"):
            prices,counts = fetch_day(Session(),"2025-08-01")
        self.assertGreaterEqual(counts["twse_count"],600)
        self.assertGreaterEqual(counts["tpex_count"],100)
        self.assertEqual(prices["3000.TWO"],20.5)

    def test_tpex_roc_date_rejected_when_wrong(self):
        with self.assertRaises(ValueError):
            normalize_day({"date":"114/08/04"},"2025-08-01")

if __name__ == "__main__":
    unittest.main()

