"""Network-free fixtures for historic TWSE / TPEx repair parsing."""
import unittest

from repair_2025 import number, parse_market, normalize_day


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


if __name__ == "__main__":
    unittest.main()
