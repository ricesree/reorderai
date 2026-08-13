"""_forecast_series should return whole units bootstrapped from real history, not a flat ADS line."""

import pandas as pd

from api.services.detect_order_service import _forecast_series
from datetime import date


class _FakeStore:
    def __init__(self, history: list[float], multiplier: float = 1.0):
        self._history = history
        self._multiplier = multiplier

    def get_daily_series(self, item_id):
        return pd.Series(self._history, dtype=float)

    def daily_uplift_multipliers(self, item_id, dates, *, alt_ids=None, allowed_types=None):
        return [self._multiplier] * len(dates)


def test_forecast_values_are_whole_units_drawn_from_history():
    history = [0.0, 1.0, 2.0, 6.0, 7.0, 0.0, 3.0]
    store = _FakeStore(history)
    rows = _forecast_series(store, "item-1", [], ads=1.3444, x_days=10, as_of=date(2026, 8, 13), uplift_types=None)

    assert len(rows) == 10
    for row in rows:
        assert row["qty"] == float(int(row["qty"]))  # whole units
        assert row["qty"] in history  # only values seen in real history (before multiplier)

    # Not a flat line: with 7 possible historical values, 10 draws should not all match.
    assert len(set(r["qty"] for r in rows)) > 1


def test_zero_ads_returns_all_zero():
    store = _FakeStore([0.0])
    rows = _forecast_series(store, "item-2", [], ads=0.0, x_days=5, as_of=date(2026, 8, 13), uplift_types=None)
    assert all(r["qty"] == 0.0 for r in rows)


def test_no_history_falls_back_to_ads_constant():
    store = _FakeStore([])
    rows = _forecast_series(store, "item-3", [], ads=2.0, x_days=4, as_of=date(2026, 8, 13), uplift_types=None)
    assert all(r["qty"] == 2.0 for r in rows)


if __name__ == "__main__":
    test_forecast_values_are_whole_units_drawn_from_history()
    test_zero_ads_returns_all_zero()
    test_no_history_falls_back_to_ads_constant()
    print("ok")
