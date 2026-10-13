"""Incremental per-tenant sales cache: full pull once, then tail refetch + replace."""

import pandas as pd

from v2.forecasting import tenant_sales_cache as tsc


class _FakeDB:
    """Stands in for WecommDatabaseConnector; records the :since param of each sales query."""

    def __init__(self, sales: pd.DataFrame):
        self.sales = sales
        self.since_calls: list = []

    def read_sql(self, query, params=None):
        if "order_items" in query:
            since = (params or {}).get("since")
            self.since_calls.append(since)
            df = self.sales
            if since is not None:
                df = df[df["sale_date"] >= pd.Timestamp(since)]
            return df.copy()
        return pd.DataFrame(
            {"product_id": [1, 2], "category_id": [5, None], "category_name": ["Dairy", None],
             "list_price": [2.0, 3.0], "is_scale": [False, False], "pack_size": [1, 6]}
        )


def _sales(days=20, qty=2.0):
    dates = pd.date_range("2026-09-01", periods=days, freq="D")
    return pd.DataFrame({"product_id": 1, "sale_date": dates, "target_sales": qty})


def test_full_then_incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(tsc, "TENANT_MODELS_DIR", tmp_path)
    db = _FakeDB(_sales())
    first = tsc.update_sales_cache("t", connector=db)
    assert db.since_calls == [None]
    assert len(first) == 20

    # Tail edit: last day's qty changes (e.g. a return) — must replace, not double count.
    db.sales.loc[db.sales.index[-1], "target_sales"] = 1.0
    second = tsc.update_sales_cache("t", connector=db)
    assert str(db.since_calls[-1]) == "2026-09-13"  # last_date 09-20 minus 7 days
    assert len(second) == 20
    assert second["target_sales"].iloc[-1] == 1.0


def test_corrupt_cache_falls_back_to_full_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(tsc, "TENANT_MODELS_DIR", tmp_path)
    path = tsc._cache_path("t")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    db = _FakeDB(_sales())
    assert len(tsc.update_sales_cache("t", connector=db)) == 20
    assert db.since_calls == [None]


def test_panel_is_dense_and_zero_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(tsc, "TENANT_MODELS_DIR", tmp_path)
    panel = tsc.load_tenant_sales_panel("t", connector=_FakeDB(_sales(days=10)))
    assert len(panel) == 2 * 10  # 2 products × 10 days
    assert panel.loc[panel["product_id"] == "2", "target_sales"].sum() == 0
    assert set(panel.columns) >= {"date", "product_id", "target_sales", "category_id", "is_perishable"}
