import builtins
from pathlib import Path
import sys

from sales_forecasting_regression.data_layer import materialize_sales_data_layer


class FakeExpression:
    def __mul__(self, other):
        return self

    def alias(self, name: str):
        return self


class FakeSeries:
    def to_list(self):
        return [2003]


class FakeSelected:
    def unique(self):
        return self

    def to_series(self):
        return FakeSeries()


class FakeFrame:
    def __init__(self) -> None:
        self.columns = ["QUANTITYORDERED", "PRICEEACH"]

    def with_columns(self, expr):
        if "SALES" not in self.columns:
            self.columns.append("SALES")
        return self

    def select(self, name: str):
        return FakeSelected()

    def filter(self, expr):
        return self

    def write_parquet(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake parquet", encoding="utf-8")


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.commands = []

    def execute(self, query, params):
        self.commands.append((query, params))
        return None

    def close(self):
        self.closed = True


class FakeDuckDbModule:
    def __init__(self) -> None:
        self.connections = []

    def connect(self, _path: str):
        connection = FakeConnection()
        self.connections.append(connection)
        return connection


class FakePolarsModule:
    def read_csv(self, _path: Path, try_parse_dates: bool = True):
        return FakeFrame()

    def col(self, _name: str):
        return FakeExpression()


def test_data_layer_raises_runtime_error_when_dependencies_missing(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("QUANTITYORDERED,PRICEEACH\n1,2\n", encoding="utf-8")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"duckdb", "polars"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    raised = False
    try:
        materialize_sales_data_layer(
            csv_path=csv_path,
            parquet_dir=tmp_path / "parquet",
            duckdb_path=tmp_path / "duckdb" / "sales.duckdb",
        )
    except RuntimeError:
        raised = True
    assert raised


def test_data_layer_happy_path_with_fake_modules(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("QUANTITYORDERED,PRICEEACH\n1,2\n", encoding="utf-8")

    fake_duckdb = FakeDuckDbModule()
    fake_polars = FakePolarsModule()

    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)
    monkeypatch.setitem(sys.modules, "polars", fake_polars)

    result = materialize_sales_data_layer(
        csv_path=csv_path,
        parquet_dir=tmp_path / "parquet",
        duckdb_path=tmp_path / "duckdb" / "sales.duckdb",
    )

    assert Path(result["parquet_dir"]).exists()
    assert Path(result["duckdb_path"]).parent.exists()
    assert len(fake_duckdb.connections) == 1
    assert fake_duckdb.connections[0].closed
    assert fake_duckdb.connections[0].commands
