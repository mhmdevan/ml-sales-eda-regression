import pytest

from california_housing_template.data import CaliforniaHousingLoader


def test_loader_returns_synthetic_when_forced() -> None:
    frame = CaliforniaHousingLoader(random_state=42).load(force_synthetic=True)
    assert not frame.empty
    assert "MedHouseVal" in frame.columns


def test_loader_raises_when_fetch_fails_and_fallback_disabled(monkeypatch) -> None:
    def failing_fetch(*args, **kwargs):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr("california_housing_template.data.fetch_california_housing", failing_fetch)

    with pytest.raises(RuntimeError):
        CaliforniaHousingLoader(random_state=42).load(
            use_synthetic_if_fetch_fails=False,
            force_synthetic=False,
        )
