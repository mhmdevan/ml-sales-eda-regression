from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_regression

from california_housing_template.config import CALIFORNIA_FEATURES, CALIFORNIA_TARGET, california_schema


class CaliforniaHousingLoader:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def load(
        self,
        use_synthetic_if_fetch_fails: bool = True,
        force_synthetic: bool = False,
    ) -> pd.DataFrame:
        if force_synthetic:
            frame = self.synthetic_frame(4000)
        else:
            try:
                dataset = fetch_california_housing(as_frame=True)
                frame = dataset.frame.copy()
            except Exception:
                if not use_synthetic_if_fetch_fails:
                    raise
                frame = self.synthetic_frame(4000)

        schema = california_schema()
        schema.validate_frame(frame)

        for feature in CALIFORNIA_FEATURES + [CALIFORNIA_TARGET]:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            frame[feature] = frame[feature].fillna(frame[feature].median())

        return frame.reset_index(drop=True)

    def synthetic_frame(self, n_rows: int) -> pd.DataFrame:
        x, y = make_regression(
            n_samples=n_rows,
            n_features=len(CALIFORNIA_FEATURES),
            n_informative=len(CALIFORNIA_FEATURES),
            noise=10.0,
            random_state=self.random_state,
        )
        rng = np.random.default_rng(self.random_state)
        frame = pd.DataFrame(x, columns=CALIFORNIA_FEATURES)
        frame["Latitude"] = rng.uniform(32.0, 42.0, size=n_rows)
        frame["Longitude"] = rng.uniform(-124.0, -114.0, size=n_rows)
        frame[CALIFORNIA_TARGET] = y / 100.0 + 2.0
        return frame
