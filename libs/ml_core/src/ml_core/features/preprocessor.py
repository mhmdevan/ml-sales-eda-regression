from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_core.features.schema import FeatureSchema


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    schema.ensure_valid()
    transformers: list[tuple[str, object, list[str]]] = []
    if schema.numeric_features:
        transformers.append(("numeric", StandardScaler(), list(schema.numeric_features)))
    if schema.categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(schema.categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
