from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


CATEGORY_COLUMNS = [
    "BAKERY",
    "CASHPOINT",
    "CONFECTIONARY",
    "DAIRY",
    "DELI",
    "DISCOUNT_BAKERY",
    "DRINKS",
    "FROZEN",
    "FRUIT_VEG",
    "GROCERY_FOOD",
    "GROCERY_HEALTH_PETS",
    "LOTTERY",
    "MEAT",
    "NEWSPAPERS_MAGAZINES",
    "PRACTICAL_ITEMS",
    "PREPARED_MEALS",
    "SEASONAL_GIFTING",
    "SOFT_DRINKS",
    "TOBACCO",
    "WORLD_FOODS",
]

ESSENTIALS_COLS = [
    "BAKERY",
    "DISCOUNT_BAKERY",
    "DAIRY",
    "FRUIT_VEG",
    "GROCERY_FOOD",
    "MEAT",
    "FROZEN",
]
INDULGENCE_COLS = ["CONFECTIONARY", "SOFT_DRINKS", "DRINKS", "SEASONAL_GIFTING"]
CONVENIENCE_COLS = ["DELI", "PREPARED_MEALS", "PRACTICAL_ITEMS", "CASHPOINT"]
LIFESTYLE_COLS = [
    "WORLD_FOODS",
    "GROCERY_HEALTH_PETS",
    "LOTTERY",
    "TOBACCO",
    "NEWSPAPERS_MAGAZINES",
]

FEATURES = [
    "recency_days",
    "baskets",
    "total_spend",
    "basket_quantity_mean",
    "basket_spend_mean",
    "essentials_ratio",
    "indulgence_ratio",
    "convenience_ratio",
    "lifestyle_ratio",
]

LOG_FEATURES = [
    "recency_days",
    "basket_spend_mean",
    "basket_quantity_mean",
    "total_spend",
    "baskets",
]

CUSTOMERS_REQUIRED = [
    "customer_number",
    "baskets",
    "total_quantity",
    "average_quantity",
    "total_spend",
    "average_spend",
]
CATEGORIES_REQUIRED = ["customer_number", *CATEGORY_COLUMNS]
BASKETS_REQUIRED = [
    "customer_number",
    "purchase_time",
    "basket_quantity",
    "basket_spend",
    "basket_categories",
]
LINEITEMS_REQUIRED = [
    "customer_number",
    "purchase_time",
    "product_id",
    "category",
    "quantity",
    "spend",
]

CUSTOMERS_NUMERIC = [
    "customer_number",
    "baskets",
    "total_quantity",
    "average_quantity",
    "total_spend",
    "average_spend",
]
CATEGORIES_NUMERIC = ["customer_number", *CATEGORY_COLUMNS]
BASKETS_NUMERIC = ["customer_number", "basket_quantity", "basket_spend"]
LINEITEMS_NUMERIC = ["customer_number", "product_id", "quantity", "spend"]


@dataclass
class SegmentationResult:
    output: pd.DataFrame
    export_data: pd.DataFrame
    training_data: pd.DataFrame
    cluster_table: pd.DataFrame
    segment_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    pca_scatter: pd.DataFrame
    segment_profile_scaled: pd.DataFrame
    k_diagnostics: pd.DataFrame
    category_percentage: pd.Series
    big4_percentage: pd.Series
    customers_per_day: pd.Series
    time_customers: pd.Series
    time_spend: pd.Series
    monthly_sales: pd.DataFrame
    top_item_categories: pd.DataFrame
    data_quality: pd.DataFrame
    stats: dict[str, Any]


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = _coerce_numeric_series(df[col])
    return df


def _drop_missing_customer_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["customer_number"]).copy()
    df["customer_number"] = df["customer_number"].astype("Int64")
    return df


def _prepare_baskets(baskets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baskets = baskets.copy()
    baskets["purchase_time"] = pd.to_datetime(baskets["purchase_time"], errors="coerce")
    baskets = baskets.dropna(subset=["purchase_time"])
    returns_baskets = baskets[
        (baskets["basket_spend"] < 0) | (baskets["basket_quantity"] < 0)
    ].copy()
    clean_baskets = baskets[
        (baskets["basket_spend"] >= 0) & (baskets["basket_quantity"] >= 0)
    ].copy()
    clean_baskets["basket_categories"] = (
        clean_baskets["basket_categories"].astype(str).fillna("")
    )
    return clean_baskets, returns_baskets


def _prepare_lineitems(lineitems: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = lineitems.copy()
    items["purchase_time"] = pd.to_datetime(items["purchase_time"], errors="coerce")
    returns_items = items[(items["spend"] < 0) | (items["quantity"] < 0)].copy()
    clean_items = items[(items["spend"] >= 0) & (items["quantity"] >= 0)].copy()
    return clean_items, returns_items


def _time_features(clean_baskets: pd.DataFrame) -> pd.DataFrame:
    analysis_date = clean_baskets["purchase_time"].max()

    recency = (
        clean_baskets.groupby("customer_number")["purchase_time"]
        .max()
        .reset_index()
    )
    recency["recency_days"] = (analysis_date - recency["purchase_time"]).dt.days
    recency = recency[["customer_number", "recency_days"]]

    ordered = clean_baskets.sort_values(["customer_number", "purchase_time"]).copy()
    ordered["prev_visit"] = ordered.groupby("customer_number")["purchase_time"].shift(1)
    ordered["days_between"] = (
        ordered["purchase_time"] - ordered["prev_visit"]
    ).dt.days
    visit_regularity = (
        ordered.groupby("customer_number")["days_between"]
        .std()
        .reset_index()
        .rename(columns={"days_between": "visit_gap_std"})
    )

    return recency.merge(visit_regularity, on="customer_number", how="left")


def _basket_stats(clean_baskets: pd.DataFrame) -> pd.DataFrame:
    return (
        clean_baskets.groupby("customer_number")
        .agg(
            basket_quantity_mean=("basket_quantity", "mean"),
            basket_quantity_std=("basket_quantity", "std"),
            basket_spend_mean=("basket_spend", "mean"),
            basket_spend_std=("basket_spend", "std"),
            basket_category_diversity=(
                "basket_categories",
                lambda x: x.astype(str).str.split(",").apply(len).mean(),
            ),
        )
        .reset_index()
    )


def _add_category_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["essentials_spend"] = df[ESSENTIALS_COLS].sum(axis=1)
    df["indulgence_spend"] = df[INDULGENCE_COLS].sum(axis=1)
    df["convenience_spend"] = df[CONVENIENCE_COLS].sum(axis=1)
    df["lifestyle_spend"] = df[LIFESTYLE_COLS].sum(axis=1)

    spend_base = df["total_spend"].replace(0, np.nan)
    df["essentials_ratio"] = df["essentials_spend"] / spend_base
    df["indulgence_ratio"] = df["indulgence_spend"] / spend_base
    df["convenience_ratio"] = df["convenience_spend"] / spend_base
    df["lifestyle_ratio"] = df["lifestyle_spend"] / spend_base
    ratio_cols = [
        "essentials_ratio",
        "indulgence_ratio",
        "convenience_ratio",
        "lifestyle_ratio",
    ]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _apply_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat in LOG_FEATURES:
        if feat in df.columns:
            df[feat] = np.log1p(df[feat].clip(lower=0))
    return df


def _stability_test(X: np.ndarray, k: int, n_runs: int = 10) -> float:
    cluster_labels = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        cluster_labels.append(km.fit_predict(X))

    ari_scores = []
    for idx in range(n_runs - 1):
        ari_scores.append(
            adjusted_rand_score(cluster_labels[idx], cluster_labels[idx + 1])
        )
    return float(np.mean(ari_scores))


def _build_k_diagnostics(X_pca: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(4, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X_pca)
        rows.append(
            {
                "k": k,
                "Inertia": float(model.inertia_),
                "Silhouette": float(silhouette_score(X_pca, labels)),
                "Stability_ARI": _stability_test(X_pca, k),
            }
        )
    return pd.DataFrame(rows)


def run_segmentation(
    customers: pd.DataFrame,
    categories: pd.DataFrame,
    baskets: pd.DataFrame,
    lineitems: pd.DataFrame,
    k_final: int = 5,
) -> SegmentationResult:
    validate_columns(customers, CUSTOMERS_REQUIRED, "customers")
    validate_columns(categories, CATEGORIES_REQUIRED, "categories")
    validate_columns(baskets, BASKETS_REQUIRED, "baskets")
    validate_columns(lineitems, LINEITEMS_REQUIRED, "lineitems")

    customers = customers.copy()
    categories = categories.copy()
    baskets = baskets.copy()
    lineitems = lineitems.copy()

    customers = _drop_missing_customer_numbers(
        _coerce_numeric_columns(customers, CUSTOMERS_NUMERIC)
    )
    categories = _drop_missing_customer_numbers(
        _coerce_numeric_columns(categories, CATEGORIES_NUMERIC)
    )
    baskets = _drop_missing_customer_numbers(
        _coerce_numeric_columns(baskets, BASKETS_NUMERIC)
    )
    lineitems = _drop_missing_customer_numbers(
        _coerce_numeric_columns(lineitems, LINEITEMS_NUMERIC)
    )

    clean_baskets, returns_baskets = _prepare_baskets(baskets)
    clean_items, returns_items = _prepare_lineitems(lineitems)

    time_features = _time_features(clean_baskets)
    basket_stats = _basket_stats(clean_baskets)

    base = customers.merge(categories, on="customer_number", how="left")
    training_data = base[base["baskets"] >= 2].copy()
    training_data = training_data.merge(time_features, on="customer_number", how="left")
    training_data = training_data.merge(basket_stats, on="customer_number", how="left")
    training_data = _add_category_ratios(training_data)
    training_data["visit_gap_std"] = training_data["visit_gap_std"].fillna(0)

    transformed_training = _apply_log_features(training_data)
    X = transformed_training[FEATURES].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=50)
    transformed_training["segment"] = kmeans.fit_predict(X_pca)
    training_data["segment"] = transformed_training["segment"].values

    f_scores, p_values = f_classif(X, transformed_training["segment"])
    feature_importance = pd.DataFrame(
        {"feature": FEATURES, "F_score": f_scores, "p_value": p_values}
    ).sort_values("F_score", ascending=False)

    export_data = base.copy()
    export_data = export_data.merge(time_features, on="customer_number", how="left")
    export_data = export_data.merge(basket_stats, on="customer_number", how="left")
    export_data = _add_category_ratios(export_data)
    export_data["visit_gap_std"] = export_data["visit_gap_std"].fillna(0)

    transformed_export = _apply_log_features(export_data)
    X_export = transformed_export[FEATURES].fillna(0)
    X_export_scaled = scaler.transform(X_export)
    X_export_pca = pca.transform(X_export_scaled)
    export_data["segment"] = kmeans.predict(X_export_pca)

    output = export_data[["customer_number", "segment"]].sort_values("customer_number")

    cluster_size = output["segment"].value_counts().sort_index()
    cluster_table = pd.DataFrame(
        {
            "segment": cluster_size.index,
            "number_of_customers": cluster_size.values,
            "percentage": (cluster_size.values / cluster_size.sum() * 100).round(2),
        }
    )

    segment_summary = (
        export_data.groupby("segment")[FEATURES].mean().assign(
            count=export_data.groupby("segment").size()
        )
    )

    pca_vis = PCA(n_components=2)
    X_2d = pca_vis.fit_transform(X_scaled)
    pca_scatter = pd.DataFrame(
        {
            "pc1": X_2d[:, 0],
            "pc2": X_2d[:, 1],
            "segment": transformed_training["segment"].values,
            "customer_number": training_data["customer_number"].values,
        }
    )

    segment_profile_scaled = export_data.groupby("segment")[FEATURES].mean()
    segment_profile_scaled = (
        segment_profile_scaled - segment_profile_scaled.mean()
    ) / segment_profile_scaled.std()

    category_percentage = (
        categories[CATEGORY_COLUMNS].sum().pipe(lambda s: (s / s.sum() * 100)).sort_values(ascending=False)
    )

    big4_percentage = (
        export_data[
            [
                "essentials_spend",
                "indulgence_spend",
                "convenience_spend",
                "lifestyle_spend",
            ]
        ]
        .sum()
        .pipe(lambda s: (s / s.sum() * 100).round(2))
    )

    ordered_baskets = clean_baskets.copy()
    ordered_baskets["day_of_week"] = ordered_baskets["purchase_time"].dt.day_name()
    ordered_baskets["hour"] = ordered_baskets["purchase_time"].dt.hour
    ordered_baskets["time_of_day"] = ordered_baskets["hour"].map(_time_of_day)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    customers_per_day = (
        ordered_baskets.groupby("day_of_week")["customer_number"]
        .nunique()
        .reindex(day_order)
        .fillna(0)
    )

    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    time_customers = (
        ordered_baskets.groupby("time_of_day")["customer_number"]
        .nunique()
        .reindex(time_order)
        .fillna(0)
    )
    time_spend = (
        ordered_baskets.groupby("time_of_day")["basket_spend"]
        .sum()
        .reindex(time_order)
        .fillna(0)
    )

    monthly_sales = (
        ordered_baskets.assign(
            year_month=ordered_baskets["purchase_time"].dt.to_period("M").astype(str)
        )
        .groupby("year_month", as_index=False)["basket_spend"]
        .sum()
        .rename(columns={"basket_spend": "sales"})
    )

    top_item_categories = (
        clean_items.groupby("category", as_index=False)
        .agg(total_spend=("spend", "sum"), total_quantity=("quantity", "sum"))
        .sort_values("total_spend", ascending=False)
        .head(10)
    )

    data_quality = pd.DataFrame(
        [
            {
                "metric": "input_customers",
                "value": int(customers["customer_number"].nunique()),
            },
            {
                "metric": "training_customers",
                "value": int(training_data["customer_number"].nunique()),
            },
            {
                "metric": "exported_customers",
                "value": int(output["customer_number"].nunique()),
            },
            {"metric": "basket_returns_removed", "value": int(len(returns_baskets))},
            {"metric": "lineitem_returns_removed", "value": int(len(returns_items))},
        ]
    )

    k_diagnostics = _build_k_diagnostics(X_pca)

    stats = {
        "total_customers": int(customers["customer_number"].nunique()),
        "segmented_customers": int(output["customer_number"].nunique()),
        "training_customers": int(training_data["customer_number"].nunique()),
        "n_segments": int(output["segment"].nunique()),
        "pca_components": int(pca.n_components_),
        "avg_basket_spend": float(clean_baskets["basket_spend"].mean()),
        "avg_basket_quantity": float(clean_baskets["basket_quantity"].mean()),
    }

    return SegmentationResult(
        output=output,
        export_data=export_data,
        training_data=training_data,
        cluster_table=cluster_table,
        segment_summary=segment_summary,
        feature_importance=feature_importance,
        pca_scatter=pca_scatter,
        segment_profile_scaled=segment_profile_scaled,
        k_diagnostics=k_diagnostics,
        category_percentage=category_percentage,
        big4_percentage=big4_percentage,
        customers_per_day=customers_per_day,
        time_customers=time_customers,
        time_spend=time_spend,
        monthly_sales=monthly_sales,
        top_item_categories=top_item_categories,
        data_quality=data_quality,
        stats=stats,
    )


def _time_of_day(hour: float | int | None) -> str:
    if pd.isna(hour):
        return "Night"
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def plot_segment_distribution(cluster_table: pd.DataFrame):
    fig = px.bar(
        cluster_table,
        x="segment",
        y="number_of_customers",
        color="segment",
        title="Segment Distribution",
        labels={"segment": "Segment", "number_of_customers": "Customers"},
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_pca_scatter(pca_scatter: pd.DataFrame):
    fig = px.scatter(
        pca_scatter,
        x="pc1",
        y="pc2",
        color="segment",
        hover_data=["customer_number"],
        title="Customer Segments in PCA Space",
        labels={"pc1": "Principal Component 1", "pc2": "Principal Component 2"},
    )
    return fig


def plot_segment_profile_heatmap(profile: pd.DataFrame):
    fig = px.imshow(
        profile,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Segment Behaviour Profile",
        labels={"x": "Features", "y": "Segment", "color": "Scaled Value"},
    )
    return fig


def plot_series_bar(series: pd.Series, title: str, xlabel: str, ylabel: str):
    frame = pd.DataFrame({"label": series.index.astype(str), "value": series.values})
    fig = px.bar(
        frame,
        x="label",
        y="value",
        title=title,
        labels={"label": xlabel, "value": ylabel},
    )
    fig.update_xaxes(tickangle=30)
    return fig


def plot_monthly_sales(monthly_sales: pd.DataFrame):
    fig = px.line(
        monthly_sales,
        x="year_month",
        y="sales",
        markers=True,
        title="Monthly Basket Sales",
        labels={"year_month": "Month", "sales": "Sales"},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_top_item_categories(top_item_categories: pd.DataFrame):
    fig = px.bar(
        top_item_categories,
        x="total_spend",
        y="category",
        orientation="h",
        title="Top Item Categories by Spend",
        labels={"total_spend": "Total Spend", "category": "Category"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig
