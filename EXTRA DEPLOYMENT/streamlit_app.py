from __future__ import annotations

import pandas as pd
import streamlit as st

from segmentation_pipeline import (
    dataframe_to_csv_bytes,
    plot_monthly_sales,
    plot_pca_scatter,
    plot_segment_distribution,
    plot_segment_profile_heatmap,
    plot_series_bar,
    plot_top_item_categories,
    run_segmentation,
)


st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv_file(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=True)
def run_pipeline(
    customers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    baskets_df: pd.DataFrame,
    lineitems_df: pd.DataFrame,
):
    return run_segmentation(customers_df, categories_df, baskets_df, lineitems_df)


st.title("Customer Segmentation Dashboard")
st.caption(
    "Upload the 4 CSV files, run segmentation, review the key charts, and download the output file."
)

with st.sidebar:
    st.header("Upload CSV Files")
    customers_file = st.file_uploader("Customers CSV", type="csv", key="customers")
    categories_file = st.file_uploader("Category Spend CSV", type="csv", key="categories")
    baskets_file = st.file_uploader("Baskets CSV", type="csv", key="baskets")
    lineitems_file = st.file_uploader("Line Items CSV", type="csv", key="lineitems")
    run_clicked = st.button("Run Segmentation", type="primary", use_container_width=True)


if not all([customers_file, categories_file, baskets_file, lineitems_file]):
    st.info("Upload all 4 CSV files to enable the pipeline.")
    st.stop()


customers_df = read_csv_file(customers_file)
categories_df = read_csv_file(categories_file)
baskets_df = read_csv_file(baskets_file)
lineitems_df = read_csv_file(lineitems_file)

st.subheader("Input Preview")
preview_cols = st.columns(4)
preview_cols[0].metric("Customers Rows", f"{len(customers_df):,}")
preview_cols[1].metric("Category Rows", f"{len(categories_df):,}")
preview_cols[2].metric("Basket Rows", f"{len(baskets_df):,}")
preview_cols[3].metric("Line Item Rows", f"{len(lineitems_df):,}")

if run_clicked:
    try:
        st.session_state["segmentation_result"] = run_pipeline(
            customers_df, categories_df, baskets_df, lineitems_df
        )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

if "segmentation_result" not in st.session_state:
    st.stop()

result = st.session_state["segmentation_result"]

stats_cols = st.columns(5)
stats_cols[0].metric("Input Customers", f"{result.stats['total_customers']:,}")
stats_cols[1].metric("Segmented Customers", f"{result.stats['segmented_customers']:,}")
stats_cols[2].metric("Training Customers", f"{result.stats['training_customers']:,}")
stats_cols[3].metric("Segments", result.stats["n_segments"])
stats_cols[4].metric("PCA Components", result.stats["pca_components"])

st.download_button(
    label="Download Output CSV",
    data=dataframe_to_csv_bytes(result.output),
    file_name="customer_segments_output.csv",
    mime="text/csv",
)

tab_output, tab_segments, tab_patterns, tab_categories, tab_quality = st.tabs(
    ["Output", "Segments", "Shopping Patterns", "Category Mix", "Data Quality"]
)

with tab_output:
    st.subheader("Output File")
    st.dataframe(result.output, use_container_width=True, height=420)
    st.subheader("Segment Counts")
    st.dataframe(result.cluster_table, use_container_width=True)

with tab_segments:
    seg_left, seg_right = st.columns(2)
    with seg_left:
        st.plotly_chart(plot_segment_distribution(result.cluster_table), use_container_width=True)
    with seg_right:
        st.plotly_chart(plot_pca_scatter(result.pca_scatter), use_container_width=True)

    st.subheader("Segment Feature Profile")
    st.plotly_chart(
        plot_segment_profile_heatmap(result.segment_profile_scaled),
        use_container_width=True,
    )

    info_left, info_right = st.columns(2)
    with info_left:
        st.subheader("Feature Importance")
        st.dataframe(result.feature_importance, use_container_width=True)
    with info_right:
        st.subheader("k Diagnostics")
        st.dataframe(result.k_diagnostics, use_container_width=True)

with tab_patterns:
    pattern_left, pattern_right = st.columns(2)
    with pattern_left:
        st.plotly_chart(
            plot_series_bar(
                result.customers_per_day,
                "Customers by Day of Week",
                "Day",
                "Unique Customers",
            ),
            use_container_width=True,
        )
    with pattern_right:
        st.plotly_chart(
            plot_series_bar(
                result.time_customers,
                "Customers by Time of Day",
                "Time of Day",
                "Unique Customers",
            ),
            use_container_width=True,
        )

    pattern_bottom_left, pattern_bottom_right = st.columns(2)
    with pattern_bottom_left:
        st.plotly_chart(
            plot_series_bar(
                result.time_spend,
                "Spend by Time of Day",
                "Time of Day",
                "Basket Spend",
            ),
            use_container_width=True,
        )
    with pattern_bottom_right:
        st.plotly_chart(plot_monthly_sales(result.monthly_sales), use_container_width=True)

with tab_categories:
    cat_left, cat_right = st.columns(2)
    with cat_left:
        st.plotly_chart(
            plot_series_bar(
                result.category_percentage.head(10),
                "Top Category Share",
                "Category",
                "Spend %",
            ),
            use_container_width=True,
        )
    with cat_right:
        st.plotly_chart(
            plot_series_bar(
                result.big4_percentage,
                "Big 4 Category Mix",
                "Group",
                "Spend %",
            ),
            use_container_width=True,
        )

    st.plotly_chart(plot_top_item_categories(result.top_item_categories), use_container_width=True)

with tab_quality:
    st.subheader("Pipeline Checks")
    st.dataframe(result.data_quality, use_container_width=True)
    st.subheader("Segment Summary")
    st.dataframe(result.segment_summary, use_container_width=True)
