from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Data Observability Hub",
    layout="wide",
)

sns.set_style("whitegrid")

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")


def reset_working_data(df: pd.DataFrame) -> None:
    """Store both original and working copies in Streamlit session state."""
    st.session_state.original_df = df.copy()
    st.session_state.cleaned_df = df.copy()
    st.session_state.last_uploaded_name = "loaded"


def get_current_df() -> pd.DataFrame:
    """Return the current working DataFrame."""
    return st.session_state.cleaned_df.copy()

def get_dataset_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return beginner-friendly dataset information."""
    info_rows = []
    for column in df.columns:
        info_rows.append(
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Non-Null Count": int(df[column].notna().sum()),
                "Null Count": int(df[column].isna().sum()),
                "Unique Values": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(info_rows)


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing count and percentage for every column."""
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100) if len(df) > 0 else 0

    summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": missing_count.values,
            "Missing %": np.round(missing_percent.values, 2),
        }
    )
    return summary.sort_values(by="Missing %", ascending=False).reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> Tuple[int, float]:
    """Return duplicate row count and percentage."""
    if len(df) == 0:
        return 0, 0.0

    duplicate_count = int(df.duplicated().sum())
    duplicate_percent = duplicate_count / len(df) * 100
    return duplicate_count, duplicate_percent


def detect_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect outliers in numeric columns using the IQR rule."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return pd.DataFrame(columns=["Column", "Outlier Count", "Outlier %"]), pd.Series(False, index=df.index)

    outlier_rows = pd.Series(False, index=df.index)
    summary_rows = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            summary_rows.append({"Column": column, "Outlier Count": 0, "Outlier %": 0.0})
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            mask = pd.Series(False, index=df.index)
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            mask = mask.fillna(False)

        outlier_rows = outlier_rows | mask
        outlier_count = int(mask.sum())
        outlier_percent = (outlier_count / len(df) * 100) if len(df) > 0 else 0.0
        summary_rows.append(
            {
                "Column": column,
                "Outlier Count": outlier_count,
                "Outlier %": round(outlier_percent, 2),
            }
        )

    return pd.DataFrame(summary_rows), outlier_rows


def detect_type_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect common data type mismatch patterns.
    This uses simple rules so the logic is easy to understand.
    """
    mismatch_rows = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        if non_null.empty:
            continue

        if series.dtype == "object":
            python_types = non_null.map(lambda x: type(x).__name__).value_counts()

            if len(python_types) > 1:
                mismatch_rows.append(
                    {
                        "Column": column,
                        "Issue": "Mixed Python value types found",
                        "Details": ", ".join([f"{idx}: {val}" for idx, val in python_types.items()]),
                    }
                )

            numeric_like = pd.to_numeric(non_null, errors="coerce")
            numeric_like_ratio = numeric_like.notna().mean()
            if 0.5 <= numeric_like_ratio < 1.0:
                mismatch_rows.append(
                    {
                        "Column": column,
                        "Issue": "Partially numeric text column",
                        "Details": f"{numeric_like_ratio * 100:.1f}% of non-null values look numeric.",
                    }
                )

            datetime_like = pd.to_datetime(non_null, errors="coerce")
            datetime_like_ratio = datetime_like.notna().mean()
            if 0.5 <= datetime_like_ratio < 1.0:
                mismatch_rows.append(
                    {
                        "Column": column,
                        "Issue": "Partially date-like text column",
                        "Details": f"{datetime_like_ratio * 100:.1f}% of non-null values look like dates.",
                    }
                )

    return pd.DataFrame(mismatch_rows)


def detect_class_imbalance(df: pd.DataFrame, target_column: Optional[str]) -> Optional[pd.DataFrame]:
    """Check class distribution for a chosen target column."""
    if not target_column or target_column not in df.columns:
        return None

    value_counts = df[target_column].value_counts(dropna=False)
    if value_counts.empty:
        return None

    imbalance_df = pd.DataFrame(
        {
            "Class": value_counts.index.astype(str),
            "Count": value_counts.values,
            "Percentage": np.round(value_counts.values / len(df) * 100, 2),
        }
    )
    return imbalance_df


def calculate_quality_score(
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    duplicate_count: int,
    outlier_mask: pd.Series,
) -> Tuple[float, List[str]]:
    """
    Calculate a simple 0-10 quality score.
    Higher missing values, duplicates, and outliers reduce the score.
    """
    if len(df) == 0:
        return 0.0, ["The dataset is empty, so the quality score is 0/10."]

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0
    duplicate_ratio = duplicate_count / len(df)
    outlier_ratio = outlier_mask.sum() / len(df)

    penalty = (missing_ratio * 4.0) + (duplicate_ratio * 3.0) + (outlier_ratio * 3.0)
    score = max(0.0, min(10.0, 10.0 - penalty * 10))

    explanation = [
        f"Missing values penalty: {missing_ratio * 100:.2f}% of all cells are missing.",
        f"Duplicate penalty: {duplicate_ratio * 100:.2f}% of rows are duplicates.",
        f"Outlier penalty: {outlier_ratio * 100:.2f}% of rows contain at least one numeric outlier.",
    ]

    return round(score, 2), explanation


def generate_ai_insights(
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    duplicate_count: int,
    outlier_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    imbalance_df: Optional[pd.DataFrame],
) -> List[str]:
    """Create human-readable, rule-based recommendations."""
    insights = []

    high_missing = missing_df[missing_df["Missing %"] >= 30]
    medium_missing = missing_df[(missing_df["Missing %"] > 0) & (missing_df["Missing %"] < 30)]

    for _, row in high_missing.iterrows():
        insights.append(
            f"Column '{row['Column']}' has {row['Missing %']:.2f}% missing values. Consider dropping it or filling values carefully based on business meaning."
        )

    for _, row in medium_missing.head(5).iterrows():
        column_name = row["Column"]
        if pd.api.types.is_numeric_dtype(df[column_name]):
            insights.append(
                f"Column '{column_name}' has {row['Missing %']:.2f}% missing values. Filling with median is often safer for numeric data."
            )
        else:
            insights.append(
                f"Column '{column_name}' has {row['Missing %']:.2f}% missing values. Filling with mode may be a simple starting point."
            )

    if duplicate_count > 0:
        insights.append(
            f"The dataset contains {duplicate_count} duplicate rows. Removing duplicates can improve training and reporting quality."
        )

    notable_outliers = outlier_df[outlier_df["Outlier Count"] > 0].sort_values(by="Outlier Count", ascending=False)
    for _, row in notable_outliers.head(5).iterrows():
        insights.append(
            f"Column '{row['Column']}' contains {int(row['Outlier Count'])} outliers based on the IQR rule. Review whether these are real extreme values or data errors."
        )

    for _, row in mismatch_df.head(5).iterrows():
        insights.append(
            f"Column '{row['Column']}' may have a data type issue: {row['Issue']}. {row['Details']}"
        )

    if imbalance_df is not None and not imbalance_df.empty:
        top_percent = imbalance_df["Percentage"].max()
        if top_percent > 70:
            dominant_class = imbalance_df.sort_values(by="Percentage", ascending=False).iloc[0]["Class"]
            insights.append(
                f"Target column appears imbalanced. Class '{dominant_class}' makes up {top_percent:.2f}% of the data, so model bias may be a risk."
            )

    if not insights:
        insights.append("No major data quality issues were detected by the current rule-based checks.")

    return insights


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    removed = before - len(cleaned)
    return cleaned, removed


def fill_missing_values(df: pd.DataFrame, columns: List[str], strategy: str) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = df.copy()
    updated_columns = []

    for column in columns:
        if column not in cleaned.columns or cleaned[column].isna().sum() == 0:
            continue

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
                updated_columns.append(column)
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
                updated_columns.append(column)
        elif strategy == "mode":
            mode_series = cleaned[column].mode(dropna=True)
            if not mode_series.empty:
                cleaned[column] = cleaned[column].fillna(mode_series.iloc[0])
                updated_columns.append(column)

    return cleaned, updated_columns


def drop_high_null_columns(df: pd.DataFrame, threshold_percent: float) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = df.copy()
    missing_pct = cleaned.isna().mean() * 100
    columns_to_drop = missing_pct[missing_pct >= threshold_percent].index.tolist()
    cleaned = cleaned.drop(columns=columns_to_drop)
    return cleaned, columns_to_drop


def remove_outliers(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    if not columns:
        return cleaned, 0

    combined_mask = pd.Series(False, index=cleaned.index)

    for column in columns:
        if column not in cleaned.columns or not pd.api.types.is_numeric_dtype(cleaned[column]):
            continue

        series = cleaned[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)
        combined_mask = combined_mask | mask.fillna(False)

    removed_count = int(combined_mask.sum())
    cleaned = cleaned.loc[~combined_mask].copy()
    return cleaned, removed_count


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def plot_missing_heatmap(df: pd.DataFrame):
    sample_df = df.copy()
    if len(sample_df) > 200:
        sample_df = sample_df.head(200)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(sample_df.isna(), cbar=False, yticklabels=False, cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Values Heatmap (up to first 200 rows)")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    st.pyplot(fig)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#1f77b4")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    st.pyplot(fig)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are needed for a correlation heatmap.")
        return

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("Data Observability Hub")
    st.caption("Upload a dataset, detect quality issues, clean the data, and download the improved result.")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

        editable_table = st.checkbox("Enable editable dataframe", value=True)
        target_column_choice = None

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file from the sidebar to begin.")
        st.markdown(
            """
            ### App Sections
            - Upload
            - Overview
            - Issues
            - Visualization
            - Cleaning
            - Download
            """
        )
        return

    try:
        loaded_df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    if (
        "cleaned_df" not in st.session_state
        or "original_df" not in st.session_state
        or st.session_state.get("uploaded_name") != uploaded_file.name
    ):
        reset_working_data(loaded_df)
        st.session_state.uploaded_name = uploaded_file.name

    current_df = get_current_df()

    with st.sidebar:
        st.subheader("Analysis Options")
        target_column_options = ["None"] + current_df.columns.tolist()
        target_column_choice = st.selectbox("Optional target column", target_column_options, index=0)
        if target_column_choice == "None":
            target_column_choice = None

        if st.button("Reset to original data", use_container_width=True):
            st.session_state.cleaned_df = st.session_state.original_df.copy()
            st.rerun()
    st.header("Upload")
    st.subheader("Dataset Preview")
    if editable_table:
        edited_df = st.data_editor(current_df, use_container_width=True, num_rows="dynamic")
        st.session_state.cleaned_df = edited_df.copy()
        current_df = edited_df.copy()
    else:
        st.dataframe(current_df, use_container_width=True)
    info_df = get_dataset_info(current_df)
    missing_df = missing_values_summary(current_df)
    duplicate_count, duplicate_percent = duplicate_summary(current_df)
    outlier_df, outlier_mask = detect_outliers_iqr(current_df)
    mismatch_df = detect_type_mismatches(current_df)
    imbalance_df = detect_class_imbalance(current_df, target_column_choice)
    quality_score, quality_explanation = calculate_quality_score(
        current_df, missing_df, duplicate_count, outlier_mask
    )
    insights = generate_ai_insights(
        current_df, missing_df, duplicate_count, outlier_df, mismatch_df, imbalance_df
    )

    upload_col1, upload_col2, upload_col3 = st.columns(3)
    upload_col1.metric("Rows", f"{current_df.shape[0]}")
    upload_col2.metric("Columns", f"{current_df.shape[1]}")
    upload_col3.metric("Quality Score", f"{quality_score}/10")

    st.header("Overview")
    overview_col1, overview_col2 = st.columns([1, 2])

    with overview_col1:
        st.subheader("Dataset Summary")
        st.write(f"Rows: **{current_df.shape[0]}**")
        st.write(f"Columns: **{current_df.shape[1]}**")
        st.write(f"Duplicate Rows: **{duplicate_count}** ({duplicate_percent:.2f}%)")
        st.write(f"Rows With Outliers: **{int(outlier_mask.sum())}**")

        st.subheader("Quality Score Explanation")
        for line in quality_explanation:
            st.write(f"- {line}")

    with overview_col2:
        st.subheader("Column Information")
        st.dataframe(info_df, use_container_width=True)
    st.header("Issues")
    issues_tab1, issues_tab2, issues_tab3, issues_tab4, issues_tab5 = st.tabs(
        ["Missing Values", "Duplicates", "Outliers", "Type Mismatches", "AI Insights"]
    )

    with issues_tab1:
        st.subheader("Missing Values Summary")
        st.dataframe(missing_df, use_container_width=True)

    with issues_tab2:
        st.subheader("Duplicate Rows")
        st.write(f"Duplicate row count: **{duplicate_count}**")
        if duplicate_count > 0:
            st.dataframe(current_df[current_df.duplicated()].head(20), use_container_width=True)
        else:
            st.success("No duplicate rows detected.")

    with issues_tab3:
        st.subheader("Outlier Detection (IQR Method)")
        if outlier_df.empty:
            st.info("No numeric columns available for outlier detection.")
        else:
            st.dataframe(outlier_df, use_container_width=True)

    with issues_tab4:
        st.subheader("Data Type Mismatch Detection")
        if mismatch_df.empty:
            st.success("No obvious type mismatch issues were detected by the simple rules.")
        else:
            st.dataframe(mismatch_df, use_container_width=True)

        if imbalance_df is not None:
            st.subheader("Class Imbalance Detection")
            st.dataframe(imbalance_df, use_container_width=True)

    with issues_tab5:
        st.subheader("AI Insights")
        for insight in insights:
            st.write(f"- {insight}")

    st.header("Visualization")
    vis_tab1, vis_tab2, vis_tab3 = st.tabs(
        ["Missing Values Heatmap", "Distribution Plots", "Correlation Heatmap"]
    )

    with vis_tab1:
        plot_missing_heatmap(current_df)

    with vis_tab2:
        numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_numeric_column = st.selectbox("Select a numeric column", numeric_columns)
            plot_distribution(current_df, selected_numeric_column)
        else:
            st.info("No numeric columns available for distribution plots.")

    with vis_tab3:
        plot_correlation_heatmap(current_df)

    # Cleaning
    st.header("Cleaning")
    clean_col1, clean_col2 = st.columns(2)

    with clean_col1:
        st.subheader("Duplicate Cleaning")
        if st.button("Remove duplicate rows", use_container_width=True):
            updated_df, removed = remove_duplicates(st.session_state.cleaned_df)
            st.session_state.cleaned_df = updated_df
            st.success(f"Removed {removed} duplicate rows.")
            st.rerun()

        st.subheader("Missing Value Handling")
        fill_columns = st.multiselect(
            "Select columns to fill missing values",
            options=st.session_state.cleaned_df.columns.tolist(),
        )
        fill_strategy = st.selectbox("Select fill strategy", ["mean", "median", "mode"])
        if st.button("Fill missing values", use_container_width=True):
            updated_df, updated_columns = fill_missing_values(
                st.session_state.cleaned_df, fill_columns, fill_strategy
            )
            st.session_state.cleaned_df = updated_df
            if updated_columns:
                st.success(f"Filled missing values in: {', '.join(updated_columns)}")
                st.rerun()
            else:
                st.warning("No columns were updated. Mean and median only work on numeric columns.")

    with clean_col2:
        st.subheader("Drop Columns With Too Many Nulls")
        null_threshold = st.slider(
            "Drop columns with missing values greater than or equal to this percentage",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
        )
        if st.button("Drop high-null columns", use_container_width=True):
            updated_df, dropped_columns = drop_high_null_columns(st.session_state.cleaned_df, null_threshold)
            st.session_state.cleaned_df = updated_df
            if dropped_columns:
                st.success(f"Dropped columns: {', '.join(dropped_columns)}")
                st.rerun()
            else:
                st.info("No columns met the selected null threshold.")

        st.subheader("Basic Outlier Removal")
        numeric_columns_for_cleaning = st.session_state.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_columns = st.multiselect(
            "Select numeric columns for outlier removal",
            options=numeric_columns_for_cleaning,
        )
        if st.button("Remove outlier rows", use_container_width=True):
            updated_df, removed_count = remove_outliers(st.session_state.cleaned_df, outlier_columns)
            st.session_state.cleaned_df = updated_df
            st.success(f"Removed {removed_count} rows containing outliers in the selected columns.")
            st.rerun()
    st.header("Download")
    st.write("Download the current cleaned dataset as a CSV file.")
    st.download_button(
        label="Download cleaned dataset",
        data=dataframe_to_csv_bytes(st.session_state.cleaned_df),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
