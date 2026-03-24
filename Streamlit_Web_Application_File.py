import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import uuid

st.set_page_config(page_title="File Uploader & Data Summary", layout="centered")
st.title("Analyzer and Detector of Data")

st.sidebar.header("📂 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

def missing_value_summary(df):
    summary = []
    placeholders = ['null', 'none', 'n/a', 'na', '-', '?', 'missing']
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            summary.append({'Missing Type': 'NaN', 'Column Name': col, 'Count': nan_count})
        if df[col].dtype == 'object':
            blank_count = (df[col] == '').sum()
            whitespace_count = df[col].str.strip().eq('').sum()
            placeholder_count = df[col].str.lower().isin(placeholders).sum()
            if blank_count > 0:
                summary.append({'Missing Type': 'Blank String', 'Column Name': col, 'Count': blank_count})
            if whitespace_count > 0:
                summary.append({'Missing Type': 'Whitespace Only', 'Column Name': col, 'Count': whitespace_count})
            if placeholder_count > 0:
                summary.append({'Missing Type': 'Placeholder Text', 'Column Name': col, 'Count': placeholder_count})
    return pd.DataFrame(summary) if summary else None

def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    return outlier_mask

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file loaded successfully")
            st.dataframe(df)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file loaded successfully")
            st.dataframe(df)
        elif file_type == 'txt':
            content = uploaded_file.read().decode("utf-8")
            st.success("✅ Text file loaded successfully")
            st.text_area("📜 File Content", content, height=300)
            df = None
        else:
            st.error("❌ Unsupported file format!")
            df = None

        if df is not None:
            # Value counts
            if 'value_count_cart' not in st.session_state:
                st.session_state['value_count_cart'] = []
            
            # Add Value Counts with Buttons
            st.subheader("🧮 Value Counts by Column")
            selected_vc_columns = st.multiselect("Select columns to show value counts and add to cart:", df.columns.tolist(), key="vc_selector")

            for col in selected_vc_columns:
                st.markdown(f"### 📊 Value Counts for **{col}**")
                vc_df = df[col].value_counts(dropna=False).reset_index()
                vc_df.columns = [col, 'Count']
                st.dataframe(vc_df)

                # Add to Cart Button
                if st.button(f"🛒 Add '{col}' Value Counts to Cart", key=f"add_cart_{col}"):
                    st.session_state['value_count_cart'].append((col, vc_df))
                    st.success(f"✅ Value counts for '{col}' added to cart.")
                
                # View the Value Count Cart
                if st.button("🛍️ View Cart of Value Counts"):
                    if st.session_state['value_count_cart']:
                        st.subheader("🛍️ Value Counts Cart")
                        for col, vc_df in st.session_state['value_count_cart']:
                            st.markdown(f"#### 🔢 Column: **{col}**")
                            st.dataframe(vc_df)
                    else:
                        st.info("🛒 Cart is currently empty.")

            # 🔎 Interactive Data Filters
            st.subheader("🔎 Interactive Data Filters") 
            filter_col = st.selectbox("Select column to filter:", df.columns)
            filter_type = st.radio("Select filter type:", ["Equals", "Range" if np.issubdtype(df[filter_col].dtype, np.number) else "Contains"])

            if filter_type == "Equals":
                selected_val = st.selectbox(f"Select value for {filter_col}:", df[filter_col].dropna().unique())
                filtered_df = df[df[filter_col] == selected_val]
            elif filter_type == "Contains":
                substring = st.text_input(f"Enter text to search in {filter_col}:")
                filtered_df = df[df[filter_col].str.contains(substring, case=False, na=False)]
            elif filter_type == "Range":
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                selected_range = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))
                filtered_df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]

            st.write("✅ Filtered Data Preview")
            st.dataframe(filtered_df, use_container_width=True)

            # Outlier Detection
            st.subheader("🚨 Outlier Detection (IQR Method)")

            if 'outlier_cart' not in st.session_state:
                st.session_state['outlier_cart'] = []

            # Detect outliers for the selected column
            outlier_mask = detect_outliers_iqr(df)
            outlier_counts = outlier_mask.sum()
            total_outliers = outlier_counts.sum()

            # Show summary of only columns with outliers
            outlier_summary = pd.DataFrame({
                "Column": outlier_counts[outlier_counts > 0].index,
                "Outlier Count": outlier_counts[outlier_counts > 0].values
            })
            
            st.dataframe(outlier_summary)
            
            if total_outliers > 0:
                st.warning(f"⚠️ Total Outliers Detected Across Columns: {int(total_outliers)}")

                # Column selector for handling
                selected_column = st.selectbox("🎯 Select a column to handle outliers:", outlier_summary["Column"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🧹 Remove Outliers from Selected Column"):
                        df_clean = df[~outlier_mask[selected_column]].reset_index(drop=True)
                        st.success(f"✅ Outliers removed from {selected_column}. Remaining rows: {df_clean.shape[0]}")
                        st.dataframe(df_clean)

                with col2:
                    if st.button("🔧 Impute Outliers in Selected Column"):
                        method = st.radio("Choose Imputation Method:", ["Mean", "Median", "Mode"], key="impute_method")
                        df_imputed = df.copy()
                        outliers = outlier_mask[selected_column]
                        if method == "Mean":
                            df_imputed.loc[outliers, selected_column] = df[selected_column].mean()
                        elif method == "Median":
                            df_imputed.loc[outliers, selected_column] = df[selected_column].median()
                        elif method == "Mode":
                            df_imputed.loc[outliers, selected_column] = df[selected_column].mode()[0]
                        st.success(f"✅ Outliers in {selected_column} imputed using {method}.")
                        st.dataframe(df_imputed.head())

                with col3:
                    if st.button("🛒 Add Column to Cart"):
                        if selected_column not in st.session_state['outlier_cart']:
                            st.session_state['outlier_cart'].append(selected_column)
                            st.success(f"✅ {selected_column} added to cart.")
                        else:
                            st.info(f"ℹ️ {selected_column} is already in the cart.")
                            
                # View cart
                if st.button("📦 View Outlier Handling Cart"):
                    st.subheader("🧾 Columns in Cart")
                    if st.session_state['outlier_cart']:
                        st.write(st.session_state['outlier_cart'])
                    else:
                        st.info("🛒 Your cart is empty.")
            else:
                st.success("🎉 No outliers detected in the dataset based on the IQR method.")

            # Missing Value Handling
            if 'missing_cart' not in st.session_state:
                st.session_state['missing_cart'] = []
   
            # Display summary
            st.subheader("🔍 Missing Value Summary")
            missing_summary = missing_value_summary(df)

            if missing_summary is not None and not missing_summary.empty:
                st.dataframe(missing_summary)
                    
            st.subheader("🛠️ Handle Missing Values by Column")
                    
            # Select column from those with missing values
            unique_columns_with_missing = missing_summary["Column Name"].unique().tolist() if missing_summary is not None else []
            if unique_columns_with_missing:
                selected_column = st.selectbox("🎯 Select a column to handle missing values:", unique_columns_with_missing)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🧹 Remove Missing Rows (Selected Column)"):
                        df = df[df[selected_column].notnull()].reset_index(drop=True)
                        st.session_state['df'] = df  # Update session state to persist changes
                        st.success(f"✅ Rows with missing values in {selected_column} removed. Remaining rows: {df.shape[0]}")
                        st.dataframe(df)
            
                with col2:
                    if st.button("🔧 Impute Missing Values (Selected Column)"):
                        df_imputed = df.copy()  # Work on a copy to avoid modifying original prematurely
                        if df[selected_column].dtype in [np.float64, np.int64]:
                            strategy = st.radio("Choose Imputation Method:", ["Mean", "Median", "Mode"], key=f"impute_numeric_{selected_column}")
                            if strategy == "Mean":
                                imputed_value = df[selected_column].mean()
                                df_imputed[selected_column] = df_imputed[selected_column].fillna(imputed_value)
                                st.success(f"✅ Imputed missing values in {selected_column} with mean ({imputed_value:.4f}).")
                            elif strategy == "Median":
                                imputed_value = df[selected_column].median()
                                df_imputed[selected_column] = df_imputed[selected_column].fillna(imputed_value)
                                st.success(f"✅ Imputed missing values in {selected_column} with median ({imputed_value:.4f}).")
                            elif strategy == "Mode":
                                imputed_value = df[selected_column].mode()[0]
                                df_imputed[selected_column] = df_imputed[selected_column].fillna(imputed_value)
                                st.success(f"✅ Imputed missing values in {selected_column} with mode ({imputed_value}).")
                        else:
                            strategy = st.radio("Choose Imputation Method:", ["Mode", "Custom Value"], key=f"impute_categorical_{selected_column}")
                            if strategy == "Mode":
                                imputed_value = df[selected_column].mode()[0]
                                df_imputed[selected_column] = df_imputed[selected_column].fillna(imputed_value)
                                st.success(f"✅ Imputed missing values in {selected_column} with mode ({imputed_value}).")
                            else:
                                custom_val = st.text_input("Enter custom value:", key=f"custom_val_{selected_column}")
                                if st.button("✅ Apply Custom Value", key=f"apply_custom_{selected_column}"):
                                    df_imputed[selected_column] = df_imputed[selected_column].fillna(custom_val)
                                    st.success(f"✅ Imputed missing values in {selected_column} with custom value ({custom_val}).")
                        
                        # Update the main dataframe and session state
                        df = df_imputed
                        st.session_state['df'] = df
                        st.dataframe(df.head())
            
                with col3:
                    if st.button("🛒 Add to Cart"):
                        if selected_column not in st.session_state['missing_cart']:
                            st.session_state['missing_cart'].append(selected_column)
                            st.success(f"✅ {selected_column} added to cart.")
                        else:
                            st.info(f"ℹ️ {selected_column} already in cart.")

                if st.button("📦 View Cart"):
                    st.subheader("🧾 Columns in Cart")
                    if st.session_state['missing_cart']:
                        st.write(st.session_state['missing_cart'])
                    else:
                        st.info("🛒 Your cart is empty.")
            else:
                st.success("🎉 No missing values detected!")
    
            # Update df from session state if modified
            if 'df' in st.session_state:
                df = st.session_state['df']

            st.subheader("🔄 Duplicate Rows")

            # Show count of duplicate rows
            duplicate_count = df.duplicated().sum()
            st.write(f"❗ **Duplicate Rows Found:** {duplicate_count}")

            # Option to remove duplicates if any
            if duplicate_count > 0:
                if st.checkbox("Remove duplicate rows"):
                    df = df.drop_duplicates()
                    st.session_state['df'] = df
                    st.success("✅ Duplicate rows removed.")
                else:
                    st.info("👆 Check the box to remove duplicates.")
            else:
                st.success("✅ No duplicate rows found.")

            st.subheader("📈 Columns with Most Unique Values")
            st.dataframe(df.nunique().sort_values(ascending=False))

            st.subheader("🧮 Select Columns for Descriptive Statistics")
            selected_columns = st.multiselect("Choose columns for descriptive analysis:", df.columns.tolist())
            if selected_columns:
                st.write("📊 Descriptive statistics for selected columns")
                st.dataframe(df[selected_columns].describe(include='all').transpose())

            # Data Visualization Section
            st.subheader("📊 Data Visualization")

            # Initialize chart configs and chart cart
            if 'chart_configs' not in st.session_state:
                st.session_state.chart_configs = []
            if 'chart_cart' not in st.session_state:
                st.session_state.chart_cart = []

            # Add chart
            if st.button("➕ Add Chart", key="add_chart"):
                st.session_state.chart_configs.append({'x': None, 'y': None, 'type': 'Scatter'})

            # Remove last chart
            if st.session_state.chart_configs and st.button("❌ Remove Last Chart", key="remove_chart"):
                st.session_state.chart_configs.pop()

            st.markdown("## 🔧 Configure Charts")

            for i, chart in enumerate(st.session_state.chart_configs):
                st.markdown(f"### Chart {i+1}")
                cols = st.columns([2, 2, 2, 1])
                
                # Select chart type
                chart['type'] = cols[0].selectbox(
                    "Chart Type",
                    ["Scatter", "Line", "Bar", "Histogram", "Box"],
                    key=f"type_{i}",
                    index=["Scatter", "Line", "Bar", "Histogram", "Box"].index(chart['type']) if chart['type'] in ["Scatter", "Line", "Bar", "Histogram", "Box"] else 0
                )
                
                # Select X-axis
                chart['x'] = cols[1].selectbox(
                    "X-axis",
                    ["None"] + df.columns.tolist(),
                    key=f"x_{i}",
                    index=0 if chart['x'] is None else df.columns.tolist().index(chart['x']) + 1 if chart['x'] in df.columns else 0
                )
                
                # Select Y-axis (required for all chart types)
                y_options = ["None"] + df.columns.tolist()
                chart['y'] = cols[2].selectbox(
                    "Y-axis",
                    y_options,
                    key=f"y_{i}",
                    index=0 if chart['y'] is None else df.columns.tolist().index(chart['y']) + 1 if chart['y'] in df.columns else 0
                )
                
                # Add to Cart Button
                if cols[3].button("🛒 Add to Cart", key=f"add_chart_cart_{i}"):
                    if chart['x'] == "None":
                        st.warning("⚠️ Please select an X-axis before adding to cart.")
                    elif chart['y'] == "None":
                        st.warning("⚠️ Please select a Y-axis before adding to cart.")
                    else:
                        chart_copy = chart.copy()  # Create a deep copy of the chart config
                        st.session_state.chart_cart.append(chart_copy)
                        st.success(f"✅ Chart {i+1} ({chart['type']}, X={chart['x']}, Y={chart['y']}) added to cart.")

            # View Chart Cart Button with improved rendering
            if st.button("🛍️ View Chart Cart", key="view_cart"):
                if st.session_state.chart_cart:
                    st.subheader("🛍️ Charts in Cart")
                    # Create a DataFrame to display chart configurations
                    cart_data = []
                    for idx, c in enumerate(st.session_state.chart_cart):
                        cart_data.append({
                            "Chart Number": idx + 1,
                            "Type": c['type'],
                            "X-axis": c['x'],
                            "Y-axis": c['y']
                        })
                    cart_df = pd.DataFrame(cart_data)
                    st.dataframe(cart_df)

                    # Render the charts
                    rows = len(st.session_state.chart_cart)
                    fig = make_subplots(
                        rows=rows, cols=1,
                        subplot_titles=[
                            f"Chart {idx+1}: {c['type']} (X={c['x']}, Y={c['y']})"
                            for idx, c in enumerate(st.session_state.chart_cart)
                        ]
                    )

                    for idx, c in enumerate(st.session_state.chart_cart):
                        row = idx + 1
                        x = c['x']
                        y = c['y']
                        ctype = c['type']

                        try:
                            # Validate data types and selections
                            if x != "None" and x in df.columns and y != "None" and y in df.columns:
                                if ctype == "Histogram":
                                    if not np.issubdtype(df[x].dtype, np.number):
                                        st.warning(f"⚠️ Histogram requires numeric X-axis. Skipping chart {idx+1} for {x}.")
                                        continue
                                    if np.issubdtype(df[y].dtype, np.number):
                                        st.warning(f"⚠️ Histogram Y-axis should be categorical for grouping. Ignoring Y-axis {y} and plotting {x}.")
                                        fig.add_trace(go.Histogram(x=df[x], name=x), row=row, col=1)
                                    else:
                                        for category in df[y].unique():
                                            subset = df[df[y] == category]
                                            fig.add_trace(
                                                go.Histogram(
                                                    x=subset[x],
                                                    name=f"{x} ({y}={category})",
                                                    opacity=0.6
                                                ),
                                                row=row, col=1
                                            )
                                        fig.update_layout(barmode='overlay')
                                elif ctype == "Box":
                                    if not np.issubdtype(df[y].dtype, np.number):
                                        st.warning(f"⚠️ Box plot requires numeric Y-axis. Skipping chart {idx+1} for {y}.")
                                        continue
                                    fig.add_trace(
                                        go.Box(
                                            x=df[x],
                                            y=df[y],
                                            name=f"{y} by {x}"
                                        ),
                                        row=row, col=1
                                    )
                                elif ctype == "Line":
                                    if not np.issubdtype(df[y].dtype, np.number) or not np.issubdtype(df[x].dtype, np.number):
                                        st.warning(f"⚠️ Line chart requires numeric X and Y axes. Skipping chart {idx+1}.")
                                        continue
                                    fig.add_trace(
                                        go.Scatter(
                                            x=df[x],
                                            y=df[y],
                                            mode='lines',
                                            name=f"{x} vs {y}"
                                        ),
                                        row=row, col=1
                                    )
                                elif ctype == "Scatter":
                                    if not np.issubdtype(df[y].dtype, np.number) or not np.issubdtype(df[x].dtype, np.number):
                                        st.warning(f"⚠️ Scatter chart requires numeric X and Y axes. Skipping chart {idx+1}.")
                                        continue
                                    fig.add_trace(
                                        go.Scatter(
                                            x=df[x],
                                            y=df[y],
                                            mode='markers',
                                            name=f"{x} vs {y}"
                                        ),
                                        row=row, col=1
                                    )
                                elif ctype == "Bar":
                                    if not np.issubdtype(df[y].dtype, np.number):
                                        st.warning(f"⚠️ Bar chart requires numeric Y-axis. Skipping chart {idx+1}.")
                                        continue
                                    fig.add_trace(
                                        go.Bar(
                                            x=df[x],
                                            y=df[y],
                                            name=f"{x} vs {y}"
                                        ),
                                        row=row, col=1
                                    )
                            else:
                                st.warning(f"⚠️ Invalid X-axis or Y-axis for chart {idx+1}. Skipping.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not render chart {idx+1}: {e}")

                    fig.update_layout(height=350 * rows, showlegend=True, title="📊 Charts in Cart")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🛒 Chart cart is currently empty.")

            # Display Current Charts
            valid_charts = [c for c in st.session_state.chart_configs if c['x'] != "None" and c['y'] != "None"]

            if valid_charts:
                st.subheader("📊 Current Charts")
                rows = len(valid_charts)
                fig = make_subplots(
                    rows=rows, cols=1,
                    subplot_titles=[
                        f"{c['type']}: X = {c['x']}, Y = {c['y']}"
                        for c in valid_charts
                    ]
                )

                for idx, c in enumerate(valid_charts):
                    row = idx + 1
                    x = c['x']
                    y = c['y']
                    ctype = c['type']

                    try:
                        # Validate data types and selections
                        if x in df.columns and y in df.columns:
                            if ctype == "Histogram":
                                if not np.issubdtype(df[x].dtype, np.number):
                                    st.warning(f"⚠️ Histogram requires numeric X-axis. Skipping chart {idx+1} for {x}.")
                                    continue
                                if np.issubdtype(df[y].dtype, np.number):
                                    st.warning(f"⚠️ Histogram Y-axis should be categorical for grouping. Ignoring Y-axis {y} and plotting {x}.")
                                    fig.add_trace(go.Histogram(x=df[x], name=x), row=row, col=1)
                                else:
                                    for category in df[y].unique():
                                        subset = df[df[y] == category]
                                        fig.add_trace(
                                            go.Histogram(
                                                x=subset[x],
                                                name=f"{x} ({y}={category})",
                                                opacity=0.6
                                            ),
                                            row=row, col=1
                                        )
                                    fig.update_layout(barmode='overlay')
                            elif ctype == "Box":
                                if not np.issubdtype(df[y].dtype, np.number):
                                    st.warning(f"⚠️ Box plot requires numeric Y-axis. Skipping chart {idx+1} for {y}.")
                                    continue
                                fig.add_trace(
                                    go.Box(
                                        x=df[x],
                                        y=df[y],
                                        name=f"{y} by {x}"
                                    ),
                                    row=row, col=1
                                )
                            elif ctype == "Line":
                                if not np.issubdtype(df[y].dtype, np.number) or not np.issubdtype(df[x].dtype, np.number):
                                    st.warning(f"⚠️ Line chart requires numeric X and Y axes. Skipping chart {idx+1}.")
                                    continue
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[x],
                                        y=df[y],
                                        mode='lines',
                                        name=f"{x} vs {y}"
                                    ),
                                    row=row, col=1
                                )
                            elif ctype == "Scatter":
                                if not np.issubdtype(df[y].dtype, np.number) or not np.issubdtype(df[x].dtype, np.number):
                                    st.warning(f"⚠️ Scatter chart requires numeric X and Y axes. Skipping chart {idx+1}.")
                                    continue
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[x],
                                        y=df[y],
                                        mode='markers',
                                        name=f"{x} vs {y}"
                                    ),
                                    row=row, col=1
                                )
                            elif ctype == "Bar":
                                if not np.issubdtype(df[y].dtype, np.number):
                                    st.warning(f"⚠️ Bar chart requires numeric Y-axis. Skipping chart {idx+1}.")
                                    continue
                                fig.add_trace(
                                    go.Bar(
                                        x=df[x],
                                        y=df[y],
                                        name=f"{x} vs {y}"
                                    ),
                                    row=row, col=1
                                )
                        else:
                            st.warning(f"⚠️ Invalid X-axis or Y-axis for chart {idx+1}. Skipping.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not render chart {idx+1}: {e}")

                fig.update_layout(height=350 * rows, showlegend=True, title="📊 Multi-Chart View")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 Add and configure at least one chart with valid X-axis and Y-axis.")

            # Top Values Insights
            st.subheader("🌟 Top Values Insights")

            vis_columns = st.multiselect("Select columns for Top Value Analysis:", df.columns.tolist(), key="top5_columns")

            if vis_columns:
                for col in vis_columns:
                    st.markdown(f"### 🔝 Top rows based on **{col}**")

                    if np.issubdtype(df[col].dtype, np.number):
                        # Get top 5 rows by highest numeric value
                        top_rows = df.sort_values(by=col, ascending=False).dropna(subset=[col]).head(5)
                    else:
                        # Get top 5 most frequent values and filter matching rows
                        top_values = df[col].dropna().value_counts().head(5).index.tolist()
                        top_rows = df[df[col].isin(top_values)]

                    st.dataframe(top_rows)

                    # Optional chart
                    if np.issubdtype(df[col].dtype, np.number):
                        fig = px.bar(top_rows, x=col, y=col, text=col)
                        fig.update_traces(textposition='auto')
                    else:
                        count_data = df[col].value_counts().loc[top_values].reset_index()
                        count_data.columns = [col, 'Count']
                        fig = px.bar(count_data, x=col, y='Count', text='Count')
                        fig.update_traces(textposition='auto')

                    st.plotly_chart(fig, use_container_width=True)

            # Remove Unique Columns
            st.subheader("📛 Remove Unique Columns")
            drop_cols = st.multiselect("Select columns to drop manually (e.g., IDs, names):", df.columns.tolist())
            if drop_cols and st.button("🗑️ Remove Selected Columns"):
                df = df.drop(columns=drop_cols)
                st.session_state['df'] = df
                st.success(f"✅ Removed columns: {', '.join(drop_cols)}")

            # Standardize Categorical Columns
            st.subheader("🔄 Standardize Categorical Columns")

            # Store standardized data globally
            if 'standardized_df' not in st.session_state:
                st.session_state.standardized_df = None

            if st.button("⚙️ Apply Standardization"):
                standardized_df = df.copy()
                for col in standardized_df.select_dtypes(include='object').columns:
                    standardized_df[col] = standardized_df[col].astype('category').cat.codes + 1
                st.session_state.standardized_df = standardized_df
                st.success("✅ Categorical columns standardized to numeric codes!")
                st.dataframe(standardized_df.head())

            # Data Splitting (Enhanced with Stratified Splitting and Visualization)
            if st.session_state.standardized_df is not None:
                st.subheader("🔀 Data Splitting")

                split_ratio = st.slider("📊 Select Train/Test Split Ratio", 0.1, 0.9, 0.7, 0.05, key="split_ratio_slider")
                target_col = st.selectbox("🎯 Select the target variable :", 
                                        ["None"] + st.session_state.standardized_df.columns.tolist(), 
                                        key="target_col_split")
                
                # Option for stratified splitting (only for classification tasks)
                stratified = False
                if target_col != "None":
                    stratified = st.checkbox("Use Stratified Splitting (for classification tasks)", value=False, key="stratified_split_checkbox")

                if st.button("📎 Split Data"):
                    from sklearn.model_selection import train_test_split
                    df_std = st.session_state.standardized_df

                    if target_col != "None":
                        X = df_std.drop(columns=[target_col])
                        y = df_std[target_col]

                        # Perform train-test split (with stratification if selected)
                        if stratified:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=42, stratify=y)
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=42)

                        from sklearn.impute import SimpleImputer
                        imputer = SimpleImputer(strategy='mean')
                        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
                        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test

                        st.success(f"✅ Data split completed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
                        st.write("📘 **X_train Sample:**")
                        st.dataframe(X_train.head())
                        st.write("📗 **y_train Sample:**")
                        st.dataframe(y_train.head())
                        st.write("📙 **X_test Sample:**")
                        st.dataframe(X_test.head())
                        st.write("📕 **y_test Sample:**")
                        st.dataframe(y_test.head())

                        # Visualize target variable distribution (if target is categorical)
                        if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) <= 10:
                            st.subheader("📊 Target Variable Distribution")
                            # Combine train and test distributions for comparison
                            train_dist = pd.Series(y_train, name='Train').value_counts(normalize=True).reset_index()
                            test_dist = pd.Series(y_test, name='Test').value_counts(normalize=True).reset_index()
                            train_dist.columns = [target_col, 'Proportion']
                            test_dist.columns = [target_col, 'Proportion']
                            train_dist['Dataset'] = 'Train'
                            test_dist['Dataset'] = 'Test'
                            dist_df = pd.concat([train_dist, test_dist], axis=0)

                            fig = px.bar(dist_df, x=target_col, y='Proportion', color='Dataset', barmode='group',
                                        title="Target Variable Distribution (Train vs Test)")
                            st.plotly_chart(fig, use_container_width=True)

                    else:
                        train_df, test_df = train_test_split(df_std, train_size=split_ratio, random_state=42)
                        st.success(f"✅ Data split completed. Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")
                        st.write("📘 **Train Sample:**")
                        st.dataframe(train_df.head())
                        st.write("📗 **Test Sample:**")
                        st.dataframe(test_df.head())
            else:
                st.warning("⚠️ Please apply standardization before splitting the data.")

            # Model Selection and Training
            st.subheader("📌 Model Selection and Training")

            # Initialize model cart in session state
            if 'model_cart' not in st.session_state:
                st.session_state.model_cart = []

            task_type = st.radio("🔍 Select Task Type:", ["Classification", "Regression"], key="task_type")

            if task_type == "Classification":
                model_name = st.selectbox("🧠 Choose Classifier:", [
                    "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)",
                    "Support Vector Machine (SVM)", "Naive Bayes", "Random Forest"
                ], key="classifier_select")
            else:
                model_name = st.selectbox("🧠 Choose Regressor:", [
                    "Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression",
                    "Ridge Regression", "Lasso Regression", "Decision Tree Regression", "Random Forest Regression"
                ], key="regressor_select")

            if st.button("🚀 Train Model"):
                if "X_train" in st.session_state and "y_train" in st.session_state:
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from sklearn.svm import SVC
                    from sklearn.naive_bayes import GaussianNB
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.pipeline import make_pipeline
                    from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, roc_auc_score, roc_curve, confusion_matrix

                    model = None
                    model_details = {'task_type': task_type, 'model_name': model_name}

                    if task_type == "Classification":
                        if model_name == "Logistic Regression":
                            model = LogisticRegression()
                        elif model_name == "Decision Tree":
                            model = DecisionTreeClassifier()
                        elif model_name == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsClassifier()
                        elif model_name == "Support Vector Machine (SVM)":
                            model = SVC(probability=True)  # Enable probability for AUC-ROC
                        elif model_name == "Naive Bayes":
                            model = GaussianNB()
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier()

                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        st.success("✅ Model Trained Successfully!")
                        st.subheader("📊 Classification Results")
                        accuracy = accuracy_score(y_test, predictions)
                        st.write("**Accuracy:**", accuracy)

                        report = classification_report(y_test, predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("📋 **Detailed Classification Report:**")
                        st.dataframe(report_df.style.format(precision=2))

                        # AUC-ROC and Curve
                        if len(set(y_test)) == 2:  # Binary classification check
                            try:
                                if hasattr(model, "predict_proba"):
                                    y_prob = model.predict_proba(X_test)[:, 1]
                                else:
                                    y_prob = model.decision_function(X_test)
                                auc = roc_auc_score(y_test, y_prob)
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                st.write(f"🔵 **AUC-ROC Score:** {auc:.4f}")

                                fig_roc = px.area(
                                    x=fpr, y=tpr,
                                    labels=dict(x="False Positive Rate", y="True Positive Rate"),
                                    title=f"ROC Curve (AUC = {auc:.4f})"
                                )
                                fig_roc.add_shape(
                                    type='line', line=dict(dash='dash'),
                                    x0=0, x1=1, y0=0, y1=1
                                )
                                st.plotly_chart(fig_roc, use_container_width=True)

                                model_details['auc_roc'] = auc
                            except Exception as e:
                                st.warning(f"⚠️ Could not compute AUC-ROC: {e}")
                                model_details['auc_roc'] = None
                        else:
                            st.info("ℹ️ ROC Curve is only applicable for binary classification.")
                            model_details['auc_roc'] = None

                        # Confusion Matrix
                        cm = confusion_matrix(y_test, predictions)
                        fig_cm = px.imshow(cm,
                                        labels=dict(x="Predicted", y="Actual", color="Count"),
                                        x=np.unique(y_test),
                                        y=np.unique(y_test),
                                        title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # Feature Importances (if available)
                        if hasattr(model, "feature_importances_"):
                            st.subheader("📌 Feature Importances")
                            importances = model.feature_importances_
                            feature_names = X_train.columns
                            importance_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Importance": importances
                            }).sort_values(by="Importance", ascending=False)

                            st.dataframe(importance_df)

                            fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances")
                            st.plotly_chart(fig, use_container_width=True)

                            model_details['feature_importances'] = importance_df.to_dict('records')
                        else:
                            st.info("ℹ️ Feature importances are not available for this model.")
                            model_details['feature_importances'] = None

                        model_details['accuracy'] = accuracy

                    else:  # Regression
                        if model_name == "Simple Linear Regression":
                            model = LinearRegression()
                            model.fit(X_train[[X_train.columns[0]]], y_train)
                            predictions = model.predict(X_test[[X_test.columns[0]]])
                        elif model_name == "Multiple Linear Regression":
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                        elif model_name == "Polynomial Regression":
                            degree = st.slider("Select Degree for Polynomial Regression", 2, 5, 2, key="poly_degree_slider")
                            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                        elif model_name == "Ridge Regression":
                            model = Ridge()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                        elif model_name == "Lasso Regression":
                            model = Lasso()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                        elif model_name == "Decision Tree Regression":
                            model = DecisionTreeRegressor()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                        elif model_name == "Random Forest Regression":
                            model = RandomForestRegressor()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)

                        st.success("✅ Model Trained Successfully!")
                        st.subheader("📊 Regression Results")

                        r2 = r2_score(y_test, predictions)
                        mse = mean_squared_error(y_test, predictions)
                        metrics = {
                            "Metric": ["R² Score", "Mean Squared Error (MSE)"],
                            "Value": [r2, mse]
                        }
                        metrics_df = pd.DataFrame(metrics)
                        st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))

                        # Actual vs. Predicted
                        fig_actual_vs_pred = px.scatter(
                            x=y_test, y=predictions,
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title="Actual vs. Predicted"
                        )
                        fig_actual_vs_pred.add_shape(
                            type='line', x0=y_test.min(), x1=y_test.max(),
                            y0=y_test.min(), y1=y_test.max(),
                            line=dict(color='red', dash='dash')
                        )
                        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

                        # Residual Plot
                        residuals = y_test - predictions
                        fig_residuals = px.scatter(
                            x=predictions, y=residuals,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title="Residual Plot"
                        )
                        fig_residuals.add_shape(
                            type='line', x0=predictions.min(), x1=predictions.max(),
                            y0=0, y1=0, line=dict(color='red', dash='dash')
                        )
                        st.plotly_chart(fig_residuals, use_container_width=True)

                        # Feature Importances (if available)
                        if hasattr(model, "feature_importances_"):
                            st.subheader("📌 Feature Importances")
                            importances = model.feature_importances_
                            feature_names = X_train.columns
                            importance_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Importance": importances
                            }).sort_values(by="Importance", ascending=False)

                            st.dataframe(importance_df)

                            fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances")
                            st.plotly_chart(fig, use_container_width=True)

                            model_details['feature_importances'] = importance_df.to_dict('records')
                        else:
                            st.info("ℹ️ Feature importances are not available for this model.")
                            model_details['feature_importances'] = None

                        model_details['r2_score'] = r2
                        model_details['mse'] = mse
                        model_details['auc_roc'] = None  # Not applicable for regression

                    # Add to Cart Button for Model
                    if st.button("🛒 Add Model to Cart", key=f"add_model_to_cart_{model_name}_{task_type}_{uuid.uuid4()}"):
                        if not any(m['model_name'] == model_name and m['task_type'] == task_type for m in st.session_state.model_cart):
                            st.session_state.model_cart.append(model_details)
                            st.success(f"✅ {model_name} ({task_type}) added to cart!")
                        else:
                            st.info(f"ℹ️ {model_name} ({task_type}) is already in the cart.")

                else:
                    st.warning("⚠️ Please split the data before training a model.")

            # View Model Cart Button
            if st.button("🛍️ View Model Cart", key="view_model_cart"):
                if st.session_state.get("model_cart"):
                    st.subheader("🛍️ Model Cart")
                    cart_data = []
                    for idx, model in enumerate(st.session_state.model_cart):
                        row = {
                            "Model ID": idx + 1,
                            "Task Type": model.get('task_type', 'N/A'),
                            "Model Name": model.get('model_name', 'N/A'),
                            "Accuracy": "N/A",
                            "AUC-ROC": "N/A",
                            "R² Score": "N/A",
                            "MSE": "N/A"
                        }
                        if model.get('task_type') == "Classification":
                            row["Accuracy"] = f"{model.get('accuracy', 'N/A'):.4f}" if model.get('accuracy') is not None else "N/A"
                            row["AUC-ROC"] = f"{model.get('auc_roc', 'N/A'):.4f}" if model.get('auc_roc') is not None else "N/A"
                        elif model.get('task_type') == "Regression":
                            row["R² Score"] = f"{model.get('r2_score', 'N/A'):.4f}" if model.get('r2_score') is not None else "N/A"
                            row["MSE"] = f"{model.get('mse', 'N/A'):.4f}" if model.get('mse') is not None else "N/A"
                        cart_data.append(row)

                    cart_df = pd.DataFrame(cart_data)
                    st.dataframe(cart_df)

                    # Optionally display feature importances if available
                    for idx, model in enumerate(st.session_state.model_cart):
                        if 'feature_importances' in model and model['feature_importances']:
                            st.markdown(f"#### Feature Importances for {model['model_name']} ({model['task_type']})")
                            importance_df = pd.DataFrame(model['feature_importances'])
                            st.dataframe(importance_df)
                            fig = px.bar(importance_df, x="Feature", y="Importance", title=f"Feature Importances for {model['model_name']}")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🛒 Model cart is currently empty.")

    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
else:
    st.info("📁 Please upload a CSV, Excel, or Text file using the sidebar!")