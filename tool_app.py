import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import io
import json
import base64
import zipfile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# App Configuration
st.set_page_config(
    page_title="Advanced Data Analysis Toolkit",
    page_icon=":bar_chart:",
    layout="wide",    
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# Advanced Data Loading Function
def load_data(file):
    try:
        extension = file.name.split('.')[-1].lower()
        if extension == 'csv':
            return pd.read_csv(file)
        elif extension in ['xls', 'xlsx']:
            return pd.read_excel(file)
        elif extension == 'json':
            return pd.read_json(file)
        elif extension == 'parquet':
            return pd.read_parquet(file)
        elif extension == 'txt':
            return pd.read_csv(file, sep='\t')  # Assuming tab-separated
        else:
            st.error(f"Unsupported file type: {extension}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Enhanced Categorical Data Handling
def handle_categorical_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if categorical_cols.empty:
        st.write("No categorical columns found.")
        return df

    encoding_method = st.selectbox("Choose Encoding Method", [
        "One-Hot Encoding", 
        "Label Encoding", 
        "Frequency Encoding",
        "Target Encoding"
    ])

    if encoding_method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=categorical_cols)
    elif encoding_method == "Label Encoding":
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    elif encoding_method == "Frequency Encoding":
        for col in categorical_cols:
            freq_dict = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_dict)
    elif encoding_method == "Target Encoding":
        st.warning("Target Encoding requires a target variable. Please specify.")
    
    return df

# Advanced Null Value Handling
def handle_null_values(df):
    st.write("Current Null Values:")
    null_summary = df.isnull().sum()
    st.dataframe(null_summary[null_summary > 0])

    handling_method = st.selectbox("Choose Method to Handle Null Values", [
        "Drop rows", 
        "Fill with mean", 
        "Fill with median", 
        "Fill with mode", 
        "Advanced Imputation"
    ])

    if handling_method == "Drop rows":
        df.dropna(inplace=True)
    elif handling_method == "Fill with mean":
        df.fillna(df.mean(), inplace=True)
    elif handling_method == "Fill with median":
        df.fillna(df.median(), inplace=True)
    elif handling_method == "Fill with mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    elif handling_method == "Advanced Imputation":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Numerical Imputation
        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
        
        # Categorical Imputation
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Advanced Outlier Detection and Handling
def handle_outliers(df):
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.empty:
        st.write("No numerical columns to check for outliers.")
        return df
    
    # Boxplot for outlier visualization
    plt.figure(figsize=(15, 6))
    numerical_df.boxplot()
    st.pyplot(plt)
    
    # Z-score method
    z_scores = np.abs(stats.zscore(numerical_df))
    threshold = st.slider("Z-Score Threshold for Outliers", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
    
    outliers = (z_scores > threshold).any(axis=1)
    outliers_count = np.sum(outliers)
    st.write(f"Detected {outliers_count} rows as outliers.")
    
    outlier_handling = st.radio("Outlier Handling Method", [
        "Remove Outliers", 
        "Clip to Threshold", 
        "Keep Outliers"
    ])
    
    if outlier_handling == "Remove Outliers":
        df = df[~outliers]
        st.write(f"Removed {outliers_count} outliers.")
    elif outlier_handling == "Clip to Threshold":
        for col in numerical_df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

# Advanced Feature Scaling
def feature_scaling(df):
    scaling_method = st.selectbox("Choose Scaling Method", [
        "Standard Scaling", 
        "Min-Max Scaling", 
        "Log Scaling", 
        "Robust Scaling"
    ])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.write("No numeric columns found for scaling.")
        return df
    
    try:
        if scaling_method == "Standard Scaling":
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaling_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaling_method == "Log Scaling":
            df[numeric_cols] = np.log1p(df[numeric_cols])
        elif scaling_method == "Robust Scaling":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    except ValueError as e:
        st.error(f"An error occurred during scaling: {e}")
    
    return df

# Advanced Data Exploration and Visualization
def explore_data(df):
    st.header("🔍 Comprehensive Data Exploration")
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)
    
    # Visualization Options
    st.subheader("Data Visualization")
    viz_type = st.selectbox("Choose Visualization", [
        "Histogram", 
        "Box Plot", 
        "Scatter Plot", 
        "Pair Plot"
    ])
    
    if viz_type == "Histogram":
        col = st.selectbox("Select Column for Histogram", numeric_cols)
        plt.figure(figsize=(10, 6))
        df[col].hist()
        plt.title(f"Histogram of {col}")
        st.pyplot(plt)
    
    elif viz_type == "Box Plot":
        col = st.selectbox("Select Column for Box Plot", numeric_cols)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        st.pyplot(plt)
    
    elif viz_type == "Scatter Plot":
        x_col = st.selectbox("X-axis Column", numeric_cols)
        y_col = st.selectbox("Y-axis Column", numeric_cols)
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(plt)
    
    elif viz_type == "Pair Plot":
        plt.figure(figsize=(15, 10))
        sns.pairplot(df[numeric_cols])
        st.pyplot(plt)

# Feature Selection
def select_features(df):
    st.header("🔬 Feature Selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if st.checkbox("Perform Univariate Feature Selection"):
        k_features = st.slider("Number of Top Features to Select", min_value=1, max_value=len(numeric_cols), value=5)
        selector = SelectKBest(score_func=f_classif, k=k_features)
        
        # You'll need a target column for this
        target_col = st.selectbox("Select Target Column", numeric_cols)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        selector.fit(X, y)
        selected_feature_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_feature_indices]
        
        st.write("Top Selected Features:")
        st.write(selected_features)

# Advanced analysis with corrected widget keys
def advanced_analysis(df):
    st.header("🔬 Advanced Data Analysis and Visualization")
    
    # Check if DataFrame is valid
    if df is None or df.empty:
        st.error("No data available for analysis. Please load a dataset first.")
        return
    
    # Tabs for different analysis types
    analysis_tabs = st.tabs([
        "Statistical Summary", 
        "Distribution Analysis", 
        "Correlation & Clustering", 
        "Advanced Visualizations", 
        "Comparative Analysis"
    ])
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Statistical Summary Tab
    with analysis_tabs[0]:
        st.subheader("Comprehensive Statistical Insights")
        
        # Descriptive Statistics
        st.write("Descriptive Statistics:")
        desc_stats = df[numeric_cols].describe()
        st.dataframe(desc_stats)
        
        # Skewness and Kurtosis
        st.write("Distribution Characteristics:")
        skew_kurt_df = pd.DataFrame({
            'Column': numeric_cols,
            'Skewness': [df[col].skew() for col in numeric_cols],
            'Kurtosis': [df[col].kurtosis() for col in numeric_cols]
        })
        st.dataframe(skew_kurt_df)
    
    # Distribution Analysis Tab
    with analysis_tabs[1]:
        st.subheader("Distribution and Probability Analysis")
        
        # Ensure we have numeric columns
        if not numeric_cols:
            st.warning("No numeric columns available for distribution analysis.")
        else:
            # Distribution Plot Selection
            distribution_type = st.selectbox(
                "Choose Distribution Visualization", 
                [
                    "Kernel Density Estimation (KDE)", 
                    "Histogram with Probability Density", 
                    "Box Plot with Distribution",
                    "Q-Q Plot for Normality"
                ],
                key="distribution_viz_type"
            )
            
            # Numeric columns selection
            selected_cols = st.multiselect(
                "Select Columns for Analysis", 
                numeric_cols,
                key="distribution_cols"
            )
            
            # Plotting based on selection
            if selected_cols:
                plt.figure(figsize=(12, 6))
                
                if distribution_type == "Kernel Density Estimation (KDE)":
                    for col in selected_cols:
                        sns.kdeplot(df[col], fill=True, label=col)
                    plt.title("Kernel Density Estimation")
                    plt.legend()
                
                elif distribution_type == "Histogram with Probability Density":
                    for col in selected_cols:
                        sns.histplot(df[col], kde=True, stat="density", label=col)
                    plt.title("Histogram with Probability Density")
                    plt.legend()
                
                elif distribution_type == "Box Plot with Distribution":
                    sns.boxenplot(data=df[selected_cols])
                    plt.title("Enhanced Box Plot")
                    plt.xticks(rotation=45)
                
                elif distribution_type == "Q-Q Plot for Normality":
                    fig, axes = plt.subplots(1, len(selected_cols), figsize=(15, 5))
                    for i, col in enumerate(selected_cols):
                        stats.probplot(df[col], dist="norm", plot=axes[i])
                        axes[i].set_title(f"Q-Q Plot: {col}")
                    plt.tight_layout()
                
                st.pyplot(plt)
    
    # Correlation & Clustering Tab
    with analysis_tabs[2]:
        st.subheader("Advanced Correlation and Clustering")
        
        # Correlation Analysis
        correlation_method = st.selectbox(
            "Correlation Method", 
            ["Pearson", "Spearman", "Kendall"],
            key="correlation_method_select"
        )
        
        # Compute Correlation
        if numeric_cols:
            if correlation_method == "Pearson":
                corr_matrix = df[numeric_cols].corr(method='pearson')
            elif correlation_method == "Spearman":
                corr_matrix = df[numeric_cols].corr(method='spearman')
            else:
                corr_matrix = df[numeric_cols].corr(method='kendall')
            
            # Correlation Visualization
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, 
                        center=0, square=True)
            plt.title(f"{correlation_method} Correlation Heatmap")
            st.pyplot(plt)
        else:
            st.warning("No numeric columns available for correlation analysis.")
    
    # Advanced Visualizations Tab
    with analysis_tabs[3]:
        st.subheader("Multi-dimensional Visualizations")
        
        if not numeric_cols:
            st.warning("No numeric columns available for advanced visualizations.")
        else:
            # Advanced Visualization Types
            viz_type = st.selectbox(
                "Choose Advanced Visualization", 
                [
                    "t-SNE Visualization",
                    "PCA Visualization"
                ],
                key="advanced_viz_type"
            )
            
            # Prepare data
            X = df[numeric_cols]
            
            if viz_type == "t-SNE Visualization":
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(X)
                
                # Create DataFrame for plotting
                tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
                
                # Optional color mapping
                color_option = st.selectbox(
                    "Color by", 
                    list(df.columns),
                    key="tsne_color_option"
                )
                
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    tsne_df['TSNE1'], 
                    tsne_df['TSNE2'], 
                    c=df[color_option], 
                    cmap='viridis'
                )
                plt.colorbar(scatter)
                plt.title("t-SNE Visualization")
                st.pyplot(plt)
            
            elif viz_type == "PCA Visualization":
                # Prepare data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create DataFrame
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                
                # Color option
                color_option = st.selectbox(
                    "Color by", 
                    list(df.columns),
                    key="pca_color_option"
                )
                
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    pca_df['PC1'], 
                    pca_df['PC2'], 
                    c=df[color_option], 
                    cmap='viridis'
                )
                plt.colorbar(scatter)
                plt.title("PCA Visualization")
                plt.xlabel(f"First Principal Component (Variance Explained: {pca.explained_variance_ratio_[0]*100:.2f}%)")
                plt.ylabel(f"Second Principal Component (Variance Explained: {pca.explained_variance_ratio_[1]*100:.2f}%)")
                st.pyplot(plt)
    
    # Comparative Analysis Tab
    with analysis_tabs[4]:
        st.subheader("Comparative and Segmentation Analysis")
        
        # Ensure we have both categorical and numeric columns
        if not categorical_cols or not numeric_cols:
            st.warning("Need both categorical and numeric columns for comparative analysis.")
        else:
            # Analysis type selection
            analysis_type = st.selectbox(
                "Choose Comparative Analysis", 
                [
                    "Grouped Statistical Comparison",
                    "Feature Importance Ranking"
                ],
                key="comparative_analysis_type"
            )
            
            if analysis_type == "Grouped Statistical Comparison":
                # Group selection and comparison
                group_col = st.selectbox(
                    "Select Grouping Column", 
                    categorical_cols,
                    key="group_comparison_column"
                )
                compare_col = st.selectbox(
                    "Select Comparison Column", 
                    numeric_cols,
                    key="compare_numeric_column"
                )
                
                # Grouped box plot
                plt.figure(figsize=(12, 6))
                sns.boxplot(x=group_col, y=compare_col, data=df)
                plt.title(f"{compare_col} Comparison by {group_col}")
                plt.xticks(rotation=45)
                st.pyplot(plt)
            
            elif analysis_type == "Feature Importance Ranking":
                # Ensure we have enough numerical columns
                if len(numeric_cols) < 2:
                    st.warning("Need at least two numeric columns for feature importance.")
                else:
                    # Target column selection
                    target_col = st.selectbox(
                        "Select Target Column", 
                        numeric_cols,
                        key="feature_importance_target"
                    )
                    
                    # Prepare features and target
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Train Random Forest
                    rf = RandomForestRegressor(n_estimators=100)
                    rf.fit(X, y)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': rf.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Plot feature importance
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x='importance', y='feature', data=feature_importance)
                    plt.title("Feature Importance Ranking")
                    st.pyplot(plt)

#Downloading..........

def download_data(df):
    """
    Provides download options for the current dataframe
    
    Args:
        df (pd.DataFrame): DataFrame to be downloaded
    """
    st.header("📥 Download Processed Data")
    
    # Choose file format
    download_format = st.selectbox("Select Download Format", [
        "CSV", 
        "Excel", 
        "JSON", 
        "Parquet"
    ])
    
    # Generate download
    if st.button("Generate Download Link"):
        try:
            # Create a file in memory
            if download_format == "CSV":
                csv = df.to_csv(index=False)
                file_ext = "csv"
                mime_type = "text/csv"
            elif download_format == "Excel":
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                csv = excel_buffer.getvalue()
                file_ext = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif download_format == "JSON":
                csv = df.to_json(orient='records')
                file_ext = "json"
                mime_type = "application/json"
            elif download_format == "Parquet":
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                parquet_buffer.seek(0)
                csv = parquet_buffer.getvalue()
                file_ext = "parquet"
                mime_type = "application/octet-stream"
            
            # Create download link
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:{mime_type};base64,{b64}" download="processed_data.{file_ext}">Download {download_format} File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error generating download: {e}")

# Modify the main function to remove the previous advanced_analysis call
def main():
    st.title("🚀 Advanced Data Analysis Toolkit")
    
    # Sidebar Navigation
    app_mode = st.sidebar.selectbox("Choose Mode", [
        "Data Loading", 
        "Data Exploration", 
        "Data Preprocessing", 
        "Feature Engineering", 
        "Advanced Analysis",
        "Data Download"
    ])
    
    # Data Loading
    if app_mode == "Data Loading":
        st.header("📤 Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'json', 'parquet', 'txt'],
            help="Supported formats: CSV, Excel, JSON, Parquet, Text"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.success("Dataset Loaded Successfully!")
                st.write("Preview of Dataset:")
                st.dataframe(df.head())
    
    # Data Exploration
    elif app_mode == "Data Exploration":
        if st.session_state.df is not None:
            explore_data(st.session_state.df)
        else:
            st.error("Please load data first.")
    
    # Data Preprocessing
    elif app_mode == "Data Preprocessing":
        if st.session_state.df is not None:
            df = st.session_state.df
            
            preprocessing_option = st.selectbox("Choose Preprocessing Task", [
                "Handle Categorical Data", 
                "Handle Null Values", 
                "Handle Outliers"
            ])
            
            if preprocessing_option == "Handle Categorical Data":
                df = handle_categorical_data(df)
            elif preprocessing_option == "Handle Null Values":
                df = handle_null_values(df)
            elif preprocessing_option == "Handle Outliers":
                df = handle_outliers(df)
            
            st.session_state.df = df
        else:
            st.error("Please load data first.")
    
    # Feature Engineering
    elif app_mode == "Feature Engineering":
        if st.session_state.df is not None:
            df = st.session_state.df
            
            feature_option = st.selectbox("Choose Feature Engineering Task", [
                "Feature Scaling", 
                "Feature Selection"
            ])
            
            if feature_option == "Feature Scaling":
                df = feature_scaling(df)
                st.session_state.df = df
            elif feature_option == "Feature Selection":
                select_features(df)
        else:
            st.error("Please load data first.")
    
    # Advanced Analysis
    elif app_mode == "Advanced Analysis":
        if st.session_state.df is not None:
            advanced_analysis(st.session_state.df)
        else:
            st.error("Please load and preprocess data first.")
    # Download Option
    elif app_mode == "Data Download":
        if st.session_state.df is not None:
            download_data(st.session_state.df)
        else:
            st.error("Please load and process data first.")


if __name__ == "__main__":
    main()
