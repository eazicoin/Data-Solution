import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
from ydata_profiling import ProfileReport
import datetime

# Set page config
st.set_page_config(page_title="Data Science Expert", layout="wide")

# Title
st.title("Data Science Expert App")
st.markdown("""
This app performs automated data analysis, visualization, and predictive modeling. 
Upload your dataset and follow the steps to get insights!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Step:", 
                          ["Upload Data", "Data Cleaning", "EDA", 
                           "Visualization", "Prediction", "Insights"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'report' not in st.session_state:
    st.session_state.report = None

# Step 1: Upload Data
if options == "Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success("Data uploaded successfully!")
            
            st.subheader("Data Preview")
            st.write(df.head())
            
            st.subheader("Basic Information")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            
            # Basic stats
            st.subheader("Basic Statistics")
            st.write(df.describe(include='all'))
            
            # Detect data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            st.write(dtype_df)
            
            # Detect missing values
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(df.isna().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
            st.write(missing_df)
            
            # Detect duplicates
            st.subheader("Duplicate Rows")
            st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Step 2: Data Cleaning
elif options == "Data Cleaning" and st.session_state.df is not None:
    st.header("Data Cleaning")
    df = st.session_state.df
    
    st.subheader("Data Issues Detected")
    
    # Create a list of issues
    issues = []
    
    # Check for missing values
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        issues.append(f"Missing values detected: {missing_values} total missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplicate rows detected: {duplicates} duplicates")
    
    # Check for outliers (simple check)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if (df[col].max() > df[col].mean() + 3 * df[col].std()) or \
           (df[col].min() < df[col].mean() - 3 * df[col].std()):
            issues.append(f"Potential outliers detected in column: {col}")
    
    # Check for inconsistent data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col])
                issues.append(f"Column '{col}' contains numeric data but is stored as text")
            except:
                pass
    
    # Display issues
    if issues:
        st.warning("The following issues were detected in your dataset:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("No major data quality issues detected!")
    
    # Cleaning options
    st.subheader("Data Cleaning Options")
    
    cleaning_options = st.multiselect(
        "Select cleaning operations to perform:",
        [
            "Remove duplicate rows",
            "Fill missing values (numeric)",
            "Fill missing values (categorical)",
            "Remove rows with missing values",
            "Remove columns with high missing values (>30%)",
            "Convert text to numeric where possible",
            "Remove outliers (for numeric columns)",
            "Standardize column names"
        ]
    )
    
    if st.button("Clean Data"):
        cleaned_df = df.copy()
        
        # Apply selected cleaning operations
        if "Remove duplicate rows" in cleaning_options:
            cleaned_df = cleaned_df.drop_duplicates()
        
        if "Fill missing values (numeric)" in cleaning_options:
            numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            imputer = SimpleImputer(strategy='mean')
            cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
        
        if "Fill missing values (categorical)" in cleaning_options:
            cat_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        
        if "Remove rows with missing values" in cleaning_options:
            cleaned_df = cleaned_df.dropna()
        
        if "Remove columns with high missing values (>30%)" in cleaning_options:
            threshold = len(cleaned_df) * 0.3
            cleaned_df = cleaned_df.dropna(axis=1, thresh=threshold)
        
        if "Convert text to numeric where possible" in cleaning_options:
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                    except:
                        pass
        
        if "Remove outliers (for numeric columns)" in cleaning_options:
            numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                cleaned_df = cleaned_df[(cleaned_df[col] <= mean + 3*std) & 
                                      (cleaned_df[col] >= mean - 3*std)]
        
        if "Standardize column names" in cleaning_options:
            cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
        
        st.session_state.cleaned_df = cleaned_df
        st.success("Data cleaning completed!")
        
        st.subheader("Cleaned Data Preview")
        st.write(cleaned_df.head())
        
        st.subheader("Cleaning Report")
        st.write(f"Original shape: {df.shape}")
        st.write(f"New shape: {cleaned_df.shape}")
        st.write(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
        st.write(f"Columns removed: {df.shape[1] - cleaned_df.shape[1]}")

elif options == "Data Cleaning" and st.session_state.df is None:
    st.warning("Please upload a dataset first in the 'Upload Data' section.")

# Step 3: EDA
elif options == "EDA" and st.session_state.cleaned_df is not None:
    st.header("Exploratory Data Analysis (EDA)")
    df = st.session_state.cleaned_df
    
    st.subheader("Automated EDA Report")
    
    if st.button("Generate Full EDA Report"):
        try:
            profile = ProfileReport(df, title="Profiling Report")
            st.session_state.report = profile
            
            # Display in Streamlit
            st_profile_report(profile)
        except Exception as e:
            st.error(f"Error generating report: {e}")
    
    st.subheader("Quick Insights")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        st.write("### Numeric Columns Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
            st.write(f"**Statistics for {selected_num_col}**")
            st.write(df[selected_num_col].describe())
            
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num_col], kde=True, ax=ax)
            st.pyplot(fig)
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            st.write("### Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    
    # Categorical columns analysis
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.write("### Categorical Columns Analysis")
        
        selected_cat_col = st.selectbox("Select categorical column:", cat_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Value Counts for {selected_cat_col}**")
            st.write(df[selected_cat_col].value_counts())
            
        with col2:
            fig, ax = plt.subplots()
            df[selected_cat_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

elif options == "EDA" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# Step 4: Visualization
elif options == "Visualization" and st.session_state.cleaned_df is not None:
    st.header("Data Visualization")
    df = st.session_state.cleaned_df
    
    st.subheader("Select Visualization Type")
    
    viz_type = st.selectbox(
        "Choose visualization type:",
        [
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Pie Chart",
            "Heatmap",
            "Pair Plot"
        ]
    )
    
    # Common options
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if viz_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-axis", numeric_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            hue = st.selectbox("Hue (optional)", [None] + list(cat_cols))
            
            fig = px.scatter(df, x=x_axis, y=y_axis, color=hue, 
                            title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 2 numeric columns for scatter plot")
    
    elif viz_type == "Line Chart":
        if len(numeric_cols) >= 1:
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            hue = st.selectbox("Hue (optional)", [None] + list(cat_cols))
            
            fig = px.line(df, x=x_axis, y=y_axis, color=hue, 
                          title=f"{y_axis} over {x_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 numeric column for line chart")
    
    elif viz_type == "Bar Chart":
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            x_axis = st.selectbox("X-axis", cat_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            hue = st.selectbox("Hue (optional)", [None] + list(cat_cols))
            
            fig = px.bar(df, x=x_axis, y=y_axis, color=hue, 
                         title=f"{y_axis} by {x_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 categorical and 1 numeric column for bar chart")
    
    elif viz_type == "Histogram":
        if len(numeric_cols) >= 1:
            col = st.selectbox("Select column:", numeric_cols)
            bins = st.slider("Number of bins", 5, 100, 20)
            hue = st.selectbox("Hue (optional)", [None] + list(cat_cols))
            
            fig = px.histogram(df, x=col, nbins=bins, color=hue, 
                              title=f"Distribution of {col}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 numeric column for histogram")
    
    elif viz_type == "Box Plot":
        if len(numeric_cols) >= 1:
            y_axis = st.selectbox("Y-axis", numeric_cols)
            x_axis = st.selectbox("X-axis (optional)", [None] + list(cat_cols))
            
            fig = px.box(df, y=y_axis, x=x_axis, 
                         title=f"Box Plot of {y_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 numeric column for box plot")
    
    elif viz_type == "Violin Plot":
        if len(numeric_cols) >= 1:
            y_axis = st.selectbox("Y-axis", numeric_cols)
            x_axis = st.selectbox("X-axis (optional)", [None] + list(cat_cols))
            
            fig = px.violin(df, y=y_axis, x=x_axis, 
                            title=f"Violin Plot of {y_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 numeric column for violin plot")
    
    elif viz_type == "Pie Chart":
        if len(cat_cols) >= 1:
            col = st.selectbox("Select column:", cat_cols)
            
            fig = px.pie(df, names=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 1 categorical column for pie chart")
    
    elif viz_type == "Heatmap":
        if len(numeric_cols) >= 2:
            fig = px.imshow(df[numeric_cols].corr(), 
                           labels=dict(x="Features", y="Features", color="Correlation"),
                           x=numeric_cols, y=numeric_cols)
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 2 numeric columns for heatmap")
    
    elif viz_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select columns for pair plot:", numeric_cols, 
                                         default=numeric_cols[:3])
            hue = st.selectbox("Hue for pair plot:", [None] + list(cat_cols))
            
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df, dimensions=selected_cols, color=hue)
                st.plotly_chart(fig)
            else:
                st.warning("Select at least 2 columns for pair plot")
        else:
            st.warning("Need at least 2 numeric columns for pair plot")

elif options == "Visualization" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# Step 5: Prediction
elif options == "Prediction" and st.session_state.cleaned_df is not None:
    st.header("Predictive Modeling")
    df = st.session_state.cleaned_df
    
    st.subheader("Select Target Variable")
    target = st.selectbox("Choose the target variable for prediction:", df.columns)
    st.session_state.target = target
    
    # Determine problem type
    if df[target].dtype in ['int64', 'float64']:
        unique_values = df[target].nunique()
        if unique_values < 10:  # Arbitrary threshold for classification
            problem_type = "classification"
        else:
            problem_type = "regression"
    else:
        problem_type = "classification"
    
    st.session_state.model_type = problem_type
    st.write(f"Automatically detected problem type: **{problem_type}**")
    
    if st.button("Train Predictive Model"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = df.drop(columns=[target])
                y = df[target]
                
                # Handle categorical features
                cat_cols = X.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Handle missing values (just in case)
                imputer = SimpleImputer(strategy='mean')
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Train model
                if problem_type == "regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = model.score(X_test, y_test)
                    mse = mean_squared_error(y_test, y_pred)
                    st.success(f"Regression Model Trained! R² Score: {score:.2f}, MSE: {mse:.2f}")
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.success(f"Classification Model Trained! Accuracy: {accuracy:.2f}")
                
                st.session_state.model = model
                
                # Feature importance
                st.subheader("Feature Importance")
                if problem_type == "regression":
                    importance = model.feature_importances_
                else:
                    importance = model.feature_importances_
                
                feat_imp = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error training model: {e}")

elif options == "Prediction" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# Step 6: Insights
elif options == "Insights" and st.session_state.cleaned_df is not None:
    st.header("Data Storytelling & Insights")
    df = st.session_state.cleaned_df
    
    st.subheader("Key Insights from Your Data")
    
    # Generate basic insights
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    st.write("### Dataset Overview")
    st.write(f"- The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    
    if len(numeric_cols) > 0:
        st.write("### Numeric Features Insights")
        for col in numeric_cols:
            st.write(f"- **{col}**:")
            st.write(f"  - Range: {df[col].min():.2f} to {df[col].max():.2f}")
            st.write(f"  - Average: {df[col].mean():.2f}")
            st.write(f"  - Median: {df[col].median():.2f}")
            st.write(f"  - Standard Deviation: {df[col].std():.2f}")
    
    if len(cat_cols) > 0:
        st.write("### Categorical Features Insights")
        for col in cat_cols:
            st.write(f"- **{col}**:")
            top_values = df[col].value_counts().head(3)
            for val, count in top_values.items():
                st.write(f"  - '{val}' appears {count} times ({count/len(df)*100:.1f}%)")
    
    # Correlation insights if available
    if len(numeric_cols) >= 2:
        st.write("### Correlation Insights")
        corr = df[numeric_cols].corr().unstack().sort_values(ascending=False)
        corr = corr[corr != 1].drop_duplicates()
        
        top_pos = corr.nlargest(1)
        top_neg = corr.nsmallest(1)
        
        for (col1, col2), val in top_pos.items():
            st.write(f"- The strongest positive correlation is between **{col1}** and **{col2}** (r = {val:.2f})")
        
        for (col1, col2), val in top_neg.items():
            st.write(f"- The strongest negative correlation is between **{col1}** and **{col2}** (r = {val:.2f})")
    
    # Prediction insights if available
    if st.session_state.model is not None:
        st.write("### Predictive Modeling Insights")
        target = st.session_state.target
        model_type = st.session_state.model_type
        
        if model_type == "regression":
            st.write(f"- The model predicts **{target}** with reasonable accuracy.")
            st.write("- Features that most influence the prediction:")
            # Get feature importance (already calculated in prediction step)
        else:
            st.write(f"- The model classifies **{target}** with reasonable accuracy.")
            st.write("- Features that most influence the classification:")
        
        # Show top 3 features
        X = df.drop(columns=[target])
        if len(cat_cols) > 0:
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        if st.session_state.model_type == "regression":
            importance = st.session_state.model.feature_importances_
        else:
            importance = st.session_state.model.feature_importances_
        
        feat_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(3)
        
        for _, row in feat_imp.iterrows():
            st.write(f"  - **{row['Feature']}** (importance: {row['Importance']:.3f})")
    
    st.write("### Recommendations")
    st.write("- Based on the analysis, consider the following actions:")
    st.write("  1. Investigate the strongest correlations for potential causal relationships")
    st.write("  2. Explore the most important features identified by the predictive model")
    if len(numeric_cols) > 0:
        st.write("  3. Examine outliers in numeric features that may need special handling")
    if len(cat_cols) > 0:
        st.write("  4. Analyze the distribution of categorical variables for imbalances")
    
    st.write("### Next Steps")
    st.write("- To dive deeper into the analysis:")
    st.write("  1. Export the cleaned dataset for further analysis")
    st.write("  2. Try different visualization types to uncover hidden patterns")
    st.write("  3. Experiment with different predictive models and parameters")

elif options == "Insights" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Science Expert App**  
Created with Streamlit, Pandas, and Scikit-learn  
[GitHub Repository](#)  
""")