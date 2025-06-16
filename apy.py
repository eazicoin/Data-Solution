import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from ydata_profiling import ProfileReport
import tempfile
import os

# App Configuration
st.set_page_config(page_title="Data Science Expert App", layout="wide")
st.title("🧠 Data Science Expert App")
st.write("Upload your dataset to get expert-level data cleaning, EDA, visualization, modeling, and insights.")

# 1. Upload Dataset
uploaded_file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    # 2. Data Cleaning
    st.subheader("🧼 Automated Data Cleaning")
    if st.button("Clean Dataset"):
        original_shape = df.shape

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Fill missing values
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include='object').columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Type conversion
        df = df.convert_dtypes()

        cleaned_shape = df.shape
        st.success(f"✅ Data cleaned. Shape: {original_shape} ➡ {cleaned_shape}")

    # 3. EDA - ydata_profiling
    st.subheader("📊 Automated EDA Report")
    if st.button("Generate EDA Report"):
        with st.spinner("Creating profile report..."):
            profile = ProfileReport(df, title="📈 Data Profile Report", explorative=True)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                profile.to_file(tmp_file.name)
                with open(tmp_file.name, "r", encoding="utf-8") as f:  # ✅ Fix UnicodeDecodeError
                    html_content = f.read()
                st.components.v1.html(html_content, height=1000, scrolling=True)

    # 4. Summary Stats & Correlation
    st.subheader("📋 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🧭 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # 5. Visualization
    st.subheader("📈 Interactive Visualizations")
    col1 = st.selectbox("Select X-axis", df.columns)
    col2 = st.selectbox("Select Y-axis", df.columns)
    chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line"])

    if st.button("Plot Chart"):
        if chart_type == "Scatter":
            st.plotly_chart(px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}"))
        elif chart_type == "Bar":
            st.plotly_chart(px.bar(df, x=col1, y=col2, title=f"{col1} vs {col2}"))
        elif chart_type == "Line":
            st.plotly_chart(px.line(df, x=col1, y=col2, title=f"{col1} vs {col2}"))

    # 6. Predictive Modeling
    st.subheader("🔮 Predictive Modeling")
    target = st.selectbox("🎯 Select Target Column", df.columns)
    feature_options = [col for col in df.columns if col != target]
    selected_features = st.multiselect("🧪 Select Feature Columns", feature_options)

    if st.button("Train Model") and selected_features:
        X = pd.get_dummies(df[selected_features])
        y = df[target]
        y_type = "classification" if y.nunique() <= 10 else "regression"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier() if y_type == "classification" else RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.success(f"✅ {y_type.title()} model trained using Random Forest.")

        if y_type == "classification":
            st.code(classification_report(y_test, preds))
        else:
            mse = mean_squared_error(y_test, preds)
            st.metric("Mean Squared Error", round(mse, 2))

        # Feature Importance
        st.subheader("📊 Feature Importance")
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(importance)

        # 7. Storytelling Insights
        st.subheader("📘 Storytelling Insights")
        st.markdown(f"""
        - **Prediction Goal:** `{target}`
        - **Top 3 Influential Features:**
        """)

        for i, feat in enumerate(importance.head(3).index):
            st.markdown(f"  {i+1}. **{feat}**")

        if y_type == "classification":
            st.markdown("""
            ✅ This model helps classify data points into known categories, which is useful for:
            - Customer segmentation
            - Fraud detection
            - Loan default prediction
            """)
        else:
            st.markdown("""
            📈 This regression model forecasts numerical outcomes. Ideal for:
            - Sales forecasting
            - Risk scoring
            - Financial projections
            """)
