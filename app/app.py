import streamlit as st
import pandas as pd
import os
import time
import joblib
import matplotlib.pyplot as plt
from preprocessing import preprocess_df
from model_trainer import train_model
from shap_explainer import explain_model
from st_aggrid import AgGrid, GridOptionsBuilder

WATCH_DIR = "watch_folder"
POLL_INTERVAL = 5  # seconds

# ---- SETUP ----
st.set_page_config(layout="wide")
st.title("💳 Real-Time Fraud Detection App")

# ---- SIDEBAR LIVE MODE ----
st.sidebar.header("🕒 Real-time Watch Mode")
watch_enabled = st.sidebar.checkbox("Enable Real-time Watch Mode", value=False)
watch_dir = st.sidebar.text_input("Folder to watch:", value=WATCH_DIR)

if watch_enabled and not os.path.exists(watch_dir):
    os.makedirs(watch_dir)
    st.sidebar.success(f"Created watch folder at: {watch_dir}")

@st.cache_data(ttl=POLL_INTERVAL)
def load_new_files(watch_dir):
    files = [f for f in os.listdir(watch_dir) if f.endswith(".csv")]
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(watch_dir, file))
        df = preprocess_df(df)
        df['source_file'] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ---- FILE INGEST ----
all_data = pd.DataFrame()
if watch_enabled:
    with st.spinner("Watching folder... Drop your CSVs now!"):
        all_data = load_new_files(watch_dir)
        time.sleep(POLL_INTERVAL)
else:
    uploaded_files = st.file_uploader("📁 Upload multiple transaction CSVs", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            df = preprocess_df(df)
            df['source_file'] = file.name
            all_data = pd.concat([all_data, df], ignore_index=True)

# ---- PROCESS DATA ----
if not all_data.empty:
    st.success(f"✅ {len(all_data)} records loaded.")

    with st.spinner("🔄 Initializing model..."):
        model, X_test, target_col = train_model(all_data)
    st.success("✅ Model ready.")

    st.subheader("📊 SHAP Feature Importance")
    explain_model(X_test)
    st.image("shap_outputs/shap_summary.png", caption="SHAP summary of top fraud features", use_container_width=True)

    def is_fraud_row(val):
        if isinstance(val, str):
            return val.strip().lower() in ['1', 'true', 'yes', 'fraud']
        if isinstance(val, (int, float, bool)):
            return val == 1 or val is True
        return False

    frauds = all_data[all_data[target_col].apply(is_fraud_row)]

    st.subheader("📿 Interactive Fraudulent Transactions")
    gb = GridOptionsBuilder.from_dataframe(frauds)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
    gridOptions = gb.build()
    AgGrid(frauds, gridOptions=gridOptions, enable_enterprise_modules=False)

    st.download_button("📅 Download Fraud Table", data=frauds.to_csv(index=False).encode('utf-8'),
                       file_name='fraud_cases.csv', mime='text/csv')

    st.markdown("## 🧠 Fraud Summary & Explanation")
    st.markdown(f"- 🔍 **{len(frauds)} fraudulent transactions** detected out of **{len(all_data)} total records**.")
    st.markdown(f"- 🗾 Fraud detected in **{frauds['source_file'].nunique()} file(s)**.")

    st.markdown("### 📌 Top Patterns in Frauds")
    top_cols = frauds.select_dtypes(include=['object', 'category']).nunique()
    top_categoricals = top_cols[top_cols < 10].index.tolist()
    if top_categoricals:
        for col in top_categoricals:
            st.markdown(f"**🗂️ {col}**")
            st.dataframe(frauds[col].value_counts().reset_index().rename(columns={'index': col, col: 'Count'}))

    st.markdown("### 📈 Summary of Numerical Features in Frauds")
    st.dataframe(frauds.select_dtypes(include='number').describe().T)

    def plot_top_fraud_features(fraud_df, full_df, top_n=10):
        fraud_means = fraud_df.mean(numeric_only=True)
        full_means = full_df.mean(numeric_only=True)
        contribution = (fraud_means - full_means).abs().sort_values(ascending=False)[:top_n]

        fig, ax = plt.subplots(figsize=(10, 5))
        contribution.plot(kind='barh', ax=ax, color='crimson')
        ax.set_title(f"Top {top_n} Feature Differences in Fraudulent vs All Transactions")
        ax.set_xlabel("Mean Difference")
        st.pyplot(fig)

    st.markdown("### 📊 Key Feature Differences")
    plot_top_fraud_features(frauds, all_data)

    st.markdown("---")
    st.markdown("🔍 _These summaries help in identifying recurring patterns in frauds across sources._")
    st.markdown("📌 _Use the SHAP chart above to understand which features contribute most to fraud predictions._")

else:
    st.info("👈 Upload one or more CSV files to begin or enable watch mode to load files automatically.")