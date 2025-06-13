# 💳 Real-Time Fraud Detection App

Detect fraudulent banking transactions in real time with the power of Machine Learning and SHAP explainability — all in a clean and intuitive Streamlit interface.

---

## ✅ What This App Does

- 🔍 Upload one or multiple banking transaction CSV files
- 🧠 Detects potentially fraudulent transactions on-the-fly
- 📊 Explains key fraud indicators using **SHAP** feature importance
- 📈 Summarizes fraud trends and provides key pattern insights
- 🧾 View, search, and download flagged fraud cases interactively

---

## ⚙️ How It Works

1. 📁 Upload CSV(s) — app supports multi-file uploads
2. 🔄 Data is automatically **preprocessed**
3. 🧠 If a model doesn’t exist, one is trained using the uploaded data
4. 🚨 App flags suspicious transactions based on learned patterns
5. 📊 **SHAP** visualization explains which features drove the model’s decision
6. 🧾 Interactive fraud table + summary of trends, patterns, and statistics

---

## 📸 Screenshots

### 📊 App Interface
![App Interface](assets/app_interface.png)

### ➕ Upload CSV Files
![Upload Demo](assets/upload_demo.png)

### 📟 View Fraud Table + Download
![Fraud Table](assets/fraud_table_demo.png)


---

## 🧪 [Try it on Streamlit →](https://detect-banking-fraud.streamlit.app/)

---

## 📦 Built With

- **Python**
- **Pandas** for data handling
- **Scikit-learn** for machine learning
- **Streamlit** for front-end interface
- **SHAP** for explainable AI
- **AgGrid** for advanced interactive tables
- **Matplotlib** for comparative feature visualization

---

## 🙋 Who This Is For

- 🧑‍💻 **FinTech Engineers** – prototype fraud systems
- 📊 **Data Scientists** – experiment with real-time fraud ML flows
- 🧠 **Learners & Students** – hands-on ML, SHAP, and Streamlit project
- 💼 **Recruiters** – evaluate candidate ML skills through working apps

---

## 📂 Project Structure



## Run Locally

```bash
.\venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py

### 🧪 Try with Example Data

You can download and test the app using our sample dataset below:

- 📄 [example dataset](data/Fraud Detection Dataset.csv)

