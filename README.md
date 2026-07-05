# 📊 Social Media Engagement Analysis & Prediction Dashboard

A professional, interactive, and modern web application built using Python, Streamlit, and Scikit-Learn. The dashboard performs advanced data analytics and utilizes a trained Decision Tree Classifier to predict whether a social media post will receive **High Engagement** or **Low Engagement**.

This project is ready to run locally or deploy to **Streamlit Community Cloud**.

---

## 🚀 Key Features

*   **🏠 Modern Home Dashboard**: Interactive project overview, algorithm description, and real-time dataset shape metrics with custom-designed KPI cards.
*   **🔍 Dataset Explorer**: Upload custom CSVs or use the bundled dataset. Generates profile reports including dimensions, missing values, duplicates, column data types, and descriptive statistics.
*   **📊 Exploratory Data Analysis (EDA)**: Dynamic interactive plots using Plotly (Histogram, Box Plot, Scatter Plot, Correlation Heatmap, Pie Chart, Bar Chart, Count Plot, and Line Chart) with runtime column selection.
*   **🤖 Machine Learning Prediction**: 
    *   Upload arbitrary CSV data.
    *   Automatically handles preprocessing, custom LabelEncoding mapping alignment, missing columns, and feature alignment.
    *   Predicts class labels (`Yes` for High Engagement, `No` for Low Engagement) along with prediction probability confidence scores.
    *   Supports real-time model evaluation (Accuracy, Confusion Matrix Heatmap, and Classification Report) if ground-truth `engagement_rate` is present in the uploaded file.
    *   Provides CSV download capability for results.
*   **🌳 Feature Importance**: Visualization of Gini importance for the Decision Tree Classifier features, identifying the key drivers of social media success.
*   **📈 Dashboard Insights**: Advanced analytics presenting categorical summaries, engagement averages (likes, shares, comments), and average sentiment & toxicity scores.
*   **🎨 Visual Analytics**: High-quality distribution and box plots mapping metrics across multiple platforms, languages, sentiment levels, emotions, and campaigns.
*   **ℹ️ Project Details & Tech Stack**: Overview of the underlying technical architecture and developer placeholders.

---

## 📂 Project Structure

```text
SocialMediaDashboard/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── model.pkl                   # Trained DecisionTreeClassifier model
├── encoders.pkl                # Dictionary of LabelEncoders for categorical fields
├── README.md                   # Project documentation
└── Social Media Engagement Dataset .csv  # Sample dataset used for analysis
```

---

## 🛠️ Technology Stack

*   **Python 3.9+**
*   **Streamlit** (UI/UX layout & routing)
*   **Pandas & NumPy** (Data processing & manipulation)
*   **Scikit-Learn** (Label Encoding & Decision Tree Classifier)
*   **Plotly Express & Graph Objects** (Interactive visual analytics)
*   **Joblib** (Model loading and serialization)

---

## 💻 Running the App Locally

### 1. Prerequisites
Ensure you have Python installed. We recommend setting up a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all package requirements:

```bash
pip install -r requirements.txt
```

### 3. Start the Application
Run the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model & Dataset Details

*   **Algorithm**: `DecisionTreeClassifier`
*   **Target Variable**: Binarized version of `engagement_rate` 
    *   `engagement_rate >= 0.10` $\rightarrow$ **High Engagement (`Yes`)**
    *   `engagement_rate < 0.10` $\rightarrow$ **Low Engagement (`No`)**
*   **Dataset Dimensions**: 7,315 entries across 28 features, including post metadata, user past engagement statistics, text content characteristics, sentiment score, emotion type, and toxicity score.
*   **Preprocessed Drops**: `mentions` is excluded during predictions to match training feature structure.

---

## 👥 Authors
*   **Developer Name** — *Final Year Machine Learning Project*
*   [GitHub Placeholder](https://github.com/)
*   [LinkedIn Placeholder](https://linkedin.com/)
