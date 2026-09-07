# 🛡️ Credit Card Fraud Detection System (FDS)

An end-to-end Machine Learning web platform developed with **Django**, **Scikit-Learn**, and **Pandas** to detect, predict, and analyze fraudulent financial transactions in real-time and batch workflows.

---

## 📌 Features

- **Interactive Exploratory Data Analysis (EDA)**: Instant statistical breakdown of class distribution (fraudulent vs legitimate), dataset shape, and missing value diagnostics.
- **Dynamic Model Training Pipeline**: Upload CSV datasets directly through the UI; automatically runs `StandardScaler` feature scaling, train-test splitting, and fits a high-precision `LogisticRegression` classification model.
- **Model & Feature Persistence**: Serializes trained classifiers, test datasets, and evaluation states directly in the database for instant recall.
- **Single Transaction Inference**: Predict fraud risk on individual transactions with precision scores.
- **Batch CSV Classification**: Process and classify hundreds/thousands of transaction records in bulk with visual risk badges.
- **Manual Parameter Input**: Test hypothetical or individual transaction parameter values (V1 through V28, Time, Amount) directly via interactive web forms.
- **Interactive DataTables Explorer**: Explore uploaded transaction datasets with server-side pagination.

---

## 🛠️ Tech Stack

- **Backend**: Python, Django 6.x
- **Machine Learning / Data Science**: Scikit-Learn (Logistic Regression, StandardScaler), Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, DataTables, Bootstrap
- **Database**: SQLite3 (relational storage with BLOB model serialization)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup & Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Run the Development Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

## 📂 Project Structure

```text
Fraud-Detection/
├── Apps/
│   └── homeApp/
│       ├── models.py       # DataFileUpload model with serialized model storage
│       ├── views.py        # ML pipelines, EDA logic, batch/single inference
│       └── urls.py         # App routes
├── FDS/
│   ├── settings.py         # Django settings & configurations
│   └── urls.py             # Root URL configuration
├── static/                 # CSS stylesheets, JavaScript, and image assets
├── templates/              # HTML UI templates (Landing, Reports, Analysis, Predictions)
├── manage.py
├── requirements.txt
└── README.md
```

---

## 📊 Sample Datasets
- `single.csv`: Clean sample of a single legitimate transaction.
- `fraud.csv`: Isolated sample of a confirmed fraudulent transaction.
- `multi.csv`: Multi-record transaction sample containing both fraudulent and legitimate rows for batch inference testing.
