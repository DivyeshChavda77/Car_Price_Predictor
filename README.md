# 🚗 Car Price Predictor

An end-to-end Machine Learning project that predicts the resale value of used cars based on vehicle specifications such as brand, fuel type, transmission, ownership history, mileage, and vehicle age.

Built using Python, Scikit-Learn, CatBoost, Flask, and Machine Learning Pipelines.

---

## 📌 Project Overview

The objective of this project is to estimate the market value of a used car using machine learning techniques.

The project includes:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing Pipeline
* Model Training & Evaluation
* Prediction Pipeline
* Flask Web Application

---

## 📂 Project Structure

```text
Car_Price_Predictor/
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── notebook/
│   ├── EDA_Car_Price_Predictor.ipynb
│   └── MODEL_TRAINING.ipynb
│
├── src/
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   └── home.html
│
├── static/
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Dataset Features

| Feature       | Description             |
| ------------- | ----------------------- |
| km_driven     | Total kilometers driven |
| fuel          | Fuel type of vehicle    |
| seller_type   | Type of seller          |
| transmission  | Manual or Automatic     |
| owner         | Ownership history       |
| brand         | Vehicle brand           |
| car_age       | Age of vehicle          |
| selling_price | Target Variable         |

---

## 🔧 Feature Engineering

The following feature engineering techniques were applied:

### Brand Extraction

Extracted vehicle brand from car name.

Example:

```python
Maruti Swift Dzire VDI
↓
Maruti
```

### Car Age

Created a new feature:

```python
car_age = Current Year - Manufacturing Year
```

### Log Transformation

Applied logarithmic transformation to target variable:

```python
np.log1p(selling_price)
```

This significantly improved model performance.

---

## 📈 Exploratory Data Analysis

Performed:

* Missing Value Analysis
* Duplicate Record Analysis
* Univariate Analysis
* Bivariate Analysis
* Outlier Detection
* Correlation Analysis

### Key Insights

* Car age negatively impacts selling price.
* Higher mileage vehicles generally have lower resale values.
* Automatic transmission vehicles command higher prices.
* Brand significantly influences resale value.
* First-owner vehicles typically sell at higher prices.

---

## 🤖 Machine Learning Models Used

The following regression models were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor

### Model Performance

| Model             | R² Score |
| ----------------- | -------: |
| Linear Regression |   0.7766 |
| Decision Tree     |   0.6124 |
| Random Forest     |   0.7552 |
| XGBoost           |   0.7765 |
| CatBoost          |   0.8078 |

### Best Model

✅ CatBoost Regressor

---

## ⚙️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* CatBoost
* XGBoost
* Flask
* Dill

---

## 🚀 Web Application

The project includes a Flask web application where users can:

* Enter vehicle details
* Predict used car resale value
* View estimated market price instantly

---

## 🖥️ Running the Project

### Clone Repository

```bash
git clone <repository-url>
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🎯 Future Improvements

* Hyperparameter Tuning
* Docker Deployment
* Cloud Deployment
* Advanced Feature Engineering
* Real-time Car Price API

---

## 👨‍💻 Author

**Divyesh Chavda**

Artificial Intelligence & Data Science Student

Passionate about Machine Learning, Data Science, and AI-based Solutions.

---

## ⭐ Project Highlights

* End-to-End Machine Learning Pipeline
* Feature Engineering
* Model Comparison
* CatBoost Implementation
* Flask Deployment
* Industry-Standard Project Structure
* Resume-Ready Project
