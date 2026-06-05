# Automated Invoice Classification (CapEx vs OpEx)

A machine learning solution that automatically classifies invoice expenses as **Capital Expenditure (CapEx)** or **Operating Expenditure (OpEx)** and integrates directly with Salesforce using an Azure-hosted prediction API.

This project was developed as part of the Georgia Tech OMSA Practicum in partnership with SalesHub.

---

## Business Problem

Organizations process large volumes of invoices every month. Determining whether an expense should be classified as:

* **CapEx (Capital Expenditure)**
* **OpEx (Operating Expenditure)**

is critical for:

* Financial reporting
* Budget planning
* Tax compliance
* Audit readiness

Manual classification is time-consuming and prone to inconsistency.

This project automates the classification process using Machine Learning and integrates the prediction directly into Salesforce workflows.

---

## Solution Overview

The solution consists of:

1. Invoice data stored in Salesforce
2. Machine Learning model trained using Python
3. Azure-hosted prediction API
4. Salesforce External Service integration
5. Salesforce Flow automation
6. Reporting dashboard for expenditure analysis

### Architecture

```text
Salesforce Invoice
        │
        ▼
Salesforce Flow
        │
        ▼
Azure Web App (Flask API)
        │
        ▼
Machine Learning Model
        │
        ▼
Prediction Result
(CapEx / OpEx)
        │
        ▼
Update Salesforce Record
        │
        ▼
Analytics Dashboard
```

---

## Dataset

The project uses a synthetic invoice dataset generated based on GAAP accounting principles.

### Features

| Feature                | Description                  |
| ---------------------- | ---------------------------- |
| Project Type           | Type of project              |
| Project Stage          | Current project stage        |
| Invoice Amount         | Invoice dollar amount        |
| Recurring Expense      | Whether expense is recurring |
| Useful Life Expectancy | Expected benefit period      |

Dataset size:

```text
403 invoices
```

Target variable:

```text
CapEx
OpEx
```

---

## Machine Learning Models Evaluated

The following models were trained and compared:

| Model                  | Accuracy |
| ---------------------- | -------- |
| Logistic Regression    | 95%      |
| Decision Tree          | 98%      |
| Random Forest          | 99%      |
| XGBoost                | 98%      |
| Support Vector Machine | 97%      |

The final implementation used a **Decision Tree Classifier** because it offered:

* High accuracy
* Fast inference
* Strong interpretability
* Easy explanation to business stakeholders

---

## Feature Engineering

### Data Preparation

* Removed non-predictive fields
* Encoded categorical variables using LabelEncoder
* Standardized numerical features using StandardScaler
* Train/Test Split: 75/25

### Features Used

```text
Project Type
Project Stage
Invoice Amount
Recurring Expense
Useful Life Expectancy
```

---

## Technology Stack

### Machine Learning

* Python
* Scikit-Learn
* XGBoost
* Pandas
* NumPy

### API Layer

* Flask
* Joblib

### Cloud

* Microsoft Azure Web Apps

### CRM

* Salesforce
* Salesforce Flow
* External Services
* Named Credentials

### Reporting

* Salesforce Analytics Studio

---

## API Endpoint

### Request

```http
POST /predict
```

Example payload:

```json
{
  "Project Type": "Software Development",
  "Project Stage": "Execution",
  "Invoice Amount": 25000,
  "Recurring Expense?": false,
  "Useful Life Expectancy": 5
}
```

### Response

```json
{
  "prediction_code": 1,
  "prediction_label": "CapEx"
}
```

---

## Model Deployment

The trained model is serialized using Joblib:

```python
joblib.dump(best_model, "model.pkl")
joblib.dump(label_encoders, "label_encoder.pkl")
```

The Flask application loads both artifacts at startup and exposes a REST API for real-time predictions.

```python
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoder.pkl")
```

---

## Salesforce Integration

### External Service

Salesforce connects to the Azure-hosted API through:

* OpenAPI Specification
* External Services
* Named Credentials

### Flow Automation

The flow:

1. Retrieves invoices requiring classification
2. Calls the Azure prediction API
3. Receives CapEx/OpEx prediction
4. Updates Salesforce records automatically

This enables near real-time invoice classification without user intervention.

---

## Dashboard

A Salesforce Analytics Studio dashboard was developed to provide:

### Features

* YTD CapEx vs OpEx summary
* Department expenditure analysis
* Project expenditure analysis
* Monthly spending trends
* Interactive filters
* Invoice detail table

Users can drill down by:

* Project
* Department
* Region
* Date range

---

## Results

### End-to-End Solution Accuracy

| Approach                    | Accuracy |
| --------------------------- | -------- |
| Rule-Based Logic            | 94.24%   |
| Salesforce Analytics Studio | 75.72%   |
| External ML Service         | 99.18%   |

The external machine learning service achieved the highest accuracy and provided the greatest flexibility for future enhancements.

---

## Future Enhancements

### Model Improvements

* Train using real production invoice data
* Automatic retraining pipeline
* User feedback loop for corrections
* Model monitoring

### NLP Enhancements

Use invoice descriptions as additional features:

* TF-IDF
* Word Embeddings
* BERT-based encodings

### Security

Production deployment should include:

* OAuth 2.0
* API Keys
* Azure Authentication
* Salesforce Named Credential security policies

---

## Repository Structure

```text
.
├── app.py
├── model.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── Classification_Model.py
```

---

## Authors
* Yaozhong Shi (Tim Shi)
* Marylyn Chen
* Nivedita Minjur


Georgia Institute of Technology

Master of Science in Analytics (OMSA)

---

## License

This repository is provided for educational and portfolio purposes. Please contact the authors before commercial use.
