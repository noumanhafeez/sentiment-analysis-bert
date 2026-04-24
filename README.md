# AI Sentiment Analyzer using Fine-Tuned BERT

A complete NLP web application that performs sentiment analysis using a fine-tuned **BERT (bert-base-uncased)** model on the IMDB movie reviews dataset.

This project includes:

- Model training using HuggingFace Transformers
- Fine-tuning BERT for binary sentiment classification
- FastAPI web application
- Single text prediction
- Bulk file upload (CSV / JSON / TXT)
- Sentiment analytics dashboard
- REST API endpoint

---

# Features

## AI / NLP Features

- Fine-tuned `bert-base-uncased`
- Binary classification:
  - POSITIVE
  - NEGATIVE
- Confidence score output
- IMDB dataset used for training

## Web App Features

- Beautiful UI with FastAPI HTML responses
- Single review sentiment prediction
- Bulk sentiment analysis from files
- Pie chart sentiment distribution
- Tabular report preview

## File Support

Upload and analyze:

- CSV
- JSON
- TXT

---

# Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Deep Learning | PyTorch |
| NLP | HuggingFace Transformers |
| Dataset | IMDB |
| Backend | FastAPI |
| Visualization | Matplotlib |
| Data Handling | Pandas |

---

# Project Structure

```bash
sentiment-analysis-bert/
│── config/
│   └── config.py
│
│── frontend/
│   └── app.py
│
│── saved_model/
│
│── src/
│   ├── dataset.py
│   ├── model.py
│   ├── prediction.py
│   └── trainer.py
│
│── utils/
│   └── logger.py
│
│── main.py
│── requirements.txt

```


## Configuration

File: config/config.py

```python
MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
NUM_LABELS = 2

TRAIN_SIZE = 5000
TEST_SIZE = 1000

BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

OUTPUT_DIR = "./saved_model"
LOG_DIR = "./logs"
```

## Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

### Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Requirements
```bash
pip install -r requirements.txt
```

## How to Run

### Run Prediction Script
```bash
python main.py
```

Example Output:
```json
{
  "text": "This movie was absolutely bad!",
  "label": "NEGATIVE",
  "confidence": 0.9842
}
```

### Run FastAPI App
```bash
uvicorn frontend.app:app --reload
```

Open browser: http://127.0.0.1:8000

## API Endpoints

### Home Page
GET /

Returns web interface.

### Predict Sentiment
POST /predict

Request:
```json
{
  "text": "This movie was amazing"
}
```

Response:
```json
{
  "text": "This movie was amazing",
  "label": "POSITIVE",
  "confidence": 0.9971
}
```

### Predict Using Form
POST /predict-form

Used by frontend form.

### Bulk Upload
POST /upload-file

Supports:
- CSV
- JSON
- TXT

Returns sentiment analytics report.

## Training Pipeline

1. Load IMDb dataset
2. Shuffle data
3. Tokenize text using BERT tokenizer
4. Load pre-trained BERT model
5. Fine-tune model using Trainer API
6. Evaluate accuracy
7. Save trained model

## Inference Flow

Input Text → Tokenizer → Fine-tuned BERT → Softmax → Prediction Label → Confidence Score

## Example Use Cases

- Product review sentiment analysis
- Customer feedback monitoring
- Movie review classification
- Social media analysis
- Brand reputation monitoring
- Survey response understanding

## Example Bulk Analysis Report

Uploaded 100 reviews:

Positive: 72 (72.0%)
Negative: 28 (28.0%)

Generated:
- Pie chart
- Table of predictions
- Confidence scores

## Why This Project is Valuable

This project proves your skills in:

- NLP
- Transformers
- Deep Learning
- API Development
- Python Backend
- Real-world Deployment
- Clean Code Architecture

## Future Improvements

- Positive / Neutral / Negative classes
- Docker containerization
- AWS deployment
- CI/CD pipeline
- Login system
- Report history
- Database integration
- Admin dashboard
- Real-time prediction API

## Author

Nouman Hafeez

Software Engineer | AI Enthusiast | NLP Developer