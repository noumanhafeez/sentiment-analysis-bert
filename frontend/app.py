# app.py

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.prediction import SentimentPredictor

import pandas as pd
import json
import io
import base64
import matplotlib
matplotlib.use("Agg")  # ✅ Fix for Docker
import matplotlib.pyplot as plt

import logging

# -----------------------------------
# Logging
# -----------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------
# Init App
# -----------------------------------
app = FastAPI(title="AI Sentiment Analyzer")
predictor = SentimentPredictor()

MAX_ROWS = 1000  # ✅ Prevent overload


# -----------------------------------
# API Schema
# -----------------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------------
# Health Check
# -----------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------------
# Helpers
# -----------------------------------
def make_chart(labels, values):
    try:
        plt.figure(figsize=(7, 5))
        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Sentiment Distribution")

        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        plt.close()
        img.seek(0)

        chart_base64 = base64.b64encode(img.read()).decode()
        return chart_base64

    except Exception as e:
        logging.error(f"Chart error: {e}")
        return None


def analyze_texts(texts):
    if not texts:
        return {
            "results": [],
            "total": 0,
            "positive": 0,
            "negative": 0,
            "pos_percent": 0,
            "neg_percent": 0,
            "chart": None
        }

    results = []
    positive = 0
    negative = 0

    texts = texts[:MAX_ROWS]  # ✅ Limit size

    for text in texts:
        try:
            pred = predictor.predict(str(text))
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            pred = {
                "text": str(text),
                "label": "error",
                "confidence": 0
            }

        results.append(pred)

        if pred["label"].lower() == "positive":
            positive += 1
        else:
            negative += 1

    total = len(results)

    pos_percent = round((positive / total) * 100, 2) if total else 0
    neg_percent = round((negative / total) * 100, 2) if total else 0

    chart = make_chart(
        ["Positive", "Negative"],
        [positive, negative]
    )

    return {
        "results": results,
        "total": total,
        "positive": positive,
        "negative": negative,
        "pos_percent": pos_percent,
        "neg_percent": neg_percent,
        "chart": chart
    }


# -----------------------------------
# Home Page
# -----------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>AI Sentiment Analyzer</title>
        <style>
            body{
                font-family:Arial;
                background:#0f172a;
                color:white;
                padding:40px;
                max-width:900px;
                margin:auto;
            }

            .card{
                background:#1e293b;
                padding:30px;
                border-radius:20px;
                margin-bottom:30px;
            }

            textarea,input,button{
                width:100%;
                padding:14px;
                margin-top:10px;
                border-radius:10px;
                border:none;
                font-size:16px;
            }

            button{
                background:#3b82f6;
                color:white;
                font-weight:bold;
                cursor:pointer;
            }

            h1,h2{
                margin-bottom:10px;
            }

            p{
                color:#cbd5e1;
            }
        </style>
    </head>

    <body>

        <h1>🤖 AI Sentiment Analyzer</h1>
        <p>Analyze single text or upload files for bulk sentiment analytics.</p>

        <div class="card">
            <h2>Single Review</h2>

            <form action="/predict-form" method="post">
                <textarea name="text" rows="6"
                placeholder="Write your review here..."></textarea>

                <button type="submit">Analyze Text</button>
            </form>
        </div>

        <div class="card">
            <h2>Bulk File Upload</h2>
            <p>Supported: CSV, JSON, TXT</p>

            <form action="/upload-file" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Upload & Analyze</button>
            </form>
        </div>

    </body>
    </html>
    """


# -----------------------------------
# Single Text UI
# -----------------------------------
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(text: str = Form(...)):
    try:
        result = predictor.predict(text)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        result = {"text": text, "label": "error", "confidence": 0}

    return f"""
    <html>
    <body style="font-family:Arial;padding:40px;background:#0f172a;color:white;">
        <h1>Prediction Result</h1>

        <p><b>Text:</b> {result["text"]}</p>
        <p><b>Label:</b> {result["label"]}</p>
        <p><b>Confidence:</b> {result["confidence"]}</p>

        <br>
        <a href="/" style="color:#3b82f6;">← Back</a>
    </body>
    </html>
    """


# -----------------------------------
# Bulk Upload
# -----------------------------------
@app.post("/upload-file", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        content = await file.read()

        texts = []

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            texts = df.iloc[:, 0].dropna().astype(str).tolist()

        elif filename.endswith(".json"):
            data = json.loads(content.decode())

            if isinstance(data, list):
                texts = [str(item) for item in data]
            else:
                texts = [str(data)]

        elif filename.endswith(".txt"):
            lines = content.decode().splitlines()
            texts = [line.strip() for line in lines if line.strip()]

        else:
            return "<h1>Unsupported File</h1><a href='/'>Back</a>"

        report = analyze_texts(texts)

        rows = ""
        for i, item in enumerate(report["results"][:20], start=1):
            rows += f"""
            <tr>
                <td>{i}</td>
                <td>{item["text"][:70]}</td>
                <td>{item["label"]}</td>
                <td>{item["confidence"]}</td>
            </tr>
            """

        chart_html = ""
        if report["chart"]:
            chart_html = f'<img src="data:image/png;base64,{report["chart"]}">'

        return f"""
        <html>
        <body style="font-family:Arial;padding:40px;background:#0f172a;color:white;">

            <h1>📊 Bulk Analysis Report</h1>

            <p>Total: {report["total"]}</p>
            <p>Positive: {report["positive"]} ({report["pos_percent"]}%)</p>
            <p>Negative: {report["negative"]} ({report["neg_percent"]}%)</p>

            {chart_html}

            <h2>Sample Results</h2>
            <table border="1" cellpadding="8">
                <tr>
                    <th>#</th>
                    <th>Text</th>
                    <th>Label</th>
                    <th>Confidence</th>
                </tr>
                {rows}
            </table>

            <br><a href="/">← Back</a>

        </body>
        </html>
        """

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return "<h1>Error processing file</h1><a href='/'>Back</a>"


# -----------------------------------
# JSON API
# -----------------------------------
@app.post("/predict")
def predict_api(request: TextRequest):
    try:
        return predictor.predict(request.text)
    except Exception as e:
        logging.error(f"API error: {e}")
        return {"error": "Prediction failed"}