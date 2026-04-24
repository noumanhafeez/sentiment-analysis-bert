# app.py

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.prediction import SentimentPredictor

import pandas as pd
import json
import io
import base64
import matplotlib.pyplot as plt

# -----------------------------------
# Init App
# -----------------------------------
app = FastAPI(title="AI Sentiment Analyzer")
predictor = SentimentPredictor()


# -----------------------------------
# API Schema
# -----------------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------------
# Helpers
# -----------------------------------
def make_chart(labels, values):
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


def analyze_texts(texts):
    results = []
    positive = 0
    negative = 0

    for text in texts:
        pred = predictor.predict(str(text))
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
    result = predictor.predict(text)

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
    filename = file.filename.lower()
    content = await file.read()

    texts = []

    # CSV
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))

        # first column used
        texts = df.iloc[:, 0].dropna().astype(str).tolist()

    # JSON
    elif filename.endswith(".json"):
        data = json.loads(content.decode())

        if isinstance(data, list):
            for item in data:
                texts.append(str(item))
        else:
            texts.append(str(data))

    # TXT
    elif filename.endswith(".txt"):
        lines = content.decode().splitlines()
        texts = [line.strip() for line in lines if line.strip()]

    else:
        return """
        <h1>Unsupported File</h1>
        <a href="/">Back</a>
        """

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

    return f"""
    <html>
    <head>
        <style>
            body{{
                font-family:Arial;
                padding:40px;
                background:#0f172a;
                color:white;
            }}

            .box{{
                background:#1e293b;
                padding:25px;
                border-radius:18px;
                margin-bottom:25px;
            }}

            table{{
                width:100%;
                border-collapse:collapse;
                margin-top:20px;
            }}

            td,th{{
                border:1px solid #334155;
                padding:10px;
                text-align:left;
            }}

            th{{
                background:#334155;
            }}

            img{{
                max-width:420px;
                border-radius:15px;
                margin-top:15px;
            }}

            a{{
                color:#3b82f6;
            }}
        </style>
    </head>

    <body>

        <h1>📊 Bulk Analysis Report</h1>

        <div class="box">
            <p><b>Total Records:</b> {report["total"]}</p>
            <p><b>Positive:</b> {report["positive"]} ({report["pos_percent"]}%)</p>
            <p><b>Negative:</b> {report["negative"]} ({report["neg_percent"]}%)</p>

            <img src="data:image/png;base64,{report["chart"]}">
        </div>

        <div class="box">
            <h2>Sample Results</h2>

            <table>
                <tr>
                    <th>#</th>
                    <th>Text</th>
                    <th>Label</th>
                    <th>Confidence</th>
                </tr>

                {rows}
            </table>
        </div>

        <a href="/">← Back Home</a>

    </body>
    </html>
    """


# -----------------------------------
# JSON API
# -----------------------------------
@app.post("/predict")
def predict_api(request: TextRequest):
    return predictor.predict(request.text)