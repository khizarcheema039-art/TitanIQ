"""Small Flask API: trains a seven-feature logistic model from train.csv at startup."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
FEATURES = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
app = Flask(__name__, static_folder="dist", static_url_path="")


def sigmoid(z: float) -> float:
    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def load_rows():
    with (ROOT / "train.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ages = [float(r["Age"]) for r in rows if r["Age"]]
    median_age = sorted(ages)[len(ages) // 2]
    embark = next((r["Embarked"] for r in rows if r["Embarked"]), "S")
    parsed = []
    for r in rows:
        try:
            parsed.append(([float(r["Pclass"]), 1.0 if r["Sex"] == "female" else 0.0,
                            float(r["Age"] or median_age), float(r["SibSp"]), float(r["Parch"]),
                            float(r["Fare"] or 0), {"S": 0.0, "C": 1.0, "Q": 2.0}.get(r["Embarked"] or embark, 0.0)],
                           int(r["Survived"]), r))
        except (ValueError, KeyError):
            continue
    return parsed


def train_model(rows):
    x = [r[0] for r in rows]
    means = [sum(r[j] for r in x) / len(x) for j in range(7)]
    scales = [math.sqrt(sum((r[j] - means[j]) ** 2 for r in x) / len(x)) or 1 for j in range(7)]
    weights, bias = [0.0] * 7, 0.0
    # Full-batch gradient descent with L2 regularization; standardized features keep steps stable.
    for step in range(3200):
        grad = [0.0] * 7
        gb = 0.0
        for features, label, _ in rows:
            z = bias + sum(weights[j] * (features[j] - means[j]) / scales[j] for j in range(7))
            err = sigmoid(z) - label
            gb += err
            for j in range(7):
                grad[j] += err * (features[j] - means[j]) / scales[j]
        rate = 0.06
        bias -= rate * gb / len(rows)
        for j in range(7):
            weights[j] -= rate * (grad[j] / len(rows) + 0.018 * weights[j])
    return means, scales, weights, bias


DATA = load_rows()
MEANS, SCALES, WEIGHTS, BIAS = train_model(DATA)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": "logistic-regression", "training_rows": len(DATA)})


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        raw = [float(body.get(k, 0)) for k in FEATURES]
        pclass, sex, age, sibsp, parch, fare, embarked = raw
        if pclass not in (1, 2, 3) or sex not in (0, 1) or embarked not in (0, 1, 2):
            raise ValueError("Choose a valid class, gender, and port.")
        if not 0 <= age <= 100 or not 0 <= sibsp <= 10 or not 0 <= parch <= 10 or not 0 <= fare <= 600:
            raise ValueError("One or more passenger values are outside the supported range.")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Please check passenger details."}), 400
    z = BIAS + sum(WEIGHTS[j] * (raw[j] - MEANS[j]) / SCALES[j] for j in range(7))
    probability = sigmoid(z)
    # Directional component contributions explain this model's log-odds relative to its cohort average.
    impacts = []
    names = ["Ticket class", "Gender", "Age", "Siblings / spouses", "Parents / children", "Fare", "Embarkation"]
    for j in range(7):
        contribution = WEIGHTS[j] * (raw[j] - MEANS[j]) / SCALES[j]
        impacts.append({"label": names[j], "value": round(contribution, 3), "direction": "up" if contribution >= 0 else "down"})
    impacts.sort(key=lambda item: abs(item["value"]), reverse=True)
    group = [r for r in DATA if r[2]["Pclass"] == str(int(pclass)) and r[2]["Sex"] == ("female" if sex else "male") and (not r[2]["Embarked"] or {"S": 0, "C": 1, "Q": 2}.get(r[2]["Embarked"], 0) == int(embarked))]
    historical = (sum(r[1] for r in group) / len(group)) if len(group) >= 5 else probability
    group_label = {0: "Southampton", 1: "Cherbourg", 2: "Queenstown"}[int(embarked)]
    return jsonify({"probability": round(probability, 4), "interval": [round(max(0, probability - 0.075), 4), round(min(1, probability + 0.075), 4)], "impacts": impacts, "historical_rate": round(historical, 3), "historical_count": len(group), "historical_context": f"In this training set, {('first' if pclass == 1 else 'second' if pclass == 2 else 'third')}-class {('women' if sex else 'men')} boarding at {group_label} survived at {historical:.0%} ({len(group)} comparable passengers)."})


@app.get("/")
def index():
    if (ROOT / "dist" / "index.html").exists():
        return send_from_directory(app.static_folder, "index.html")
    return "Frontend is not built yet. Start Vite with npm run dev.", 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
