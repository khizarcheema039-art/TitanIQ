# The Passenger Ledger

The Passenger Ledger is an interactive Titanic passenger manifest. Enter a passenger's recorded details to see an estimate from a logistic regression model trained on the Kaggle Titanic training data.

The estimate is a teaching aid based on a small historical dataset. It is not a claim about what would have happened to any individual passenger.

## Features

- Passenger inputs for ticket class, sex, age, family aboard, fare, and port of embarkation
- Survival estimates from a Flask API
- A historical comparison drawn from passengers with similar class, sex, and port entries
- A React interface built with Vite

## Requirements

- Python 3.10 or newer
- Node.js 20.19 or newer (or 22.12 or newer) and npm

## Run locally

From the project directory, install the frontend dependencies:

```sh
npm ci
```

Create and activate a Python virtual environment, then install the API dependency:

```sh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Then install Flask:

```sh
python -m pip install -r requirements.txt
```

Start the API in one terminal:

```sh
python app.py
```

Start the frontend in a second terminal:

```sh
npm run dev
```

Open the local address printed by Vite. The development server forwards `/api` requests to the Flask service on `127.0.0.1:5000`.

## Production build

Build the frontend and serve it locally with Flask:

```sh
npm run build
python app.py
```

Flask serves the built files from `dist`. Set the `PORT` environment variable to change its port; the default is `5000`.

## Data and model

The app reads `train.csv` when `app.py` starts and trains a logistic regression model using seven fields: ticket class, sex, age, siblings or spouses aboard, parents or children aboard, fare, and embarkation port. Missing ages are filled with the training-set median, missing fares with zero, and missing ports with the first recorded port. The `test.csv` file and `titanic survival prediction.ipynb` notebook are included as project data and reference material; the app does not use them to train the model.

The interface also displays a fixed plus-or-minus 7.5 percentage point range. It is a presentation aid, not a statistically calibrated confidence interval.

The dataset comes from Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) competition. Review Kaggle's terms before redistributing the data. This repository does not currently include a software license; without one, standard copyright applies to the project code.

## Project files

- `app.py` — Flask API and model training
- `src/` — React interface and styles
- `train.csv` — labeled training data used by the app
- `test.csv` — competition test data
- `titanic survival prediction.ipynb` — exploratory notebook

