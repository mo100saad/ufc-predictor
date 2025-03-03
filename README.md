# UFC Predictor

**Author:** Mohammad Saad  
**GitHub Repository:** [UFC Predictor](https://github.com/mo100saad/ufc-predictor/)  
**Tech Stack:** PyTorch, Flask, SQLite, Node.js  

## Overview

UFC Predictor is a **web-based machine learning application** designed to predict fight outcomes in the Ultimate Fighting Championship (UFC). The model utilizes **fighter statistics and historical fight data** to estimate the probability of victory for each competitor. The application integrates a **PyTorch-based neural network** for predictions, a **Flask REST API** for backend communication, and an **SQLite database** for efficient data storage and retrieval.

## Features

- **AI-Driven Predictions**  
  - A neural network model trained on historical UFC fight data to predict match outcomes.
  - Utilizes **PyTorch** for deep learning-based fight analysis.
  
- **RESTful API**  
  - Built with **Flask**, providing endpoints for users to input fighter statistics and receive predictions.
  - Supports JSON-based requests and responses.

- **Data Management with SQLite**  
  - Stores fighter statistics, fight history, and model outputs for future reference.
  - Enables efficient querying and retrieval of past fights.

- **Web Interface with React.js**  
  - Implements a **frontend UI** using **Node.js** for users to interact with the model.
  - Allows users to input fighter details and visualize predictions.

## Installation and Setup

### Prerequisites

Ensure that you have the following installed:

- Python 3.8+
- Flask
- PyTorch
- SQLite3
- Node.js & npm

### Step 1: Clone the Repository

```bash
git clone https://github.com/mo100saad/ufc-predictor.git
cd ufc-predictor
```

### Step 2: Install Dependencies

#### Backend (Python & Flask)
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend (Node.js)

```bash
cd frontend
npm install
```

### Step 3: Set Up the Database

Initialize the SQLite database:

```bash
python setup_database.py
```

### Step 4: Run the Application

#### Start the Flask Backend
```bash
python app.py
```

#### Start the Node.js Frontend
```bash
cd frontend
npm start
```

The application should now be accessible at `http://localhost:3000`.

## API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| POST   | `/predict`            | Predicts fight outcomes based on input data. |
| GET    | `/fighters`           | Retrieves stored fighter statistics. |
| POST   | `/add_fighter`        | Adds a new fighter to the database. |
| DELETE | `/delete_fighter/{id}` | Removes a fighter record. |

### Example API Request (Predict Fight)

```json
{
  "fighter1": {
    "name": "Conor McGregor",
    "height": 175,
    "weight": 70,
    "reach": 188,
    "win_streak": 5
  },
  "fighter2": {
    "name": "Khabib Nurmagomedov",
    "height": 178,
    "weight": 70,
    "reach": 178,
    "win_streak": 10
  }
}
```

### Example Response

```json
{
  "fighter1": {
    "name": "Conor McGregor",
    "win_probability": 35.2
  },
  "fighter2": {
    "name": "Khabib Nurmagomedov",
    "win_probability": 64.8
  }
}
```

## Model Training

The model is trained using historical UFC fight data, focusing on:

- Fighter physical attributes (height, reach, weight class)
- Recent fight records (win/loss streaks, knockouts, submissions)
- Opponent strength and ranking

Training is implemented in **PyTorch**, utilizing a multi-layer neural network to optimize predictions.

To retrain the model with new data:

```bash
python train_model.py
```

## Future Improvements

- **Enhanced Feature Engineering:** Include more variables such as fight styles, strikes landed per minute, and grappling statistics.
- **Data Augmentation:** Increase dataset size with enriched fight history.
- **Deployment:** Host the application on **AWS/GCP** for real-time predictions.
- **Improved UI:** Create an interactive visualization dashboard.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
