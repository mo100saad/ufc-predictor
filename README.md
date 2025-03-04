<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg" width="100" height="100" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" width="100" height="100" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="100" height="100" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/Tailwind_CSS_Logo.svg" width="100" height="100" />
</p>

# Mo's UFC Predictor

UFC Predictor is a machine learning-powered web application that predicts fight outcomes in the Ultimate Fighting Championship (UFC). Using advanced neural networks and a comprehensive database of fighter statistics, this tool provides data-driven predictions for upcoming UFC matchups with detailed probability analysis.

### Live Demo: COMING SOON

## Features

### AI-Driven Fight Predictions
- **Neural Network Model**: Trained on thousands of historical UFC fights
- **Multi-Strategy Prediction**: Uses multiple prediction approaches for accurate results
- **Probability Analysis**: Detailed win probability with confidence ratings
- **Physical Matchup Evaluation**: Accounts for weight class, reach, and other physical advantages

### Fighter Database
- **Comprehensive Statistics**: Track records, physical attributes, and performance metrics
- **Fighter Comparisons**: Side-by-side comparison of fighting styles and statistics
- **Historical Data**: Access to past UFC fight results for analysis

### Modern Web Interface
- **Responsive Design**: Built with React and Tailwind CSS for a seamless experience on all devices
- **Interactive Visualizations**: Dynamic charts for prediction and fighter comparison
- **User-Friendly**: Intuitive interface for both casual fans and MMA analysts

## Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework for fight prediction model
- **Flask**: REST API framework
- **SQLite**: Database for fighter and fight data
- **Pandas/NumPy**: Data processing and analysis

### Frontend
- **React**: UI component library
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Data visualization
- **Axios**: HTTP client for API communication

## Installation and Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm 7+

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/mo100saad/ufc-predictor.git
cd ufc-predictor/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database and train the model
python main.py --init-and-train

# Start the Flask server
python main.py
```

### Frontend Setup
```bash
# Navigate to the frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```
The application will be available at `http://localhost:3000` with the API running at `http://localhost:5000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/api/health` | Health check for API status |
| POST | `/api/predict` | Predicts fight outcome between two fighters |
| GET | `/api/fighters` | Returns list of all fighters in the database |
| GET | `/api/fighter/{name}` | Retrieves detailed stats for a specific fighter |
| GET | `/api/fights` | Returns list of recorded fights |
| POST | `/api/manage-csv` | Manages CSV data synchronization |

### Example API Request (Predict Fight)
```json
{
  "fighter1": {
    "name": "Conor McGregor",
    "height": 175,
    "weight": 155,
    "reach": 188,
    "stance": "Southpaw",
    "wins": 22,
    "losses": 6,
    "sig_strikes_per_min": 5.32,
    "takedown_avg": 0.7
  },
  "fighter2": {
    "name": "Khabib Nurmagomedov",
    "height": 178,
    "weight": 155,
    "reach": 178,
    "stance": "Orthodox",
    "wins": 29,
    "losses": 0,
    "sig_strikes_per_min": 4.10,
    "takedown_avg": 5.35
  }
}
```

### Example Response
```json
{
  "prediction": {
    "fighter1_name": "Conor McGregor",
    "fighter2_name": "Khabib Nurmagomedov",
    "fighter1_win_probability": 0.37,
    "fighter2_win_probability": 0.63,
    "predicted_winner": "fighter2",
    "confidence_level": "Medium"
  }
}
```

## Model Architecture

### Neural Network
- **2 fully-connected hidden layers** (64 → 32 neurons)
- **Dropout layers** (0.4, 0.3) for regularization
- **ReLU activation** for hidden layers, **sigmoid** for output

### Feature Engineering
- Direct fighter comparisons (weight, height, reach advantages)
- Performance metrics differentials (striking, grappling, etc.)
- Win record and streak analysis

### Position Bias Correction
- Direction-preserving data augmentation
- Multi-perspective prediction to remove corner bias
- Physical reality checks for extreme mismatches

## Deployment

### Frontend (AWS)
- **AWS Amplify**: Hosts the React frontend
- **Amazon CloudFront**: CDN for global content delivery
- **Route 53**: DNS management

### Backend (Render)
- **Render Web Services**: Hosts the Flask API
- **PostgreSQL Database**: Used in production (instead of SQLite)
- **Automated CI/CD**: Continuous deployment from GitHub

## Environment Variables

For local development, create a `.env` file in the backend directory with:
```ini
DEBUG=True
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///data/ufc_fights.db
```
For production deployment on Render, configure these environment variables in the Render dashboard.

## Future Roadmap

### Enhanced Prediction Model
- Fighter style matchup analysis
- Time-series analysis of fighter performance trends
- Inclusion of coach and camp data

### Advanced Features
- User accounts with saved predictions
- Live odds comparison with major sportsbooks
- Betting strategy recommendations

### Expanded Coverage
- Prediction for preliminary card fights
- Historical analysis of prediction accuracy
- Fighter career trajectory projections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/amazing-feature
# Commit your changes
git commit -m 'Add some amazing feature'
# Push to the branch
git push origin feature/amazing-feature
# Open a Pull Request
```
---

## 🔒 Copyright & Usage Disclaimer  

**This UFC Predictor project is the original work and intellectual property of Mohammad Saad.**  
It **may not be copied, redistributed, or used commercially** in any form without explicit permission. However, you are welcome to use it **as inspiration** for your own projects.  

If you wish to reference certain aspects of the design, functionality, or methodology, please provide **proper attribution**.  

---


## 📜 License
This project is **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** license. See the [LICENSE](LICENSE) file for details.

---

## Contact
**Mohammad Saad** - [@mo100saad](https://github.com/mo100saad)

Project Link: [GitHub Repository](https://github.com/mo100saad/ufc-predictor)
