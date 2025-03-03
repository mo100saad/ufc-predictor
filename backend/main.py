# main.py - Main application entry point

import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import traceback
from werkzeug.utils import secure_filename

from config import DATABASE_PATH, CSV_FILE_PATH, MODEL_PATH
from database import get_db_connection, init_db, import_csv_to_db, get_fight_data_for_training
from model import ModelTrainer
from data_loader import preprocess_dataset, create_sample_dataset
from utils import feature_engineering, get_fighter_stats, calculate_elo_ratings

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

app.config['SECRET_KEY'] = 'ufc-predictor-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.dirname(CSV_FILE_PATH)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app)  # Enable Cross-Origin Resource Sharing

# Ensure directories exist
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

#------------------------
# API Routes
#------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is working"""
    return jsonify({"status": "ok"})

@app.route('/api/init-database', methods=['POST'])
def initialize_database():
    """Initialize the database and import data from CSV"""
    try:
        # Check if CSV file exists
        if not os.path.exists(CSV_FILE_PATH):
            # Create sample dataset if no file exists
            create_sample_dataset()
        
        # Initialize database schema
        init_db()
        
        # Import data from CSV
        import_csv_to_db()
        
        return jsonify({"message": "Database initialized successfully"})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the fight prediction model using the database data"""
    try:
        # Check if database exists
        if not os.path.exists(DATABASE_PATH):
            # Initialize database if it doesn't exist
            initialize_database()
        
        # Get fight data for training
        fight_data = get_fight_data_for_training()
        
        if fight_data.empty:
            return jsonify({"error": "No fight data available for training."}), 400
        
        # Apply feature engineering
        fight_data = feature_engineering(fight_data)
        
        # Train the model
        trainer = ModelTrainer(fight_data)
        trainer.train_model()
        
        return jsonify({
            "message": "Model trained successfully",
            "model_path": MODEL_PATH
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_fight():
    """Predict the outcome of a fight between two fighters"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            # Train model if it doesn't exist
            train_model()
        
        # Get fighters data from request
        data = request.get_json()
        
        if not data or 'fighter1' not in data or 'fighter2' not in data:
            return jsonify({"error": "Invalid request. fighter1 and fighter2 data required."}), 400
        
        # Format fighter data
        fighter1 = data['fighter1']
        fighter2 = data['fighter2']
        
        # Prepare feature dictionaries
        fighter1_features = {
            'fighter1_height': fighter1.get('height', 0),
            'fighter1_weight': fighter1.get('weight', 0),
            'fighter1_reach': fighter1.get('reach', 0),
            'fighter1_sig_strikes_per_min': fighter1.get('sig_strikes_per_min', 0),
            'fighter1_takedown_avg': fighter1.get('takedown_avg', 0),
            'fighter1_sub_avg': fighter1.get('sub_avg', 0),
            'fighter1_win_streak': fighter1.get('win_streak', 0)
        }
        
        fighter2_features = {
            'fighter2_height': fighter2.get('height', 0),
            'fighter2_weight': fighter2.get('weight', 0),
            'fighter2_reach': fighter2.get('reach', 0),
            'fighter2_sig_strikes_per_min': fighter2.get('sig_strikes_per_min', 0),
            'fighter2_takedown_avg': fighter2.get('takedown_avg', 0),
            'fighter2_sub_avg': fighter2.get('sub_avg', 0),
            'fighter2_win_streak': fighter2.get('win_streak', 0)
        }
        
        # Get fight data for training (needed to initialize the model with correct input size)
        fight_data = get_fight_data_for_training()
        
        # Apply feature engineering to ensure consistency with training data
        fight_data = feature_engineering(fight_data)
        
        # Make prediction
        trainer = ModelTrainer(fight_data)
        probability = trainer.predict_fight(fighter1_features, fighter2_features)
        
        return jsonify({
            "prediction": {
                "fighter1_name": fighter1.get('name', 'Fighter 1'),
                "fighter2_name": fighter2.get('name', 'Fighter 2'),
                "fighter1_win_probability": probability,
                "fighter2_win_probability": 1 - probability
            }
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/fighters', methods=['GET'])
def get_fighters():
    """Get a list of all fighters in the database"""
    try:
        # Check if database exists
        if not os.path.exists(DATABASE_PATH):
            return jsonify({"error": "Database not found. Please initialize database first."}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM fighters')
        fighters = cursor.fetchall()
        
        conn.close()
        
        # Convert to list of dictionaries
        fighters_list = []
        for fighter in fighters:
            fighters_list.append({
                'id': fighter['id'],
                'name': fighter['name'],
                'height': fighter['height'],
                'weight': fighter['weight'],
                'reach': fighter['reach'],
                'stance': fighter['stance'],
                'wins': fighter['wins'],
                'losses': fighter['losses'],
                'draws': fighter['draws']
            })
        
        return jsonify({"fighters": fighters_list})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/fights', methods=['GET'])
def get_fights():
    """Get a list of all fights in the database"""
    try:
        if not os.path.exists(DATABASE_PATH):
            return jsonify({"error": "Database not found. Please initialize database first."}), 400
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT f.*, 
               f1.name as fighter1_name, 
               f2.name as fighter2_name,
               w.name as winner_name
        FROM fights f
        JOIN fighters f1 ON f.fighter1_id = f1.id
        JOIN fighters f2 ON f.fighter2_id = f2.id
        LEFT JOIN fighters w ON f.winner_id = w.id
        ''')
        fights = cursor.fetchall()
        
        conn.close()
        
        # Convert to list of dictionaries
        fights_list = []
        for fight in fights:
            fights_list.append({
                'id': fight['id'],
                'fighter1_name': fight['fighter1_name'],
                'fighter2_name': fight['fighter2_name'],
                'weight_class': fight['weight_class'],
                'method': fight['method'],
                'rounds': fight['rounds'],
                'time': fight['time'],
                'date': fight['date'],
                'winner_name': fight['winner_name']
            })
        
        return jsonify({"fights": fights_list})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/fighter/<name>', methods=['GET'])
def get_fighter_by_name(name):
    """Get information about a specific fighter"""
    try:
        # Check if database exists
        if not os.path.exists(DATABASE_PATH):
            return jsonify({"error": "Database not found. Please initialize database first."}), 400
        
        fighter_stats = get_fighter_stats(name)
        
        if not fighter_stats:
            return jsonify({"error": f"Fighter '{name}' not found."}), 404
        
        return jsonify({"fighter": fighter_stats})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload and process a new CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and file.filename.endswith('.csv'):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'ufc_data.csv')
            file.save(filepath)
            
            # Clear and reinitialize the database
            if os.path.exists(DATABASE_PATH):
                os.remove(DATABASE_PATH)
            
            init_db()
            import_csv_to_db()
            
            return jsonify({"message": "CSV file uploaded and processed successfully"})
        else:
            return jsonify({"error": "File must be a CSV"}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

#------------------------
# UI Routes
#------------------------

@app.route('/', methods=['GET'])
def index():
    """Main page of the UFC Fight Predictor"""
    return render_template('index.html')

@app.route('/fighters', methods=['GET'])
def fighters_page():
    """Page showing all fighters"""
    return render_template('fighters.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    """Page for predicting fight outcomes"""
    return render_template('predict.html')

@app.route('/admin', methods=['GET'])
def admin_page():
    """Admin page for data management"""
    return render_template('admin.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=False)