# main.py - Main application entry point
import argparse
import sys
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import traceback
from werkzeug.utils import secure_filename

from config import DATABASE_PATH, CSV_FILE_PATH, MODEL_PATH, TRAINING_CSV_PATH, CSV_SYNC_ON_STARTUP
from database import get_db_connection, init_db, import_csv_to_db, get_fight_data_for_training
from model import UFCPredictor, predict_fight
from data_processor import preprocess_data, create_advantage_features, select_features
from data_loader import preprocess_dataset, create_sample_dataset
from utils import feature_engineering, get_fighter_stats, calculate_elo_ratings
from csv_sync import verify_csv_consistency, sync_csv_files

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

# Initialize CSV files if configured
if CSV_SYNC_ON_STARTUP:
    try:
        # Verify that CSV files are consistent on startup
        if not verify_csv_consistency():
            print("CSV files are inconsistent or missing. Attempting to synchronize...")
            sync_csv_files()
    except Exception as e:
        print(f"Error checking CSV consistency on startup: {e}")
        traceback.print_exc()

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
        
        # Also generate training data if it doesn't exist
        if not os.path.exists(TRAINING_CSV_PATH):
            sync_csv_files()
        
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
        
        # Get fight data for training - will use training CSV if available
        fight_data = get_fight_data_for_training()
        
        if fight_data.empty:
            return jsonify({"error": "No fight data available for training."}), 400
        
        # Apply feature engineering
        fight_data = feature_engineering(fight_data)
        
        # Preprocess data for model
        processed_data = preprocess_data(fight_data)
        processed_data = create_advantage_features(processed_data)
        processed_data = select_features(processed_data)
        
        # Prepare data for training
        from data_processor import prepare_training_data, create_dataloaders
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(processed_data)
        train_loader, val_loader, test_loader, scaler, feature_columns = create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Save feature columns and scaler for later use
        import joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(feature_columns, os.path.join(os.path.dirname(MODEL_PATH), 'feature_columns.pkl'))
        joblib.dump(scaler, os.path.join(os.path.dirname(MODEL_PATH), 'scaler.pkl'))
        
        # Create and train model
        from data_processor import train_model as train_model_function
        model = UFCPredictor(input_size=len(feature_columns))
        model, history = train_model_function(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_path=MODEL_PATH
        )
        
        # Evaluate model
        from data_processor import evaluate_model
        metrics = evaluate_model(model, test_loader)
        
        return jsonify({
            "message": "Model trained successfully",
            "model_path": MODEL_PATH,
            "metrics": metrics
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
            return jsonify({"error": "Model not trained. Please train the model first."}), 400
        
        # Get fighters data from request
        data = request.get_json()
        
        if not data or 'fighter1' not in data or 'fighter2' not in data:
            return jsonify({"error": "Invalid request. fighter1 and fighter2 data required."}), 400
        
        # Format fighter data for the model
        fighter1 = data['fighter1']
        fighter2 = data['fighter2']
        
        # Load the model and required files
        import torch
        import joblib
        
        model = UFCPredictor(input_size=len(joblib.load(os.path.join(os.path.dirname(MODEL_PATH), 'feature_columns.pkl'))))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        scaler = joblib.load(os.path.join(os.path.dirname(MODEL_PATH), 'scaler.pkl'))
        feature_columns = joblib.load(os.path.join(os.path.dirname(MODEL_PATH), 'feature_columns.pkl'))
        
        # Make prediction
        prediction = predict_fight(model, fighter1, fighter2, scaler, feature_columns)
        
        # Format the response
        response = {
            "prediction": {
                "fighter1_name": fighter1.get('name', 'Fighter 1'),
                "fighter2_name": fighter2.get('name', 'Fighter 2'),
                "fighter1_win_probability": prediction['probability_red_wins'],
                "fighter2_win_probability": prediction['probability_blue_wins'],
                "predicted_winner": "fighter1" if prediction['predicted_winner'] == 'Red' else "fighter2",
                "confidence_level": prediction['confidence_level']
            }
        }
        
        return jsonify(response)
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

@app.route('/api/manage-csv', methods=['GET', 'POST'])
def manage_csv_files():
    """API endpoint to manage CSV files (verify, sync, etc.)"""
    try:
        if request.method == 'GET':
            # Check CSV consistency and return status
            is_consistent = verify_csv_consistency()
            main_exists = os.path.exists(CSV_FILE_PATH)
            training_exists = os.path.exists(TRAINING_CSV_PATH)
            
            return jsonify({
                "status": "ok",
                "is_consistent": is_consistent,
                "main_csv_exists": main_exists,
                "training_csv_exists": training_exists,
                "main_csv_path": CSV_FILE_PATH,
                "training_csv_path": TRAINING_CSV_PATH
            })
        
        elif request.method == 'POST':
            # Process action from request
            data = request.get_json()
            action = data.get('action')
            
            if action == 'sync':
                # Synchronize CSV files
                sync_csv_files()
                return jsonify({
                    "status": "ok",
                    "message": "CSV files synchronized successfully"
                })
            else:
                return jsonify({
                    "error": f"Unknown action: {action}"
                }), 400
                
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
        
        # Check if this is specifically for training data
        is_training = request.form.get('is_training', 'false').lower() == 'true'
        
        if file and file.filename.endswith('.csv'):
            # Save the file to the appropriate location
            filename = secure_filename(file.filename)
            
            if is_training:
                filepath = TRAINING_CSV_PATH
                file.save(filepath)
                message = "Training CSV file uploaded successfully"
            else:
                filepath = CSV_FILE_PATH
                file.save(filepath)
                
                # Clear and reinitialize the database if this is the main data
                if os.path.exists(DATABASE_PATH):
                    os.remove(DATABASE_PATH)
                
                init_db()
                import_csv_to_db()
                
                # Also update the training data
                sync_csv_files()
                
                message = "Main CSV file uploaded and processed successfully"
            
            return jsonify({"message": message})
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
def parse_args():
    """Parse command line arguments for UFC Predictor"""
    parser = argparse.ArgumentParser(description="UFC Fight Predictor Backend")
    parser.add_argument('--init-and-train', action='store_true', 
                      help='Initialize database and train model before starting the server')
    parser.add_argument('--init-db', action='store_true',
                      help='Initialize the database only')
    parser.add_argument('--train', action='store_true',
                      help='Train the model only')
    parser.add_argument('--sync-csv', action='store_true',
                      help='Synchronize CSV files')
    parser.add_argument('--no-server', action='store_true',
                      help='Do not start the Flask server after other operations')
    return parser.parse_args()

# Replace the "if __name__ == '__main__':" section with this:
if __name__ == '__main__':
    args = parse_args()
    
    # Process CLI arguments
    if args.sync_csv:
        print("Synchronizing CSV files...")
        sync_csv_files()
    
    if args.init_db or args.init_and_train:
        print("Initializing database...")
        if not os.path.exists(CSV_FILE_PATH):
            print(f"Main CSV file not found at {CSV_FILE_PATH}. Creating sample dataset...")
            create_sample_dataset()
        
        init_db()
        import_csv_to_db()
        print("Database initialized successfully")
    
    if args.train or args.init_and_train:
        print("Training model...")
        # Get fight data for training
        fight_data = get_fight_data_for_training()
        
        if fight_data.empty:
            print("No fight data available for training.")
            sys.exit(1)
        
        # Apply feature engineering
        fight_data = feature_engineering(fight_data)
        
        # Preprocess data for model
        processed_data = preprocess_data(fight_data)
        
        # Train the model
        trainer = ModelTrainer(processed_data)
        history, metrics = trainer.train_model()
        
        print(f"Model trained successfully and saved to {MODEL_PATH}")
        print(f"Test metrics: {metrics}")
    
    # Start the server unless --no-server was specified
    if not args.no_server:
        print("Starting UFC Predictor API server...")
        app.run(debug=False)
    else:
        print("Server start skipped due to --no-server flag")
# Run the app
if __name__ == '__main__':
    app.run(debug=False)