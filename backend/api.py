from flask import Blueprint, request, jsonify
import os
import logging
import pandas as pd
import numpy as np
import joblib
import torch
from database import get_db_connection, search_fighters, get_fighter_details, get_all_fighters
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ufc_api')

# Configuration
DATA_DIR = 'data'
MODEL_DIR = 'models'
DATABASE_PATH = os.path.join(DATA_DIR, 'ufc_fighters.db')
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'pytorch_model.pth')
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.joblib')

# Create Blueprint for API endpoints
api_bp = Blueprint('api', __name__, url_prefix='/api')

#----------------
# Helper Functions
#----------------
def format_fighter_for_prediction(fighter_data):
    """Convert fighter data from database to format needed for prediction"""
    # Log the incoming fighter data to help debug
    logger.info(f"Formatting fighter data for prediction: {fighter_data}")
    
    # Create a mapping for column names
    column_mapping = {
        'name': 'name',
        'height': 'height',
        'weight': 'weight',
        'reach': 'reach',
        'stance': 'stance',
        'age': 'age',
        'wins': 'wins',
        'losses': 'losses',
        'SLpM': 'sig_strikes_per_min',
        'sig_str_acc': 'sig_strike_accuracy',
        'SApM': 'sig_strikes_absorbed_per_min',
        'str_def': 'sig_strike_defense',
        'td_avg': 'takedown_avg',
        'td_acc': 'takedown_accuracy',
        'td_def': 'takedown_defense',
        'sub_avg': 'sub_avg',
        # New mappings for enhanced data
        'win_by_ko': 'win_by_ko',
        'win_by_sub': 'win_by_sub',
        'win_by_dec': 'win_by_dec',
        'current_win_streak': 'current_win_streak',
        'current_lose_streak': 'current_lose_streak',
        'longest_win_streak': 'longest_win_streak',
        'weight_class_rank': 'weight_class_rank',
        'p4p_rank': 'p4p_rank',
        # Add frontend-specific mappings
        'sig_strikes_per_min': 'sig_strikes_per_min',
        'sig_strike_accuracy': 'sig_strike_accuracy',
        'sig_strikes_absorbed_per_min': 'sig_strikes_absorbed_per_min',
        'sig_strike_defense': 'sig_strike_defense',
        'takedown_avg': 'takedown_avg',
        'takedown_accuracy': 'takedown_accuracy',
        'takedown_defense': 'takedown_defense'
    }
    
    # Convert to dict with standardized column names for prediction
    prediction_data = {}
    for db_col, pred_col in column_mapping.items():
        if db_col in fighter_data:
            # Handle null values
            value = fighter_data[db_col]
            if value is None:
                if db_col in ['wins', 'losses', 'sig_strikes_per_min', 'takedown_avg', 'sub_avg']:
                    value = 0
                elif db_col in ['sig_strike_accuracy', 'sig_strike_defense', 'takedown_accuracy', 'takedown_defense']:
                    value = 50  # default to 50%
                else:
                    continue  # skip if null and not handled above
            
            prediction_data[pred_col] = value
    
    # Add aliases for win breakdown columns to ensure compatibility with UI
    if 'win_by_ko' in fighter_data and 'win_by_KO_TKO' not in prediction_data:
        prediction_data['win_by_KO_TKO'] = fighter_data['win_by_ko']
    
    if 'win_by_sub' in fighter_data and 'win_by_SUB' not in prediction_data:
        prediction_data['win_by_SUB'] = fighter_data['win_by_sub']
    
    if 'win_by_dec' in fighter_data and 'win_by_DEC' not in prediction_data:
        prediction_data['win_by_DEC'] = fighter_data['win_by_dec']
    
    logger.info(f"Formatted prediction data: {prediction_data}")
    return prediction_data

def get_fighter_stats(fighter_id):
    """Get fighter statistics from the database including fight history if available"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM fighters WHERE id = ?', (fighter_id,))
    fighter = cursor.fetchone()
    
    if fighter:
        # Convert row to dict
        fighter_dict = {}
        for idx, col in enumerate(cursor.description):
            fighter_dict[col[0]] = fighter[idx]
        
        # Get fight history if the fights table exists
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fights'")
            if cursor.fetchone():
                cursor.execute('''
                    SELECT 
                        f.id as fight_id,
                        f.date,
                        f.method,
                        f.rounds,
                        f.weight_class,
                        CASE 
                            WHEN f.fighter1_id = ? THEN f2.name 
                            ELSE f1.name 
                        END as opponent,
                        CASE 
                            WHEN f.winner_id = ? THEN 'Win'
                            WHEN f.winner_id IS NULL THEN 'Draw'
                            ELSE 'Loss'
                        END as result
                    FROM 
                        fights f
                    JOIN 
                        fighters f1 ON f.fighter1_id = f1.id
                    JOIN 
                        fighters f2 ON f.fighter2_id = f2.id
                    WHERE 
                        f.fighter1_id = ? OR f.fighter2_id = ?
                    ORDER BY 
                        f.date DESC
                ''', (fighter_id, fighter_id, fighter_id, fighter_id))
                
                fighter_dict['fight_history'] = []
                for row in cursor.fetchall():
                    fight = {}
                    for idx, col in enumerate(cursor.description):
                        fight[col[0]] = row[idx]
                    fighter_dict['fight_history'].append(fight)
        except Exception as e:
            logger.warning(f"Could not get fight history: {e}")
        
        conn.close()
        return fighter_dict
    
    conn.close()
    return None

#----------------
# API Routes
#----------------

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@api_bp.route('/fighters', methods=['GET'])
def get_fighters():
    """Get all fighters endpoint"""
    try:
        fighters_data = get_all_fighters()
        
        # Clean fighters data to ensure JSON serializable
        cleaned_fighters = []
        for fighter in fighters_data:
            cleaned_fighter = {}
            for key, value in fighter.items():
                # Handle binary data
                if isinstance(value, bytes):
                    try:
                        # Try to decode, or set to None if not possible
                        cleaned_fighter[key] = value.decode('utf-8', errors='ignore') or None
                    except:
                        cleaned_fighter[key] = None
                else:
                    cleaned_fighter[key] = value
            cleaned_fighters.append(cleaned_fighter)
            
        return jsonify({"fighters": cleaned_fighters})
    except Exception as e:
        logger.error(f"Error getting fighters: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/fighters/search', methods=['GET'])
def search_fighters_endpoint():
    """Search fighters endpoint"""
    query = request.args.get('q', '')
    if not query or len(query) < 2:  # Allow shorter search terms
        return jsonify({"fighters": []})
    
    try:
        fighters = search_fighters(query)
        return jsonify({"fighters": fighters})
    except Exception as e:
        logger.error(f"Error searching fighters: {e}")
        return jsonify({"error": str(e)}), 500
        
@api_bp.route('/search', methods=['GET'])
def search_endpoint():
    """Alternative search endpoint - for compatibility with frontend"""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify({"fighters": []})
    
    try:
        fighters = search_fighters(query)
        return jsonify({"fighters": fighters})
    except Exception as e:
        logger.error(f"Error searching fighters: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/fighters/<int:fighter_id>', methods=['GET'])
def get_fighter_endpoint(fighter_id):
    """Get fighter by ID endpoint"""
    try:
        fighter = get_fighter_details(fighter_id)
        if fighter:
            return jsonify({"fighter": fighter})
        else:
            return jsonify({"error": "Fighter not found"}), 404
    except Exception as e:
        logger.error(f"Error getting fighter {fighter_id}: {e}")
        return jsonify({"error": str(e)}), 500
@api_bp.route('/fighters/<string:fighter_name>', methods=['GET'])
def get_fighter_by_name_endpoint(fighter_name):
    try:
        # Enhanced logging
        print(f"=== FIGHTER SEARCH ===")
        print(f"Searching for fighter: {fighter_name}")
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Search strategies
        search_queries = [
            f'%{fighter_name}%',      # Contains the name
            fighter_name,             # Exact match
            fighter_name.lower(),     # Case-insensitive match
            fighter_name.strip(),     # Remove any leading/trailing spaces
        ]
        
        fighter = None
        for query in search_queries:
            print(f"Trying search query: '{query}'")
            cursor.execute('SELECT * FROM fighters WHERE LOWER(TRIM(name)) LIKE LOWER(TRIM(?))', (query,))
            fighter = cursor.fetchone()
            if fighter:
                break
        
        if fighter:
            # Convert row to dict and ensure all values are JSON-serializable
            fighter_dict = {}
            for idx, col in enumerate(cursor.description):
                column_name = col[0]
                value = fighter[idx]
                
                # Process binary data
                if isinstance(value, bytes):
                    try:
                        # Try to decode as string, or set to None
                        fighter_dict[column_name] = value.decode('utf-8', errors='ignore') or None
                    except:
                        fighter_dict[column_name] = None
                else:
                    fighter_dict[column_name] = value
            
            print(f"=== FIGHTER FOUND ===")
            conn.close()
            return jsonify({"fighter": fighter_dict})
        else:
            print(f"=== NO FIGHTER FOUND ===")
            conn.close()
            return jsonify({"error": "Fighter not found"}), 404
    
    except Exception as e:
        print(f"=== COMPLETE ERROR ===")
        print(f"Error for fighter {fighter_name}: {e}")
        return jsonify({"error": str(e)}), 500
    
@api_bp.route('/predict', methods=['POST'])
def predict_fight_endpoint():
    """Predict fight outcome endpoint"""
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Check if we have fighter IDs or data (support both camelCase and snake_case)
        if ('fighter1Id' in data and 'fighter2Id' in data) or ('fighter1_id' in data and 'fighter2_id' in data):
            # Get fighter IDs (support both formats)
            fighter1_id = data.get('fighter1Id') or data.get('fighter1_id')
            fighter2_id = data.get('fighter2Id') or data.get('fighter2_id')
            
            # Get fighter data using our helper function that includes fight history
            fighter1_data = get_fighter_stats(fighter1_id)
            fighter2_data = get_fighter_stats(fighter2_id)
            
            if not fighter1_data or not fighter2_data:
                return jsonify({"error": "One or both fighters not found"}), 404
            
            # Convert to prediction format
            fighter1 = format_fighter_for_prediction(fighter1_data)
            fighter2 = format_fighter_for_prediction(fighter2_data)
            
        elif 'fighter1' in data and 'fighter2' in data:
            # Direct fighter objects passed from frontend
            fighter1_data = data['fighter1']
            fighter2_data = data['fighter2']
            
            # Convert to prediction format
            fighter1 = format_fighter_for_prediction(fighter1_data)
            fighter2 = format_fighter_for_prediction(fighter2_data)
            
            # Log what we're using for prediction
            logger.info(f"Using fighter1 data: {fighter1}")
            logger.info(f"Using fighter2 data: {fighter2}")
        else:
            return jsonify({"error": "Must provide fighter1Id and fighter2Id, or fighter1 and fighter2 data"}), 400
        
        # Load model - IMPORTANT: Try ensemble model first since we retrained it
        from model import predict_fight, load_model
        try:
            # Try both models - prioritizing the ensemble model
            if os.path.exists(ENSEMBLE_MODEL_PATH):
                logger.info(f"Loading ensemble model from {ENSEMBLE_MODEL_PATH}")
                model, scaler, feature_columns, is_pytorch = load_model(ENSEMBLE_MODEL_PATH)
            elif os.path.exists(PYTORCH_MODEL_PATH):
                logger.info(f"Loading PyTorch model from {PYTORCH_MODEL_PATH}")
                model, scaler, feature_columns, is_pytorch = load_model(PYTORCH_MODEL_PATH)
            else:
                return jsonify({"error": "No trained models found"}), 500
                
            if model is None:
                return jsonify({"error": "Failed to load model"}), 500
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({"error": f"Error loading model: {str(e)}"}), 500
        
        # Make prediction
        prediction = predict_fight(model, fighter1, fighter2, scaler, feature_columns, is_pytorch)
        
        # Format prediction results (support both camelCase and snake_case)
        result = {
            "fighter1Name": fighter1.get('name', 'Fighter 1'),
            "fighter2Name": fighter2.get('name', 'Fighter 2'),
            "fighter1WinProbability": float(prediction['probability_fighter1_wins']),
            "fighter2WinProbability": float(prediction['probability_fighter2_wins']),
            "predictedWinner": fighter1.get('name', 'Fighter 1') if prediction['predicted_winner'] == 'fighter1' else fighter2.get('name', 'Fighter 2'),
            "confidenceLevel": prediction['confidence_level'],
            # Add snake_case versions for compatibility
            "fighter1_name": fighter1.get('name', 'Fighter 1'),
            "fighter2_name": fighter2.get('name', 'Fighter 2'),
            "fighter1_win_probability": float(prediction['probability_fighter1_wins']),
            "fighter2_win_probability": float(prediction['probability_fighter2_wins']),
            "predicted_winner": fighter1.get('name', 'Fighter 1') if prediction['predicted_winner'] == 'fighter1' else fighter2.get('name', 'Fighter 2'),
            "confidence_level": prediction['confidence_level']
        }
        
        return jsonify({"prediction": result})
    
    except Exception as e:
        logger.error(f"Error predicting fight: {e}")
        return jsonify({"error": str(e)}), 500


'''
@api_bp.route('/predict', methods=['POST'])
def predict_fight_endpoint():
    """Predict fight outcome endpoint"""
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Check if we have fighter IDs or data (support both camelCase and snake_case)
        if ('fighter1Id' in data and 'fighter2Id' in data) or ('fighter1_id' in data and 'fighter2_id' in data):
            # Get fighter IDs (support both formats)
            fighter1_id = data.get('fighter1Id') or data.get('fighter1_id')
            fighter2_id = data.get('fighter2Id') or data.get('fighter2_id')
            
            # Get fighter data using our helper function that includes fight history
            fighter1_data = get_fighter_stats(fighter1_id)
            fighter2_data = get_fighter_stats(fighter2_id)
            
            if not fighter1_data or not fighter2_data:
                return jsonify({"error": "One or both fighters not found"}), 404
            
            # Convert to prediction format
            fighter1 = format_fighter_for_prediction(fighter1_data)
            fighter2 = format_fighter_for_prediction(fighter2_data)
            
        elif 'fighter1' in data and 'fighter2' in data:
            # Direct fighter objects passed from frontend
            fighter1_data = data['fighter1']
            fighter2_data = data['fighter2']
            
            # Convert to prediction format
            fighter1 = format_fighter_for_prediction(fighter1_data)
            fighter2 = format_fighter_for_prediction(fighter2_data)
            
            # Log what we're using for prediction
            logger.info(f"Using fighter1 data: {fighter1}")
            logger.info(f"Using fighter2 data: {fighter2}")
        else:
            return jsonify({"error": "Must provide fighter1Id and fighter2Id, or fighter1 and fighter2 data"}), 400
        
        # Load model
        from model import predict_fight, load_model
        try:
            # Try both models - pytorch_model.pth or ensemble_model.joblib
            if os.path.exists(PYTORCH_MODEL_PATH):
                model, scaler, feature_columns, is_pytorch = load_model(PYTORCH_MODEL_PATH)
            elif os.path.exists(ENSEMBLE_MODEL_PATH):
                model, scaler, feature_columns, is_pytorch = load_model(ENSEMBLE_MODEL_PATH)
            else:
                return jsonify({"error": "No trained models found"}), 500
                
            if model is None:
                return jsonify({"error": "Failed to load model"}), 500
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({"error": f"Error loading model: {str(e)}"}), 500
        
        # Make prediction
        prediction = predict_fight(model, fighter1, fighter2, scaler, feature_columns)
        
        # Format prediction results (support both camelCase and snake_case)
        result = {
            "fighter1Name": fighter1.get('name', 'Fighter 1'),
            "fighter2Name": fighter2.get('name', 'Fighter 2'),
            "fighter1WinProbability": float(prediction['probability_fighter1_wins']),
            "fighter2WinProbability": float(prediction['probability_fighter2_wins']),
            "predictedWinner": fighter1.get('name', 'Fighter 1') if prediction['predicted_winner'] == 'fighter1' else fighter2.get('name', 'Fighter 2'),
            "confidenceLevel": prediction['confidence_level'],
            # Add snake_case versions for compatibility
            "fighter1_name": fighter1.get('name', 'Fighter 1'),
            "fighter2_name": fighter2.get('name', 'Fighter 2'),
            "fighter1_win_probability": float(prediction['probability_fighter1_wins']),
            "fighter2_win_probability": float(prediction['probability_fighter2_wins']),
            "predicted_winner": fighter1.get('name', 'Fighter 1') if prediction['predicted_winner'] == 'fighter1' else fighter2.get('name', 'Fighter 2'),
            "confidence_level": prediction['confidence_level']
        }
        
        return jsonify({"prediction": result})
    
    except Exception as e:
        logger.error(f"Error predicting fight: {e}")
        return jsonify({"error": str(e)}), 500
'''
@api_bp.route('/compare', methods=['GET'])
def compare_fighters_endpoint():
    """Compare two fighters endpoint"""
    fighter1_id = request.args.get('fighter1Id') or request.args.get('fighter1_id')
    fighter2_id = request.args.get('fighter2Id') or request.args.get('fighter2_id')
    
    if not fighter1_id or not fighter2_id:
        return jsonify({"error": "Must provide fighter1Id/fighter1_id and fighter2Id/fighter2_id"}), 400
    
    try:
        # Get fighter data with the new helper function that includes fight history
        fighter1 = get_fighter_stats(fighter1_id)
        fighter2 = get_fighter_stats(fighter2_id)
        
        if not fighter1 or not fighter2:
            return jsonify({"error": "One or both fighters not found"}), 404
        
        # Prepare comparison data
        comparison = {}
        stats_to_compare = [
            'height', 'weight', 'reach', 'wins', 'losses',
            'SLpM', 'sig_str_acc', 'SApM', 'str_def',
            'td_avg', 'td_acc', 'td_def', 'sub_avg'
        ]
        
        for stat in stats_to_compare:
            if stat in fighter1 and stat in fighter2:
                # Determine advantage
                if stat in ['losses', 'SApM']:
                    # Lower is better
                    if fighter1[stat] < fighter2[stat]:
                        advantage = 'fighter1'
                    elif fighter2[stat] < fighter1[stat]:
                        advantage = 'fighter2'
                    else:
                        advantage = 'even'
                else:
                    # Higher is better
                    if fighter1[stat] > fighter2[stat]:
                        advantage = 'fighter1'
                    elif fighter2[stat] > fighter1[stat]:
                        advantage = 'fighter2'
                    else:
                        advantage = 'even'
                
                comparison[stat] = {
                    'fighter1Value': fighter1[stat],
                    'fighter2Value': fighter2[stat],
                    'advantage': advantage,
                    'diff': fighter1[stat] - fighter2[stat]
                }
        
        # Make prediction too
        prediction = None
        try:
            # Convert to prediction format
            fighter1_pred = format_fighter_for_prediction(fighter1)
            fighter2_pred = format_fighter_for_prediction(fighter2)
            
            # Load model
            from model import predict_fight, load_model
            try:
                # Try both models
                if os.path.exists(PYTORCH_MODEL_PATH):
                    model, scaler, feature_columns, is_pytorch = load_model(PYTORCH_MODEL_PATH)
                elif os.path.exists(ENSEMBLE_MODEL_PATH):
                    model, scaler, feature_columns, is_pytorch = load_model(ENSEMBLE_MODEL_PATH)
                else:
                    logger.warning("No trained models found for comparison")
                    return None
                    
                if model is not None:
                    # Make prediction
                    pred_result = predict_fight(model, fighter1_pred, fighter2_pred, scaler, feature_columns)
                    
                    prediction = {
                        "fighter1WinProbability": float(pred_result['probability_fighter1_wins']),
                        "fighter2WinProbability": float(pred_result['probability_fighter2_wins']),
                        "predictedWinner": fighter1["name"] if pred_result['predicted_winner'] == 'fighter1' else fighter2["name"],
                        "confidenceLevel": pred_result['confidence_level'],
                        # Add snake_case versions for compatibility
                        "fighter1_win_probability": float(pred_result['probability_fighter1_wins']),
                        "fighter2_win_probability": float(pred_result['probability_fighter2_wins']),
                        "predicted_winner": fighter1["name"] if pred_result['predicted_winner'] == 'fighter1' else fighter2["name"],
                        "confidence_level": pred_result['confidence_level']
                    }
            except Exception as e:
                logger.error(f"Error loading model for comparison: {e}")
                # Continue without prediction
        except Exception as e:
            logger.error(f"Error making prediction for comparison: {e}")
            # Continue without prediction
        
        return jsonify({
            "fighter1": fighter1,
            "fighter2": fighter2,
            "comparison": comparison,
            "prediction": prediction
        })
    
    except Exception as e:
        logger.error(f"Error comparing fighters: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/admin/stats', methods=['GET'])
def admin_stats_endpoint():
    """Get admin dashboard statistics"""
    try:
        # Connect to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get fighter count
        cursor.execute('SELECT COUNT(*) FROM fighters')
        fighter_count = cursor.fetchone()[0]
        
        # Model info
        model_info = {}
        if os.path.exists(PYTORCH_MODEL_PATH):
            model_info['pytorch'] = {
                'path': PYTORCH_MODEL_PATH,
                'size': os.path.getsize(PYTORCH_MODEL_PATH) / 1024,  # KB
                'lastModified': os.path.getmtime(PYTORCH_MODEL_PATH)
            }
        
        if os.path.exists(ENSEMBLE_MODEL_PATH):
            model_info['ensemble'] = {
                'path': ENSEMBLE_MODEL_PATH,
                'size': os.path.getsize(ENSEMBLE_MODEL_PATH) / 1024,  # KB
                'lastModified': os.path.getmtime(ENSEMBLE_MODEL_PATH)
            }
        
        # Top fighters by win percentage
        cursor.execute('''
            SELECT id, name, wins, losses, CAST(wins AS FLOAT) / (wins + losses) as win_pct
            FROM fighters
            WHERE (wins + losses) >= 5
            ORDER BY win_pct DESC
            LIMIT 10
        ''')
        
        top_fighters = []
        for row in cursor.fetchall():
            top_fighters.append({
                'id': row[0],
                'name': row[1],
                'wins': row[2],
                'losses': row[3],
                'winPercentage': row[4]
            })
        
        conn.close()
        
        return jsonify({
            'fighterCount': fighter_count,
            'modelInfo': model_info,
            'topFighters': top_fighters
        })
    
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/admin/upload', methods=['POST'])
def admin_upload_endpoint():
    """Upload data files endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Get file type
        file_type = request.form.get('type', 'fighter_stats')
        
        # Save file to appropriate location
        if file_type == 'fighter_stats':
            save_path = os.path.join(DATA_DIR, 'fighter_stats.csv')
        elif file_type == 'ufc_dataset':
            save_path = os.path.join(DATA_DIR, 'ufc_dataset.csv')
        else:
            return jsonify({"error": "Invalid file type"}), 400
        
        file.save(save_path)
        
        # Optionally trigger data processing
        if file_type == 'fighter_stats':
            # Import dynamically to avoid circular import
            from database import import_fighter_stats_to_db
            success = import_fighter_stats_to_db()
            if not success:
                return jsonify({"error": "Failed to import fighter stats"}), 500
        
        return jsonify({"message": f"File uploaded successfully as {file_type}", "path": save_path})
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/admin/retrain', methods=['POST'])
def admin_retrain_endpoint():
    """Retrain model endpoint"""
    try:
        # Import dynamically to avoid circular import
        from main import train_models
        
        # Start training in a separate thread to not block the API
        import threading
        
        def train_thread_func():
            try:
                best_model_path, best_is_pytorch = train_models()
                logger.info(f"Training complete. Best model: {best_model_path}")
            except Exception as e:
                logger.error(f"Error during training: {e}")
        
        thread = threading.Thread(target=train_thread_func)
        thread.daemon = True  # Daemonize thread
        thread.start()
        
        return jsonify({"message": "Training started in background"})
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({"error": str(e)}), 500

def register_api(app):
    """Register the API blueprint with the Flask app"""
    app.register_blueprint(api_bp)
    return app