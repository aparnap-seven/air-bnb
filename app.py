from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open('airbnb_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load features
with open('features.json', 'r') as f:
    feature_data = json.load(f)

numerical_features = feature_data['numerical_features']
categorical_features = feature_data['categorical_features']
all_features = feature_data['all_features']

print("✓ Model and encoders loaded successfully")

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Check if API is running"""
    return jsonify({
        'status': 'OK',
        'message': 'Airbnb Price Prediction API is running'
    }), 200

# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Airbnb price based on input features
    
    Expected JSON input:
    {
        "Number of Reviews": 50,
        "Reviews per Month": 2.5,
        "Minimum Nights": 30,
        "Availability 365": 200,
        "Beds": 1,
        "Years_as_Host": 3.5,
        "Neighbourhood": "Manhattan",
        "Room Type": "Entire home/apt",
        "Neighbourhood_RoomType": "Manhattan_Entire home/apt"
    }
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that data is provided
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please send JSON data with prediction features'
            }), 400
        
        # ====================================================================
        # VALIDATE AND PREPARE INPUT DATA
        # ====================================================================
        
        # Check required fields
        required_numerical = numerical_features
        required_categorical = categorical_features
        
        # Validate numerical features
        input_numerical = {}
        for feature in required_numerical:
            if feature not in data:
                return jsonify({
                    'error': f'Missing feature: {feature}',
                    'message': f'Required numerical feature "{feature}" not provided',
                    'required_features': required_numerical + required_categorical
                }), 400
            
            try:
                input_numerical[feature] = float(data[feature])
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Invalid data type for {feature}',
                    'message': f'Feature "{feature}" must be a number',
                    'received': data[feature]
                }), 400
        
        # Validate categorical features
        input_categorical = {}
        for feature in required_categorical:
            if feature not in data:
                return jsonify({
                    'error': f'Missing feature: {feature}',
                    'message': f'Required categorical feature "{feature}" not provided',
                    'required_features': required_numerical + required_categorical
                }), 400
            
            input_categorical[feature] = str(data[feature])
        
        # ====================================================================
        # ENCODE CATEGORICAL FEATURES
        # ====================================================================
        
        input_encoded = {}
        
        # Encode individual categorical features
        for feature in [f for f in required_categorical if f != 'Neighbourhood_RoomType']:
            try:
                encoded_value = label_encoders[feature].transform([input_categorical[feature]])[0]
                input_encoded[feature + '_encoded'] = encoded_value
            except ValueError:
                # Handle unknown categories
                return jsonify({
                    'error': f'Unknown value for {feature}',
                    'message': f'"{input_categorical[feature]}" is not a valid {feature}',
                    'valid_values': list(label_encoders[feature].classes_)
                }), 400
        
        # Encode interaction feature
        try:
            interaction_value = input_categorical['Neighbourhood'] + '_' + input_categorical['Room Type']
            encoded_interaction = label_encoders['Neighbourhood_RoomType'].transform([interaction_value])[0]
            input_encoded['Neighbourhood_RoomType_encoded'] = encoded_interaction
        except ValueError:
            return jsonify({
                'error': 'Invalid Neighbourhood_RoomType combination',
                'message': f'The combination "{input_categorical["Neighbourhood"]}" + "{input_categorical["Room Type"]}" is not recognized'
            }), 400
        
        # ====================================================================
        # PREPARE FEATURE VECTOR IN CORRECT ORDER
        # ====================================================================
        
        # Create feature vector in exact order as training
        feature_vector = []
        
        # Add numerical features first
        for feature in numerical_features:
            feature_vector.append(input_numerical[feature])
        
        # Add encoded categorical features in order
        for feature in categorical_features:
            feature_vector.append(input_encoded[feature + '_encoded'])
        
        # Convert to numpy array and reshape for model
        X_pred = np.array(feature_vector).reshape(1, -1)
        
        # ====================================================================
        # MAKE PREDICTION
        # ====================================================================
        
        predicted_price = model.predict(X_pred)[0]
        
        # Ensure price is positive
        predicted_price = max(0, predicted_price)
        
        # ====================================================================
        # PREPARE RESPONSE
        # ====================================================================
        
        response = {
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'currency': 'USD',
            'confidence_interval': {
                'low': round(max(0, predicted_price - 149.04), 2),
                'high': round(predicted_price + 149.04, 2),
                'rmse': 149.04,
                'interpretation': 'Model prediction is typically within ±$149 from actual price'
            },
            'model_metrics': {
                'r2_score': 0.3449,
                'rmse': 149.04,
                'mae': 55.87,
                'accuracy': '34.49%',
                'note': 'Model explains 34% of price variance. Use prediction as guidance only.'
            },
            'input_data': {
                'numerical': input_numerical,
                'categorical': input_categorical
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Handle unexpected errors
        return jsonify({
            'error': 'Prediction error',
            'message': str(e),
            'status': 'failed'
        }), 500

# ============================================================================
# INFO ENDPOINT - Shows available options
# ============================================================================
@app.route('/info', methods=['GET'])
def info():
    """Get information about available features and valid values"""
    return jsonify({
        'model_info': {
            'type': 'Random Forest Regressor',
            'trees': 50,
            'max_depth': 8,
            'r2_score': 0.3449,
            'rmse': 149.04,
            'training_period': '2008-2015'
        },
        'numerical_features': numerical_features,
        'categorical_features': {
            'Neighbourhood': list(label_encoders['Neighbourhood'].classes_),
            'Room Type': list(label_encoders['Room Type'].classes_)
        },
        'example_request': {
            'Number of Reviews': 50,
            'Reviews per Month': 2.5,
            'Minimum Nights': 30,
            'Availability 365': 200,
            'Beds': 1,
            'Years_as_Host': 3.5,
            'Neighbourhood': 'Manhattan',
            'Room Type': 'Entire home/apt',
            'Neighbourhood_RoomType': 'Manhattan_Entire home/apt'
        },
        'endpoint': '/predict',
        'method': 'POST'
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/health', '/predict', '/info']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors (wrong HTTP method)"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'Use POST method for /predict endpoint'
    }), 405

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    # Run on localhost:5000
    # Change debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)