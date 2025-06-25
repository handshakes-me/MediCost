from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import traceback

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Function to train a new model
def train_new_model():
    try:
        print("Training a new model...")
        # Load the dataset
        print("Loading dataset...")
        data = pd.read_csv('insurance.csv')
        print(f"Dataset loaded with {len(data)} rows")
        
        # Check if required columns exist
        required_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'expenses']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
            
        print("Preprocessing data...")
        # Preprocess the data
        le = LabelEncoder()
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            print(f"Encoding column: {col}")
            data[col] = le.fit_transform(data[col])
        
        # Split the data
        print("Splitting data...")
        X = data.drop('expenses', axis=1)
        y = data['expenses']
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Train the model
        print("Training model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Save the new model
        model_path = 'new_model.pkl'
        print(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        
        # Verify the model was saved
        if os.path.exists(model_path):
            print("New model trained and saved successfully!")
            return model
        else:
            print("Error: Failed to save the model file")
            return None
            
    except Exception as e:
        print("Error in train_new_model:")
        print(traceback.format_exc())
        return None

# Initialize model variable
model = None

print("\n=== Starting Model Loading Process ===")
print(f"Current working directory: {os.getcwd()}")
print("Contents of current directory:", os.listdir('.'))

def generate_indian_healthcare_data(num_samples=1000):
    """Generate synthetic healthcare data for Indian demographics"""
    np.random.seed(42)
    
    # Generate realistic Indian demographic data
    age = np.random.normal(35, 15, num_samples).astype(int)  # Mean age around 35
    age = np.clip(age, 1, 100)  # Ensure age is between 1-100
    
    # Gender distribution (slightly more males in Indian population)
    gender = np.random.choice([0, 1], size=num_samples, p=[0.48, 0.52])  # 48% female, 52% male
    
    # BMI distribution (Indians have lower BMIs on average but higher body fat at lower BMIs)
    bmi = np.random.normal(24, 4, num_samples)  # Mean BMI ~24 (healthy range for Indians is 18-23)
    bmi = np.clip(bmi, 15, 50)  # Reasonable BMI range
    
    # Children (0-5 is common in Indian families)
    children = np.random.poisson(1.5, num_samples)  # Mean ~1.5 children
    children = np.clip(children, 0, 5)
    
    # Smoker (lower smoking rates in India compared to West)
    smoker = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])  # ~10% smokers
    
    # Regions in India (0=North, 1=South, 2=East, 3=West, 4=Central, 5=North-East)
    region = np.random.randint(0, 6, num_samples)
    
    # Base cost calculation based on Indian healthcare costs (in INR)
    # Average cost is around 10,000-50,000 INR for minor procedures
    base_cost = 10000 + (age ** 1.5) * 10  # Age increases cost
    base_cost += bmi * 200  # Higher BMI increases cost
    base_cost += children * 5000  # Each child adds some cost
    base_cost *= (1 + smoker * 0.8)  # Smokers pay 80% more
    
    # Add some randomness and convert to integer (INR)
    charges = np.random.normal(base_cost, 5000).astype(int)
    charges = np.clip(charges, 2000, 200000)  # Reasonable range for Indian healthcare
    
    # Create feature matrix and target
    X = np.column_stack((age, gender, bmi, children, smoker, region))
    y = charges
    
    return X, y

def create_dummy_model():
    """Create a model trained on Indian healthcare data"""
    from sklearn.ensemble import RandomForestRegressor
    print("Training model on Indian healthcare data...")
    
    # Generate Indian healthcare data
    X, y = generate_indian_healthcare_data(5000)  # Generate 5000 samples
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Save the model for future use
    joblib.dump(model, 'indian_healthcare_model.pkl')
    print("Indian healthcare model trained and saved successfully")
    return model

try:
    # Try to load existing models first
    model_paths = ['rf_tuned.pkl', 'new_model.pkl']
    for path in model_paths:
        try:
            if os.path.exists(path):
                print(f"\nAttempting to load model from: {os.path.abspath(path)}")
                model = joblib.load(path)
                print(f"Successfully loaded model from {path}")
                print(f"Model type: {type(model)}")
                break
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    # If no model loaded, try to train a new one
    if model is None:
        print("\nNo valid model found. Attempting to train a new one...")
        model = train_new_model()
    
    # If still no model, create a dummy one
    if model is None:
        print("\nFailed to train a model. Creating a dummy model...")
        model = create_dummy_model()
        
except Exception as e:
    print(f"\nUnexpected error during model initialization: {str(e)}")
    print(traceback.format_exc())
    print("Creating a dummy model as fallback...")
    model = create_dummy_model()

print("\n=== Model initialization complete ===")
@app.route('/')

def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return render_template('op.html', pred='Error: Model is not available. Please try again later.')
    
    if request.method == 'GET':
        return redirect(url_for('hello_world'))
    
    try:
        # Debug: Print form data
        print("\n=== Form Data ===")
        print(dict(request.form))
        
        # Get form data with defaults
        form_data = {
            'age': request.form.get('age', ''),
            'sex': request.form.get('sex', '').lower(),
            'bmi': request.form.get('bmi', ''),
            'children': request.form.get('children', ''),
            'smoker': request.form.get('smoker', '').lower(),
            'region': request.form.get('region', '0')
        }
        
        # Validate required fields
        for field in ['age', 'bmi', 'children']:
            if not form_data[field]:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert and validate data types
        try:
            age = int(form_data['age'])
            bmi = float(form_data['bmi'])
            children = int(form_data['children'])
            sex = 1 if form_data['sex'] == 'male' else 0
            smoker = 1 if form_data['smoker'] == 'yes' else 0
            region = int(form_data['region'])
        except ValueError as e:
            raise ValueError(f"Invalid input format: {str(e)}")
        
        # Validate input ranges
        if not (0 < age < 120):
            raise ValueError("Age must be between 1 and 120")
        if bmi <= 0:
            raise ValueError("BMI must be a positive number")
        if children < 0:
            raise ValueError("Number of children cannot be negative")
        
        # Create feature array in the correct order
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format the prediction in Indian Rupees with proper formatting
        def format_inr(amount):
            try:
                amount = float(amount)
                s, *d = f"{amount:,.2f}".replace(".00", "").partition(".")
                parts = []
                # Add commas as thousand separators
                s = "{:,}".format(int(amount))
                return s
            except (ValueError, TypeError):
                return str(amount)
            
        formatted_cost = format_inr(prediction)
        
        return render_template('op.html', 
                            pred=formatted_cost,  # Just the number, no currency symbol
                            status='success',
                            cost=float(prediction))
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"\n=== Prediction Error ===\n{error_msg}")
        print(f"Form data: {dict(request.form) if request.method == 'POST' else 'No form data'}")
        traceback.print_exc()
        return render_template('op.html', pred=error_msg, status='error')

# Add a simple home route for testing
@app.route('/test')
def test():
    return "Flask server is running!"

# Add a route to retrain the model if needed
@app.route('/retrain', methods=['GET'])
def retrain_model():
    global model
    model = train_new_model()
    if model is not None:
        return "Model retrained successfully!"
    return "Failed to retrain model."

if __name__ == '__main__':
    app.run(debug=True)