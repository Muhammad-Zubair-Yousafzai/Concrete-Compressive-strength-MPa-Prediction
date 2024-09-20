from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('model.pkl')  # Load your best model
scaler = joblib.load('scaler.pkl')  # Load the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        cement = float(request.form.get('cement', 0))
        water = float(request.form.get('water', 0))
        coarse_aggregate = float(request.form.get('coarse_aggregate', 0))
        fine_aggregate = float(request.form.get('fine_aggregate', 0))
        age_day = float(request.form.get('age_day', 0))
        blast_furnace_slag = float(request.form.get('blast_furnace_slag', 0))
        fly_ash = float(request.form.get('fly_ash', 0))
        superplasticizer = float(request.form.get('superplasticizer', 0))
        sand = float(request.form.get('sand', 0))  # Add sand here

        # Prepare the features for prediction
        features = [
            cement, blast_furnace_slag, fly_ash, water,
            superplasticizer, coarse_aggregate, sand, fine_aggregate, age_day
        ]
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        output = prediction[0]

        # Return the prediction and input values
        return jsonify({
            'prediction': f'{output:.2f} MPa',
            'input_values': {
                'Cement': cement,
                'Water': water,
                'Coarse Aggregate': coarse_aggregate,
                'Fine Aggregate': fine_aggregate,
                'Age (days)': age_day,
                'Blast Furnace Slag': blast_furnace_slag,
                'Fly Ash': fly_ash,
                'Superplasticizer': superplasticizer,
                'Sand': sand
            }
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {e}'})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == "__main__":
    app.run(debug=False)
