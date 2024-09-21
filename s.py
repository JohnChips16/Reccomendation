from flask import Flask, request, jsonify
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import joblib
import os

app = Flask(__name__)

# Global variable to store the trained model
model = None

# Load people's profile data from a JSON file
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Example path to people's profile JSON file
PEOPLE_PROFILES_FILE = 'data.json'

# Example training data: 
# This can be pre-loaded or generated on the fly
people_profiles = load_json_data(PEOPLE_PROFILES_FILE)

# Convert the data to a DataFrame and prepare for processing
def prepare_data(people_profiles):
    df = pd.DataFrame(people_profiles)
    df['combined_text'] = df['description'] + ' ' + df['skills'].apply(' '.join)
    return df

df_people = prepare_data(people_profiles)

# Function to train the model
def train_model(df):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),  # Text processing
        ('svd', TruncatedSVD(n_components=100)),  # Dimensionality reduction
        ('model', NearestNeighbors(n_neighbors=5, metric='cosine'))  # Nearest Neighbor search
    ])
    
    # Fit the pipeline on the people's profiles combined text
    pipeline.fit(df['combined_text'])
    
    return pipeline

# Route to train the model and update the global model variable
@app.route('/train', methods=['POST'])
def train():
    global model
    # Train the model on the preloaded data
    df = df_people  # Use the data from people's profiles
    model = train_model(df)

    return jsonify({'status': 'Model trained successfully.'})

# Route to save the trained model
@app.route('/save', methods=['POST'])
def save_model():
    global model
    if model is None:
        return jsonify({'error': 'No model found. Train the model first.'}), 400

    model_path = 'trained_model.pkl'
    joblib.dump(model, model_path)

    return jsonify({'status': f'Model saved at {model_path}.'})

# Route to load a saved model
@app.route('/load', methods=['POST'])
def load_model():
    global model
    model_path = 'trained_model.pkl'

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found. Please save a model first.'}), 400

    model = joblib.load(model_path)

    return jsonify({'status': 'Model loaded successfully.'})

# Route to compare a profile against the existing people profiles
@app.route('/compare', methods=['POST'])
def compare_profiles():
    global model
    if model is None:
        return jsonify({'error': 'No model found. Train or load a model first.'}), 400

    # Input profile data (from the POST request)
    profile_data = request.json

    # Prepare the input profile for comparison
    input_profile = {
        'description': profile_data.get('description', ''),
        'skills': ' '.join(profile_data.get('skills', []))
    }
    
    combined_input_text = input_profile['description'] + ' ' + input_profile['skills']

    # Transform the input data and compare it with people's profiles
    tfidf_matrix = model.named_steps['tfidf'].transform([combined_input_text])
    svd_matrix = model.named_steps['svd'].transform(tfidf_matrix)

    people_tfidf_matrix = model.named_steps['tfidf'].transform(df_people['combined_text'])
    people_svd_matrix = model.named_steps['svd'].transform(people_tfidf_matrix)

    # Find the nearest neighbors (most similar profiles)
    distances, indices = model.named_steps['model'].kneighbors(svd_matrix)

    # Get the top matches and return their profile data
    top_matches = df_people.iloc[indices[0]].copy()
    top_matches['similarity'] = 1 - distances[0]  # Inverse of distance to get similarity

    return jsonify(top_matches[['name', 'similarity']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
