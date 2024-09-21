import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Load your profile and people profile data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Example profile data (your profile and people profiles)
my_profile = {
    "description": "Software engineer skilled in Python, machine learning, and APIs.",
    "location": "New York",
    "awards": [{"title": "Best Engineer", "year": 2022}],
    "skills": ["Python", "Machine Learning", "APIs"]
}

people_profiles = [
    {
        "name": "Thomas",
        "description": "Experienced in Python, deep learning, and API development.",
        "location": "San Francisco",
        "awards": [{"title": "AI Developer", "year": 2021}],
        "skills": ["Python", "Deep Learning", "APIs"]
    },
    {
        "name": "Anna",
        "description": "Data scientist specializing in Python, SQL, and visualization.",
        "location": "Boston",
        "awards": [{"title": "Data Expert", "year": 2023}],
        "skills": ["Python", "SQL", "Visualization"]
    }
    # Add more profiles...
]

# Convert data to DataFrame
def prepare_data(my_profile, people_profiles):
    profiles = people_profiles + [{"name": "You", **my_profile}]
    df = pd.DataFrame(profiles)
    return df

df = prepare_data(my_profile, people_profiles)

# Text-based features (description and skills combined)
df['combined_text'] = df['description'] + ' ' + df['skills'].apply(' '.join)

# Pipeline for processing text and clustering
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),          # Convert text to numerical vectors
    ('kmeans', KMeans())                   # Cluster similar profiles
])

# Hyperparameter tuning (e.g., number of clusters)
param_grid = {
    'kmeans__n_clusters': [2, 3, 4, 5],    # Try different numbers of clusters
    'tfidf__max_features': [50, 100, 200], # Adjust number of TF-IDF features
}

# Grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3)

# Fit the model
grid_search.fit(df['combined_text'])

# Best pipeline
best_model = grid_search.best_estimator_

# Get clustering labels
df['cluster'] = best_model.named_steps['kmeans'].labels_

# Evaluate using silhouette score
silhouette_avg = silhouette_score(best_model.named_steps['tfidf'].transform(df['combined_text']),
                                  df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Cosine similarity between your profile and others
cos_sim = cosine_similarity(
    best_model.named_steps['tfidf'].transform([df.loc[df['name'] == 'You', 'combined_text'].values[0]]),
    best_model.named_steps['tfidf'].transform(df['combined_text'])
)

# Add cosine similarity to DataFrame (excluding yourself)
df['similarity'] = cos_sim[0]
df_sorted = df[df['name'] != 'You'].sort_values(by='similarity', ascending=False)

# Print sorted names and similarity scores
print(df_sorted[['name', 'similarity']])

# Output the top matches
print("Top matches (best to least):")
print(df_sorted[['name', 'similarity']].to_string(index=False))
