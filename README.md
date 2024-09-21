# Profile Matching API with Flask and Machine Learning
# ©2024 URBAN NETWORKS
## Intellectual Property
This project is released under the MIT License. All contributions and code within this repository are freely available for use and modification under the terms of this license.

### Important:
- Please ensure that any third-party code integrated into this project respects the terms of its respective licenses.
- Contributors retain copyright to their contributions but agree to license them under the same open-source terms upon submission of a pull request.
- If you use or modify this code, we request (but do not require) attribution to the original author and this repository.

## Description
This project is a Flask-based REST API that allows users to compare their profiles (including descriptions and skills) against a dataset of people profiles. It uses machine learning techniques such as TF-IDF for text processing, SVD for dimensionality reduction, and Nearest Neighbors for profile matching. The API returns the most similar profiles based on cosine similarity.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)


## Installation
To run this project locally, follow these steps:

   1. git clone https://github.com/JohnChips16/Reccomendation.git
   2. cd Reccomendation
   3. python3 s.py


## 5. **Usage**
0. it's flexible, so you can adjust depends on what fields you are comparing it.
1. train the model with given json data
   /train
2. save the model
   /save
3. load the trained model
   /load
4. and make comparison
   /compare

   with following curl command
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
        "description": "Experienced in Python, deep learning, cybersecurity analyst, and API development.",                                                                             "location": "San Francisco",
        "awards": [{"title": "AI Developer", "year": 2021}, {"title": "Cyber Defender", "year": 2021}],                                                                                 "skills": ["Python", "Deep Learning", "APIs", "Jenkins"]
    }' http://localhost:5000/compare

## **response**
```markdown
[
  {
    "name": "Thomas",
    "similarity": 0.9933721509868334
  },
  {
    "name": "Anna",
    "similarity": 0.1529370996048529
  },
  {
    "name": "David",
    "similarity": 0.11578778255017896
  },
  {
    "name": "Sarah",
    "similarity": 0.08631453763573982
  },
  {
    "name": "James",
    "similarity": 0.0651742734494497
  }
] 




