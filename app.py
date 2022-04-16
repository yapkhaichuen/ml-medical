# Import important libraries
import json
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask_cors import CORS

# Create the Flask app
app = Flask(__name__)
CORS(app)


# Home page
@app.route('/')
def home():
    return """
        <p>Welcome to the ML-Medical API.</h1>
        <p>However this is not what you're looking for.</h1>
        <p>Please use the following endpoints to access the ML-Medical API.</p>
        <ul>
            <li>/cardiac</li>
        </ul>
    """
    



# Heart attack prediction
@app.route('/cardiac', methods=['POST'])
def cardiac():
    request_data = request.get_json()
    if request_data:
        if 'age' in request_data:
            age = request_data['age']
            # Patient age in years
        if 'sex' in request_data:
            sex = request_data['sex']
            # Patient gender ( 0 = female; 1 = male)
        if 'cp' in request_data:
            cp = request_data['cp']
            # Chest pain type ( 0 = typical angina, 1 = atypical angina, 2 = non-aginal pain, 3 = asymptomatic)
        if 'trestbps' in request_data:
            trestbps = request_data['trestbps']
            # Resting blood pressure (in mm Hg on admission to the hospital or self testing)
        if 'chol' in request_data:
            chol = request_data['chol']
            # Serum cholestoral in mg/dl
        if 'fbs' in request_data:
            fbs = request_data['fbs']
            # Fasting blood sugar > 120 mg/dl (0 = false; 1 = true)
        if 'restecg' in request_data:
            restecg = request_data['restecg']
            # Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
        if 'thalach' in request_data:
            thalach = request_data['thalach']
            # Maximum heart rate achieved
        if 'exang' in request_data:
            exang = request_data['exang']
            # Chest pain (angina) after exercise (0 = false; 1 = true)
        if 'oldpeak' in request_data:
            oldpeak = request_data['oldpeak']
            # ST depression induced by exercise relative to rest (in mm Hg)
        if 'slope' in request_data:
            slope = request_data['slope']
            # Slope of peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
        if 'ca' in request_data:
            ca = request_data['ca']
            # Number of major vessels (0-3) colored by flourosopy
        if 'thal' in request_data:
            thal = request_data['thal']
            # Thalassemia (0 = normal; 1 = fixed defect; 2 = reversable defect)

    # Prepare and parse the data
    df_heart = pd.read_csv('/home/khaichuen/ml-medical/heart.csv')
    dups_data = df_heart.duplicated()
    data_heart = df_heart.drop_duplicates()
    X = data_heart.drop('target',axis=1)
    Y = data_heart['target']

    # Define data
    def user_input_features():
        age_fresh = age
        sex_fresh = sex
        cp_fresh = cp
        trestbps_fresh = trestbps
        chol_fresh = chol
        fbs_fresh = fbs
        restecg_fresh = restecg
        thalach_fresh = thalach
        exang_fresh = exang
        oldpeak_fresh = oldpeak
        slope_fresh = slope
        ca_fresh = ca
        thal_fresh = thal
        data = {'age': age_fresh,
                'sex': sex_fresh,
                'cp': cp_fresh,
                'trestbps': trestbps_fresh,
                'chol': chol_fresh,
                'fbs': fbs_fresh,
                'restecg': restecg_fresh,
                'thalach': thalach_fresh,
                'exang': exang_fresh,
                'oldpeak': oldpeak_fresh,
                'slope': slope_fresh,
                'ca': ca_fresh,
                'thal': thal_fresh
                }

        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()
    print(df)

    # Create the model
    knn = KNeighborsClassifier(n_neighbors=35)
    knn.fit(X,Y)
    prediction = knn.predict(df)

    # Return the prediction
    heart_attack_clf = np.array(['Low Risk', 'High Risk'])
    print(heart_attack_clf[prediction])
    prediction_proba = knn.predict_proba(df)

    risk = str(heart_attack_clf[prediction])
    risk = risk[2:] # Remove the first two characters
    risk = risk[:-2] # Remove the last two characters

    probability = str(prediction_proba)
    probability = probability[2:] # Remove the first two characters
    probability = probability[:-2] # Remove the last two characters

    # Parse all data for API to return
    data_fresh = {
        "age": age,
        "ca": ca,
        "chol": chol,
        "cp": cp,
        "exang": exang,
        "fbs": fbs,
        "oldpeak": oldpeak,
        "restecg": restecg,
        "sex": sex,
        "slope": slope,
        "thal": thal,
        "thalach": thalach,
        "trestbps": trestbps,
        "risk": risk,
        "prediction_probability" : probability
    }

    return jsonify(data_fresh)


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
