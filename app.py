# Import important libraries
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create the Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return """
        <p>Welcome to the ML-Medical API.</h1>
        <p>However this is not what you're looking for.</h1>
        <p>Please use the following endpoints to access the ML-Medical API.</p>
        <ul>
            <li>/cardiac</li>
            <li>/diabetes</li>
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

# Diabetes prediction
@app.route('/diabetes', methods=['POST'])
def diabetes():
    request_data = request.get_json()
    if request_data:
        if 'age' in request_data:
            age = request_data['age']
            # Patient age in years
        if 'gender' in request_data:
            gender = request_data['gender']
            # Patient gender ( 0 = female; 1 = male)
        if 'polyuria' in request_data:
            polyuria = request_data['polyuria']
            # Excessive urination ( 0 = false, 1 = true)
        if 'polydipsia' in request_data:
            polydipsia = request_data['polydipsia']
            # Extreme thirst ( 0 = false, 1 = true)
        if 'weight' in request_data:
            weight = request_data['weight']
            # Sudden weight loss ( 0 = false, 1 = true)
        if 'weakness' in request_data:
            weakness = request_data['weakness']
            # Feeling weak with no reason ( 0 = false, 1 = true)
        if 'polyphagia' in request_data:
            polyphagia = request_data['polyphagia']
            # Eating excessive amounts of food ( 0 = false, 1 = true)
        if 'genital_thrush' in request_data:
            genital_thrush = request_data['genital_thrush']
            # Yeast infection in the genital area ( 0 = false, 1 = true)
        if 'visual_blurring' in request_data:
            visual_blurring = request_data['visual_blurring']
            # High blood sugar causes the lens of the eye to swell, which changes your ability to see ( 0 = false, 1 = true)
        if 'itching' in request_data:
            itching = request_data['itching']
            # Itching with no reason ( 0 = false, 1 = true)
        if 'irritability' in request_data:
            irritability = request_data['irritability']
            # Irritability with no reason ( 0 = false, 1 = true)
        if 'delayed_healing' in request_data:
            delayed_healing = request_data['delayed_healing']
            # Delayed healing of the skin and/or mucous membranes ( 0 = false, 1 = true)
        if 'partial_paresis' in request_data:
            partial_paresis = request_data['partial_paresis']
            # Weakening of a muscle or group of muscles ( 0 = false, 1 = true)
        if 'muscle_stiffness' in request_data:
            muscle_stiffness = request_data['muscle_stiffness']
            # Feeling stiff in the limbs ( 0 = false, 1 = true)
        if 'alopecia' in request_data:
            alopecia = request_data['alopecia']
            # Hair loss ( 0 = false, 1 = true)
        if 'obesity' in request_data:
            obesity = request_data['obesity']
            # Obesity ( 0 = false, 1 = true)
    
    # Importing the dataset
    actual_patient_data = pd.read_csv('/home/khaichuen/ml-medical/diabetes.csv')
    converted_data=pd.get_dummies(actual_patient_data, prefix=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
           'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
           'Itching', 'Irritability', 'delayed healing', 'partial paresis',
           'muscle stiffness', 'Alopecia', 'Obesity', 'class'], drop_first=True)

    # Training the model
    X_train, X_test, y_train, y_test = train_test_split(converted_data.drop('class_Positive', axis=1),converted_data['class_Positive'], test_size=0.3, random_state=0)
    
    sc=StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    RF_classifier.fit(X_train, y_train)

    process_prediction = RF_classifier.predict(sc.transform(np.array([[int(age),int(gender),int(polyuria),int(polydipsia),int(weight),int(weakness),int(polyphagia),int(genital_thrush),int(visual_blurring),int(itching),int(irritability), int(delayed_healing),int(partial_paresis),int(muscle_stiffness),int(alopecia),int(obesity)]])))

    if process_prediction == 1:
        prediction = 'High risk'
    else:
        prediction = 'Low risk'

    data = {
        "age" : age,
        "gender" : gender,
        "polyuria" : polyuria,
        "polydipsia" : polydipsia,
        "weight": weight,
        "weakness": weakness,
        "polyphagia": polyphagia,
        "genital_thrush": genital_thrush,
        "visual_blurring": visual_blurring,
        "itching": itching,
        "irritability": irritability,
        "delayed_healing": delayed_healing,
        "partial_paresis": partial_paresis,
        "muscle_stiffness": muscle_stiffness,
        "alopecia": alopecia,
        "obesity": obesity,
        "prediction": prediction
    }

    return jsonify(data)

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
