# Import important libraries
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

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
            <li>/hypertension</li>
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

    print("Cardiac activated")

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

    # Create the model
    knn = KNeighborsClassifier(n_neighbors=35)
    knn.fit(X,Y)
    prediction = knn.predict(df)

    # Return the prediction
    prediction_proba = knn.predict_proba(df)
    probability = str(prediction_proba)
    probability_new = probability.replace(']','')

    # Hacky way to get the probability of the prediction (might change later)
    if len(probability_new) == 9:
        probability_final = float(probability_new[6:]) # Remove the first 11 and last 1 characters
    else:
        probability_final = float(probability_new[13:][:-1]) # Remove the first 11 and last 1 characters

    # Return the prediction based on the probability
    if probability_final > 0.5:
        risk = 'High Risk'
    else:
        risk = 'Low Risk'

    # Server log
    print(df)
    print("Risk = " + risk)
    print(prediction_proba)
    print(probability_final)
    
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
        "prediction_probability" : probability_final
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
        if 'sex' in request_data:
            sex = request_data['sex']
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

    print("Diabetes activated")

    # Importing the dataset
    actual_patient_data = pd.read_csv('/home/khaichuen/ml-medical/diabetes.csv')
    converted_data=pd.get_dummies(actual_patient_data, prefix=['Sex', 'Polyuria', 'Polydipsia', 'sudden weight loss',
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

    process_prediction = RF_classifier.predict(sc.transform(np.array([[int(age),int(sex),int(polyuria),int(polydipsia),int(weight),int(weakness),int(polyphagia),int(genital_thrush),int(visual_blurring),int(itching),int(irritability), int(delayed_healing),int(partial_paresis),int(muscle_stiffness),int(alopecia),int(obesity)]])))

    if process_prediction == 1:
        prediction = 'High risk'
    else:
        prediction = 'Low risk'

    data = {
        "age" : age,
        "sex" : sex,
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

@app.route('/hypertension', methods=['POST'])
def hypertension():
    request_data = request.get_json()
    if request_data:
        if 'age' in request_data:
            age = request_data['age']
            # Patient age in years
        if 'bmi' in request_data:
            bmi = request_data['bmi']
            # BMI index 10-50 range no more no less
        if 'drinking' in request_data:
            drinking = request_data['drinking']
            # Drinking status: 0 = no, 1 = yes
        if 'exercise' in request_data:
            exercise = request_data['exercise']
            # Exercise time in hours/week 1-3
        if 'sex' in request_data:
            sex = request_data['sex']
            # gender 1 = male, 0 = female
        if 'junk' in request_data:
            junk = request_data['junk']
            # junk food consumption 1-3 times a week
        if 'sleep' in request_data:
            sleep = request_data['sleep']
            # sleep rating score 1-3
        if 'smoking' in request_data:
            smoking = request_data['smoking']
            # smoking status: 0 = no, 1 = yes
    
    print("Hypertension activated")

    def user_input_features():
        Age = age
        Bmi = bmi
        #Drinking = st.sidebar.slider('DRINKING ', 0, 1, 0)
        Drinking = drinking
        #Exercise = st.sidebar.slider('EXERCISE PER WEEK', 1, 3, 1)
        Exercise = exercise
        Gender = sex
        Junk = junk
        Sleep = sleep
        Smoking = smoking
        data = {'Age': Age,
                'Bmi': Bmi,
                'Drinking': Drinking,
                'Exercise': Exercise,
                'Gender': Gender,
                'Junk': Junk,
                'Sleep': Sleep,
                'Smoking': Smoking,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    
    data = {'Age': age,
            'Bmi': bmi,
            'Drinking': drinking,
            'Exercise': exercise,
            'Sex': sex,
            'Junk': junk,
            'Sleep': sleep,
            'Smoking': smoking
            }

    print(data)
    
    df_input = user_input_features()


    df = pd.read_csv('/home/khaichuen/ml-medical/hypertension.csv')
    X = df.iloc[:, 0:8].values
    y = df.iloc[:, 8:11].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler as ss
    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    classifier= RandomForestRegressor(n_estimators = 300, random_state = 0)
    classifier.fit(X_train,y_train)

    prediction = classifier.predict(df_input)
    #prediction_proba = classifier.score(X_test, y_test)

    ans = prediction.flatten()
    a = ans[0] #depression
    b = ans[1] #diabetes is not considered due to another page alredy 
    c = ans[2] #hypertension
    if(a<50 and b<50 and c<50):
        result = "Fit and healthy"
    elif(a>50 and a<70 and a>b and a>c):
        result = 'Low risk of depression'
    elif(b>50 and b<70 and b>c and b>a):
        result = 'High risk of hyperglycemia'
    elif(c>50 and c<70 and c>a and c>b):
        result = "Low risk of hypertension"       
    elif(a>50 and a>b and a>c):
        result = "High risk of depression"
    elif (b>50 and b>a and b>c):
         result = "High risk of hyperglycemia "
    elif (c>50 and c>a and c>b):     
         result = "High risk of hypertension"

    prediction_proba = classifier.score(X_test,y_test)
    
    print(prediction_proba)
    print(result)

    data_fresh = {
        "age": age,
        "bmi": bmi,
        "drinking": drinking,
        "exercise": exercise,
        "sex": sex,
        "junk": junk,
        "sleep": sleep,
        "smoking": smoking,
        "risk": result,
        "probability": prediction_proba
    }

    return jsonify(data_fresh)

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
