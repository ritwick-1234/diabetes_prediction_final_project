from flask import Flask, render_template, request, redirect, url_for,session
import mysql.connector
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key=os.urandom(24)

users={}
# Establish database connection
conn = mysql.connector.connect(
    host='bpnzb7jhgtwbssiuxyin-mysql.services.clever-cloud.com',
    user='uehzwawa3yrglua7',
    password='0QMxaEXCLFSAJKgWWSz3',
    database='bpnzb7jhgtwbssiuxyin'
)

cursor = conn.cursor()
# with open('svm_model_abcdefgh.pkl', 'rb') as f:
#    svm_model=pickle.load(open('svm_model_abcdefgh.pkl','rb'))
# filename = 'svm_model_abc.pkl'
# classifier = pickle.load(open(filename, 'rb'))
# @app.route('/register')
# def register():
#     return render_template('register.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/home')
# def home():
#     if 'user_id' in session:
#         return render_template('home.html')
#     else:
#         return redirect('/login')

# @app.route('/login')
# def login():
    
#     return render_template('login.html')
# @app.route('/')
# @app.route('/register')
# def register():
#     return render_template('register.html')

# @app.route('/home')
# def home():
#     if 'user_id' in session:
#         return render_template('home.html')
#     else:
#         return redirect('/')

# @app.route('/predict')
@app.route('/')
@app.route('/register')
def register():
    
    
    return render_template('register.html')

@app.route('/login')
def login():
    
    return render_template('login.html')


@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/register')


@app.route('/predict')
def predict():
     return redirect("https://svmfinalfordiabetesprediction-l3nb98jmvovdbzsq4ljrt7.streamlit.app/")
# def std_scalar(df):
#     std_X=StandardScaler()
#     x=pd.DataFrame(std_X.fit_transform(df))
#     return x
# def pipeline(features):
#     steps=[('scaler',StandardScaler()),('SVM', )]
#     pipe=Pipeline(steps)
#     return pipe.fit_transform(features)

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    
    cursor.execute("SELECT * FROM `flask_1` WHERE `email` = %s AND `password` = %s", (email, password))
    users = cursor.fetchall()
    
    if len(users) > 0:
        session['user_id']=users[0][0]
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login', message='Login failed, please try again.'))
       

# @app.route('/add_user', methods=['POST'])
# def add_user():
#     first_name = request.form.get('first_name')
#     last_name = request.form.get('last_name')
#     email = request.form.get('Uemail')
#     password = request.form.get('Upassword')
#     confirm_password = request.form.get('confirm_password')
#     phone_number = request.form.get('phone_number')

#     # Check if password and confirm_password match
#     if password != confirm_password:
#         return "Passwords do not match", 400

#     try:
#         cursor.execute("""
#             INSERT INTO `flask_1` (`user_id`,`first_name`, `last_name`, `email`, `password`, `phone_number`)
#             VALUES (NULL,%s, %s, %s, %s, %s)
#         """, (first_name, last_name, email, password, phone_number))
#         conn.commit()
#     except Exception as e:
#         return f"An error occurred: {str(e)}", 500

#     return "Your registration is successful", 200
#     cursor.execute("""select * from `flask_1` where `email` like '{}'""".format(email) )
#     myuser=cursor.fetchall()
#     session['user_id']=myuser[0][0]
#     return redirect('/')
@app.route('/add_user', methods=['POST'])
def add_user():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('Uemail')
    password = request.form.get('Upassword')
    confirm_password = request.form.get('confirm_password')
    phone_number = request.form.get('phone_number')

    # Check if password and confirm_password match
    if password != confirm_password:
        return "Passwords do not match", 400

    try:
        # Insert user into database
        cursor.execute("""
            INSERT INTO `flask_1` (`user_id`, `first_name`, `last_name`, `email`, `password`, `phone_number`)
            VALUES (NULL, %s, %s, %s, %s, %s)
        """, (first_name, last_name, email, password, phone_number))
        conn.commit()

        # Retrieve inserted user to get user_id
        cursor.execute("SELECT * FROM `flask_1` WHERE `email` = %s", (email,))
        myuser = cursor.fetchall()

        # Store user_id in session
        session['user_id'] = myuser[0][0]

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

    # Redirect to home page
    return redirect('/')


@app.route('/logout')
def logout():
    # This will display a logout alert and then redirect to the login page
    return '''
        <script>
            alert("You are logged out.");
            window.location.href = "/";
        </script>
    '''
@app.route('/home_1')
def home_1():
    return render_template('home_1.html')
@app.route('/about_project')
def about_project():
    return render_template('about_project.html')
@app.route('/contact')
def contact():
    return redirect('https://contactpage-d3yzpumewz4zwdsmttdybf.streamlit.app/')



# df1 = pd.read_csv('diabetes.csv')

# # Renaming DiabetesPedigreeFunction as DPF
# df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# # Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
# # df_copy = df1.copy(deep=True)
# df1[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df1[['Glucose', 'BloodPressure',
#                                                                                     'SkinThickness', 'Insulin',
#                                                                                     'BMI']].replace(0, np.nan)

# # Replacing NaN value by mean, median depending upon distribution
# df1['Glucose'].fillna(df1['Glucose'].mean(), inplace=True)
# df1['BloodPressure'].fillna(df1['BloodPressure'].mean(), inplace=True)
# df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace=True)
# df1['Insulin'].fillna(df1['Insulin'].median(), inplace=True)
# df1['BMI'].fillna(df1['BMI'].median(), inplace=True)

# # Model Building

# X = df1.drop(columns='Outcome')
# y = df1['Outcome']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# # Creating Random Forest Model

# classifier = RandomForestClassifier(n_estimators=20)
# classifier.fit(X_train, y_train)

# # Creating a pickle file for the classifier
# filename = 'svm_model_abcdefgh.pkl'
# pickle.dump(classifier, open(filename, 'wb'))
#####################################################################










@app.route('/predict_diabetes')
def predict_diabetes():
    pass

    
    # if request.method == 'POST':
    #     preg = request.form['pregnancies']
    #     glucose = request.form['glucose']
    #     bp = request.form['bloodpressure']
    #     st = request.form['skinthickness']
    #     insulin = request.form['insulin']
    #     bmi = request.form['bmi']
    #     dpf = request.form['dpf']
    #     age = request.form['age']

    #     data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
    #     my_prediction = classifier.predict(data)

    #     return render_template('predict_diabetes.html', prediction=my_prediction)




     # Get input values from the form and convert to float
    # features =[float() for x in request.form.values()] #float(request.form[field]) for field in request.form

    # # Convert features list to NumPy array and reshape to (1, num_features)
    # final_features = [np.array(features)]

    # # Perform feature scaling using std_scalar function
    

    # # Perform prediction using SVM model
    # prediction = svm_model.predict(final_features)

    # # Determine prediction result
    
    # if prediction == 1:
    #     result="The preson is diabetic" 
    # else:
    #     return "not diabetic"
        

    # # Extract individual feature values for rendering
    # Pregnancies = request.form['Pregnancies']
    # Glucose = request.form['Glucose']
    # BloodPressure = request.form['BloodPressure']
    # SkinThickness = request.form['SkinThickness']
    # Insulin = request.form['Insulin']
    # BMI = request.form['BMI']
    # DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    # Age = request.form['Age']

    # # Render template with prediction result and input values
    # return render_template('predict_diabetes.html', preg=Pregnancies, bp=BloodPressure,
    #                        gluc=Glucose, st=SkinThickness, ins=Insulin, bmi=BMI,
    #                        dbf=DiabetesPedigreeFunction, age=Age, res=result)

     

    # Get input values from the form
    
    
    #Feature tranform and prediction using pipeline
    # We can now use predictions from this feature_tranformed variable
    #feature_tranformed= pipeline(final_features)



    # Using standard scalar method
    
    # Pregnancies = int(request.form['Pregnancies'])
    # Glucose = int(request.form['Glucose'])
    # BloodPressure = int(request.form['BloodPressure'])
    # SkinThickness = int(request.form['SkinThickness'])
    # Insulin = int(request.form['Insulin'])
    # BMI = float(request.form['BMI'])
    # DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    # Age = int(request.form['Age'])

    # # Prepare input data for prediction
    # input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]])

    # # Perform prediction
    # prediction = model.predict(input_data)

    # # Determine prediction result
    # if prediction[0] == 1:
    #     result = " oops! The patient seems to be Diabetic"
    # else:
    #     result = " Hurrah! The patient seems not  to be Diabetic"

    # # Render template with prediction result
    # return render_template('predict_diabetes.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
