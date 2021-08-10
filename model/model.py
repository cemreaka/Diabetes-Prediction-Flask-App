import pandas as p
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DiabetesPrediction = p.read_csv(
    'C:\\Users\\cemre\\Documents\\repos\\Diabetes-Prediction-Flask-App\\Dataset\\diabetes.csv')

X = DiabetesPrediction[['Pregnancies', 'Glucose', 'BloodPressure',
                        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = DiabetesPrediction['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=10)

lm = LogisticRegression(solver='liblinear')
lm.fit(X_train, Y_train)

pickle.dump(lm, open('model.pickle', 'wb'))
