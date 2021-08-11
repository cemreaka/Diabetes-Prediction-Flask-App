import pandas as p
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DiabetesPrediction = p.read_csv(
    'C:\\Users\\cemre\\Documents\\repos\\Diabetes-Prediction-Flask-App\\Dataset\\diabetes.csv')

X = DiabetesPrediction[['Pregnancies', 'Glucose', 'BloodPressure',
                        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = DiabetesPrediction['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=3)

lm = LogisticRegression(solver='liblinear')
lm.fit(X_train, y_train)
print(lm.score(X_test, y_test))
pickle.dump(lm, open('model.pickle', 'wb'))
