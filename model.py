import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib



iris = load_iris()
X, y = iris.data , iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 11)

model = RandomForestClassifier()
model.fit(Xtrain, ytrain)

joblib.dump(model,'rf_model.joblib')