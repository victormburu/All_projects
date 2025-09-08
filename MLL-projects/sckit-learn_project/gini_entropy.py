import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.utils.multiclass import type_of_target
from pip._vendor.distro import __main__
#import seaborn as sns
def importdata():
    diabetes_data = pd.read_csv(r"../sckit-learn_project/Multiclass_Diabetes_Dataset.csv")
    print("dataset length:", len(diabetes_data))
    print("dataset shape:", diabetes_data.shape)
    print("dataset columns:", diabetes_data.columns)
    print("dataset description:", diabetes_data.describe())
    print("dataset head:", diabetes_data.head())
    return diabetes_data

def splitdataset(diabetes_data):
    X = diabetes_data.iloc[:, :-1]
    Y = diabetes_data.iloc[:, -1].astype('int')
# Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test
# Creating and training the model

def train_using_gini(X_train , X_test, y_train):
    clf_gini = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=3, random_state=100, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    clf_entropy = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=3, random_state=100, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print("Report : \n", classification_report(y_test, y_pred))
    
def plot_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(12,8))
    tree.plot_tree(clf_object.estimators_[0], feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree using Gini Index")
    plt.show()
    
if __name__ == '__main__':
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
    feature_names = data.columns[:-1]
    print(type_of_target(Y))
    class_names = ['0', '1', '2']
    plot_tree(clf_gini, feature_names, class_names)
    
    print("Feature Importances (Gini):", clf_gini.feature_importances_) 
    print("Feature Importances (Entropy):", clf_entropy.feature_importances_)
    