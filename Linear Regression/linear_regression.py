# Author: Nishank Bhatnagar
# Machine Learning: Implements Linear Regression Algorithm and K-Fold cross validation technique
# evaluates using Iris Dataset

import numpy as np
import math
from sklearn.metrics import accuracy_score
import pandas as pd

"""
    This function Checks the Accuracy of Predicted class label vs the Actual Class label
    
    returns Accuracy percentage 
"""
def check_accuracy(actual,predicted):
    return accuracy_score(actual,predicted)*100

"""
    Normalizes the Predicted class value to nearest approximation
"""
def norm_class_value(X):
    for i in range(len(X)):
        dec = X[i] - int(X[i])
        if dec < 0.5:
            X[i] = math.floor(X[i])
        elif dec >= 0.5:
            X[i] = math.ceil(X[i])
    return X

""" Multi Fit : 
        Parameters :-
            X => DataSet X of one or more variable in form of matrix of [n_samples,n_features]
            Y => DataSet Y is the Class matrix in form of [n_samples,1]
            
        Fits the data by calculating Ordinary Least Square
        Formula: B = inverse(X'X).X'Y
        
        returns a vector of Coefficient values corresponding to Xij   
"""
def fit(X_values,Y_values):
    inverse = np.linalg.inv(np.dot(X_values.T, X_values))
    trans_prod = np.dot(X_values.T, Y_values)
    coeff = np.dot(inverse, trans_prod)
    return coeff


""" Multi Predict : 
        Parameters :-
            X => Testing DataSet X of one or more variable in form of matrix of [n_samples,n_features]
            B_Estimator => Vector of n_sample Coefficient value

        Predicts the Class value given a data set
        Formula: Y = B_estimator. X

        returns an array with predicted Values  
"""
def predict(x,b_estimator):
    return norm_class_value(np.dot(x, b_estimator))

""" Residual Error : 
        Parameters :-
            actual_values => Vector of testing Dataset Class Labels 
            predicted_values => Vector of n_sample predicted Class labels 

        Finds the square Residual Error vector and Root mean square
        Formula: 
        residual_error = actual_values - predicted_values
        root_mean_sqaure = Sum of square of residual_error / length of values exisiting

        returns an array with predicted Values  
"""
def residual_error(actual_values,predicted_values):
    e_residual = []
    square_error_total = 0.0

    for i in range(len(predicted_values)):
        e_residual.append(actual_values[i] - predicted_values[i])
        square_error_total += pow(e_residual[i],2)
    root_mean_square = square_error_total/float(len(actual_values))
    error_res = pd.DataFrame(e_residual, columns=['Residual Error'])
    return (error_res,root_mean_square)

""" K Fold Validation : 
        Parameters :-
            Data =>  DataSet of one or more variable in form of matrix of [n_samples,n_features]
            k => K value used to split the Dataset
            classifier => Function of Classifer to be used [In our case only 1 Linear Regression therefore Fit]

        Breaks the dataset into K parts where K-1 Dataset is used as Training Data and rest is used as Testing data

        returns an array with accuracy scores of the predicted class label from Testing Datasets   
"""
def k_fold_validation(data, k, classifier):
    accuracy_data = []

    if k <= 1:
        print("Choose K greater than 1")
        return
    else:
        split = int(len(data) / k)
        for i in range(1, k + 1):
            rng_from, rng_to = split * (i - 1), split * i
            training_data = data.iloc[0:rng_from, :].append(data.iloc[rng_to:, :], ignore_index=True)
            validation_data = data.iloc[rng_from:rng_to]
            training_X, training_Y = training_data.iloc[:, :4], training_data.iloc[:, 4:]
            coeff_val = fit(training_X, training_Y)
            testing_X, testing_Y = validation_data.iloc[:, :4], validation_data.iloc[:, 4:]
            test_prediction = predict(testing_X, coeff_val)
            accuracy_data.append(check_accuracy(testing_Y, test_prediction))

    return accuracy_data

if __name__ == '__main__':

    col_name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    # Replaces the Class Value from String to Numerical Value
    class_val = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}

    data_iris = pd.read_csv("iris_dataset.csv", names=col_name, header=None)
    data_iris = data_iris.replace({'class': class_val})
    print("The DataSet Iris :- ")
    print(data_iris)

    """
        Next step is to check the Training functionality of the Model and 
        its ability to predict Class Labels given a random Dataset  
    """
    print("===========================================")
    print("Training the model with Iris Dataset ....")
    X_value_of_dataset = data_iris.iloc[:,:4]
    Y_value_of_dataset = data_iris.iloc[:,4:]
    coeff_of_data = fit(X_value_of_dataset,Y_value_of_dataset)
    print("===========================================")
    print("The Coeffecient of dataset calculated : ",coeff_of_data.T)

    """ 
        Testing the predict function using the calculated Coeff 
        with known result to compare Predicted Vs Actual
    """

    test_data_iris = pd.read_csv("iris_dataset_test.csv", names=col_name, header=None)
    test_data_iris = test_data_iris.replace({'class': class_val})
    test_data_X,test_data_Y = test_data_iris.iloc[:,:4],test_data_iris.iloc[:,4:]
    print("===========================================")
    print('Predicting Values .... ')
    predicted_values = predict(test_data_X,coeff_of_data)
    print('The Predicted Values are : ', predicted_values.T)
    print('The Actual Values are : ', test_data_Y.values.T)
    print("===========================================")

    print('The accuracy score of Predicted values Vs the Actual values is : {}%'.format(check_accuracy(test_data_Y,predicted_values)))
    print("===========================================")
    res_error, root_mean_sq = residual_error(test_data_Y.values,predicted_values)
    print("The Residual error :")
    print(res_error.T)
    print("Root mean square : ")
    print(root_mean_sq)

    """
        Performing K Fold Cross Validation on the DataSet
    """
    print("After performing K-Fold with K=3 Cross Validation : ")
    shuffled_data = data_iris.sample(frac=1)
    accuracy_array = k_fold_validation(shuffled_data,3,fit)
    print(" ")

    print(accuracy_array)
    print("Average accuracy count : {}%".format(sum(accuracy_array)/len(accuracy_array)))
    print("===========================================")
