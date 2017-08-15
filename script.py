import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args      # labeli is 50000 x 1

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    
    ones = np.ones((n_data, 1))
    data = np.append(ones, train_data, axis = 1) #50000 x 716
    # Initial weights is 716 x 1

    #theta = np.reshape(sigmoid(np.dot(data, np.transpose(initialWeights))), (n_data, 1)) # theta is 50000 x 1
    theta = np.reshape(sigmoid(np.dot(data, initialWeights)), (n_data, 1)) # theta is 50000 x 1
    
    #errorsum = 0
    #gradientsum = np.zeros((1, n_features + 1))

    log_theta = np.log(theta)
    #print(log_theta.shape)
    #print('++++++++++++++++')

    subtract_label = np.subtract(ones, labeli)
    #print(subtract_label.shape)
    #print('++++++++++++++++')

    subtract_theta_log = np.log((np.subtract(ones, theta)))
    #print(subtract_theta_log.shape)
    #print('++++++++++++++++')

    errorsum = np.add(np.dot(np.transpose(log_theta), labeli), np.dot(np.transpose(subtract_theta_log), subtract_label))
    
    #mult_subtracts = np.multiply(subtract_label, subtract_theta_log)
    #mult_first_part = np.transpose(np.multiply(log_theta, labeli))
    #errorsum = np.add(mult_first_part, mult_subtracts)

    #print(errorsum.shape)
    error = (-1/n_data)*errorsum
    #error = (-1/n_data)*(np.sum(errorsum))

    #print(error)

    subtract_theta_labeli = np.subtract(theta, labeli)

    #print(subtract_theta_labeli.shape)
    #print(data.shape)

    #error_grad = ((1/n_data)*np.dot(np.transpose(subtract_theta_labeli), data)).flatten()
    error_grad = ((1/n_data)*np.dot(data.T, subtract_theta_labeli)).flatten()
    
    #error_grad = ((1/n_data)*np.multiply(subtract_theta_labeli, data)).flatten()
    '''
    gsum = np.zeros((1, n_feature + 1))
    for i in range(n_data):
        gsum = np.add(gsum,(subtract_theta_labeli[i]*data[i,:]))
    error_grad = np.multiply((1/n_data),gsum).flatten()
    '''

    #print(error_grad.shape)
    '''
    for i in range(n_data):
    	errorsum = errorsum + ((labeli[i]*np.log(theta[i])) + ((1 - labeli[i])*(np.log(1 - theta[i]))))
    	gradientsum = np.add(gradientsum,((theta[i] - labeli[i])*data[i,:]))
    error = (-1/n_data)*errorsum
    error_grad = np.multiply((1/n_data),gradientsum)
    error_grad = np.transpose(error_grad).flatten()
    '''
    
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    ones = np.ones((data.shape[0], 1))
    data = np.append(ones, data, axis = 1)
    #sig = sigmoid((np.dot(np.transpose(W), np.transpose(data))))
    probabilities = sigmoid(np.dot(data, W))
    #label = np.argmax(probabilities, axis=1)

    for i in range(data.shape[0]):
        label[i] = np.argmax(probabilities[i,:])

    #print(label)
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    ones = np.ones((n_data, 1))
    data = np.append(ones, train_data, axis = 1) #50000 x 716
    initialWeights = params.reshape((n_feature + 1, n_class)) # 716 x 10
    theta_Num = np.exp(np.dot(data, initialWeights)) # 50000 x 10
    for row in range(n_data):
    	theta_Num[row] /= np.sum(theta_Num[row])
    log_Theta = np.log(theta_Num)
    error = -np.sum(labeli*log_Theta)
    '''
    error_array = np.dot(np.transpose(log_Theta), labeli)
    print(error_array)
    for i in range(error_array.shape[0]):
    	error = error + error_array[0][i]
    error = -1*error
    '''
    
    theta_Subtract_label = np.subtract(theta_Num,labeli)
    error_grad = np.dot(data.T, theta_Subtract_label).flatten()
    
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = data.shape[0]
    label = np.zeros((n_data, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    ones = np.ones((n_data, 1))
    data = np.append(ones, data, axis = 1) #50000 x 716
    probabilities = sigmoid(np.dot(data, W))
    for i in range(data.shape[0]):
        label[i] = np.argmax(probabilities[i,:])

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))

initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

C = [1.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]

print('\nLinear Kernel with all other parameters default')
clf = SVC(kernel = 'linear')
clf.fit(train_data,train_label)
predicted_label = clf.predict(train_data)
print('\n Training set Accuracy:' + str(accuracy_score(predicted_label, train_label)*100) + '%')
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(accuracy_score(predicted_label, validation_label)*100) + '%')
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(accuracy_score(predicted_label, test_label)*100) + '%')

print('\nRBF Kernel with Gamma = 1 and all other parameters default')
clf = SVC(kernel = 'rbf', gamma = 1.0)
clf.fit(train_data,train_label)
predicted_label = clf.predict(train_data)
print('\n Training set Accuracy:' + str(accuracy_score(predicted_label, train_label)*100) + '%')
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(accuracy_score(predicted_label, validation_label)*100) + '%')
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(accuracy_score(predicted_label, test_label)*100) + '%')

print('\nRBF Kernel with Gamma = default')
clf = SVC(kernel = 'rbf', gamma = 'auto')
clf.fit(train_data,train_label)
predicted_label = clf.predict(train_data)
print('\n Training set Accuracy:' + str(accuracy_score(predicted_label, train_label)*100) + '%')
predicted_label = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(accuracy_score(predicted_label, validation_label)*100) + '%')
predicted_label = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(accuracy_score(predicted_label, test_label)*100) + '%')

size_C = len(C)
for c in C:
	print('\nRBF Kernel with Gamma = default and C = ' + str(c))
	clf = SVC(kernel = 'rbf', gamma = 'auto', C = c)
	clf.fit(train_data,train_label)
	predicted_label = clf.predict(train_data)
	print('\n Training set Accuracy:' + str(accuracy_score(predicted_label, train_label)*100) + '%')
	predicted_label = clf.predict(validation_data)
	print('\n Validation set Accuracy:' + str(accuracy_score(predicted_label, validation_label)*100) + '%')
	predicted_label = clf.predict(test_data)
	print('\n Testing set Accuracy:' + str(accuracy_score(predicted_label, test_label)*100) + '%')


f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
