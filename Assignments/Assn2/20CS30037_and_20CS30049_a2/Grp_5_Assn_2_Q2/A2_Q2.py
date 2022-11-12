# Assignment 2: Machine Learning
# Question 2: Supervised Learning (SVM, MLP, Feature Selection - Forward Selection, Ensemble Learning - Max Voting)
# Group No 5: 
#      20CS30037 - Pranav Nyati
#      20CS300449 - Shreyas Jena

#################################### IMPORTS ####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import random
from cmath import sqrt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

##################################### HELPER FUNCTIONS #############################

# function to split the data into train and test sets
def train_test_split(df, train_frac):

    train_num = round(train_frac*df.shape[0])
    train_idx = random.sample(range(df.shape[0] - 1), train_num)

    # splitting the train and test datasets based on indexes sampled
    train_df = df.iloc[train_idx]
    test_df = df.drop(train_df.index)

    return train_df, test_df

# funtion to split the dataframe into a dataframe with attribute values and a pandas series with the labels
def df_feature_label_split(df, label_name):
    df_features = df.drop(label_name, axis = 1)
    df_label = df[label_name]

    return df_features, df_label



#################################### READING AND PREPROCESSING DATA #######################################

# Reading the data from the file
dataframe = pd.read_csv('wine.csv', header= None)

# Adding the column names to the dataframe
dataframe.columns = ['Class Label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash' , 'Magnesium', 'Total phenols', 'Flavanoids', 
                     'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']


# Standard Scalar Normalization of the data

df_new = dataframe.copy(deep=True)
class_labels = dataframe.iloc[:, 0]

for i in range(1, 14):
    temp = dataframe.iloc[:, i]
    mean = temp.sum()/len(temp)
    variance =  ((temp - mean)**2).sum()/len(temp)
    sd = np.sqrt(variance)
    df_new.iloc[:, i] = (dataframe.iloc[:, i] - mean)/sd
    

# Train Test Split of the data = 80:20 split
train_data, test_data = train_test_split(df_new, train_frac=0.8)

# Spltting the train and test datasets into features and class labels
train_df_features, train_df_label = df_feature_label_split(train_data, 'Class Label')
test_df_features, test_df_label = df_feature_label_split(test_data,'Class Label')


#################################### BINARY SVM CLASSIFIER  #######################################
print("\nTRAINING OF BINARY SVM CLASSIFIERS WITH DIFFERENT KERNELS\n")

# Initialization of the SVM binary classifiers using the following three kernels:
# (A) Linear    (B) Quadratic   (C) Radial Basis Function 

# binary_SVM_clf1  => uses LINEAR kernel
binary_SVM_clf1 = svm.SVC(kernel='linear')
binary_SVM_clf1 = binary_SVM_clf1.fit(train_df_features, train_df_label)

predictions_SVM_clf1 = binary_SVM_clf1.predict(test_df_features)
accuracy_SVM_clf1 = accuracy_score(test_df_label, predictions_SVM_clf1)

# binary_SVM_clf2  => uses QUADRATIC kernel 
binary_SVM_clf2 = svm.SVC(kernel='poly', degree=2)
binary_SVM_clf2 = binary_SVM_clf2.fit(train_df_features, train_df_label)

predictions_SVM_clf2 = binary_SVM_clf2.predict(test_df_features)
accuracy_SVM_clf2 = accuracy_score(test_df_label, predictions_SVM_clf2)

# binary_SVM_clf3  => uses RADIAL BASIS FUNCTION
binary_SVM_clf3 = svm.SVC(kernel='rbf')
binary_SVM_clf3 = binary_SVM_clf3.fit(train_df_features, train_df_label)

predictions_SVM_clf3 = binary_SVM_clf3.predict(test_df_features)
accuracy_SVM_clf3 = accuracy_score(test_df_label, predictions_SVM_clf3)

# Printing the accuracies of the three SVM classifiers
print("Accuracy of SVM classifier using Linear Kernel: ", accuracy_SVM_clf1, '\n')
print("Accuracy of SVM classifier using Quadratic Kernel: ", accuracy_SVM_clf2, '\n')
print("Accuracy of SVM classifier using Radial Basis Function Kernel: ", accuracy_SVM_clf3, '\n')


#################################### MULTI LAYER PERCEPTRON CLASSIFIER #######################################
print("\nTRAINING OF TWO DIFFERENT MULTI-LAYER PERCEPTRONS\n ")

counts_MLP1 = 0
counts_MLP2 = 0

num_repeats = 20

accuracy_MLP_clfs = np.zeros((num_repeats, 2), dtype=np.float64)


# hyperparameters for the MLP Classifier
learn_rate1 = 0.001
optimizer = 'sgd'
max_iterations = 3000
batch_sz = 32

for i in range(0, num_repeats):

    # new train test split for each iteration
    train_data, test_data = train_test_split(df_new, train_frac=0.8)
    train_df_features, train_df_label = df_feature_label_split(train_data, 'Class Label')
    test_df_features, test_df_label = df_feature_label_split(test_data,'Class Label')

    # MLP Classifier 1
    MLP_clf1 = MLPClassifier((16), activation = 'relu', solver = optimizer, learning_rate_init = learn_rate1, batch_size = batch_sz, max_iter= max_iterations)
    MLP_clf1 = MLP_clf1.fit(train_df_features, train_df_label)
    predictions_MLP_clf1 = MLP_clf1.predict(test_df_features)
    accuracy_MLP_clf1 = accuracy_score(test_df_label, predictions_MLP_clf1)
    accuracy_MLP_clfs[i, 0] = accuracy_MLP_clf1


    # MLP Classifier 2
    MLP_clf2 = MLPClassifier((256, 16), activation = 'relu', solver = optimizer, learning_rate_init = learn_rate1, batch_size = batch_sz, max_iter= max_iterations)
    MLP_clf2 = MLP_clf2.fit(train_df_features, train_df_label)
    predictions_MLP_clf2 = MLP_clf2.predict(test_df_features)
    accuracy_MLP_clf2 = accuracy_score(test_df_label, predictions_MLP_clf2)
    accuracy_MLP_clfs[i, 1] = accuracy_MLP_clf2

    if(accuracy_MLP_clf1 - accuracy_MLP_clf2 >= 0.00001):
        counts_MLP1 += 1
    else:
        counts_MLP2 += 1


# Printing the accuracies of the two MLP classifiers averaged over 30 runs
# print(counts_MLP1)
print("Average Accuracy of MLP classifier using 1 layer with 16 hidden units: ", accuracy_MLP_clfs[:, 0].mean(), '\n')
# print(counts_MLP2)
print("Average Accuracy of MLP classifier using 2 layers with 256 and 16 hidden units: ", accuracy_MLP_clfs[:, 1].mean(), '\n')


# Selection of best MLP model out of the two MLP models
# Selection criterion:  The model which achieves higher test accuracy more num of times in training over 30 different random train-test split , is the best model

if (counts_MLP1 >= counts_MLP2):
    best_MLP_clf_layers = (16)
   
else:
    best_MLP_clf_layers = (256, 16)


################################### BEST MLP MODEL WITH VARIED LEARNING RATES #########################################
print("\nTRAINING THE BEST MODEL FROM PREVIOUS STAGE FOR DIFFERENT LEARNING RATES\n")

learn_rate_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
accuracy_dict = {}

for learn_rate in learn_rate_list:
    best_MLP_clf = MLPClassifier(best_MLP_clf_layers, activation = 'relu', solver = optimizer, learning_rate_init = learn_rate, batch_size = 32, max_iter= max_iterations)
    best_MLP_clf = best_MLP_clf.fit(train_df_features, train_df_label)
    preds_best_MLP_clf = best_MLP_clf.predict(test_df_features)
    accuracy_best_MLP_clf = accuracy_score(test_df_label, preds_best_MLP_clf)
    accuracy_dict[str(learn_rate)] = round(accuracy_best_MLP_clf*100, 4)
    print("Accuracy of best MLP classifier with learning rate = ", learn_rate, " is: ", accuracy_best_MLP_clf, '\n')
    plt.figure()
    plt.plot(best_MLP_clf.loss_curve_)
    plt.title("Loss Curve for best MLP model " +  str(best_MLP_clf_layers)+ " with learning rate = " + str(learn_rate))
    plt.xlabel("No of Iterations")
    plt.ylabel("Training Loss")
    plt.savefig("output_plots/MLP_loss_curve_learn_rate_" + str(learn_rate) + ".png")

# Plotting the final accuracies of the best MLP model for different learning rates
plt.figure()
plt.bar(list(accuracy_dict.keys()), list(accuracy_dict.values()), width= 0.5, label = accuracy_dict.values())
plt.title("Final accuracy for different initial learning rates of the best model")
i = 0
for learn_rate in learn_rate_list:
    plt.text(i, accuracy_dict[str(learn_rate)] + 4, str(accuracy_dict[str(learn_rate)]), ha='center', va='top')
    i += 1

plt.xlabel("Initial Learning Rates")
plt.ylabel("Accuracy (in %)")
plt.savefig("output_plots/MLP_final_accurcacy_vs_learn_rate.png")


################################### FEATURE SELECTION USING FORWARD SELECTION #########################################
print("\n FEATURE SELECTION USING FORWARD SELECTION\n")

feature_list = list(train_df_features.columns)
forward_feature_list = []   # list of features selected by forward selection

# dictionary to store the updated feature list after each iteration of forward selection
feature_list_per_iteration = {}
num_features = len(feature_list)

# numpy array to store the accuracies of the models with different number of features
accuracy_list = np.zeros(num_features, dtype=np.float32)
i = 0

# while loop of forward selection to select the next best feature which when added to the current feature list, gives the best accuracy
while True:

    # if the addition of previous best feature to the feature list already gives accuracy of 100%, then stop the loop
    if accuracy_list[i-1] > 0.999999:
        break

    
    max_accuracy = 0.0
    max_accuracy_attr = ""
    for feature in feature_list:
        feature_list_temp = forward_feature_list.copy() + [feature]
        train_df_features_temp = train_df_features[feature_list_temp]
        MLP_clf = MLPClassifier(best_MLP_clf_layers, activation = 'relu', solver = optimizer, learning_rate_init = learn_rate1, batch_size = 32, max_iter= max_iterations)
        MLP_clf = MLP_clf.fit(train_df_features_temp, train_df_label)
        predictions_MLP_clf = MLP_clf.predict(test_df_features[feature_list_temp])
        accuracy_MLP_clf = accuracy_score(test_df_label, predictions_MLP_clf)
        if accuracy_MLP_clf - max_accuracy > 0.000001:
            max_accuracy = accuracy_MLP_clf
            max_accuracy_attr = feature
        
    forward_feature_list.append(max_accuracy_attr)
    feature_list.remove(max_accuracy_attr)
    accuracy_list[i] = max_accuracy
    feature_list_per_iteration[i+1] = forward_feature_list.copy()
    print("Iteration: ", i+1, " Accuracy: ", max_accuracy, " Feature: ", max_accuracy_attr)
    
    if i > 0:
        if accuracy_list[i] - accuracy_list[i-1] < 0.0000001:
            break

    if len(feature_list) == 0:
        break

    print("Forward Feature List for iteration ", i+1, " is ", feature_list_per_iteration[i+1])
    i += 1


######################################## ENSEMLE LEARNING (MAX VOTING) #########################################
print("\n ENSEMBLE LEARNING USING MAX VOTING\n")

# Ensemble Learning - Max Voting

# Models in the ensemble:
#       1. binary_SVM_clf2 => SVM using Quadratic Kernel
#       2. binary_SVM_clf3 => SVM using Radial Basis Function
#       3. MLP_clf2 => Multi-Layer Perceptron with 2 hidden layers of 256 and 16 neurons respectively

# Predictions from the 3 models
#  predictions_SVM_clf2 => SVM using Quadratic Kernel
#  predictions_SVM_clf3 => SVM using Radial Basis Function
#  predictions_MLP_clf2 => Multi-Layer Perceptron with 2 hidden layers of 256 and 16 neurons respectively


# Training the best MLP model for a random train-test split

# a new train-test split is created to train the best MLP model
train_data, test_data = train_test_split(df_new, train_frac=0.8)
train_df_features, train_df_label = df_feature_label_split(train_data, 'Class Label')
test_df_features, test_df_label = df_feature_label_split(test_data,'Class Label')

# training the best MLP model
best_MLP_clf_new = MLPClassifier(best_MLP_clf_layers, activation = 'relu', solver = optimizer, learning_rate_init = learn_rate1, batch_size = 32, max_iter= max_iterations)
best_MLP_clf_new = best_MLP_clf_new.fit(train_df_features, train_df_label)
# predictions from the best MLP model
predictions_best_MLP_clf_new = best_MLP_clf.predict(test_df_features)
accuracy_best_MLP_clf_new = accuracy_score(test_df_label, predictions_best_MLP_clf_new)


# Concatenating the predictions from the 3 models
all_model_preds = np.column_stack((predictions_SVM_clf2, predictions_SVM_clf3, predictions_best_MLP_clf_new))

# Assigning the ensemble prediction as the class label with highest frequency for each test sample (MAX VOTING)
ensemble_preds = np.zeros(all_model_preds.shape[0], dtype=np.int32)
for i in range(all_model_preds.shape[0]):
    ensemble_preds[i] = np.bincount(all_model_preds[i]).argmax()

# print([predictions_SVM_clf2[i] for i in range(len(predictions_SVM_clf2))])
# print([predictions_SVM_clf3[i] for i in range(len(predictions_SVM_clf3))])
# print([predictions_best_MLP_clf_new[i] for i in range(len(predictions_best_MLP_clf_new))])
# print([ensemble_preds[i] for i in range(len(ensemble_preds))])

# Accuracy of the ensemble model
accuracy_max_voting = accuracy_score(test_df_label, ensemble_preds)

# Printing the accuracies of the 3 models and the ensemble model for comparison
print("Accuracy of Binary SVM using Quadratic Kernel: ", accuracy_SVM_clf2)
print("Accuracy of Binary SVM using Radial Basis Function: ", accuracy_SVM_clf3)
print("Accuracy of the best Multi-Layer Perceptron model: ", accuracy_best_MLP_clf_new)
print("Accuracy of Ensemble Learning using Max Voting: ", accuracy_max_voting)

print("\n Execution is complete. \n")