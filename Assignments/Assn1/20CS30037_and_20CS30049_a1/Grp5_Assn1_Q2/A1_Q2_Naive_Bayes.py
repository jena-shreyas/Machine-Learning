import numpy as np
import pandas as pd
from math import sqrt, exp, pi

class NaiveBayesClassifier(object):
    def __init__(self, alpha = 0):
        self.alpha = alpha

    def read_file(self, filename):
        self.data = pd.read_csv(filename)
        self.data.drop(columns = 'ID', inplace = True)
        print("Data samples before outlier removal : {}".format(self.data.shape[0]))

    def preprocess(self):

        # Find numerical and categorical features
        cat_headers = [key for key in self.data.keys() if self.data[key].dtype == 'object']
        num_headers = [key for key in self.data.keys() if self.data[key].dtype != 'object']
        
        # store indices of numerical columns, we'll use it later while calculating likelihoods
        self.num_col_indexes = list()

        for i, key in enumerate(self.data.keys()):
            if key in num_headers:
                self.num_col_indexes.append(i)

        # Replace nans in numerical feature columns with their mean
        for num_head in num_headers:
            self.data[num_head] = self.data[num_head].fillna(round(self.data[num_head].mean()))

        # Replace nans in numerical feature columns with their mode
        for cat_head in cat_headers:
            self.data[cat_head] = self.data[cat_head].fillna(self.data[cat_head].mode().iloc[0])

        self.cat_maps = dict()
        # Convert categorical features to label encoded values
        for cat_head in cat_headers:
            self.cat_maps[cat_head] = dict()
            for i, val in enumerate(np.unique(self.data[cat_head])):
                self.cat_maps[cat_head][val] = i
            self.data[cat_head] = self.data[cat_head].apply(lambda x: self.cat_maps[cat_head][x])

        # Define map b/w output classes and their labels
        output_head = self.data.keys()[-1]
        self.output_map = self.cat_maps[output_head]

        # Calculate mean, std_dev and outlier threshold for numerical categories
        # Find indices of samples with outlier values and remove them
        self.num_stats = dict()
        outliers = dict()
        outlier_indexes = list()

        for i, key in enumerate(self.data.keys()):
            if key in num_headers:
                mean = self.data[key].mean()
                std = self.data[key].std()
                threshold = mean + 3*std
                self.num_stats[i] = mean, std

                outliers[key] = self.data.loc[self.data[key] > threshold]         # Extract instances for which given feature's value exceeds its threshold
                outlier_indexes = outlier_indexes + list(outliers[key].index)          # Store the corresponding indices for these instances

        # Drop samples with outlier values
        outlier_indexes = list(set(outlier_indexes))        # To obtain a list of unique indices for instances having outlier values
        self.data.drop(outlier_indexes, inplace = True)     # Drop data instances with outlier values

        print("Data samples after outlier removal : {}".format(self.data.shape[0]))

        # Normalize numerical values
        for i in self.num_col_indexes:
            mean = self.num_stats[i][0]
            std = self.num_stats[i][1]
            self.data[num_head] = (self.data[num_head] - mean) / std


    def split(self, ratio = 0.8):
        # Split data into training and testing sets (numpy arrays)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        
        self.classes = np.unique(self.y)
        num_examples = self.X.shape[0]
        num_train = int(ratio * num_examples)

        # Shuffle data
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        self.X_train = self.X[indices[:num_train]]
        self.X_test = self.X[indices[num_train:]]
        self.y_train = self.y[indices[:num_train]]
        self.y_test = self.y[indices[num_train:]]

        print("Training set size : {}".format(self.X_train.shape[0]))
        print("Testing set size : {}".format(self.X_test.shape[0]))

        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def compute_priors(self, y):
        priors = dict()
        for c in self.classes:
            priors[c] = np.sum(y == c) / len(y)
        return priors

    def gaussian_probability(self, x, mean, std):

        exponent = exp(-((x-mean)**2 / (2 * std**2)))
        return (1 / (sqrt(2 * pi) * std)) * exponent

    def compute_likelihoods(self, X, y):

        likelihoods = dict()
        for c in self.classes:
            likelihoods[c] = dict()

            for i in range(X.shape[1]):
                # If feature is categorical, we compute and store likelihoods for each unique value
                # If feature is numerical, we won't store likehoods, will directly compute gaussian probability
                if i not in self.num_col_indexes:
                    
                    likelihoods[c][i] = dict()
                    for val in np.unique(X[:, i]):

                        num_unique_vals = len(np.unique(X[:, i]))
                        num_cond = (X[:, i] == val) & (y == c)
                        denom_cond = (y == c)
                        likelihoods[c][i][val] = (np.sum(num_cond) + self.alpha)/ (np.sum(denom_cond) + self.alpha * num_unique_vals)

        return likelihoods

    def train(self, X, y, num_folds):
        # Split data into num_folds folds
        X_folds = np.array_split(X, num_folds)
        y_folds = np.array_split(y, num_folds)

        # Perform num_folds-fold cross validation
        accuracies = []
        for i in range(num_folds):
            X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
            y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
            X_val = X_folds[i]
            y_val = y_folds[i]

            self.priors = self.compute_priors(y_train)
            self.likelihoods = self.compute_likelihoods(X_train, y_train)

            y_pred_val = self.predict(X_val)
            accuracy = np.sum(y_pred_val == y_val) / len(y_val)
            accuracies.append(accuracy)

        val_accuracy = np.mean(accuracies)
        print("Laplace smoothing factor : {}".format(self.alpha))
        print("{}-fold validation accuracy : {}".format(num_folds, val_accuracy))
        return val_accuracy

    def predict(self, X):
    # compute posterior probabilities using the calculated priors and likelihoods
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = 0

                for i in range(len(x)):
                    if i in self.num_col_indexes:
                        mean = self.num_stats[i][0]
                        std = self.num_stats[i][1]
                        likelihood += np.log(self.gaussian_probability(x[i], mean, std))

                    else:
                        likelihood += np.log(self.likelihoods[c][i][x[i]])
                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

    def test(self, X, y):
    # test the model
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        print("Test accuracy : {}".format(accuracy))
        return accuracy


if __name__ == '__main__':
    print("\nWITHOUT LAPLACE SMOOTHING ...\n")
    bayes_classifier = NaiveBayesClassifier()
    bayes_classifier.read_file('Dataset_A.csv')
    bayes_classifier.preprocess()
    X_train, y_train, X_test, y_test = bayes_classifier.split(ratio = 0.8)
    k_fold_val_accuracy = bayes_classifier.train(X_train, y_train, num_folds = 10)
    test_accuracy = bayes_classifier.test(X_test, y_test)

    print("\nUSING LAPLACE SMOOTHING FACTOR ...\n")
    alpha = float(input("Enter Laplace smoothing factor : "))
    bayes_classifier = NaiveBayesClassifier(alpha=alpha)
    bayes_classifier.read_file('Dataset_A.csv')
    bayes_classifier.preprocess()
    X_train, y_train, X_test, y_test = bayes_classifier.split(ratio = 0.8)
    k_fold_val_accuracy = bayes_classifier.train(X_train, y_train, num_folds = 10)
    test_accuracy = bayes_classifier.test(X_test, y_test)
