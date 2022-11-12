# Assignment 1: Machine Learning
# Question 1: Decision Tree Classifier using ID3 Algorithm
# Group No 5: 
#      20CS30037 - Pranav Nyati
#      20CS300449 - Shreyas Jena

#################################### IMPORTS ####################################

# import libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

#######################################    CLASS DECLARATIONS     ##############################################

# Declaration of Node Class and DecisionTreeClassifier Class


# CLASS Node - This class is used to create a node in the decision tree
class Node:
    
    def __init__(self):
        self.attr_name = ''             # stores the attribute name for that node
        self.outcome_label = None       # only for a leaf node, it stores the outcome value 
        self.parent = None              # parent is an object of class Node
        self.children = []              # list to store children nodes of this node
        self.leaf_flag = False          # flag to check if the node is a leaf node
        self.parent_edge_label = ''     # stores the attr_val of the parent node from which current node has been derived
        self.depth = 0

    # method to check if current node is a leaf node
    def is_leaf_node(self):
        return self.leaf_flag

    # method to count the no of nodes in the subtree rooted at the current node 
    def subtree_node_count(self):
        if self.is_leaf_node():
            return 1
        else:
            count = 1
            for child in self.children:
                count += child.subtree_node_count()
            return count

    # method to find the depth of the subtree rooted at the current node
    def subtree_depth(self):
        if self.is_leaf_node():
            return 0
        else:
            depth = 1
            for child in self.children:
                depth = max(depth, 1 + child.subtree_depth())
            return depth


# CLASS DecisionTreeClassifier - This class is used to create a decision tree classifier

class DecisionTreeModel:

    def __init__(self):

        self.root = None          # root of the Decision Tree
        self.depth = 0            # depth of the Decision Tree
        self.num_nodes = 0


    # class method to train the ID3 tree from train data
    def train_ID3_tree(self, train_df, train_labels):

        attr_list = list(train_df.columns)
        attr_list.remove('ID')
        depth_ctr = 0
        self.root = self.decision_tree_recur(train_df, train_labels, attr_list, None, '', depth_ctr)
        return
        
    # class method that recursively builds the decision tree
    def decision_tree_recur(self, train_df_subset, train_labels_subset, attr_list, parent_node, parent_attr_val, depth_ctr):

        # if all the training samples of the training subset have the same output label
        if (len(train_labels_subset.unique()) == 1):
           leaf_node = Node()
           leaf_node.parent = parent_node
           leaf_node.leaf_flag = True
           leaf_node.parent_edge_label = parent_attr_val
           leaf_node.outcome_label = train_labels_subset.unique()[0]
           leaf_node.num_samples = train_labels_subset.shape[0]
           leaf_node.depth = depth_ctr
       
           return leaf_node

        # if all the features have been once covered in the decision tree, but all training examples at this node have different output label, 
        # then return the most probable class
        if (len(attr_list) == 0):

            leaf_node = Node()
            leaf_node.parent = parent_node
            leaf_node.leaf_flag = True
            leaf_node.parent_edge_label = parent_attr_val
            leaf_node.outcome_label = train_labels_subset.value_counts().idxmax()
            leaf_node.num_samples = train_labels_subset.shape[0]
            leaf_node.depth = depth_ctr

            return leaf_node

        # else select the best attribute that maximizes the info gain
        max_gain_attr, max_info_gain = find_max_gain_attr(train_df_subset, train_labels_subset, attr_list)
        if (max_info_gain >= 0.1):     # only select a node if it leads to an info gain of >= 0.1, else make it leaf node
            non_leaf_node = Node()
            non_leaf_node.attr_name = max_gain_attr
            non_leaf_node.parent = parent_node
            non_leaf_node.parent_edge_label = parent_attr_val
            non_leaf_node.children = []
            non_leaf_node.leaf_flag = False
            non_leaf_node.depth = depth_ctr
            non_leaf_node.num_samples = train_labels_subset.shape[0]

        else:
            leaf_node = Node()
            leaf_node.parent = parent_node
            leaf_node.leaf_flag = True
            leaf_node.parent_edge_label = parent_attr_val
            leaf_node.outcome_label = train_labels_subset.value_counts().idxmax()
            leaf_node.depth = depth_ctr
            leaf_node.num_samples = train_labels_subset.shape[0]

            return leaf_node

        # remove the max_gain_attr to ensure that it does not appear as an attribute down the tree again
        if (max_gain_attr in attr_list):
            attr_list_sub = attr_list.copy()
            attr_list_sub.remove(max_gain_attr)
             
        # recursively train the children node 
        for attr_val in list(train_df_subset[max_gain_attr].unique()):

            # to obtain the subset of data corresponding to a particular attr value of the max_gain_attr
            train_df_sub = train_df_subset[train_df_subset[max_gain_attr] == attr_val]
            train_labels_sub = train_labels_subset[train_df_subset[max_gain_attr] == attr_val]

            new_child = self.decision_tree_recur(train_df_sub, train_labels_sub, attr_list_sub, non_leaf_node, attr_val, depth_ctr+1)
            non_leaf_node.children.append(new_child)
            
        return non_leaf_node
        
    # method to predict the output for test data using trained ID3 tree
    def predict_test_labels(self, test_df_features):
        
        test_labels_pred = pd.Series(index = test_df_features.index, dtype = object)
        for index, row in test_df_features.iterrows():
            node = self.root
            while node.is_leaf_node() == False:
                found = False
                for child in node.children:
                    if child.parent_edge_label == row[node.attr_name]:
                        node = child
                        found = True
                        break

                if found == False:
                    break
            test_labels_pred[index] = node.outcome_label

        return test_labels_pred

    # method to calculate the accuracy of prediction
    def calc_test_accuracy(self, test_df_features, test_df_labels):

        test_labels_pred = self.predict_test_labels(test_df_features)
        accuracy = (test_labels_pred == test_df_labels).sum()/test_df_labels.shape[0]
        return accuracy

    # method to print the trained ID3 model hierarchically to a text file
    def print_ID3_tree_model(self, output_file):

        print('Printing the decision tree model:', file= output_file)
        print('---------------------------------', file= output_file)
        depth = 0
        self.print_subtree_recur(self.root, depth, output_file)

        return

    # method to recursively print the decision tree
    def print_subtree_recur(self, node, depth, output_file):
        
        if (node.is_leaf_node()):
            print('LEAF NODE: ', file= output_file)
            if (node.depth  == 0):
                print('ROOT NODE', file= output_file)
            else:
                print('PARENT ATTRIBUTE: ' + str(node.parent.attr_name), file= output_file)
                print('PARENT ATTRIBUTE VALUE: ' + str(node.parent_edge_label), file= output_file)
            
            print('DEPTH OF THE NODE: ' + str(node.depth), file= output_file)
            print('OUTCOME CATEGORY: ' + str(node.outcome_label), file= output_file)
            print('---------------------------------', file= output_file)
            return

        print('NONE LEAF NODE: ', file= output_file)
        if (node.depth  == 0):
            print('ROOT NODE', file= output_file)
        else:
            print('PARENT ATTRIBUTE: ' + str(node.parent.attr_name), file= output_file)
            print('PARENT ATTRIBUTE VALUE: ' + str(node.parent_edge_label), file= output_file)
        print('DEPTH OF THE NODE: ' + str(node.depth), file= output_file)
        print('NODE ATTRIBUTE : ' + str(node.attr_name),  file= output_file)
        print('NUM OF NODES IN ITS SUBTREE: ' + str(node.subtree_node_count()), file= output_file)
        print('---------------------------------', file= output_file)

        for child in node.children:
            self.print_subtree_recur(child, depth+1, output_file)

        return

    # method to count the no of nodes in the tree
    def ID3_node_count(self):
        if self.root.is_leaf_node():
            return 1
        else:
            count = 1
            for child in self.root.children:
                count += child.subtree_node_count()
            return count

    # method to calculate the depth of the tree 
    def calc_tree_depth(self):
        if self.root.is_leaf_node():
            self.depth = 0
            return 0
        else:
            depth = 1
            for child in self.root.children:
                depth = max(depth, child.subtree_depth() + 1)
            self.depth = depth
            return depth
    
    # method for pruning the trained tree to prevent overfitting using REDUCED ERROR PRUNING
    def reduced_error_pruning(self, val_df_features, val_df_labels, output_file):

        # Validation accuracy before pruning
        val_accuracy_before_pruning = self.calc_test_accuracy(val_df_features, val_df_labels)
        print('\nVAL ACCURACY (before pruning): ' + str(val_accuracy_before_pruning), file= output_file)

        # Pruning
        self.prune_subtree_recur(self.root, val_df_features, val_df_labels, val_df_features, val_df_labels)

        # Validation accuracy after pruning
        val_accuracy_after_pruning = self.calc_test_accuracy(val_df_features, val_df_labels)
       
        print('\nVAL ACCURACY (after pruning): ' + str(val_accuracy_after_pruning) + '\n', file= output_file)
        
        return

    # recursive pruning of the nodes
    def prune_subtree_recur(self, node, val_df_features, val_df_labels, val_df_stat, val_df_stat_labels):
            
        if node.leaf_flag == True:
            return

        # Prune children if it not a leaf node
        for child in node.children:

            val_df_sub = val_df_features[val_df_features[node.attr_name] == child.parent_edge_label]
            val_labels_sub = val_df_labels[val_df_features[node.attr_name] == child.parent_edge_label]

            if len(val_df_sub) == 0:
                continue
            self.prune_subtree_recur(child, val_df_sub, val_labels_sub, val_df_stat, val_df_stat_labels)

        val_accuracy_before_pruning = self.calc_test_accuracy(val_df_stat, val_df_stat_labels)

        # Prune current node if it is not a leaf node
        node.leaf_flag = True
        node.outcome_label = val_df_labels.value_counts().idxmax()
        val_accuracy_after_pruning = self.calc_test_accuracy(val_df_stat, val_df_stat_labels)

        # If accuracy decreases, then revert back to the original node
        if (val_accuracy_after_pruning < val_accuracy_before_pruning + 0.0001):
            node.leaf_flag = False
            node.outcome_label = ''
           
        return

    # method to help calculate the variation of accuracy with respect to the depth of the tree
    def calc_depth_vs_accuracy(self, test_df_features, test_df_labels):

        depth_vs_acc = {}
        
        Max_depth = self.depth
        
        for i in range(Max_depth + 1):
            temp_tree = deepcopy(self)
            temp_tree.flag_same_depth(temp_tree.root, i)

            temp_acc = temp_tree.calc_test_accuracy(test_df_features, test_df_labels)
            depth_vs_acc[i] = temp_acc


        return depth_vs_acc

    # method to flag the nodes at a particular depth as leaf nodes to calculate variation of accuracy with depth
    def flag_same_depth(self, node, depth):
        
        if node.leaf_flag == True:
            return

        if node.depth == depth:
            node.leaf_flag = True
            return

        for child in node.children:
            self.flag_same_depth(child, depth)

        return

#######################################   HELPER FUNCTIONS    ##################################################

# Declaration of the HELPER FUNCTIONS:

# function to calculate the entropy of the total dataset
def calc_total_attr_entr(train_labels):
    
    total_attr_entr = 0
    for category in list(train_labels.unique()):
        fraction = train_labels.value_counts()[category]/(train_labels.shape[0])
        total_attr_entr += -fraction*np.log2(fraction)
    return total_attr_entr

# function to calculate the entropy of a subset of data with a particular attr-val for an attribute
def calc_attr_val_entr(train_df, train_labels, attr_name, attr_val):

    attr_val_entr = 0
    train_df_subset = train_df[train_df[attr_name] == attr_val]
    train_labels_subset = train_labels[train_df[attr_name] == attr_val]

    attr_val_entr = calc_total_attr_entr(train_labels_subset)

    return attr_val_entr

# function to calculate the info gain of an attribute
def calc_attr_info_gain(train_df, train_labels, attr_name):

    attr_info_gain = 0
    total_attr_entropy = calc_total_attr_entr(train_labels)
    attr_val_entr_dict = {}
    attr_info_gain += total_attr_entropy

    for attr_val in list(train_df[attr_name].unique()):
        attr_val_entr = calc_attr_val_entr(train_df, train_labels, attr_name, attr_val)
        attr_val_entr_dict[attr_val] = attr_val_entr
        attr_info_gain -= (train_df[attr_name].value_counts()[attr_val]/train_df.shape[0])*attr_val_entr

    return (attr_info_gain, attr_val_entr_dict)

# finding the attribute with max gain among the attributes in attr_list
def find_max_gain_attr(train_df, train_labels, attr_list):

    max_gain_attr = None
    max_info_gain = -1

    for attr_name in list(attr_list):
        if attr_name != 'ID':
            attr_info_gain, attr_entr_dict = calc_attr_info_gain(train_df, train_labels, attr_name)
            if attr_info_gain > max_info_gain:
                max_info_gain = attr_info_gain
                max_gain_attr = attr_name

    return (max_gain_attr, max_info_gain)

# function to split the data into train and test sets
def train_test_split(df, train_frac, random_state):
    train_df = df.sample(frac = train_frac, random_state = random_state)
    test_df = df.drop(train_df.index)

    return train_df, test_df

# funtion to split the dataframe into a dataframe with attribute values and a pandas series with the labels
def df_feature_label_split(df, label_name):
    df_features = df.drop(label_name, axis = 1)
    df_label = df[label_name]

    return df_features, df_label


#######################################     READ DATA     #######################################

# Importing the dataset 
df = pd.read_csv('Dataset_A.csv')

# Missing value stats for the dataset
Missing_vals = df.isnull().sum()
print('Missing values in the dataset: ')
print(Missing_vals)
print("\n")

#########################################    DATA CLEANING (HANDLING NANS)      #############################################

# HANDLING THE MISSING VALUES (NaN values) IN THE DATASET

df_new = df.copy()

# Bucket the Age into 10 bins to make it categorical
df_new['Age'] = pd.cut(df_new['Age'], 10, labels = False)

# Ever_Married attribute is a missing value in 140 train examples. Replace it by the mode
df_new['Ever_Married'] = df['Ever_Married'].fillna(df['Ever_Married'].mode().iloc[0])

# Attribute Graduated is a missing value in 78 train examples. Since, it is a categorical attribute, replace all the NaNs by attribute value with max frequency
df_new['Graduated'] = df['Graduated'].fillna(df['Graduated'].mode().iloc[0])

# Attribute Profession is a missing value in 124 train examples. Since, it is a categorical attribute, replace all the NaNs by attribute value with max frequency
df_new['Profession'] = df['Profession'].fillna(df['Profession'].mode().iloc[0])

# Attribute Work_Experience is missing value in 829 train examples. Replace it by the mode 
# df_new['Work_Experience'] = df['Work_Experience'].fillna(round(df['Work_Experience'].mean()))
df_new['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].mode().iloc[0])

# Replace the NaNs in the Family_Size attribute by the mean
# df_new['Family_Size'] = df['Family_Size'].fillna(round(df['Family_Size'].mean()))
df_new['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].mode().iloc[0])

# Replace Var_1 NaNs by the max_frequency category
df_new['Var_1'] = df['Var_1'].fillna(df['Var_1'].mode().iloc[0])

# Missing value stats for the dataset after handling the missing values (Now no missing values present)
print("No missing values in the dataset after handling the missing values:")
print(df_new.isnull().sum())
print("\n\n")

#######################################   TRAINING AND TESTING   #############################################

# Training the Decision Tree Model on 10 different train-test splits and evaluating the best model on the test set

with open('A1_Q1_Output.txt', "w") as f:

    # dictinary to store the model trained (after doing reduced error pruning) corresponding to each train-test split
    models = {}

    MaxTestAccuracy = -10  
    MaxModel = None    # to store the model with max accuracy
    MaxModelNo = -1    # index to store which of the 10 random splits results in best model


    f.write("Training the Decision Tree Model on 10 different train-test splits and evaluating the best model on the test set\n\n")

    for i in range(10):
        
        f.write("RANDOM SPLIT NO : " + str(i) + "\n")
        print("RANDOM SPLIT NO : " + str(i) + "\n")
        f.write('\n')
        train_df, test_df = train_test_split(df_new, 0.8, i)

        train_df_new, val_df = train_test_split(train_df, 0.8, i)

        f.write("Length of train set: " + str(len(train_df_new)) + "\n")
        f.write("Length of validation set: " + str(len(val_df)) + "\n")
        f.write("Length of test set: " + str(len(test_df)) + "\n")

        # Split the train and test dataframes into dataframes with columns of attribute values and pandas series with the labels(outcomes)
        train_df_features, train_df_label = df_feature_label_split(train_df_new, 'Segmentation')
        test_df_features, test_df_label = df_feature_label_split(test_df, 'Segmentation')
        val_df_features, val_df_label = df_feature_label_split(val_df, 'Segmentation')

        model = DecisionTreeModel()   # declare an object of class DecisionTreeModel()

        # Training the model
        print("Training the  ID3 model on the train dataset")
        model.train_ID3_tree(train_df_features, train_df_label)
        print('')
        print('Training Id3 model completed')
        print('')

        f.write("\n")
        f.write("Number of nodes in the tree before pruning: " + str(model.ID3_node_count()) + "\n")
        f.write("Depth of the tree before pruning: " + str(model.calc_tree_depth()) + "\n")
        
        # Test accuracy on the dataset before pruning
        f.write("\n")
        test_acc_bef_prun = model.calc_test_accuracy(test_df_features, test_df_label)
        f.write("TEST ACCURACY (before pruning): " + str(test_acc_bef_prun) + "\n")

        # Reduced Error Pruning
        print('Starting pruning')
        model.reduced_error_pruning(val_df_features, val_df_label, f)
        print('')
        print('Pruning done')
        print('\n')

        models[i] = model

        f.write("\n")
        f.write("Number of nodes in the tree after pruning: " + str(model.ID3_node_count()) + "\n")
        f.write("Depth of the tree after pruning: " + str(model.calc_tree_depth()) + "\n")

        # Test accuracy on the dataset after pruning
        f.write("\n")
        test_acc_aft_prun = model.calc_test_accuracy(test_df_features, test_df_label)

        if (test_acc_aft_prun > MaxTestAccuracy):
            MaxTestAccuracy = test_acc_aft_prun
            MaxModel = model
            MaxModelNo = i

        f.write('(TEST ACCURACY) after pruning: ' + str(test_acc_aft_prun) + "\n")
        f.write("\n")

    
    # writing the parameters of the best model to the output file
    f.write("------------------------------------------------------------\n")
    f.write("TEST ACCURACY OF BEST MODEL: " + str(MaxTestAccuracy) + "\n")
    f.write("NUMBER OF NODES IN THE BEST MODEL (AFTER PRUNING): " + str(MaxModel.ID3_node_count()) + "\n")  
    f.write("DEPTH OF THE BEST MODEL (AFTER PRUNING): " + str(MaxModel.calc_tree_depth()) + "\n")

f.close()

###################################    PRINT BEST MODEL HIEARARCHY     ########################################

# Printing the Best Model Tree (after pruning) in hierarchical format to a text file
f1 = open('A1_Q1_BestModel.txt', 'w')

print("\nPrinting the Best Model Tree (after pruning) in hierarchical format to an output file:\n")
f1.write("\nPrinting the Best Model Tree (after pruning) in hierarchical format:\n\n")

MaxModel.print_ID3_tree_model(f1)
f1.close()

###################################    ACCURACY VS DEPTH PLOTS     ########################################

# Plotting the depth vs accuracy plot for the best model after pruning

depth_based_acc = MaxModel.calc_depth_vs_accuracy(test_df_features, test_df_label)
plt.plot(depth_based_acc.keys(), [val*100 for val in depth_based_acc.values()])
plt.xlabel("Depth of the tree")
plt.ylabel("Test Set Accuracy (in %) ")
plt.title("Depth vs Accuracy plot for ID3 Decision Tree")
plt.show()


#######################################    END OF CODE   ###############################################