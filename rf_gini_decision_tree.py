import pandas as pd
import numpy as np

class Node():
    def __init__(self,feature=None,feature_value=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.feature_value = feature_value
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree():
    def __init__(self,max_depth=10,min_samples_split=5,n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        #
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        #
        self.root = self.grow_tree(X,y, current_gini=float('inf'))

    def get_leaf_value(self, y):
        #
        # print("calculating leaf value")
        # print(f"Y: {y}")
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]
        # np.argmax receives a list of counts, and returns the index of the maximum count

    def grow_tree(self, X, y, current_gini, depth=0):
        # print(f"X shape before best split: {X.shape}")
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.get_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False) #random index of a feature in X we will find best split for
        # find the best split
        cpX = X.copy()
        cpY = y.copy()
        best_feature_indx, best_threshold, best_gini = self.best_split(cpX, cpY, feat_idxs, current_gini)
        # print(f"X shape after best split: {X.shape}")
        if best_feature_indx is None:
            # print("No valid split found")
            return Node(value=self.get_leaf_value(y))
        # create the child nodes
        if best_threshold is None:
            # print(f'splitting for feature {X.columns[best_feature_indx]}')
            original_unique_values = X[X.columns[best_feature_indx]].unique()
            split_value = original_unique_values[0]
            # print(f"split value: {split_value}")

            left_mask = X[X.columns[best_feature_indx]] == split_value
            right_mask = X[X.columns[best_feature_indx]] != split_value
        else:
            # print(f'splitting for feature {X.columns[best_feature_indx]} at threshold {best_threshold}')
            left_mask = X[X.columns[best_feature_indx]] <= best_threshold
            right_mask = X[X.columns[best_feature_indx]] > best_threshold

        # print(f"left mask: {left_mask}")
        # print(f"right mask: {right_mask}")
        left_X = X[left_mask]
        right_X = X[right_mask]
        left_y = y[left_mask]
        right_y = y[right_mask]

        left_child = self.grow_tree(left_X, left_y, current_gini=best_gini, depth=depth + 1)
        right_child = self.grow_tree(right_X, right_y, current_gini=best_gini, depth=depth + 1)
        return Node(feature=X.columns[best_feature_indx], feature_value=split_value if best_threshold is None else None, threshold=best_threshold, left=left_child, right=right_child)

    def determine_gini_purity(self,X,feat_indx,y):
        #consider case when the feature is categorical, or when the feature is numerical
        feature_data = X.iloc[:, feat_indx]
        if feature_data.dtype == 'object' or isinstance(feature_data.iloc[0], str):
            return self.gini_categorical(X, feat_indx, y)
        else:
            return self.gini_numeric(X, feat_indx, y)

    def gini_categorical(self,X,feat_indx,y):
        class_labels = X[X.columns[feat_indx]].unique()
        predict_values = y.unique()
        # print(f'predict-values: {predict_values}')
        # print(f'classifications for feature {X.columns[feat_indx]}: {class_labels}')
        
        total_gini_impurity = 0
        for label in class_labels:
            # print(f'calculating gini impurity for feature {X.columns[feat_indx]} at index {feat_indx} with label {label}')
            feature_values = X[X.columns[feat_indx]]
            mask = feature_values == label
            y_masked = y[mask]
            gini = 1 - (y_masked[y_masked == predict_values[0]].shape[0]/y_masked .shape[0])**2 - (y_masked[y_masked == predict_values[1]].shape[0]/y_masked.shape[0])**2
            total_gini_impurity += gini * y_masked.shape[0]/y.shape[0]
        # print(f'total gini impurity for feature {X.columns[feat_indx]}: {total_gini_impurity}')

        return (total_gini_impurity, None)
    
    def gini_numeric(self,X,feat_indx,y):
        numeric_values = sorted(X[X.columns[feat_indx]].unique())
        predict_values = y.unique()

        if len(numeric_values)==2:
            threshold = np.mean(numeric_values)

            split_mask_1 = X[X.columns[feat_indx]] <= threshold
            split_mask_2 = X[X.columns[feat_indx]] > threshold
            y_split_1 = y[split_mask_1]
            y_split_2 = y[split_mask_2]

            gini1 = 1 - (y_split_1[y_split_1 == predict_values[0]].shape[0] / y_split_1.shape[0])**2 - (y_split_1[y_split_1 == predict_values[1]].shape[0] / y_split_1.shape[0])**2
            gini2 = 1 - (y_split_2[y_split_2 == predict_values[0]].shape[0] / y_split_2.shape[0])**2 - (y_split_2[y_split_2 == predict_values[1]].shape[0] / y_split_2.shape[0])**2

            weighted_gini = (y_split_1.shape[0] * gini1 + y_split_2.shape[0] * gini2) / y.shape[0]
            return (weighted_gini, threshold)
            

        min_gini = float('inf')
        min_gini_threshold = None
        for i in range(len(numeric_values)-2):
            threshold = (numeric_values[i] + numeric_values[i+1]) /2
            # split the data by this threshold
            split_mask_1 = X[X.columns[feat_indx]] <= threshold
            split_mask_2 = X[X.columns[feat_indx]] > threshold
            y_split_1 = y[split_mask_1]
            y_split_2 = y[split_mask_2]

            gini1 = 1 - (y_split_1[y_split_1 == predict_values[0]].shape[0] / y_split_1.shape[0])**2 - (y_split_1[y_split_1 == predict_values[1]].shape[0] / y_split_1.shape[0])**2
            gini2 = 1 - (y_split_2[y_split_2 == predict_values[0]].shape[0] / y_split_2.shape[0])**2 - (y_split_2[y_split_2 == predict_values[1]].shape[0] / y_split_2.shape[0])**2

            weighted_gini = (y_split_1.shape[0] * gini1 + y_split_2.shape[0] * gini2) / y.shape[0]
            if weighted_gini < min_gini:
                min_gini = weighted_gini
                min_gini_threshold = threshold

        return (min_gini, min_gini_threshold)

    def best_split(self, X, y, feat_indxs, old_gini):
        split_feature_indx, split_threshold = None, None
    
        min_gini = old_gini
        for feat_indx in feat_indxs:
            if len(X[X.columns[feat_indx]].unique()) <= 1:
                # print(f"Skipping feature {X.columns[feat_indx]} - only one unique value")
                continue

            gini, threshold = self.determine_gini_purity(X, feat_indx, y)
            # print(f'gini impurity for feature {feat_indx}: {gini}')
            if gini < min_gini and old_gini - gini > 0.01:
                min_gini = gini
                split_feature_indx = feat_indx
                split_threshold = threshold

        return (split_feature_indx, split_threshold, min_gini)
    

    def predict(self,X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T  
        # predictions = []
        for index, row in X.iterrows():
            prediction = self.traverse_tree(self.root, row)
        return prediction

    def traverse_tree(self, node, X):
        if node.is_leaf_node():
            # print("reached leafnode")
            return node.value
        
    
        if node.threshold is None: # this means the feature of the current node is categorical
            # print(f"reached categorical variable: {node.feature}")
            # print(f"Sample feature value: {X[node.feature]}")
            if X[node.feature] == node.feature_value:
                return self.traverse_tree(node.left, X)
            else:
                return self.traverse_tree(node.right, X)

        else: # this means the feature of the current node is numerical
            # print(f"reached numerical variable: {node.feature}")
            # print(f"Sample feature value: {X[node.feature]}")
            if X[node.feature] <= node.threshold:
                return self.traverse_tree(node.left, X)
            else:
                return self.traverse_tree(node.right, X)



