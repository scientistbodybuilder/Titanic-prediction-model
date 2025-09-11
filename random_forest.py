import rf_gini_decision_tree
import pandas as pd
import numpy as np
import math
from collections import Counter

data = pd.read_csv('Titanic-Dataset.csv')
# print(data.head())

# Data Cleaning
df  = data.drop(['Name','Ticket','Cabin','Embarked'],axis='columns')
df = df.fillna({'Age': df['Age'].median()})

def func(x):
    if x < 8:
        return 0
    elif x >= 8 and x < 15:
        return 1
    elif x >= 15 and x < 31:
        return 2
    else:
        return 3

df['Family'] = df['SibSp'] + df['Parch']
df['Fare Level'] = df.apply(lambda x: func(x['Fare']), axis=1)
df = df.drop(['SibSp','Parch'],axis='columns')
df = df[['Age', 'Sex', 'Family', 'Pclass', 'Fare Level', 'Survived']]
df.reset_index(inplace=True, drop=True)

train_data = df.sample(frac=0.8, random_state=50)
val_data = df.drop(train_data.index)
# val_data.to_excel('rf Validation_data.xlsx', index=False)

# At time to make a split, instead of considering all available features to find the lowest impurity split,
# randomly select a subset of the variables, and select the best split from that.
# The number of variables used per set, should be constant, althought the variables themselves are selected randomly?
#To evalute the randomly forest, we can use out of bag accuracy, by testing it against rows of data that were never included in any bootstrapped sample?
# we a constant number of m predictors, where commonly m is sqrt(p) where p is full number of possible predictors we have

class RandomForest():
    def __init__(self,max_trees,feature_subset_size,df):
        self.max_trees = max_trees
        self.df = df
        self.feature_subset_size = min(feature_subset_size, len(self.df.columns) - 2)
        self.bootstrap_list = []
        self.forest = []
        self.predictions = []

    def grow_forest(self):
        self.forest = []
        for bootstrap_df in self.bootstrap_list:
            X = bootstrap_df[bootstrap_df.columns[:-1]]
            y = bootstrap_df[bootstrap_df.columns[-1]]
            
            dt = rf_gini_decision_tree.DecisionTree(n_features=self.feature_subset_size)
            dt.fit(X,y)

            self.forest.append(dt)


    def initialize_bootstraps(self):
        self.bootstrap_list = []
        for i in range(self.max_trees):
            bstr = self.df.sample(n=len(self.df), replace=True, random_state=i)
            self.bootstrap_list.append(bstr)

    def predict(self,X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T  
        #clear old predictions
        self.predictions = []
        for index, row in X.iterrows():
            predictions = []
            for dt in self.forest:
                pred = dt.predict(row)
                predictions.append(pred)


            final_prediction = Counter(predictions).most_common(1)[0][0]
            self.predictions.append(final_prediction)

        return self.predictions


# TRAIN

n_features = len(train_data.columns) - 1 
optimal_subset = max(1, int(math.sqrt(n_features)))  
rf = RandomForest(150,optimal_subset,train_data)

print("initialize the bootstrap")
rf.initialize_bootstraps()

print("grow the forest")
rf.grow_forest()

# prediction

X = val_data[val_data.columns[:-1]]
print(f"test data columns: {X.columns}")
print(f"We have {len(rf.forest)} trees in the forst")

print("Predicting")
predictions = rf.predict(X)

# EVAL Results

result_col = pd.DataFrame({'Predictions': predictions})
total = val_data.shape[0]

correct = (val_data['Survived'].reset_index(drop=True) == result_col['Predictions'].reset_index(drop=True)).sum()
accuracy = float(correct / total)
print(f"accuracy: {accuracy}")