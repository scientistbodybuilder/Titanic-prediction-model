import pandas as pd
import numpy as np
import openpyxl
import gini_decision_tree


data = pd.read_csv('Titanic-Dataset.csv')
# print(data.head())

# Data Cleaning
df  = data.drop(['Name','Ticket','Cabin','Embarked'],axis='columns')
df = df.fillna({'Age': df['Age'].median()})

df['Family'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp','Parch'],axis='columns')
df = df[['Age', 'Sex', 'Family', 'Pclass', 'PassengerId', 'Survived']]
df.reset_index(inplace=True, drop=True)

train_data = df.sample(frac=0.8, random_state=46)
val_data = df.drop(train_data.index)
# val_data.to_excel('DT Validation_data.xlsx', index=False)


X = train_data[train_data.columns[:-2]] #avoid the passengerId
y = train_data[train_data.columns[-1]]
# print(f"X: {X}")
# print(f"y: {y}")

dt = gini_decision_tree.DecisionTree(max_depth=100)
dt.fit(X, y)


predict_results = dt.predict(val_data[val_data.columns[:-1]])

result_df = pd.DataFrame({'id':[], 'prediction':[]})
for x in predict_results:
    result_df = pd.concat([result_df,pd.DataFrame({'id': [x['id']], 'prediction': [x['prediction']]})], axis=0)

# result_df.to_excel('DT Predictions.xlsx', index=False)

total = val_data.shape[0]

correct = 0
actual_survived = val_data.sort_values(by='PassengerId')['Survived'].to_list()
predicted_survived = result_df.sort_values(by='id')['prediction'].to_list()
for i in range(total):
    if actual_survived[i] ==  predicted_survived[i]:
        correct +=1

accuracy = float(correct / total)
print(f"accuracy: {accuracy}")