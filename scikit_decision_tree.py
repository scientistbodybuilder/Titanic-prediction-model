from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd

data = pd.read_csv('Titanic-Dataset.csv')

# Data Cleaning
df = data.drop(['Name','Ticket','PassengerId','Fare','Cabin','Embarked'], axis='columns')
df = df.fillna({'Age': df['Age'].median()})

# Encode categorical variable
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Family'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp','Parch'], axis='columns')

# Separate features and target
X = df.drop('Survived', axis=1)  # Features only
y = df['Survived']               # Target only

# Standard split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

clf = DecisionTreeClassifier(random_state=46)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")