import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data = pd.read_csv('titanic.csv')
print(data.columns.tolist())

# Display basic information about the dataset
print(data.info())
print("\nFirst few rows:\n", data.head())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Basic statistical summary
print(data.describe())

# Survival rate by gender
sns.barplot(x='Sex_male', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.show()

# Age distribution
plt.figure(figsize=(10, 5))
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Create a new feature 'FamilySize'
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Feature 'IsAlone'
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

# Select features for the model
features = ['Pclass', 'Age', 'Fare', 'Sex_male', 'IsAlone']
X = data[features]
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
predictions = rf_model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Feature importance
feature_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importances")
plt.show()

# Survival correlation heatmap
correlation = data[features + ['Survived']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features with Survival')
plt.show()

