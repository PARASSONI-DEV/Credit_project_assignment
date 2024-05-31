import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# Load the dataset
url = "https://github.com/YBIFoundation/Dataset/raw/main/Credit%20Default.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()

# Display basic information about the dataset
data.info()

# Check for missing values
data.isnull().sum()
from sklearn.model_selection import train_test_split

# Define the target variable and features
X = data.drop(columns=['Default'])
y = data['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Load the dataset
url = "https://github.com/YBIFoundation/Dataset/raw/main/Credit%20Default.csv"
data = pd.read_csv(url)

# Display basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Define the target variable and features
X = data.drop(columns=['Default'])
y = data['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
