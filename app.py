import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Preprocessing the dataset
# Selecting necessary columns
data = data[['text', 'target']]

# Handle missing values
data = data.dropna(subset=['text', 'target'])  # Remove rows with missing values

# Ensure target contains only valid numeric values
data = data[data['target'].str.strip().isin(['0', '1'])]  # Keep rows with '0' or '1' as target
data['target'] = data['target'].astype(int)  # Convert target to integer

# Splitting the dataset into training and testing sets
X = data['text']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluating the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to classify new email content
def classify_email(email_content):
    email_vec = vectorizer.transform([email_content])
    prediction = model.predict(email_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

print("Hello! You've entered the Email Classification System!")
while True:
    user_input = input("\nPlease type the content of the email (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        print("Closing the Email Classification System. Have a great day!")
        break
    print(f"The email content: '{user_input}' \n \n has been classified as: {classify_email(user_input)}")



