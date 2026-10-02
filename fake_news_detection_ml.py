import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Remove empty rows
data = data.dropna()

# Features and labels
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train ML model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Fake News Detection Model")
print("--------------------------")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Try your own news
while True:
    news = input("\nEnter news text (or type 'exit' to stop): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)[0]

    print("Prediction:", prediction)