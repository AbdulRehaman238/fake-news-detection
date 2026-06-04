import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Split data
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Accuracy:", round(score * 100, 2), "%")

# Test custom news
news = input("Enter news text: ")

news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)

print("Prediction:", prediction[0])
