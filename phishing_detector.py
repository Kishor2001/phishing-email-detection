from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

emails = [
    "Win money now click here",
    "Claim your free prize",
    "Your account is suspended verify immediately",
    "Meeting scheduled tomorrow",
    "Project report attached",
    "Team meeting at 10 AM",
    "Congratulations you won lottery",
    "Update your bank details now",
    "Monthly project review",
    "Please find attached document"
]

labels = [
    "Phishing",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Safe",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

test_email = ["Click here to claim your free reward"]
test_vector = vectorizer.transform(test_email)

result = model.predict(test_vector)

print("\nTest Email Prediction:", result[0])