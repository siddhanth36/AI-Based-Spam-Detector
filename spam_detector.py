# spam_detector.py
"""
SMS Spam Detection Model
Train a Naive Bayes classifier to identify spam messages.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download and load the dataset
def load_data():
    """Loads the SMS Spam Collection dataset from the UCI repository."""
    print("Loading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    file_name = "SMSSpamCollection"
    df = pd.read_csv(url, sep='\t', names=['label', 'message'], compression='zip')
    return df

# Explore the data
def explore_data(df):
    """Prints basic information about the dataset."""
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 entries:")
    print(df.head())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPercentage of spam: {(df['label'].value_counts()['spam'] / len(df)) * 100:.2f}%")

# Preprocess and train model
def train_model(df):
    """Preprocesses data, trains the model, and evaluates performance."""
    # Convert labels to numerical values
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # Define features and target
    X = df['message']
    y = df['label_num']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} test samples.")

    # Text vectorization - Convert words to numbers
    vectorizer = CountVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} words.")

    # Train Naive Bayes classifier
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_features, y_train)

    return model, vectorizer, X_test_features, y_test, X_test

# Evaluate the model
def evaluate_model(model, X_test_features, y_test):
    """Evaluates the model and prints performance metrics."""
    # Make predictions
    y_pred = model.predict(X_test_features)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")

    # Detailed performance report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('assets/confusion_matrix.png', bbox_inches='tight', dpi=100)
    plt.close()
    print("Confusion matrix saved to 'assets/confusion_matrix.png'")

# Predict new messages
def predict_message(model, vectorizer, message):
    """Classifies a new message as ham or spam."""
    features = vectorizer.transform([message])
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 0:
        result = "HAM ✅"
        confidence = probability[0][0]
    else:
        result = "SPAM 🚨"
        confidence = probability[0][1]

    print(f"\nMessage: '{message}'")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")

def main():
    """Main function to run the spam detection pipeline."""
    print("=" * 50)
    print("SMS Spam Detector Training Pipeline")
    print("=" * 50)

    # Load and explore data
    df = load_data()
    explore_data(df)

    # Train and evaluate model
    model, vectorizer, X_test_features, y_test, X_test = train_model(df)
    evaluate_model(model, X_test_features, y_test)

    # Test with sample messages
    print("\n" + "=" * 50)
    print("Testing with Sample Messages")
    print("=" * 50)

    test_messages = [
        "Hey, are we still meeting for lunch today?",
        "Congratulations! You've won a free $1000 Walmart gift card! Click here to claim now!",
        "Your package has been shipped. Track it here: https://example.com/track/123",
        "Nah I don't think he goes to usf, he lives around here though",
        "URGENT: Your bank account has been suspended. Please verify your details immediately: http://fake-bank-login.com"
    ]

    for message in test_messages:
        predict_message(model, vectorizer, message)
        print("-" * 40)

if __name__ == "__main__":
    main()
