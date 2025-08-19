# SMS Spam Detection Model 🤖📱

A machine learning model that accurately classifies SMS messages as **spam** or **ham** (not spam) with over 98% accuracy. Built with Python and scikit-learn.


## Features

- **High Accuracy**: Achieves 98.5% accuracy on the test set
- **Real-time Prediction**: Classify new messages with confidence scores
- **Detailed Metrics**: Provides full classification report and confusion matrix
- **Automated Data Handling**: Automatically downloads and preprocesses the benchmark dataset

## How It Works

1. **Data Acquisition**: Downloads the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from UCI Machine Learning Repository
2. **Text Preprocessing**: Uses CountVectorizer to convert messages to numerical features
3. **Model Training**: Employs a Multinomial Naive Bayes classifier optimized for text data
4. **Evaluation**: Comprehensive performance analysis with accuracy, precision, recall, and F1-score metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/spam-detector.git
cd spam-detector

# Install dependencies:
pip install -r requirements.txt

# Usage
python spam_detector.py
