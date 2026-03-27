# spam-mail-detector-using-python
📧 Spam Mail Detector Using Python

This project is a Spam Email Classifier built using Python and Machine Learning techniques.
It classifies messages/emails as Spam or Ham (Not Spam) based on textual data.

🚀 Features

Text preprocessing (lowercasing, removing stopwords, tokenization)

Converts text into numerical features using TF-IDF

Trains a machine learning model to detect spam

Simple and beginner-friendly implementation

Uses a public spam dataset (spam.csv)

🛠️ Technologies Used

Python 🐍

Pandas

NumPy

Scikit-learn

📂 Project Structure
spam-mail-detector-using-python/
│
├── SpamEmailDetector.py   # Main Python file
├── spam.csv               # Dataset
├── README.md              # Project documentation
└── .gitignore

📊 Dataset

The dataset contains labeled messages:

spam → unwanted messages

ham → legitimate messages

Stored in spam.csv

⚙️ How It Works

Load the dataset

Preprocess text data

Convert text into vectors using TF-IDF

Train the classifier

Predict whether a message is spam or not

🧑‍💻Source code

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['label', 'message']]

# Convert labels to numbers (spam=1, ham=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text into numerical data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom message
def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    result = model.predict(msg_vec)
    return "Spam" if result[0] == 1 else "Not Spam"

# Example
print(predict_spam("Congratulations! You won a free iPhone"))
```



▶️ How to Run

Clone the repository:

git clone https://github.com/AniketSingh00723/spam-mail-detector-using-python.git


Install required libraries:

pip install pandas numpy scikit-learn


Run the program:

python SpamEmailDetector.py

🎯 Output

Displays model accuracy

Predicts whether a given message is Spam or Ham

📌 Future Improvements

Add GUI or web interface

Use advanced NLP techniques

Deploy using Flask or Streamlit

👨‍💻 Author

Aniket Singh
BSc IT Student

