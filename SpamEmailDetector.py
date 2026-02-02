#SPAM MAIL DETECTOR#


import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,f1_score

#DOWNLOADING NLTK RESOURCES
nltk.download('stopwords')

#DATASET:SPAM SMS COLLECTION(spam.csv)
df = pd.read_csv("spam.csv", encoding = "latin-1")

#keeps only required columns
df = df[['v1','v2']]
df.columns = ['label','message']

#text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['message'] = df['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state = 42
)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("===== Model Performance =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred, pos_label='spam'))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Custom Prediction Function
# -------------------------------
def predict_spam(message):
    message = preprocess_text(message)
    message_vector = vectorizer.transform([message])
    return model.predict(message_vector)[0]

# -------------------------------
# Test with Sample Messages
# -------------------------------
print("\n===== Sample Predictions =====")
msg1 = "Congratulations! You have won a free lottery. Call now!"
msg2 = "Hey, are you coming to college today?"

print(f"Message: {msg1}")
print("Prediction:", predict_spam(msg1))

print("\nMessage:", msg2)
print("Prediction:", predict_spam(msg2))