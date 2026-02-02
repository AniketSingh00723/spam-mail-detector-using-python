# Spam Email Detector using Python (Bulletproof Version)

import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords (first time only)
nltk.download('stopwords')

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Remove completely empty columns (very important)
df = df.dropna(axis=1, how='all')

# Normalize column names
df.columns = df.columns.str.lower()

print("Detected columns:", df.columns.tolist())

# -----------------------------
# 2. Detect text & label safely
# -----------------------------
if len(df.columns) == 1:
    # Only text present → create dummy labels (for safety)
    text_col = df.columns[0]
    df['label'] = 'ham'
    label_col = 'label'
else:
    label_col = df.columns[0]
    text_col = df.columns[1]

X = df[text_col].astype(str)
y = df[label_col].astype(str)

# -----------------------------
# 3. Text Preprocessing
# -----------------------------
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

X = X.apply(clean_text)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6. Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# 8. Prediction Function
# -----------------------------
def predict_spam(message):
    message = clean_text(message)
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]

# -----------------------------
# 9. Test Messages
# -----------------------------
msg1 = "Congratulations! You won a free prize"
msg2 = "Are we attending lecture tomorrow?"

print("\nMessage:", msg1)
print("Prediction:", predict_spam(msg1))

print("\nMessage:", msg2)
print("Prediction:", predict_spam(msg2))
