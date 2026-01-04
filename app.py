import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("amazon.csv", engine="python", on_bad_lines="skip")
df = df[['Text', 'Score']]
df.dropna(inplace=True)
df['Text'] = df['Text'].astype(str)

threshold = df['Score'].median()
df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= threshold else 0)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['Text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("Sentiment Analysis App")
st.write(f"Model Accuracy: {acc:.4f}")

user_input = st.text_area(" extracting columns Text and Score from amazon.csv")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        if prediction == 1:
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")
