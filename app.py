import streamlit as st
import pickle
import re

# Load the model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define stopwords and clean_text function
stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
             'for', 'and', 'or', 'of', 'with', 'by', 'as', 'from', 'that', 'this', 'be', 'it'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 2]
    return " ".join(words)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline and click Predict to see if it's REAL or FAKE.")

user_input = st.text_area("Paste news text here:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("✅ This news is REAL.")
    else:
        st.error("🚫 This news is FAKE.")
