# 📰 Fake News Detection using Python & Machine Learning

A machine learning project that detects whether a news article is *REAL or FAKE* using TF-IDF vectorization and a Naive Bayes classifier. The model is deployed with a Streamlit web app for real-time predictions.

---

## ✅ Features

- Preprocessed and cleaned a real-world dataset of *44,000+ news articles*
- Used *TF-IDF vectorization* for feature extraction
- Trained a *Multinomial Naive Bayes* classifier with over *99% accuracy*
- Built a fully functional *Streamlit web app* for real-time user predictions
- Evaluated model using accuracy, precision, recall, and F1-score

---

## 🛠 Tech Stack

- *Languages:* Python
- *Libraries:* pandas, scikit-learn, Streamlit, NumPy, re
- *ML Techniques:* TF-IDF, Naive Bayes
- *Tools:* Jupyter Notebook, Google Drive (for large file hosting)

---

## 🧠 How It Works

1. Load and clean text data from real and fake news datasets
2. Convert text to numerical features using *TF-IDF*
3. Train a Naive Bayes classifier
4. Save model and vectorizer to .pkl files
5. Use *Streamlit* to build a web app for predicting user input in real time

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/fake-news-detection

# 2. Navigate into the folder
cd fake-news-detection

# 3. Install requirements (optional)
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
