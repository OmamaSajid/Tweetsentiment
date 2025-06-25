# 🧠 TweetSensei - Twitter Sentiment Classifier

**TweetSensei** is a machine learning project that classifies the sentiment of tweets as **Positive (1)** or **Negative (0)** using the **Sentiment140** dataset. It uses natural language processing (NLP) and logistic regression to train a model that can analyze tweet tone and predict sentiment in real time.

---

## 🔍 Project Overview

This project:
- Loads and preprocesses tweets using NLTK.
- Converts text into TF-IDF vectors.
- Trains a logistic regression model for binary sentiment classification.
- Predicts sentiment of new, user-written tweets directly from code input.

---

## 📊 Dataset: Sentiment140

- **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Original Labels**:
  - `0` → Negative
  - `4` → Positive

> 🛠 **Modification**: For simplicity and binary classification, this project maps label `4 → 1`, so the sentiment labels are now:
>
> - `0` → Negative  
> - `1` → Positive

- **Fields Used**:
  - `target`: sentiment label (0 or 1)
  - `text`: the tweet content

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- NLTK (Stopwords, Stemming)
- scikit-learn (TF-IDF, Logistic Regression)
- Regex (`re`)

---
---

## 🛠️ How to Use This Project

You have **two options** to run TweetSensei:

### ✅ 1. Use Pre-trained Model (Quick Start)


- Upload these files to your Colab session:
  - `trained_model.sav`
 

- Then, load them like this:
  ```python
  import pickle

  model = pickle.load(open('trained_model.sav', 'rb'))
If you want to train the model from the beginning:

Open the notebook tweetsensei.ipynb in Google Colab.

Download the dataset from Kaggle:

kaggle datasets download -d kazanova/sentiment140
unzip sentiment140.zip

And enjoyyy


Thanks for checking out TweetSensei!
🐦 Happy Tweet Sniffing — and see you next project!
Bye Bye! 👋


