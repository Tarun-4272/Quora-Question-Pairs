# Quora Question Pairs - Duplicate Detection (NLP Project)

This project focuses on identifying whether two Quora questions are duplicates using classical machine learning techniques and custom feature engineering. The solution involves building an end-to-end NLP pipeline, training an XGBoost model, and deploying the application using Streamlit.

---

## 🔍 Problem Statement

Quora aims to reduce duplicate content by identifying semantically similar questions. Given a pair of questions, the task is to predict whether they are semantically equivalent (i.e., duplicates).

---

## ✨ Key Features

- End-to-end pipeline for binary classification of duplicate question pairs.
- Extensive text preprocessing including:
  - Lowercasing, stemming, punctuation and stopword removal, contraction expansion.
- 30+ handcrafted features to measure similarity between question pairs:
  - TF-IDF cosine similarity
  - Fuzzy string matching (token set ratio, token sort ratio)
  - Word overlap ratios
  - Levenshtein distance
  - Length differences

---

## 📂 Dataset

- Source: [Quora Question Pairs Dataset (Kaggle)](https://www.kaggle.com/competitions/quora-question-pairs)
- ~400,000 labeled question pairs with a binary target (`is_duplicate`)

---

## 🔧 Workflow

1. **Data Cleaning & Preprocessing**  
   Normalize text, handle contractions, remove stopwords, punctuation, etc.

2. **Feature Engineering**  
   Create 30+ similarity-based features capturing lexical, semantic, and syntactic patterns.

3. **Model Training**  
   Train an XGBoost classifier using stratified 5-fold cross-validation.

4. **Evaluation**  
   Evaluate using log-loss and recall metrics. Best model achieved:
   - 📉 **Log-Loss**: `0.357`
   - 🎯 **+12% improvement in recall** over baseline models

5. **Deployment**  
   Build a **Streamlit** app to interactively test the model with custom question pairs.

---

## 🚀 Deployment

Launch the Streamlit app locally:

```bash
streamlit run app.py
