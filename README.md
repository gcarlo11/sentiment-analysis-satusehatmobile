# 🩺 Sentiment Analysis of SATUSEHAT Mobile App Reviews

This project performs **sentiment analysis** on user reviews of the **SATUSEHAT Mobile** application — Indonesia’s national health platform.
The main objective is to classify each review into **positive**, **neutral**, or **negative** sentiment categories using **machine learning** techniques.

Two models are developed and compared:

* **Support Vector Machine (SVM)**
* **Naïve Bayes (MultinomialNB)**

The project also addresses **class imbalance** with the **Synthetic Minority Over-sampling Technique (SMOTE)** and visualizes the most frequent words per sentiment using **WordClouds**.

---

## 📂 Dataset

The dataset was **self-scraped** from the **Google Play Store** and stored as `satusehat_reviews.csv`.
Each entry contains:

* The review text
* A corresponding star rating (1–5)

Sentiment labels are derived as follows:

* **Positive (positif):** 4–5 stars
* **Neutral (netral):** 3 stars
* **Negative (negatif):** 1–2 stars

---

## ⚙️ Methodology

The project follows a **standard machine learning pipeline** for text classification in Indonesian:

### 1. Data Loading

* Load the dataset using **pandas** and inspect for missing or duplicate entries.

### 2. Labeling

* Convert numerical ratings into categorical sentiment labels (`positif`, `netral`, `negatif`).

### 3. Text Preprocessing

Raw review text is cleaned and standardized through:

* **Cleansing:** Remove non-alphanumeric characters and punctuation.
* **Case Folding:** Convert text to lowercase.
* **Tokenization:** Split sentences into individual words.
* **Stopword Filtering:** Remove common Indonesian stopwords via `nltk`.
* **Stemming:** Reduce words to their root form using the **Sastrawi** stemmer.

### 4. Feature Extraction

* Apply **TF–IDF vectorization** to transform preprocessed text into numerical features representing word importance.

### 5. Handling Class Imbalance

* Since positive reviews dominate, apply **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples of minority classes for balanced model training.

### 6. Model Training and Evaluation

Two supervised models are trained and compared:

* **Support Vector Machine (SVM)** with linear kernel
* **Naïve Bayes (MultinomialNB)**

Models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

### 7. Visualization

* **WordClouds** illustrate the most frequent terms per sentiment class, providing qualitative insights into user feedback trends.

---

## 📊 Results

| Model           | Accuracy   | Remarks                                                                                                        |
| :-------------- | :--------- | :------------------------------------------------------------------------------------------------------------- |
| **SVM**         | **0.9417** | Achieved high precision and recall across all classes, performing best on “neutral” and “positive” sentiments. |
| **Naïve Bayes** | **0.8842** | Delivered solid results but slightly lower overall accuracy than SVM.                                          |

### 🔤 WordCloud Insights

* **Positive** reviews include words about *good service, ease of use, and helpful features*.
* **Negative** reviews highlight issues such as *bugs, login problems, and slow performance*.

---

## 🧰 Requirements

Install dependencies before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk Sastrawi wordcloud imbalanced-learn
```

---

## 🚀 How to Run

1. Clone this repository or download the notebook file:

   ```
   sentiment-analysis-satusehatmobile.ipynb
   ```
2. Install the dependencies listed above.
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook sentiment-analysis-satusehatmobile.ipynb
   ```
4. Run all cells sequentially to reproduce data preprocessing, model training, and evaluation.

---

## 🔮 Future Improvements

* Expand the dataset with newer reviews.
* Apply **deep learning** models such as **LSTM** or **IndoBERT** for better semantic understanding.
* Integrate model explainability using **SHAP** or **LIME**.
* Develop a lightweight **dashboard** for real-time sentiment tracking.

---

## 🧑‍💻 Author

**Rasyid**
Computer Science student with an interest in **Data Science, Machine Learning, and Artificial Intelligence**.

