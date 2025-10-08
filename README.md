# Sentiment Analysis of SATUSEHAT Mobile App Reviews

#### Overview

This project performs sentiment analysis on user reviews of the SATUSEHAT mobile application. The primary goal is to classify the sentiment of each review as "positive," "neutral," or "negative." This is achieved by building and evaluating machine learning models, specifically Support Vector Machine (SVM) and Naive Bayes. The project also addresses class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE) and visualizes the most frequent words for each sentiment class using a WordCloud.

#### Dataset

The dataset used for this analysis was self-scraped from the Google Play Store and is stored in a CSV file named `satusehat_reviews.csv`. It contains user reviews, each with a corresponding star rating.

#### Methodology

The project follows a standard machine learning pipeline for text classification:

1.  **Data Loading**: The `satusehat_reviews.csv` file is loaded into a pandas DataFrame.
2.  **Labeling**: Review scores are converted into sentiment labels:
    * `'positif'` for scores of 4 and 5
    * `'netral'` for a score of 3
    * `'negatif'` for scores of 1 and 2
3.  **Text Preprocessing**: The raw text data is cleaned and prepared for the models through a series of steps:
    * **Cleansing**: Non-alphanumeric characters are removed.
    * **Case Folding**: All text is converted to lowercase.
    * **Tokenization**: Text is split into individual words (tokens).
    * **Stopword Filtering**: Common Indonesian stopwords are removed using the `nltk` library.
    * **Stemming**: Words are reduced to their root form using the `Sastrawi` library.
4.  **Feature Extraction**: The preprocessed text is converted into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This technique represents the importance of a word in a document relative to a collection of documents.
5.  **Handling Class Imbalance**: The dataset is likely to have an unequal distribution of sentiment classes (e.g., more positive reviews than negative ones). The SMOTE (Synthetic Minority Over-sampling Technique) algorithm is used to over-sample the minority classes, creating a balanced dataset for training the models.
6.  **Model Training and Evaluation**:
    * The dataset is split into training and testing sets.
    * **SVM (Support Vector Machine)**: A linear kernel SVM model is trained on the balanced data.
    * **Naive Bayes (MultinomialNB)**: A Naive Bayes model is also trained for comparison.
    * Both models are evaluated using common metrics such as **accuracy score**, a **classification report** (precision, recall, f1-score), and a **confusion matrix**.
7.  **Visualization**: Word clouds are generated for each sentiment class (`'positif'`, `'netral'`, and `'negatif'`) to visually identify the most frequently used words in each category.

#### Results

The models' performance is detailed in the output of the notebook, including:

* **SVM Performance**:
    * Accuracy: 0.9417
    * A classification report showing high precision, recall, and f1-scores across all three classes. The model performs particularly well on the "neutral" and "positive" classes.
* **Naive Bayes Performance**:
    * Accuracy: 0.8842
    * The classification report shows good performance, though slightly less accurate than the SVM model.

#### Word Clouds

The word clouds illustrate the keywords associated with each sentiment, providing insights into user feedback. For example:
* The "positive" word cloud likely features words related to good service, ease of use, and helpful features.
* The "negative" word cloud might show words related to bugs, slow performance, difficulty of use, or login issues.

This project demonstrates a complete process for text-based sentiment analysis, from data preparation to model evaluation and insightful visualization.
