**TODO**:

- The data section is finished
- next comes text preprocessing

# Introduction 

If you are someone that wants to learn about NLP and Sentiment Analysis, this repository is for you. We will do X,Y,Z in a easy and simple way from beginning to end... {This could be the title instead of "Introduction"}

This repository explores various NLP techniques for sentiment analysis of reviews on the Netflix app in the Google Play Store. It offers a detailed, yet easy to follow guide on sentiment analysis starting from basic Exploratory Data Analysis (EDA) and proceeding to text pre-processing and machine learning model training and evaluation. The pros and cons of different text pre-processing techniques, as well as a performance comparisson between them, are presented. Finally, various machine learning methods are explored, from standard ML algorithms, such as Logistic Regression, to more advanced ones, like RNNs and Transformers.


![Sentiment_Analysis](https://miro.medium.com/v2/1*_JW1JaMpK_fVGld8pd1_JQ.gif)

# Data

## Dataset

The dataset we will use is [`netflix_reviews.csv`](DATASETS/netflix_reviews.ipynb) from [Kaggle](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data). It contains reviews for the Netflix App in the Google Play Store.

The dataset consists of 8 columns:

| Column Name            | Description                                        |
|------------------------|----------------------------------------------------|
| `reviewId`             | A unique identifier for each review.               |
| `userName`             | The name of the user who submitted the review.     |
| *`content`*              | The actual text of the review.                     |
| *`score`*                | The rating given, ranging from 1 to 5.             |
| `thumbsUpCount`        | The number of "thumbs up" the review received.     |
| `reviewCreatedVersion` | The version of the app used when the review was written. |
| `at`                   | The date and time the review was posted.           |
| `appVersion`           | The version of the app used when the review was written. |

For our analysis, we will focus on two aspects of the reviews. The `content` will be the input for our models, and the `score` will be the output.
For our analysis we will focus on two aspects of the reviews. The `content` will be the input to our models and the `score` will be the output.

## Analysis and Visualization

You can get an initial grasp of the dataset here: [`data_visualization.ipynb`](DATA/data_visualization.ipynb). Where we will perform some data analysis and visualization. This step isn't important, but if you want a deeper understanding of the dataset, you can look into it.

## Clean Up

This section outlines the steps to prepare a cleaned dataset. First, we remove unnecessary columns to simplify the dataset. Then, we drop duplicate rows and any rows with missing values. You can view all these steps in: [`data_cleanup.ipynb`](DATA/data_cleanup.ipynb).

Then we save the cleaned dataset as: [`cleaned_data.csv`](DATASETS/cleaned_data.csv).

# Text

At this point, we need to deal with the `content` of the review. This is because it often contains unwanted characters, such as punctuation, emojis, and characters repeated many times. To address this, we first pre-process the text to clean it up. Then, we convert the words into numbers through a process called text vectorization. This is because our models operate with numerical data, not text.

## Text Pre-Processing

The text pre-processing is done in: [`text_preprocessing.ipynb`](TEXT/text_preprocessing.ipynb).

The pre-processing goes as follows:
- We load the cleaned dataset: [`cleaned_data.csv`](DATA/cleaned_data.csv).
- We clean up the text using several techniques (turn the text into lowercase, replace emojis, remove repeated characters) to be ready for vectorization.

The pre-processed dataset is saved as: [`preprocessed_text.csv`](DATASETS/preprocessed_text.csv).

## Text Vectorization

At this point we are ready for turning our words into numbers. There are many ways to do so. The one we explored in this repository are:
- BoW (Bag of Words): [`bagofwords.ipynb`](TEXT/bagofwords.ipynb)
- TF-IDF (Term Frequency - Inverse Document Frequency): [`tfidf.ipynb`](TEXT/tfidf.ipynb)
- Word2Vec (Word 2 Vector)
- GloVe (Global Vectors)

We will give you some general guidelines for which one to pick, and the pros and cons of each one.

# Picking a Model

At this point we have different ways to deal with the text vectorization and many different models we can pick. So we have a lot of possible combinations. I guess we can just show some of them.

TODO

# Outline

- Why is this important?
  	- Why people should read this thing.
- What kind of data did we use?
	- Where did we get it from?
	- How did we analyze and visualize the data?
- How did we deal with the Text?
	- How did we pre-process the text? (Clean up)
	- How did we turn words into numbers?
		- Bag-of-Words
		- TF-IDF
		- Word2Vec
		- GloVe
- What kind of models did we use?
	- Naive Bayes
	- SVM
	- LogisticRegression
	- RandomForest
	- XGBoost
	- LSTM/GRU/MultiHead Attention custom model
	- Transformers (?)
	- BERT/ ROBERTA
- How did we evaluate our model?
- How can we put our best model into production?
	- Like having a website. With a text box. And you get as an output the rating of the review. From 1 to 5 stars.