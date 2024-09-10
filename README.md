**TODO**:

There are 3 ways we can treat this problem:
1. Treat it as a regression problem. So take the ratings, and use minmax normalization to narmalize them. So 5 becomes 1. And 1 becomes 0. Then for the models it becomes a simple regression problem.
2. Treat it as a classification problem. We can do as we did: 1-2 negative, 3 neutral, 4-5 positive. (We could even test using 5 different classes for each star). Then for the models it becomes a simple classification problem.
3. Use "ordinal classification". This makes sense. But it's not easy to implement. Like there is no straight forward way to approach this.

**Some stuff to consider**: If our models classify something as 5 stars (positive) but in reality it was 1 star (or negative), that's a big error. But this error doesn't show up in a standard "classification" model. Because each class is treated **indipendently**. On the other hand, in a regression model, if the predicted label was 5 and the true label was 1, the loss function (like RMSE) would be simply, 5-1 = 4. So I guess with a classificaiton loss function we just know if we are "right/wrong" but with a regression loss function we know how much "right/wrong" we are.

Something really interesting to read: [StackExchange](https://stats.stackexchange.com/questions/222073/classification-with-ordered-classes)

(Regression seems to be the most reasonable approach).

# Mastering Sentiment-Analysis: A Step-by-Step Guide 

Ready to learn about Natural Language Processing (NLP) and Sentiment Analysis? This repository is perfect for you. We'll walk you through the whole process, from understanding your data to setting up a sentiment analysis model, in a simple and straightforward way.

We'll start with some basic data exploration using user reviews of the Netflix app from the Google Play Store. Then, we'll move on to text pre-processing. You'll learn how to clean and prepare text data, explore different text vectorization methods, and compare various machine learning models—from logistic regression to advanced transformers.

By the end, you'll not only understand sentiment analysis but also know how to build and deploy your own models. Whether you're new to NLP or looking to brush up on your skills, this guide will help you master sentiment analysis with ease.

![Sentiment_Analysis](https://miro.medium.com/v2/1*_JW1JaMpK_fVGld8pd1_JQ.gif)

# Data

## Dataset

The dataset we will use is [`netflix_reviews.csv`](DATASETS/netflix_reviews.ipynb) from [Kaggle](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data). It contains reviews for the Netflix App in the Google Play Store.

The dataset consists of 8 columns:

| Column Name            | Description                                        |
|------------------------|----------------------------------------------------|
| `reviewId`             | A unique identifier for each review.               |
| `userName`             | The name of the user who submitted the review.     |
| **`content`**              | The actual text of the review.                     |
| **`score`**                | The rating given, ranging from 1 to 5.             |
| `thumbsUpCount`        | The number of "thumbs up" the review received.     |
| `reviewCreatedVersion` | The version of the app used when the review was written. |
| `at`                   | The date and time the review was posted.           |
| `appVersion`           | The version of the app used when the review was written. |

For our analysis we will focus on two aspects of the reviews. The `content` will be the input to our models and the `score` will be the output.

## Analysis and Visualization

You can get an initial grasp of the dataset here: [`data_visualization.ipynb`](DATA/data_visualization.ipynb). Where we will perform some data analysis and visualization. This step isn't important, but if you want a deeper understanding of the dataset, you can look into it.

## Clean Up

This section outlines the steps to prepare a cleaned dataset. First, we remove unnecessary columns to simplify the dataset. Then, we drop duplicate rows and any rows with missing values. In addition, we categorize the ratings by transforming the score column. In this step, we convert ratings of 4 and 5 stars to "positive," 3 stars to "neutral," and 1 and 2 stars to "negative."

You can see all these steps in: [`data_cleanup.ipynb`](DATA/data_cleanup.ipynb).

We then save the cleaned up dataset as: [`cleaned_data.csv`](DATASETS/cleaned_data.csv).

# Text

At this point, we need to deal with the `content` of the review. This is because it often contains unwanted characters, such as punctuation, emojis, and characters repeated many times. To address this, we first **pre-process** the text to clean it up. We then convert the words into numbers through a process called text **vectorisation**. This is because our models work with numerical data, not text.

## Text Pre-Processing

The text pre-processing is done in: [`text_preprocessing.ipynb`](TEXT/text_preprocessing.ipynb).

The pre-processing goes as follows:
- We load the cleaned dataset: [`cleaned_data.csv`](DATA/cleaned_data.csv).
- We clean up the text using several techniques (turn the text into lowercase, replace emojis, remove repeated characters) to be ready for vectorization.

The pre-processed dataset is saved as: [`preprocessed_text.csv`](DATASETS/preprocessed_text.csv).

## Text Vectorization

At this point, we are ready to turn our words into numbers. There are many ways to do this. The ones we have explored in this repository are:
- BoW (Bag of Words): [`bagofwords.ipynb`](TEXT/bagofwords.ipynb)
- TF-IDF (Term Frequency - Inverse Document Frequency): [`tfidf.ipynb`](TEXT/tfidf.ipynb)
- Word2Vec (Word 2 Vector): [`word2vec.ipynb`](TEXT/word2vec.ipynb)
- GloVe (Global Vectors): [`glove.ipynb`](TEXT/glove.ipynb)

We will give you some general guidelines on which one to choose, and the pros and cons of each.

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
