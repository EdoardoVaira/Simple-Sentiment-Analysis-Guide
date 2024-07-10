**TODO**:

- Understand what the `reviewCreatedVersion` in the dataset means.
- We should use a "word cloud" somewhere.
- Is saying 1, 2 -> negative / 3 -> neutral / 4, 5 -> positive the best way to handle this?  

# Introduction 

*What is this Repository?* (To Fix)

This repository explores various NLP techniques for sentiment analysis of reviews on the Netflix app in the App Store.

*Why is this important and how can this be valuable to you?* (To Fix)

- It will give you a basic understanding of Sentiment Analysis
- You will see the pros/cons of different techniques
- It will be simple
- Easy to understand
- Fast to learn

# Data

## Dataset

The dataset we will use for all these experiments is [Netflix Reviews \[DAILY UPDATED\]](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data) from [Kaggle](https://www.kaggle.com/).

The dataset consists of 8 columns:

| Column Name            | Description                                        |
|------------------------|----------------------------------------------------|
| `reviewId`             | A unique identifier for each review.               |
| `userName`             | The name of the user who submitted the review.     |
| `content`              | The actual text of the review.                     |
| `score`                | The rating given, ranging from 1 to 5.             |
| `thumbsUpCount`        | The number of "thumbs up" the review received.     |
| `reviewCreatedVersion` | __TODO__ - Details needed.                         |
| `at`                   | The date and time the review was posted.           |
| `appVersion`           | The version of the app used when the review was written. |


## Analysis and Visualization

You can get an initial grasp of the dataset here [`data_visualization.ipynb`](data_visualization.ipynb). This step isn't important. But if you want a deeper understanding of the dataset, you can look into it.

## Clean Up

This section details the steps taken to prepare a cleaned dataset. We start by removing unnecessary columns to simplify the dataset. Next, we eliminate duplicate rows and drop any rows with missing values. We also add a new column to categorize the sentiment of each review. You can see all of these steps in [`data_cleanup.ipynb`](data_cleanup.ipynb).

The cleaned dataset is saved as [`cleaned_data.csv`](cleaned_data.csv).

# Text

At this point we are ready for handling the content of the review... *TODO*

## Text Pre-Processing

The text pre-processing is done by the [`text_preprocessing.ipynb`](text_preprocessing.ipynb)

The pre-processing goes as follows:
- Load the cleaned dataset.
- Modify the text using several techniques (turn text into lowercase, replace emojis, remove unwanted numbers, punctuations and stop words, lemmatize) to be ready for vectorization.

The pre-processed Dataframe is saved in the [`preprocessed_text.csv`](preprocessed_text.csv) file.

## From Words to Numbers

TODO

# Picking a Model

At this point we have different ways to deal with the text (BoW - GloVe - Word2Vec ...) and many different models we can pick. So we have a lot of possible combinations. I guess we can just show some of them.

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