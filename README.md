**TODO**:

- Understand what the `reviewCreatedVersion` in the dataset means.
- We should use a "word cloud" somewhere.


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

This section is just to create a cleaned up dataset. By removing the useless columns. And maybe adding some extra useful columns. And Removing the duplicates. And filling the missing values.

This is done in the file [`data_cleanup.ipynb`](data_cleanup.ipynb)

This will generate a new dataset: [`cleaned_data.csv`](cleaned_data.csv)

# Text

## Text Pre-Processing

The text pre-processing is done by the [`text_preprocessing.ipynb`](text_preprocessing.ipynb)

The pre-processing goes as follows:
- Load the cleaned dataset.
- Modify the text using several techniques (turn text into lowercase, replace emojis, remove unwanted numbers, punctuations and stop words, lemmatize) to be ready for vectorization.

The pre-processed Dataframe is saved in the [`preprocessed_text.csv`](preprocessed_text.csv) file.

## From Words to Numbers

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