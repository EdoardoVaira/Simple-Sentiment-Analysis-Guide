**TODO**:

# Introduction 

*What is this Repository?* (To Fix)

This repository explores various NLP techniques for sentiment analysis of reviews on the Netflix app in the Google Play Store. It offers a detailed, yet easy to follow guide on sentiment analysis starting from basic Exploratory Data Analysis (EDA) and proceeding to text pre-processing and machine learning model training and evaluation. The pros and cons of different text pre-processing techniques, as well as a performance comparisson between them, are presented. Finally, various machine learning methods are explored, from standard ML algorithms, such as Logistic Regression, to more advanced ones, like RNNs and Transformers.

*Why is this important and how can this be valuable to you?* (To Fix)

- It will give you a basic understanding of Sentiment Analysis
- You will see the pros/cons of different techniques
- It will be simple
- Easy to understand
- Fast to learn

# Data

## Dataset

The dataset we will use is [`netflix_reviews.csv`](DATASETS/netflix_reviews.ipynb) from [Kaggle](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data). It contains reviews for the Netflix App in the Google Play Store.

The dataset consists of 8 columns:

| Column Name            | Description                                        |
|------------------------|----------------------------------------------------|
| `reviewId`             | A unique identifier for each review.               |
| `userName`             | The name of the user who submitted the review.     |
| `content`              | The actual text of the review.                     |
| `score`                | The rating given, ranging from 1 to 5.             |
| `thumbsUpCount`        | The number of "thumbs up" the review received.     |
| `reviewCreatedVersion` | The version of the app used when the review was written. |
| `at`                   | The date and time the review was posted.           |
| `appVersion`           | The version of the app used when the review was written. |

For our analysis, we will focus on two aspects of the reviews. The `content` will be the input for our models, and the `score` will be the output.

## Analysis and Visualization

You can get an initial grasp of the dataset here: [`data_visualization.ipynb`](DATA/data_visualization.ipynb). Where we will perform some data analysis and visualization. This step isn't important, but if you want a deeper understanding of the dataset, you can look into it.

## Clean Up

This section outlines the steps to prepare a cleaned dataset. First, we remove unnecessary columns to simplify the dataset. Then, we drop duplicate rows and any rows with missing values. We also add a new column to categorize the sentiment of each review. You can view all these steps in [`data_cleanup.ipynb`](DATA/data_cleanup.ipynb).

Then we save the cleaned dataset as [`cleaned_data.csv`](DATA/cleaned_data.csv).

# Text

At this point we are ready for handling the content of the review. In order to feed the text sequences to our models, several steps need to be taken. Firstly, some text pre-processing is needed to clean up the texts and make them as simple as possible. Then, the text sequences need to be vectorized from words into numbers, so our models can understand them and train on them.

## Text Pre-Processing

The text pre-processing is done in the [`text_preprocessing.ipynb`](text_preprocessing.ipynb).

The pre-processing goes as follows:
- Load the cleaned dataset.
- Modify the text using several techniques (turn text into lowercase, replace emojis, fix contractions, remove unwanted numbers, punctuations and symbols, lemmatize) to be ready for vectorization.

The pre-processed Dataframe is saved in the [`preprocessed_text.csv`](preprocessed_text.csv) file.

## From Words to Numbers

The vectorization methods explored are the following:
- BoW (Bag of Words): [`bagofwords.ipynb`](TEXT/bagofwords.ipynb)
- TF-IDF (Term Frequency - Inverse Document Frequency)
- Word2Vec (Word 2 Vector)
- GloVe (Global Vectors)

They are extensively analyzed in the notebook. We decided not to save them in a seperate Dataframe, since the resulting size was 1.5 GB. Instead we run the necessary code again when training our models, as the vectorizing process is quite fast. Later when we evaluate our models, we will also compare how the different techniques perform for each model.

# Picking a Model

At this point we have different ways to deal with the text vectorization and many different models we can pick. So we have a lot of possible combinations. I guess we can just show some of them.

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