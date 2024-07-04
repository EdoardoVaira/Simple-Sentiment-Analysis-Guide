
## Introduction 

This will be a study of different NLP methods for Sentiment Analysis. We will start with simple approaches and then try more complex ones.

For the entire "project" we will keep everything as simple as possible.

---
## Outline

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

---

## Data

### Dataset

The dataset we will use for all these experiments is [Netflix Reviews \[DAILY UPDATED\]](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated/data) from [Kaggle](https://www.kaggle.com/).

We have 8 columns:

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


### Analysis and Visualization

You can get an initial grasp of the dataset here [`data_visualization.ipynb`](data_visualization.ipynb)

**TODO**:

I think in this section we have to do this:

- Cleaning Up the dataset
	- Removing the duplicates
	- Removing random shit we don't need (Like dropping some columns. Like "reviewId" is useless.)
- Visualizing and understanding the dataset

---

## Text

### Text Pre-Processing

The text pre-processing is done by the [`text_preprocessing.ipynb`](text_preprocessing.ipynb)

The pre-processing goes as follows:
- Drop the unneeded columns.
- Drop missing and duplicated values.
- Add a column with the sentiment label, according to the scores.
- Modify the text using several techniques (turn text into lowercase, replace emojis, remove unwanted numbers, punctuations and stop words, lemmatize) to be ready for vectorization.

The pre-processed Dataframe is saved in the [`preprocessed_text.csv`](preprocessed_text.csv) file.

### From Words to Numbers

TODO
