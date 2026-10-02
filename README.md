# Fake News Detection Using Machine Learning

## Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to classify news text as REAL or FAKE.

The text is converted into numerical features using TF-IDF, and a Logistic Regression model is trained to make predictions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

## How It Works

1. Load the news dataset.
2. Remove empty records.
3. Split the data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression model.
6. Test the model and calculate accuracy.
7. Enter new news text to get a REAL/FAKE prediction.

## Dataset

A small synthetic demonstration dataset was used to test the project pipeline. It is intended for educational demonstration and does not represent real-world news accuracy.

## Running the Project

Run the following command in the terminal:

```bash
python fake_news_detection_ml.py