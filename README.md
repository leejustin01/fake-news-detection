# Fake News Detection

This project implements a fake news detection pipeline that classifies news articles as **real** or **fake** using **Sentence-BERT (SBERT)** embeddings and a downstream machine learning classifier.

## Overview

The system takes a news article (text or URL), extracts and preprocesses the content, generates semantic embeddings using SBERT, and predicts whether the article is likely fake or real along with a confidence score.

This project is part of an exploration into NLP-based misinformation detection.

## Tech Stack

- Python  
- PyTorch  
- Sentence-BERT (`sentence-transformers`)  
- scikit-learn  
- pandas / NumPy  

## Dataset

Fake and Real News Dataset (Kaggle):  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset consists of two CSV files containing labeled fake and real news articles.

## Pipeline

1. Load and preprocess article text  
2. Generate embeddings using SBERT  
3. Train a classifier on the embeddings  
4. Predict label and confidence for unseen articles  


For a detailed explanation of the methods and analysis, see [Methods](docs/methods.md).


## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```
Training and inference scripts can be run from the command line.

## Current Status

- SBERT-based preprocessing implemented
- Classifier training in progress
- Web app integration planned

## Future Work

- Improve model performance with hyperparameter tuning
- Add article URL parsing
- Deploy as an interactive web application
