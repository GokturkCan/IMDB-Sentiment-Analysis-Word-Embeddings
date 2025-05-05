# IMDB-Sentiment-Analysis-Word-Embeddings
Project Overview
This repository contains our implementation and analysis of sentiment classification on the IMDb movie reviews dataset. We compare two different approaches for text representation:

Traditional TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
Pre-trained GloVe (Global Vectors for Word Representation) embeddings

The project demonstrates how different text representation techniques impact model performance in sentiment analysis tasks, with detailed comparisons and visualizations.
Dataset
We use the IMDb reviews dataset from TensorFlow Datasets (TFDS), containing:

25,000 training examples
25,000 test examples
Binary sentiment labels (0 for negative, 1 for positive)
Movie reviews of varying lengths and complexity

Methods
TF-IDF Approach (Baseline)

Text preprocessing (lowercase, HTML removal, stopword removal, etc.)
TF-IDF vectorization with 10,000 features and unigram+bigram range
Logistic Regression classification

Word Embedding Approach

Same text preprocessing pipeline
Pre-trained GloVe embeddings (100 dimensions)
Document-level representation via word vector averaging
Logistic Regression classification

Key Findings

TF-IDF with Logistic Regression achieved 88.66% accuracy
GloVe embeddings with Logistic Regression achieved 79.98% accuracy
Analysis of why TF-IDF outperformed simple averaging of word embeddings
Visualizations of embedding space using TensorFlow Projector

Repository Structure

IMDB_Sentiment_Analysis.ipynb: Main Jupyter notebook containing all code
Word_Embeddings_in_NLP_Report.pdf: Detailed project report
vectors.tsv & metadata.tsv: Word embedding data for visualization
visualization/: Directory containing TensorFlow Projector screenshots
README.md: This file

Setup and Usage

Clone this repository

bashgit clone https://github.com/[username]/IMDB-Sentiment-Analysis-Word-Embeddings.git

Install the required dependencies

bashpip install tensorflow tensorflow-datasets numpy pandas scikit-learn nltk matplotlib seaborn requests

Run the Jupyter notebook

bashjupyter notebook IMDB_Sentiment_Analysis.ipynb
Alternatively, you can open the notebook in Google Colab.
Requirements

Python 3.6+
TensorFlow 2.x
TensorFlow Datasets
NumPy
Pandas
scikit-learn
NLTK
Matplotlib
Seaborn
Requests

Word Embedding Visualization
We visualized our word embeddings using TensorFlow Projector (https://projector.tensorflow.org/). The repository includes:

TSV files containing the embeddings and metadata
Screenshots of various visualizations showing semantic relationships between words

To replicate the visualizations:

Go to https://projector.tensorflow.org/
Load the vectors.tsv and metadata.tsv files
Experiment with different dimensionality reduction techniques (PCA, t-SNE, UMAP)
