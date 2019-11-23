# Disaster Response Pipeline Project

![Web App Banner](/imgs/banner.png?raw=true)

## 1. Overview

This project is part of [Data Science Nanodegree Program by Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The dataset, provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/), contains real messages that were sent during disaster events. The aim of this project is to build a machine learning pipeline to classify these messages so that it can be sent to an appropriate disaster relief agency. The project also includes a web page interface where an emergency worker can input a new message and get classification results in several categories.

## 2. Getting Started

## 2.1. Dependencies

**Python 3.6.5** was used to build this project. No extra libraries are necessary beyond the Anaconda distribution. Libraries used:  

*   SQLite Database: **SQLAlchemy**
*   Machine Learning: **NumPy**, **Pandas**, **Scikit-Learn**
*   Natural Language Process: **NLTK**
*   Web App Interface: **Flask**
*   Data Visualization: **Plotly**

## 2.2. Installation

Clone this repository

`git clone https://github.com/alghsaleh/disaster-response.git`

## 2.3. File Descriptions

*   `data/disaster_messages.csv`: Dataset contains original messages in its original language, their English translation, and their genre (`direct`, `news`, or `social`).
*   `data/disaster_categories.csv`: Dataset contains dozens of classes for message content and are noted in column titles with a simple binary (0=no, 1=yes).
*   `data/process_data.py`: Python script that loads in `disaster_messages.csv` and `disaster_categories.csv`, merges them together, performs all necessary data preprocessing, and outputs a cleaned SQLite database ready for machine learning.
*   `models/train_classifier.py`: Python script that loads in cleaned SQLite database, tokenizes text, performs essential splitting (first into features and targets and then into training and testing datasets), builds an optimized pipeline for text processing via GridSearchCV, displays important metrics, and outputs the model as pickle file.
*   `app/run.py`: Python script that runs the web app interface.
*   `app/templates/*`: HTML files containing the necessary elements to put together the web app interface.
*   `imgs/*`: PNG files for banner and screenshots used in this `README.md` file.

## 2.4. Usage

1. To run ETL pipeline that preprocesses data and stores it in SQLite database; run the following command: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run the ML pipeline that builds, train, and saves the model: run the following command: ``python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl``
3. To start running the web page interface; run the following command in the `app`'s directory: `python run.py` and then visit: `http://0.0.0.0:3001/`

## 2.5. Screenshots

1. The home page shows two graphs about the distribution of the training dataset. \
![Home Page](/imgs/screenshot1.png?raw=true)
2. The message classification page highlights predicted categories. \
![Message Classification Page](/imgs/screenshot2.png?raw=true)

## 3. Acknowledgements

*   [Udacity](https://www.udacity.com/) for preparing and providing guidelines to complete this code.
*   [Figure Eight](https://www.figure-eight.com/) for providing disaster messages dataset.

## 4. License

This work is published with love from [Saudi Arabia](https://www.visitsaudi.com/en) under [Apache License 2.0](/LICENSE).
