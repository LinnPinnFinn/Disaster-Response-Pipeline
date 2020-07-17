# Disaster Response Pipeline Project

### Description:

A web app that classifies disaster messages to assist in sending the messages to the appropriate disaster relief agency. The web app also includes visualizations of the data. The dataset used to train the classifier is provided by [Figure Eight](https://www.figure-eight.com/). 

Please note that the dataset is imbalanced (some category labels have very few examples) and a classifier trained on imbalanced data tends to favor the majority classes (the classes with the most samples). This could be minimized by re-sampling the data (under or over-sampling), but in this case the imbalance between the classes has been minimized only by training the classifier based on the F1-score, rather than accuracy, as a performance metric. Using accuracy as a performance metric for a classifier trained on imbalanced data might not give a true picutre of the effectiveness of the model. The model might have a total accuracy score of 98% but might in reality only predict majority classes, and if the goal is to predict minority classes the model is of no use. Instead, the f1-score - which is a weighted average of precision and recall - is a better measure since it penalizes extreme values.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
### Screenshots:

Message Classifier Results Page:
![](https://github.com/LinnPinnFinn/Udacity-Data-Scientist-Nanodegree-Portfolio/blob/master/Project%20-%20Disaster%20Response%20Pipeline/screenshots/Screenshot%202019-11-01%20at%2019.13.22.png)

Data Visualization 1:
![](https://github.com/LinnPinnFinn/Udacity-Data-Scientist-Nanodegree-Portfolio/blob/master/Project%20-%20Disaster%20Response%20Pipeline/screenshots/Screenshot%202019-11-01%20at%2019.11.43.png)

Data Visualization 2:
![](https://github.com/LinnPinnFinn/Udacity-Data-Scientist-Nanodegree-Portfolio/blob/master/Project%20-%20Disaster%20Response%20Pipeline/screenshots/Screenshot%202019-11-01%20at%2019.11.58.png)

Data Visualization 3:
![](https://github.com/LinnPinnFinn/Udacity-Data-Scientist-Nanodegree-Portfolio/blob/master/Project%20-%20Disaster%20Response%20Pipeline/screenshots/Screenshot%202019-11-01%20at%2019.12.29.png)