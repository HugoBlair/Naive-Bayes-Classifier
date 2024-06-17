# Naive-Bayes-Classifier
This project implements a Naive Bayes Classifier for predicting recurrence events in breast cancer patients. The classifier is trained on historical data, calculates the conditional probabilities of various features, and uses these probabilities to predict the class of new instances.

## Features

    age: Age of the patient (e.g., "10-19", "20-29", etc.)
    menopause: Menopause status ("lt40", "ge40", "premeno")
    tumor-size: Size of the tumor (e.g., "0-4", "5-9", etc.)
    inv-nodes: Number of involved nodes (e.g., "0-2", "3-5", etc.)
    node-caps: Presence of node caps ("yes", "no")
    deg-malig: Degree of malignancy ("1", "2", "3")
    breast: Breast side ("left", "right")
    breast-quad: Breast quadrant ("left_up", "left_low", "right_up", "right_low", "central")
    irradiat: Received irradiation ("yes", "no")

## Class Labels

    no-recurrence-events: No recurrence of cancer events
    recurrence-events: Recurrence of cancer events

## Project Structure

    NaiveBayesClassifier.java: Main Java class implementing the Naive Bayes Classifier
    Instance: Inner class representing a single data instance
    data: Directory containing training and testing datasets

## Installation:

### Clone the repository:
```
git clone https://github.com/HugoBlair/Naive-Bayes-Classifier.git
```
```
cd path/to/Naive-Bayes-Classifier
```
### Compile the Java code:
```
javac NaiveBayesClassifier.java
```
### Run the classifier:
```
java NaiveBayesClassifier breast_cancer_testing.csv breast_cancer_training.csv
```
