# machine_learning_project-supervised-learning

## Project Outcomes

Goal:
- Use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements.
- Utilize the project dataset and requirements to hone the following skills:
    - Exploratory data analysis using Pandas and Python methods and functions such as utility methods available in Pandas dataframes
    - Data Preprocessing and feature engineering
    - Training a machine learning model using 2 methodologies:
        - Logistic Regression Classification
        - Random Forest Classifier
    - Model Evaluation and Interpretation of Findings on Model Performance
- Practise using Cross-Validation techniques such as K-Folds Cross-Validation
- Explore using GridSearchCV and RandomizedSearchCV for the Random Forest Classification section, especially in exploration of Hyperparameter Tuning
- Explore differing measures of "scoring", including differentiating between Accuracy, Recall, Precision, F1
- Explore the different measure of Model performance for a Classifier vs. a Regressor

### Duration:
- Project is intended to take: Approximately 3 hours and 20 minutes.
- Project took closer to 16 hours to complete, including getting more familiar with Scikit Learn documentation on RandomForestClassifer, GridSearchCV, RandomizedSearchCV and the parameters available for each class.

### Project Description:
(Kept from the Original Lighthouse Labs Project Description)

In this project, you will apply supervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

The data set for this project is the "Diabetes" dataset from the National Institute of Diabetes and Digestive and Kidney Diseases 
The project will involve the following tasks:

-	Exploratory data analysis and pre-processing: We will import and clean the data sets, analyze and visualize the relationships between the different variables, handle missing values and outliers, and perform feature engineering as needed.
-	Supervised learning: We will use the Diabetes dataset to build a machine learning model that can predict whether a patient has diabetes or not, using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We will select at least two models, including one ensemble model, and compare their performance.

The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

### Conclusions:
The following conclusions are also available within the Python Notebook entitled "Supervised Learning - Project.ipynb" and are copied here for ease of the reader:

### Discussion of Results for the Training Section

#### Results from the Training of 3 Models:

| Generic Model Name  | Cross-Validation Used                    | Hyper Parameters                                                                                                                                                                                                                 | Training Models Scored On      (Best Model Chosen by What Highest Measure?) | Training Recall Score Achieved | Peformance on Test Data:      Recall Score | Performance on Test Data:      F1 Score |
|---------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------------|--------------------------------------------|-----------------------------------------|
|                     |                                          |                                                                                                                                                                                                                                  |                                                                             |                                |                                            |                                         |
| Logistic Regression | None                                     |                                                                                                                                                                                                                                  | N/A                                                                         | N/A                            | 0.67                                       | 0.66                                    |
| Logistic Regression | K-Folds (10 folds)                       |                                                                                                                                                                                                                                  | Recall                                                                      | 0.643                          | 0.6                                        | 0.706                                   |
| Random Forest       | K-Folds (5 folds) with GridSearch        | param_grid = {          'n_estimators' : [100,   200],          'max_features' : ['sqrt',   'log2'],          'max_depth': [2, 9],          'criterion': ['gini']      }                                                         | Recall                                                                      | 0.588                          | 0.667                                      | 0.727                                   |
| Random Forest       | K-Folds (10 folds) with RandomizedSearch | param_randomized = {          'n_estimators' : [100, 200, 300,   400, 500],          'max_features' : ['sqrt', 'log2',   None],          'max_depth': [3, 5, 10],          'criterion': ['gini', 'entropy',   'log_loss']      } | Recall                                                                      | 0.576                          | 0.6                                        | 0.679                                   |

#### Reflections on the Model Training Measures:

Findings from Model Training:

* The plain-jane Logistic Regression with no cross-validation performed the best on Test data using Recall score (which we earlier decided was the most important performance measure for a "disease state" predictive model).

* When using F1 score on Test data as a measure, which is a harmonized mean of Recall and Precision, the Random Forest model performed best.

* For all CV models, the scoring method was "Recall" instead of Accuracy or Precision, again because this was earlier deemed the most important performance measure for a "disease state" predictive model.  Perhaps if "Accuracy" had been left as the scoring mechanism for the model selected from Training data, the performance measures or outcomes might have been different with Testing data.

* It was somewhat unsurprising that the GridSearch with definite inclusion of RandomForest max_depth of 9, did better even with fewer folds, than RandomizedSearch.

* It was an interesting discovery how much computation power/time was required for GridSearch, which meant its hyperparameters needed to be limited, so amount of tuning needed to be limited.  Had the same exhaustive list of hyperparmeters as given to RandomizedSearch had been given to GridSearch, it may well have had even higher performance statistics on Test data!

* I had expected that a Random Forest model would do better than Logistic Regression, and expected that especially with Cross-Validation and Hyperparameter Tuning thrown into the mix.  I was surprised that with all the extra computation effort and tuning, the Performance (Recall score) for Test data (0.667) was approximately the same as plain-jane basic Logistic Regression (0.67).  This speaks to the power of the Logistic Regression approach.  **However**, when looking at a blended measure of performance, the F1 score (blend between Recall and Precision), the Random Forest GridSearch method was highest of all at 0.727.  Undoubtedly, given more time, hyperparameters, and computational power, we may be able to get an even more performant model using the Random Forest with CV/Grid Search methodology.  It is very promising!

#### Overall Conclusions

1. The best performing model (when using Recall as the primary performance measure, on Test data) was the regular, non-CV Logistic Regression.  This was somewhat unexpected given the extra cross-validation efforts and hyperparameter tuning available to the Random Forest approach.

2. Random Forest with Grid Search seems promising, especially if:
    - Given more computational power and time, to allow more hyperparameters
    - A different metric is used to measure performance or "predictive accuracy" rather than Recall (such as F1 score)

3. Chi-Squared statistical tests and feature selection methodologies are very likely to be valuable in order to slim down the features in this model, and potentially increase predictive power, performance and "accuracy".

4. Though the dataset was not unbalanced, it might be interesting to attempt oversampling techniques to bolster the dataset and see if a better predictive model could be generated.

### Next Steps

This exercise demonstrated to me the areas where I would like to gain more understanding and expertise in, including:
1. Understand the best practise tests required to determine from a statistical significance perspective, whether Features are related to each other (more than correlation matrix)
2. Understand how the above plays into Feature Selection methodologies
3. Understand ways to train models to interpolate missing data when there is missing data
4. Understand appropriate ways to replace Outliers, and ways to know from a statistical significance and propriety perspective, when replacing Outliers will negatively impact predictive value of the model.
5. More research to better understand the n_jobs parameters for GridSearchCV and RandomizedSearchCV classes, to see how to set this on different types of computers and on my laptop specifically, to know how to maximize the use of my computing environment without risking program crash.
6. Understand and employ tools like Optuna and Ray to better scale the computing environment to be able to safely add computing power for processor-intensive activities like GridSearch with large numbers of hyperparameter methods.