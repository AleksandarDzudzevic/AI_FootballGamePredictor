# AI Football Game Predicting Project  ⚽️ 

This project investigates the power of machine learning to predict the outcome of football matches to better understand the pros and cons of using different models when performing predictions based on large data sets, in this case, all football games from the major 5 leagues that have been played throughout nine years! 

## Project Goals

* Leverage machine learning models to forecast football match results.
* Analyze the impact of various features on prediction accuracy.
* Identify the optimal model for football match outcome prediction.

## Technical Stack

* **Data Analysis:** pandas
* **Machine Learning:** scikit-learn
* **Visualization:** matplotlib
* **Data Persistence:** joblib

## Data

The project utilizes historical data from top European leagues:

* English Premier League
* Spanish La Liga
* French Ligue 1
* German Bundesliga
* Italian Serie A

The data encompasses features like:

* Home and Away Team Names
* Half-time and Full-time Goals Scored by Each Team
* Shots on Target, Yellow/Red Cards, etc.

### Preprocessing Steps

The code performs essential data preprocessing tasks:

* **Label Encoding:** Categorical features (e.g., team names) are transformed into numerical labels using `LabelEncoder`.
* **Feature Selection:** A relevant subset of features is chosen for model training.
* **Missing Value Handling:**  Rows with missing values (data points with missing information) are removed. 

### Employed Models

The project trains and evaluates the performance of various machine learning models for football match outcome prediction:

* **Logistic Regression**
* **Gaussian Naive Bayes**
* **Random Forest**

**Evaluation Metrics:**

The code utilizes F1-score and accuracy to assess the performance of each model.

### Cross-Validation

The project leverages cross-validation to ensure the generalizability of the models. Cross-validation involves splitting the data into training and testing sets multiple times for a more robust evaluation.

### Expected Results

The project aims to uncover:

* Which model demonstrates the best performance in predicting football match outcomes?
* How the number of input features influences the accuracy and execution time of the models.

The final step involves generating plots to visualize the impact of the number of features on model performance and execution speed.

### Running the Project

1. Ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`).
2. Modify the data folder paths in the code to point to your data location.
3. Run the Python script.

The script will process the data, train the models, perform cross-validation, and generate plots visualizing the results.

### Contributing

We welcome contributions to this project! Feel free to submit pull requests with improvements, bug fixes, or new features. 

Let's work together to enhance the world of football match prediction through the power of AI!  
