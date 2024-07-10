from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
from sklearn.metrics import f1_score
from os import path, makedirs, walk
from joblib import dump, load
import json
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

# Utility Functions


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Model trained in {:2f} seconds".format(end-start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made Predictions in {:2f} seconds".format(end-start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return f1_score(target, y_pred, average='micro'), acc


def model(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))


def derive_clean_sheet(src):
    arr = []
    n_rows = src.shape[0]

    for data in range(n_rows):

        #[HTHG, HTAG]
        values = src.iloc[data].values
        cs = [0, 0]

        if values[0] == 0:
            cs[1] = 1

        if values[1] == 0:
            cs[0] = 1

        arr.append(cs)

    return arr


# Data gathering

en_data_folder = '/Users/aleksandardzudzevic/PycharmProjects/CompSciEE/data/english-premier-league_zip'
es_data_folder = '/Users/aleksandardzudzevic/PycharmProjects/CompSciEE/data/spanish-la-liga_zip'
fr_data_folder = '/Users/aleksandardzudzevic/PycharmProjects/CompSciEE/data/french-ligue-1_zip'
ge_data_folder = '/Users/aleksandardzudzevic/PycharmProjects/CompSciEE/data/german-bundesliga_zip'
it_data_folder = '/Users/aleksandardzudzevic/PycharmProjects/CompSciEE/data/italian-serie-a_zip'

data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]

season_range = (9, 18)

data_files = []
for data_folder in data_folders:
    for season in range(season_range[0], season_range[1] + 1):
        data_files.append(
            f'{data_folder}/data/season-{season:02d}{(season+1):02d}_csv.csv')
data_frames = []

for data_file in data_files:
    print('here',data_file)
    if path.exists(data_file):
        data_frames.append(pd.read_csv(data_file))
data = pd.concat(data_frames).reset_index()
print(data)

# Pre processing

input_filter = ['home_encoded', 'away_encoded', 'HTHG', 'HTAG', 'HS',
                'AS', 'HST', 'AST', 'HR', 'AR']
output_filter = ['FTR']

cols_to_consider = input_filter + output_filter

encoder = LabelEncoder()
home_encoded = encoder.fit_transform(data['HomeTeam'])
home_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['home_encoded'] = home_encoded

encoder = LabelEncoder()
away_encoded = encoder.fit_transform(data['AwayTeam'])
away_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['away_encoded'] = away_encoded

# Deriving Clean Sheet
# htg_df = data[['HTHG', 'HTAG']]
# cs_data = derive_clean_sheet(htg_df)
# cs_df = pd.rame(cs_data, columns=['HTCS', 'ATCS'])

# data = pd.concat([data, cs_df], axis=1)

data = data[cols_to_consider]

print(data[data.isna().any(axis=1)])
data = data.dropna(axis=0)

# Training & Testing

X = data[input_filter]
Y = data['FTR']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

svc_classifier = SVC(random_state=100, kernel='rbf')
lr_classifier = LogisticRegression(multi_class='ovr', max_iter=500)
nbClassifier = GaussianNB()
dtClassifier = DecisionTreeClassifier()
rfClassifier = RandomForestClassifier()

X = data[input_filter]
Y = data['FTR']

# Combine classifiers into a dictionary


input_filters = ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR']

# Combine classifiers into a dictionary
classifiers = {
    "Logistic Regression": lr_classifier,
    "Gaussian Naive Bayes": nbClassifier,
    "Random Forest": rfClassifier
}

# Create an empty DataFrame to store the results
results_data = []

# Iterate over the input filters and gradually add one more input filter each time
for i in range(2, len(input_filters) + 1):
    print(f"Number of Input Filters: {i}")
    print("=" * 20)

    # Select the current set of input filters
    current_input_filters = input_filters[:i]

    # Prepare the X data using the current set of input filters
    X = data[current_input_filters]
    Y = data['FTR']

    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name} Classifier (Cross-Validation)")
        print("-" * 20)

        start_cv = time()
        cv_results = cross_validate(clf, X, Y, cv = 10, scoring = 'accuracy', return_train_score = False)
        end_cv = time()

        cv_accuracy_mean = np.mean(cv_results['test_score'])
        cv_time_mean = np.mean(cv_results['fit_time'])

        print("Cross-Validation Results:")
        print("-" * 20)
        print(f"Mean Accuracy: {cv_accuracy_mean:.4f}")
        print(f"Mean Time Needed: {cv_time_mean:.4f} seconds")

        cv_accuracy_std = np.std(cv_results['test_score'])
        results_data.append({
            "Number of Input Filters": i,
            "Classifier": clf_name,
            "Mean Accuracy": cv_accuracy_mean,
            "Accuracy Std Dev": cv_accuracy_std,
            "Mean Time": cv_time_mean
        })
results_df = pd.DataFrame(results_data)

# Print the results as a table without the Accuracy Std Dev column
print("\nResults Table:")
print("=" * 20)
print(results_df.drop(columns=['Accuracy Std Dev']))

# Create a separate DataFrame for each classifier
lr_results = results_df[results_df['Classifier'] == 'Logistic Regression']
nb_results = results_df[results_df['Classifier'] == 'Gaussian Naive Bayes']
rf_results = results_df[results_df['Classifier'] == 'Random Forest']

# Plotting accuracy for each classifier with error bars for standard deviation and connecting lines
plt.figure(figsize=(10, 6))
plt.errorbar(lr_results['Number of Input Filters'], lr_results['Mean Accuracy'], yerr=0.5*lr_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Logistic Regression', color='tab:blue')
plt.errorbar(nb_results['Number of Input Filters'], nb_results['Mean Accuracy'], yerr=0.5*nb_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Gaussian Naive Bayes', color='tab:orange')
plt.errorbar(rf_results['Number of Input Filters'], rf_results['Mean Accuracy'], yerr=0.5*rf_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Random Forest', color='tab:green')

# Adding lines to connect the points
plt.plot(lr_results['Number of Input Filters'], lr_results['Mean Accuracy'], marker='o', color='tab:blue', label='_nolegend_')
plt.plot(nb_results['Number of Input Filters'], nb_results['Mean Accuracy'], marker='o', color='tab:orange', label='_nolegend_')
plt.plot(rf_results['Number of Input Filters'], rf_results['Mean Accuracy'], marker='o', color='tab:green', label='_nolegend_')

plt.xlabel('Number of Input Filters')
plt.ylabel('Accuracy (Percentage)')
plt.title('Accuracy vs Number of Input Filters')
plt.legend()
plt.grid()
plt.show()

# Plotting execution speed for all classifiers with standard deviation
plt.figure(figsize=(10, 6))
plt.errorbar(lr_results['Number of Input Filters'], lr_results['Mean Time'], yerr=lr_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Logistic Regression', color='tab:blue')
plt.errorbar(nb_results['Number of Input Filters'], nb_results['Mean Time'], yerr=nb_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Gaussian Naive Bayes', color='tab:orange')
plt.errorbar(rf_results['Number of Input Filters'], rf_results['Mean Time'], yerr=rf_results['Accuracy Std Dev'], fmt='o', capsize=5, label='Random Forest', color='tab:green')

# Adding lines to connect the points
plt.plot(lr_results['Number of Input Filters'], lr_results['Mean Time'], marker='o', color='tab:blue', label='_nolegend_')
plt.plot(nb_results['Number of Input Filters'], nb_results['Mean Time'], marker='o', color='tab:orange', label='_nolegend_')
plt.plot(rf_results['Number of Input Filters'], rf_results['Mean Time'], marker='o', color='tab:green', label='_nolegend_')

plt.xlabel('Number of Input Filters')
plt.ylabel('Mean Execution Time (seconds)')
plt.title('Execution Time vs Number of Input Filters')
plt.legend()
plt.grid()
plt.show()
