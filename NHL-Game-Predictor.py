# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

from array import array

iris=load_iris()

#team stats from last 5 games:
stats_feature_names = ['Goals For', 'Goals Against', 'Points', 'Penalty Kill %']#taedit
previous_games_stats = np.array(
              [ #GF  GA pts PenaltyKill%
                [2,  4, 0, 100], # <- WILL BE PREDICTED, ENTER DATA HERE
                [1,  2, 1, 80],
                [3,  6, 0, 50],

                [3,  2, 2, 100],
                [4,  4, 2, 80],
                [5,  4, 2, 67],
                [0,  3, 0, 100],
                [4,  4, 2, 100],

                #random data:
                [2,  9, 0, 70],
                [9,  1, 19, 100],
              ])

stats_target = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
stats_df = pd.DataFrame(previous_games_stats, columns=stats_feature_names)


print("previous_game_stats: \n" + str(previous_games_stats) + "\n feature_names: " + str(stats_feature_names) + "\n stats_df: \n" + str(stats_df))


stats_target_names = np.array(['Win', 'Loss'])

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
stats_df['outcome'] = pd.Categorical.from_codes(stats_target, stats_target_names)

stats_df['outcome'] = np.array(
              [ #should match up with array of nhl stats above
                "Loss",
                "Loss",
                "Loss",
                "Win",
                "Win",
                "Loss",
                "Win",
                "Win",

                "Loss",
                "Win"
              ])

# View the top 5 rows
#df.head()

# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

#
stats_df['is_train'] = np.random.uniform(0, 1, len(stats_df)) <= .75
stats_df['is_train'][0] = False

print("~~~stats_df~~~")
print(stats_df)
print("~~~end~~~\n\n\n")

print("~~~stats_df['is_train']~~~")
print(stats_df['is_train'])
print("~~~end~~~\n\n\n")


# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
stats_train, stats_test = stats_df[stats_df['is_train']==True], stats_df[stats_df['is_train']==False]
#stats_train['is_train'][0] = False

print("~~~stats_train~~~")
print(stats_train)
print("~~~end~~~\n\n\n")

print("~~~stats_test~~~")
print(stats_test)
print("~~~end~~~\n\n\n")

# Create a list of the feature column's names
features = df.columns[:4]
stats_features = stats_df.columns[:4]

print("stats_Features: " + str(stats_features))
print("iris Features: " + str(features))

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]
stats_y = pd.factorize(stats_train['outcome'])[0]

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)
stats_clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)
stats_clf.fit(stats_train[stats_features], stats_y)


stats_clf.predict_proba(stats_test[stats_features])[0:10]
print("~~~Confidence:~~~")
print(stats_clf.predict_proba(stats_test[stats_features])[0:10])
print("~~~end~~~\n\n\n")



print("~~~~~~~Now that we have predicted the species of all plants in the test data, we can compare our predicted species with the that plant's actual species.~~~~~~~\n")

# Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]
stats_preds = stats_target_names[stats_clf.predict(stats_test[stats_features])]

print("~~~PREDICTED outcome of game:~~~")
print(stats_preds)
print("~~~end~~~\n\n\n")


print("~~~ACTUAL outcome of game:~~~")
print(stats_test['outcome'])
print("~~~end~~~\n\n\n")

pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
print("~~~confusion matrix~~~\nColumns represent what we predicted for the outcome of the game, and rows represent the actual outcome of the game.\n")
print(pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']))
print("~~~end~~~\n\n\n")

pd.crosstab(stats_test['outcome'], stats_preds, rownames=['Actual Outcome'], colnames=['Predicted Outcome'])
print("~~~confusion matrix~~~\nColumns represent what we predicted for the outcome of the game, and rows represent the actual outcome of the game.\n")
print(pd.crosstab(stats_test['outcome'], stats_preds, rownames=['Actual Outcome'], colnames=['Predicted Outcome']))
print("~~~end~~~\n\n\n")

