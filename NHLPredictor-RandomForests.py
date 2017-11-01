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
#Create an object called iris with the iris data
# iris = {'data': array([[ 5.1,  3.5,  1.4,  0.2],
#        [ 4.9,  3. ,  1.4,  0.2],
#        [ 4.7,  3.2,  1.3,  0.2],
#        [ 4.6,  3.1,  1.5,  0.2],
#        [ 5. ,  3.6,  1.4,  0.2],
#        [ 5.4,  3.9,  1.7,  0.4],
#        [ 4.6,  3.4,  1.4,  0.3],
#        [ 5. ,  3.4,  1.5,  0.2],
#        [ 4.4,  2.9,  1.4,  0.2],
#        [ 4.9,  3.1,  1.5,  0.1],
#        [ 5.4,  3.7,  1.5,  0.2],
#        [ 4.8,  3.4,  1.6,  0.2],
#        [ 4.8,  3. ,  1.4,  0.1],
#        [ 4.3,  3. ,  1.1,  0.1],
#        [ 5.8,  4. ,  1.2,  0.2],
#        [ 5.7,  4.4,  1.5,  0.4],
#        [ 5.4,  3.9,  1.3,  0.4],
#        [ 5.1,  3.5,  1.4,  0.3],
#        [ 5.7,  3.8,  1.7,  0.3],
#        [ 5.1,  3.8,  1.5,  0.3],
#        [ 5.4,  3.4,  1.7,  0.2],
#        [ 5.1,  3.7,  1.5,  0.4],
#        [ 4.6,  3.6,  1. ,  0.2],
#        [ 5.1,  3.3,  1.7,  0.5],
#        [ 4.8,  3.4,  1.9,  0.2],
#        [ 5. ,  3. ,  1.6,  0.2],
#        [ 5. ,  3.4,  1.6,  0.4],
#        [ 5.2,  3.5,  1.5,  0.2],
#        [ 5.2,  3.4,  1.4,  0.2],
#        [ 4.7,  3.2,  1.6,  0.2],
#        [ 4.8,  3.1,  1.6,  0.2],
#        [ 5.4,  3.4,  1.5,  0.4],
#        [ 5.2,  4.1,  1.5,  0.1],
#        [ 5.5,  4.2,  1.4,  0.2],
#        [ 4.9,  3.1,  1.5,  0.1],
#        [ 5. ,  3.2,  1.2,  0.2],
#        [ 5.5,  3.5,  1.3,  0.2],
#        [ 4.9,  3.1,  1.5,  0.1],
#        [ 4.4,  3. ,  1.3,  0.2],
#        [ 5.1,  3.4,  1.5,  0.2],
#        [ 5. ,  3.5,  1.3,  0.3],
#        [ 4.5,  2.3,  1.3,  0.3],
#        [ 4.4,  3.2,  1.3,  0.2],
#        [ 5. ,  3.5,  1.6,  0.6],
#        [ 5.1,  3.8,  1.9,  0.4],
#        [ 4.8,  3. ,  1.4,  0.3],
#        [ 5.1,  3.8,  1.6,  0.2],
#        [ 4.6,  3.2,  1.4,  0.2],
#        [ 5.3,  3.7,  1.5,  0.2],
#        [ 5. ,  3.3,  1.4,  0.2],
#        [ 7. ,  3.2,  4.7,  1.4],
#        [ 6.4,  3.2,  4.5,  1.5],
#        [ 6.9,  3.1,  4.9,  1.5],
#        [ 5.5,  2.3,  4. ,  1.3],
#        [ 6.5,  2.8,  4.6,  1.5],
#        [ 5.7,  2.8,  4.5,  1.3],
#        [ 6.3,  3.3,  4.7,  1.6],
#        [ 4.9,  2.4,  3.3,  1. ],
#        [ 6.6,  2.9,  4.6,  1.3],
#        [ 5.2,  2.7,  3.9,  1.4],
#        [ 5. ,  2. ,  3.5,  1. ],
#        [ 5.9,  3. ,  4.2,  1.5],
#        [ 6. ,  2.2,  4. ,  1. ],
#        [ 6.1,  2.9,  4.7,  1.4],
#        [ 5.6,  2.9,  3.6,  1.3],
#        [ 6.7,  3.1,  4.4,  1.4],
#        [ 5.6,  3. ,  4.5,  1.5],
#        [ 5.8,  2.7,  4.1,  1. ],
#        [ 6.2,  2.2,  4.5,  1.5],
#        [ 5.6,  2.5,  3.9,  1.1],
#        [ 5.9,  3.2,  4.8,  1.8],
#        [ 6.1,  2.8,  4. ,  1.3],
#        [ 6.3,  2.5,  4.9,  1.5],
#        [ 6.1,  2.8,  4.7,  1.2],
#        [ 6.4,  2.9,  4.3,  1.3],
#        [ 6.6,  3. ,  4.4,  1.4],
#        [ 6.8,  2.8,  4.8,  1.4],
#        [ 6.7,  3. ,  5. ,  1.7],
#        [ 6. ,  2.9,  4.5,  1.5],
#        [ 5.7,  2.6,  3.5,  1. ],
#        [ 5.5,  2.4,  3.8,  1.1],
#        [ 5.5,  2.4,  3.7,  1. ],
#        [ 5.8,  2.7,  3.9,  1.2],
#        [ 6. ,  2.7,  5.1,  1.6],
#        [ 5.4,  3. ,  4.5,  1.5],
#        [ 6. ,  3.4,  4.5,  1.6],
#        [ 6.7,  3.1,  4.7,  1.5],
#        [ 6.3,  2.3,  4.4,  1.3],
#        [ 5.6,  3. ,  4.1,  1.3],
#        [ 5.5,  2.5,  4. ,  1.3],
#        [ 5.5,  2.6,  4.4,  1.2],
#        [ 6.1,  3. ,  4.6,  1.4],
#        [ 5.8,  2.6,  4. ,  1.2],
#        [ 5. ,  2.3,  3.3,  1. ],
#        [ 5.6,  2.7,  4.2,  1.3],
#        [ 5.7,  3. ,  4.2,  1.2],
#        [ 5.7,  2.9,  4.2,  1.3],
#        [ 6.2,  2.9,  4.3,  1.3],
#        [ 5.1,  2.5,  3. ,  1.1],
#        [ 5.7,  2.8,  4.1,  1.3],
#        [ 6.3,  3.3,  6. ,  2.5],
#        [ 5.8,  2.7,  5.1,  1.9],
#        [ 7.1,  3. ,  5.9,  2.1],
#        [ 6.3,  2.9,  5.6,  1.8],
#        [ 6.5,  3. ,  5.8,  2.2],
#        [ 7.6,  3. ,  6.6,  2.1],
#        [ 4.9,  2.5,  4.5,  1.7],
#        [ 7.3,  2.9,  6.3,  1.8],
#        [ 6.7,  2.5,  5.8,  1.8],
#        [ 7.2,  3.6,  6.1,  2.5],
#        [ 6.5,  3.2,  5.1,  2. ],
#        [ 6.4,  2.7,  5.3,  1.9],
#        [ 6.8,  3. ,  5.5,  2.1],
#        [ 5.7,  2.5,  5. ,  2. ],
#        [ 5.8,  2.8,  5.1,  2.4],
#        [ 6.4,  3.2,  5.3,  2.3],
#        [ 6.5,  3. ,  5.5,  1.8],
#        [ 7.7,  3.8,  6.7,  2.2],
#        [ 7.7,  2.6,  6.9,  2.3],
#        [ 6. ,  2.2,  5. ,  1.5],
#        [ 6.9,  3.2,  5.7,  2.3],
#        [ 5.6,  2.8,  4.9,  2. ],
#        [ 7.7,  2.8,  6.7,  2. ],
#        [ 6.3,  2.7,  4.9,  1.8],
#        [ 6.7,  3.3,  5.7,  2.1],
#        [ 7.2,  3.2,  6. ,  1.8],
#        [ 6.2,  2.8,  4.8,  1.8],
#        [ 6.1,  3. ,  4.9,  1.8],
#        [ 6.4,  2.8,  5.6,  2.1],
#        [ 7.2,  3. ,  5.8,  1.6],
#        [ 7.4,  2.8,  6.1,  1.9],
#        [ 7.9,  3.8,  6.4,  2. ],
#        [ 6.4,  2.8,  5.6,  2.2],
#        [ 6.3,  2.8,  5.1,  1.5],
#        [ 6.1,  2.6,  5.6,  1.4],
#        [ 7.7,  3. ,  6.1,  2.3],
#        [ 6.3,  3.4,  5.6,  2.4],
#        [ 6.4,  3.1,  5.5,  1.8],
#        [ 6. ,  3. ,  4.8,  1.8],
#        [ 6.9,  3.1,  5.4,  2.1],
#        [ 6.7,  3.1,  5.6,  2.4],
#        [ 6.9,  3.1,  5.1,  2.3],
#        [ 5.8,  2.7,  5.1,  1.9],
#        [ 6.8,  3.2,  5.9,  2.3],
#        [ 6.7,  3.3,  5.7,  2.5],
#        [ 6.7,  3. ,  5.2,  2.3],
#        [ 6.3,  2.5,  5. ,  1.9],
#        [ 6.5,  3. ,  5.2,  2. ],
#        [ 6.2,  3.4,  5.4,  2.3],
#        [ 5.9,  3. ,  5.1,  1.8]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'target_names': array(['setosa', 'versicolor', 'virginica'],
#       dtype='<U10'), 'DESCR': 'Iris Plants Database\n====================\n\nNotes\n-----\nData Set Characteristics:\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThis is a copy of UCI ML iris datasets.\nhttp://archive.ics.uci.edu/ml/datasets/Iris\n\nThe famous Iris database, first used by Sir R.A Fisher\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\nReferences\n----------\n   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...\n', 'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']}


#team stats from last 5 games:

stats_feature_names = ['Goals For', 'Goals Against', 'Points', 'Current Injuries']

previous_5_game_stats = np.array(
              [ #GF  GA pts currentInjuries
                [13, 6, 15, 1], #1
                [15, 4, 17, 0], #1
                [11, 2, 13, 1], #1
                [4,  8,  5, 4], #0
                [6,  9,  7, 6], #0
                [3,  12, 4, 3]  #0
              ])


# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
stats_df = pd.DataFrame(previous_5_game_stats, columns=stats_feature_names)

# View the top 5 rows
df.head()

print("Iris.data: " + str(iris.data) + "\n Iris.feature_names: " + str(iris.feature_names) + "\n df: " + str(df))
print("previous_5_game_stats: \n" + str(previous_5_game_stats) + "\n feature_names: " + str(stats_feature_names) + "\n stats_df: \n" + str(stats_df))




stats_target = [1,1,1,0,0,0]
stats_target_names = ['win, loss']

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
stats_df['outcome'] = pd.Categorical.from_codes(stats_target, iris.target_names)

print("~~~iris.target~~~")
print(iris.target)
print("~~~end~~~\n\n\n")

print("~~~iris.target_names~~~")
print(iris.target_names)
print("~~~end~~~\n\n\n")

print("~~~stats_target_names~~~")
print(stats_target_names)
print("~~~end~~~\n\n\n")

print("~~~stats_target~~~")
print(stats_target)
print("~~~end~~~\n\n\n")

print(pd.Categorical.from_codes(iris.target, iris.target_names))
stats_df['outcome'] = np.array(
              [ #GF  GA pts currentInjuries
                "Win",
                "Win",
                "Win",
                "Loss",
                "Loss",
                "Loss"
              ])

# View the top 5 rows
#df.head()

print(df['species'])
print(stats_df['outcome'])



# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
stats_df['is_train'] = np.random.uniform(0, 1, len(stats_df)) <= .75

print("~~~df.head~~~")
print(df.head())
print("~~~end~~~\n\n\n")

print("~~~stats_df (not .head() cuz there's only 6~~~")
print(stats_df)
print("~~~end~~~\n\n\n")




# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
stats_train, stats_test = stats_df[stats_df['is_train']==True], stats_df[stats_df['is_train']==False]



print("~~~iris test/train~~~")
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print("~~~end~~~\n\n\n")


print("~~~stats test/train~~~")
print('Number of observations in the training data:', len(stats_train))
print('Number of observations in the test data:',len(stats_test))
print("~~~end~~~\n\n\n")






# Create a list of the feature column's names
features = df.columns[:4]
stats_features = stats_df.columns[:4]

# View features
print("~~~features~~~")
print(features)
print("~~~end~~~\n\n\n")

print("~~~stats_features~~~")
print(stats_features)
print("~~~end~~~\n\n\n")





# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]
stats_y = pd.factorize(stats_train['outcome'])[0]
#
# # View target
print("~~~y~~~")
print(y)
print("~~~end~~~\n\n\n")
#
print("~~~stats_y~~~")
print(stats_y)
print("~~~end~~~\n\n\n")



# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)
stats_clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)
stats_clf.fit(stats_train[stats_features], stats_y)

print("~~~Train the Classifier to take the training features and learn how they relateto the training y (the species) ~~~")
print(clf.fit(train[features], y))
print("~~~end~~~\n\n\n")

print("~~~Stats fitting:~~~")
print(stats_clf.fit(stats_train[stats_features], stats_y))
print("~~~end~~~\n\n\n")


clf.predict(test[features])
print("~~~IRIS predict test features:~~~")
print(clf.predict(test[features]))
print("~~~end~~~\n\n\n")

#stats_clf.predict(stats_test[stats_features])
print("~~~STATS predict test features:~~~")
print(stats_clf.predict(stats_test[stats_features]))
print("~~~end~~~\n\n\n")


clf.predict_proba(test[features])[0:10]
print("~~~iris predict proba:~~~")
print(clf.predict_proba(test[features])[0:10])
print("~~~end~~~\n\n\n")

stats_clf.predict_proba(stats_test[stats_features])[0:10]
print("~~~stats predict proba:~~~")
print(stats_clf.predict_proba(stats_test[stats_features])[0:10])
print("~~~end~~~\n\n\n")






print("~~~~~~~Now that we have predicted the species of all plants in the test data, we can compare our predicted species with the that plant's actual species.~~~~~~~\n")

# Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]
stats_preds = stats_target_names[stats_clf.predict(stats_test[stats_features])[0]]



print("~~~iris prediction/actual species~~~")
print(preds)
print("~~~end~~~\n\n\n")

print("~~~stats prediction/actual outcome of game~~~")
print(stats_preds)
print("~~~end~~~\n\n\n")



