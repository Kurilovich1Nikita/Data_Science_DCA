# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import itertools
import re
import csv
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans  # for clustering
from sklearn.tree import DecisionTreeClassifier  # for decision tree mining
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
global read_file
global accuracy_scores
accuracy_scores = []
plaintext = ''

def parseFile():
    global read_file
    with open('log_files/mHealth_subject2.log', 'r+') as data:
        plaintext = data.read()
        plaintext = plaintext.split()
        #print(plaintext)

    final = [plaintext[i:i+24] for i in range(0,len(plaintext),24)]
    #print(final[0])

    file = open('Output/mHealth_subject2.csv', 'w+', newline ='')

    with file:
        write = csv.writer(file)
        write.writerows(final)

    read_file = pd.read_csv (r'Output/mHealth_subject2.csv', header = None)
    read_file.columns = ['acceleration from the chest sensor (X axis)','acceleration from the chest sensor (Y axis)',
    'acceleration from the chest sensor (Z axis)', 'electrocardiogram signal (lead 1)',
    'electrocardiogram signal (lead 2)', 'acceleration from the left-ankle sensor (X axis)',
    'acceleration from the left-ankle sensor (Y axis)', 'acceleration from the left-ankle sensor (Z axis)',
    'gyro from the left-ankle sensor (X axis)', 'gyro from the left-ankle sensor (Y axis)',
    'gyro from the left-ankle sensor (Z axis)', 'magnetometer from the left-ankle sensor (X axis)',
    'magnetometer from the left-ankle sensor (Y axis)', 'magnetometer from the left-ankle sensor (Z axis)',
    'acceleration from the right-lower-arm sensor (X axis)', 'acceleration from the right-lower-arm sensor (Y axis)',
    'acceleration from the right-lower-arm sensor (Z axis)', 'gyro from the right-lower-arm sensor (X axis)',
    'gyro from the right-lower-arm sensor (Y axis)', 'gyro from the right-lower-arm sensor (Z axis)',
    'magnetometer from the right-lower-arm sensor (X axis)', 'magnetometer from the right-lower-arm sensor (Y axis)',
    'magnetometer from the right-lower-arm sensor (Z axis)', 'Label (0 for the null class)']

    read_file.to_csv(r'Output/mHealth_subject2.csv', index=None)

parseFile()

df_without_0 = read_file[read_file['Label (0 for the null class)'] != 0]

df0 = read_file[read_file['Label (0 for the null class)'] == 0]
df1 = read_file[read_file['Label (0 for the null class)'] == 1]
df2 = read_file[read_file['Label (0 for the null class)'] == 2]
df3 = read_file[read_file['Label (0 for the null class)'] == 3]
df4 = read_file[read_file['Label (0 for the null class)'] == 4]
df5 = read_file[read_file['Label (0 for the null class)'] == 5]
df6 = read_file[read_file['Label (0 for the null class)'] == 6]
df7 = read_file[read_file['Label (0 for the null class)'] == 7]
df8 = read_file[read_file['Label (0 for the null class)'] == 8]
df9 = read_file[read_file['Label (0 for the null class)'] == 9]
df10 = read_file[read_file['Label (0 for the null class)'] == 10]
df11 = read_file[read_file['Label (0 for the null class)'] == 11]
df12 = read_file[read_file['Label (0 for the null class)'] == 12]

ax0 = df0[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax1 = df1[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax2 = df2[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax3 = df3[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax4 = df4[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax5 = df5[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax6 = df6[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax7 = df7[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax8 = df8[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax9 = df9[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax10 = df10[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax11 = df11[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

ax12 = df12[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='DarkBlue')

plt.show()

bar_df = pd.DataFrame({'label':['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], 'count': read_file['Label (0 for the null class)'].value_counts()})

ax = bar_df.plot.bar(x='label', y='count', rot=0)

plt.show()

bar_df = pd.DataFrame({'label':['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], 'count': df_without_0['Label (0 for the null class)'].value_counts()})

ax = bar_df.plot.bar(x='label', y='count', rot=0)

plt.show()

ax13 = df_without_0[['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)', 'Label (0 for the null class)']].plot.scatter(x='acceleration from the chest sensor (X axis)',
                      y='acceleration from the chest sensor (Y axis)',
                      c='Label (0 for the null class)',
                      colormap='viridis')

plt.show()

#The supervised learning classification

# Define X_iris and y_iris
def initializeMLData(df_without_0, data_columns):
    print('Data which ML algorithm is being run on.')
    print(data_columns)

    X_log_statistics = df_without_0[data_columns]
    y_log_statistics = df_without_0['Label (0 for the null class)']
    print('Activity types:', df_without_0['Label (0 for the null class)'].unique())

    # Normalize
    #scaler_logs = StandardScaler().fit(X_log_statistics)
    minmax_scaler_logs = MinMaxScaler().fit(X_log_statistics)
    #X_log_statistics = scaler_logs.transform(X_log_statistics)
    X_log_statistics = minmax_scaler_logs.transform(X_log_statistics)
    print('The length of X_log_statistics: {}'.format(len(X_log_statistics)))

    #plt.scatter(X_log_statistics[:,0], X_log_statistics[:,1], edgecolors='k', c=y_log_statistics)

    # Split in train and test sets
    X_train_logs, X_test_logs, y_train_logs, y_test_logs = train_test_split(X_log_statistics, y_log_statistics, test_size=0.25)
    print('Train shape:', X_train_logs.shape, y_train_logs.shape)
    print('Test shape:', X_test_logs.shape, y_test_logs.shape)

    #plt.scatter(X_train_logs[:,0], X_train_logs[:,1], edgecolors='k', c=y_train_logs)

    def plot_nearest_neighbors(X_train, X_test, Y_train, Y_test, k, classlabels, featurelabels, weight):
        print('Number of training points: ', X_train.size)
        possible_classes = Y_train.unique()

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
        distances, indices = nbrs.kneighbors(X_test)

        nb_of_classes = classlabels.unique().size

        h = .02  # step size in the mesh

        ## Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        ## we create an instance of Neighbours Classifier and fit the data.
        clf_data = KNeighborsClassifier(k, weights=weight)
        clf_data.fit(X_train, Y_train)

        ## Plot the decision boundary. For that, we will assign a color to each
        ## possible point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z_data = clf_data.predict(np.c_[xx.ravel(), yy.ravel()])

        ## Put the result into a color plot
        Z_data = Z_data.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z_data, cmap=cmap_light)

        for i in possible_classes:
            x1s_data = X_train[:, 0][Y_train.values == i]
            x2s_data = X_train[:, 1][Y_train.values == i]
            plt.scatter(x1s_data, x2s_data, cmap=cmap_bold, edgecolors='k', label=i)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("%i-Class classification (k = %i, weights = '%s')" % (nb_of_classes, k, weight))
        plt.xlabel(featurelabels[0])
        plt.ylabel(featurelabels[1])
        plt.legend()
        plt.show()

        return clf_data

    clf_activities = plot_nearest_neighbors(X_train_logs, X_test_logs, y_train_logs, y_test_logs, 5, df_without_0['Label (0 for the null class)'], data_columns, 'distance')

    #Evaluate performance (with test set)
    print('Number of test points: ',X_test_logs.size)
    y_pred_logs = clf_activities.predict(X_test_logs)

    #print(y_pred_logs)

    print("Accuracy score:")
    print(metrics.accuracy_score(y_test_logs, y_pred_logs))
    global accuracy_scores
    accuracy_scores.append(metrics.accuracy_score(y_test_logs, y_pred_logs))

    #Confusion Matrix

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_logs, y_pred_logs))

    #Dummy Classifier

    two_dimensional_values = df_without_0[data_columns]
    class_labels = df_without_0['Label (0 for the null class)']

    X = np.array(two_dimensional_values)
    y = np.array(class_labels)

    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(X,y)

    print(dummy_classifier.predict(X))
    print("most_frequent dummy classifier score:")
    print(dummy_classifier.score(X,y))

    new_dummy_classifier = DummyClassifier(strategy="stratified")
    new_dummy_classifier.fit(X,y)

    print(new_dummy_classifier.predict(X))
    print("stratified dummy classifier score:")
    print(new_dummy_classifier.score(X,y))

initializeMLData(df_without_0, ['acceleration from the chest sensor (X axis)', 'acceleration from the chest sensor (Y axis)'])
initializeMLData(df_without_0, ['acceleration from the left-ankle sensor (X axis)', 'acceleration from the left-ankle sensor (Y axis)'])
initializeMLData(df_without_0, ['acceleration from the right-lower-arm sensor (X axis)', 'acceleration from the right-lower-arm sensor (Y axis)'])
initializeMLData(df_without_0, ['gyro from the left-ankle sensor (X axis)', 'gyro from the left-ankle sensor (Y axis)'])
initializeMLData(df_without_0, ['gyro from the right-lower-arm sensor (X axis)', 'gyro from the right-lower-arm sensor (Y axis)'])
initializeMLData(df_without_0, ['magnetometer from the left-ankle sensor (X axis)', 'magnetometer from the left-ankle sensor (Y axis)'])
initializeMLData(df_without_0, ['magnetometer from the right-lower-arm sensor (X axis)', 'magnetometer from the right-lower-arm sensor (Y axis)'])

print(accuracy_scores)
