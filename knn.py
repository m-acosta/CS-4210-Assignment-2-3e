#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: knn.py
# SPECIFICATION: Take a set of labeled points, and using leave-one-out 
#               cross validation, find the error rate of the KNN algorithm.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

correct_predictions = 0
incorrect_predictions = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    X = [0] * len(db)
    for counter, point in enumerate(db):
       X[counter] = [float(db[counter][0]), float(db[counter][1])]
    # remove the instance used for test
    X.pop(i)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    ClassValue = {
       '-': 1.0,
       '+': 2.0,
    }
    Y = [0] * len(db) #create empty list
    for j, row in enumerate(db):
        Y[j] = ClassValue[row[2]] #transform label based on dictionary
    # remove the instance used for test
    Y.pop(i)

    #store the test sample of this iteration in the vector testSample
    testSample = instance.copy()
    testSample[2] = ClassValue[testSample[2]] #transform label based on dictionary
    testSample = [float(i) for i in testSample]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([[testSample[0], testSample[1]]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted == testSample[2]:
       correct_predictions += 1
    else:
       incorrect_predictions += 1

#print the error rate
error_rate = incorrect_predictions / (correct_predictions + incorrect_predictions)
print("Error rate is " + str(error_rate))