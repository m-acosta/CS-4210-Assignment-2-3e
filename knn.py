#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 50 minutes
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

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    X = db.copy() #make a copy of file read in
    # remove the class value
    for line in X:
       for feature in line:
          feature = float(feature)
    # remove the instance used for test
    testSample = X.pop(i)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    ClassValue = {
       '-': float(1),
       '+': float(2),
    }
    Y = [0] * len(X) #create empty list
    for i, row in enumerate(X):
        Y[i] = ClassValue[row[2]] #transform label based on dictionary

    #store the test sample of this iteration in the vector testSample
    testSample[2] = ClassValue[testSample[2]] #transform label based on dictionary

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

#print the error rate
#--> add your Python code here