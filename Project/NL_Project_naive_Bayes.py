import pandas as pd
import numpy as np
import copy
import pdb

#load data
data_folder = "c:\\Users\\Nelson\\SG-DAT-NL\\Project\\data\\"
pickle_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
print "loading %s" % pickle_file
one_minute_dataframe = pd.read_pickle(pickle_file)

print one_minute_dataframe.columns
pdb.set_trace()

standard_periods_list = [1,5,10,20,40,80]

#---Bernoulli NB
#prepare X and y
feature_columns = []
response_column = '5_min_updown'
for period in standard_periods_list:
    feature_columns.append('%s_min_prior_updown' % period)
data_columns = copy.copy(feature_columns)
data_columns.append(response_column)
dataset = one_minute_dataframe[data_columns]
dataset.dropna(inplace = True)
X = dataset[feature_columns]
y = dataset[response_column]

from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y)

def accuracy_report(_clf):
    print "-----"
    print "Accuracy: %0.2f%%" % (100 * _clf.score(X, y))

    #Print the accuracy on the test and training dataset
    training_accuracy = _clf.score(xtrain, ytrain)
    test_accuracy = _clf.score(xtest, ytest)

    print "Accuracy on training data: %0.2f%%" % (100 * training_accuracy)
    print "Accuracy on test data: %0.2f%%" % (100 * test_accuracy)

def make_predictions(_clf):
    y_pred = _clf.predict(X)
    training_y_pred = _clf.predict(xtrain)
    testing_y_pred = _clf.predict(xtest)
    return y_pred, training_y_pred, testing_y_pred

from sklearn.metrics import accuracy_score
def report_accuracy_score(_clf):
    y_pred, training_y_pred, testing_y_pred = make_predictions(_clf)
    print "-----"
    print "Accuracy score: %0.2f%%" % (100 * accuracy_score(y, y_pred))
    print "Accuracy score on training data: %0.2f%%" % (100 * accuracy_score(ytrain, training_y_pred))
    print "Accuracy score on test data: %0.2f%%" % (100 * accuracy_score(ytest, testing_y_pred))

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import confusion_matrix
def report_confusion_matrix(_clf, plot_cm = False):
    y_pred, training_y_pred, testing_y_pred = make_predictions(_clf)
    print "-----"
    print "Confusion matrix (on entire data):"
    cm = confusion_matrix(y, y_pred)
    print(cm)
    if plot_cm:
        plt.figure()
        plot_confusion_matrix(cm)
        plt.show()

from sklearn.metrics import roc_curve, auc
def report_roc_curve(_clf, clf_name = None):
    y_pred, training_y_pred, testing_y_pred = make_predictions(_clf)
    print "-----"
    print "ROC (on entire data)"
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for %s' % clf_name)
    plt.legend(loc="lower right")
    plt.show()    


from sklearn.naive_bayes import BernoulliNB
print "\n>>>BernoulliNB:"
clf_b = BernoulliNB().fit(xtrain, ytrain)
accuracy_report(clf_b)
report_accuracy_score(clf_b)
report_confusion_matrix(clf_b)
report_roc_curve(clf_b, 'BernoulliNB')

#---Logistic Regression: binary features
from sklearn.linear_model import LogisticRegression
print "\n>>>Logistic Regression with binary features:"
clf_lrb = LogisticRegression().fit(xtrain, ytrain)
accuracy_report(clf_lrb)
report_accuracy_score(clf_lrb)
report_confusion_matrix(clf_lrb)
report_roc_curve(clf_lrb, 'Logistic Regression with binary features')

#---Skipping MultinomialNB as we don't really have count data features

#---Gaussian NB
feature_columns = []
response_column = '5_min_updown'
for period in standard_periods_list:
    feature_columns.append('%s_min_prior' % period)
data_columns = copy.copy(feature_columns)
data_columns.append(response_column)
dataset = one_minute_dataframe[data_columns]
dataset.dropna(inplace = True)
X = dataset[feature_columns]
y = dataset[response_column]

xtrain, xtest, ytrain, ytest = train_test_split(X, y)

from sklearn.naive_bayes import GaussianNB
print "\n>>>GaussianNB:"
clf_g = GaussianNB().fit(xtrain, ytrain)
accuracy_report(clf_g)
report_accuracy_score(clf_g)
report_confusion_matrix(clf_g)
report_roc_curve(clf_g, 'GaussianNB')

#---Logistic Regression: continuous features
print "\n>>>Logistic Regression with continuous features:"
clf_lrc = LogisticRegression().fit(xtrain, ytrain)
accuracy_report(clf_lrc)
report_accuracy_score(clf_lrc)
report_confusion_matrix(clf_lrc)
report_roc_curve(clf_lrc, 'Logistic Regression with continuous features')

