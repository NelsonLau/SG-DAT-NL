import pandas as pd
import copy
import time
import sys
import itertools
import pdb
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

SVM_FRAC = 0.1
outfile = 'svm_output.txt'
outputfile = open(outfile,'a')

def make_predictions(_clf):
    y_pred = _clf.predict(X)
    training_y_pred = _clf.predict(xtrain)
    testing_y_pred = _clf.predict(xtest)
    return y_pred, training_y_pred, testing_y_pred

def report_accuracy_score(_clf):
    y_pred, training_y_pred, testing_y_pred = make_predictions(_clf)
    print "-----"
    print "Accuracy score: %0.2f%%" % (100 * accuracy_score(y, y_pred))
    print "Accuracy score on training data: %0.2f%%" % (100 * accuracy_score(ytrain, training_y_pred))
    print "Accuracy score on test data: %0.2f%%" % (100 * accuracy_score(ytest, testing_y_pred))
    outputfile.write("Accuracy score: %0.2f%%\n" % (100 * accuracy_score(y, y_pred)))
    outputfile.write("Accuracy score on training data: %0.2f%%\n" % (100 * accuracy_score(ytrain, training_y_pred)))
    outputfile.write("Accuracy score on test data: %0.2f%%\n" % (100 * accuracy_score(ytest, testing_y_pred)))

#load data
data_folder = "c:\\Users\\Nelson\\SG-DAT-NL\\Project\\data\\"
pickle_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
print "loading %s" % pickle_file
one_minute_dataframe = pd.read_pickle(pickle_file)

# total_predictors_list = ['NUMBER_TICKS', 'DayOfWeek', 'Hour', '1_min_prior', '5_min_prior', '10_min_prior', '20_min_prior', '40_min_prior', '80_min_prior', '1_min_prior_updown', '5_min_prior_updown', '10_min_prior_updown', '20_min_prior_updown', '40_min_prior_updown', '80_min_prior_updown', '1_last_vs_ma', '5_last_vs_ma', '10_last_vs_ma', '20_last_vs_ma', '40_last_vs_ma', '80_last_vs_ma', '1_last_vs_ma_updown', '5_last_vs_ma_updown', '10_last_vs_ma_updown', '20_last_vs_ma_updown', '40_last_vs_ma_updown', '80_last_vs_ma_updown', '1_min_volume', '5_min_volume', '10_min_volume', '20_min_volume', '40_min_volume', '80_min_volume', '1_min_busd', '5_min_busd', '10_min_busd', '20_min_busd', '40_min_busd', '80_min_busd', '1_min_busd_updown', '5_min_busd_updown', '10_min_busd_updown', '20_min_busd_updown', '40_min_busd_updown', '80_min_busd_updown', '1_min_hilo', '5_min_hilo', '10_min_hilo', '20_min_hilo', '40_min_hilo', '80_min_hilo', '1_min_price_vs_range', '5_min_price_vs_range', '10_min_price_vs_range', '20_min_price_vs_range', '40_min_price_vs_range', '80_min_price_vs_range']

total_predictors_list = ['NUMBER_TICKS', 'DayOfWeek', 'Hour', '1_min_prior', '5_min_prior', '10_min_prior', '20_min_prior', '40_min_prior', '80_min_prior', '1_min_prior_updown', '5_min_prior_updown', '10_min_prior_updown', '20_min_prior_updown', '40_min_prior_updown', '80_min_prior_updown', '1_last_vs_ma', '5_last_vs_ma', '10_last_vs_ma', '20_last_vs_ma', '40_last_vs_ma', '80_last_vs_ma', '1_last_vs_ma_updown', '5_last_vs_ma_updown', '10_last_vs_ma_updown', '20_last_vs_ma_updown', '40_last_vs_ma_updown', '80_last_vs_ma_updown', '5_min_volume', '10_min_volume', '20_min_volume', '40_min_volume', '80_min_volume', '1_min_busd', '5_min_busd', '10_min_busd', '20_min_busd', '40_min_busd', '80_min_busd', '1_min_busd_updown', '5_min_busd_updown', '10_min_busd_updown', '20_min_busd_updown', '40_min_busd_updown', '80_min_busd_updown', '1_min_hilo', '5_min_hilo', '10_min_hilo', '20_min_hilo', '40_min_hilo', '80_min_hilo', '1_min_price_vs_range', '5_min_price_vs_range', '10_min_price_vs_range', '20_min_price_vs_range', '40_min_price_vs_range', '80_min_price_vs_range'] #1 min volume gets stuck... create timeout?


response_list = ['1_min_updown', '5_min_updown', '10_min_updown','20_min_updown', '40_min_updown', '80_min_updown','1_min_cat', '5_min_cat', '10_min_cat','20_min_cat', '40_min_cat', '80_min_cat']

two_list = list(itertools.combinations(total_predictors_list,2))
three_list = list(itertools.combinations(total_predictors_list,3))
predictor_combo_list = two_list + three_list

for response in response_list:
    for predictor_combo in predictor_combo_list:
        predictor_combo = list(predictor_combo)
        title_string = "\n---Working on %s %s\n" % (",".join(predictor_combo), response) 
        print title_string
        outputfile.write(title_string)
        all_columns_list = copy.copy(predictor_combo)
        all_columns_list.append(response)

        print "copying dataframe"
        total_dataframe = pd.DataFrame.copy(one_minute_dataframe[all_columns_list])
        total_dataframe.dropna(inplace=True)
        sample_dataframe = total_dataframe.sample(frac=SVM_FRAC)
        X = sample_dataframe[predictor_combo]
        y = sample_dataframe[response]
        xtrain, xtest, ytrain, ytest = train_test_split(X, y)

        print "fitting Linear SVM"
        outputfile.write("fitting Linear SVM\n")
        clf_linear = SVC(kernel='linear')
        clf_linear.fit(xtrain, ytrain)
        print "scoring"
        report_accuracy_score(clf_linear)

        print "fitting RBF SVM"
        outputfile.write("fitting RBF SVM\n")
        clf_linear = SVC(kernel='rbf')
        clf_linear.fit(xtrain, ytrain)
        print "scoring"
        report_accuracy_score(clf_linear)

outputfile.close()