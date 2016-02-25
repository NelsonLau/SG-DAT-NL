import pandas as pd
import numpy as np
import copy
import random
import pdb
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

#The silhouette score is slow to compute for large datasets, but seems relatively stable across samples, so we can sample it an take an average
SILHOUETTE_SAMPLE_SIZE = 1000
SILHOUETTE_TRIES = 10
MAX_CLUSTERS = 5

#load data
data_folder = "c:\\Users\\Nelson\\SG-DAT-NL\\Project\\data\\"
pickle_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
print "loading %s" % pickle_file
one_minute_dataframe = pd.read_pickle(pickle_file)

total_predictors_list = ['NUMBER_TICKS', 'DayOfWeek', 'Hour', '1_min_prior', '5_min_prior', '10_min_prior', '20_min_prior', '40_min_prior', '80_min_prior', '1_min_prior_updown', '5_min_prior_updown', '10_min_prior_updown', '20_min_prior_updown', '40_min_prior_updown', '80_min_prior_updown', '1_last_vs_ma', '5_last_vs_ma', '10_last_vs_ma', '20_last_vs_ma', '40_last_vs_ma', '80_last_vs_ma', '1_last_vs_ma_updown', '5_last_vs_ma_updown', '10_last_vs_ma_updown', '20_last_vs_ma_updown', '40_last_vs_ma_updown', '80_last_vs_ma_updown', '1_min_volume', '5_min_volume', '10_min_volume', '20_min_volume', '40_min_volume', '80_min_volume', '1_min_busd', '5_min_busd', '10_min_busd', '20_min_busd', '40_min_busd', '80_min_busd', '1_min_busd_updown', '5_min_busd_updown', '10_min_busd_updown', '20_min_busd_updown', '40_min_busd_updown', '80_min_busd_updown', '1_min_hilo', '5_min_hilo', '10_min_hilo', '20_min_hilo', '40_min_hilo', '80_min_hilo', '1_min_price_vs_range', '5_min_price_vs_range', '10_min_price_vs_range', '20_min_price_vs_range', '40_min_price_vs_range', '80_min_price_vs_range']

print "---First we try to find best clustering factors on a standalone basis, we'll use what we know about the variables to whittle down the set after that"
predictor_performance = {}
for predictor in total_predictors_list:
    try:
        print ">>>trying %s" % predictor
        print "copying dataframe"
        X = pd.DataFrame.copy(one_minute_dataframe[[predictor]])
        X.dropna(inplace=True)

        print "analyzing"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        performance_by_num_clusters = []
        for num_clusters in range(2,MAX_CLUSTERS+1):
            print "Trying %s cluster solution" % num_clusters
            km = KMeans(n_clusters=num_clusters, random_state=1)
            km.fit(X_scaled)
            X['km_cluster'] = km.labels_

            silhouette_list = []
            print "silhouette calculation (total %s):" % SILHOUETTE_TRIES
            for tries in range(SILHOUETTE_TRIES):
                try:
                    print " %s " % str(tries + 1),
                    sample_list = random.sample(range(len(X_scaled)),SILHOUETTE_SAMPLE_SIZE)
                    sample_list.sort
                    X_scaled_sample = X_scaled[sample_list]
                    km_labels_sample = km.labels_[sample_list]
                    silhouette_list.append(metrics.silhouette_score(X_scaled_sample, km_labels_sample))
                except:
                    print "F",
            print ""
            performance_by_num_clusters.append(np.mean(silhouette_list))
        predictor_performance[predictor] = np.max(performance_by_num_clusters)
    except:
        print "***Failed for %s" % predictor

print predictor_performance
sorted_predictor_performance = sorted(predictor_performance.items(), key = operator.itemgetter(1), reverse = True)
for predictor, performance in sorted_predictor_performance:
    print "%s\t%0.5f" % (predictor, performance)