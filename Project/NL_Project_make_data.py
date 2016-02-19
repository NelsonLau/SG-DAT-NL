import pandas as pd
import numpy as np
import pdb

standard_periods_list = [1,5,10,20,40,80] #We'll use this set of periods repeatedly e.g., look at 1, 5, .., 80 minute returns, but also look at 1, 5, ..., 80 minute prior moves, etc
data_folder = "c:\\Users\\Nelson\\SG-DAT-NL\\Project\\data\\"

#---Read data from xlsx
load_from_save = False
if not load_from_save:
    data_file = "nifty 1 min bars.xlsx"
    data_file_path = data_folder + data_file
    print "reading %s" % data_file_path
    one_minute_dataframe = pd.read_excel(data_file_path,sheetname='Sheet1',header=2,parse_cols="A:G")
    daily_dataframe = pd.read_excel(data_file_path,sheetname='Sheet2',header=1,parse_cols="A:G")

    #---Add futures ticker to one_minute_dataframe
    print "adding futures ticker"
    one_minute_dataframe['date_only'] = one_minute_dataframe.Date.dt.to_period('D')
    daily_dataframe['date_only'] = daily_dataframe.Date.dt.to_period('D')
    one_minute_dataframe = pd.merge(one_minute_dataframe, daily_dataframe[['FUT_CUR_GEN_TICKER', 'date_only']], on='date_only', how='left')
    del one_minute_dataframe['date_only'] #we don't need this column

    #---Add DayOfWeek & Hour as columns (possible features or interaction features)
    print "adding DayOfWeek & Hour"
    one_minute_dataframe['DayOfWeek'] = one_minute_dataframe.Date.dt.dayofweek
    one_minute_dataframe['Hour'] = one_minute_dataframe.Date.dt.hour

    #---Set a time-series index on one_minute_dataframe
    print "setting time-series index"
    one_minute_dataframe.set_index('Date', inplace = True)

    #---Make response columns
    #Make x-minute-ahead return response columns
    print "making x-minute-ahead return response columns"
    def make_return_column(dataframe,mins_ahead,threshold):
        column_name = '%s_min_return' % mins_ahead
        successfully_made_column = False
        while not successfully_made_column:
            dataframe[column_name] = dataframe.resample('T',loffset='-%smin' % mins_ahead,fill_method='ffill',limit=mins_ahead - 1).LAST_PRICE.pct_change(periods=mins_ahead)
            if abs(dataframe[column_name]).max()  > 0:
                successfully_made_column = True #there is some intermittent error where a column of zeroes is returned. We want to avoid that.
            else:
                print "***retrying %s" % column_name
        dataframe.loc[abs(dataframe[column_name]) < threshold,column_name] = 0

    tick_size = 0.05
    max_price = one_minute_dataframe.LAST_PRICE.max()
    threshold = tick_size / max_price / 10.0 #returns of abs value smaller than this are just float precision issues
    for mins_ahead in standard_periods_list:
        make_return_column(one_minute_dataframe, mins_ahead, threshold)

    #Make x-minute-ahead binary responses
    def binarize_column(dataframe,column_name,binary_column_name):
        dataframe[binary_column_name] = dataframe[column_name][abs(dataframe[column_name]) > 0].apply(lambda x: 1 if x > 0 else 0)

    print "making x-minute-ahead binary response columns"
    for x in standard_periods_list:
        binarize_column(one_minute_dataframe,'%s_min_return' % x, '%s_min_updown' % x)

    #Make x-minute-ahead categorical responses
    print "making x-minute-ahead categorical response columns"
    percentile = 1.0/3.0 #we want categories 0,1,2 - dividng the returns into "significantly -ve, close to 0, significantly +ve"
    for x in standard_periods_list:
        cutoff = one_minute_dataframe['%s_min_return' % x].abs().quantile(percentile)
        bins = [one_minute_dataframe['%s_min_return' % x].min(),-cutoff,cutoff,one_minute_dataframe['%s_min_return' % x].max()]
        print "\t bins are %s" % bins
        one_minute_dataframe['%s_min_cat' % x] = pd.cut(one_minute_dataframe['%s_min_return' % x], bins, labels=False)    

    #---Make feature columns
    #Make x-minute prior return feature columns
    print "making x-minute prior return feature columns"
    def make_prior_column(dataframe,mins_prior):
        column_name = '%s_min_prior' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:
            dataframe[column_name] = -dataframe.resample('T',loffset='%smin' % mins_prior,how='last',fill_method='bfill',limit=mins_prior).LAST_PRICE.pct_change(periods=-mins_prior)
            if abs(dataframe[column_name]).max()  > 0:
                successfully_made_column = True #there is some intermittent error where a column of zeroes is returned. We want to avoid that.
            else:
                print "***retrying %s" % column_name

    for x in standard_periods_list:
        make_prior_column(one_minute_dataframe,x)

    print "making x-minute prior return feature columns"
    for x in standard_periods_list:
        binarize_column(one_minute_dataframe,'%s_min_prior' % x, '%s_min_prior_updown' % x)

    #make ma
    print "making x-minute moving average feature columns"
    def make_ma_column(dataframe,mins_prior):
        column_name = '%s_min_ma' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:
            dataframe[column_name] = pd.rolling_sum(dataframe.LAST_PRICE,mins_prior,freq='T',min_periods=1) / pd.rolling_count(dataframe.LAST_PRICE,mins_prior,freq='T')
            if abs(dataframe[column_name]).max()  > 0:
                successfully_made_column = True #there is some intermittent error where a column of zeroes is returned. We want to avoid that.
            else:
                print "***retrying %s" % column_name

    for x in standard_periods_list:
        make_ma_column(one_minute_dataframe,x)

    print "making x-minute difference versus moving average feature columns"
    def make_lvma_column(dataframe,mins_prior):
        column_name = '%s_last_vs_ma' % mins_prior
        dataframe[column_name] = dataframe.LAST_PRICE - dataframe['%s_min_ma' % mins_prior]

    for x in standard_periods_list:
        make_lvma_column(one_minute_dataframe,x)

    print "making x-minute difference versus moving average feature columns"
    for x in standard_periods_list:
        binarize_column(one_minute_dataframe,'%s_last_vs_ma' % x, '%s_last_vs_ma_updown' % x)

    #make buying-up-selling-down column
    print "making buying-up-selling-down feature columns"
    one_minute_dataframe['busd'] = np.sign(one_minute_dataframe['1_min_prior']) * one_minute_dataframe.VOLUME
    one_minute_dataframe.loc[(one_minute_dataframe.index.hour == 11) & (one_minute_dataframe.index.minute == 45), 'busd'] = np.sign(one_minute_dataframe.LAST_PRICE - one_minute_dataframe.OPEN) * one_minute_dataframe.VOLUME #for the opening minute, use last-open rather than the usual last - previous last

    def make_busd_column(dataframe,mins_prior):
        column_name = '%s_min_busd' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:        
            dataframe[column_name] = pd.rolling_sum(dataframe.busd,mins_prior, freq='T', min_periods = 1) / pd.rolling_count(dataframe.busd,mins_prior, freq='T')
            if abs(dataframe[column_name]).max()  > 0:
                successfully_made_column = True #there is some intermittent error where a column of zeroes is returned. We want to avoid that.
            else:
                print "***retrying %s" % column_name

    for x in standard_periods_list:
        make_busd_column(one_minute_dataframe,x)

    print "making x-minute busd binary feature columns"
    for x in standard_periods_list:
        binarize_column(one_minute_dataframe,'%s_min_busd' % x, '%s_min_busd_updown' % x)        

else:
    load_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
    one_minute_dataframe = pd.read_pickle(load_file)


#---pickle
save_file = True
if save_file:
    pickle_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
    one_minute_dataframe.to_pickle(pickle_file)

#---Write output file
output_file_name = "nifty_one_minute.csv"
output_file_path = data_folder + output_file_name
print "writing %s" % output_file_path
one_minute_dataframe.to_csv(output_file_path)