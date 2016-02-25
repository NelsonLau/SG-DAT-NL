import pandas as pd
import numpy as np
import time
import pdb

standard_periods_list = [1,5,10,20,40,80] #We'll use this set of periods repeatedly e.g., look at 1, 5, .., 80 minute returns, but also look at 1, 5, ..., 80 minute prior moves, etc
data_folder = "c:\\Users\\Nelson\\SG-DAT-NL\\Project\\data\\"


trying_something_new = True
if trying_something_new:
    load_from_save = True
    save_file = False
else:
    print "This will overwrite the pkl file, please check; ctrl-c to quit"
    time.sleep(3)
    print "Starting!"
    load_from_save = False
    save_file = True

def assess_column(dataframe, column_name, additional_check_list = None):
    #there is some intermittent error where a column of zeroes, or of all +inf or -inf is returned. We want to avoid that.
    passed = False
    if abs(dataframe[column_name]).max() > 0 and abs(dataframe[column_name]).min() < np.inf:
        if additional_check_list is not None:
            if 'range' in additional_check_list:
                if dataframe[column_name].min() == dataframe[column_name].max():
                    passed = False
                else:
                    passed = True
        else:
            passed = True
    else:
        passed = False
    if not passed:
        print "***retrying %s" % column_name
    return passed

#---Read data from xlsx
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
            successfully_made_column = assess_column(dataframe, column_name)
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
            successfully_made_column = assess_column(dataframe, column_name)

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
            successfully_made_column = assess_column(dataframe, column_name)

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

    #make volume column
    print "making rolling average volume feature columns"
    def make_volume_column(dataframe, mins_prior):
        column_name = '%s_min_volume' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:        
            dataframe[column_name] = pd.rolling_sum(dataframe.VOLUME,mins_prior, freq='T', min_periods = 1) / pd.rolling_count(dataframe.VOLUME,mins_prior, freq='T')
            successfully_made_column = assess_column(dataframe, column_name)

    for x in standard_periods_list:
        make_volume_column(one_minute_dataframe,x)

    #make buying-up-selling-down feature columns
    print "making buying-up-selling-down feature columns"
    one_minute_dataframe['busd'] = np.sign(one_minute_dataframe['1_min_prior']) * one_minute_dataframe.VOLUME
    one_minute_dataframe.loc[(one_minute_dataframe.index.hour == 11) & (one_minute_dataframe.index.minute == 45), 'busd'] = np.sign(one_minute_dataframe.LAST_PRICE - one_minute_dataframe.OPEN) * one_minute_dataframe.VOLUME #for the opening minute, use last-open rather than the usual last - previous last

    def make_busd_column(dataframe,mins_prior):
        column_name = '%s_min_busd' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:        
            dataframe[column_name] = pd.rolling_sum(dataframe.busd,mins_prior, freq='T', min_periods = 1) / pd.rolling_count(dataframe.busd,mins_prior, freq='T')
            successfully_made_column = assess_column(dataframe, column_name)

    for x in standard_periods_list:
        make_busd_column(one_minute_dataframe,x)

    print "making x-minute busd binary feature columns"
    for x in standard_periods_list:
        binarize_column(one_minute_dataframe,'%s_min_busd' % x, '%s_min_busd_updown' % x)        

    #make price relative to min-max range feature columns
    print "making min/max over last x minutes feature columns"
    def make_price_range_column(dataframe,mins_prior,price_type):
        assert (price_type == 'min' or price_type == 'max')
        column_name = '%s_min_%s' % (mins_prior, price_type)
        successfully_made_column = False
        while not successfully_made_column:        
            if price_type == 'min':
                dataframe[column_name] = pd.rolling_min(dataframe.LOW,mins_prior, freq='T', min_periods = 1)
            else:
                dataframe[column_name] = pd.rolling_max(dataframe.HIGH,mins_prior, freq='T', min_periods = 1)
            successfully_made_column = assess_column(dataframe, column_name)

    for x in standard_periods_list:
        for price_type in ['min','max']:
            make_price_range_column(one_minute_dataframe,x,price_type)

    #make x-min high-low range feature column
    print "making x-min high-low range feature columns"
    def make_high_low_range_column(dataframe,mins_prior):
        column_name = '%s_min_hilo' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:        
            dataframe[column_name] = dataframe['%s_min_max' % mins_prior] - dataframe['%s_min_min' % mins_prior] 
            successfully_made_column = assess_column(dataframe, column_name)

    for x in standard_periods_list:
        make_high_low_range_column(one_minute_dataframe,x)

    #make current price vs x-min high-low range (1-min ago) feature columns i.e., breakout signal
    print "making current price vs x-min high-low range (1-min ago) feature columns"
    def make_price_vs_range_column(dataframe,mins_prior):
        column_name = '%s_min_price_vs_range' % mins_prior
        successfully_made_column = False
        while not successfully_made_column:        
            dataframe[column_name] =  2.0 * (dataframe.LAST_PRICE - dataframe.resample('T',loffset='1min',how='last',fill_method='bfill')['%s_min_min' % x])/ (dataframe.resample('T',loffset='1min',how='last',fill_method='bfill')['%s_min_max' % x] - dataframe.resample('T',loffset='1min',how='last',fill_method='bfill')['%s_min_min' % x]) - 1.0 #we x2 and then -1 so it becomes centered around 0, and -1 means at the lowest in range, +1 means at the highest. You could be outside [-1, 1], that means it's a breakout
            successfully_made_column = assess_column(dataframe, column_name, ['range'])             

    for x in standard_periods_list:
        make_price_vs_range_column(one_minute_dataframe,x)

else:
    load_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
    one_minute_dataframe = pd.read_pickle(load_file)




#---pickle
if save_file:
    pickle_file = data_folder + 'nifty_one_minute_checkpoint.pkl'
    one_minute_dataframe.to_pickle(pickle_file)


#---Write output file
output_file_name = "nifty_one_minute.csv"
output_file_path = data_folder + output_file_name
print "writing %s" % output_file_path
one_minute_dataframe.to_csv(output_file_path)