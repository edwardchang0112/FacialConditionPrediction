import numpy as np
import csv
import pandas as pd
import datetime

def readcsv(file_name, factor):
    dataset = pd.read_csv(file_name)
    return dataset[factor]

def moving_window(ts_data, window_size):
    ts_data_win = []
    for i in range(len(ts_data)-window_size+1):
        ts_data_win.append(ts_data[i:i+window_size])
    return ts_data_win

def data_rolling(ts_data, window_size):
    ts_data_arr = np.asarray(ts_data)
    ts_data_rol = moving_window(ts_data_arr, window_size)
    return ts_data_rol

def date_diff(ts_date_win):
    date_diff_list = []
    for date_row in ts_date_win:
        #print("date_row = ", date_row)
        date_a = datetime.date(int(str(date_row[-1])[:4]), int(str(date_row[-1])[5:7]), int(str(date_row[-1])[8:10]))
        date_diff_sublist = []
        for date in date_row:
            date_diff_sublist.append(0.1*(date_a - datetime.date(int(str(date)[:4]), int(str(date)[5:7]), int(str(date)[8:10]))).days)
        date_diff_list.append(date_diff_sublist[:-1])
    date_diff_list = np.stack((date_diff_list), axis=0)
    return date_diff_list

def input_output_gen(mode, ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_avg_win, ts_skinhealth_avg_win,
                     ts_temperature, ts_humidity, ts_hydration, ts_skinhealth, age_win, window_size, ts_skincare_ratio_win):
    date_diff_list = date_diff(ts_date_win)
    X_skin = np.hstack((ts_temp_win[:-1], ts_humid_win[:-1], ts_temp_win[1:], ts_humid_win[1:], ts_hydration_avg_win[:-1], ts_skinhealth_avg_win[:-1]))
    X_skin = np.hstack((date_diff_list, X_skin, np.vstack((ts_skincare_ratio_win[:-1][:, -1]*10)))) # only use the latest term of skincare ratio
    
    Y = np.stack((ts_hydration[window_size:])).reshape(-1, 1)
    #Y4 = np.stack((ts_skinhealth[window_size:])).reshape(-1, 1)
    return X_skin, Y

def input_normalization(input):
    input_nor = input / 100
    bias = np.stack((0.1*np.ones((len(input_nor), 1))), axis=0)
    input_nor_bias = np.hstack((input_nor, bias))
    return input_nor_bias

def data_load1(file_name, mssql_data):
    window_size = 1
    if file_name != None:
        ts_date = readcsv(file_name, 'Date')
        ts_temperature = readcsv(file_name, 'Temperature')
        ts_humidity = readcsv(file_name, 'Humidity')
        ts_hydration = readcsv(file_name, 'Avg3_Hydration')
        ts_hydration_diff = np.asarray(ts_hydration)[1:] - np.asarray(ts_hydration)[:-1]
        ts_skinhealth = readcsv(file_name, 'Avg3_Skinhealth')
        ts_skinhealth_diff = np.asarray(ts_skinhealth)[1:] - np.asarray(ts_skinhealth)[:-1]

    ts_date_win = np.stack((data_rolling(ts_date, window_size + 1)), axis=0)
    date_diff_list = date_diff(ts_date_win)

    ts_temperature_arr = np.vstack((ts_temperature))
    ts_humidity_arr = np.vstack((ts_humidity))
    ts_hydration_arr = np.vstack((ts_hydration))
    ts_hydration_diff_arr = np.vstack((ts_hydration_diff))
    ts_skinhealth_arr = np.vstack((ts_skinhealth))
    ts_skinhealth_diff_arr = np.vstack((ts_skinhealth_diff))

    original_data = np.hstack((date_diff_list, ts_temperature_arr[1:], ts_humidity_arr[1:], ts_hydration_arr[:-1], ts_hydration_diff_arr,
                          ts_skinhealth_arr[:-1], ts_skinhealth_diff_arr))
    return original_data

def data_load(file_name, window_size, mssql_data, anomaly_filter_model):
    print("file_name = ", file_name)
    window_size_1 = 1 # for anomaly filter
    if file_name != None:
        age = readcsv(file_name, 'Age')
        ts_date = readcsv(file_name, 'Date')
        ts_temperature = readcsv(file_name, 'Temperature')
        ts_humidity = readcsv(file_name, 'Humidity')
        ts_hydration = readcsv(file_name, 'Avg3_Hydration')
        ts_hydration_diff = np.asarray(ts_hydration)[1:] - np.asarray(ts_hydration)[:-1]
        ts_skinhealth = readcsv(file_name, 'Avg3_Skinhealth')
        ts_skinhealth_diff = np.asarray(ts_skinhealth)[1:] - np.asarray(ts_skinhealth)[:-1]
        ts_skincare_ratio = readcsv(file_name, 'Skincare_Ratio')

        ts_date_win = np.stack((data_rolling(ts_date, window_size_1 + 1)), axis=0)
        date_diff_list = date_diff(ts_date_win)

        ts_temperature_arr = np.vstack((ts_temperature))
        ts_humidity_arr = np.vstack((ts_humidity))
        ts_hydration_arr = np.vstack((ts_hydration))
        ts_hydration_diff_arr = np.vstack((ts_hydration_diff))
        ts_skinhealth_arr = np.vstack((ts_skinhealth))
        ts_skinhealth_diff_arr = np.vstack((ts_skinhealth_diff))

        original_data = np.hstack((date_diff_list, ts_temperature_arr[1:], ts_humidity_arr[1:], ts_hydration_arr[:-1],
                                   ts_hydration_diff_arr, ts_skinhealth_arr[:-1], ts_skinhealth_diff_arr))

        ''' anomaly data filtered by Isolation Forest '''
        for i in range(len(original_data)):
            anomaly = anomaly_filter_model.predict(original_data[i].reshape(1, -1))
            if anomaly == -1:
                # use drop in pandas dataframe will not affect the order issue of dropping input data one by one
                age = age.drop([i+1])
                ts_date = ts_date.drop([i+1])
                ts_temperature = ts_temperature.drop([i+1])
                ts_humidity = ts_humidity.drop([i+1])
                ts_hydration = ts_hydration.drop([i+1])
                ts_skinhealth = ts_skinhealth.drop([i+1])
                ts_skincare_ratio = ts_skincare_ratio.drop([i+1])
    else:
        pass
        '''
        # for MSSQL use
        ts_date = mssql_data[:, 0]
        ts_temperature = mssql_data[:, 1]
        ts_humidity = mssql_data[:, 2]
        ts_hydration = mssql_data[:, 3]
        ts_skinhealth = mssql_data[:, 4]
        '''

    ts_date_win = np.stack((data_rolling(ts_date, window_size + 1)), axis=0)
    ts_temp_win = np.stack((data_rolling(ts_temperature, window_size)), axis=0)
    ts_humid_win = np.stack((data_rolling(ts_humidity, window_size)), axis=0)
    ts_hydration_win = np.stack((data_rolling(ts_hydration, window_size)), axis=0)
    ts_skinhealth_win = np.stack((data_rolling(ts_skinhealth, window_size)), axis=0)
    age_win = np.stack((data_rolling(age, window_size)), axis=0)
    ts_skincare_ratio_win = np.stack((data_rolling(ts_skincare_ratio, window_size)), axis=0)

    return ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_win, ts_skinhealth_win, np.asarray(ts_temperature).reshape(-1, 1), \
               np.asarray(ts_humidity).reshape(-1, 1), np.asarray(ts_hydration).reshape(-1, 1), \
               np.asarray(ts_skinhealth).reshape(-1, 1), age_win, ts_skincare_ratio_win

def store_csv(data, filename):
    with open(filename+'data'+str(datetime.datetime.now().strftime("%Y_%m_%d"))+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(data)

