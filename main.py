import numpy as np
import xgboost as xgb
from Data_Preprocessing import input_output_gen, data_load, store_csv, input_normalization, data_load1
import time
from Load_BatchData_txt import Read_batch_files_fromtxt
from Anomaly_filter import build_iForest_model, iForest_anomaly_detection

if __name__ == '__main__':
    ''' Load some real data '''
    file_path = 'P2_Use_Skincare/Use_Skincare/'
    txtfile_name = 'User_Files.txt'
    file_list = Read_batch_files_fromtxt(txtfile_name)
    print("file_list = ", file_list)

    ''' K-fold cross validation '''
    window_size = 3
    cross_error = []
    Lin_combine_cross_error = []
    XGB_model_combine_cross_error = []
    NN_cross_error = []
    start_time = time.time()
    All_feature_importance_all = []
    Combine_pred_x_data = []
    Combine_pred_y_data = []

    '''Anomaly data filter by IsolationForest'''
    original_data_all = []
    for file in file_list:
        print("file = ", file)
        original_data = data_load1(str(file_path)+file, None)

        original_data_all.append(original_data.tolist())
        #print("original_data = ", original_data)

    original_data_all = np.vstack((original_data_all))
    iForest_model_fit = build_iForest_model(original_data_all)

    for file in file_list:
        print("test_file = ", file)
        test_X_all = []
        test_Y_all = []
        test_X_skin_all = []
        test_Y_skin_all = []
        test_X_phyInfo_all = []
        test_Y_phyInfo_all = []

        ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_win, ts_skinhealth_win, \
        ts_temperature_arr, ts_humidity_arr, ts_hydration_arr, ts_skinhealth_arr, age_win, ts_skincare_ratio_win = data_load(str(file_path)+file, window_size, None, iForest_model_fit)

        mode = 'test'
        X_skin_test, Y_skin_test = input_output_gen(mode, ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_win, ts_skinhealth_win, ts_temperature_arr,
                                ts_humidity_arr, ts_hydration_arr, ts_skinhealth_arr, age_win, window_size, ts_skincare_ratio_win)

        test_X_skin_all.append(X_skin_test.tolist())
        test_Y_skin_all.append(Y_skin_test.tolist())
        
        file_list_copy = file_list.copy()
        file_list_copy.remove(file)

        train_X_all = []
        train_Y_all = []

        train_X_skin_all = []
        train_Y_skin_all = []
        train_X_phyInfo_all = []
        train_Y_phyInfo_all = []

        for train_file in file_list_copy:  # for taining
            print("train_file = ", train_file)
            Age_index = file_list.index(train_file)
            ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_avg_win, \
            ts_skinhealth_win, ts_temperature_arr, ts_humidity_arr, \
            ts_hydration_avg_arr, ts_skinhealth_avg_arr, age_win, ts_skincare_ratio_win = data_load(str(file_path)+
            train_file, window_size, None, iForest_model_fit)

            mode = 'train'
            X_skin_train, Y_skin_train = input_output_gen(mode, ts_date_win, ts_temp_win, ts_humid_win, ts_hydration_avg_win,
            ts_skinhealth_win, ts_temperature_arr, ts_humidity_arr, ts_hydration_avg_arr, ts_skinhealth_avg_arr, age_win, window_size, ts_skincare_ratio_win)

            train_X_skin_all.append(X_skin_train.tolist())
            train_Y_skin_all.append(Y_skin_train.tolist())
            
        test_X_skin_all = np.vstack((test_X_skin_all))
        test_Y_skin_all = np.vstack((test_Y_skin_all))

        train_X_skin_all = np.vstack((train_X_skin_all))
        train_Y_skin_all = np.vstack((train_Y_skin_all))
        
        '''Data normalization'''
        print("train_X_skin_all = ", train_X_skin_all)
        train_x_skin = input_normalization(train_X_skin_all)

        train_y_skin = train_Y_skin_all / 100
        test_x_skin = input_normalization(test_X_skin_all)
        test_y_skin = test_Y_skin_all / 100

        XGB_skin_model_global = xgb.XGBRegressor(n_estimators=100,learning_rate=0.01)
        XGB_skin_model_global.fit(train_x_skin, train_y_skin)
        XGB_skin_model_global.save_model('XGBmodel3_hydration_Phase2Window3_ForSkincareRatio123.model')

        '''Local XGBooost model'''
        XGB_skin_model_user = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1)
        XGB_phyInfo_model_user = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1)

        initial = True
        pred_list = []
        Lin_combine_pred_list = []
        XGB_model_pred_list = []
        NN_pred_list = []

        ''' in order to avoid changing any value in test_y_skin '''
        test_y_data_skin_temp = test_y_skin.copy() 
        test_y_data_skin_temp_2 = test_y_skin.copy()
        test_y_data_skin_temp_flatten = np.stack(test_y_data_skin_temp_2, axis=1)

        #window_size_personal = 3
        for i in range(len(test_x_skin)):
            XGB_skin_model_user.fit(np.vstack((train_x_skin, test_x_skin[:i+1])), np.vstack((train_y_skin, test_y_data_skin_temp[:i+1])))
            '''Prediction value from models'''
            pred_skin_model_global = XGB_skin_model_global.predict([test_x_skin[i]])
            pred_list.append(pred_skin_model_global[0])

        test_y_data_temp_flatten = np.stack(test_y_data_skin_temp, axis=1)[0]
        #print("test_y_data_temp_flatten = ", test_y_data_temp_flatten)
        pred_list = np.hstack((pred_list))
        error = np.mean(abs(pred_list * 100 - test_y_data_temp_flatten * 100))
        print("error = ", error)
        cross_error.append(error)

    store_csv(cross_error, 'XGB_cross_error_1skin_0phyInfo_3days_separatetrain_')
    print("total_cost_time = ", time.time()-start_time)
