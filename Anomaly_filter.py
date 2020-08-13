from sklearn.ensemble import IsolationForest
import numpy as np

def build_iForest_model(train_data):
    random_state = np.random.RandomState(1)
    iForest_model = IsolationForest(random_state=random_state, behaviour='new')
    iForest_model.fit(train_data)
    return iForest_model

def iForest_anomaly_detection(model, input_data):
    anomaly = model.predict(input_data)
    return anomaly
