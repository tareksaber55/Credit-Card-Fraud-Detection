import pandas as pd
import joblib
from credit_fraud_utils_data import report

def load_data(path, scaler):
    df_test = pd.read_csv(path)
    df_test['Hour'] = df_test['Time'] // 3600 % 24


    x_test = df_test.drop(columns=['Time','Class'])
    t_test = df_test['Class']

    if scaler is not None:
        x_test = scaler.transform(x_test)

    return x_test, t_test


def test():
    model_dict = joblib.load('model_dict.joblib')
    model = model_dict['model']
    threshold = model_dict['threshold']
    scaler = model_dict['scaler']

    x_test, t_test = load_data(r'data\test.csv', scaler)
    report(model, x_test, t_test, threshold)


if __name__ == '__main__':
    test()
