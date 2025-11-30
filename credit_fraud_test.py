import pandas as pd
import joblib
from credit_fraud_utils_data import report

def load_data(path):
    df_test = pd.read_csv(path)
    df_test['Hour'] = df_test['Time'] // 3600 % 24


    x_test = df_test.drop(columns=['Time','Class'])
    t_test = df_test['Class']

    return x_test, t_test


def test():
    model_dict = joblib.load('model_dictionary.joblib')
    model = model_dict['model']
    threshold = model_dict['threshold']


    x_test, t_test = load_data(r'data\test.csv')
    report(model, x_test, t_test, threshold)


if __name__ == '__main__':
    test()
