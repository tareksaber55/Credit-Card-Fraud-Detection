import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from credit_fraud_utils_data import *
import joblib
def logistic(x,t,cost_sensitive=False):
    #Cost-Sensitive approach
    if cost_sensitive:
        model = LogisticRegression(fit_intercept=True,class_weight='balanced')
        model.fit(x,t)
        return model
    else:
        model = LogisticRegression(fit_intercept=True)
        model.fit(x,t)
        return model
    
def randomforest(x_train,t_train,max_depth,n_estimators):
    model = RandomForestClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
    model.fit(x_train, t_train)
    return model
            

def neuralnetwork(x_train,t_train,layer,cost_sensitive = False):
    model = MLPClassifier(
    hidden_layer_sizes=layer,   
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
    if (cost_sensitive):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes= np.unique(t_train),
            y=t_train
        )
        weight_dict = {0 : class_weights[0], 1 : class_weights[1]}
        model.fit(x_train, t_train, sample_weight=[weight_dict[cls] for cls in t_train])
    else:
        model.fit(x_train, t_train)
    return model



if __name__ == '__main__':
    scaler,X_train,X_val,t_train,t_val = load_data(r'data\newtrain.csv',r'data\val.csv',1)
    model = randomforest(X_train,t_train,10,45)
    best_threshold = f1_scores(model,X_val,t_val)
    report(model,X_val,t_val,best_threshold)
    model_dict = {
        'model' : model,
        'threshold' : best_threshold,
        'scaler' : scaler
    }
    joblib.dump(model_dict, 'model_dict.joblib')
