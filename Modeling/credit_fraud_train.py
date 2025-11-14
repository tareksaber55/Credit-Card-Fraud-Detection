import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from credit_fraud_utils_data import *
import argparse
import joblib


def gridsearch(x_train,t_train,model,params,scaler,sampling,factor):
    features = ["Amount","Hour"]
    if(scaler == 'StandardScaler'):
        scaler = ColumnTransformer(transformers=[('StandardScaler',StandardScaler(),features)],remainder='passthrough') 
    elif(scaler == 'MinMaxScaler'):
        scaler = ColumnTransformer(transformers=[('minmax_scaler',MinMaxScaler(),features)],remainder='passthrough') 
    elif(scaler == 'RobustScaler'):
        scaler = ColumnTransformer(transformers=[('robust_scaler',RobustScaler(),features)],remainder='passthrough')
    else:
        scaler = None
    minority_count = sum(t_train == 1)
    majority_count = sum(t_train == 0)
    if(sampling == 'UnderSample'):
        sampler = RandomUnderSampler(random_state=42,sampling_strategy={0:int(minority_count*factor)})
    elif(sampling == 'OverSample'):
        sampler = SMOTE(random_state=42,sampling_strategy={1:int(majority_count/factor)})
    elif(sampling == 'Both'):
        sampler = SMOTEENN(random_state=42)
    else:
        sampler = None
    if(sampler and scaler):
        pipe = imbPipeline([('scaler',scaler),('sampler',sampler),('model',model)])
    elif(scaler):
        pipe = Pipeline([('scaler',scaler),('model',model)])
    elif(sampler):
        pipe = imbPipeline([('sampler',sampler),('model',model)])
    else:
        pipe = Pipeline([('model',model)])

    print(f'running grid with scaler {scaler} and sampler {sampler}')
    cv = StratifiedKFold(n_splits=2,random_state=42,shuffle=True)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(x_train,t_train)
    print(f'best params are {grid.best_params_}')
    print(f'best f1 score is {grid.best_score_}')
    return grid.best_estimator_,grid.best_params_

if __name__ == '__main__':
    # Arg Parser
    parser = argparse.ArgumentParser(description='Traning and Validation Model')
    parser.add_argument('--model',type=str,default='RandomForest',
                        choices=['LogisticRegression','RandomForest','NeuralNetwork','VotingClassifier'],
                        help='model to train')
    parser.add_argument('--scaler',type=str,default='StandardScaler',
                        choices=['StandardScaler','MinMaxScaler','RobustScaler',None],
                        help='Scaler to scale data')
    parser.add_argument('--sampler',type=str,default=None,
                        choices=['UnderSample','OverSample','Both',None],
                        help='Sampling Strategy to handle imbalanced')
    parser.add_argument('--factor',type=float,default=2.0,
                        help='Sampling factor (eg. 0.5 = undersample , 1.5 = oversample)')
    parser.add_argument("--train", type=str, default=r"data\newtrain.csv",
                        help="Path to training CSV file.")
    parser.add_argument("--val", type=str, default=r"data\val.csv",
                        help="Path to val CSV file.")
    args = parser.parse_args()
    
    # load data
    x_train,x_val,t_train,t_val = load_data(args.train,args.val)
    # define the model and params
    if(args.model == 'LogisticRegression'):
        model = LogisticRegression()
        params = {'model__alpha':[0.01,0.1,1,10]}
    elif(args.model == 'RandomForest'):
        model = RandomForestClassifier()
        params = {'model__n_estimators':[20,25,30,35,40,45,50],
                  'model__max_depth':[3,5,7,9]}
    elif(args.model == 'NeuralNetwork'):
        model = MLPClassifier()
        params = {'model__hidden_layer_sizes':[(64,32,16,8),(16,8)]}
    else:
        lr = LogisticRegression(random_state=42, max_iter=500)
        rf = RandomForestClassifier(random_state=42)
        nn = MLPClassifier(random_state=42, max_iter=500)
        model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('nn', nn)], voting='soft')
        params = {
            'model__lr__C': [0.1, 1],
            'model__rf__n_estimators': [25,30,50],
            'model__nn__hidden_layer_sizes': [(32,16), (16,8)]
        }

    # training and validation
    best_model , best_params = gridsearch(x_train,t_train,model,
                                          params,args.scaler,args.sampler,
                                          args.factor)
    eval(best_model,x_val,t_val)
    best_threshold = f1_scores(best_model,x_val,t_val)
    report(best_model,x_val,t_val,best_threshold)
    model_dictionary = {
        'model' : best_model,
        'threshold' : best_threshold
    }
    joblib.dump(model_dictionary, 'model_dictionary.joblib')