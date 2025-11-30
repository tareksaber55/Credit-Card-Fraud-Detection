import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from credit_fraud_utils_data import *
import argparse
import joblib


def gridsearch(x_train,t_train,model,params,scaler,sampling,factor,grid):
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
    if(grid):
        print(f'running grid with scaler {scaler} and sampler {sampler}')
        cv = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(x_train,t_train)
        print(f'best params are {grid.best_params_}')
        print(f'best f1 score is {grid.best_score_}')
        return grid.best_estimator_
    else:
        pipe.fit(x_train,t_train)
        return pipe

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
    parser.add_argument("--gridsearch",action='store_true' ,
                        help="use grid search")
    parser.add_argument('--outliers_features',type=str,nargs='+',
                        help=(
        "List of feature names on which outliers should be removed. "
        "If omitted, no outlier removal is performed."
    ))
    parser.add_argument('--outliers_factor',type=float,default=1.5,
                        help='when factor increase the number of deleted outliers decrease and vice versa')
    
    args = parser.parse_args()
    
    # load data
    x_train,x_val,t_train,t_val = load_data(args.train,args.val,args)
    # define the model and params
    if(args.model == 'LogisticRegression'):
        model = LogisticRegression(random_state=42)
        params = {'model__C':[0.01,0.1,1],
                  'model__penalty':['l1','l2'],
                  'model__class_weight':['balanced',None]}
    elif(args.model == 'RandomForest'):      
        model = RandomForestClassifier(random_state=42,max_depth=10,n_estimators=45)        
        params = {'model__n_estimators':[40,45,50,60,70,80],
                  'model__max_depth':[9,10,11,12],
                  'model__class_weight':['balanced',None]}
    elif(args.model == 'NeuralNetwork'):
        model = MLPClassifier(random_state=42,hidden_layer_sizes=(16,8))
        params = {'model__hidden_layer_sizes':[(64,32,16,8),(32,16,8),(16,8)]}
    else:
        lr = LogisticRegression(random_state=42,C=1,penalty='l2')
        rf = RandomForestClassifier(random_state=42,max_depth=10,n_estimators=45)
        nn = MLPClassifier(random_state=42,hidden_layer_sizes=(16,8))
        model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('nn', nn)], voting='soft',weights=[0.5,2.5,1])
        params = {
            'model__lr__C': [0.1, 1],
            'model__rf__n_estimators': [25,30,50],
            'model__nn__hidden_layer_sizes': [(32,16), (16,8)]
        }

    # training and validation
    best_model = gridsearch(x_train,t_train,model,
                                          params,args.scaler,args.sampler,
                                          args.factor,args.gridsearch)
    eval(best_model,x_val,t_val)
    best_threshold = bestthreshold(best_model,x_val,t_val)
    report(best_model,x_val,t_val,best_threshold)
    model_dictionary = {
        'model' : best_model,
        'threshold' : best_threshold
    }
    joblib.dump(model_dictionary, 'model_dictionary.joblib')