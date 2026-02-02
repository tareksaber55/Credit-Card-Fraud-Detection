from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from credit_fraud_utils_data import *
import argparse



def gridsearch(x_train,t_train,model,params,args):
    steps = []
    if args.scaler != 'None':
        features = ["Amount","Hour"]
        # Mapping for scalers
        scaler_map = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }     
        scaler_transformer = ColumnTransformer(
            transformers=[(args.scaler, scaler_map[args.scaler], features)],
            remainder='passthrough'
        )
        steps.append(('scaler', scaler_transformer))

    
    if args.sampler != 'None':
        # Count classes
        minority_count = sum(t_train == 1)
        majority_count = sum(t_train == 0)

        # Select sampler
        if args.sampler == 'UnderSample':
            sampler = RandomUnderSampler(random_state=42, sampling_strategy={0: int(minority_count * args.factor)})
        elif args.sampler == 'OverSample':
            sampler = SMOTE(random_state=42, sampling_strategy={1: int(majority_count / args.factor)})
        else:
            sampler = SMOTEENN(random_state=42)
        steps.append(('sampler', sampler))

    steps.append(('model', model))

    # Use imbalanced-learn pipeline if sampler exists, otherwise sklearn pipeline
    pipe = imbPipeline(steps) if args.sampler != 'None' else Pipeline(steps)

    cv = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=cv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(x_train,t_train)
    print(f'best params are {grid.best_params_}')
    return grid.best_estimator_,grid.best_params_


if __name__ == '__main__':
    # Arg Parser
    parser = argparse.ArgumentParser(description='Traning and Testing Model')
    parser.add_argument('--model',type=str,default='RandomForest',
                        choices=['LogisticRegression','RandomForest','NeuralNetwork','VotingClassifier','KNN'],
                        help='model to train')
    parser.add_argument('--scaler',type=str,default='StandardScaler',
                        choices=['StandardScaler','MinMaxScaler','RobustScaler','None'],
                        help='Scaler to scale data')
    parser.add_argument('--sampler',type=str,default='None',
                        choices=['UnderSample','OverSample','Both','None'],
                        help='Sampling Strategy to handle imbalanced')
    parser.add_argument('--factor',type=float,default=2.0,
                        help='Sampling factor (eg. 0.5 = undersample , 1.5 = oversample)')
    parser.add_argument("--train_dataset", type=str, default=r"data_\new_train.csv",
                        help="Path to training CSV file.")
    parser.add_argument("--val_dataset", type=str, default=r"data_\val.csv",
                        help="Path to val CSV file, this small dataset used for selecting best threshold")
    parser.add_argument("--test_dataset", type=str, default=r"data_\test.csv",
                        help="Path to Test CSV file")
    parser.add_argument('--outliers_features',type=str,nargs='+',
                        help=(
        "List of feature names on which outliers should be removed. "
        "If omitted, no outlier removal is performed."
    ))
    parser.add_argument('--outliers_factor',type=float,default=1.5,
                        help='when factor increase the number of deleted outliers decrease and vice versa')
    args = parser.parse_args()
    
    # load data
    x_train,t_train,x_val,t_val,x_test,t_test = load_data(args.train_dataset,args.val_dataset,args.test_dataset,args)
    # define the model and params
    if(args.model == 'LogisticRegression'):
        model = LogisticRegression(random_state=42)
        params = {'model__C':[0.01,0.1,1],
                  'model__penalty':['none','l2'],
                  'model__class_weight':['balanced',None]}
    elif(args.model == 'RandomForest'):      
        model = RandomForestClassifier(random_state=42)        
        params = {'model__n_estimators':[20,35,40,45,50],
                  'model__max_depth':[7,9,10],
                  'model__class_weight':['balanced',None]}
    elif(args.model == 'NeuralNetwork'):
        model = MLPClassifier(random_state=42)
        params = {'model__hidden_layer_sizes':[(64,32,16,8),(32,16,8),(16,8)],
                  'model__activation':['relu','tanh','logistic','identity']}
    elif(args.model == 'KNN'):
        model = KNN()
        params = {'model__n_neighbors':[3,5,7]}
    else:
        # combine the best estimators only
        lr = LogisticRegression(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        nn = MLPClassifier(random_state=42)
        model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('nn', nn)], voting='soft')
        params = {
            'model__lr__C': [1],
            'model__lr__penalty':['l2'],
            'model__lr__class_weight':[None],
            'model__rf__n_estimators': [35],
            'model__rf__max_depth': [9],
            'model__rf__class_weight': [None],
            'model__nn__hidden_layer_sizes': [(16,8)],
            'model__nn__activation': ['tanh'],
            'model__weights':[[1,2,2],[1,2,1.5],[1,1,1]]
        }


    # training and validation
    best_model,params = gridsearch(x_train,t_train,model,params,args)
    best_threshold = bestthreshold(best_model,x_val,t_val)
    f1,avg_precision = report(best_model,x_test,t_test,best_threshold)
    save_results_to_excel(args, params ,f1, avg_precision,best_threshold)
    save_model(best_model,best_threshold)