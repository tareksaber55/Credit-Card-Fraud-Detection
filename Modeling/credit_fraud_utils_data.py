from matplotlib import pyplot as plt
from matplotlib.pylab import shuffle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,average_precision_score,f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(train_path,val_path,preproccesing = 0):
    df_train = pd.read_csv(train_path)
    df_val =  pd.read_csv(val_path)

    X_train = df_train.drop(['Class'],axis=1)
    t_train = df_train['Class']
    
    
    X_val = df_val.drop(['Class'],axis=1)
    t_val = df_val['Class']


    scaler = None
    features = ["Time","Amount"]
    if(preproccesing == 1):
        scaler = ColumnTransformer(transformers=[('standard_scaler',StandardScaler(),features)],remainder='passthrough') 
    elif(preproccesing == 2):
        scaler = ColumnTransformer(transformers=[('minmax_scaler',MinMaxScaler(),features)],remainder='passthrough') 
    elif(preproccesing == 3):
        scaler = ColumnTransformer(transformers=[('robust_scaler',RobustScaler(),features)],remainder='passthrough') 
    if(scaler):
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
    return scaler,X_train,X_val,t_train,t_val


def eval(model,x,t):
    t_prob = model.predict_proba(x)[:, 1]
    precision,recall,threshold = precision_recall_curve(t,t_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1_score = (2*precision*recall)/(recall+precision)
    max_idx = np.argmax(f1_score)
    avg_pr = average_precision_score(t,t_prob)
    print(f'the best threshold {threshold[max_idx]}, F1-Score {f1_score[max_idx]}')
    print(f'average precision : {avg_pr}')
    


def undersample(x, t, factor=1):
    minority_count = sum(t == 1)
    majority_target = int(factor * minority_count)
    rus = RandomUnderSampler(
        random_state=42,
        sampling_strategy={0: majority_target , 1: minority_count}
    )
    return rus.fit_resample(x, t)




def oversample(x,t,factor=1):
    majority_size = sum(t==0)
    new_size = majority_size//factor
    x,t = SMOTE(random_state=42,sampling_strategy={1:new_size},k_neighbors=3).fit_resample(x,t)
    return x,t




def under_over_sample(x,t):
    majority_size = sum(t==0)
    minority_size = sum(t==1) 
    new_size = majority_size//2
    x,t = RandomUnderSampler(
        random_state=1,
        sampling_strategy={0: new_size}
    ).fit_resample(x,t)
    x,t = SMOTE(
        random_state=1,
        sampling_strategy={1:new_size},
        k_neighbors=3).fit_resample(x,t)
    return x,t



def show_distributions(df):

    fig,axs = plt.subplots(6,5,figsize=(15,12))

    for i,ax in enumerate(axs.flat):
        feature = df.columns[i]
        ax.boxplot([df[df.Class == 0][feature] ,
                   df[df.Class == 1][feature]])
        
        ax.set_title(feature)
        ax.set_xticks([1,2])
        ax.set_xticklabels(['No Fraud','Fraud'])
    plt.suptitle('Distribution of Features Per Class')
    plt.tight_layout()
    plt.show()



def del_outliers(df,cols):
    df_0 = df[df.Class == 0]
    df_1 = df[df.Class == 1]
    for col in cols:
        q1 = df_0[col].quantile(0.25)
        q3 = df_0[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        df_0 = df_0[(df_0[col] < upper) & (df_0[col] > lower)]
    df = pd.concat([df_0,df_1],axis=0).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df




def f1_scores(model,x,t):
    t_prob = model.predict_proba(x)[:,1]
    precision,recall,threshold = precision_recall_curve(t,t_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1_score = (2*precision*recall)/(recall+precision)
    
    plt.plot(threshold,f1_score)
    plt.xlabel('Thresholds')
    plt.ylabel('F1-Scores')
    plt.title('F1-Score Results')
    plt.show()

    max_idx = np.argmax(f1_score)
    print(f'the best threshold {threshold[max_idx]}, F1-Score {f1_score[max_idx]}')
    return threshold[max_idx]





def report(model,x,t, threshold=0.2):
    t_prob = model.predict_proba(x)[:,1]
    t_pred = (t_prob>=threshold).astype(int)
    cm = confusion_matrix(t,t_pred)
    cr = classification_report(t,t_pred)
    f1 = f1_score(t,t_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print(f"F1-Score: {f1:.4f}")
    print(f'average_precision: {average_precision_score(t,t_prob)}')