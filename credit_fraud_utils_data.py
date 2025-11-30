from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,average_precision_score,f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler , RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


def load_data(train_path,val_path,args):
    df_train = pd.read_csv(train_path)
    df_val =  pd.read_csv(val_path)
   
    if(args.outliers_features):
        df_train = del_outliers(df_train,args.outliers_features,args.outliers_factor)

    # we cannot work with time as it is , we should convert it to Hours of day 
    df_train['Hour'] = df_train['Time'] // 3600 % 24
    df_val['Hour'] = df_val['Time'] // 3600 % 24

    X_train = df_train.drop(['Time','Class'],axis=1)
    t_train = df_train['Class']
    
    X_val = df_val.drop(['Time','Class'],axis=1)
    t_val = df_val['Class']

    return X_train,X_val,t_train,t_val


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



def del_outliers(df,cols,factor):
    """
    Remove outliers only from Class == 0 using the IQR method.
    """
    df_0 = df[df.Class == 0]
    df_1 = df[df.Class == 1]
    for col in cols:
        q1 = df_0[col].quantile(0.25)
        q3 = df_0[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor*iqr
        upper = q3 + factor*iqr
        df_0 = df_0[(df_0[col] <= upper) & (df_0[col] >= lower)]
    df = pd.concat([df_0,df_1],axis=0).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
    return df




def bestthreshold(model,x,t,plot=True):
    t_prob = model.predict_proba(x)[:,1]
    precision,recall,threshold = precision_recall_curve(t,t_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1_score = (2*precision*recall)/(recall+precision)
    if plot:
        plt.plot(threshold,f1_score)
        plt.xlabel('Thresholds')
        plt.ylabel('F1-Scores')
        plt.title('F1-Score Results')
        plt.show()

    max_idx = np.argmax(f1_score)
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