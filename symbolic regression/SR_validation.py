import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt   
from sklearn.metrics import r2_score    
from sklearn.model_selection import train_test_split    
  
data = pd.read_csv('Opt_XGB_descriptor-nohighMW.csv')    
# data = pd.read_csv('Cor_descriptor-nohighMW.csv')    
  
X = data[['VE3_DzZ', 'VE3_Dzm', 'Sm', 'ATSC0p', 'MW', 'ATSC0m',
       'VR3_D', 'Atomic_Mass_range', 'ATS4m', 'ATSC0Z', 'AATSC0m', 'AATS2Z',
       'ATS5m', 'ATS3m', 'MID_X', 'AATS3Z', 'nS', 'BCUTs-1l', 'ATS1Z', 'MIC3',
       'MIC5', 'ATS0m', 'BCUTv-1h', 'Atomic_Mass_skew', 'Atomic_Mass_mean',
       'AATSC0Z', 'MIC2', 'MIC0', 'AATS0v', 'AATS0Z', 'ZMIC5', 'MIC4',
       'AATS1Z']]
y = data['log2(AE)']    
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)    
  
# 定义复杂的计算公式为lambda函数  
def AE_formula(row):  
    VE3_DzZ = row['VE3_DzZ']  
    VE3_Dzm = row['VE3_Dzm'] 
    MW = row['MW']  
    Sm = row['Sm']  
    ATSC0p = row['ATSC0p']   
    ATSC0m = row['ATSC0m']  
    VR3_D = row['VR3_D']  
    ATS4m = row['ATS4m'] 
    ATS5m = row['ATS5m']  
    ATS3m = row['ATS3m']  
    ATS1Z = row['ATS1Z']   
    BCUTv_1h = row['BCUTv-1h']       
      

    return (2.7498167 * np.log(MW)) - np.log(ATS3m + ATSC0Z) # eq. 2
    # return (6.5425587 + (2.6068673 * VE3_DzZ)) - VR3_D # eq. 3   76779 row
    # return (2.4686925 * np.log(MW + MW)) - np.log(896.3235 + ATS3m) # eq. 4
  
# 使用apply方法将函数应用到X_train和X_test的每一行  
y_pred_train = X_train.apply(AE_formula, axis=1)   
y_pred_test = X_test.apply(AE_formula, axis=1)  
  
# 计算R²值  
r2_train = r2_score(y_train, y_pred_train)    
r2_test = r2_score(y_test, y_pred_test)  

#======== AE plot ========================
plt.rc('font',family='Times New Roman',weight='normal')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 42, # 26
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 42,
}
plt.figure(figsize=(12.0,11.5)) #(8.0,7.5)
#plt.title("Traing dataset with r2=",font1)

y_train = y_train.ravel()
y_pred_train = y_pred_train.ravel()
y_test = y_test.ravel()
y_pred_test = y_pred_test.ravel()

plt.plot(y_train,y_pred_train,color='#C0C0C0',marker='o',linestyle='', markersize=10, markerfacecolor='#80C149',alpha=1) #8
plt.plot(y_test,y_pred_test,color='#C0C0C0',marker='o',linestyle='', markersize=10, markerfacecolor='#b80d57',alpha=0.4)

# 添加R²值到图表上  
plt.text(0.05, 0.88, f'R² Train: {r2_train:.2f}', transform=plt.gca().transAxes, fontsize=44) # 26    
plt.text(0.05, 0.75, f'R² Test: {r2_test:.2f}', transform=plt.gca().transAxes, fontsize=44)  

plt.legend(labels=["Training data","Test data"],loc="lower right",fontsize=38, frameon=True) # 22
title='MD calculated log2(AE) [kcal/mol]' 
title1='SR Predicted log2(AE) [kcal/mol]'

plt.xlabel(title,font1)
plt.ylabel(title1,font1)

plt.xlim((5, 10))
plt.ylim((5, 10))
plt.plot([5, 10],[5, 10], color='k', linewidth=5.0, linestyle='--') # 2.5
my_x_ticks = np.arange(5, 10, 1.0)
my_y_ticks = np.arange(5, 10, 1.0)  

plt.xticks(my_x_ticks,size=38) #18
plt.yticks(my_y_ticks,size=38)
plt.tick_params(width=4.0, length=10.0) # 2.0
bwith = 4.0 #2.0
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.savefig('./eq3-nohighMW-AE_rs_accuracy-3D.png', dpi=600)
