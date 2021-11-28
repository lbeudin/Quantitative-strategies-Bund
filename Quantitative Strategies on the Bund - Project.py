# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:37:30 2021

@author: Léa
"""


from pandas import read_csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import ensemble
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from scipy.ndimage.interpolation import shift
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xlwings as xw


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Datasbase Management
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

filename = 'data2_week_momentum2.csv'
dataset = read_csv(filename,delimiter=';',index_col=0)
col =['Yield Bund signal','Yield Bund Response','Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1','FUT','Years to maturity','Conv CTD Fwd Rsk','Coupon','Yield','Yield D-1']           
array = dataset.loc[:,col]
array = array.dropna()    

col =['Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1']           
response ='Yield Bund Response'
responseCl = 'Yield Bund signal'


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         CSV
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def exportCSV(dataframe,title):
    wb = xw.Book()  
    wb.sheets[0].range('A1').value=dataframe
    wb.save(title)
    wb.close()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Graphes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def signal_graphe(prediction,real,title):
    real=np.cumsum(real)
    prediction=np.cumsum(prediction)
    prediction = pd.DataFrame(prediction)
    real = pd.DataFrame(real)
    conc = prediction
    conc[1] =real 
    conc = pd.DataFrame(conc)
    plt.plot(conc)
    plt.title("Signal Pedicted vs signal real "+title)
    plt.xlabel('Date')
    plt.legend(['prediction','real'])
    plt.ylabel('Signal cumulated Up and Down')
    plt.show()

def graphsimple(prediction,real):
    real.columns=['real Yield']
    prediction=pd.DataFrame(prediction)
    prediction.columns=['pred yield']
    real=pd.DataFrame(real)
    real.append(prediction['pred yield'])
    conc= real.join(prediction.set_index(real.index[:len(prediction)]))
    plt.plot(conc)
    plt.title("Linear Regression - prediction and real yield")
    plt.legend(['real', 'prediction'])
    plt.ylim(1,-1)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         SIGNAL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#create a signal for regression if Y_pred volatility is above a limit, yield is predicted to go up or down
def signal_vol(prediction,real,X):
    nbWeek=X[0]
    limVol=X[1]    
    FIXE = 52
    currVol = pd.DataFrame(np.array(real.rolling(FIXE).std(ddof=0))/np.sqrt(52)).iloc[:,0]   
    RealPred = pd.DataFrame(shift(real,1))
    for i in range(2,FIXE,1):
        RealPred[i] =  shift(real,i)
    RealPred["pred"] = np.array(prediction)
    RealPred.columns=range(0,FIXE,1)
    avg=RealPred.sum(axis=1)/FIXE
    for i in range(0,FIXE,1):
        RealPred[i]=(RealPred[i]-avg)**2
    predVol = RealPred.sum(axis=1)
    predVol= np.sqrt(predVol/FIXE)/np.sqrt(52)
    return np.where(predVol>limVol,np.where(prediction>0,1,-1),0)

#create a signal for regression if Y_pred is above limvol, yield is predicted to go up or down
def signalRegression(Y_pred,limVol):
    return np.where(abs(Y_pred)>limVol,np.where(Y_pred>0,1,-1),0)
  

#fonction qui calcul le pnl avec une strategie prise sur les futures
def pnl_fut(Y_pred,array,start,limVol,title):
    y_real_1 = pd.DataFrame(array.loc[:,'Yield D-1'])
    real = pd.DataFrame(array.loc[:,'Yield'])
    Y_pred= (pd.DataFrame(Y_pred))
    duration = np.zeros(len(y_real_1))
    duration[0]=0
    pn = np.zeros(len(y_real_1)-start)
    pn[0]=0
    N=0  
    for i in range(0,len(Y_pred)):
        if(abs(Y_pred.iloc[i,:][0])>limVol):
            if(Y_pred.iloc[i,:][0]>0):
                N=N-1
            else :
                N=N+1  
        pn[i]=-9.5*(real.iloc[i+start,:][0]-y_real_1.iloc[i+start,:][0])*N
        
    plt.plot(np.cumsum(pn))
    plt.title("PNL - Margin Call "+title)
    plt.xlabel('Date')
    plt.ylabel('PNL')
    plt.show()
    return pn

def signal_CL_2threshold(prediction,X):
    prediction = np.array(prediction)
    return np.where(prediction>X[1],1,np.where(prediction<X[0],-1,0))

#calculate bond price
def bond_price(par, T, ytm, coup, freq):
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + \
            par/(1+ytm/freq)**(freq*T)
    return price

#Fonction qui calcul le pnl en cas de stratégie d'achat du bund
def pnl(prediction,array,start,title):
    pl = np.zeros(len(prediction))
    cash=0
    n=0
     #-duration*deltaY
    prediction = np.array(prediction)
    for i in range(0,len(prediction)):
        bond = bond_price(100, array.iloc[i+start,9],  array.iloc[i+start,12]/100,  array.iloc[i+start,11], 1)
        if(prediction[i]==-1):
            cash = cash-bond# we buy bond
            n=n+1
        else :
            if(prediction[i]==1):
                cash = cash+bond 
                n=n-1
                #we sell bond
        pl[i]=cash+n*bond
    plt.plot(pl)
    plt.title("PNL "+title)
    plt.xlabel('Date')
    plt.ylabel('PNL')
    plt.show()
    print(np.mean(pl))
    return pl
 
#fonction qui calcul le pnl pour les classification strategies sur les future   
def pnlCL(Y_pred,array,X,start,title):
    y_real_1 = pd.DataFrame(array.loc[:,'Yield D-1'])
    Y_pred= pd.DataFrame(Y_pred)
    Yreal =pd.DataFrame(array.loc[:,'Yield'])
    pn = np.zeros(len(y_real_1)-start)
    pn[0]=0
    N=0
    limVol=X[1]
    limvol2 = X[0]
    for i in range(0,len(Y_pred)):
        if(Y_pred.iloc[i,:][0]>limVol):
            N=N-1
        else :
            if(Y_pred.iloc[i,:][0]<limvol2):
                N=N+1        
        pn[i]=-9.5*(Yreal.iloc[i+start,:][0]-y_real_1.iloc[i+start,:][0])*N
    plt.plot(np.cumsum(pn))
    plt.title("PNL - Margin Call "+title)
    plt.xlabel('Date')
    plt.ylabel('PNL')
    plt.show()
    return pn

    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Methods and Results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#fonction qui gère le backtest pour les méthodes de regression
def Regresson_results(title,model,dataset,col,response,start,signal,stop,X):
    prediction = startReg(model,dataset,col,response,title,start,stop)
    real = dataset.loc[:,response]
    real=real.iloc[start:]
    pnl_fut(prediction,array,start,X[1],title)
    print("MSE  "+title,metrics.mean_squared_error(prediction,real))
    print("MEA "+title,metrics.mean_absolute_error(prediction,real)) 
             
    signal_prediction = signal(prediction,X[1])
    signal_real = signal(pd.DataFrame(real),X[1])
    signal_real = signal_real.reshape(len(signal_real))
    signal_prediction = signal_prediction.reshape(len(signal_prediction))
    daily_error=signal_prediction-signal_real
    error_nb = len(daily_error) - np.count_nonzero(daily_error)
    graphsimple(prediction,real)
    print("Success in % : " +str(error_nb/len(daily_error)))
    signal_graphe(signal_prediction,signal_real,title)
    confusionmatrix = confusion_matrix(signal_real,signal_prediction)#pd.crosstab(signal_real,signal_prediction, rownames=['Real'], colnames=['Predicted'])
    print (confusionmatrix)
    #excel
    predt = np.array(prediction).reshape(len(prediction),1)
    realt = np.array(real).reshape(len(real),1)
    sigpredt = signal_prediction.reshape(len(signal_prediction),1)
    test = np.append(predt,realt,axis=1)
    test = np.append(test,sigpredt,axis=1)
    test = pd.DataFrame(test)
    test.columns = ['prediction yield','real yield','signal']
    exportCSV(test,"signal.csv")
    return signal_prediction

#fonction qui gère le backtest pour les méthodes de classification
def Regresson_resultsCL(title,model,dataset,col,response,responseCl,start,stop,X):
    prediction = startReg(model,dataset,col,responseCl,title,start,stop)
    real = dataset.loc[:,response]
    real=real.iloc[start:]
    signal_real =  pd.DataFrame(signalRegression(real,X[2]))
    signal_prediction = pd.DataFrame(signal_CL_2threshold(prediction,X))
    daily_error=signal_prediction-signal_real
    error_nb = len(daily_error) - np.count_nonzero(daily_error)
    print("Success signal modified in % : " +str(error_nb/len(daily_error)))
    signal_graphe(signal_prediction,signal_real,title)
    confusionmatrix = pd.crosstab(signal_real.iloc[:][0],signal_prediction.iloc[:][0] ,rownames=['Real'], colnames=['Predicted'])
    print(confusionmatrix)
    pnlCL(prediction,dataset,X,start,title)
    #excel
    realt = np.array(signal_real).reshape(len(signal_real),1)
    sigpredt = np.array(signal_prediction).reshape(len(signal_prediction),1)
    test = np.append(realt,sigpredt,axis=1)
    test = pd.DataFrame(test)
    test.columns = ['prediction signal','real signal']
    exportCSV(test,"signal.csv")
    return signal_prediction

#fonction qui gère le réapprentissage, retourne le Y prediction    
def startReg(model,dataset,col,response,title,start,stop):
    prediction=np.zeros(len(dataset))
    for i in range(start,len(dataset)):
        X =dataset.loc[:,col]
        X_pred = X.iloc[i,:]
        X = X.iloc[i-stop:i,:]
        Y = dataset.loc[:,response]
        Y_pred = Y.iloc[i]
        Y=Y.iloc[i-stop:i]
        prediction[i] = model(Y,X,Y_pred,X_pred)
    prediction = pd.DataFrame(prediction) 
    return prediction.iloc[start:,:]

#fonction qui gère l'environnement réel
def start_real_environnement(model,dataset,col,response,title,start,stop,signal,limvol):
    prediction=np.zeros(len(dataset))
    for i in range(start,len(dataset)):
        X =dataset.loc[:,col]
        X_pred = X.iloc[i,:]
        X = X.iloc[i-stop:i,:]
        Y = dataset.loc[:,response]
        Y_pred = 0
        Y=Y.iloc[i-stop:i]
        prediction[i] = model(Y,X,Y_pred,X_pred)
    return signal(pd.DataFrame(prediction).iloc[len(prediction)-stop:,:],limvol)

#deux méthodes de calibration pour le gradiant boost
def calib(X,Xparams):#title,model,dataset,col,response,start,signal,stop,X):
    prediction = startReg(Xparams[1],Xparams[2],Xparams[3],Xparams[4],Xparams[0],int(X[0]),int(X[0]))
    real = Xparams[2].loc[:,Xparams[4]]
    real=real.iloc[int(X[0]):]
    signal_prediction = Xparams[5](prediction.iloc[:,0],real,[int(X[1]),X[2]])
    signal_real =  Xparams[5](real,real,[int(X[1]),X[2]])
    daily_error=signal_prediction-signal_real
    error_nb = len(daily_error) - np.count_nonzero(daily_error)
    return (1-error_nb/len(daily_error))

def neldermead(funct, X,step,stop_stagne,stop, ittBreak,X_params):
    nb_params = len(X)
    F0 = funct(X,X_params)
    m = 0
    k = 0
    refl =1
    exp =2
    contr=-1/2
    red=0.5
    params = [[X, F0]]
    for i in range(nb_params):
        vect = copy.copy(X)
        vect[i] = vect[i] + step
        params.append([vect,funct(vect,X_params)])
    while 1:
        params.sort(key=lambda x: x[1])
        F = params[0][1]
        if ittBreak<= k :
            return params[0]
        k =k+1
        if F<F0-stop_stagne:
            m = 0
            F0 = F
        else:
            m =m+1
        if m >= stop:
            return params[0]
            
        centroid = [0.] * nb_params
        for cen in params[:-1]:
            for i, c in enumerate(cen[0]):
                centroid[i] += c / (len(params)-1)
        newParam_refl = centroid + refl*(centroid - np.array(params[-1][0]))
        refl_r = funct(newParam_refl,X_params)
        if params[0][1] <= refl_r < params[-2][1]:
            del params[-1]
            params.append([newParam_refl, refl_r])
            continue
        if refl_r < params[0][1]:
            newParam_exp = centroid + exp*(centroid - np.array(params[-1][0]))
            exp_e = funct(newParam_exp,X_params)
            if exp_e < refl_r:
                del params[-1]
                params.append([newParam_exp,exp_e])
                continue
            else:
                del params[-1]
                params.append([newParam_refl, refl_r])
                continue
        newParam_contr = centroid + contr*(centroid -  np.array(params[-1][0]))
        contr_c = funct(newParam_contr,X_params)
        if contr_c < params[-1][1]:
            del params[-1]
            params.append([newParam_contr, contr_c])
            continue
        par = params[0][0]
        params2 = []
        for li in params:
            newParam_red = par + red*(li[0] - np.array(par))
            red_r = funct(newParam_red,X_params)
            params2.append([newParam_red, red_r])
        params = params2


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Models
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
#Cette partie rassemble les modèles de machine learning
def multipleRegressionLinear(Y,X,Y_pred,X_pred):
    reg = sm.OLS(Y,X)
    resReg = reg.fit()
    prediction = reg.predict(resReg.params,X_pred )
    return prediction

def logReg(Y,X,Y_pred,X_pred):
    logreg = LogisticRegression()
    sc = StandardScaler()
    X= sc.fit_transform(X)
    X_pred = np.array(X_pred).reshape(1,-1)
    X_pred = sc.transform(X_pred)
    modelLogReg = logreg.fit(X,Y)
    prediction = modelLogReg.predict(X_pred)
    return prediction

def logRegProb(Y,X,Y_pred,X_pred):
    logreg = LogisticRegression()
    sc = StandardScaler()
    X= sc.fit_transform(X)
    X_pred = np.array(X_pred).reshape(1,-1)
    X_pred = sc.transform(X_pred)
    modelLogReg = logreg.fit(X,Y)
    prediction = modelLogReg.predict_proba(X_pred)
    return prediction[0,0]

def PCA_Red2(Y,X,Y_pred,X_pred):
    pca = PCA(0.90)
    X_reduced = pca.fit_transform(scale(X))
    pls = PLSRegression(n_components=3, scale=False)
    pls.fit(scale(X_reduced),Y)
    X_pred =np.array(X_pred).reshape(1,-1)
    X_pred=pca.transform(scale(X_pred))
    prediction = pls.predict(X_pred)
    return prediction

def gradientBoost(Y,X,Y_pred,X_pred):
    params={'n_estimators':3650,'max_depth':22,'learning_rate':0.4,'criterion':'mse','loss':'quantile'}
    gradiantboost = ensemble.GradientBoostingRegressor(**params)
    gradiantboost.fit(X,Y)
    X_pred = np.array(X_pred).reshape(1,-1)
    prediction = gradiantboost.predict(X_pred)
    return prediction

def polynomial(Y,X,Y_pred,X_pred):
    polynomial_features= PolynomialFeatures(degree=2)
    X_pred = np.array(X_pred).reshape(1,-1)
    X_poly = polynomial_features.fit_transform(X)
    X_poly_pred = polynomial_features.fit_transform(X_pred)
    model = LinearRegression()
    model.fit(X_poly, Y)
    prediction = model.predict(X_poly_pred)
    return prediction

def KNN(Y,X,Y_pred,X_pred):
    knnr = KNeighborsRegressor(n_neighbors = 10)
    knnr.fit(X, Y)
    X_pred = np.array(X_pred).reshape(1,-1)
    prediction = knnr.predict(X_pred)
    return prediction

def TreeCart(Y,X,Y_pred,X_pred):
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X,Y)
    X_pred = np.array(X_pred).reshape(1,-1)
    ypredprob = pd.DataFrame(clf.predict_proba(X_pred))
    #prediction = ypredprob.applymap(lambda x: 1 if x<0.4 else -1)
    prediction = ypredprob.iloc[:,1]
    return prediction

def KNN_CL(Y,X,Y_pred,X_pred):
    knn = KNeighborsClassifier(n_neighbors=7)  
    knn.fit(X, Y) 
    X_pred = np.array(X_pred).reshape(1,-1)
    ypredprob = pd.DataFrame(knn.predict_proba(X_pred))
    #prediction = ypredprob.applymap(lambda x: 1 if x<threshold else 0)
    prediction = ypredprob.iloc[:,1]
    return prediction

def RandomForest(Y,X,Y_pred,X_pred):
    nb_tree = 2000
    sc = StandardScaler()    
    X= sc.fit_transform(X)
    X_pred = np.array(X_pred).reshape(1,-1)
    X_pred = sc.transform(X_pred)
    regressor = RandomForestRegressor(n_estimators=nb_tree, random_state=0)
    regressor.fit(X, Y)
    prediction = regressor.predict(X_pred)
    #prediction =  np.where(prediction<threshold,1,0)
    return prediction

def LDA(Y,X,Y_pred,X_pred):
    sc = StandardScaler()
    X= sc.fit_transform(X)
    X_pred= np.array(X_pred).reshape(1,-1)

    X_pred = sc.transform(X_pred)
    regi = LinearDiscriminantAnalysis()
    resReg = regi.fit(X,Y)
    ypredprob = pd.DataFrame(resReg.predict_proba(X_pred))
    #prediction = signal(ypredprob,threshold)
    prediction = ypredprob.iloc[:,1]
    return prediction

def logRegCL(Y,X,Y_pred,X_pred):
    logreg = LogisticRegression()
    logreg.fit(X,Y)
    pred_proba_df = pd.DataFrame(logreg.predict_proba(X_pred))
    #prediction = signal(pred_proba_df,threshold)
    return pred_proba_df


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Results & Backtest
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
#dans cette partie on étudie les modèles de machine learning avec backtest et PNL 
"""Multiple Linear regression """
tic = time.perf_counter()
Xparams =[0,0.05]
prediction = Regresson_results("Multiple Reg",multipleRegressionLinear,array,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,100,"Multiple Reg Bund Strat")
toc = time.perf_counter()
print("Time :",str(toc-tic))
print(prediction[len(prediction)-1])


"""Regression avec les moyennes movibles"""
tic = time.perf_counter()
Xparams =[5,0.05]
mean =  array.ewm(span=4, adjust=False).mean()
mean[response]=array.loc[:,response]
prediction=Regresson_results("Mobile Mean Reg",multipleRegressionLinear,mean,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,100,"Mobile Mean Reg Y sans mean Bund ")
toc = time.perf_counter()
print("Time :",str(toc-tic))
       
"""Regression avec les moyennes movibles Y en moyenne mobile"""
tic = time.perf_counter()
Xparams =[5,0.05]
mean =  array.ewm(span=4, adjust=False).mean()
prediction=Regresson_results("Mobile Mean Reg",multipleRegressionLinear,mean,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,100,"Mobile Mean Reg Bund")
toc = time.perf_counter()
print("Time :",str(toc-tic))

""" Polynomiale regression Degré 2 """
tic = time.perf_counter()
Xparams =[0,0.06]
prediction = Regresson_results("Poly 2 Reg",polynomial,array,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,50,"Poly 2 Reg")
toc = time.perf_counter()
print("Time :",str(toc-tic))

""" KNN Regressor """
tic = time.perf_counter()
Xparams =[5,0.05]
prediction = Regresson_results("KNN Reg",KNN,array,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,50,"KNN Reg")
toc = time.perf_counter()
print("Time :",str(toc-tic))


""" PCA  """   
filename2 = 'data2_week_momentum_pca.csv'
dataset2 = read_csv(filename2,delimiter=';',index_col=0)
response ='Yield Bund Response'
col2 =['Yield Bund Response','Italy 10Y','EURUSD','GBPUSD','20 DBR','2Y DBR','WTI','EUROSTOX','5Y DBR','DAX','EURIBOR 3M','Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1','FUT','Years to maturity','Conv CTD Fwd Rsk','Coupon']           
#	RS EUR 10Y	German CPI	Fed Fund rate	Conv CTD Fwd Rsk FUT VOL	
array2 = dataset2.loc[:,col2]
array2 = array2.dropna()       
col2 =['Italy 10Y','EURUSD','GBPUSD','20 DBR','2Y DBR','WTI','EUROSTOX','5Y DBR','DAX','EURIBOR 3M','Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1','FUT','Years to maturity','Conv CTD Fwd Rsk','Coupon']           
tic = time.perf_counter()
Xparams =[0,0.1]
prediction = Regresson_results(" PCA ",PCA_Red2,array2,col2,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,50)
toc = time.perf_counter()
print("Time :",str(toc-tic))

#on utilise la method avec les paramètres calibrés prend 2H à se lancer
""" Gradient Boost Method """
tic = time.perf_counter()
Xparams =[5,0.05]
prediction=Regresson_results("Gradiant Boost",gradientBoost,array,col,response,100,signalRegression,100,Xparams)
#pl = pnl(prediction,array,50,"Gradiant Boost")
toc = time.perf_counter()
print("Time :",str(toc-tic))




""" Classification """
"""Log regression"""
tic = time.perf_counter()
Xparams =[0.40,0.75,0.05]
prediction=Regresson_resultsCL("Log ",logRegProb,array,col,response,responseCl,50,50,Xparams)
#pl = pnl(prediction,array,50," Log ")
toc = time.perf_counter()
print("Time :",str(toc-tic))

"""Log regression avec les moyennes movibles"""
tic = time.perf_counter()
mean =  array.ewm(span=4, adjust=False).mean()
rendement = (array-mean)/mean
rendement = rendement.dropna() 
Xparams =[0.350,0.85,0.03]
prediction=Regresson_resultsCL("Log Moving Average",logReg,array,col,response,responseCl,50,50,Xparams)
pl = pnl(prediction,array,50,"Log Moving Average")
toc = time.perf_counter()
print("Time :",str(toc-tic))

"""KNN """
tic = time.perf_counter()
Xparams =[0.40,0.75,0.05]
prediction=Regresson_resultsCL("KNN Cl",KNN_CL,array,col,response,responseCl,50,50,Xparams)
pl = pnl(prediction,array,50," KNN CL ")
toc = time.perf_counter()
print("Time :",str(toc-tic))

"""Regression Tree Cart"""
tic = time.perf_counter()
Xparams =[0.4,0.55,0.05]
prediction=Regresson_resultsCL("tree Cart",TreeCart,array,col,response,responseCl,50,50,Xparams)
pl = pnl(prediction,array,50," Tree Cart ")
toc = time.perf_counter()
print("Time :",str(toc-tic))

""" Random Forest """
tic = time.perf_counter()
Xparams =[0.4,0.55,0.05]
prediction=Regresson_resultsCL("Randome Forest",RandomForest,array,col,response,responseCl,50,50,Xparams)
toc = time.perf_counter()
print("Time :",str(toc-tic))

""" Linear Discriminent analysis"""
tic = time.perf_counter()
Xparams =[0.30,0.75,0.05]
prediction=Regresson_resultsCL("LDA",LDA,array,col,response,responseCl,100,100,Xparams)
toc = time.perf_counter()
print("Time :",str(toc-tic))




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        Utilisation
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
#Nombre de semaine pour sortir un signal
stop=1 
filename = 'data2_week_momentum2_tt.csv'
dataset = read_csv(filename,delimiter=';',index_col=0)
col =['Yield Bund Response','Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1']           
array = dataset.loc[:,col]
dataset.loc[stop:,col] = 0
array = array.dropna()    
col =['Momentum 3M 6M','BRENT','Spread Corpo','Spread 2A 10A','MSCI EMU','Yield Bund D-1']           
response ='Yield Bund Response'


methode = multipleRegressionLinear
signal_trading =start_real_environnement(methode,array,col,response,"Multiple Reg",100,1,signalRegression,0.05)
if signal_trading==0:
    print("Signal Hold")
if signal_trading==1:
    print("Signal Sell")
if signal_trading==-1:
    print("Signal Buy")



