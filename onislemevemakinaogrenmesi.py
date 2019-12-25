# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from gorsellestirme import gorsellestirme
gorsellestirme()

scaler = preprocessing.MinMaxScaler()

def k_cross_validation(x, y, estimator):
    success = cross_val_score(estimator = estimator, X=x, y=y , cv = 10)
    return(success.mean())
    
    
#maaliyet ve Doğruluk oranlarını hesaplamak için kullanılan fonksiyon
#maaliyet --> (0*TruePositive + 0*TrueNegative + 1*FalseNegative + 2*FalsePositive)
def dogruluk_oranları(cm):
    dogruluk = (cm[0][0]+cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    hassaslik = (cm[0][0]/(cm[0][0]+cm[1][0]))                 #anma(recall)
    #ozgunluk =  (cm[1][1]/(cm[1][1]+cm[0][1]))
    kesinlik = (cm[0][0]/(cm[0][0]+cm[0][1]))
    maaliyet =  ((cm[1][0] * 1) + (cm[0][1] * 2) )
    F1_score = (2*hassaslik*kesinlik)/(hassaslik+kesinlik)
    Algoritmaların_degerleri={'dogruluk':dogruluk,
                              'kesinlik':kesinlik,
                              'F1-score':F1_score,
                              'maaliyet':maaliyet}

    return dogruluk, kesinlik,maaliyet, F1_score, Algoritmaların_degerleri
    

def dogruluk_(cm):
    dogruluk = (cm[0][0]+cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    return dogruluk

dataset = pd.read_csv('german.csv')


#kategorik verileri nümerikleştirme (encoding)
'''
Customer Risk Type : müşterinin riskli olup olmadığı
1: riskli, 2:riskli değil
'''

dataset['Customer Risk Type'] = dataset['Customer Risk Type'].map({1:1,2:0 });
'''
Checking Account : Mevcut çek hesabının durumu
A11: ... <0 DM
A12: 0 <= ... <200 DM
A13: ...> = 200 DM
A14: çek hesabı yok
'''
dataset['Checking Account'] = dataset['Checking Account'].map({'A11':0,'A14':0,'A12':1,'A13':2})
'''
Credit History: Kredi geçmişi
A30: Kredi alınmadı / krediler uygun şekilde ödendi
A31: bu bankadaki tüm krediler usulüne uygun olarak geri ödendi
A32: mevcut krediler şimdiye kadar usulüne göre geri ödendi
A33: Geçmişte ödeme gecikmesi
A34: Kritik hesap / mevcut diğer krediler (bu bankada değil)
alınmadı,ödendi (0) - gecikme(1) - riskli (2)
'''
dataset['Credit History'] = dataset['Credit History'].map({'A34':2,'A30':0,'A33':1,'A31':0,'A32':0})
'''
Purpose : hangi amaçla kredi aldığı
A40: araba (yeni), A41: araba (kullanılmış), A42: mobilya / ekipman, A43: radyo / televizyon, 
A44: ev aletleri, A45: tadilatlar, A46: eğitim, A48: yeniden eğitim, A49: iş
araba  - ev  - egitim  - is  olarak ayrıldı
'''
dataset['Purpose'] = dataset['Purpose'].map({'A410':'diger','A47':'diger','A43':'ev','A46':'egitim','A45':'ev','A44':'ev','A41':'araba','A40':'araba','A42':'ev','A49':'is','A48':'egitim'})
'''
Savings Account : Tasarruf hesabı / tahviller
A61: ... <100 DM
A62: 100 <= ... <500 DM
A63: 500 <= ... <1000 DM
A64: ..> = 1000 DM
A65: bilinmeyen / tasarruf hesabı yok
'''
dataset['Savings Account'] = dataset['Savings Account'].map({'A65':0,'A61':1,'A62':2,'A63':3,'A64':4})
'''
Present employment since : şimdiye kadarki iş deneyimi
A71: işsiz
A72: ... <1 yıl
A73: 1 <= ... <4 yıl
A74: 4 <= ... <7 yıl
A75: ..> = 7 yıl
'''
dataset['Present employment since'] = dataset['Present employment since'].map({'A71':0,'A72':1,'A73':2,'A74':3,'A75':4})
'''
Other Debtors/Guarantors : Diğer borçları / kefiller
A101: yok
A102: ortak başvuru sahibi
A103: garantör
'''
dataset['Other Debtors/Guarantors'] = dataset['Other Debtors/Guarantors'].map({'A101':0,'A102':1,'A103':2})
'''
Property : özellik
A121: emlak
A122: toplum tasarruf sözleşmesi / hayat sigortası yapılması
A123: araba veya diğer
A124: bilinmeyen / özellik yok
'''
dataset['Property'] = dataset['Property'].map({'A124':0,'A123':1,'A122':2,'A121':3})
'''
Other installment plans : Diğer taksit planları
A141: banka
A142: mağazalar
A143: yok
var ya da yok şeklinde ayrıldı
'''
dataset['Other installment plans'] = dataset['Other installment plans'].map({'A143':0,'A141':1,'A142':1})
'''
Housing : Konut(ev sahibi mi )
A151: kira
A152: kendi
A153: diğer
'''
dataset['Housing'] = dataset['Housing'].map({'A153':0,'A151':1,'A152':2})
'''
Job : Meslek
A171: işsiz / vasıfsız - ikamet etmeyen
A172: vasıfsız - ikamet
A173: yetenekli çalışan / resmi
A174: yönetim / serbest meslek / yüksek nitelikli çalışan / memur
'''
dataset['Job'] = dataset['Job'].map({'A171':0,'A172':0,'A173':1,'A174':2})
'''
Telephone : Telefon
A191: yok
A192: kayıtlı
'''
dataset['Telephone'] = dataset['Telephone'].map({'A191':0,'A192':1})
'''
Foreign Worker : yabancı çalışan
A201: evet
A202: hayır
'''
dataset['Foreign Worker'] = dataset['Foreign Worker'].map({'A201':0,'A202':1})
    
'''
Personal Status and Sex : medeni durum ve cinsiyet
A91: Erkek: boşanmış / ayrılmış
A92: Kadın: boşanmış / ayrılmış / evli
A93: erkek: bekar
A94: erkek: evli / dul
A95: kadın: bekar
erkek ve kadın olarak ayrıldı
'''
dataset['Personal Status and Sex'] = dataset['Personal Status and Sex'].map({'A91':0,'A92':1,'A93':0,'A94':0,'A95':1})

#One Hot Encoding
dataset = pd.concat([dataset.iloc[:,:2],pd.get_dummies(dataset['Purpose']),dataset.iloc[:,3:]],axis = 1)
dataset = dataset.drop(['Purpose'],axis=1)

#Korelasyon analizi
corr = dataset.corr()   
sns.heatmap(corr)
plt.show()
#Korelasyon analizi sonraso 'Duration' özniteliği silindi
dataset = dataset.drop(['Duration'], axis = 1)

describe = dataset.describe() 


  
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1:]

#min-max scale kullanıldı.
scaler = preprocessing.MinMaxScaler()       
X = scaler.fit_transform(X)  

#Dataset train ve test olarak ayrıldı 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
    
#Algoritmaların denenip karşılaştırılması amacıyla:
algoritmaların_maaliyeti = {}
algoritmaların_dogruluk_değeri= {}
algoritmaların_kesinlik_değeri = {}
algoritmaların_F1_score_değeri = {}
train_kümesi_dogruluk_degeri = {}
algoritmaların_k_cross_accuracy_değeri= {}

#1. Lojistik regresyon
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)                   
logr_y_pred = logr.predict(X_test) 
#Lojistik regresyon - karmaşıklık matrisi         
cm_logr = confusion_matrix(y_test,logr_y_pred)    
dogruluk,kesinlik,maaliyet, F1_score, lojistik_regresyon_degerleri  = dogruluk_oranları(cm_logr)        
k_cross_accuracy = k_cross_validation(X, Y, logr)
algoritmaların_k_cross_accuracy_değeri['Lojistik regresyonun k cross validation doğruluk değeri'] = k_cross_accuracy 
algoritmaların_dogruluk_değeri['Lojistik regresyon doğruluk değeri'] = dogruluk    
algoritmaların_kesinlik_değeri['Lojistik regresyon kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['Lojistik regresyon maaliyeti'] = maaliyet 
algoritmaların_F1_score_değeri['Lojistik regresyon F1_score değeri'] = F1_score 
lojistik_regresyon_degerleri['k cross validation accuracy'] = k_cross_accuracy



# 2. KNN    
from sklearn.neighbors import KNeighborsClassifier   
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)   
knn_y_pred = knn.predict(X_test) 
 #KNN - karmaşıklık matrisi   
cm_knn = confusion_matrix(y_test,knn_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, knn_degerleri = dogruluk_oranları(cm_knn)
k_cross_accuracy = k_cross_validation(X, Y, knn)
algoritmaların_k_cross_accuracy_değeri["KNN'in k cross validation doğruluk değeri"] = k_cross_accuracy 
algoritmaların_dogruluk_değeri['KNN doğruluk değeri'] = dogruluk
algoritmaların_kesinlik_değeri['KNN kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['KNN maaliyeti'] = maaliyet  
algoritmaların_F1_score_değeri['KNN F1_score değeri'] = F1_score 
knn_degerleri['k cross validation accuracy'] = k_cross_accuracy



# 3. SVC (SVM sınıflandırıcısı)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svs_y_pred = svc.predict(X_test)
#SVC - karmaşıklık matrisi   
cm_svc= confusion_matrix(y_test,svs_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, SVM_degerleri = dogruluk_oranları(cm_svc)
k_cross_accuracy = k_cross_validation(X, Y, svc)
algoritmaların_k_cross_accuracy_değeri["SVM'in k cross validation doğruluk değeri"] = k_cross_accuracy 
algoritmaların_dogruluk_değeri['SVM doğruluk değeri'] = dogruluk   
algoritmaların_kesinlik_değeri['SVM kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['SVM maaliyeti'] = maaliyet
algoritmaların_F1_score_değeri['SVM F1_score değeri'] = F1_score 
SVM_degerleri['k cross validation accuracy'] = k_cross_accuracy



# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)    
gnb_y_pred = gnb.predict(X_test)
#Naive Bayes - karmaşıklık matrisi  
cm_gnb= confusion_matrix(y_test,gnb_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, Naive_Bayes_degerleri = dogruluk_oranları(cm_gnb)
k_cross_accuracy = k_cross_validation(X, Y, gnb)
algoritmaların_k_cross_accuracy_değeri["Naive Bayes'in k cross validation doğruluk değeri"] = k_cross_accuracy 
algoritmaların_dogruluk_değeri['Naive Bayes doğruluk değeri'] = dogruluk  
algoritmaların_kesinlik_değeri['Naive Bayes kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['Naive Bayes maaliyeti'] = maaliyet
algoritmaların_F1_score_değeri['Naive Bayes F1_score değeri'] = F1_score 
Naive_Bayes_degerleri['k cross validation accuracy'] = k_cross_accuracy


# 5. Karar Ağacı
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dtc = base_estimator = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf = 5, max_depth=20))
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)
#Karar Ağacı - karmaşıklık matrisi    
cm_dtc = confusion_matrix(y_test,dtc_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, Karar_Agacı_degerleri = dogruluk_oranları(cm_dtc)
k_cross_accuracy = k_cross_validation(X, Y, dtc)
algoritmaların_k_cross_accuracy_değeri["Karar Ağacı'nın k cross validation doğruluk değeri"] = k_cross_accuracy 
algoritmaların_dogruluk_değeri['Karar Ağacı doğruluk değeri'] = dogruluk   
algoritmaların_kesinlik_değeri['Karar Ağacı kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['Karar Ağacı maaliyeti'] = maaliyet
algoritmaların_F1_score_değeri['Karar Ağacı F1_score değeri'] = F1_score 
Karar_Agacı_degerleri['k cross validation accuracy'] = k_cross_accuracy

    
# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(min_samples_leaf = 10)
rfc = RandomForestClassifier(min_samples_leaf = 5, max_depth=20)
rfc.fit(X_train,y_train)    
rfc_y_pred = rfc.predict(X_test)
#Random Forest - karmaşıklık matrisi 
cm_rfc = confusion_matrix(y_test,rfc_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, Random_Forest_degerleri = dogruluk_oranları(cm_rfc)
k_cross_accuracy = k_cross_validation(X, Y, rfc)
algoritmaların_k_cross_accuracy_değeri["Random Forest'ın k cross validation doğruluk değeri"] = k_cross_accuracy
algoritmaların_dogruluk_değeri['Random Forest doğruluk değeri'] = dogruluk   
algoritmaların_kesinlik_değeri['Random Forest kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['Random Forest maaliyeti'] = maaliyet
algoritmaların_F1_score_değeri['Random Forest F1_score değeri'] = F1_score 
Random_Forest_degerleri['k cross validation accuracy'] = k_cross_accuracy
#en önemli ilk 3 özniteliğin bulunması
önemlilik = rfc.feature_importances_


  
#ensemble learning:  ('rfc',rfc), ('logr',logr),
from sklearn.ensemble import VotingClassifier
ens = VotingClassifier(estimators=[('dtc', dtc), ('gnb',gnb), ('rfc',rfc), ('logr',logr)])
ens.fit(X_train, y_train)
ens_y_pred = ens.predict(X_test)
#confusion matrix of RandomForestClassifier
cm_ens = confusion_matrix(y_test,ens_y_pred)
dogruluk,kesinlik,maaliyet, F1_score, kollektif_ogrenme_degerleri = dogruluk_oranları(cm_ens)
k_cross_accuracy = k_cross_validation(X, Y, ens)
algoritmaların_k_cross_accuracy_değeri["Kollektif öğrenmenin k cross validation doğruluk değeri"] = k_cross_accuracy
algoritmaların_dogruluk_değeri['Kollektif öğrenmenin doğruluk değeri'] = dogruluk    
algoritmaların_kesinlik_değeri['Kollektif öğrenmenin kesinlik değeri'] = kesinlik
algoritmaların_maaliyeti['Kollektif öğrenmenin maaliyeti'] = maaliyet
algoritmaların_F1_score_değeri['Kollektif öğrenmenin F1_score değeri'] = F1_score 
kollektif_ogrenme_degerleri['k cross validation accuracy'] = k_cross_accuracy