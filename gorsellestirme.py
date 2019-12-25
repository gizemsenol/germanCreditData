# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def gorsellestirme():
    
    #Veri tanıtımı
    gorsel_data = pd.read_csv('german.csv')
    describe = gorsel_data.describe()

    gorsel_data['Housing'] = gorsel_data['Housing'].map({'A153':'diger','A151':'kira','A152':'kendi'}) 
    gorsel_data['Purpose'] = gorsel_data['Purpose'].map({'A43':'radyo/tv','A46':'eğitim','A45':'tadilat','A44':'ev aleti',
               'A41':'araba','A40':'araba(yeni)','A42':'mobilya','A49':'iş','A48':'yeniden eğitim'})
    gorsel_data['Personal Status and Sex'] = gorsel_data['Personal Status and Sex'].map({'A91':'erkek','A92':'kadın',
               'A93':'erkek','A95':'kadın','A94':'erkek'})
    gorsel_data['Job'] = gorsel_data['Job'].map({'A171':'işsiz','A172':'vasıfsız','A173':'yetenekli','A174':'yüksek nitelikli'})
    gorsel_data['Foreign Worker'] = gorsel_data['Foreign Worker'].map({'A201':'evet','A202':'hayır'})
    
    #hangi kategoride kaç veri var?
    
    gorsel_data['Checking Account'].value_counts().plot(kind='pie')
    plt.show()
    
    describe = gorsel_data['Duration'].describe()
    sns.distplot(gorsel_data['Duration'], hist = False)
    plt.show()
    
    gorsel_data['Credit History'].value_counts().plot(kind='pie')
    plt.show()
    
    sns.countplot(y="Purpose", data=gorsel_data)
    plt.show()
    
    describe = gorsel_data['Credit amount'].describe()
    sns.distplot(gorsel_data['Credit amount'], hist = False)
    plt.show()
    
    gorsel_data['Savings Account'].value_counts().plot(kind='pie')
    plt.show()
    
    gorsel_data['Present employment since'].value_counts().plot(kind='pie')
    plt.show()
    
    sns.countplot(y="Installment rate", data=gorsel_data)
    plt.show()
    
    gorsel_data['Personal Status and Sex'].value_counts().plot(kind='pie')
    plt.show()
    
    sns.countplot(y="Other Debtors/Guarantors", data=gorsel_data)
    plt.show()
    
    describe = gorsel_data['Present Residence Since'].describe()
    sns.distplot(gorsel_data['Present Residence Since'], hist = False)
    plt.show()
    
    gorsel_data['Property'].value_counts().plot(kind='pie')
    plt.show()
    
    describe = gorsel_data['Age'].describe()
    sns.distplot(gorsel_data['Age'], hist = False)
    plt.show()
    
    sns.countplot(y="Other installment plans", data=gorsel_data)
    plt.show()
            
    gorsel_data['Housing'].value_counts().plot(kind='pie')
    plt.show()
    
    sns.countplot(x="Existing Credits", data=gorsel_data)
    describe = gorsel_data['Existing Credits'].describe()
    plt.show()
    
    gorsel_data['Job'].value_counts().plot(kind='pie')
    plt.show()
    
    sns.countplot(x="Liable to Provide", data=gorsel_data)
    describe = gorsel_data['Liable to Provide'].describe()
    plt.show()
    
    gorsel_data['Telephone'].value_counts().plot(kind='pie')
    plt.show()
    
    gorsel_data['Foreign Worker'].value_counts().plot(kind='pie')
    plt.show()
    
    gorsel_data['Customer Risk Type'].value_counts().plot(kind='pie')
    plt.show()
     
    
    #görselleştirme
    gorsel_data = pd.read_csv('german.csv')

    
    emekli = gorsel_data[(gorsel_data['Age']) >= 65]
    emekli['Age'] = 'emekli'
    yasli = gorsel_data[(gorsel_data['Age']) > gorsel_data['Age'].median()]
    yasli = yasli[(yasli['Age']) < 65]
    yasli['Age'] = 'yaşlı'
    genc = gorsel_data[(gorsel_data['Age']) <= gorsel_data['Age'].median()]
    genc['Age'] = 'genç'
    gorsel_data = pd.concat([yasli,genc,emekli],axis = 0)
    
    gorsel_data['Personal Status and Sex'] = gorsel_data['Personal Status and Sex'].map({'A91':'erkek','A92':'kadın','A93':'erkek','A94':'erkek','A95':'kadın'})

    
    #Present Residence Since : ne zamandan beri ikamet ettiği 
    sns.barplot(x = 'Present Residence Since', y = 'Customer Risk Type', hue = 'Job', data = gorsel_data)
    plt.show() #yeni ikamet adresini değiştirmiş ve işsiz olan kişiler riskli grupta.
    sns.barplot(x = 'Present Residence Since', y = 'Customer Risk Type', hue = 'Housing', data = gorsel_data)
    plt.show() #ikamet adresi yakın zamanda değişmiş ve kirada olanlar riskli kategoride.
    #tablo 4.2.1
    
    #age ve Liable to Provide
    sns.barplot(x = 'Age', y = 'Customer Risk Type', hue = 'Liable to Provide', data = gorsel_data)
    plt.show() #Ortalama Yaşın üzerinde olan kişilerde bakmakla yükümlü olduğu kişi sayısı fazla ise daha riskli fakat ortlama yaşı  altı için tam tersi
    #tablo 4.2.2

    #age ve Personal Status and Sex
    sns.barplot(x = 'Age', y = 'Customer Risk Type', hue = 'Personal Status and Sex', data = gorsel_data)
    plt.show() #Yaş ilerledikçe kadınlar erkeklerden daha az riskli olur.
    #tablo 4.2.3