

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model

bank = pd.read_csv('/Users/VyHo/Downloads/bank-additional/bank-additional-full.csv',sep = ';')
#check the headers
bank.head()
#check the column names
bank.columns.values
#check data types
bank.dtypes
#convert a variable to binary. converting true/false to 1/0
bank['y'] = (bank['y'] == 'yes').astype(int)
bank['education'].unique()

#replacing row values

bank['education']=np.where(bank['education'] =='basic.9y', 'Basic',
bank['education'])
bank['education']=np.where(bank['education'] =='basic.6y', 'Basic',
bank['education'])
bank['education']=np.where(bank['education'] =='basic.4y', 'Basic',
bank['education'])
bank['education']=np.where(bank['education'] =='university.degree',
'University Degree', bank['education'])
bank['education']=np.where(bank['education'] =='professional.course',
'Professional Course', bank['education'])
bank['education']=np.where(bank['education'] =='high.school', 'High School', bank['education'])
bank['education']=np.where(bank['education'] =='illiterate',
'Illiterate', bank['education'])
bank['education']=np.where(bank['education'] =='unknown', 'Unknown',
bank['education'])

bank['y'].value_counts()
bank.groupby('y').mean() #shows the means for all numeric variables groupby the dependent variable
bank.groupby('education').mean()

#plotting education levels (x) vs. the dependent variable
pd.crosstab(bank.education,bank.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Level')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
#the graph shows education level is a great predictor



table = pd.crosstab(bank.marital,bank.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital')
plt.ylabel('Proportion of Customers')

#the marital proportion graph shows marital status is not a good predictor 

pd.crosstab(bank.day_of_week,bank.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')

#day of week seem to be a good predictor

bank.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

pd.crosstab(bank.poutcome,bank.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')

#poutcome - marketing camp outcome. marketing camp sucess -> purchase success

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:

    cat_list = pd.get_dummies(bank[var], prefix=var) #built in function that dummifize the given list of columns
    bank1=bank.join(cat_list)
    bank=bank1
    
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
#dropping the original columns that are not dummy columns

bank_vars = bank.columns.values.tolist()

to_keep = [i for i in bank_vars if i not in cat_vars]

bank_final = bank[to_keep]
bank_final.columns.values
#making list of independent variables and dependent variable

bank_final_vars = bank_final.columns.values.tolist()
Y = ['y']
X = [i for i in bank_final_vars if i not in Y]

#final table name is bank_final

#features selection


model = LogisticRegression()

rfe = RFE(model,12)
rfe = rfe.fit(bank_final[X],bank_final[Y])
print(rfe.support_)
print(rfe.ranking_)

#using the boolean list to spit out the new array
boolean = rfe.support_.tolist()
new_X = list(compress(X,boolean))

X = bank_final[new_X]
Y = bank_final['y']

#run logistic regression using statsmodels.api

logit_model = sm.Logit(Y,X)
result = logit_model.fit()
print(result.summary())

clf = linear_model.LogisticRegression()
clf.fit(X,Y)
clf.score(X,Y)

pd.DataFrame(zip(X.columns,np.transpose(clf.coef_))) #what is clf.coef

#what is spliting data to traning and testing sets
