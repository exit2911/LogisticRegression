
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




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
