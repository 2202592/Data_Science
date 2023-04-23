#!/usr/bin/env python
# coding: utf-8

# # DATA PREPARATION

# In[5]:


import os #"""os used for file extraction"""
import glob #"""glob used for folder/file identification"""
import zipfile #"""zipfile used for unzipping"""
import numpy as np #"""numpy used for statistics purpose"""
import pandas as pd #"""pandas used for importing data and creating dataframes"""
import seaborn as sns #"""seaborn used for visulization"""
import matplotlib.pyplot as plt #"""matplot used for visulization"""
from datetime import timedelta, datetime
import warnings #"""warnings used to ignore warning windows"""
warnings.filterwarnings('ignore')


# In[27]:


survey


# In[8]:


df = pd.DataFrame()

#Creating date time based on individual columns of time, date, start time, end time etc.
survey = pd.read_excel(r'C:\Users\pc\Desktop\Data Science A2\FAREED\SurveyResults.xlsx', usecols=['ID', 'Start time', 'End time', 'date', 'Stress level'], dtype={'ID': str})
survey['Stress level'].replace('na', np.nan, inplace=True)
survey.dropna(inplace=True)

survey['Start datetime'] =  pd.to_datetime(survey['date'].map(str) + ' ' + survey['Start time'].map(str))
survey['End datetime'] =  pd.to_datetime(survey['date'].map(str) + ' ' + survey['End time'].map(str))
survey.drop(['Start time', 'End time', 'date'], axis=1, inplace=True)

# Convert SurveyResults.xlsx to GMT-00:00
daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

survey1 = survey[survey['End datetime'] <= daylight].copy()
survey1['Start datetime'] = survey1['Start datetime'].apply(lambda x: x + timedelta(hours=5))
survey1['End datetime'] = survey1['End datetime'].apply(lambda x: x + timedelta(hours=5))

survey2 = survey.loc[survey['End datetime'] > daylight].copy()
survey2['Start datetime'] = survey2['Start datetime'].apply(lambda x: x + timedelta(hours=6))
survey2['End datetime'] = survey2['End datetime'].apply(lambda x: x + timedelta(hours=6))

survey = pd.concat([survey1, survey2], ignore_index=True)
survey['datetime'] = survey['End datetime'] - survey['Start datetime']

#Creating minutes columns for labeling 
def minutes(x):
    return x.seconds/60
survey['time'] = survey['datetime'].apply(minutes)
survey.reset_index(drop=True, inplace=True)

#"""Glob will help us access to all folders inside the specified location"""
folders = glob.glob(os.path.join("Data/*"))

i = 0
#"""Looping through the folders so we can get all the data one by one and merge it into one single csv"""
while i < len(folders):
    
    #only running for one file. rest all fines is same
    if i == 1:
        break
    
    folder = folders[i]
    #"""Glob will help us access to all the files inside the folder"""
    files = glob.glob(os.path.join(folder, '*.zip'))
    #"""creating new dataframe to append all the data from each folder"""
    joined_list = pd.DataFrame()
    j = 0
    
    #"""looping through each files to unzip because each folder have multiple zip files and each zip file have
    #multiple csv we need to merge"""
    while j < len(files):
        file = files[j]
        
        #"""zipfile helps us unzip the files and access the csv. it will unzip in the .ipynb directory"""
        with zipfile.ZipFile(file) as zf:
            zf.extractall()
            #"""accessing the data"""
            acc = pd.read_csv('ACC.csv', index_col=None, header=0)
            bvp = pd.read_csv('BVP.csv', index_col=None, header=0)
            eda = pd.read_csv('EDA.csv', index_col=None, header=0)
            hr = pd.read_csv('HR.csv', index_col=None, header=0)
            temp = pd.read_csv('TEMP.csv', index_col=None, header=0)

            acc = acc.rename(columns={acc.columns[0]: 'acx', acc.columns[1]: 'acy', acc.columns[2]: 'acz'})
            bvp = bvp.rename(columns={bvp.columns[0]: 'blood_vol_pulse'})
            eda = eda.rename(columns={eda.columns[0]: 'eda'})
            hr = hr.rename(columns={hr.columns[0]: 'h_rate'})
            temp = temp.rename(columns={temp.columns[0]: 'temp'})
            print("Zip file extracted and extracted csv's")
            
            #Creating DateTime attribute for each dataset

            def process_df(df):
                start_timestamp = df.iloc[0,0]
                sample_rate = df.iloc[1,0]
                new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
                new_df['datetime'] = [(start_timestamp + i/sample_rate) for i in range(len(new_df))]
                return new_df
            
            acc = process_df(acc)
            bvp = process_df(bvp)
            eda = process_df(eda)
            hr = process_df(hr)
            temp = process_df(temp)
            print('Date Time Created')

            #Merging the Data #taken from original code because this was the best feasible method
            joined = acc.merge(eda, on='datetime', how='outer')
            joined = joined.merge(hr, on='datetime', how='outer')
            joined = joined.merge(temp, on='datetime', how='outer')
            print('Merging is done')
            
            #Fill all the null value by forward fill and backward fill instead of removing and losing the data
            
            joined.fillna(method='ffill', inplace=True)
            joined.fillna(method='bfill', inplace=True)
            joined.reset_index(inplace = True,drop = True)
            print('Fill null value is done')
            
            #"""appending the data of each folder to the variable"""
            joined_list = joined_list.append(joined)
                 
        j += 1
        joined_list['id'] = folders[i][-2:]
    #"""concating the folders together to final dataframe to save the csv at the end"""
    df = pd.concat([df,joined_list],ignore_index = True).reset_index(drop=True)
    
    print(i, ' :- is Done')
    i += 1
    
print('DONE')


# In[9]:


joined_list


# In[13]:


df = pd.DataFrame()


# In[14]:


#labeling only 15th folder because computation will take 2 hours. remaining are folders are same.
def labeling(x):
    if x>survey.time.max():
        return 2
    else:
        return 0
joined_list['Label'] = joined_list['datetime'].apply(labeling)


# In[15]:


#temporary asign for analysis
df = joined_list.copy()


# In[16]:


df


# In[17]:


#"""Singular duplicate value checker"""
df.duplicated().sum()


# In[18]:


#"""Dropping duplicates values"""
df = df.drop_duplicates().reset_index(drop = True)


# In[19]:


#"""Dropping nan values which was replaced for ibi csv which was empty for some zip files"""
df.dropna(inplace = True)


# In[20]:


#"""Saving the CSV so we can avoid multiple execution repeatatively"""
df.to_csv('allData.csv')


# # Data Preprocessing and Extraction is completed 

# In[3]:


import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc

from sklearn.metrics import classification_report


# In[4]:


#"""Loading the saved csv which was unzipped and merged from all the folders"""
new_df = pd.read_csv('allData.csv')
new_df


# In[5]:


#"""dropping unnamed: 0 because it was created when saving .to_csv which could have been avoided by ignore_index"""
new_df.drop({'Unnamed: 0','id'},axis = 1,inplace = True)


# In[6]:


gc.collect()


# In[7]:


#"""singular null value checker"""
new_df.isnull().sum()


# In[8]:


#"""info function to get the null value, columns name, data types etc."""
new_df.info()


# In[9]:


#"""Describe function to get quantile, mean and description of the data"""
new_df.describe()


# In[10]:


sns.heatmap(new_df.isnull(), cmap='viridis')
plt.show()


# In[11]:


new_df['Label'].unique()


# In[12]:


#Accidentally made labels 0 and 2 so changing 2 to 1
new_df["Label"] = new_df["Label"].replace(2,1)


# In[13]:


new_df


# In[14]:


#"""dependent variable value check for imbalance data""" """The data is imbalanced"""
new_df['Label'].value_counts()


# In[16]:


#seaborn visulization for explaining the imbalanced data


# In[17]:


sns.countplot(new_df['Label'])


# In[18]:


print("The Data is Imbalanced and the 0's label is :- ",float(len(new_df[new_df['Label'] == 0])/len(new_df[new_df['Label'] == 1]))," Times greater than 1's label")


# In[19]:


#Downsampling 0's to balance the data to match 1's shape
from sklearn.utils import resample

zeros = new_df[new_df['Label'] == 0]
ones = new_df[new_df['Label'] == 1]

print(zeros.shape)
print(ones.shape)

zeros_downsample = resample(zeros, replace=True, n_samples=len(ones), random_state=42)

print(zeros_downsample.shape)

final_df = pd.concat([zeros_downsample, ones])


# In[21]:


sns.countplot(final_df['Label'])


# In[22]:


final_df.shape


# In[23]:


final_df.corr()


# In[24]:


#"""Heat-map to check correlation visulization with the dependent variable"""
fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(final_df.corr().transpose(), cmap='inferno',annot = True)
plt.show()


# In[25]:


#"""Box plot to check the outlier of the data, which can be handled by Z-score during modelling"""
fig, ax = plt.subplots(figsize=(20,20))
plt.boxplot(final_df)
plt.show()


# In[26]:


#All the data attribute appeard to be normally distributed except fotr 4th attribute and that us fine because trhat us datetime 
#attribute


# # Spliting the data to train the model and test the model performance

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X = final_df.drop({'datetime','Label'},axis = 1)
Y = final_df['Label']


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)


# In[39]:


train_class_counts = pd.Series(y_train).value_counts()
test_class_counts = pd.Series(y_test).value_counts()

df = pd.DataFrame({'Training data': train_class_counts, 'Testing data': test_class_counts})
df.plot(kind='bar')
plt.show()


# In[30]:


gc.collect()


# # Feature Selection

# In[31]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[32]:


# feature selection """Pre defined fucntion of feature selection taken from web"""
def select_features(X_train, y_train, X_test):
 # configure to select all features
 fs = SelectKBest(score_func=f_classif, k='all')
 # learn relationship from training data
 fs.fit(X_train, y_train)
 # transform train input data
 X_train_fs = fs.transform(X_train)
 # transform test input data
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs, fs


# In[33]:


#Tells us the scores of features which are best for model building 
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))


# In[34]:


#Shows Which Feature is best for model building
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# In[35]:


gc.collect()


# # Model Building

# # SGD Classifier

# In[40]:


from sklearn import linear_model

SGD = linear_model.SGDClassifier()
SGD.fit(X_train,y_train)


# In[41]:


SGD_pred = SGD.predict(X_test)


# In[42]:


print(classification_report(y_test, SGD_pred))


# In[52]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[53]:


#Confusion matrix representing True positive(TP), True Negative(NP), False Positive(FP), False Negative(FN)
con1 = confusion_matrix(y_test, SGD_pred)
con1


# In[60]:


disp = ConfusionMatrixDisplay(confusion_matrix=con1)
disp.plot()


# # Naive Bayes

# In[62]:


from sklearn.naive_bayes import BernoulliNB

NB = BernoulliNB()

NB.fit(X_train,y_train)


# In[63]:


NB_pred = NB.predict(X_test)


# In[64]:


print(classification_report(y_test, NB_pred))


# In[65]:


#Confusion matrix representing True positive(TP), True Negative(NP), False Positive(FP), False Negative(FN)
con2 = confusion_matrix(y_test, NB_pred)
con2


# In[66]:


disp2 = ConfusionMatrixDisplay(confusion_matrix=con2)
disp2.plot()


# # Random Forest

# In[67]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(max_depth=2, random_state=0)
RF.fit(X_train,y_train)


# In[68]:


RF_pred = RF.predict(X_test)


# In[69]:


print(classification_report(y_test, RF_pred))


# In[70]:


#Confusion matrix representing True positive(TP), True Negative(NP), False Positive(FP), False Negative(FN)
con3 = confusion_matrix(y_test, RF_pred)
con3


# In[71]:


disp3 = ConfusionMatrixDisplay(confusion_matrix=con3)
disp3.plot()


# In[ ]:




