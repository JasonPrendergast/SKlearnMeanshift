#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_csv('titanic.csv')
print(df.head())
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())
def handle_non_numerical_data(df):
    #colums df
    columns = df.columns.values
    for column in columns:
        #each column create a list of unique values
        text_digit_vals = {}
        def convert_to_int(val):
           return text_digit_vals[val]
#       looking for data type within the olumn which is not a number value
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
           # print(df[column].name)
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            #unique values given number id as new value to replace text with a int
            for unique in unique_elements:
               # print(unique)
                #check if value exists in list of unique values
                 #eg [bob,1],[glob,2],[frog,3]
                if unique not in text_digit_vals:
                    #increment new unique value within the list
                   
                    text_digit_vals[unique] = x
                    x+=1
#           Call convert to int inside columns 
            df[column] = list(map(convert_to_int, df[column]))

    return df
df = handle_non_numerical_data(df)
print(df.head())
#X is data - survived
X = np.array(df.drop(['survived'], 1).astype(float))
#convert to scala for improved accuracy
X = preprocessing.scale(X)
#y is the value of survived to be checked against after the classifier makes a prediction eg test data for the model
y = np.array(df['survived'])
#Call MATH MAGIC random nodes selected then eucidian distance across whole data set BOOM and it just works... mental
clf = MeanShift()
#pickle time?
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
#moddify copied static dataset to add cluster_group column 
original_df['cluster_group']=np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
    
n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    #print(temp_df.head())

    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    survival_rate = len(survival_cluster) / len(temp_df)
    #print(i,survival_rate)
    survival_rates[i] = survival_rate

print('survival_rates') 
print(survival_rates)

for i in range(n_clusters_):
    print('################ survival_rates per group#####################')
    print(survival_rates[i])
    print(i)
    print('############### Raw Grouping ######################')
    print(original_df[ (original_df['cluster_group']==i) ])
    print('############## describe #######################')
    print(original_df[ (original_df['cluster_group']==i) ].describe())
    #print('#####################################')
    cluster_current = (original_df[ (original_df['cluster_group']==i) ])
    pclasses = len(np.unique(cluster_current['pclass']))
    print('PCLASS')
    print(pclasses)
    for pclass in range(3):
        print('###### survival rate per pclass in each group #########')
        pclass = pclass+1
        print(pclass)
        cluster_current_fc = (cluster_current[ (cluster_current['pclass']==pclass) ])
        print(cluster_current_fc.describe())
   
    
    

#print(original_df[ (original_df['cluster_group']==1) ])
#print(original_df[ (original_df['cluster_group']==2) ])
#print(original_df[ (original_df['cluster_group']==3) ])
