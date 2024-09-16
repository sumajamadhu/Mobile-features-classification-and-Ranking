# Mobile features
import pandas as pd
import numpy as np
import pickle
data1=pd.read_csv('MobileTest.csv')
data2=pd.read_csv('MobileTrain.csv')
#outlier detection
df = pd.concat([data1.assign(ind="test"), data2.assign(ind="train")])
df=df.drop('id',axis=1)
Q1=np.percentile(df['fc'],25,interpolation='midpoint')
Q2=np.percentile(df['fc'],50,interpolation='midpoint')
Q3=np.percentile(df['fc'],75,interpolation='midpoint')
IQR=Q3-Q1
low_lim=Q1-1.5*IQR
up_lim=Q3+1.5*IQR
outlier=[]
for x in df['fc']:
    if((x>up_lim) or (x<low_lim)):
        outlier.append(x)
ind1=df['fc']>up_lim   
for i in df.loc[ind1].index:
  df.drop(i,inplace=True)
Q1=np.percentile(df['px_height'],25,interpolation='midpoint')
Q2=np.percentile(df['px_height'],50,interpolation='midpoint')
Q3=np.percentile(df['px_height'],75,interpolation='midpoint')
IQR=Q3-Q1
low_lim=Q1-1.5*IQR
up_lim=Q3+1.5*IQR
outlier=[]
for x in df['px_height']:
   if((x>up_lim) or (x<low_lim)):
      outlier.append(x)
ind1=df['fc']>up_lim   
for i in df.loc[ind1].index:
  df.drop(i,inplace=True)
#splitting as train and test
test,train=df[df['ind'].eq("test")],df[df['ind'].eq("train")]
test=test.drop('ind',axis=1)
train=train.drop('ind',axis=1)
#splitting as x and y
Y=train['price_range']
X=train.drop(['price_range','touch_screen','four_g','wifi','dual_sim','blue','three_g','m_dep','n_cores','pc'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=.2)
#Modeling
from sklearn.svm import SVC
svm_linear=SVC(kernel='linear')
a25=svm_linear.fit(x_train,y_train)
#Saving the model to disk
pickle.dump(a25,open('mobile.pkl','wb') )

