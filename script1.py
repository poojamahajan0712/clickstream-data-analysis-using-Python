# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:42:43 2019

@author: Pooja.Mahajan
"""

import pandas as pd ##pandas
import numpy as np  ##numpy
import datetime
df=pd.read_csv("clickstreamdata.csv")

###converting to date time
df['datetime1']=pd.to_datetime(df['datetime'])

###sorting by time and uuid
df['diff']=df.sort_values(['UUID','datetime1'],ascending=[True,True]).groupby('UUID')['datetime1'].diff()

df['val']=1

for j in range(1,df.shape[0]):
    if((df.iloc[j,0]==df.iloc[j-1,0]) & (df.iloc[j,5]>datetime.timedelta(minutes=30))):
         df.iloc[j,6]=df.iloc[j-1,6]+1

###1. No of sessions
no_of_sessions=pd.DataFrame(df.groupby('UUID')['val'].max())
no_of_sessions.reset_index(level=0, inplace=True)

max_time=pd.DataFrame(df.groupby('UUID')['datetime1'].max())
max_time.reset_index(level=0, inplace=True)

ts=pd.merge(no_of_sessions,max_time,on=['UUID'],how='left')


########################################################################
#Subcategory to be classified into types "Sale", "Rent", "Others", "NULL"
df['subcat']=np.where(df['subcategory_name'].str.contains('Sale'),"Sale",
         (np.where(df['subcategory_name'].str.contains('Rent'),"Rent",
         (np.where(df['subcategory_name']=="NULL","NULL","Others")))))





##2. last subcategory  visited by UUID
last_sub=pd.merge(ts,df[['UUID','subcat','datetime1']],on=['UUID','datetime1'],how='left')

last_sub_cat=last_sub[['UUID','subcat']]



###3.Max subcategory visited by UUID
no_of_subcat=pd.DataFrame(df.groupby(['UUID','subcat']).size())
no_of_subcat.reset_index(level=0, inplace=True)
no_of_subcat.reset_index(level=0, inplace=True)

max_subcat=pd.DataFrame(no_of_subcat.groupby(['UUID','subcat'])[0].max())
max_subcat.reset_index(level=0, inplace=True)
max_subcat.reset_index(level=0, inplace=True)

max_Subcategory=max_subcat[['UUID','subcat']]


