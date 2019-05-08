# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:42:43 2019

@author: Pooja.Mahajan
"""
##importing pandas and numpy
import pandas as pd
import numpy as np
import datetime
df=pd.read_csv("D:/Pooja/quikr/Assignment0/clickstreamdata.csv")

###converting to date time
df['datetime1']=pd.to_datetime(df['datetime'])

###sorting by time and uuid and groupby by user to calculate difference between time in subsequent event records

df['diff']=df.sort_values(['UUID','datetime1'],ascending=[True,True]).groupby('UUID')['datetime1'].diff()

##creating a column val with defualt value 1 to capture no. of sessions in cumulative manner
df['val']=1

##iterating through each record , for same user if diff is  gt 30 min inc val column value by 1 from the prev rec val value for that user 
for j in range(1,df.shape[0]):
    if((df.iloc[j,0]==df.iloc[j-1,0]) & (df.iloc[j,5]>datetime.timedelta(minutes=30))):
         df.iloc[j,6]=df.iloc[j-1,6]+1
    elif((df.iloc[j,0]==df.iloc[j-1,0])):
        df.iloc[j,6]=df.iloc[j-1,6]

########################1. No of sessions
no_of_sessions=pd.DataFrame(df.groupby('UUID')['val'].max())
no_of_sessions.reset_index(level=0, inplace=True)

### Filling missing values in Subcategory
df['subcategory_name']=df['subcategory_name'].fillna("NULL")

###merged previous dataframe to get uuid ,no. of sessions and latest timestamp.
max_time=pd.DataFrame(df.groupby('UUID')['datetime1'].max())
max_time.reset_index(level=0, inplace=True)

ts=pd.merge(no_of_sessions,max_time,on=['UUID'],how='left')


####################2. last subcategory  visited by UUID
last_sub=pd.merge(ts,df[['UUID','subcategory_name','datetime1']],on=['UUID','datetime1'],how='left')

last_sub_cat=last_sub[['UUID','subcategory_name']]



###################3.Max subcategory visited by UUID
no_of_subcat=pd.DataFrame(df.groupby(['UUID','subcategory_name']).size())
no_of_subcat.reset_index(level=0, inplace=True)
no_of_subcat.reset_index(level=0, inplace=True)

    
max_subcat=no_of_subcat.loc[no_of_subcat.groupby(['UUID'])[0].idxmax()]


max_Subcategory=max_subcat[['UUID','subcategory_name']]



res1=pd.merge(no_of_sessions,last_sub_cat,on=['UUID'],how="left")
res1.rename(columns={'val':'no_of_sessions','subcategory_name':'last_subcat_visited'},inplace=True)
res1=pd.merge(res1,max_Subcategory,on=['UUID'],how="left")
res1.rename(columns={'subcategory_name':'max_subcat_visited'},inplace=True)

########################################################################################################################
#Subcategory to be classified into types "Sale", "Rent", "Others", "NULL"

res1['last_subcattype_visited']=np.where(res1['last_subcat_visited'].str.contains('Sale'),"Sale",
         (np.where(res1['last_subcat_visited'].str.contains('Rent'),"Rent",
         (np.where(res1['last_subcat_visited']=="NULL",res1['last_subcat_visited'],"Others")))))

res1['max_subcattype_visited']=np.where(res1['max_subcat_visited'].str.contains('Sale'),"Sale",
         (np.where(res1['max_subcat_visited'].str.contains('Rent'),"Rent",
         (np.where(res1['max_subcat_visited']=="NULL",res1['max_subcat_visited'],"Others")))))



res1.to_csv("assignment0.csv")











