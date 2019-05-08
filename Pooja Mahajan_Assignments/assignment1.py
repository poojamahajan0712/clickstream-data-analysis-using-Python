# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:49:04 2019

@author: Pooja.Mahajan
"""

import pandas as pd
import numpy as np

mn=pd.read_csv("D:/Pooja/quikr/Assignment1/realestate_brokers_main.csv",sep=";")
tx=pd.read_csv("D:/Pooja/quikr/Assignment1/realestatebrokeraccounttransaction.csv",sep=";")
action=pd.read_csv("D:/Pooja/quikr/Assignment1/action_mapping.csv",sep=",")


#1. Modification of field transaction_type to be applied to TX:
#IF (transaction_type = 'adjustment_add' AND remark contains 'CFP') THEN 'topup'
#ELSE transaction_type"""


tx['transaction_type']=np.where((tx['transaction_type']=='adjustment_add') & (tx['remark'].str.contains('CFP')),"topup",tx['transaction_type'])

#2. Find the pre and post transaction balance for each transaction in TX by mapping current wallet balance from MN 
#and tracing back into the transaction history based on increments and decrements."""

tx=pd.merge(tx,mn,on='broker_id',how='left')
tx['transaction_time']=pd.to_datetime(tx['transaction_time'])
###sorting transactions by each brokerid in decreasing order of transaction time
tx=tx.sort_values(['broker_id','transaction_time'],ascending=[True,False])

###creating two columns pre and post .
tx['post']=tx['account_credits']
tx['pre']=0
tx1=tx[1:1000]

###updating "post" column by keeping latest transaction with "accountcredit" value from mn and setting rest to 0
for i in range(1,tx1.shape[0]):
    if(tx1.iloc[i,0]==tx1.iloc[i-1,0]):
        tx1.iloc[i,7]=0
tx1['flag']=np.where(tx1['post']!=0,1,0)

####updating pre column for latest transaction for each broker i.e flag=1
for i in range(tx1.shape[0]):
    print(i)
    if(tx1.iloc[i,9]==1):
        if((tx1.iloc[i,2]=='transfer') | (tx1.iloc[i,2]=='deduct') | (tx1.iloc[i,2]=='adjustment_deduct') |  (tx1.iloc[i,2]=='ad_transfer')):
            tx1.iloc[i,8]=tx1.iloc[i,7]+tx1.iloc[i,1]
        elif((tx1.iloc[i,2]=='topup') |(tx1.iloc[i,2]=='receive') |(tx1.iloc[i,2]=='refund') |(tx1.iloc[i,2]=='adjustment_add') |(tx1.iloc[i,2]=='moneyback') |(tx1.iloc[i,2]=='manual corrections') ):
            tx1.iloc[i,8]=tx1.iloc[i,7]-tx1.iloc[i,1]


####updating pre and post column for other transactions where flag=0  , index 7 - "post" and 8 - "pre"    
for i in range(1,tx1.shape[0]):
    if((tx1.iloc[i,9]!=1) & (tx1.iloc[i,0]==tx1.iloc[i-1,0])):
         tx1.iloc[i,7]=tx1.iloc[i-1,8]
         if((tx1.iloc[i,2]=='transfer') | (tx1.iloc[i,2]=='deduct') | (tx1.iloc[i,2]=='adjustment_deduct') |  (tx1.iloc[i,2]=='ad_transfer')):
            tx1.iloc[i,8]=tx1.iloc[i,7]+tx1.iloc[i,1]
         elif((tx1.iloc[i,2]=='topup') |(tx1.iloc[i,2]=='receive') |(tx1.iloc[i,2]=='refund') |(tx1.iloc[i,2]=='adjustment_add') |(tx1.iloc[i,2]=='moneyback') |(tx1.iloc[i,2]=='manual corrections') ):
            tx1.iloc[i,8]=tx1.iloc[i,7]-tx1.iloc[i,1]


#3. For each transaction in TX, find the previous datetime when wallet balance went below 20 credits.
# The most recent time in the past should be considered. Call this field previous_expiry.
#If balance has not dropped below 20 in the past then take value 01Jan1980."""
tx1['prev_expiry']=0
tx1=tx1.sort_values(['broker_id','transaction_time'],ascending=[True,True])

for i in range(1,tx1.shape[0]):
    if((tx1.iloc[i,0]==tx1.iloc[i-1,0])):
        if(tx1.iloc[i,7]<20):
            tx1.iloc[i,10]=tx1.iloc[i-1,3]
        else:
            tx1.iloc[i,10]=pd.to_datetime("1980-01-01")

tx1['prev_expiry']=np.where(tx1['prev_expiry']==0,pd.to_datetime("1980-01-01"),tx1['prev_expiry'])
       
            
#4. For each transaction in TX, find the datatime in the future when wallet balance would go below 20 credits.
#The most recent time in the future should be considered. Call this field next_expiry.
#If balance has not dropped below 20 in the future then take value 01Jan2050.

tx1['next_expiry']=0
tx1=tx1.sort_values(['broker_id','transaction_time'],ascending=[True,True])

for i in range(1,tx1.shape[0]):
    if((tx1.iloc[i,0]==tx1.iloc[i-1,0])):
        if(tx1.iloc[i-1,7]<20):
            tx1.iloc[i-1,11]=tx1.iloc[i,3]
        else:
            tx1.iloc[i-1,11]=pd.to_datetime("2050-01-01")

tx1['next_expiry']=np.where(tx1['next_expiry']==0,pd.to_datetime("2050-01-01"),tx1['next_expiry'])




#5. Subset broker_id,transaction_time,account_credits,previous_expiry,next_expiry for only transaction_type = 'topup'

op=tx1[tx1['transaction_type']=='topup']
op=op[['broker_id','transaction_time','account_credits','prev_expiry','next_expiry']]

op.to_csv('assignment1.csv',index=False)








                     
