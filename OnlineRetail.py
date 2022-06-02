# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:11:27 2022

@author: Adira Islah (6181801078)
"""

import pandas as pd
import pycountry_convert as pc
from apyori import apriori
from datetime import datetime
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('OnlineRetail.csv')
df.head()

df['Country'].value_counts() # melihat ada country apa saja

# untuk cek apakah kota tersebut terdapat kode benuanya tidak (continent_name)
country_code = pc.country_name_to_country_alpha2("Australia", cn_name_format = "default")
print(country_code)
continent_name = pc.country_alpha2_to_continent_code(country_code)
print(continent_name)

#Penyiapan data

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df.dropna() # Membersihkan missing value
df = df[(df.Quantity>0)] 

df= df[~df['Country'].str.contains('Unspecified')] # hapus country bernama Unspecified

#buat list code benua
res_code = []
for index, row in df.iterrows():
    if (row['Country']=="EIRE"):
        res_code.append("EU")
    elif(row['Country']=="Channel Islands"):
        res_code.append("EU")
    elif(row['Country']=="European Community"):
        res_code.append("EU")
    elif(row['Country']=="RSA"):
        res_code.append("AF")
    else:
        inputcountrycode = pc.country_name_to_country_alpha2(row['Country'], cn_name_format="default")
        continent_code = pc.country_alpha2_to_continent_code(inputcountrycode)
        res_code.append(continent_name)

df ['benua'] = res_code # masukin list res_code ke dataframe
print (df)

#============BENUA EROPA===============

#filter hanya untuk benua eropa
basket = (df[df['benua'] =="EU"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

#Applying Apriori
apriori_start = datetime.now()
apriori_frequent_itemsets_eu = apriori(basket_sets, min_support=0.02, use_colnames=True)
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))

#hasil aturan asosiasi 
rules_apriori_eu = association_rules(apriori_frequent_itemsets_eu, metric="lift", min_threshold=3)
rules_apriori_eu = rules_apriori_eu.sort_values(['confidence', 'lift'], ascending = [False, False]) # mengurutkan confidence dan lift yang paling tinggi
rules_apriori_eu.head()

rules_apriori_eu[ (rules_apriori_eu['lift'] > 3) & (rules_apriori_eu['confidence'] >= 0.5)]
rules_apriori_eu.head()

#Applying FP-Growth
fpgrowth_start = datetime.now()
fpgrowth_frequent_itemsets_eu = fpgrowth(basket_sets, min_support=0.02, use_colnames=True)
fpgrowth_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("FP-Growth Running Time: ", str(running_time))

#asc_rules2_2 = association_rules(fpgrowth_frequent_itemsets2,  metric="confidence", min_threshold=0.2)
fp_rules_eu = association_rules(fpgrowth_frequent_itemsets_eu,  metric="lift", min_threshold=3)

#asc_rules2 = asc_rules2[ (asc_rules2['lift'] > 3) ]
fp_rules_eu_2 = fp_rules_eu[ (fp_rules_eu['confidence'] >= 0.5) & (fp_rules_eu['lift'] > 3) ]

#============BENUA ASIA=============

#filter hanya untuk benua asia
basket2 = (df[df['benua'] =="AS"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units2(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_sets2 = basket2.applymap(encode_units2)

#Applying Apriori
apriori_start = datetime.now()
apriori_frequent_itemsets_as = apriori(basket_sets2, min_support=0.07, use_colnames=True)
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))

rules_apriori_as = association_rules(apriori_frequent_itemsets_as, metric="lift", min_threshold=3)
rules_apriori_as = rules_apriori_as.sort_values(['confidence', 'lift'], ascending = [False, False]) # mengurutkan confidence dan lift yang paling tinggi
rules_apriori_as.head()

rules_apriori_as[ (rules_apriori_as['lift'] > 3) & (rules_apriori_as['confidence'] >= 0.5)]
rules_apriori_as.head()

#Applying FP-Growth
fpgrowth_start = datetime.now()
fpgrowth_frequent_itemsets_as = fpgrowth(basket_sets2, min_support=0.07, use_colnames=True)
fpgrowth_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("FP-Growth Running Time: ", str(running_time))

#asc_rules3_2 = association_rules(fpgrowth_frequent_itemsets3,  metric="confidence", min_threshold=0.2)
fp_rules_as = association_rules(fpgrowth_frequent_itemsets_as,  metric="lift", min_threshold=3)
#asc_rules3_3 = asc_rules3[ (asc_rules3['lift'] > 3) ]
fp_rules_as_2 = fp_rules_as[ (fp_rules_as['confidence'] >= 0.5) & (fp_rules_as['lift'] > 3) ]

#===========BENUA AUSTRALIA===========

#filter hanya untuk benua australia
basket3 = (df[df['Country'] =="Australia"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units3(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets3 = basket3.applymap(encode_units3)

#Applying Apriori
apriori_start = datetime.now()
apriori_frequent_itemsets_au = apriori(basket_sets3, min_support=0.07, use_colnames=True)
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))

rules_apriori_au = association_rules(apriori_frequent_itemsets_au, metric="lift", min_threshold=3)
# mengurutkan confidence dan lift yang paling tinggi
rules_apriori_au = rules_apriori_au.sort_values(['confidence', 'lift'], ascending = [False, False])
rules_apriori_au.head()

rules_apriori_au[ (rules_apriori_au['lift'] > 3) & (rules_apriori_au['confidence'] >= 0.5)]
rules_apriori_au.head()

#Applying FP-Growth
fpgrowth_start = datetime.now()
fpgrowth_frequent_itemsets_au = fpgrowth(basket_sets3, min_support=0.07, use_colnames=True)
fpgrowth_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("FP-Growth Running Time: ", str(running_time))


#asc_rules3_2 = association_rules(fpgrowth_frequent_itemsets3,  metric="confidence", min_threshold=0.2)
fp_rules_au = association_rules(fpgrowth_frequent_itemsets_au,  metric="lift", min_threshold=3)
#asc_rules3_3 = asc_rules3[ (asc_rules3['lift'] > 3) ]
fp_rules_au_2 = fp_rules_au[ (fp_rules_au['confidence'] >= 0.5) & (fp_rules_au['lift'] > 3) ]

