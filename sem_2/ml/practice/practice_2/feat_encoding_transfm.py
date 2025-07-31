from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
import pandas as pd
#Load dataset
df=pd.read_csv('/home/ibab/Downloads/archive/Heart.csv',index_col=0)
print(df['ChestPain'].unique())
#feature encoding
#for chest pain type
chest_pain_order=[['typical','nontypical','nonanginal','asymptomatic']]
#using skl
ord_enc=OrdinalEncoder(categories=chest_pain_order)
#trans the row
df[['ChestPain']]=ord_enc.fit_transform(df[['ChestPain']])
#Ohe
#Thal
print(df['Thal'].unique())
ohe=OneHotEncoder(sparse_output=False)
thal_enc=ohe.fit_transform(df[['Thal']])
thal_df=pd.DataFrame(thal_enc,columns=ohe.get_feature_names_out(['Thal']))
df=pd.concat([df.drop(columns=['Thal']),thal_df],axis=1)
print(df.head())
print(df.columns)
