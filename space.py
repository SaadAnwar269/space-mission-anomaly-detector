# Hey This is Saad Anwar and this is my prototype implementation for the Selection Round.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
data=pd.read_csv('Dataset/space_missions_dataset.csv')

###print(data.head())

print(data.info())
#printed info to get an idea of datatypes of all parameters

#Checking for missing values using isnull function cascaded with sum
print(data.isnull().sum())

cols_cat=data.select_dtypes(include='object').columns
###print("\n Categorical Columns\n",data[cols_cat].head())
print(data[cols_cat].shape)

#Showcasing 2 different approaches to split into columns. Might be ineffective at some point but worth practicing
num_cols=data.drop(columns=data.select_dtypes(include='object').columns)
###print("\nNumerical Columns \n",num_cols.head())
print(num_cols.shape)

print("Unique Values of each categorical column : \n", data[cols_cat].nunique())
#Since the unique values of Target Type, Target Name, Misssion Type and Launch Vehicle are easily quanitifiable.
#We can enumerate these to use as one hot encoded values or label encoding based on type of data

encoded_cat = pd.get_dummies(data[cols_cat], columns=['Target Type', 'Target Name', 'Mission Type', 'Launch Vehicle'])

#print(encoded_cat.head())

#Now that we are done with the categorical column preprocessing let's preprocess the Numerical Columns to make sure none of them overpowers the other

scaler=StandardScaler()
scaled_num=scaler.fit_transform(num_cols)
scaled_num = pd.DataFrame(scaled_num, columns=num_cols.columns)


new_data=pd.concat([encoded_cat,scaled_num],axis=1)
#print(new_data.head())
#For Isolation Forest, we will need to remove the categorical columns which haven't been converted to numeric
new_data_clean=new_data.drop(columns=['Launch Date','Mission ID','Mission Name'])
#Now that the general preprocessing has been done, We will move to detecting anomalies in the data such as missions which did not yield much but cost a lot or similar anomalies

from sklearn.ensemble import IsolationForest

#Isolation forest is tree based anomaly detector that makes use of unsupervised learning techniques
iso_for=IsolationForest(n_estimators=100,contamination=0.15,random_state=42)

#I have set contamination to 0.05 initally based on the anomalies I have observed in the dataset being minimal.
#This value will be set and optimized later on basis of trial and error method
iso_for.fit(new_data_clean)
anomaly_labels=iso_for.predict(new_data_clean)
new_data_clean['Anomaly']= anomaly_labels
result= data.copy()
result['Anomaly']=anomaly_labels
print('The total number of anomalies detected with contamination set to 15% are : ',result['Anomaly'].value_counts())
#Printing out some for the console output
anomalies = result[result['Anomaly'] == -1]
print(anomalies.head())

#Now after detecting these anomalies, we need to sumarize on what basis are these different from the entries being considered normal

#We will compare against normal ones how they differ
print(result.groupby('Anomaly').mean(numeric_only=True))

#As per my intuition I believe that a space mission costing absurd amount of $$ might be considered anomalous so I will print some based on highest $$
print(anomalies.sort_values(by='Mission Cost (billion USD)', ascending=False).head())

#Similarly Low Scientific Yield is Also a cause to worry so I will consider that as well
print(anomalies.sort_values(by='Scientific Yield (points)').head())


#After detection it is always more presentable to add infographics so Iwwill use matplotplib to visualize results
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
colors = result['Anomaly'].map({1: 'blue', -1: 'red'})  # red = anomaly

plt.scatter(result['Mission Cost (billion USD)'],
            result['Scientific Yield (points)'],
            c=colors, alpha=0.6)

plt.xlabel("Mission Cost (billion USD)")
plt.ylabel("Scientific Yield (points)")
plt.title("Mission Cost vs Scientific Yield (Anomalies in Red)")
plt.grid(True)
plt.show()

#The red dots represent missions that stood out as being potentially anomolous. Though most of them are correctly identified there are some mis identifications as well such as 2 red dots at the top right corner.

anomalies.to_csv('Outputs/anomalies_15%.csv', index=False)

# Saving to a csv file to compare against different levels of contamination to effectively compare and perform trial and error based optimization

# For now I have plotted just the Cost against yield plot however there are many factors to determine whether a mission is anomalous or not and for that the complete report will be added to the Outputs folder for each level of contamination
feature_columns = [col for col in result.columns if col not in ['Anomaly', 'Mission ID', 'Mission Name', 'Launch Date', 'Launch Vehicle']]
stats = result[feature_columns].describe().T[['mean', 'std']]

def explain_anomaly(row, stats, threshold=2):
    reasons = []
    for col in stats.index:
        value = row[col]
        mean = stats.loc[col, 'mean']
        std = stats.loc[col, 'std']
        if std == 0:
            continue  # skip if no variation
        z_score = (value - mean) / std
        if abs(z_score) > threshold:
            direction = "high" if z_score > 0 else "low"
            reasons.append(f"{col} is unusually {direction} (z={z_score:.2f})")

    if not reasons:
        reasons.append("Subtle multivariate anomaly (no single feature has a high deviation)")

    return reasons

#Based on the z score I have added an explanation of the anomaly if possible to interpret on the deviation of a single feature but if not heavily influenced by a single feature and it is still being detected as anomalous then it is due to the subtle deviation of multiple variables.

anomalies['Reasons'] = anomalies.apply(lambda row: explain_anomaly(row, stats), axis=1)

anomalies[['Mission ID', 'Mission Name', 'Reasons']].to_csv("Outputs/anomaly_report_15%.csv", index=False)

#I have tried this out on different contamination levels for validation

#Since a simple statistical model would not understand the relevance of the attribute itself so it cannot infer directly to anomalous bad missions or good missions it just captures the outliers for now... To Identify specifically anomolous missions in which the yield is low and cost is high that would need to be weighted and explained explicityl
#Onw way to achieve this is to use rule based anomaly detection to specifically capture our required anomalies

rule_based_flags = (result['Mission Cost (billion USD)'] > 2.0) & (result['Scientific Yield (points)'] < 30)
result['Rule_Based_Anomaly'] = rule_based_flags.astype(int)
rule_based_anomalies = result[result['Rule_Based_Anomaly'] == 1]
rule_based_anomalies[['Mission ID', 'Mission Name', 'Mission Cost (billion USD)', 'Scientific Yield (points)']].to_csv("Outputs/rule_based_anomalies.csv", index=False)

#This is a preliminary effort on my part and I would love to keep further expanding my knowledge and potentially work on more specialized provided datasets to improve my work as well as showcase my ability

