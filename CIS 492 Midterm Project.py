#!/usr/bin/env python
# coding: utf-8

# In[131]:


#Task 1 (20 points): Get familiar with the dataset. You can do some statistic analysis (table or plotting) for
#each feature. You can also compare the feature difference between the normal data and abnormal data. Put
#your analysis within 2 pages in the report.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[135]:


def process_file(filename, anomaly=1):
    df = pd.read_csv(filename)
    df["anomaly"] = anomaly
    if anomaly:
        df["id"] = int(filename.split("_")[-1][:-4])
    else:
        
        df["id"] = int(filename.split("\\")[-1][:-4])
    return df


# In[136]:


train = pd.read_csv("normal/training-data.csv")


# In[137]:


data = []

#for file in os.listdir("anomaly/VALID_ANOMALY"):
   #df = process_file(os.path.join("anomaly/VALID_ANOMALY", file), 1)
   #data.append(df)


# In[139]:


for file in os.listdir("anomaly/TEST_ANOMALY"):
    df = process_file(os.path.join("anomaly/TEST_ANOMALY", file), 1)
    data.append(df)


# In[140]:


for file in os.listdir("normal/TEST_NORMAL"):
     df = process_file(os.path.join("normal/TEST_NORMAL", file), 0)
     data.append(df)


# In[141]:


#for file in os.listdir("normal/VALID_NORMAL"):
    #df = process_file(os.path.join("normal/VALID_NORMAL", file), 0)
    #data.append(df)


# In[142]:


#df = process_file("normal/training-data.csv", 0)
#data.append(df)


# In[143]:


df = pd.concat(data)


# In[144]:


df.groupby(["id", "anomaly"], as_index=False)["lq_d"].agg(["min", "max"]).groupby("anomaly").agg(["mean", "std"])


# In[145]:


plt.scatter(df["cmp_b_s"], df["cmp_a_s"], c=df["anomaly"].replace({1:"red", 0:"blue"}))


# In[146]:


plt.scatter(df["cmp_b_d"], df["cmp_a_d"], c=df["anomaly"].replace({1:"red", 0:"blue"}))


# In[147]:


#Task 2 (30 points): Using independent Gaussian analysis for anomaly detection. The threshold can be
#tuned using the validation dataset. Report the True Positive, False Positive, True Negative, and False
#Negative, Precision, Recall, and F1 score


# In[148]:


#pick features
train = pd.read_csv("train/training-data.csv")
features = ["f1_d", "f2_d", "pr_d", "lq_d", "cmp_a_d","cmp_b_d", "cmp_c_d"]


# In[149]:


#compute the mean and standard deviation for the features
model_data = train[features]
model = model_data.agg(["mean", "std"])
model


# In[150]:


def p(x, model):
    """
    x is an observation (of only the features we want)
    model is the mean and standard deviation of the features in the training examples
    """
    # print(x)
    prob = 1
    for feature in model.columns:
        mu, sigma = model.loc["mean", feature], np.sqrt(model.loc["std", feature])
        prob *= (1/(sigma*(2*np.pi)**0.5)) * np.exp(-(x[feature] - mu)**2/(2*sigma**2))
    return prob



# In[151]:


validation = pd.read_csv("valid/0.csv")
x = validation[features]
p(x.iloc[0], model)

p(x, model).hist(bins=40)


# In[152]:


results = p(df, model)


# In[153]:


anomaly_probability_threshold = 0.000034
df["anomaly_predicted"] = (results < anomaly_probability_threshold).astype(int)


# In[154]:


y = pd.read_csv("submission.txt", header=None, sep=" ")
#y.columns = ["id", "anomaly"]
# y = y.set_index("id")["anomaly"]
y = df.groupby("id")["anomaly"].max()


# In[155]:


(df.groupby("id")["anomaly_predicted"].mean()).hist(bins=20)


# In[156]:


(df.groupby("id")["anomaly_predicted"].mean())


# In[157]:


pct_process_threshold = 0.1232
pred = (df.groupby("id")["anomaly_predicted"].mean() > pct_process_threshold).astype(int)
(pred.sort_index() == y).mean()


# In[158]:


y.mean(), pred.mean()


# In[159]:


pd.concat([pred, y], axis=1)


# In[160]:


true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
for i in range(0, len(y)):
    if y.loc[i] == 1 and pred.loc[i] == 1:
        true_positives += 1
    elif y.loc[i] == 0 and pred.loc[i] == 0:
        true_negatives += 1
    elif y.loc[i] == 1 and pred.loc[i] == 0:
        false_negatives += 1
    elif y.loc[i] == 0 and pred.loc[i] == 1:
        false_positives += 1


precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f_1_score = (2 * (precision * recall)) / (precision + recall)


# In[161]:


print("true positives: ", true_positives)
print("false positives: ", false_positives)
print("false negatives: ", false_negatives)
print("true negatives: ", true_negatives)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f_1_score)


# In[162]:


#Task 3 (45 points): Using Multi-variate Gaussian analysis for anomaly detection. The threshold can be
#tuned using the validation dataset. Report the True Positive, False Positive, True Negative, and False
#Negative, Precision, Recall, and F1 score.


# In[163]:


def m_p(x, model):
    #x is the collection data being used to train 
    #number of features: d
    
    d = len(features)
    
    mu = np.zeros((d, 1))
    
    for i in range(0, len(x)):
        a = np.array(x.iloc[i])
        b = np.array([a]).T
        mu += b
    
    mu /= len(x)
    
    # find the covariance matrix
    
    #intialize the matrix to the number of features by number of features 
    sigma = np.zeros((d, d))
    
    for i in range(0, len(x)):
        a = np.array(x.iloc[i])
        sigma += (b - mu) @ (b - mu).T

    sigma /= len(x)
    
    p = (1 / (((2 *np.pi)**(d/2)) * (np.abs(np.linalg.det(sigma))**(0.5)))) * np.exp(-0.5 * ((b-mu).T @ np.linalg.inv(sigma) @  (b-mu)))
    
    return p[0][0]
    
        


# In[164]:


m_p(train[features], model)


# In[165]:


results = p(df, model)


# In[166]:


anomaly_probability_threshold = 0.000034
df["anomaly_predicted"] = (results < anomaly_probability_threshold).astype(int)


# In[167]:


y = pd.read_csv("submission.txt", header=None, sep=" ")
#y.columns = ["id", "anomaly"]
# y = y.set_index("id")["anomaly"]
y = df.groupby("id")["anomaly"].max()


# In[168]:


pct_process_threshold = 0.1232
pred = (df.groupby("id")["anomaly_predicted"].mean() > pct_process_threshold).astype(int)
(pred.sort_index() == y).mean()


# In[169]:


y.mean(), pred.mean()


# In[170]:


pd.concat([pred, y], axis=1)


# In[171]:


true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
for i in range(0, len(y)):
    if y.loc[i] == 1 and pred.loc[i] == 1:
        true_positives += 1
    elif y.loc[i] == 0 and pred.loc[i] == 0:
        true_negatives += 1
    elif y.loc[i] == 1 and pred.loc[i] == 0:
        false_negatives += 1
    elif y.loc[i] == 0 and pred.loc[i] == 1:
        false_positives += 1


precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f_1_score = (2 * (precision * recall)) / (precision + recall)


# In[173]:


print("true positives: ", true_positives)
print("false positives: ", false_positives)
print("false negatives: ", false_negatives)
print("true negatives: ", true_negatives)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f_1_score)



# In[ ]:




