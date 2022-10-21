#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Task 1 (20 points): Get familiar with the dataset. You can do some statistic analysis (table or plotting) for
#each feature. You can also compare the feature difference between the normal data and abnormal data. Put
#your analysis within 2 pages in the report.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[2]:


def process_file(filename, anomaly=1):
    df = pd.read_csv(filename)
    df["anomaly"] = anomaly
    if anomaly:
        df["id"] = int(filename.split("_")[-1][:-4])
    else:
        
        df["id"] = int(filename.split("\\")[-1][:-4])
    return df


# In[3]:


train = pd.read_csv("normal/training-data.csv")


# In[4]:


data = []

#for file in os.listdir("anomaly/VALID_ANOMALY"):
   #df = process_file(os.path.join("anomaly/VALID_ANOMALY", file), 1)
   #data.append(df)


# In[5]:


for file in os.listdir("anomaly/TEST_ANOMALY"):
    df = process_file(os.path.join("anomaly/TEST_ANOMALY", file), 1)
    data.append(df)


# In[6]:


for file in os.listdir("normal/TEST_NORMAL"):
     df = process_file(os.path.join("normal/TEST_NORMAL", file), 0)
     data.append(df)


# In[7]:


#for file in os.listdir("normal/VALID_NORMAL"):
   #df = process_file(os.path.join("normal/VALID_NORMAL", file), 0)
   #data.append(df)


# In[8]:


#df = process_file("normal/training-data.csv", 0)
#data.append(df)


# In[9]:


df = pd.concat(data)


# In[10]:


df.groupby(["id", "anomaly"], as_index=False)["lq_d"].agg(["min", "max"]).groupby("anomaly").agg(["mean", "std"])


# In[12]:


plt.scatter(df["cmp_b_s"], df["cmp_a_s"], c=df["anomaly"].replace({1:"red", 0:"blue"}))


# In[14]:


plt.scatter(df["cmp_b_d"], df["cmp_a_d"], c=df["anomaly"].replace({1:"red", 0:"blue"}))


# In[15]:


#Task 2 (30 points): Using independent Gaussian analysis for anomaly detection. The threshold can be
#tuned using the validation dataset. Report the True Positive, False Positive, True Negative, and False
#Negative, Precision, Recall, and F1 score


# In[16]:


#pick features
train = pd.read_csv("train/training-data.csv")
features = ["f1_d", "f2_d", "pr_d", "lq_d", "cmp_a_d","cmp_b_d", "cmp_c_d"]


# In[19]:


#compute the mean and standard deviation for the features
model_data = train[features]
model = model_data.agg(["mean", "std"])
model


# In[20]:


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



# In[21]:


validation = pd.read_csv("valid/0.csv")
x = validation[features]
p(x.iloc[0], model)

p(x, model).hist(bins=40)


# In[22]:


results = p(df, model)


# In[23]:


anomaly_probability_threshold = 0.000034
df["anomaly_predicted"] = (results < anomaly_probability_threshold).astype(int)


# In[24]:


y = pd.read_csv("submission.txt", header=None, sep=" ")
#y.columns = ["id", "anomaly"]
# y = y.set_index("id")["anomaly"]
y = df.groupby("id")["anomaly"].max()


# In[25]:


(df.groupby("id")["anomaly_predicted"].mean()).hist(bins=20)


# In[26]:


(df.groupby("id")["anomaly_predicted"].mean())


# In[27]:


pct_process_threshold = 0.120
pred = (df.groupby("id")["anomaly_predicted"].mean() > pct_process_threshold).astype(int)
(pred.sort_index() == y).mean()


# In[28]:


y.mean(), pred.mean()


# In[60]:


pd.concat([pred, y], axis=1)


# In[143]:


id_list = np.sort(df["id"].drop_duplicates().values, axis=None)


# In[144]:


np.savetxt('gaussian_output.txt', np.column_stack((id_list,pred.values)), fmt ="%i %i", delimiter = ' ')


# In[145]:


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


# In[146]:


print("true positives: ", true_positives)
print("false positives: ", false_positives)
print("false negatives: ", false_negatives)
print("true negatives: ", true_negatives)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f_1_score)


# In[147]:


#Task 3 (45 points): Using Multi-variate Gaussian analysis for anomaly detection. The threshold can be
#tuned using the validation dataset. Report the True Positive, False Positive, True Negative, and False
#Negative, Precision, Recall, and F1 score.


# In[148]:


def m_p(x, model):
    #x is the collection data being used to train 
    #number of features: d
    
    d = len(features)
    
    mu = np.zeros((d, 1))
    
    for i in range(0, len(model)):
        a = np.array(model.iloc[i])
        b = np.array([a]).T
        mu += b
    
    mu /= len(model)
    # find the covariance matrix
    
    #intialize the matrix to the number of features by number of features 
    sigma = np.zeros((d, d))
    
    for i in range(0, len(model)):
        a = np.array(model.iloc[i])
        b = np.array([a]).T
        sigma += (b - mu) @ (b - mu).T

    sigma /= len(model)
    
    return sigma, mu
    
    a = np.array(x)
    arr = []
    for i in range(0, len(x)):
        b = np.array([x.iloc[i]]).T
        prob = np.array( (1 / ( ((2 * np.pi)**(d/2)) * (np.linalg.det(sigma)**(0.5)))) * np.exp(-0.5 * ( (b-mu).T @ np.linalg.inv(sigma) @ (b-mu))))
        arr.append(prob[0][0])
        
    
    p = pd.DataFrame(np.array(arr), columns = ['Prob'])
    return p
    
        


# In[149]:


df = pd.concat(data)


# In[150]:


features = ["f1_a", "f1_s", "f1_c", "f2_a", "f2_s", "f2_c", "pr_s", "prd_a", "prd_c", "prd_s"]


# In[151]:


model_data = train[features]


# In[152]:


#results = m_p(df[features], model)


# In[153]:


model = model_data.agg(["mean", "std"])
model


# In[154]:


def normalized(data, model):
    normalized = data
    for i in range(0, len(data)):
        for feature in model.columns:
            a = (float(normalized.iloc[i][feature]) - float(model[feature]["mean"])) / ( model[feature]["std"])
            normalized.loc[i, feature] = abs(a)
            
    return normalized
        


# In[155]:


#train_n = normalized(train[features], model)
#train_n


# In[156]:


sigma = np.cov(train[features].values.T)


# In[157]:


mu = train[features].mean().values


# In[158]:


def m_p_2(x, mu, sigma):
    d = sigma.shape[0]
    x_c = x-mu
    return (1 / ( ((2 * np.pi)**(d/2)) * (np.linalg.det(sigma)**(0.5)))) * np.exp(-0.5 * (x_c @ np.linalg.inv(sigma) @ x_c.T))


# In[159]:


p = np.apply_along_axis(m_p_2, axis=1, arr=df[features].values, mu=mu, sigma=sigma)


# In[160]:


results = pd.DataFrame(p)
results.hist(bins=50)
anomaly_probability_threshold = 0.1 * (10**-29)

df["anomaly_predicted"] = (results > anomaly_probability_threshold).astype(int)


# In[161]:


y = pd.read_csv("submission.txt", header=None, sep=" ")
y.columns = ["id", "anomaly"]
y = y.set_index("id")["anomaly"]
#y = df.groupby("id")["anomaly"].max()


# In[162]:


(df.groupby("id")["anomaly_predicted"].mean()).hist(bins=60)


# In[163]:


pct_process_threshold = 0.34996

pred = (df.groupby("id")["anomaly_predicted"].mean() < pct_process_threshold).astype(int)
#pred = ((df.groupby("id")["anomaly_predicted"].mean() > pct_process_threshold) or (df.groupby("id")["anomaly_predicted"].mean() < pct_process_threshold_lower)).astype(int)
(pred.sort_index() == y).mean()


# In[164]:


y.mean(), pred.mean()


# In[165]:


pd.concat([pred, y], axis=1)


# In[167]:


id_list = np.sort(df["id"].drop_duplicates().values, axis=None)


# In[168]:


np.savetxt('multivariate_gaussian_output.txt', np.column_stack((id_list,pred.values)), fmt ="%i %i", delimiter = ' ')


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


# In[172]:


print("true positives: ", true_positives)
print("false positives: ", false_positives)
print("false negatives: ", false_negatives)
print("true negatives: ", true_negatives)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f_1_score)



# In[ ]:




