#!/usr/bin/env python
# coding: utf-8

# In[3]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('C:/Users/PonfaJr/Desktop/mlModel/Untitled Folder 1/autoM.csv')


# In[5]:


data.columns


# In[6]:


df=data.drop(columns=['symboling','normalized-losses'])
df


# In[7]:


df.info()


# In[8]:


df["price"].describe()


# In[9]:


df[df.price=="?"]


# In[10]:


df["price"][129]=28400
df["price"][45]=11000
df["price"][44]=16000
df["price"][9]=37400


# In[11]:


df.iloc[9]


# In[12]:


df.describe()


# In[13]:


df.head()


# In[14]:


df["Price"]=df["price"].astype(int)
df.head(5)


# In[15]:


df["Price"].dtypes


# In[16]:


df.drop(columns=["price"],inplace=True)


# In[17]:


df.horsepower[df.horsepower=="?"]=88


# In[18]:


df["Horsepower"]=df.horsepower.astype(int)


# In[19]:


df.drop(columns=["horsepower"],inplace=True)


# In[20]:


df.info()


# In[21]:


df[(df["Price"]==df.Price.max())|(df["Horsepower"]==df.Horsepower.max())]


# In[22]:


df["body-style"].unique()


# In[23]:


df.groupby("body-style").nunique()["engine-size"]


# In[24]:


df["Price"].mean()


# In[25]:


df["make"].unique()


# In[26]:


df["Price"].unique()


# In[27]:


df[df['make']=='mercedes-benz'].sort_values(by="Price")


# In[28]:


df["Price"].plot()


# In[29]:


df[df=="?"].count()


# In[30]:


df[df["peak-rpm"]=="?"] = 1
df["peak-rpm"]=df["peak-rpm"].astype(int)
df["peak-rpm"].max()


# In[31]:


df.head(1)


# In[32]:


df5=np.arange(205)
df["new"]=pd.DataFrame(df5)
df


# In[33]:


df["make"].value_counts()


# In[34]:


toy=df[df.make=="toyota"]
nis=df[df.make=="nissan"]
maz=df[df.make=="mazda"]
hon=df[df.make=="honda"]
vis=pd.concat([toy,nis,maz,hon])


# In[35]:


vis[vis.Price==vis["Price"].max()]


# In[36]:


plt.plot(df.new[:32],toy.Price,marker="o",label="toyota",linestyle="--")
plt.plot(df.new[:18],nis.Price,marker="o",label="nissan",linestyle="--")
plt.plot(df.new[:17],maz.Price,marker="o",label="mazda",linestyle="--")
plt.plot(df.new[:13],hon.Price,marker="o",label="honda",linestyle="--")
plt.title("Price prediction")
plt.xlabel("Car makers")
plt.ylabel("Price")
plt.legend()
plt.legend()
plt.show()


# In[37]:


df.head()


# In[38]:


price=pd.cut(df.Price,np.arange(0,45000,1000))
df.pivot_table("Horsepower",index=price,aggfunc=np.mean)
#18000 to 19000 price is enough for making 180 horsepower cars


# In[39]:


df[df["Horsepower"]>=190]["Price"].min()


# In[40]:


df[df["Price"]==19699]


# In[41]:


df50=df.head(21)
df50


# In[42]:


df50.make


# In[43]:


g= sns.FacetGrid(df50,col="make")
g.map(plt.hist,"Price",bins=20)


# In[44]:


make_g=df.groupby(by="make")
make_g.count()


# In[45]:


h=sns.FacetGrid(df,"make",size=5.5,aspect=1.6)
h.map(plt.hist,"Price",bins=20)
#in this case i find that the BMW car has the various price range cars


# In[48]:


h=sns.FacetGrid(df,"make",size=5.5,aspect=1.6)
h.map(sns.pointplot,"Price","Horsepower",bins=20)


# In[49]:


df50






