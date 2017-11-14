
# coding: utf-8

# In[1]:


import pcm
import read_dataset


# In[2]:


print("_________ Bengio Team Projet: Machine Learning Data Mining ___________")


# In[4]:


X,T = read_dataset.read_letter_recognition_dataset()
print (X)
print (T)


# In[3]:


pcm.pcm()


# In[17]:


get_ipython().magic('run main.py -d')

