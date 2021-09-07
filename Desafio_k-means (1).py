#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


df = pd.read_csv(r"C:\Users\guilh\OneDrive\Desktop\projetos\dados_desafio_k-means\29013.csv")


# In[41]:


df


# In[42]:


df = df.rename(columns={'0': 'errado','1.4326407534765973': 'A', '7.8207214321166525' :  'B', '0.1' : 'C'})


# In[43]:


df


# In[44]:


df = df.drop(columns=['errado'])


# In[45]:


df


# In[46]:


sb.pairplot(df,hue='C')


# In[47]:


pred = df.drop(columns=['A', 'B'])


# In[48]:


pred


# In[49]:


Xteste = df.iloc[:, 0:2].values


# In[50]:


Xteste


# In[51]:


from sklearn.cluster import KMeans


# In[52]:


kmeans = KMeans(n_clusters=4, random_state=0)


# In[53]:


kmeans.fit(Xteste)


# In[54]:


kmeans.labels_


# In[63]:


Yteste = kmeans.labels_


# In[64]:


Yteste


# In[55]:


df['Kclasses'] = kmeans.labels_


# In[56]:


sb.pairplot(df,hue='Kclasses')


# In[65]:


from sklearn.metrics import confusion_matrix
import pylab as pl

cm = confusion_matrix(Yteste, pred)
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()


# In[ ]:





# In[ ]:





# In[ ]:




