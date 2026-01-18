#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import arch
print(arch.__version__)



# In[5]:


pip install arch --upgrade


# In[6]:


from arch.univariate import GARCH, EGARCH, TGARCH, Normal, Studentst


# In[7]:


from arch import arch_model
from arch.univariate import GARCH, EGARCH, TARCH, Normal, StudentsT


# In[8]:


import pymc3 as pm
import theano.tensor as tt


# In[ ]:




