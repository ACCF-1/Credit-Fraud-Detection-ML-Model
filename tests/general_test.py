# In[0]

import pandas as pd
import numpy as np
import ydata_profiling as yp


df = pd.DataFrame(np.random.rand(100,5), columns=['a','b','c','d','e'])

profile = yp.ProfileReport(df, title='test', html={'style':{'full_width':True}})
# %%
