import json
import pandas as pd
#import ijson

import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import numpy as np
import random
import pandas as pd
import ast

d = pd.read_csv('filtered.csv')
# dw = d.loc[438]['attributes']
# di = ast.literal_eval(dw)
# for y in range(len(d)):
#     dw = d.loc[y]['attributes']
#     di = ast.literal_eval(dw)
#     for x in di:
#         #print(x)

print(d.shape)
print(list(list(np.where(d['attributes'].isna()))[0]))
d = d.drop(list(list(np.where(d['attributes'].isna()))[0]), axis=0)
print(d.shape)
d = d.reset_index(drop=True)
#for x in d['attributes']:
lens = [len(ast.literal_eval(x)) for x in d['attributes']]
res_list = [i for i in range(len(lens)) if lens[i] < 3]
print(res_list)
d = d.drop(res_list, axis=0)
#d = d[len(ast.literal_eval(d['attributes'])) > 3]
#d.drop(d.loc[len(ast.literal_eval(['attributes'])) < 3],axis = 0)
print(d.shape)
d.to_csv('filtered_dropped.csv')
