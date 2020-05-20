from math import sqrt
import numpy as np
from collections import Counter
import warnings 
import pandas as pd
import random
#from matplotlib import style
#style.use('fivethirtyeight')
#dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_features=[5,7]
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.show()
def k_nearest_neighbors(data, predict, k=3) :
    if len(data)>=k:
        warnings.warn("k is bigger than the classes!")
    distances=[]
    for group in data:
        for features in data[group]:
            Udistances=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([Udistances,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    votes_result=Counter(votes).most_common(1)[0][0]
    confidence=Counter(votes).most_common(1)[0][1]/k
    return votes_result,confidence
data=pd.read_csv('BreastCancer.csv')
data.replace('?',-99999,inplace=True)
data.drop(['id'],1,inplace=True)
full_data=data.astype(float).values.tolist()
random.shuffle(full_data) #must be converted to list first


#slicing
test_size=.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]


#dictionary to the algrothim
for i in train_data:
     train_set[i[-1]].append(i[:-1]) 
for i in test_data:
    test_set[i[-1]].append(i[:-1]) 

#accuracy
correct=0
total=0
for group in test_set:
    for lists in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set, lists, k=5)
        if vote == group:
            correct+=1
        total+=1
print('accuray:', correct/total)
    