import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use("fivethirtyeight")
#xs=np.array([1,2,3,4,5,6],dtype=np.float64) #datatype is in numpy 
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val+ random.randrange(-variance,variance) #needs an intial value
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step 
    xs = [i for i in range(hm)]
    return np.array(xs , dtype=np.float64),np.array(ys,dtype=np.float64) #order makes different


def best_fit_slope_and_intercept(xs,ys): 
    m=(((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)**2)-mean(xs**2)))
    b= mean(ys)-m*mean(xs)
    return m , b


def squared_error(ys_orig,ys_reg):
    return sum((ys_reg-ys_orig)**2)

    
def coeffecient_of_determination(ys_orig,ys_reg):
    mean_line=[mean(ys_orig) for y in ys]
    mean_error=squared_error(ys_orig,mean_line)
    reg_error=squared_error(ys_orig,ys_reg)
    return 1 - (reg_error/mean_error)

xs,ys=create_dataset(50,10,2,correlation='pos') #WHERE YOU PUT IT MAKES A DIFFERENCCCEEEEEE!!!!
m,b= best_fit_slope_and_intercept(xs,ys) #we want the variables to be global
Regression_line=[(m*x)+b for x in xs] #the[] because it will create a list
r2=coeffecient_of_determination(ys,Regression_line)
plt.scatter(xs,ys)
plt.plot(xs,Regression_line)
plt.show()
