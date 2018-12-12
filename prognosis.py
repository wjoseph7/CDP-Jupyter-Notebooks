import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import scipy.stats as stats


#Create column names
cols = ["id","outcome","time"]
for i in range(32):
    cols.append(str(i))

#Read data into dataframe            
df = pd.read_csv("wpbc.data",header=None,names=cols)

#We only want to examine the data where the cancer recurred
df=df[df["outcome"]=="R"]
#Remove 4 rows with missing values
df=df[df["31"]!="?"]
#Remove unnecessary outcome and id columns
del df["outcome"]
del df["id"]

#Convert dtype of 31st column from object to float
df["31"] = df["31"].astype(float)

#Compute correlation matrix variable with max correlation
cors = np.corrcoef(df,rowvar=False)[0][1:]
corsAbs = abs(cors)
print(cors)
print(np.argmax(corsAbs))

#make scatter plot
plt.scatter(df['0'],df['time'],color="red")
plt.xlabel("Radius")
plt.ylabel("Recurrence Time")

#Make linear model
X = df.drop("time", axis=1)
y = df.drop(list(df)[1:],axis=1)
lm = sk.LinearRegression()
fit=lm.fit(X[["0"]],y)

#Diagnostics
residuals = y-lm.predict(X[['0']])
residuals = residuals['time']
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].scatter(X[['0']],y, color='red')
ax[0][0].plot(X[['0']],lm.predict(X[['0']]))
ax[0][1].scatter(residuals,lm.predict(X[['0']]))
ax[1][0].hist(residuals)
stats.probplot(residuals, dist="norm", plot=ax[1][1])

#Polynomial basis functions
def plotPoly(n):
    plt.scatter(X[['0']],y,color='red')
    polyX = X[['0']].values[:,0]
    fit = np.polyfit(polyX,y,n)[:,0]
    fitted = 0
    print(fit)
    for i in range(n+1):
        fitted += fit[i]*np.linspace(12,30)**(n-i)
    print(fitted)
    plt.plot(np.linspace(12,30),fitted)
plotPoly(5)
plt.show()
#Jacknife MSE optimization
x = X[['0']].values[:,0]
y = y.values[:,0]
def poly(n, xp, yp, xpred, yreal):
    fit = np.polyfit(xp,yp,n)
    fitted = 0
    for i in range(n+1):
        fitted += fit[i]*xpred**(n-i)
    residual = (yreal-fitted)**2
    return residual
MSE = [] 
for p in range(1,6):
    SEn = []
    for i in range(len(X)):
        xJack = np.delete(x,[i])
        yJack = np.delete(y,[i])
        SEn.append(poly(p,xJack,yJack,x[i],y[i]))
    MSE.append(SEn)
plt.boxplot(MSE)
plt.xlabel("Poly. degree")
plt.ylabel("MSE")

#UQ with Jackknife
pointSet = []
exes = np.linspace(12,30)
for i in range(len(X)):
    xJack = np.delete(x,[i])
    yJack = np.delete(y,[i])
    fit = np.polyfit(xJack,yJack,2)
    fitted = 0
    for j in range(2+1):
        fitted += fit[j]*exes**(2-j)
    pointSet.append(fitted)
mean = []
sd = []
for i in range(len(exes)):
    mu = 0
    s = []
    for j in range(len(pointSet)):
        mu += pointSet[j][i]
        s.append(pointSet[j][i])
    mu /= len(pointSet)
    mean.append(mu)
    sd.append(np.std(s))
snorm = [s/max(sd) for s in sd]
errorPlus = [x+2*s*np.std(y) for x,s in zip(mean,snorm)]
errorMinus = [x-2*s*np.std(y) for x,s in zip(mean,snorm)]
plt.scatter(x,y,color="green")
plt.plot(exes,mean,color="blue")
plt.plot(exes,errorPlus,color="red")
plt.plot(exes,errorMinus,color="red")
plt.xlabel("Radius")
plt.ylabel("Recurrence Time")











plt.show()



"""
#Diagnostics
residuals = y-lm.predict(X[['0']])
residuals = residuals['time']
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].scatter(X[['0']],y, color='red')
ax[0][0].plot(X[['0']],lm.predict(X[['0']]))
ax[0][0].set_xlabel("Radius")
ax[0][0].set_ylabel("Recurrence Time")
ax[0][1].scatter(residuals,lm.predict(X[['0']]))
ax[0][1].set_xlabel("Residuals")
ax[0][1].set_ylabel("Fitted")
ax[1][0].hist(residuals)
ax[1][0].set_xlabel("Residuals Histogram")
stats.probplot(residuals, dist="norm", plot=ax[1][1])
ax[1][1].set_title("")
"""

