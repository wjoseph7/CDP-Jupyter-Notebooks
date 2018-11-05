import random as rand
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#rand.seed(0)

wheel = ("0-28-9-26-30-11-7-20-32-17-5-22-34-15-3-24-"
         "36-13-1-00-27-10-25-29-12-8-19-31-18-6-21-"
         "33-16-4-23-35-14-2")
wheel = wheel.split("-")

def Trial():
    return wheel[rand.randint(0,len(wheel)-1)]

trials = []

for i in range(1000000):
    trials.append(Trial())

plt.hist(trials,bins=len(wheel))
plt.title("1 Million Roulette Trials",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.xlabel("Slot Number",fontsize=20)
plt.show()
exit()
def oddTrial():
    trial = int(Trial())
    if trial&1:
        return str(trial)
    else:
        return oddTrial()

oddTrials = []

for i in range(10):
    oddTrials.append(oddTrial())

def MontyHall(verbose, n, switch=False):

    if verbose:
        for i in range(n):
            print("x",end="")
        print("")
        print("^")
            
    doors = []
    for i in range(n):
        doors.append(0)
    randIndex = npr.randint(low=0, high=n, size=1)[0]
    doors[randIndex]=1
    
    randIndex2 = -1

    while randIndex2 == -1 or randIndex2 == randIndex or randIndex2 == 0:
        randIndex2 = npr.randint(low=0, high=n, size=1)[0]

    if verbose:
        print("")
        print("")
    
        for i in range(n):
            if i!=randIndex2:
                print("x",end="")
            else:
                print(doors[randIndex2],end="")

        print("")
        print("^")

        print("")
        print("")

    if verbose:
        doorNumber = input("Which door would you like to open? ")
        doorNumber = int(doorNumber)
    elif switch:
        doorNumber = switch if randIndex2 != 1 else n-1
    else:
        doorNumber = 0

    if verbose:
        for i in range(n):
            if i!=randIndex2 and i!=doorNumber:
                print("x",end="")
            elif i==randIndex2:
                print(doors[randIndex2],end="")
            else:
                print(doors[doorNumber],end="")

        print("")

        for i in range(n):
            if i!=doorNumber:
                print(" ", end="")
            else:
                print("^")
        print("")

    if doors[doorNumber]:
        if verbose: print("You got it!")
        return 1
    else:
        if verbose: print("Better luck next time!")
        return 0
    
MontyHall(True, 3)

winStay = []
winSwitch = []

for i in range(100000):
    winStay.append(MontyHall(False,3))
    winSwitch.append(MontyHall(False,3,True))

pStay = np.mean(winStay)
pSwitch = np.mean(winSwitch)

print("")
print("")

print("Probability of Winning if you stay = ", pStay)
print("Probability of Winning if you switch = ", pSwitch)
