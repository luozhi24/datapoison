import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import attack
from scipy import optimize
import time
import os

i = int(sys.argv[1])

rho = 1/2
nu = 1 #math.sqrt(2)
deltal = 0.05
c = 2*math.sqrt(1/(1-rho))
n = 10000000
random.seed()
###revise
K = i/10

path = "./" + sys.argv[1]
if os.path.isdir(path):
    pass
else:
    os.makedirs(path)

att = attack.attack(sys.argv[1])

hmax = 1
comhmax = 1
ranhmax = 1
hcthmax = 1

class node:
    def __init__(self, h, i):
        self.h = h
        self.i = i
        self.U = float('inf')
        self.B = float('inf')
        self.sumreward = 0
        self.T = 0
        #self.tau = 0
        self.start = 0
        self.end = 0
        self.represent = -1
        self.leaf = 1


def f(x):
    return 0.5*(math.sin(13*x)*math.sin(27*x)+1)

def oppof(x):
    return -1*0.5*(math.sin(13*x)*math.sin(27*x)+1)

def tand(t):
    #return 2**(math.floor(math.log(t))+1)
    return 2**(math.floor(math.log(t))+1)

def delta(t):
    return min((rho/(3*nu))**(1/8)*deltal/t, 1)

def iternode(h, i):
    return str(h) + "," + str(i)

def caltau(t, h):
    return c**2*math.log(1/delta(tand(t)))*rho**(-2*h)/nu**2

'''
opti = optimize.minimize(oppof, 24, method='SLSQP')
maxv = -1 * opti.fun
'''
opti = optimize.minimize(oppof, 0.85, method='SLSQP')
maxv = -1 * opti.fun
worti = optimize.minimize(f, 0.65, method='SLSQP')
minv = worti.fun

print(opti.x[0])
print(worti.x[0])
print(maxv)
print(minv)
time.sleep(5)

maintree = dict()
comtree = dict()
rantree = dict()
hcttree = dict()

root = node(0, 1)
root.T = 1
root.leaf = 0
root.start = 0
root.end = 1
#root.tau = 1
#print(iternode(0, 1))

maintree.update({iternode(0, 1): root})
maintree.update({iternode(1, 1): node(1, 1)})
maintree.update({iternode(1, 2): node(1, 2)})
maintree[iternode(1, 1)].start = 0
maintree[iternode(1, 1)].end = 0.5
maintree[iternode(1, 2)].start = 0.5
maintree[iternode(1, 2)].end = 1

comtree.update({iternode(0, 1): root})
comtree.update({iternode(1, 1): node(1, 1)})
comtree.update({iternode(1, 2): node(1, 2)})
comtree[iternode(1, 1)].start = 0
comtree[iternode(1, 1)].end = 0.5
comtree[iternode(1, 2)].start = 0.5
comtree[iternode(1, 2)].end = 1

rantree.update({iternode(0, 1): root})
rantree.update({iternode(1, 1): node(1, 1)})
rantree.update({iternode(1, 2): node(1, 2)})
rantree[iternode(1, 1)].start = 0
rantree[iternode(1, 1)].end = 0.5
rantree[iternode(1, 2)].start = 0.5
rantree[iternode(1, 2)].end = 1

hcttree.update({iternode(0, 1): root})
hcttree.update({iternode(1, 1): node(1, 1)})
hcttree.update({iternode(1, 2): node(1, 2)})
hcttree[iternode(1, 1)].start = 0
hcttree[iternode(1, 1)].end = 0.5
hcttree[iternode(1, 2)].start = 0.5
hcttree[iternode(1, 2)].end = 1

#print(maintree[iternode(1, 2)].i)

def opttraverse(t):
    tau = 1
    ht = 0
    it = 1
    nodeiter = iternode(ht, it)
    while maintree[nodeiter].leaf == 0 and maintree[nodeiter].T >= tau:
        if maintree[iternode(ht+1, 2*it-1)].B >= maintree[iternode(ht+1, 2*it)].B:
            ht = ht+1
            it = 2*it-1
        else:
            ht = ht+1
            it = 2*it
        tau = caltau(t, ht)
        #print(maintree[nodeiter].T, tau)
        nodeiter = iternode(ht, it)
    return ht, it

def updateB(ht, it):
    nodeiter = iternode(ht, it)
    if maintree[nodeiter].leaf == 1:
        maintree[nodeiter].B = maintree[nodeiter].U
    else:
        maintree[nodeiter].B = min(maintree[nodeiter].U, max(maintree[iternode(ht+1, 2*it-1)].B, maintree[iternode(ht+1, 2*it)].B))

    h = ht - 1
    if it % 2 == 0:
        i = int(it/2)
    else:
        i = int((it+1)/2)

    while h != 0:
        nodeiter = iternode(h, i)
        maintree[nodeiter].B = min(maintree[nodeiter].U, max(maintree[iternode(h+1, 2*i-1)].B, maintree[iternode(h+1, 2*i)].B))
        h = h-1
        if i % 2 == 0:
            i = int(i/2)
        else:
            i = int((i+1)/2)
    return

def calU(treekey, time):
    return maintree[treekey].sumreward/maintree[treekey].T + nu * rho**(maintree[treekey].h) + math.sqrt(c**2*math.log(1/delta(tand(time)))/maintree[treekey].T)

def comopttraverse(t):
    tau = 1
    ht = 0
    it = 1
    nodeiter = iternode(ht, it)
    while comtree[nodeiter].leaf == 0 and comtree[nodeiter].T >= tau:
        if comtree[iternode(ht+1, 2*it-1)].B >= comtree[iternode(ht+1, 2*it)].B:
            ht = ht+1
            it = 2*it-1
        else:
            ht = ht+1
            it = 2*it
        tau = caltau(t, ht)
        #print(maintree[nodeiter].T, tau)
        nodeiter = iternode(ht, it)
    return ht, it

def comupdateB(ht, it):
    nodeiter = iternode(ht, it)
    if comtree[nodeiter].leaf == 1:
        comtree[nodeiter].B = comtree[nodeiter].U
    else:
        comtree[nodeiter].B = min(comtree[nodeiter].U, max(comtree[iternode(ht+1, 2*it-1)].B, comtree[iternode(ht+1, 2*it)].B))

    h = ht - 1
    if it % 2 == 0:
        i = int(it/2)
    else:
        i = int((it+1)/2)

    while h != 0:
        nodeiter = iternode(h, i)
        comtree[nodeiter].B = min(comtree[nodeiter].U, max(comtree[iternode(h+1, 2*i-1)].B, comtree[iternode(h+1, 2*i)].B))
        h = h-1
        if i % 2 == 0:
            i = int(i/2)
        else:
            i = int((i+1)/2)
    return

def comcalU(treekey, time):
    return comtree[treekey].sumreward/comtree[treekey].T + nu * rho**(comtree[treekey].h) + math.sqrt(c**2*math.log(1/delta(tand(time)))/comtree[treekey].T)

def ranopttraverse(t):
    tau = 1
    ht = 0
    it = 1
    nodeiter = iternode(ht, it)
    while rantree[nodeiter].leaf == 0 and rantree[nodeiter].T >= tau:
        if rantree[iternode(ht+1, 2*it-1)].B >= rantree[iternode(ht+1, 2*it)].B:
            ht = ht+1
            it = 2*it-1
        else:
            ht = ht+1
            it = 2*it
        tau = caltau(t, ht)
        #print(maintree[nodeiter].T, tau)
        nodeiter = iternode(ht, it)
    return ht, it

def ranupdateB(ht, it):
    nodeiter = iternode(ht, it)
    if rantree[nodeiter].leaf == 1:
        rantree[nodeiter].B = rantree[nodeiter].U
    else:
        rantree[nodeiter].B = min(rantree[nodeiter].U, max(rantree[iternode(ht+1, 2*it-1)].B, rantree[iternode(ht+1, 2*it)].B))

    h = ht - 1
    if it % 2 == 0:
        i = int(it/2)
    else:
        i = int((it+1)/2)

    while h != 0:
        nodeiter = iternode(h, i)
        rantree[nodeiter].B = min(rantree[nodeiter].U, max(rantree[iternode(h+1, 2*i-1)].B, rantree[iternode(h+1, 2*i)].B))
        h = h-1
        if i % 2 == 0:
            i = int(i/2)
        else:
            i = int((i+1)/2)
    return

def rancalU(treekey, time):
    return rantree[treekey].sumreward/rantree[treekey].T + nu * rho**(rantree[treekey].h) + math.sqrt(c**2*math.log(1/delta(tand(time)))/rantree[treekey].T)

def hctopttraverse(t):
    tau = 1
    ht = 0
    it = 1
    nodeiter = iternode(ht, it)
    while hcttree[nodeiter].leaf == 0 and hcttree[nodeiter].T >= tau:
        if hcttree[iternode(ht+1, 2*it-1)].B >= hcttree[iternode(ht+1, 2*it)].B:
            ht = ht+1
            it = 2*it-1
        else:
            ht = ht+1
            it = 2*it
        tau = caltau(t, ht)
        #print(maintree[nodeiter].T, tau)
        nodeiter = iternode(ht, it)
    return ht, it

def hctupdateB(ht, it):
    nodeiter = iternode(ht, it)
    if hcttree[nodeiter].leaf == 1:
        hcttree[nodeiter].B = hcttree[nodeiter].U
    else:
        hcttree[nodeiter].B = min(hcttree[nodeiter].U, max(hcttree[iternode(ht+1, 2*it-1)].B, hcttree[iternode(ht+1, 2*it)].B))

    h = ht - 1
    if it % 2 == 0:
        i = int(it/2)
    else:
        i = int((it+1)/2)

    while h != 0:
        nodeiter = iternode(h, i)
        hcttree[nodeiter].B = min(hcttree[nodeiter].U, max(hcttree[iternode(h+1, 2*i-1)].B, hcttree[iternode(h+1, 2*i)].B))
        h = h-1
        if i % 2 == 0:
            i = int(i/2)
        else:
            i = int((i+1)/2)
    return

def hctcalU(treekey, time):
    return hcttree[treekey].sumreward/hcttree[treekey].T + nu * rho**(hcttree[treekey].h) + math.sqrt(c**2*math.log(1/delta(tand(time)))/hcttree[treekey].T)

t = 1
reward = np.zeros(n+1)
sumreward = np.zeros(n+1)
regret = np.zeros(n+1)
attsum = np.zeros(n+1)
target = np.zeros(n+1)

comreward = np.zeros(n+1)
comsumreward = np.zeros(n+1)
comregret = np.zeros(n+1)
comattsum = np.zeros(n+1)
comtarget = np.zeros(n+1)

ranreward = np.zeros(n+1)
ransumreward = np.zeros(n+1)
ranregret = np.zeros(n+1)
ranattsum = np.zeros(n+1)
rantarget = np.zeros(n+1)

hctreward = np.zeros(n+1)
hctsumreward = np.zeros(n+1)
hctregret = np.zeros(n+1)
hctattsum = np.zeros(n+1)
hcttarget = np.zeros(n+1)

timelist = np.zeros(n+1)

while t <= n:
    if t == tand(t):
        for key in maintree.keys():
            if maintree[key].h == 0:
                continue
            if maintree[key].T != 0:
                maintree[key].U = calU(key, t)

        for h in range(0, hmax):
            for key in maintree.keys():
                if maintree[key].h == hmax - h:
                    if maintree[key].leaf == 1:
                        maintree[key].B = maintree[key].U
                    else:
                        hnow = maintree[key].h
                        inow = maintree[key].i
                        maintree[key].B = min(maintree[key].U, max(maintree[iternode(hnow+1, 2*inow-1)].B, maintree[iternode(hnow+1, 2*inow)].B))

    if t == tand(t):
        for key in comtree.keys():
            if comtree[key].h == 0:
                continue
            if comtree[key].T != 0:
                comtree[key].U = comcalU(key, t)

        for h in range(0, comhmax):
            for key in comtree.keys():
                if comtree[key].h == comhmax - h:
                    if comtree[key].leaf == 1:
                        comtree[key].B = comtree[key].U
                    else:
                        hnow = comtree[key].h
                        inow = comtree[key].i
                        comtree[key].B = min(comtree[key].U, max(comtree[iternode(hnow+1, 2*inow-1)].B, comtree[iternode(hnow+1, 2*inow)].B))

    if t == tand(t):
        for key in rantree.keys():
            if rantree[key].h == 0:
                continue
            if rantree[key].T != 0:
                rantree[key].U = rancalU(key, t)

        for h in range(0, ranhmax):
            for key in rantree.keys():
                if rantree[key].h == ranhmax - h:
                    if rantree[key].leaf == 1:
                        rantree[key].B = rantree[key].U
                    else:
                        hnow = rantree[key].h
                        inow = rantree[key].i
                        rantree[key].B = min(rantree[key].U, max(rantree[iternode(hnow+1, 2*inow-1)].B, rantree[iternode(hnow+1, 2*inow)].B))

    if t == tand(t):
        for key in hcttree.keys():
            if hcttree[key].h == 0:
                continue
            if hcttree[key].T != 0:
                hcttree[key].U = hctcalU(key, t)

        for h in range(0, hcthmax):
            for key in hcttree.keys():
                if hcttree[key].h == hcthmax - h:
                    if hcttree[key].leaf == 1:
                        hcttree[key].B = hcttree[key].U
                    else:
                        hnow = hcttree[key].h
                        inow = hcttree[key].i
                        hcttree[key].B = min(hcttree[key].U, max(hcttree[iternode(hnow+1, 2*inow-1)].B, hcttree[iternode(hnow+1, 2*inow)].B))

    ht, it = opttraverse(t)
    print(ht, it)
    nodeiter = iternode(ht, it)
    if maintree[nodeiter].represent == -1:
        maintree[nodeiter].represent = maintree[nodeiter].start + (maintree[nodeiter].end - maintree[nodeiter].start)*random.random()

    comht, comit = comopttraverse(t)
    print("com:",comht, comit)
    comnodeiter = iternode(comht, comit)
    if comtree[comnodeiter].represent == -1:
        comtree[comnodeiter].represent = comtree[comnodeiter].start + (comtree[comnodeiter].end - comtree[comnodeiter].start)*random.random()

    ranht, ranit = ranopttraverse(t)
    print("ran:",ranht, ranit)
    rannodeiter = iternode(ranht, ranit)
    if rantree[rannodeiter].represent == -1:
        rantree[rannodeiter].represent = rantree[rannodeiter].start + (rantree[rannodeiter].end - rantree[rannodeiter].start)*random.random()

    hctht, hctit = hctopttraverse(t)
    print("hct:",hctht, hctit)
    hctnodeiter = iternode(hctht, hctit)
    if hcttree[hctnodeiter].represent == -1:
        hcttree[hctnodeiter].represent = hcttree[hctnodeiter].start + (hcttree[hctnodeiter].end - hcttree[hctnodeiter].start)*random.random()
    
    """
    if K >= maintree[nodeiter].start and K < maintree[nodeiter].end:
        meanreward = f(maintree[nodeiter].represent)
        reward[t] = random.gauss(meanreward, 1)
        attsum[t] = attsum[t-1]
    else:
    """
    reward[t], attackornot, cost, hj, ij = att.judge(maintree[nodeiter].represent, t, K)
    if hj != ht or ij != it:
        print("judge error!\n")
        exit(0)
    
    if attackornot:
        attsum[t] = attsum[t-1] + cost
        target[t] = target[t-1]
    else:
        attsum[t] = attsum[t-1]
        target[t] = target[t-1] + 1

    meanreward = f(comtree[comnodeiter].represent)
    comreward[t] = random.gauss(meanreward, 0.5)
    if K >= comtree[comnodeiter].start and K < comtree[comnodeiter].end:
        #meanreward = f(comtree[comnodeiter].represent)
        #comreward[t] = random.gauss(meanreward, 0.5)
        comattsum[t] = comattsum[t-1]
        comtarget[t] = comtarget[t-1] + 1
    else:
        #comreward[t] = random.gauss(minv, 1)
        cost = meanreward - f(K) + 1
        comreward[t] = comreward[t] - max(cost, 0)
        comattsum[t] = comattsum[t-1] + max(cost, 0)
        comtarget[t] = comtarget[t-1]

    meanreward = f(rantree[rannodeiter].represent)
    ranreward[t] = random.gauss(meanreward, 0.5)
    if K >= rantree[rannodeiter].start and K < rantree[rannodeiter].end:
        #meanreward = f(rantree[rannodeiter].represent)
        #ranreward[t] = random.gauss(meanreward, 1)
        ranattsum[t] = ranattsum[t-1]
        rantarget[t] = rantarget[t-1] + 1
    else:
        cost = random.random()
        #cost = 2*cost
        print("random cost: ", cost)
        #meanreward = f(arm)
        ranreward[t] = ranreward[t] - max(cost, 0)
        ranattsum[t] = ranattsum[t-1] + max(cost, 0)
        rantarget[t] = rantarget[t-1]

    meanreward = f(hcttree[hctnodeiter].represent)
    hctreward[t] = random.gauss(meanreward, 0.5)
    if K >= hcttree[hctnodeiter].start and K < hcttree[hctnodeiter].end:
        hcttarget[t] = hcttarget[t-1] + 1
    else:
        hcttarget[t] = hcttarget[t-1]

    sumreward[t] = sumreward[t-1] + reward[t]
    regret[t] = t * maxv - sumreward[t]
    #timelist[t] = t
    maintree[nodeiter].T += 1
    maintree[nodeiter].sumreward += reward[t]
    #t += 1

    comsumreward[t] = comsumreward[t-1] + comreward[t]
    comregret[t] = t * maxv - comsumreward[t]
    #timelist[t] = t
    comtree[comnodeiter].T += 1
    comtree[comnodeiter].sumreward += comreward[t]

    ransumreward[t] = ransumreward[t-1] + ranreward[t]
    ranregret[t] = t * maxv - ransumreward[t]
    #timelist[t] = t
    rantree[rannodeiter].T += 1
    rantree[rannodeiter].sumreward += ranreward[t]

    hctsumreward[t] = hctsumreward[t-1] + hctreward[t]
    hctregret[t] = t * maxv - hctsumreward[t]
    #timelist[t] = t
    hcttree[hctnodeiter].T += 1
    hcttree[hctnodeiter].sumreward += hctreward[t]

    timelist[t] = t
    t += 1

    maintree[nodeiter].U = calU(nodeiter, t)
    comtree[comnodeiter].U = comcalU(comnodeiter, t)
    rantree[rannodeiter].U = rancalU(rannodeiter, t)
    hcttree[hctnodeiter].U = hctcalU(hctnodeiter, t)
    #print(maintree[nodeiter].U)

    updateB(ht, it)
    comupdateB(comht, comit)
    ranupdateB(ranht, ranit)
    hctupdateB(hctht, hctit)
    #print(maintree[nodeiter].T, caltau(t, ht))
    print()
    if maintree[nodeiter].T >= caltau(t, ht) and maintree[nodeiter].leaf == 1:
        maintree[nodeiter].leaf = 0
        maintree.update({iternode(ht+1, 2*it-1): node(ht+1, 2*it-1)})
        maintree[iternode(ht+1, 2*it-1)].start = maintree[iternode(ht, it)].start
        maintree[iternode(ht+1, 2*it-1)].end = (maintree[iternode(ht, it)].start + maintree[iternode(ht, it)].end) / 2

        maintree.update({iternode(ht+1, 2*it): node(ht+1, 2*it)})
        maintree[iternode(ht+1, 2*it)].start = (maintree[iternode(ht, it)].start + maintree[iternode(ht, it)].end) / 2
        maintree[iternode(ht+1, 2*it)].end = maintree[iternode(ht, it)].end

        hmax = max(hmax, ht+1)

    if comtree[comnodeiter].T >= caltau(t, comht) and comtree[comnodeiter].leaf == 1:
        comtree[comnodeiter].leaf = 0
        comtree.update({iternode(comht+1, 2*comit-1): node(comht+1, 2*comit-1)})
        comtree[iternode(comht+1, 2*comit-1)].start = comtree[iternode(comht, comit)].start
        comtree[iternode(comht+1, 2*comit-1)].end = (comtree[iternode(comht, comit)].start + comtree[iternode(comht, comit)].end) / 2

        comtree.update({iternode(comht+1, 2*comit): node(comht+1, 2*comit)})
        comtree[iternode(comht+1, 2*comit)].start = (comtree[iternode(comht, comit)].start + comtree[iternode(comht, comit)].end) / 2
        comtree[iternode(comht+1, 2*comit)].end = comtree[iternode(comht, comit)].end

        comhmax = max(comhmax, comht+1)

    if rantree[rannodeiter].T >= caltau(t, ranht) and rantree[rannodeiter].leaf == 1:
        rantree[rannodeiter].leaf = 0
        rantree.update({iternode(ranht+1, 2*ranit-1): node(ranht+1, 2*ranit-1)})
        rantree[iternode(ranht+1, 2*ranit-1)].start = rantree[iternode(ranht, ranit)].start
        rantree[iternode(ranht+1, 2*ranit-1)].end = (rantree[iternode(ranht, ranit)].start + rantree[iternode(ranht, ranit)].end) / 2

        rantree.update({iternode(ranht+1, 2*ranit): node(ranht+1, 2*ranit)})
        rantree[iternode(ranht+1, 2*ranit)].start = (rantree[iternode(ranht, ranit)].start + rantree[iternode(ranht, ranit)].end) / 2
        rantree[iternode(ranht+1, 2*ranit)].end = rantree[iternode(ranht, ranit)].end

        ranhmax = max(ranhmax, ranht+1)

    if hcttree[hctnodeiter].T >= caltau(t, hctht) and hcttree[hctnodeiter].leaf == 1:
        hcttree[hctnodeiter].leaf = 0
        hcttree.update({iternode(hctht+1, 2*hctit-1): node(hctht+1, 2*hctit-1)})
        hcttree[iternode(hctht+1, 2*hctit-1)].start = hcttree[iternode(hctht, hctit)].start
        hcttree[iternode(hctht+1, 2*hctit-1)].end = (hcttree[iternode(hctht, hctit)].start + hcttree[iternode(hctht, hctit)].end) / 2

        hcttree.update({iternode(hctht+1, 2*hctit): node(hctht+1, 2*hctit)})
        hcttree[iternode(hctht+1, 2*hctit)].start = (hcttree[iternode(hctht, hctit)].start + hcttree[iternode(hctht, hctit)].end) / 2
        hcttree[iternode(hctht+1, 2*hctit)].end = hcttree[iternode(hctht, hctit)].end

        hcthmax = max(hcthmax, hctht+1)

att.fp.close()

'''
path = "./" + sys.argv[1]
if os.path.isdir(path):
    pass
else:
    os.makedirs(path)
'''

fpregret = open(path+"/regret","w")
for i in range(len(regret)):
    fpregret.write(str(regret[i]))
    fpregret.write(" ")
fpregret.write("\n")
for i in range(len(comregret)):
    fpregret.write(str(comregret[i]))
    fpregret.write(" ")
fpregret.write("\n")
for i in range(len(ranregret)):
    fpregret.write(str(ranregret[i]))
    fpregret.write(" ")
fpregret.write("\n")
for i in range(len(hctregret)):
    fpregret.write(str(hctregret[i]))
    fpregret.write(" ")
fpregret.write("\n")
fpregret.close()

fpcost = open(path+"/cost","w")
for i in range(len(attsum)):
    fpcost.write(str(attsum[i]))
    fpcost.write(" ")
fpcost.write("\n")
for i in range(len(comattsum)):
    fpcost.write(str(comattsum[i]))
    fpcost.write(" ")
fpcost.write("\n")
for i in range(len(ranattsum)):
    fpcost.write(str(ranattsum[i]))
    fpcost.write(" ")
fpcost.write("\n")
fpcost.close()

fptarget = open(path+"/target","w")
for i in range(len(target)):
    fptarget.write(str(target[i]))
    fptarget.write(" ")
fptarget.write("\n")
for i in range(len(comtarget)):
    fptarget.write(str(comtarget[i]))
    fptarget.write(" ")
fptarget.write("\n")
for i in range(len(rantarget)):
    fptarget.write(str(rantarget[i]))
    fptarget.write(" ")
fptarget.write("\n")
for i in range(len(hcttarget)):
    fptarget.write(str(hcttarget[i]))
    fptarget.write(" ")
fptarget.write("\n")
fptarget.close()

"""
fig, ax = plt.subplots(1, 1)#, figsize=(6, 4))
#print(hmax)
ax.plot(timelist, regret, color='cyan',linestyle='--',linewidth=2)
ax.plot(timelist, comregret, color='red',linestyle='-.',linewidth=2)
ax.plot(timelist, ranregret, color='brown',linestyle=':',linewidth=2)
ax.plot(timelist, hctregret, color='green',linestyle='-',linewidth=2)
#plt.title('Attack HCT regret')
ax.legend(labels=["Proposed attack", "Oracle attack","Random attack","Without attack"], fontsize=14)#ncol=4)
ax.set_xlabel("t",fontsize=14)
ax.set_ylabel("R(t)",fontsize=14)
ax.set_title("K={}".format(K), fontsize=14)
plt.show()
fig, ax = plt.subplots(1, 1)
ax.plot(timelist, attsum, color='cyan', linestyle='--',linewidth=2)
ax.plot(timelist, comattsum, color='red', linestyle='-.',linewidth=2)
ax.plot(timelist, ranattsum, color='brown', linestyle=':',linewidth=2)
ax.legend(labels=["Proposed attack", "Oracle attack","Random attack"], fontsize=14)#ncol=3)
ax.set_xlabel("t",fontsize=14)
ax.set_ylabel("C(t)",fontsize=14)
ax.set_title("K={}".format(K), fontsize=14)

axins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.6, 0.2, 1, 1),
                   bbox_transform=ax.transAxes)

axins.plot(timelist, attsum, color='cyan', linestyle='--',linewidth=2)
axins.plot(timelist, comattsum, color='red', linestyle='-.',linewidth=2)
axins.set_xlim(int(0.8e7), n)
axins.set_ylim(min(attsum[int(0.8e7)],comattsum[int(0.8e7)])-500,max(attsum[n],comattsum[n])+500)

mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
plt.show()

fig, ax = plt.subplots(1, 1)#, figsize=(6, 4))
#print(hmax)
ax.plot(timelist, target, color='cyan',linestyle='--',linewidth=2)
ax.plot(timelist, comtarget, color='red',linestyle='-.',linewidth=2)
ax.plot(timelist, rantarget, color='brown',linestyle=':',linewidth=2)
ax.plot(timelist, hcttarget, color='green',linestyle='-',linewidth=2)
#plt.title('Attack HCT regret')
ax.legend(labels=["Proposed attack", "Oracle attack","Random attack","Without attack"], fontsize=14)#ncol=4)
ax.set_xlabel("t",fontsize=14)
ax.set_ylabel("Target nodes pulls",fontsize=14)
ax.set_title("K={}".format(K), fontsize=14)
plt.show()
"""



