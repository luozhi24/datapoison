import numpy as np
import math
import random
from scipy import optimize
#from func import *

rho = 1/2
nu = 1
delta = 0.05
n = 10000000
###revise

"""
hctrho = 1/2
hctnu = 1
hctdelta  = 0.05
c = 2*math.sqrt(1/(1-hctrho))
"""

def f(x):
    return 0.5*(math.sin(13*x)*math.sin(27*x)+1)

class node:
    def __init__(self, h, i):
        self.h = h
        self.i = i
        self.L = float('-inf')
        self.B = float('-inf')
        self.sumreward = 0
        self.T = 0
        self.start = 0
        self.end = 0
        self.represent = -1
        self.leaf = 1

def iternode(h, i):
    return str(h) + "," + str(i)

class attack:
    def __init__(self, targetarm):
        #self.attacktree = dict()
        self.T1 = dict()
        root = node(0,1)
        root.T = 1
        #root.leaf = 0
        root.start = 0
        root.end = 1#*upper_use

        self.T1.update({iternode(0,1): root})

        #self.nodenum = 3
        #self.tm = 1
        self.reward = np.zeros(n+1)
        #self.nodenum = np.zeros(n+1)
        self.nodenum = 1
        self.brother = 0
        self.brotherkey = iternode(0, 1)
        self.fp = open("./" + targetarm + "/infor","w")

    def calB(self, nodeiter):
        sumT = self.T1[nodeiter].T
        return math.sqrt(1/(2*sumT)*math.log(math.pi**2*sumT**2*self.nodenum/(3*delta)))

    def judge(self, xt, t, k):
        h = 0
        i = 1
        #tau = 1
        flag = 0
        attackornot = 0

        nodeiter = iternode(h, i)
        print(k)

        if self.brother:
            nodeiter = self.brotherkey
            self.T1[nodeiter].represent = xt
            flag = 1
            self.brother = 0
            '''
            for key in self.T1.keys():
                if self.T1[key].h == 0:
                    continue
                if self.T1[key].represent == -1:
                    self.T1[key].represent = xt
                    nodeiter = key
                    flag = 1
                    self.brother = 0
                    break
            '''
        
        if flag == 0:
            for key in self.T1.keys():
                if xt == self.T1[key].represent:
                    flag = 1
                    nodeiter = key
                    break

        if flag == 0:
            for key in self.T1.keys():
                if self.T1[key].leaf == 1:
                    if xt >= self.T1[key].start and xt < self.T1[key].end:
                        nodeiter = key
                        break
            
            h = self.T1[nodeiter].h
            i = self.T1[nodeiter].i
            self.T1[nodeiter].leaf = 0
            self.T1.update({iternode(h+1, 2*i-1): node(h+1, 2*i-1)})
            self.T1[iternode(h+1, 2*i-1)].start = self.T1[iternode(h, i)].start
            self.T1[iternode(h+1, 2*i-1)].end = (self.T1[iternode(h, i)].start + self.T1[iternode(h, i)].end) / 2
            if xt >= self.T1[iternode(h+1, 2*i-1)].start and xt < self.T1[iternode(h+1, 2*i-1)].end:
                nodeiter = iternode(h+1, 2*i-1)
                self.brotherkey = iternode(h+1, 2*i)

            self.T1.update({iternode(h+1, 2*i): node(h+1, 2*i)})
            self.T1[iternode(h+1, 2*i)].start = (self.T1[iternode(h, i)].start + self.T1[iternode(h, i)].end) / 2
            self.T1[iternode(h+1, 2*i)].end = self.T1[iternode(h, i)].end
            if xt >= self.T1[iternode(h+1, 2*i)].start and xt < self.T1[iternode(h+1, 2*i)].end:
                nodeiter = iternode(h+1, 2*i)
                self.brotherkey = iternode(h+1, 2*i-1)

            self.T1[nodeiter].represent = xt
            self.nodenum += 2
            self.brother = 1
            
        
        meanreward = f(xt)
        rt = random.gauss(meanreward, 0.5)
        self.T1[nodeiter].T += 1
        self.T1[nodeiter].sumreward += rt
        
        """
        if self.T1[nodeiter].represent == -1:
            self.T1[nodeiter].represent = xt
        """
        #print("nodenum:", self.nodenum)

        if k >= self.T1[nodeiter].start and k < self.T1[nodeiter].end:
            attackornot = 0
            return rt, attackornot, 0, self.T1[nodeiter].h, self.T1[nodeiter].i
        
        attackornot = 1
        
        """
        h = self.T1[nodeiter].h
        i = self.T1[nodeiter].i
        while 1:
            h = h - 1
            if i % 2 == 0:
                i = int(i / 2)
            else:
                i = int((i + 1) / 2)
            if k >= self.T1[iternode(h, i)].start and k < self.T1[iternode(h, i)].end:
                break
        """
        estimin = float('inf')
        estimax = float('-inf')

        for key in self.T1.keys():
            if self.T1[key].T == 0 or self.T1[key].h == 0:
                continue
            esti = self.calB(key)
            mean = self.T1[key].sumreward/self.T1[key].T

            if mean - esti < estimin:
                estimin = mean - esti
                meanmin = mean
                esmin = esti
            
            if mean + esti > estimax:
                estimax = mean + esti
                meanmax = mean
                esmax = esti

        self.fp.write("node: " + str(nodeiter) + "\n")
        self.fp.write("represent: " + str(xt) + "\n")
        self.fp.write("mean-reward: " + str(meanreward) + "\n")

        self.fp.write("meanmax: " + str(meanmax) + "\n")
        self.fp.write("esmax: " + str(esmax) + "\n")
        self.fp.write("max: " + str(estimax) + "\n")

        self.fp.write("meanmin: " + str(meanmin) + "\n")
        self.fp.write("esmin: " + str(esmin) + "\n")
        self.fp.write("min: " + str(estimin) + "\n")

        Kesti = float('inf')
        Kkey = iternode(0,1)
        h = 0
        i = 1

        while 1:
            h = h + 1
            i = 2 * i - 1

            Kkey = iternode(h,i)
            if k>=self.T1[Kkey].start and k<self.T1[Kkey].end:
                pass
            else:
                i = i + 1
                Kkey = iternode(h,i)

            if self.T1[Kkey].leaf == 1:
                break

        if self.T1[Kkey].T != 0:
            Kesti = self.T1[Kkey].sumreward/self.T1[Kkey].T - self.calB(Kkey)

        nowesti = self.T1[nodeiter].sumreward/self.T1[nodeiter].T + self.calB(nodeiter)

            
        #estimate2 = self.T1[iternode(h, i)].sumreward/self.T1[iternode(h, i)].T - self.calB(iternode(h, i))
        #print("estimate:",max(estimate1-estimate2, 0))
        self.fp.write("K: " + str(self.T1[Kkey].represent) + "\n")
        self.fp.write("f(K): " + str(f(self.T1[Kkey].represent)) + "\n")
        self.fp.write("Kesti: " + str(Kesti) + "\n")

        self.fp.write("nowesti: " + str(nowesti) + "\n")
        self.fp.write("cost: " + str(max(nowesti-Kesti+(estimax-estimin), 0)) + "\n\n")
        
        return rt - max(nowesti-Kesti+(estimax-estimin), 0), attackornot, max(nowesti-Kesti+(estimax-estimin), 0), self.T1[nodeiter].h, self.T1[nodeiter].i
        
        
        
    
        
        
