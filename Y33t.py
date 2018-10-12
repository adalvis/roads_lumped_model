# -*- coding: utf-8 -*-
"""
Purpose: To yeet all your friends
Authors: Panda and Sebulbous
Date: 06/09/2018    
"""

import numpy as np
import matplotlib.pyplot as plt

def yeet(n):
    print('YEET '*n)
    
def yotenheim(n):
      for i in range(n):
          if i < 6:
              print('This is not the yotenheim you\'re looking for.')
          else:
              print('1/6 YOTENHEIM')

def y33t(n):
    string = "yeet "
    for i in range(3,n):
        string +="y"
        for j in range(i):
            string += "e"
        string += "t "    
    return string    

def ratio(string):
    numYeet = 0
    numE = 0
    
    for i in range(len(string)):
        if string[i] == 'y':
            numYeet += 1
        elif string[i] == 'e':
            numE += 1;
    
    eYeetRatio = numE*1.0/numYeet    
    print('Your ratio of e to yeet is',eYeetRatio)
    return eYeetRatio

def graphYeetRatio(numYeet):
    # Graph e to yeet ratio vs yeet string parameter n
    yeetX = np.empty(numYeet)
    yeetY = np.empty(numYeet)
    
    for i in range(numYeet):
        yeetX[i] = i
        yeetY[i] = ratio(y33t(i))
        
    plt.plot(yeetX,yeetY)
    plt.show()    



yeet(69)              
yotenheim(10)        
string = y33t(6)
print(string)
ratio(string)
graphYeetRatio(50)