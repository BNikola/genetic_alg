# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:08:58 2019

@author: Nikola
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

lista = np.array([v for v in range(10, 20)])


# Function energy
def energy(s):
    s = np.array(s)
    return s ** 2 + 10 * np.cos(s - 5)


a = energy(lista)


def neighbour(s, D):
    sn = s + D[0] * (2 * np.random.uniform() - 1)
    D[0] = 1.003 * D[0]
    return sn, D


D = np.array([2, 0])


def probability(old, new, temp):
    if (new < old):
        return 1
    else:
        return 0.1 * math.e ** ((old - new) / temp)


def temperature(k, TMax):
    return TMax * np.sqrt(1 - k ** 2)


print(temperature(0.001, 223))

##################################################
# Program
##################################################

s0 = -8  # pocetno stanje
kMax = 1000  # maksimalan broj iteracija
corr = np.array([2, 0])  # prva vrijednost je sirina, druga je korekcioni faktor
TMax = 10  # maksimalna temp

x = np.arange(-10, 10, 0.01)
y = energy(x)

s = s0
e = energy(s)

fig = plt.figure()
ax = fig.add_subplot(111)
print(s)
print(e)
ax.plot(x, y, 'c-', s, e, 'ro')
plt.ion()
plt.show()
plt.title("test")
# inicijalizacija brojaca
k = 0
while (k < kMax):
    ax.cla()
    ax.plot(x, y, 'c-', s, e, 'ro')
    plt.title("Broj ponavljanja: " + str(k))
    T = temperature(k / kMax, TMax)
    [sNew, corr] = neighbour(s, corr)
    eNew = energy(sNew)
    ax.plot(s, e, 'ro')
    if (eNew < e):
        corr[1] = e - eNew
    else:
        corr[1] = 0
    r = np.random.uniform()
    p = probability(e, eNew, T)
    if (p > r):
        s = sNew
        e = eNew
        ax.plot(s,e, 'ro')
    k = k + 1
    plt.pause(0.01)
plt.pause(5)    # pauzira na 5 sekundi
fig.ginput()    # ceka dok korisnik ne klikne (default jednom)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# print(s)
# print(e)
# ax.plot(x, y, 'c-', s, e, 'ro')
# plt.show()




