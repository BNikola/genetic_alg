# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:13:50 2019

@author: Nikola
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


# region functions
def z1(x, y):
    rez = 3 * (1 - x)**2 * math.e ** (-(x**2 + (y+1)**2)) - 10*(x/5 - x**3 - y**5)*math.e**(-(x**2 + y**2)) - (1/3) * math.e ** (-((x+1)**2 + y**2))
    return rez

def encode(x, Gd, Gg, n):
    return np.int64(np.floor((x-Gd) * (2**n - 1) / (Gg - Gd)))

def decode(d, Gd, Gg, n):
    return Gd + (Gg - Gd) * d / (2**n - 1)

def show(ft):
    global X, Y, Z, population_size, Gd, Gg, n, population_x, population_y, population_x_coded, population_y_coded, \
        population_value, population_p, population_q, \
        population_x_bin, population_y_bin, population_fitness, r_x, r_y, best, best_x, best_y, do_display, fig, ax, minOrMax
    population_x_coded = encode(population_x, Gd, Gg, n)
    population_y_coded = encode(population_y, Gd, Gg, n)
    binary_list_x = [np.int64(list(np.binary_repr(i,n))) for i in population_x_coded]
    binary_list_y = [np.int64(list(np.binary_repr(i,n))) for i in population_y_coded]
    population_x_bin = np.array(binary_list_x)
    population_y_bin = np.array(binary_list_y)
    population_value = z1(population_x, population_y)

    if minOrMax == 1:
        # min
        M = max(population_value)
        population_fitness = M - population_value
    else:
        # max
        M = min(population_value)
        population_fitness = population_value - M

    total_fitness = np.sum(population_fitness)
    population_p = population_fitness / total_fitness
    population_q = np.array([np.sum(population_p[:i]) for i in range(1, len(population_p) + 1)])

    # formatting the data frame
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 15)

    df = pd.DataFrame()
    if ft == 1:
        df['sb_x'] = r_x
    df['Pop_x']      = population_x
    df['Pop_x_c']    = population_x_coded

    if ft == 1:
        df['sb_y'] = r_y
    df['Pop_y']      = population_y
    df['Pop_y_c']    = population_y_coded

    df['Population value'] = population_value
    df['Population fitness'] = population_fitness
    df['Population p'] = population_p
    df['Population q'] = population_q

    print(df)

    # choosing the best
    if minOrMax == 1:
        # min
        best_x = population_x[np.argmin(population_value)]
        best_y = population_y[np.argmin(population_value)]
    else:
        # max
        best_x = population_x[np.argmax(population_value)]
        best_y = population_y[np.argmax(population_value)]
    ax.cla()
    cs = ax.contourf(X, Y, Z, np.arange(-10, 10, 0.7), cmap=cm.RdBu_r)
    ax.autoscale(False)
    ax.scatter(population_x, population_y, zorder=10)
    ax.scatter(best_x, best_y, zorder=15, c='lightgreen', s=66)
    best = z1(best_x, best_y)
    ax.annotate(np.round(best, 3), (best_x + 0.1, best_y + 0.1), c='g')
    plt.title("Broj ponavljanja: " + str(ft) + " best(x: " + str("{:5.4f}").format(best_x)  + ", " + str("{:5.4f}").format(best_y) + ")")
    plt.pause(0.2)





# endregion

# region Parameters
do_display = 1
minOrMax = 1
x = np.arange(-3.0, 3.0, 0.01)
y = np.arange(-3.0, 3.0, 0.01)
X,Y = np.meshgrid(x,y)
Z = z1(X,Y)

population_size = 30
max_iter        = 1000
max_same        = 60
prob_co         = 0.85
prob_mut        = 0.15
Gd              = -3
Gg              = 3
elit            = 1
prec            = 3

n = np.int64(np.ceil(np.log((Gg-Gd) * 10 ** prec + 1) / np.log(2)))

fig = plt.figure()
ax = fig.add_subplot(111)
cs = ax.contourf(X, Y, Z, np.arange(-10, 10, 0.7), cmap=cm.RdBu_r)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.autoscale(False)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# plt.show()

# region Population data
population_x        = np.zeros(population_size)
population_y        = np.zeros(population_size)
population_x_coded  = population_x
population_y_coded  = population_y
population_value    = population_x
population_p        = population_x
population_q        = population_x
population_fitness  = population_x
next_gen_x            = population_x
next_gen_x_coded      = np.int64(population_x)
next_gen_y            = population_y
next_gen_y_coded      = np.int64(population_y)
population_x_bin      = np.zeros((population_size, n), dtype=np.int64)
population_y_bin      = np.zeros((population_size, n), dtype=np.int64)
r_x                   = np.random.uniform(size=population_size)
r_y                   = np.random.uniform(size=population_size)

population_x = (Gg - Gd) * r_x + Gd
population_y = (Gg - Gd) * r_y + Gd
print(n)
print(population_x)
print(population_y)

ax.scatter(population_x, population_y)

show(1)


done = 0
bc      = 1
iter    = 1
best_x    = 0
best_y    = 0
cBest_x   = best_x
cBest_y   = best_y
cBest     = best

# while loop
# todo - make: done == 0
while done == 0:
    tmp_v = population_value
    tmp_p_x = population_x
    tmp_p_y = population_y

    if minOrMax == 1:
        # min
        ind = np.argpartition(tmp_v, 2*elit)[:2*elit]
    else:
        # max
        ind = np.argpartition(tmp_v, -2*elit)[-2*elit:][::-1]

    if elit > 0:
        next_gen_x[:2*elit] = tmp_p_x[ind]
        next_gen_y[:2*elit] = tmp_p_y[ind]
    for i in range(2*elit, population_size):
        r = np.random.uniform()
        j = 0
        while population_q[j] < r:
            j += 1
        next_gen_x[i] = population_x[j]
        next_gen_y[i] = population_y[j]
    next_gen_x_coded = encode(next_gen_x, Gd, Gg, n)
    population_x = next_gen_x
    population_x_coded = next_gen_x_coded

    next_gen_y_coded = encode(next_gen_y, Gd, Gg, n)
    population_y = next_gen_y
    population_y_coded = next_gen_y_coded

    #rekombinacija
    # todo - ispis rekombinacije
    for i in range(elit, np.int64(np.floor(population_size/2))):
        r = np.random.uniform()
        da = 0
        tr_x = 0
        tr_y = 0
        if r < prob_co:                                               # da li se vrsi rekombinacija u i-tom paru
            for j in range(2):
                da = 1
                r_x  = np.random.uniform()
                if r_x > 0.3 or r_x < 0.8:
                    r_y = np.random.uniform()
                else:
                    r_y = r_x
                tr_x = np.int64(np.floor(r_x * (n - 1)))  # tacka u kojoj se vrsi rekombinacija za x
                tr_y = np.int64(np.floor(r_y * (n - 1)))  # tacka u kojoj se vrsi rekombinacija za y
                temp = 2 ** n - 1                                         # najveci broj sa n bita
                # Za x
                m1_x = np.bitwise_and(temp * 2 ** (tr_x + 1), temp)             # prva maska
                m2_x = np.bitwise_xor(m1_x, temp)                             # druga maska

                # print(np.binary_repr(m1_x, width=n))
                # print(np.binary_repr(m2_x, width=n))

                next_gen_x_coded[2*i - 1] = np.bitwise_or(np.bitwise_and(population_x_coded[2*i - 1], m1_x), np.bitwise_and(population_x_coded[2 * i], m2_x))
                next_gen_x_coded[2*i]     = np.bitwise_or(np.bitwise_and(population_x_coded[2*i - 1], m2_x), np.bitwise_and(population_x_coded[2 * i], m1_x))

                next_gen_x[2*i - 1] = decode(next_gen_x_coded[2*i - 1], Gd, Gg, n)
                next_gen_x[2*i]     = decode(next_gen_x_coded[2*i], Gd, Gg, n)

                # za Y
                m1_y = np.bitwise_and(temp * 2 ** (tr_y + 1), temp)  # prva maska
                m2_y = np.bitwise_xor(m1_y, temp)  # druga maska

                # print(np.binary_repr(m1_y, width=n))
                # print(np.binary_repr(m2_y, width=n))

                next_gen_y_coded[2 * i - 1] = np.bitwise_or(np.bitwise_and(population_y_coded[2 * i - 1], m1_y),
                                                            np.bitwise_and(population_y_coded[2 * i], m2_y))
                next_gen_y_coded[2 * i] = np.bitwise_or(np.bitwise_and(population_y_coded[2 * i - 1], m2_y),
                                                        np.bitwise_and(population_y_coded[2 * i], m1_y))

                next_gen_y[2 * i - 1] = decode(next_gen_y_coded[2 * i - 1], Gd, Gg, n)
                next_gen_y[2 * i] = decode(next_gen_y_coded[2 * i], Gd, Gg, n)

    population_x = next_gen_x
    population_x_coded = next_gen_x_coded

    population_y = next_gen_y
    population_y_coded = next_gen_y_coded

    # mutacija
    # todo ispis mutacije
    for i in range(2*elit, population_size):
        r_x = np.random.uniform()
        r_y = np.random.uniform()
        da = 0
        tr = 0
        if r_x < prob_mut:
            da = 1
            #todo - change random uniform to normal!
            tr = np.int64(np.floor(np.random.uniform() * n))
            next_gen_x_coded[i] = np.bitwise_xor(2**tr, population_x_coded[i])
            next_gen_x[i] = decode(next_gen_x_coded[i], Gd, Gg, n)
        # if r_x < prob_mut:
        #     da = 1
            if tr < 0.3 or tr > 0.8:
                tr = np.int64(np.floor((np.random.uniform()) * n))
            next_gen_y_coded[i] = np.bitwise_xor(2**tr, population_y_coded[i])
            next_gen_y[i] = decode(next_gen_y_coded[i], Gd, Gg, n)

    population_x = next_gen_x
    population_x_coded = next_gen_x_coded

    population_y = next_gen_y
    population_y_coded = next_gen_y_coded

    iter += 1
    show(iter)

    print(str(bc) + "---" + str(best) + "---" + str(cBest))
    if cBest == best:
        bc += 1
    else:
        cBest = best
        bc = 0
    if bc > max_same or iter > max_iter:
        done = 1

# endregion

# endregion


