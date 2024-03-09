# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 02:08:04 2023

@author: Spare
"""
from random import sample
import random
import numpy as np
import matplotlib.pyplot as plt

N = 50  # Size of the lattice
q = 10  # Number of spin states
num_steps = 100  # Number of simulation steps

# Initialize the lattice
S = np.array([[random.randint(0, q-1) for _ in range(N)] for _ in range(N)])
# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()


for step in range(num_steps):
    print(step)
    for ni in range(0,N-1):
        for nj in range(0,N-1):

            ssub_00 = S[ni-1, nj-1]
            ssub_01 = S[ni, nj-1]
            ssub_02 = S[ni+1, nj-1]
            ssub_10 = S[ni-1, nj]
            ssub_11 = S[ni, nj]
            ssub_12 = S[ni+1, nj]
            ssub_20 = S[ni-1, nj+1]
            ssub_21 = S[ni, nj+1]
            ssub_22 = S[ni+1, nj+1]

            a = int(ssub_11 == ssub_00)
            b = int(ssub_11 == ssub_01)
            c = int(ssub_11 == ssub_02)
            d = int(ssub_11 == ssub_10)
            e = int(ssub_11 == ssub_12)
            f = int(ssub_11 == ssub_20)
            g = int(ssub_11 == ssub_21)
            h = int(ssub_11 == ssub_22)
            e0 = a+b+c+d+e+f+g+h

            Neigh = np.array([[ssub_00, ssub_10, ssub_20],
                              [ssub_01, ssub_11, ssub_21],
                              [ssub_02, ssub_12, ssub_22]])

            Neigh = set([x for x in Neigh.flatten() if x != ssub_11])
            if len(Neigh)==1:
                new_state = sample(Neigh, 1)[0]
                a = int(new_state == ssub_00)
                b = int(new_state == ssub_01)
                c = int(new_state == ssub_02)
                d = int(new_state == ssub_10)
                e = int(new_state == ssub_12)
                f = int(new_state == ssub_20)
                g = int(new_state == ssub_21)
                h = int(new_state == ssub_22)
                e1 = a+b+c+d+e+f+g+h
                if e0-e1 < 0.0:
                    states[ni][nj] = new_state

# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()

size = 10
states = np.random.randint(1, 50, (size, size))
for step in range(num_steps):

    for ni in range(1, size-1):
        for nj in range(1, size-1):
            current_sate = states[ni][nj]
            a = current_sate == states[ni-1][nj+0]
            b = current_sate == states[ni-1][nj-1]
            c = current_sate == states[ni+0][nj-1]
            d = current_sate == states[ni+1][nj-1]
            e = current_sate == states[ni+1][nj+0]
            f = current_sate == states[ni+1][nj+1]
            g = current_sate == states[ni+0][nj+1]
            h = current_sate == states[ni-0][nj+1]
            e0 = 1*a+1*b+1*c+1*d+1*e+1*f+1*g+1*h
            new_state = random.randint(0, q-1)
            a = new_state == states[ni-1][nj+0]
            b = new_state == states[ni-1][nj-1]
            c = new_state == states[ni+0][nj-1]
            d = new_state == states[ni+1][nj-1]
            e = new_state == states[ni+1][nj+0]
            f = new_state == states[ni+1][nj+1]
            g = new_state == states[ni+0][nj+1]
            h = new_state == states[ni-0][nj+1]
            e1 = 1*a+1*b+1*c+1*d+1*e+1*f+1*g+1*h
            if e0-e1 < 0.0:
                states[ni][nj] = new_state






import numpy as np
from numpy cimport ndarray
cimport cython
from libc.stdlib cimport rand

@cython.boundscheck(False)
@cython.wraparound(False)
def update_states(int num_steps, int S):
    cdef int size = 10
    cdef int[:, :] states = np.random.randint(1, S, (size, size))
    cdef int ni, nj, step
    cdef int current_state, new_state
    cdef int a, e0, e1

    # Seed the random number generator
    rand.seed()

    # Perform the simulation steps
    for step in range(num_steps):
        print(step)
        for ni in range(1, size-1):
            for nj in range(1, size-1):
                current_state = states[ni, nj]
                a = int(current_state == states[ni-1, nj+0])
                e0 = a

                new_state = rand.randint(0, S-1)
                a = new_state == states[ni-1, nj+0]
                e1 = a

                if e0 - e1 < 0.0:
                    states[ni, nj] = new_state

    return states  # Or any other desired output
