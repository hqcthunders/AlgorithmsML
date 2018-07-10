#!/usr/bin/env python3
# Softmax Regression is one of algorithms most used in Classification Problems.
# vector a
# a(i) = exp(z(i)) / ∑(j=0 -> d-1)exp(z(j))
import numpy as np


def softmax(z):
    sum_exp = sum(np.exp(z))
    a = np.array([np.exp(z[i])/sum_exp for i in range(len(z))])
    return a


def main():
    np.random.seed(1)
    z = np.random.rand(10)
    print(softmax(z))


if __name__ == '__main__':
    main()
