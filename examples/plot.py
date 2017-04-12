#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# an array of x values
x = np.arange(0, 7, .1)

# y values. The whole vector at once! sin() is applied to each component
y = np.sin(x)

# a vector of 0, same shape as x
y2 = np.zeros_like(x)

# element-wise works too
for i in range(0, len(x)):
    y2 = 1.0/2 * np.sin(2 * x) 

# list comprehension - short and powerful
y3 = [1.0/3 * np.sin(3*t) for t in x]

plt.plot(x, y,  label='this is y')
plt.plot(x, y2, label='this is y2')
plt.plot(x, y3, label='this is y3')
plt.legend(loc='upper right')

plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()


plt.savefig('sin.png')
# can also save pdf, eps, svg, ...

# show the plot(s)
# halts the program for as long as the plot windows are open
plt.show()









