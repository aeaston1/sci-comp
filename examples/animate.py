#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import animation



# example from:
# http://nbviewer.ipython.org/github/jakevdp/matplotlib_pydata2013/blob/master/notebooks/05_Animations.ipynb

# other nice resource:
# https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()

# set up axes with given limits
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# note, we use the object-oriented matplotlib interface this time
# that's why these commands look different
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.ylabel('y')    # still works, but mixing looks messy
 

# initialization function: plot what is common to each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called repeatedly
# should return the figure elements that have changed
x = np.linspace(0, 2, 100)

def animate(i):
    print(i)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)


# save movie to file. Requires ffmpeg to be installed.
# anim.save('animation.mp4', fps=20) 

# anim.save('animation.gif', fps=20) # didn't work :(

plt.show()


