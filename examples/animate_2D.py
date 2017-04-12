import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Producing an animated plot of 2D data with a colormap
# http://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate
# http://venusch.blogspot.nl/2012/12/2d-animation-in-xy-plane-using-scatter.html

N = 30
time = 30
c=np.zeros(shape=(time,N,N))  #[time, y, x]

# generate some example data to show
for t in range(time):
    for y in range (N):
        for x in range (N):
            xx = x - N/2.0
            yy = y - N/2.0
            c[t,y,x] = np.hypot(xx, yy) * ( xx * np.cos(t/time*2.0*np.pi) + yy * np.sin(t/time*2.0*np.pi) )

fig = plt.figure()
ax = plt.axes() #xlim=(0, 10), ylim=(0, 10))
im = plt.imshow(c[1,:,:]) 

#show a color bar
plt.colorbar()

#a text label, used later to show frame number
txt = plt.text(1,1,'')

# function for creating an empty plot
def init():
    im.set_data(c[1,:,:])
    return [im]

# i is the frame number
def animate(i,fig,im):
    a=im.get_array()
    a=c[i,:,:]
    im.set_array(a)

    # if you want the color scale to update for each frame
    # may be confusing, but can be good for noticing
    # if the values grow out of the color scale
    # im.autoscale()

    #plot the frame number
    txt.set_text(i)

    # return a list of all the plot bobjects that have been changed
    # note: must be a list, even if it has only one element: [im]
    return[im, txt]


ani=animation.FuncAnimation(fig,animate,init_func=init,fargs=(fig,im),frames=time)
# fargs is extra arguments to pass to the animation function. The frame number i is always passed.

# save a video file
# ani.save('animation.mp4', fps=20)

# save an animated gif
# from http://www.jamesphoughton.com/2013/08/making-gif-animations-with-matplotlib.html
#ani.save('animation.gif', writer='imagemagick', fps=4);

plt.show()




            



