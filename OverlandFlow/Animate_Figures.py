"""
Author: Amanda Manaster
Date: 04/05/2017
Purpose: Create animation of figures from OverlandFlow_create_own_grid.py
"""
import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation
from JSAnimation import HTMLWriter

#initialize figure
fig = plt.figure(frameon = False, figsize=(6,10))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

#create empty list for images 
myimages = []

#loops through images
for i in range(9):

    #read in figure
    fname = 'C:/Users/Amanda/Desktop/Python/Slope%i_200.png' % i 
    img = mgimg.imread(fname)
    imgplot = plt.imshow(img)

    #append image to the list
    myimages.append([imgplot])

#animate
my_anim = animation.ArtistAnimation(fig, myimages)

#save animation
my_anim.save('C:/Users/Amanda/Desktop/Python/HydrographElevation_200.html',  
             writer=HTMLWriter(embed_frames = True))
