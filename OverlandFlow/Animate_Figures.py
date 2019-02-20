"""
Author: Amanda Manaster
Date: 04/05/2017
Purpose: Create animation of figures from OverlandFlow_create_own_grid.py
"""
import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation
import numpy as np
plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.8-Q16\magick.exe'


#initialize figure
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

#create empty list for images 
myimages = []

T = np.arange(0,1100,50)

#loops through images
for i in range(len(T)):

    #read in figure
    fname = 'C:/Users/Amanda/Desktop/3DFigs/year%i.png' % T[i] 
    img = mgimg.imread(fname)
    imgplot = plt.imshow(img)

    #append image to the list
    myimages.append([imgplot])

#animate
my_anim = animation.ArtistAnimation(fig, myimages)

writer = animation.ImageMagickFileWriter(fps = 2)

#save animation
my_anim.save('C:/Users/Amanda/Desktop/3D_1000yrs.gif', writer=writer, dpi = 300)
