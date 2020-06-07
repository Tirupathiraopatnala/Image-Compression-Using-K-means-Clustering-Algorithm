'''Task-1
    Importing Libraries'''

from __future__ import print_function
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use("ggplot")

from skimage import io #for image processing
from sklearn.cluster import KMeans #KMeans algorithm 

from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider # for interactive input using a picklist
import ipywidgets as widgets


''' Task 2
    Data Preprocessing ''' 
    
mg  = io.imread('images/1-Saint-Basils-Cathedral.jpg') #don't forgot chane the file path
ax = plt.axes(xticks=[],yticks=[])
ax.imshow(img);


img.shape #pixel dimensions of the image

img_data = (img/255.0).reshape(-1,3) #reshapiing it to a Vector
img_data.shape



''' Task 3 
    Visualizing the Color Space using Point Clouds'''
    
from plot_utils import plot_utils 


x = plot_utils(img_data,title="Input image")
x.colorSpace()


'''Task 4
    Reduced color space using k-means'''
    
from sklearn .cluster import MiniBatchKMeans


kmeans = MiniBatchKMeans(16).fit(img_data)
k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]


y = plot_utils(img_data,colors=k_colors,title="Reduced Colors")
y.colorSpace()



'''Task 5 
    K-means Image Compression with Interactive Controls'''
    
    img_dir = 'images/'  # give your own Path 



@interact
def color_compression(image= os.listdir(img_dir),k=IntSlider(min=1,max=256,step=1,
                                                            continuous_update=False,
                                                           layout=dict(width='100%'))):
    input_img = io.imread(img_dir+image)
    img_data = (input_img/255.0).reshape(-1,3)
    kmeans = MiniBatchKMeans(k).fit(img_data)
   
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    k_img = np.reshape(k_colors,(input_img.shape))
   
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.suptitle('K-means Image Compression',fontsize=20)
   
    ax1.set_title('compressed')
    ax1.set_xticks([])

    ax1.set_yticks([])
    ax1.imshow(k_img)
   
    ax2.set_title('Original')
    ax2.set_xticks([])

    ax2.set_yticks([])
    ax2.imshow(input_img)
   
    plt.subplots_adjust(top=0.85)
    plt.show()