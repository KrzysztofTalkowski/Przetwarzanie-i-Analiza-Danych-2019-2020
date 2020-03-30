#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import pylab as py
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


# In[ ]:





# In[13]:


def sin(f = 1, T = 1, Fs = 128, phi =0 ):
    
    
    dt = 1.0/Fs
    t = np.arange(0,T,dt)
    s = np.sin(2*np.pi*f*t + phi)
    return (s,t)


# In[14]:


(s,t) = sin(f=10,Fs=20)
py.plot(t,s)
py.show()


# In[15]:


(s,t) = sin(f=10,Fs=21)
py.plot(t,s)
py.show()


# In[16]:


(s,t) = sin(f=10,Fs=30)
py.plot(t,s)
py.show()


# In[17]:


(s,t) = sin(f=10,Fs=45)
py.plot(t,s)
py.show()


# In[18]:


(s,t) = sin(f=10,Fs=50)
py.plot(t,s)
py.show()


# In[19]:


(s,t) = sin(f=10,Fs=100)
py.plot(t,s)
py.show()


# In[20]:


(s,t) = sin(f=10,Fs=150)
py.plot(t,s)
py.show()


# In[21]:


(s,t) = sin(f=10,Fs=200)
py.plot(t,s)
py.show()


# In[22]:


(s,t) = sin(f=10,Fs=250)
py.plot(t,s)
py.show()


# In[23]:


(s,t) = sin(f=10,Fs=1000)
py.plot(t,s)
py.show()


# In[24]:


# 2.4
# Istnieje, jest to twierdzenie Nyquista-Shannona 
# Aby wiernie odtworzyć sygnał maksymalna częstotliwość sygnału nie powinna przekraczać połowy częstotliwości próbkowania
# Jest to tzw. częstotliwość Nyquista tzn. częstotliwość Nyquista jest równa połowie częstotliwości próbkowania


# In[63]:


# 2.5
# Aliasing i jest to nieodwracalne zniekształcenie sygnału w procenie próbkowania wynikające z niespełnienia założeń
# twierdzenia o próbkowaniu


# In[ ]:





# [title] (img/widok.png)

# In[69]:


from IPython.display import Image
Image(filename="widok.png")


# In[81]:


widok=mpimg.imread('widok.png', 0)
imgplot = plt.imshow(widok)
plt.show()


# In[82]:


print(widok.ndim)


# In[85]:


print(widok.shape)


# In[86]:


print('Maximum RGB value in this image {}'.format(widok.max()))


# In[88]:


print('Minimum RGB value in this image {}'.format(widok.min()))


# In[100]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2)])

def rgb_to_gray2(widok):
        graywidok2 = np.zeros(widok.shape)
        R = np.array(widok[:, :, 0])
        G = np.array(widok[:, :, 1])
        B = np.array(widok[:, :, 2])

        Avg = (R+G+B)
        graywidok2 = widok

        for i in range(3):
           graywidok2[:,:,i] = Avg/3

        return graywidok2  


# In[101]:


def rgb_to_gray3(widok):
        graywidok3 = np.zeros(widok.shape)
        R = np.array(widok[:, :, 0])
        G = np.array(widok[:, :, 1])
        B = np.array(widok[:, :, 2])

        Avg = (0.21*R+0.72*G+0.07*B)
        graywidok3 = widok

        for i in range(3):
           graywidok3[:,:,i] = Avg

        return graywidok3 


# In[106]:


widok = mpimg.imread('widok.png', 0)     
gray = rgb2gray(widok)    
plt.imshow(gray,cmap = plt.get_cmap(name = 'gray'))
plt.show()


# In[120]:


widok = mpimg.imread('widok.png',0)   
gray2 = rgb_to_gray2(widok)  
plt.imshow(graywidok2)
plt.show()


# In[123]:


widok = mpimg.imread('widok.png', 0)   
gray3 = rgb_to_gray3(widok)  
plt.imshow(gray3)
plt.show()


# In[126]:


plt.hist(gray.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()
plt.hist(gray2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()
plt.hist(gray3.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()


# In[127]:


plt.hist(gray3.ravel(), bins=16, range=(0.0, 1.0), fc='k', ec='k')
plt.axvline(gray3.mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()


m=gray3.mean()
print(gray3)


# In[207]:


# 3.7 Tutaj niestety nie potrafiłem tego zrobić po wielu próbach


# In[233]:


image = mpimg.imread('triangle.png', 0)   
gradient = rgb_to_gray3(image)  
plt.imshow(gradient)
plt.show()


# In[244]:


def hist(histogram):
    norm=np.linalg.norm(histogram)
    return histogram/norm

ab=hist(gradient[0])
ab=np.mean(ab)

print(ab)


# In[254]:


black = (image>ab)*255
plt.imshow(black)
plt.show()
print(image)


# In[ ]:




