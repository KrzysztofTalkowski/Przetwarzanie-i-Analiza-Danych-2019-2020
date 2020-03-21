#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
a = np.array([1,2,3,4,5,6,7])
a


# In[7]:


b = np.array([[1,2,3], [4,5,6]])
b


# In[9]:


b = np.transpose(b)
b


# In[16]:


c = np.arange (0,100,1)
c


# In[18]:


d = np.arange(0,2,0.2)
d


# In[19]:


e = np.arange(0,100,5)
e


# In[23]:


f = np.random.rand(5,4)
np.round(f,2)


# In[30]:


g = np.random.randint(1,1000,100)
g


# In[31]:


h = np.zeros((3,2))
h


# In[32]:


i = np.ones((3,2))
i


# In[67]:


j = np.random.randint(100,size=(5,5),dtype='int32')
j


# In[98]:


a = np.random.uniform(0,10, size=(3,3))

a


# In[99]:


b = a.astype(int)
b


# In[102]:


c =np.round(a,0)
c


# In[104]:


d = c.astype(int)
d
# As type bierze wartości integerów takie jakie są(pełne),a round zaokrągla
# mamy więc sytuację taką, że dla astype 3,7 to 3, a dla round to 4.


# In[106]:


b = np.array([[1,2,3,4,5],[6,7,8,9,10]],dtype=np.int32)
b.ndim


# In[107]:


b.size


# In[113]:


b[(0,1)],b[(0,3)]


# In[115]:


b[(0)]


# In[120]:


b[(0,0)],b[(1,0)]


# In[122]:


c = np.random.randint(low = 0, high = 100,size=(20,7))
c


# In[127]:


c[:,0:4]


# In[128]:


a = np.random.randint(low = 0, high = 10,size=(3,3))
b = np.random.randint(low = 0, high = 10,size=(3,3))
a + b


# In[129]:


a * b


# In[130]:


a / b


# In[131]:


a^b


# In[135]:


a >= 4


# In[141]:


#Sprawdź czy wartość macierzy a 1¿= ¡=4.
# Tego nie zrozumiałem, stwierdziłem, że może coś takiego :
a[0,1] == 4


# In[143]:


np.diag(b)


# In[144]:


np.trace(b)


# In[145]:


np.sum(b)


# In[146]:


np.min(b)


# In[147]:


np.max(b)


# In[148]:


np.std(b)


# In[150]:


b


# In[151]:


b.mean(0)


# In[152]:


b.mean(1)


# In[159]:


a = np.random.randint(1,100,50)
a


# In[160]:


b = a.reshape((10,5))
b


# In[163]:


c = np.resize(a,(10,5))
c


# In[165]:


a = np.random.randint(1,100,5)
b = np.random.randint(1,100,4)
b_new = b[:, np.newaxis]
a+b_new


# In[166]:


a = np.random.randn(5,5)
a


# In[169]:


b = np.sort(a, axis=-1)
b


# In[170]:


c = np.sort(a, axis=-2)
c


# In[175]:


b=np.array([(1,'MZ','mazowieckie'),(2,'ZP','zachodniopomorskie'),(3,'ML','małopolskie')])
b


# In[176]:


c = np.matrix(b)
c


# In[181]:


d = np.sort(c, axis = 1)
d


# In[184]:


d[1,2]


# In[186]:


####3####
a = np.random.randint(1,100,size = (10,5))
a


# In[188]:


np.trace(a)


# In[189]:


np.diag(a)


# In[193]:


a = np.random.rand(3,3)
b = np.random.rand(3,3)
c = a * b
c


# In[200]:


a = np.random.randint(1,100,size = (5,5))
a
b = np.random.randint(1,100,size = (5,5))
b
a_m = np.matrix(a)
a_m
b_m = np.matrix(b)
b_m
a_m + b_m


# In[209]:


a = np.random.randint(1,10, size=(4,5))
a
b = np.random.randint(1,10, size=(5,4))
b
b_new = b.reshape(4,5)
a+b_new


# In[210]:


b_new


# In[211]:


b


# In[212]:


b_new[:,3] * b_new [:,4]


# In[213]:


a[:,3] * a[:,4]


# In[214]:


(b_new[:,3] * b_new [:,4]) * (a[:,3] * a[:,4])


# In[217]:


a = np.random.normal(0,1,(3,3))
a


# In[229]:


b = np.random.normal(0,1,(3,3))
b


# In[230]:


c = np.random.uniform(0,1,(3,3))
c


# In[232]:


a.mean()


# In[233]:


b.mean()


# In[234]:


c.mean()


# In[235]:


np.std(a)


# In[236]:


np.std(b)


# In[237]:


np.std(c)


# In[238]:


np.var(a)


# In[239]:


np.var(b)


# In[240]:


np.var(c)


# In[241]:


d = np.random.uniform(0,1,(3,3))
d


# In[242]:


np.std(d)


# In[243]:


d.mean()


# In[245]:


np.var(d)


# In[246]:


## średnia z random normal jest niższa niż z random uniform, 
## ale odchylenie standardowe i wariancja są wyższe


# In[255]:


a = np.random.randint(1,4, (3,3))
a


# In[256]:


b = np.random.randint(1,4,(3,3))
b


# In[257]:


a * b


# In[259]:


np.dot(a,b)


# In[260]:


### dot to iloczyn skalarny,a * to mnożenie macierzy, które mnoży każdy element
### jednej macierzy, przez odpowiadający element drugiej macierzy


# In[266]:


from numpy.lib.stride_tricks import as_strided
a = np.random.randint(1,4, size=(6,6))
a


# In[274]:


as_strided(a, strides=(3*5, ), shape=(2, ))


# In[275]:


b = np.random.randint(1,10, size=(2,3))
b


# In[276]:


c = np.random.randint(1,10, size=(5,4))
c


# In[277]:


d = np.vstack((b,c))
d


# In[278]:


e = np.hstack((b,c))
e


# In[279]:


## v stack - vertically, h stack horizontally
## zależy czy wolimy mieć tablicę z większą ilością kolumn/wierszy - 
## zależy od sytuacji w jakiej się znajdujemy


# In[ ]:




