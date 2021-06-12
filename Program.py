#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries 

import matplotlib.pyplot as plt
import cv2
import numpy as np
import easyocr
from pylab import rcParams
from IPython.display import Image
rcParams['figure.figsize'] = 8, 16


# In[2]:


# Calling EasyOCR with 'en' for english.

reader = easyocr.Reader(['en'])


# In[3]:


# Displaying sample image

image = cv2.imread('Image Path')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
plt.imshow(image)


# In[4]:


# List, format - [bounding box coordinates, text, confidence score]

output = reader.readtext('Image Path')
output


# In[5]:


# Number of text localisations in image.

num = len(output)


# In[6]:


# Initialising arrays for creating bounding boxes.

x_min = np.zeros(num)
y_min = np.zeros(num)
x_max = np.zeros(num)
y_max = np.zeros(num)


# In[7]:


# Collecting bounding box coordinates and sketching as rectangles in sample image.

for j in range (0,num):
    x_min[j], y_min[j] = [int(min(idx)) for idx in zip(*output[j][0])]
    x_max[j], y_max[j] = [int(max(idx)) for idx in zip(*output[j][0])]
    cv2.rectangle(image,(int(x_min[j]),int(y_min[j])),(int(x_max[j]),int(y_max[j])),(0,255,0),2)


# In[8]:


# Text Detection (Displaying sample image with bounding boxes around text).
    
plt.imshow(image)


# In[9]:


# Text Recognition

for i in range (0,3):
    print(output[i][1])


# In[ ]:




