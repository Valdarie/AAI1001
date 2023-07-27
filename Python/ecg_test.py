#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Preprocessing

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


# In[2]:


dir = Path('archive/test')
filepaths = list(dir.glob('**/*.png'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath', dtype=str)
labels = pd.Series(labels, name='Label', dtype=str)

dataframe_test = pd.concat([filepaths , labels] , axis=1)
dataframe_test


# In[3]:


dataframe_test['Label'].value_counts()


# ### 2. Data Balancing

# In[4]:


samples = []
for category in ['N','M','Q','V','S','F']:
    category_slice = dataframe_test.query('Label == @category')
    samples.append(category_slice.sample(160, random_state=1))

dataframe_test = pd.concat(samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
dataframe_test['Label'].value_counts()


# In[5]:


dataframe_test


# ### 3. Image Preprocessing and Data Augmentation

# In[6]:


def gray_torgb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    image = cv2.merge((image, image, image))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


# In[7]:


test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=gray_torgb,
    rescale=1./255
)


# In[8]:


size = 224
color_mode = 'rgb'
batch_size = 32

test_images = test_generator.flow_from_dataframe(
    dataframe=dataframe_test,        # Input dataframe containing file paths and labels
    x_col='Filepath',                # Name of the dataframe column containing file paths
    y_col='Label',                   # Name of the dataframe column containing labels
    target_size=(size, size),        # Target size for resizing the images
    color_mode=color_mode,           # Color mode for the images
    class_mode='categorical',        # Type of label encoding ('categorical' for one-hot encoded labels)
    batch_size=batch_size,           # Number of images in each batch
    shuffle=False                    # No shuffle of data during testing
)


# In[9]:


test_images.class_indices


# In[10]:


plt.imshow(test_images[0][0][3])


# ### 4. Model Evaluation

# In[11]:


from keras.models import load_model

def load_model(model_path):
    best_model = load_model(model_path)
    return best_model


# In[12]:


from keras.models import load_model

def evaluate_model(model_path, test_images):
    # Load the pre-trained model
    best_model = load_model(model_path)
    
    # Evaluate the model on the test images
    results = best_model.evaluate(test_images, verbose=0)
    
    # Print the evaluation results
    print('Test Loss     : {:.4f}'.format(results[0]))
    print('Test Accuracy : {:.4f}%'.format(results[1] * 100))
    print('Test Precision: {:.4f}%'.format(results[2] * 100))
    print('Test Recall   : {:.4f}%'.format(results[3] * 100))
    print('Test AUC      : {:.4f}'.format(results[4]))

evaluate_model('ECG_Model.h5', test_images)


# ### 5. Predictions

# In[13]:


best_model = load_model('ECG_Model.h5')
y_pred = best_model.predict(test_images)
y_pred


# In[14]:


# Convert the predicted probabilities to class labels
# np.argmax() is used to find the index of the maximum probability along the specified axis (axis=1 in this case).
y_pred = np.argmax(y_pred, axis=1)
y_pred


# ### Confusion Matrix and Classification Report

# In[15]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(test_images.labels, y_pred)
print(cm)

report = classification_report(test_images.labels, y_pred, target_names=['F', 'M', 'N', 'Q', 'S', 'V'],digits=4)
print(report)

def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df
report = report_to_df(report)

report.to_csv('classification_report.csv', index=True)


# ### Confusion Matrix Plotting

# In[16]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, without normalization')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar()

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.rcParams['font.size'] = '20'
    
plot_confusion_matrix(cm, classes=['F', 'M', 'N', 'Q', 'S', 'V'],normalize=False,title='Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix' + '.jpg', dpi=500, bbox_inches='tight')


# In[18]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, without normalization')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar()

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.rcParams['font.size'] = '7'
    
plot_confusion_matrix(cm, classes=['F', 'M', 'N', 'Q', 'S', 'V'],normalize=True,title='Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('normalized_confusion_matrix' + '.jpg', dpi=500, bbox_inches='tight')

