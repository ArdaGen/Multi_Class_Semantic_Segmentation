"""
@author: ardag
Code is inspired from https://github.com/bnsreenu/python_for_microscopists.git
and https://github.com/bnsreenu/python_for_image_processing_APEER.git
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model


## Load validation images from the folder
img_dir = "val_images/"
img_list = os.listdir(img_dir)
sorted_img_list = sorted(img_list)


## Load validation masks from the folder
label_dir = "val_masks/"
label_list = os.listdir(label_dir)
sorted_label_list = sorted(label_list)


## Load model
model = load_model("model.hdf5", compile=False)


## Calculate average DICE score (15 images and 15 masks in the validation set)
def dice_score(model,n_classes=3):

    # List of DICE scores
    class1_Dice_list =[]
    class2_Dice_list =[]
    class3_Dice_list =[]
    
    from tensorflow.keras.metrics import MeanIoU
    IOU_keras = MeanIoU(num_classes=n_classes) 
    model.call = tf.function(model.call)
    
    num=0
    for i in range(15):
        
        # Prediction of validation image
        val_image = cv2.imread(img_dir+sorted_img_list[i], 0)
        val_scaled = val_image.astype('float32') /255.
        val_scaled = np.expand_dims(val_scaled, axis=0)
        val_scaled = np.expand_dims(val_scaled, axis=3)
        val_pred = model.call(val_scaled)
        val_pred_argmax = np.argmax(val_pred, axis=3)
        prediction= val_pred_argmax.astype(np.uint8)
        
        
        # Mask from the validation data
        val_label = cv2.imread(label_dir+sorted_label_list[i], 0)
        val_label = np.expand_dims(val_label, axis=0)
        
        # IOU between prediction and mask image
        IOU_keras.reset_state()
        IOU_keras.update_state(val_label,prediction)
                
        # Get values from the confusion matrix
        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        
        # Calculate Dice scores for three classes
        class1_Dice = (2*values[0,0])/((2*values[0,0]) + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
        class2_Dice = (2*values[1,1])/((2*values[1,1]) + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
        class3_Dice = (2*values[2,2])/((2*values[2,2]) + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
        
        class1_Dice_list.append(class1_Dice)
        class2_Dice_list.append(class2_Dice)
        class3_Dice_list.append(class3_Dice)
        num+=1
    
    # Average DICE score for each class
    def Total_Perclass_Dice(Perclass_Dice_list):
      print('Bkg Dice,{: .3f}'.format(sum(Perclass_Dice_list)/len(Perclass_Dice_list)))
    Total_Perclass_Dice(class1_Dice_list)
    
    def Total_Perclass_Dice(Perclass_Dice_list):
      print('Al Dice ,{: .3f}'.format(sum(Perclass_Dice_list)/len(Perclass_Dice_list)))
    Total_Perclass_Dice(class2_Dice_list)
    
    def Total_Perclass_Dice(Perclass_Dice_list):
      print('Pt Dice ,{: .3f}'.format(sum(Perclass_Dice_list)/len(Perclass_Dice_list)))
    Total_Perclass_Dice(class3_Dice_list)
    return 


## Calculate average Precision score 
def precision_score(model,n_classes=3):
    
    # List of Precision scores
    class1_precision_list =[]
    class2_precision_list =[]
    class3_precision_list =[]
    
    from tensorflow.keras.metrics import MeanIoU 
    IOU_keras = MeanIoU(num_classes=n_classes) 

    num=0
    for i in range(15):
        
        # Prediction of validation image
        val_image = cv2.imread(img_dir+sorted_img_list[i], 0)
        val_scaled = val_image.astype('float32') /255.
        val_scaled = np.expand_dims(val_scaled, axis=0)
        val_scaled = np.expand_dims(val_scaled, axis=3)
        val_pred = model.predict(val_scaled)
        val_pred_argmax = np.argmax(val_pred, axis=3)
        prediction= val_pred_argmax.astype(np.uint8)
        
        
        # Mask from the validation data
        val_label = cv2.imread(label_dir+sorted_label_list[i], 0)
        val_label = np.expand_dims(val_label, axis=0)
        
        # IOU between prediction and mask image
        IOU_keras.reset_state()
        IOU_keras.update_state(val_label,prediction)
        
        
        
        # Get values from the confusion matrix
        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        
        # Calculate Precision scores for three classes
        class1_Precision = values[0,0]/(values[0,0] + values[1,0] + values[2,0])
        class2_Precision = values[1,1]/(values[1,1] + values[0,1] + values[2,1])
        class3_Precision = values[2,2]/(values[2,2] + values[0,2] + values[1,2])
    
        class1_precision_list.append(class1_Precision)
        class2_precision_list.append(class2_Precision)
        class3_precision_list.append(class3_Precision)
        num+=1
        
    # Average Precision score for each class
    def Total_Perclass_Precision(Perclass_Precision_list):
      print('Bkg precision,{: .3f}'.format(sum(Perclass_Precision_list)/len(Perclass_Precision_list)))
    Total_Perclass_Precision(class1_precision_list)
    
    
    def Total_Perclass_Precision(Perclass_Precision_list):
      print('Al precision,{: .3f}'.format(sum(Perclass_Precision_list)/len(Perclass_Precision_list)))
    Total_Perclass_Precision(class2_precision_list)
    
    def Total_Perclass_Precision(Perclass_Precision_list):
      print('Pt precision,{: .3f}'.format(sum(Perclass_Precision_list)/len(Perclass_Precision_list)))
    Total_Perclass_Precision(class3_precision_list)
    return


## Calculate average RECALL score
def recall_score(model,n_classes=3):
    
    # List of Recall scores
    class1_recall_list =[]
    class2_recall_list =[]
    class3_recall_list =[]
    
    from tensorflow.keras.metrics import MeanIoU 
    IOU_keras = MeanIoU(num_classes=n_classes) 
    
    num=0
    for i in range(15):
        
        # Prediction of validation image
        val_image = cv2.imread(img_dir+sorted_img_list[i], 0)
        val_scaled = val_image.astype('float32') /255.
        val_scaled = np.expand_dims(val_scaled, axis=0)
        val_scaled = np.expand_dims(val_scaled, axis=3)
        val_pred = model.predict(val_scaled)
        val_pred_argmax = np.argmax(val_pred, axis=3)
        prediction= val_pred_argmax.astype(np.uint8)

        
        # Mask from the validation data
        val_label = cv2.imread(label_dir+sorted_label_list[i], 0)
        val_label = np.expand_dims(val_label, axis=0)
        
        # IOU between prediction and mask image
        IOU_keras.reset_state()
        IOU_keras.update_state(val_label,prediction)
        
        
        
        # Get values from the confusion matrix
        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        
        # Calculate Recall scores for three classes
        class1_Recall = values[0,0]/(values[0,0] + values[0,1] + values[0,2])
        class2_Recall = values[1,1]/(values[1,1] + values[1,0] + values[1,2])
        class3_Recall = values[2,2]/(values[2,2] + values[2,0] + values[2,1])
    
        class1_recall_list.append(class1_Recall)
        class2_recall_list.append(class2_Recall)
        class3_recall_list.append(class3_Recall)
        num+=1
        
    # Average Recall score for each class
    def Total_Perclass_Recall(Perclass_Recall_list):
      print('Bkg recall,{: .3f}'.format(sum(Perclass_Recall_list)/len(Perclass_Recall_list)))
    Total_Perclass_Recall(class1_recall_list)
    
    def Total_Perclass_Recall(Perclass_Recall_list):
      print('Al recall,{: .3f}'.format(sum(Perclass_Recall_list)/len(Perclass_Recall_list)))
    Total_Perclass_Recall(class2_recall_list)
    
    def Total_Perclass_Recall(Perclass_Recall_list):
      print('Pt recall,{: .3f}'.format(sum(Perclass_Recall_list)/len(Perclass_Recall_list)))
    Total_Perclass_Recall(class3_recall_list)
    return


