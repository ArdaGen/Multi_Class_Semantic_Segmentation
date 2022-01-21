"""

@author: ardag

Code is inspired from https://github.com/bnsreenu/python_for_microscopists.git
and https://github.com/bnsreenu/python_for_image_processing_APEER.git

"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("model.hdf5", compile=False)




from tensorflow.keras.metrics import MeanIoU 
n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes) 


# Load Validation Image
val_image = cv2.imread("data_for_training_and_testing/val_images/val/stem_images_al_10007.tifpatch_01.tif", 0)
val_scaled = val_image.astype('float32') /255.
val_scaled = np.expand_dims(val_scaled, axis=0)
val_scaled = np.expand_dims(val_scaled, axis=3)
val_pred = model.predict(val_scaled)
val_pred_argmax = np.argmax(val_pred, axis=3)
prediction= val_pred_argmax.astype(np.uint8)

# Load Validation Mask
val_label = cv2.imread("data_for_training_and_testing/val_masks/val/set1_0012.tifpatch_01.tif", 0)
val_label = np.expand_dims(val_label, axis=0)


IOU_keras.reset_state()
IOU_keras.update_state(val_label,prediction)
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)


#To calculate I0U for three class
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
#print(values)

class1_IoU = values[0,0]/(values[0,0] + values[1,0] + values[2,0] + values[0,1]+ values[0,2])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[2,1] + values[0,1]+ values[1,2])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])


print("IoU for class 1 is: ", class1_IoU)
print("IoU for class 2 is: ", class2_IoU)
print("IoU for class 3 is: ", class3_IoU)

#To calculate DSC for three class
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_Dice = (2*values[0,0])/((2*values[0,0]) + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
class2_Dice = (2*values[1,1])/((2*values[1,1]) + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_Dice = (2*values[2,2])/((2*values[2,2]) + values[2,0] + values[2,1] + values[0,2]+ values[1,2])


print("Dice for class 1 is: ", class1_Dice)
print("Dice for class 2 is: ", class2_Dice)
print("Dice for class 3 is: ", class3_Dice)

 
# To calcualte Recall for three class : Recall

class1_Recall = values[0,0]/(values[0,0] + values[0,1] + values[0,2])
class2_Recall = values[1,1]/(values[1,1] + values[1,0] + values[1,2])
class3_Recall = values[2,2]/(values[2,2] + values[2,0] + values[2,1])

print("Recall for class 1 is: ", class1_Recall)
print("Recall for class 2 is: ", class2_Recall)
print("Recall for class 3 is: ", class3_Recall)

# To calculate Precision for three class: Precision

class1_Precision = values[0,0]/(values[0,0] + values[1,0] + values[2,0])
class2_Precision = values[1,1]/(values[1,1] + values[0,1] + values[2,1])
class3_Precision = values[2,2]/(values[2,2] + values[0,2] + values[1,2])

print("Precision for class 1 is: ", class1_Precision )
print("Precision for class 2 is: ", class2_Precision )
print("Precision for class 3 is: ", class3_Precision )
