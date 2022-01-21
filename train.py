"""

@author: ardag

Code is inspired from https://github.com/bnsreenu/python_for_microscopists.git
and https://github.com/bnsreenu/python_for_image_processing_APEER.git

Code/Library used for Focal Loss can be found at https://github.com/artemmavrin/focal-loss.git

"""



import os
import numpy as np
from matplotlib import pyplot as plt

seed=24
batch_size= 2


# Prepare data for integer labeling
def preprocess_data(img, mask, num_class):
    img = img.astype('float')/255.
    mask = mask.astype(np.int64)
    return (img,mask)

# Data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainGenerator(train_img_path, train_mask_path, num_class):
    
    img_data_gen_args = dict(rotation_range=45,horizontal_flip=True,vertical_flip=True,
                             width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1, shear_range= 0.1, fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        color_mode='grayscale',
        batch_size = batch_size,
        target_size = (512,512),
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        target_size = (512,512),
        seed = seed )
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)


train_img_path = "data_for_training_and_testing/train_images"
train_mask_path = "data_for_training_and_testing/train_masks"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=3)


val_img_path = "data_for_training_and_testing/val_images"
val_mask_path = "data_for_training_and_testing/val_masks"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=3)


x, y = train_img_gen.__next__()


for i in range(0,1):
    image = x[i,:,:,0]
    mask = y[i,:,:,0] #mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


x_val, y_val = val_img_gen.__next__()


for i in range(0,1):
    image = x_val[i,:,:,0]
    mask = y_val[i,:,:,0] #mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


num_train_imgs = len(os.listdir('data_for_training_and_testing/train_images/train'))
num_val_images = len(os.listdir('data_for_training_and_testing/val_images/val'))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]
n_classes=3


# Compile and train model

from focal_loss import SparseCategoricalFocalLoss
from unet import unet_model

def get_model():
    return unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
base_learning_rate = 0.0005
class_weights=np.array([0.2,0.3,0.5])
import tensorflow as tf
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss= SparseCategoricalFocalLoss(gamma=1,class_weight=(class_weights)),metrics = ['accuracy'])
model.summary()

initial_epochs = 100

history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs= initial_epochs,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)


# Plot accuracy and validation results

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training')
plt.plot(epochs, val_loss, 'r', label='Validation')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accurcacy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model
model.save('model.hdf5')
















