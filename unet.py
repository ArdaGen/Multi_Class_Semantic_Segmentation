
"""

@author: ardag

Code is inspired from https://github.com/bnsreenu/python_for_microscopists.git
and https://github.com/bnsreenu/python_for_image_processing_APEER.git

"""


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Conv2DTranspose,Activation
from tensorflow.keras.models import Model

def unet_model(n_classes=2, IMG_HEIGHT=512, IMG_WIDTH=512, IMG_CHANNELS=1):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = inputs
    c1 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(s)
    c1 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c1)
    p1 = AveragePooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c2)
    p2 = AveragePooling2D((2, 2))(c2)
     
    c3 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(p2)
    c3 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c3)
    p3 = AveragePooling2D((2, 2))(c3)
     
    c4 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(p3)
    c4 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(c4)
    p4 = AveragePooling2D(pool_size=(2, 2))(c4)
      
    c5 = Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(p4)
    c5 = Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(c5)
    p5 = AveragePooling2D(pool_size=(2, 2))(c5)

    c6 = Conv2D(1024, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(p5)
    c6 = Conv2D(1024, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(c6)
   

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2),  padding='same' )(c6)
    u6 = concatenate([u6, c5], axis=3)
    c7 = Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(u6)
    c7 = Conv2D(512, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(c7)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2),  padding='same')(c7)
    u7 = concatenate([u7, c4],axis=3)
    c8 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(u7)
    c8 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same' )(c8)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2),  padding='same')(c8)
    u8 = concatenate([u8, c3],axis=3)
    c9 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(u8)
    c9 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c9)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2),  padding='same')(c9)
    u9 = concatenate([u9, c2], axis=3)
    c10 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(u9)
    c10 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c10)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2),  padding='same')(c10)
    u10 = concatenate([u10, c1], axis=3)
    c11 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(u10)
    c11 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_normal',  padding='same')(c11)


    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c11)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model