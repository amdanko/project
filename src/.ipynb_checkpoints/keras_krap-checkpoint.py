import os
import time
import numpy as np
import keras
from keras import backend as K
from rob import rgb_images, pad_images
from keras.layers.noise import GaussianNoise
from keras.layers import concatenate
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.layers import UpSampling2D, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import softmax 

def dice_coef(y_true, y_pred,smooth=1):
    ''' Metric used for CNN training'''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)




def dice_coef_loss(y_true, y_pred):
    ''' Loss function'''
    return -dice_coef(y_true, y_pred)



def vanilla_unet(patch_size = (None,None),learning_rate = 1e-5,dropout=0):
    ''' Get U-Net model with gaussian noise and dropout'''
    
    inputs = Input((patch_size[0], patch_size[1],3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4) if dropout > 0 else pool4

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6) if dropout > 0 else up6
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7) if dropout > 0 else up7
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8) if dropout > 0 else up8
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9) if dropout > 0 else up9
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    opt = Adam(lr=learning_rate, decay = 1e-6)
    model.compile(optimizer= opt,loss=dice_coef_loss, metrics=[dice_coef])

    return model



def rob_unet(patch_size = (None,None),learning_rate = 1e-5,dropout=0):
    ''' Get U-Net model with gaussian noise and dropout'''

    gaussian_noise_std = 0.025
    dropout = 0.25

    inputs = Input((patch_size[0], patch_size[1],3))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_with_noise)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    opt = Adam(lr=learning_rate, decay = 1e-6)
    model.compile(optimizer= opt,loss=dice_coef_loss, metrics=[dice_coef])

    return model





def dil_unet(patch_size = (None,None),learning_rate = 1e-5,dropout=0):
    ''' Get U-Net model with gaussian noise and dropout'''


    dropout = 0.25

    inputs = Input((patch_size[0], patch_size[1],3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=1)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=1)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=1)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',dilation_rate=1)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',dilation_rate=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=1)(pool4)
#    DROP
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=2)(conv5)
#    DROP
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',dilation_rate=4)(conv5)
#    DROP
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    opt = Adam(lr=learning_rate, decay = 1e-6)
    model.compile(optimizer= opt,loss=dice_coef_loss, metrics=[dice_coef])

    return model





def train_seg(train_patch, train_label, val_patch, val_label, img_gen,
              model, model_path, model_name="cool_name", epochs=100,
              early_stop=1, monitor='val_dice_coef',patience=15):

    
    earlyStopping = EarlyStopping(monitor=monitor, patience=patience,
                                               verbose=1, mode='max')

    checkpoint = ModelCheckpoint(os.path.join(model_path,model_name+'.hdf5'), mode = 'max', monitor=monitor,
                                 verbose=1, save_best_only=True, save_weights_only = True)

    if (early_stop): 
        callback= [checkpoint,earlyStopping]
    else:
        callback=checkpoint

    training_time = time.time()
    
        # if i am augumenting it on the fly then can i just make..however many steps i want... ? 
    HERE_WE_GO_FOLKS = model.fit_generator(img_gen,
                     epochs=epochs,
                     steps_per_epoch= train_patch.shape[0] / 32,
                     validation_data= (val_patch,val_label),
                     verbose=1,
                     callbacks=callback)

    training_time= time.time() - training_time
    
    monitor_history =  np.array(HERE_WE_GO_FOLKS.history[monitor])
    np.save(os.path.join(model_path,model_name+"_"+monitor+'.npy'),monitor_history )

    score= max(HERE_WE_GO_FOLKS.history[monitor])
    score=float(score)
    run_dict= {'score':score,
               'time': training_time,
               'system': os.uname()[1]}
    
    return run_dict



def seg_img(model, test_img,mean,std, period=16):
    
    img_rgb = rgb_images(test_img,period)
    img_rgb = 100.0*(img_rgb-img_rgb.min(axis=(1,2),keepdims = True))/(img_rgb.max(axis=(1,2),keepdims = True)
                                                                       -img_rgb.min(axis=(1,2),keepdims = True))
    img_rgb-=mean
    img_rgb/=std
    img_rgb,nw,nz  = pad_images(img_rgb)

    print "Segmenting test image..."
    predict = model.predict(img_rgb)
    predict = predict[:,:-nw,:-nz,0]

    return predict