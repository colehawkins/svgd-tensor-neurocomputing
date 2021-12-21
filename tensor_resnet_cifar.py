from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10,cifar100
import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' #filter out info logs and warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #filter out info logs and warnings

from utils import get_normal_prior,count_removed_params,estimate_rank_from_prior_variances,get_svgd_update,make_particles,get_tt_prior,create_posterior_loss,get_mean_log_likelihood, make_batch_dict
from multiplicativeBayesDense import BayesKerasDense
from resnet_mult_conv import reshapedBayesKerasConv


#64.68 for resnet cifar100

tensorized_model = True
normal_prior = False
#prior_mult_factor = 1e-7 was the one for the current experiments
prior_mult_factor = 1.0
mult_factor = 10


low_rank_init = 0
#low_rank_init = 2

batch_size = 128
epochs = 200

padding = 'valid'
sess = tf.InteractiveSession()
tf.set_random_seed(1)

particle_init_lr = 1e-5
num_particles = 8

data_augmentation = True

subtract_pixel_mean = True

num_svgd_steps = 5000
svgd_stepsize = 1e-5

n = 18
depth = n * 6 + 2
# Model version
version = 1

#%%

# Model name, depth and version
model_type = 'TensorResNet%dv%d' % (depth, version)

# Load the CIFAR100 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
num_classes = 100

#%%
# Input image dimensions.
input_shape = x_train.shape[1:]

num_steps_per_epoch = np.floor(x_train.shape[0]/batch_size)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%%
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch > 180:
        lr *= 5e-2
    elif epoch > 140:
        lr *= 1e-1
    elif epoch > 80:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def tensor_resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    

    
    input_shape = inputs.shape.as_list()
    input_channels = input_shape[-1]
    
    if input_channels == 3 or kernel_size==1 or input_channels<32:
        conv = Conv2D(num_filters,
              kernel_size=kernel_size,
              strides=strides,
              padding='same',
              kernel_initializer="he_normal")
        
    else:
        conv_tt_rank = 20
            
        if input_channels == 32:
                input_channel_modes = [4,8]
                     
        elif input_channels == 64:
                input_channel_modes = [4,4,4]

    
        if num_filters == 16:
            
            output_channel_modes = [4,4]
    
        elif num_filters == 32:
            
            output_channel_modes = [4,8]
                        
        elif num_filters == 64:
                
            output_channel_modes = [4,4,4]    
                
        conv = reshapedBayesKerasConv(input_channels=input_channels,
                                          input_channel_modes=input_channel_modes, 
                                          output_channels=num_filters, padding='same',
                                          output_channel_modes=output_channel_modes, 
                                          kernel_size=[kernel_size,kernel_size],
                                          tt_rank=conv_tt_rank, strides=strides,
                                          bias_initializer=0.0,use_bias=True,low_rank_init=low_rank_init)

    x = inputs

    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x


def resnet_v1(input_shape, depth, num_classes=10, tensorized_model = False):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if tensorized_model:
        res_layer = tensor_resnet_layer
    else: 
        res_layer = resnet_layer
    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = res_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = res_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = res_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = res_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = resnet_v1(input_shape=input_shape,num_classes = num_classes, depth=depth,tensorized_model = tensorized_model)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

model_params = model.count_params()
model.summary()


model_description = ''
if tensorized_model:
    if normal_prior:
        tt_prior = get_normal_prior(model)
        prior = 'normal'
        model_description = 'tensorized_model_normal_prior' 
    else:
        tt_prior = get_tt_prior(model)
        prior = 'tt'
        model_description = 'tensorized_model_tt_prior' 
else:    
    tt_prior = get_normal_prior(model)
    prior = 'normal'
    model_description = 'normal_model_normal_prior' 


    
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_low_rank.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
#checkpoint = ModelCheckpoint(filepath=filepath,
#                             monitor='val_acc',
#                             verbose=1,
#                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)




#callbacks = [checkpoint, lr_reducer, lr_scheduler]
callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(x_train)


##%%
#i = 0
#j = 0
#for layer in model.layers:
#    if hasattr(layer,'filters') and hasattr(layer,'tt_rank'):
#        print(np.prod(layer.filter.shape)+np.prod(layer.b.shape))
#        j+=1
#    
#        if layer.kernel_size[0]>1 and layer.filters>16:
#            print(j,layer)
#            i+=1
#        
#print(i)
##%%
#
#model.layers[11].kernel_size

#%%

tt_prior = prior_mult_factor*tt_prior

model.compile(loss=create_posterior_loss(tt_prior,mult_factor = mult_factor),
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy','categorical_crossentropy'])

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    steps_per_epoch = np.floor(x_train.shape[0]/batch_size),
                    epochs=epochs, verbose=1, workers=10,
                    callbacks=callbacks)
#

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#%%
#def get_new_param_number(old_param_number,)
removed_params = count_removed_params(model,sess,threshold= 1e-1)
compression_ratio = (model_params-removed_params)/model_params
print(compression_ratio)
#%%
print(model_params)
print(model_params-removed_params)
#%%

#path = './saved_models/'+model_description
#keras.models.save_model(model,path)


#%%
svgd_batch_size = 8

if svgd_batch_size == 32:
    
    num_svgd_steps = 5000 
    
elif svgd_batch_size == 8:
    num_svgd_steps = 10000    

opt = Adam(lr=lr_schedule(epochs))

model_list, priors_list = make_particles(model, num_particles, opt,x_train,y_train,svgd_batch_size,datagen,prior = prior,steps_per_epoch = 10)

#%%
historical_grad_assign_op,particle_update_ops,input_output_placeholders = get_svgd_update(model_list, priors_list, svgd_batch_size,svgd_stepsize,sess)
#%%

for i in range(0,num_svgd_steps):
   # print(i)
    temp_dict = make_batch_dict(x_train, y_train, svgd_batch_size, input_output_placeholders, num_particles)
    sess.run([historical_grad_assign_op,particle_update_ops],temp_dict)
    
    if i%1000==0:
        print(i)
        #print("Getting accuracy in step number ",ii)
#        new_plot_test(model_list,x_test,y_test,5)
#        accuracy = np.mean(get_accuracy(model_list,x_test,y_test)) 
        log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
        #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
        print( "log-lik "+str(log_lik))
        #plt.imshow(x_test[0])
        
log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
        #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
print( "log-lik "+str(log_lik))
#%%

for i in range(len(model_list)):
    print("Model ",i)
    temp_model = model_list[i]
    removed_params = count_removed_params(temp_model,sess,threshold= 1e-1)
    compression_ratio = (model_params-removed_params)/model_params
    print("Compression ratio ",compression_ratio)

    print("Initial params", model_params)
    print("Remaining params ",model_params-removed_params)


#%%

#def resnet_count_removed_params(model,sess,threshold = 1e-1):
#    total_removed_params = 0
#    j = 0
#    i = 0
#    for layer in model.layers:
#        if hasattr(layer,'filters'):
#            j+=1
#    
#            if hasattr(layer,'tt_rank'):
#                full_param = np.prod(layer.filter.shape.as_list())+np.prod(layer.b.shape.as_list())
#                prior_variances = sess.run(layer.prior_variances)
#                new_tt_rank = estimate_rank_from_prior_variances(prior_variances,threshold)
#                print("Layer ",j)
#                print("Inferred TT-rank",new_tt_rank)
#                print("Parameter number", [full_param,layer.count_params(),layer.count_params(new_tt_rank)])
#                i+=1
#                
#    print(i)
##%%
#resnet_count_removed_params(model,sess)
    

#%%

#from refactored_utils import save_model
#save_model(model,'cifar_resnet_110','367compression_with_904_accuracy')
