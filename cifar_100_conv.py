import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

from multiplicativeBayesDense import BayesKerasDense
from resnet_mult_conv import reshapedBayesKerasConv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES'] = '0' #filter out info logs and warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #filter out info logs and warnings
import tensorflow_probability as tfp
tfd = tfp.distributions

from utils import get_cifar_100_conv_tt_prior,get_normal_prior,show_variance_and_weights,count_removed_params,get_svgd_update,make_particles,get_tt_prior,create_posterior_loss,get_mean_log_likelihood, make_batch_dict

tensorized_model = True
normal_prior = False

low_rank_init = 0
mult_factor = 2
padding = 'valid'
sess = tf.InteractiveSession()
tf.set_random_seed(5)
num_particles = 20
learning_rate = 1e-3
batch_size = 128
num_classes = 100
verbose = 1
tf_precision = tf.float32
svgd_stepsize = 1e-5
num_svgd_steps = 5000
#%%
#init_epochs_per_learning_rate = 30
#learning_rate_steps = 4

init_epochs_per_learning_rate = 40
learning_rate_steps = 3


data_augmentation = True

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

output_shape = y_train.shape[1]

model = Sequential()

kernel_size = [3,3]

fc_tt_rank = 30
conv_tt_rank = 30


fc1 = BayesKerasDense([4,8,4,8],[4,8,4,4],tt_rank=fc_tt_rank,low_rank_init=low_rank_init)
fc2 = BayesKerasDense([32,16],[10,10],tt_rank=fc_tt_rank,low_rank_init=low_rank_init)

tt_conv1 = reshapedBayesKerasConv(128,[4*8*4], 256,[8*4*8], [3,3],tt_rank=conv_tt_rank, strides=1,bias_initializer=0.1,use_bias=True,low_rank_init=low_rank_init,padding=padding)
tt_conv2 = reshapedBayesKerasConv(256, [8*4*8], 256,[8*4*8], [3,3],tt_rank=conv_tt_rank, strides=1,bias_initializer=0.1,use_bias=True,low_rank_init=low_rank_init,padding=padding)
tt_conv3 = reshapedBayesKerasConv(256,[8*4*8], 256,[8*4*8], [3,3],tt_rank=conv_tt_rank, strides=1,bias_initializer=0.1,use_bias=True,low_rank_init=low_rank_init,padding=padding)


if tensorized_model:
    model.add(Conv2D(128, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.3))
    model.add(tt_conv1)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
    
    model.add(tt_conv2)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.3))
    model.add(tt_conv3)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5)))
    model.add(Flatten())
    #model.add(Dense(256))
    model.add(fc1)
    model.add(Activation('relu'))
    model.add(fc2)
    model.add(Activation('softmax'))

else:
    model.add(Conv2D(128, (3, 3), padding='same',
                     input_shape=x_train.shape[1:],strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), padding=padding,strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
    
    model.add(Conv2D(256, (3, 3), padding=padding,strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), padding=padding,strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

optimizer = keras.optimizers.Adam(lr=learning_rate)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(x_train)
model_params = model.count_params()

if tensorized_model:
    if normal_prior:
        tt_prior = get_normal_prior(model)
        prior = 'normal'
    else:
        #tt_prior = get_cifar_100_conv_tt_prior(model)
        tt_prior= get_tt_prior(model)
        prior = 'tt'

else:    
    tt_prior = get_normal_prior(model)
    prior = 'normal'
#%%

model.summary()
#%%
for i in range(0,learning_rate_steps):
   # print("With tt prior, starting step ",i)
    optimizer = keras.optimizers.Adam(lr=learning_rate*(10**(-i)))
    
    model.compile(optimizer=optimizer, loss=create_posterior_loss(tt_prior,mult_factor=mult_factor), metrics=['categorical_accuracy','categorical_crossentropy'])

    model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch= np.floor(x_train.shape[0]/batch_size),
                    epochs=init_epochs_per_learning_rate,
                    validation_data=(x_test, y_test),
                    workers=4)

    
#print(sess.run(fc1.prior_variances))
#print(sess.run(fc2.prior_variances))
#print(sess.run(tt_conv1.prior_variances))
#print(sess.run(tt_conv2.prior_variances))
#print(sess.run(tt_conv3.prior_variances))
#%%
    
removed_params = count_removed_params(model,sess,threshold = 1e-1)
compressed_size = (model_params-removed_params)/model_params
print(compressed_size)
 
#%%    
svgd_batch_size = 32
model_list, priors_list = make_particles(model, num_particles, optimizer,x_train,y_train,svgd_batch_size,datagen,prior = prior,steps_per_epoch = 10)

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
sizes = []

for temp_model in model_list:
    removed_params = count_removed_params(temp_model,sess,threshold = 1e-1)
#    compressed_size = (model_params-removed_params)/model_params
    sizes.append(model_params-removed_params)
    
print("Mean ",np.mean(sizes))
print("Sum ",np.sum(sizes))
 