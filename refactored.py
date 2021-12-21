from __future__ import print_function
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
from multiplicativeBayesDense import BayesKerasDense
from multBayesConv import reshapedBayesKerasConv


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #filter out info logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #filter out info logs and warnings

from refactored_utils import save_model,load_model,resnet_v1,lr_schedule,fit_particles,get_normal_prior,count_removed_params,estimate_rank_from_prior_variances,get_svgd_update,make_particles,get_tt_prior,create_posterior_loss,get_mean_log_likelihood, make_batch_dict
tensorized_model = False
normal_prior = True
experiment = 'cifar_resnet'

if tensorized_model:
    if normal_prior:
        model_type = 'tensorized_normal_prior'
    else:
        model_type = 'tensorized_tt_prior'
else:
    model_type = 'full_normal_prior'



svgd_batch_size = 32
num_particles = 20

low_rank_init = 0
mult_factor = 10


batch_size = 128
epochs = 200

padding = 'valid'
sess = tf.InteractiveSession()
tf.set_random_seed(1)


particle_init_lr = 1e-5


data_augmentation = True
num_classes = 10
subtract_pixel_mean = True

num_svgd_steps = 5000
svgd_stepsize = 1e-5


n = 8
depth = n * 6 + 2
# Model version
version = 1

#%%


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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


model = resnet_v1(input_shape=input_shape, depth=depth,tensorized_model = tensorized_model)

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
    else:
        tt_prior = get_tt_prior(model)
        prior = 'tt'
else:    
    tt_prior = get_normal_prior(model)
    prior = 'normal'

model.compile(loss=create_posterior_loss(tt_prior,mult_factor = mult_factor),
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy','categorical_crossentropy'])
    
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

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
#%%
# Fit the model on the batches generated by datagen.flow().
datgen = datagen.flow(x_train, y_train, batch_size=batch_size)
import time

model.fit_generator(datgen,
                    validation_data=(x_test, y_test),
                    steps_per_epoch = num_steps_per_epoch,
                    epochs=epochs, verbose=1, workers=4,
                    callbacks=callbacks)

#%%

print(experiment,model_type)
#%%



save_model(model,experiment,model_type+'_init')
#%%

#from  multBayesConv import reshapedBayesKerasConv
#custom_objects={'BayesKerasDense': BayesKerasDense,'reshapedBayesKerasConv':reshapedBayesKerasConv}
#newest_load = load_model(experiment,'tensorized_tt_prior_init',custom_objects)

##%%
#from keras.models import model_from_json
##%%
#with open('./saved_models/'+experiment+'/'+model_type+'_model_structure.json','r') as infile:    
##    data = json.load(infile)
#    loaded_model_json = infile.read()
#    new_model = model_from_json(loaded_model_json,custom_objects=custom_objects)
##%%
#new_model.compile(loss='categorical_crossentropy',
#              optimizer=Adam(lr=lr_schedule(0)),
#              metrics=['accuracy'])
#
#new_tt_prior = get_tt_prior(new_model)
##%%
#new_model.compile(loss=create_posterior_loss(tt_prior,mult_factor = mult_factor),
#              optimizer=Adam(lr=lr_schedule(0)),
#              metrics=['accuracy','categorical_crossentropy'])

#new_model.load_weights('./saved_models/'+experiment+'/'+model_type+'_weights.h5')



#%%
#model_list, priors_list,update_ops = make_particles(model, num_particles, opt,x_train,y_train,svgd_batch_size,datagen,prior = prior,steps_per_epoch = 10)
t = time.time()
opt = Adam(lr=lr_schedule(200))
model_list, priors_list, update_ops= make_particles(model, num_particles, opt,x_train,y_train,svgd_batch_size,datagen,prior = prior,steps_per_epoch = 10)

print("Time to get the update ops is ",time.time()-t)



#%%
#sess.graph.finalize()

t = time.time()
for i in range(num_particles):
    
    print("Setting weights for model ",i)
        
    print(time.time()-t)
    t = time.time()
    model.fit_generator(datgen,
                        steps_per_epoch = 20,
                        epochs=1, verbose=0, workers=2)
   # sess.run(update_ops[i])
    sess.run(update_ops[i])    
    
#%%
    
#for i in range(num_particles):
#    save_model(model_list[i],experiment,model_type+str(i))
#    
    
#%%
#t = time.time()
#for i in range(num_particles):
#    
#    print("Setting weights for model ",i)
#        
#    print(time.time()-t)
#    t = time.time()
#    model.fit_generator(datgen,
#                        steps_per_epoch = 10,
#                        epochs=1, verbose=0, workers=1)
#   # sess.run(update_ops[i])
#    model_list[i].set_weights(model.get_weights())
##%%
#
#test = model_list[0]
#test.get_weights()[0][0,0,0]
##%%
#
#sess.run(update_ops[0])
##%%
#len(update_ops)
##%%
#
#model.get_weights()[0][0,0,0]
##%% 
#test.metrics_names
##%%
#test.evaluate(x_test,y_test)    
##sess.run(update_ops)
##%%
#model.evaluate(x_test,y_test)
#
##things to try
#run just run stuff from update opds, might be faster

#%%
#fit_particles(model_list,model,x_train,y_train,svgd_batch_size,datgen,steps_per_epoch=1)
#%%
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

scores_2 = model_list[3].evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores_2[0])
print('Test accuracy:', scores_2[1])
#removed_params = count_removed_params(model,sess)
#compression_ratio = (model_params-removed_params)/model_params


#%%

if svgd_batch_size == 32:
    print("keeping normal svgd batch")    
    num_svgd_steps = 7500 
    
elif svgd_batch_size == 8:
    num_svgd_steps = 30000    

#historical_grad_assign_op,particle_update_ops,input_output_placeholders = get_svgd_update(model_list, priors_list, svgd_batch_size,svgd_stepsize,sess)
historical_grad_assign_op,particle_update_ops,input_output_placeholders,historical_grad = get_svgd_update(model_list, priors_list, svgd_batch_size,svgd_stepsize)
historical_grad_init = tf.variables_initializer([historical_grad])
sess.run(historical_grad_init)
#%%
print("Starting svgd")
num_svgd_steps = 10000
for i in range(0,num_svgd_steps):
   # print(i)
    temp_dict = make_batch_dict(x_train, y_train, svgd_batch_size, input_output_placeholders, num_particles)
    sess.run([historical_grad_assign_op,particle_update_ops],temp_dict)
    if i%1000==0:
        print(i)
        if i%5000==0 and i>0:
            print(i)
            #print("Getting accuracy in step number ",ii)
    #        new_plot_test(model_list,x_test,y_test,5)
    #        accuracy = np.mean(get_accuracy(model_list,x_test,y_test)) 
            log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
            #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
            print( "log-lik "+str(log_lik))
            #plt.imshow(x_test[0])
        
        
        #%%

for a in model_list:
    print(a.evaluate(x_test,y_test))
#%%
log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
        #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
print( "log-lik "+str(log_lik))