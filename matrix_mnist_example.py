import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras.models import Sequential
from keras.layers import Activation, Flatten
import numpy as np
#import t3f
import keras
#from keras import backend
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from multiplicativeBayesDense import BayesKerasDense
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #filter out info logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #filter out info logs and warnings
from utils import make_particle_updates,make_ada_update,svgd_kernel,tf_model_unpack,apply_low_rank_update,apply_update, np_svgd_kernel,get_particles,get_placeholders_and_gradients, gradient_unpack, make_batch_dict, get_accuracy, ada_update
from utils import get_true_prior_variances,show_variance_and_weights,count_removed_params,estimate_rank_from_prior_variances,get_svgd_update,make_particles,get_tt_prior,create_posterior_loss,get_mean_log_likelihood, make_batch_dict
import tensorflow_probability as tfp
tfd = tfp.distributions

tf_precision = tf.float32

num_particles = 100
batch_size = 128
loss_const = 1/10000000
new_const = 1000*loss_const


num_svgd_steps = 50000
init_epochs = 100
num_epochs = 1
temp_output_dims = [5,5,5,5]
tt_layer_size = len(temp_output_dims)
max_tt_rank = 20
invert_param = False
optimizer_step_size = 1e-3
svgd_stepsize = 1e-4
verbose = 1


save_dir = os.path.join(os.getcwd(), 'mnist_saved_models')
model_name = 'mnist_%s_low_rank.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

callbacks = [checkpoint]

train, test = mnist.load_data()

x_train,y_train = train

x_test,y_test = test

x_train = x_train / 127.5 - 1.0
x_test = x_test / 127.5 - 1.0


sess = tf.InteractiveSession()
tf.set_random_seed(1)


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

output_shape = y_train.shape[1]


#from BayesKerasDense import BayesKerasDense

tt_layer1 = BayesKerasDense(input_dims=[7, 4, 7, 4], output_dims=temp_output_dims,
                           tt_rank=max_tt_rank, activation='relu',bias_initializer=1e-3)
tt_layer2 = BayesKerasDense(input_dims=[25, 25], output_dims=[5,2],
                           tt_rank=max_tt_rank, activation=None,bias_initializer=1e-3)
#%%
from keras.layers import Dense
first_model = Sequential([
        Flatten(input_shape=(28, 28)),
        tt_layer1,
        tt_layer2,
#        tt_layer2,
        Activation('softmax')
        ])
optimizer = optimizers.Adam(lr=optimizer_step_size)

model_params = first_model.count_params()

tt_prior = get_tt_prior(first_model)

first_model.compile(optimizer=optimizer, loss=create_posterior_loss(tt_prior,const = new_const), metrics=['categorical_accuracy','categorical_crossentropy'])
first_model.summary()
#%%



first_model.fit(x_train, y_train, epochs=init_epochs, batch_size=batch_size, validation_data=(x_test, y_test),verbose = verbose)
#%%

#sess.run(tf.global_variables_initializer())
layer = tt_layer2
prior_variances = layer.prior_variances
cores = layer.matrix.tt_cores

for i in range(len(prior_variances)+1):
    show_variance_and_weights(prior_variances,i,cores,sess)

#true_prior_variances = get_true_prior_variances(prior_variances)

    
#%%

cores = tt_layer1.matrix.tt_cores
#%%

i = 3
show_variance_and_weights(prior_variances,i,cores,sess)
#%%
plt.imshow(b, cmap='hot', interpolation='nearest')
plt.show()



#%%
removed_params = count_removed_params(first_model,sess)
compressed_size = (model_params-removed_params)/model_params
#sess.run(tt_layer.prior_variances)
#%%

model_list, priors_list = make_particles(first_model, num_particles, optimizer,x_train,y_train,batch_size,x_test = x_test,y_test = y_test)
#%%
historical_grad_assign_op,particle_update_ops,input_output_placeholders = get_svgd_update(model_list, priors_list, batch_size,svgd_stepsize,sess)

#%%

for i in range(0,num_svgd_steps):
   # print(i)
    temp_dict = make_batch_dict(x_train, y_train, batch_size, input_output_placeholders, num_particles)
    sess.run([historical_grad_assign_op,particle_update_ops],temp_dict)
    
    if i%10000==0:
        print(i)
        #print("Getting accuracy in step number ",ii)
#        new_plot_test(model_list,x_test,y_test,5)
#        accuracy = np.mean(get_accuracy(model_list,x_test,y_test)) 
        log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
        #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
        print( "log-lik "+str(log_lik))
        #plt.imshow(x_test[0])

#%%
compression_list = []
for model in model_list:
    removed_params = count_removed_params(first_model,sess)
    compressed_size = (model_params-removed_params)/model_params     
    compression_list.append(compressed_size)        
    
#%%
#out = []
#temp = 7
#for model in model_list:
#    pred = model.predict(x_test[0:1,:,:])[0]
#    out.append(pred[temp])
#
#ax = plt.axes()
#plt.xlim(.95,1)
#
#plot = sns.kdeplot(out)
#ax.set_xticks([0,1],['0','1'])
#ax.set_yticks([])        
#plt.xlim(.999,1)
#sns.despine(left=True)
#sns.set_context("poster")
##ax.set_xticks([0,1],['0','1'])
#
#
#ax.set_xticks([0,1])        
#
#tensor_core = 0
#sess.run(model_list[0].layers[1].weights[tensor_core][:,0,0,:])

#tt_layer = first_model.layers[1]
##first_core = tt_layer.trainable_weights[0]

#%%
    
num_test_examples = x_test.shape[0]    
i = 0
x_test_example = x_test[i]
y_test_example = y_test[i]
#%%
out_list = []
for model in model_list:
    out_list.append(model.predict(x_test))

#%%
#num_outputs = 10
#raw_final_outputs = []
#for i in range(num_test_examples):
#    temp = 
#    for j in range(0,num_outputs):
        
    

#for i in range(num_test_examples):
#
#
#def get_classification_accuracy(x_test,y_test,model_list):