import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras.models import Sequential
from keras.layers import Activation, Flatten,Dense
import numpy as np
#import t3f
#from keras.datasets import mnist
from keras.datasets import mnist,fashion_mnist
from keras.utils import to_categorical
from keras import optimizers

from multiplicativeBayesDense import BayesKerasDense
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' #filter out info logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #filter out info logs and warnings
from utils import make_batch_dict,get_normal_prior,count_removed_params,get_svgd_update,make_particles,get_tt_prior,create_posterior_loss,get_mean_log_likelihood
import tensorflow_probability as tfp
tfd = tfp.distributions

normal_prior = False
tensorized = True
normal_scale = 10
tf_precision = tf.float32
low_rank_init = 1  
mult_factor = 100
low_rank_threshold = 1e-1
num_particles = 20
batch_size = 128
num_svgd_steps = 5000
init_epochs = 1
num_epochs = 1
temp_output_dims = [5,5,5,5]
tt_layer_size = len(temp_output_dims)
max_tt_rank = 20
invert_param = False
optimizer_step_size = 1e-3
svgd_stepsize = 1e-4
verbose = 1
rng=1
svgd_subsampling_ratio = .1
constant_subsample = False


tf.set_random_seed(rng)


#load training data

#train, test = mnist.load_data()
train, test = fashion_mnist.load_data()

x_train,y_train = train
x_test,y_test = test
x_train = x_train / 127.5 - 1.0
x_test = x_test / 127.5 - 1.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
output_shape = y_train.shape[1]

sess = tf.InteractiveSession()

if tensorized:

    tt_layer1 = BayesKerasDense(input_dims=[7, 4, 7, 4], output_dims=temp_output_dims,
                           tt_rank=max_tt_rank, activation='relu',bias_initializer=1e-3,low_rank_init=low_rank_init)
    tt_layer2 = BayesKerasDense(input_dims=[25, 25], output_dims=[5,2],
                           tt_rank=max_tt_rank, activation=None,bias_initializer=1e-3,low_rank_init=low_rank_init)

    first_model = Sequential([
            Flatten(input_shape=(28, 28)),
            tt_layer1,
            tt_layer2,
            Activation('softmax')
            ])
else: 

    first_model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(625),
            Dense(10),
            Activation('softmax')
            ])


optimizer = optimizers.Adam(lr=optimizer_step_size)
model_params = first_model.count_params()

if normal_prior:
    tt_prior = get_normal_prior(first_model,scale = normal_scale)
else:
    tt_prior = get_tt_prior(first_model)
#%%
if verbose:
    first_model.summary()
    
first_model.compile(optimizer=optimizer, loss=create_posterior_loss(tt_prior,mult_factor= mult_factor), metrics=['categorical_accuracy','categorical_crossentropy'])
first_model.fit(x_train, y_train, epochs=init_epochs, batch_size=batch_size, validation_data=(x_test, y_test),verbose = verbose)
#%%
removed_params = count_removed_params(first_model,sess,threshold = 1e-1)
compressed_size = (model_params-removed_params)/model_params
    
removed_params2 = count_removed_params(first_model,sess,threshold = 1e-2)
compressed_size2 = (model_params-removed_params2)/model_params


if normal_prior:
    prior = 'normal'
else:
    prior = 'tt'

model_list, priors_list = make_particles(first_model, num_particles, optimizer,x_train,y_train,batch_size,x_test = x_test,y_test = y_test,prior = prior,steps_per_epoch = 10)    

historical_grad_assign_op,particle_update_ops,input_output_placeholders = get_svgd_update(model_list, priors_list, batch_size,svgd_stepsize,sess,svgd_subsampling_ratio,constant_subsample)


for i in range(0,num_svgd_steps):
    
    temp_dict = make_batch_dict(x_train, y_train, batch_size, input_output_placeholders, num_particles)
    sess.run([historical_grad_assign_op,particle_update_ops],temp_dict)
    
    if i%1000==0 and i>0:
        print(i)
        #print("Getting accuracy in step number ",ii)
#        new_plot_test(model_list,x_test,y_test,5)
#        accuracy = np.mean(get_accuracy(model_list,x_test,y_test)) 
        log_lik = get_mean_log_likelihood(model_list,x_test,y_test)
        #print("Accuracy "+str(accuracy)+" log-lik "+str(log_lik))
        print( "log-lik "+str(log_lik))
        #plt.imshow(x_test[0])

##%%
#compression_list = []
#for model in model_list:
#    removed_params = count_removed_params(first_model,sess,threshold = low_rank_threshold)
#    compressed_size = (model_params-removed_params)/model_params     
#    compression_list.append(compressed_size)        

#%%
list1 = []
for temp_model in model_list:
    removed_params = count_removed_params(temp_model,sess,threshold = 1e-2)
    list1.append(model_params-removed_params2)
    print("Compressed size ",model_params-removed_params," or ",model_params-removed_params2)


#%%
plt.figure()
out = []

index = 10
plt.imshow(x_test[index,:,:])
true_out= np.argmax(y_test[index])
for model in model_list:
    pred = model.predict(x_test[index:index+1,:,:])[0]
    out.append(pred[true_out])
plt.figure()
ax = plt.axes()
#plt.xlim(.95,1)

plot = sns.kdeplot(out)
ax.set_xticks([0,1],['0','1'])
ax.set_yticks([])        
#plt.xlim(.999,1)
sns.despine(left=True)
sns.set_context("poster")
#ax.set_xticks([0,1],['0','1'])
ax.set_xticks([0,1])        

plt.show()

    
num_test_examples = x_test.shape[0]    
i = 0
x_test_example = x_test[i]
y_test_example = y_test[i]

out_list = []
for model in model_list:
    out_list.append(model.predict(x_test))