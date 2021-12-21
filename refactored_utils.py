import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import t3f
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import backend
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import BatchNormalization, Conv2D,Input,AveragePooling2D
from keras.regularizers import l2
from multBayesConv import reshapedBayesKerasConv
from keras import Model

tf_precision = tf.float32



tf_precision = tf.float32


from keras.models import model_from_json

def save_model(model,experiment,model_type):
#with open('./saved_models/'+temp+'_weights.h5','w') as outfile:
    model.save_weights('./saved_models/'+experiment+'/'+model_type+'_weights.h5')
        #    json.dumps(weights,outfile)
    with open('./saved_models/'+experiment+'/'+model_type+'_model_structure.json','w') as outfile:    
        outfile.write(model.to_json())
    

def load_model(experiment,model_type,custom_objects = None):
    with open('./saved_models/'+experiment+'/'+model_type+'_model_structure.json','r') as infile:    
    #    data = json.load(infile)
        loaded_model_json = infile.read()
        new_model = model_from_json(loaded_model_json,custom_objects=custom_objects)
    new_model.load_weights('./saved_models/'+experiment+'/'+model_type+'_weights.h5')
    return new_model


def count_removed_params(model,sess,threshold = 1e-1):
    removed_params = 0
    for layer in model.layers[1:]:
        if hasattr(layer,'prior_variances'):
            prior_variances = sess.run(layer.prior_variances)
            new_tt_rank = estimate_rank_from_prior_variances(prior_variances,threshold)
            print(new_tt_rank)
            print("I am layer ",layer)
            print("with parameter number ",layer.count_params(new_tt_rank))
            removed_params += layer.count_params()-layer.count_params(new_tt_rank)
    return removed_params

def estimate_rank_from_prior_variances(prior_variances,threshold= 1e-1):

    tt_rank = len(prior_variances)*[0]
    
    for i in range(len(prior_variances)):
        for x in prior_variances[i]:
            if x>threshold:
                tt_rank[i]+=1
        tt_rank[i] = np.max([1,tt_rank[i]])
    return [1]+tt_rank+[1]

def get_svgd_update(model_list, priors_list, batch_size,svgd_stepsize):
    model_gradients_list, input_output_placeholders,unpacked_model_gradients_list,unpacked_model_weights_list = get_particle_gradients_and_weights(model_list, priors_list, batch_size)

#    with tf.device('/gpu:1'):
    unpacked_gradients = tf.transpose(tf.stack(unpacked_model_gradients_list,axis = 1))
    unpacked_model_weights = tf.transpose(tf.stack(unpacked_model_weights_list,axis = 1))
        
    neg_unpacked_gradients = -unpacked_gradients
    
    historical_grad = tf.Variable(tf.zeros(shape = unpacked_gradients.get_shape().as_list()),dtype=tf_precision)
    
    kernel_matrix, kernel_derivative = svgd_kernel(unpacked_model_weights)
    
    update = tf.matmul(kernel_matrix,neg_unpacked_gradients)+kernel_derivative
    
    adj_update,historical_grad_assign_op = make_ada_update(update,historical_grad)
    
    adj_update = svgd_stepsize*adj_update
    
    particle_update_ops = make_particle_updates(adj_update,model_list)

    return historical_grad_assign_op,particle_update_ops,input_output_placeholders,historical_grad

def get_particle_gradients_and_weights(model_list, priors_list, batch_size):

    model_gradients_list =[]
    input_output_placeholders = []
    unpacked_model_gradients_list = []
    unpacked_model_weights_list = []

    num_models = len(model_list)
    for i in range(num_models):
                    
            
#        with tf.device(string):
        
        temp = get_placeholders_and_gradients(model_list[i],batch_size,model_list[i].output_shape[1])
        input_output_placeholders = input_output_placeholders+temp[0]
        
        model_gradients_list.append(temp[1])
        unpacked_model_gradients_list.append(gradient_unpack(temp[1]))
        unpacked_model_weights_list.append(tf_model_unpack(model_list[i]))
        
    return model_gradients_list, input_output_placeholders,unpacked_model_gradients_list,unpacked_model_weights_list

import time

def make_particles(first_model, num_particles, optimizer,x_train,y_train,batch_size,datagen=None,x_test = None, y_test = None, verbose = 0, prior = 'tt',steps_per_epoch = None):
    
    if steps_per_epoch is None:
        steps_per_epoch = np.floor(x_train.shape[0]/batch_size)
    
    model_list = []
    priors_list = []

    if prior=='tt':
        get_prior = get_tt_prior
    elif prior == 'normal':
        get_prior = get_normal_prior
    
    t = time.time()

    update_ops = []

#    weights_for_update = first_model.get_weights()

    for i in range(0,num_particles):
        print(time.time()-t)        
        t = time.time()
   #     gc.collect()
#        with tf.device(string):
        print("Training model ",str(i))
 #       tf.set_random_seed(i)
    
    #check if low-rank gets copied over as well
        temp_model = keras.models.clone_model(first_model)
        
        priors_list.append(get_prior(temp_model))
        
        update_ops.append([tf.assign(to_var, from_var) for to_var, from_var in
              zip(temp_model.trainable_weights, first_model.trainable_weights)])
    
    #    optimizer = optimizers.Adam(lr=1e-4)
        #temp_model.set_weights(weights_for_update)
        temp_model.compile(optimizer=optimizer, loss=create_posterior_loss(priors_list[i]), metrics=['categorical_accuracy','categorical_crossentropy'])
        model_list.append(temp_model)     
    
    return model_list, priors_list, update_ops 

def fit_particles(model_list,first_model,x_train,y_train,batch_size,datagen=None,x_test = None, y_test = None, verbose = 0,steps_per_epoch = 1):

    for i in range(len(model_list)):
#        temp_model = model_list[i]

        if datagen is not None:
            model_list[i].fit_generator(datagen,
                            steps_per_epoch= steps_per_epoch,
                            epochs=1,
                            workers=2)
        else:        
                model_list[i].fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test),verbose=verbose)
        


def get_true_prior_variances(prior_variances):
    true = []
    for i in range(len(prior_variances)+1):
        if i==0:
            true.append(tf.pow(prior_variances[i],2))
        elif i==(len(prior_variances)):
            true.append(tf.pow(prior_variances[i-1],2))
        else:
            true.append(tf.einsum('i,j->ij', prior_variances[i-1], prior_variances[i]))
    return true

def show_variance_and_weights(prior_variances,i,cores,sess,rank_threshold = 1e-1):
            
    core = cores[i]    
    #a = np.abs(sess.run(core[:,0,0,:]))
    #b = np.abs(sess.run(right_core[:,0,0,:]))
    if i==0:
        variance = np.expand_dims(sess.run(tf.pow(prior_variances[i],2)),axis = 0)
    elif i==(len(prior_variances)):
        variance = np.expand_dims(sess.run(tf.pow(prior_variances[i-1],2)),axis = 0)
    else:
        variance= sess.run(tf.einsum('i,j->ij', prior_variances[i-1], prior_variances[i]))

    print(variance.shape)
    a = np.sum(np.abs(sess.run(core)),axis = (1,2))
    
    plt.imshow(a, cmap='hot', interpolation='None',vmin = 0,vmax = 10)
    plt.title("Summed elements core"+str(i))
    plt.show()
    plt.imshow(variance, cmap='hot', interpolation='nearest',vmin = 0,vmax = rank_threshold)
    plt.title("Variance core "+str(i))
    plt.show()

def get_tt_prior(model, invert_param = False,scale = 100):
    temp_prior = tf.squeeze(tf.zeros([1]))

    dist = tfd.Normal(loc = 0.0, scale =scale)

    for layer in model.layers:
        if hasattr(layer,'tt_rank'):
            temp_prior += layer.create_layer_prior(bias_variance = scale,invert=invert_param)
        else:
            for weight in layer.trainable_weights:
                temp_prior+=tf.reduce_sum(dist.log_prob(weight))
    return temp_prior

def get_normal_prior(model, scale = 10):
    dist = tfd.Normal(loc = 0.0, scale =scale)
    temp_prior = tf.squeeze(tf.zeros([1]))

    for layer in model.layers:
        for weight in layer.trainable_weights:
            temp_prior += tf.reduce_sum(dist.log_prob(weight))    
    return temp_prior

def create_posterior_loss(tt_prior, const = 1/10000000, mult_factor = 1.0):
    def custom_loss(y_true,y_pred):
        return keras.losses.categorical_crossentropy(y_true,y_pred)-mult_factor*const*tt_prior
    return custom_loss


def get_mean_log_likelihood(model_list,x_test,y_test):

    temp = []
    for model in model_list:
        temp.append(model.evaluate(x_test,y_test,verbose=0)[2])        
        
    return np.mean(temp)
#def get_mean_keras_log_likelihood(model_list,x_test,y_test,sess):
#      
#    log_likelihood_list = []        
#    for model in model_list:
#        #to_reduce = -keras.losses.categorical_crossentropy(model.predict(x_test),y_test) 
#        log_likelihood_list.append(log_loss(y_test,model.predict(x_test)))
#        
#    mean_log_likelihood = -np.mean(log_likelihood_list)
#    return mean_log_likelihood

def build_first_model(x_train, y_train, batch_size,x_test,y_test):
    first_model = Sequential()
    first_model.add(Flatten(input_shape=(28, 28)))
    tt_layer = t3f.nn.KerasDense(input_dims=[7, 4, 7, 4], output_dims=[5, 5, 5, 5],
                                 tt_rank=4, activation='relu',
                                 bias_initializer=1e-3)
    
    first_model.add(tt_layer)
    first_model.add(Dense(10))
    first_model.add(Activation('softmax'))
    optimizer = optimizers.Adam(lr=1e-2)
    first_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    first_model.fit(x_train, y_train, epochs=2, batch_size=batch_size, validation_data=(x_test, y_test),verbose = 0)
    return first_model

def plot_test(model_list,x_test,y_test, index):
    test_out = [] 
    test_point = np.reshape(x_test[index],[1,28,28])
    true_out = y_test[index]
    for i in range(len(model_list)):
        test_out.append(model_list[i].predict(test_point))
    
    test_out = np.squeeze(np.stack(test_out,axis = 1))
    print("True out is: ",true_out)
    for i in range(0,10):
        sns.kdeplot(test_out[:,i]).set_title(str(i))  
        plt.show()

def new_plot_test(model_list,x_test,y_test, point_index):
    num_models = len(model_list)

    test_point = np.reshape(x_test[point_index],[1,28,28])
    true_out = y_test[point_index]
    test_out = [] 
    for i in range(num_models):
        test_out.append(model_list[i].predict(test_point))
    
    summed_out = np.sum(np.squeeze(np.stack(test_out,axis = 0)),axis = 0)
    print("True out is: ",true_out)
    plt.scatter(range(0,10),summed_out/num_models)
    plt.show()
    
def get_accuracy(model_list,x_test,y_test):
    num_models = len(model_list)

    test_out = [] 
    for i in range(num_models):
        test_out.append(model_list[i].evaluate(x_test,y_test,verbose=0)[1])
        
    return test_out

def apply_low_rank_update(low_rank_assignments_list,low_rank_placeholders_list,update,sess):
    
    for i in range(len(low_rank_assignments_list)):
        temp_dict = {low_rank_placeholders_list[i][0]:update[i,0], low_rank_placeholders_list[i][1]:update[i,1:]}
        sess.run(low_rank_assignments_list[i], feed_dict = temp_dict)



def make_particle_updates(adj_update,model_list):
    
    update_list = []

    for i in range(len(model_list)):

        start_index = 0
        end_index = 0
        
        for weight in model_list[i].trainable_weights:
        
            end_index += np.prod(weight.shape)
            delta = tf.reshape(adj_update[i,start_index:end_index],weight.shape)
            update_list.append(weight.assign_add(delta))
            start_index = end_index
    return update_list

def ada_update(update,historical_grad,n_iter):
    update = np.clip(update,-1e8,1e8)
    alpha = .9
    eps = 1e-6
    step_size = 1e-3
    if n_iter==0:#np.linalg.norm(historical_grad)<1e-8:
        new_historical_grad = np.square(update)#np.clip(np.square(update),-1e8,1e8)
    else:
        new_historical_grad = alpha*historical_grad+(1-alpha)*np.square(update)

    adj_grad = np.divide(update,np.sqrt(eps+new_historical_grad))        
    return step_size*adj_grad, historical_grad


def make_ada_update(param_update,historical_grad):
    alpha = 0.9
    eps = 1e-6
    new_historical_grad = tf.cond(tf.linalg.norm(
            historical_grad)<eps/10,lambda:historical_grad+ tf.pow(param_update,2),lambda: alpha * historical_grad + (1 - alpha) * tf.pow(param_update,2))
    
    adj_grad = tf.divide(param_update , tf.sqrt(eps+new_historical_grad))
    
    update_op = historical_grad.assign(new_historical_grad)

    return adj_grad,update_op


def non_tensor_ada_update(update,historical_grad,n_iter):
    update = np.clip(update,-1e8,1e8)
    alpha = .9
    eps = 1e-6
    step_size = 1e-4
    if n_iter==0:#np.linalg.norm(historical_grad)<1e-8:
        new_historical_grad = np.square(update)#np.clip(np.square(update),-1e8,1e8)
    else:
        new_historical_grad = alpha*historical_grad+(1-alpha)*np.square(update)

    adj_grad = np.divide(update,np.sqrt(eps+new_historical_grad))        
    return step_size*adj_grad, historical_grad

def np_svgd_kernel(X0):
    XY = np.matmul(X0, np.transpose(X0))
    X2_ = np.sum(np.square(X0), axis=1)

    x2 = np.reshape( X2_, [np.shape(X0)[0],1] )
    
    X2e = np.repeat(x2, np.shape(X0)[0],axis = 1)
    H = np.subtract(np.add(X2e, np.transpose(X2e) ), 2 * XY)

    V = np.reshape(H, [-1,1]) 


    h = np.median(V)
    h = np.sqrt(0.5 * h / np.log( np.shape(X0)[0] + 1.0))

    # compute the rbf kernel
    Kxy = np.exp(-H / h ** 2 / 2.0)

    dxkxy = -np.matmul(Kxy, X0)
    sumkxy = np.expand_dims(np.sum(Kxy, axis=1), 1) 
    dxkxy = np.add(dxkxy, np.multiply(X0, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy,h)

def apply_update(update,model):

    start_index = 0
    for layer in model.layers:
        layer_weights = []
        for weight in layer.trainable_weights:
            weight_shape = weight.get_shape().as_list()
            end_index = start_index+np.prod(weight_shape)
            layer_weights.append(np.reshape(update[start_index:end_index],weight_shape))    
            #print(start_index,end_index)
            start_index = end_index
    
        layer.set_weights(layer_weights)

def get_particles(model_list,low_rank_parameters_list):
    weight_vectors = []
    for model in model_list:
        weight_vectors.append(model_unpack(model))
    
    particles = np.stack(weight_vectors,axis = 0)
    temp_list = []
    for i in range(len(low_rank_parameters_list)):
        temp = low_rank_parameters_list[i]
        temp[1] = np.reshape(temp[1],[1,])
        temp = np.concatenate([temp[1],temp[0]],axis = 0)
        temp_list.append(temp)
        
    temp_stack = np.stack(temp_list,axis = 0)
        
    particles = np.concatenate([particles,temp_stack],axis = 1) 
    
    return particles

def get_non_low_rank_particles(model_list):
    weight_vectors = []
    for model in model_list:
        weight_vectors.append(model_unpack(model))
    
    particles = np.stack(weight_vectors,axis = 0)

    return particles

def make_batch_dict(x_train, y_train, batch_size,input_output_placeholders, num_particles):
    batch_indices = np.random.choice(x_train.shape[0],batch_size)
    training_batch_input = x_train[batch_indices]
    true_output = y_train[batch_indices]
    temp_dict = make_dict(training_batch_input,true_output, input_output_placeholders, num_particles)
    return temp_dict

def get_placeholders_and_gradients(model,batch_size,output_shape):
    input_placeholder = model.input
    output_tensor = model.output
    output_placeholder = tf.placeholder(tf.float32,shape = [batch_size,output_shape])
    loss = model.loss_functions[0](output_placeholder,output_tensor)
    gradients = backend.gradients(loss, model.trainable_weights)

    return [input_placeholder,output_placeholder],gradients

def make_dict(training_batch_input,true_output, input_output_placeholders, num_particles):
    temp = [training_batch_input,true_output]
    new = []
    for _ in range(0,num_particles):
        new+= temp 
    
    out = zip(input_output_placeholders,new)        

    return dict(out)

def tf_model_unpack(model):
    #def unpack(model):
    
    weights = model.trainable_weights
    
    temp_list = []
    
    for x in weights:
        temp_list.append(tf.reshape(x,[-1]))
    
    stacked = tf.concat(temp_list,axis = 0)    
    return stacked


def model_unpack(model):
    #def unpack(model):
    
    temp_model = model
    weights = temp_model.get_weights()
    
    temp_list = []
    
    for x in weights:
        temp_list.append(np.reshape(x,[-1]))
    
    stacked = np.concatenate(temp_list,axis = 0)    
    return stacked

def gradient_unpack(gradients):
    #def unpack(model):
        
    temp_list = []
    
    for x in gradients:
        temp_list.append(tf.reshape(x,[-1]))
    
    stacked = tf.concat(temp_list,axis = 0)    
    return stacked

def svgd_kernel(X0):
    XY = tf.matmul(X0, tf.transpose(X0))
    X2_ = tf.reduce_sum(tf.square(X0), axis=1)

    x2 = tf.reshape( X2_, shape=( tf.shape(X0)[0], 1) )
    
    X2e = tf.tile(x2, [1, tf.shape(X0)[0] ] )
    H = tf.subtract(tf.add(X2e, tf.transpose(X2e) ), 2 * XY)

    V = tf.reshape(H, [-1,1]) 

    # median distance
    def get_median(v):
        v = tf.reshape(v, [-1])
        m = v.get_shape()[0]//2
        return tf.nn.top_k(v, m).values[m-1]
    h = get_median(V)
    h = tf.sqrt(0.5 * h / tf.log( tf.cast( tf.shape(X0)[0] , tf.float32) + 1.0))

    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.0)

    dxkxy = -tf.matmul(Kxy, X0)
    sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1) 
    dxkxy = tf.add(dxkxy, tf.multiply(X0, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy)

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
    
    if input_channels == 3:# or kernel_size==1:
        conv = Conv2D(num_filters,
              kernel_size=kernel_size,
              strides=strides,
              padding='same',
              kernel_initializer="he_normal",
              kernel_regularizer=l2(1e-4))
        
    else:
        conv_tt_rank = 20
        if input_channels == 16:
         #       conv_tt_rank = 10
                input_channel_modes = [4,4]
            
        elif input_channels == 32:
                input_channel_modes = [4,8]
                     
        elif input_channels == 64:
                input_channel_modes = [4,4,4]

    
        if num_filters == 16:
         #   conv_tt_rank = 10

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
                                          bias_initializer=0.0,use_bias=True)

    x = inputs

    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x

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
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

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