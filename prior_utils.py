import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import keras
import numpy as np

tf_precision = tf.float32

def make_custom_loss(model,local_low_rank,global_low_rank,tt_different =True):

    prior = make_prior(model,local_low_rank,global_low_rank,tt_different)
    
    def custom_loss(y_true, y_pred):
        return keras.losses.categorical_crossentropy(y_true,y_pred)-prior
    return custom_loss, prior

def make_prior(model,local_low_rank,global_low_rank,tt_different):
    
    prior= tf.zeros([1])
    
    for layer in model.layers:
        
        config = layer.get_config()
        
        if ('tt_rank' in config) and tt_different==True:
            prior+=tt_prior(layer,local_low_rank,global_low_rank)
        else:
            prior+=non_tt_prior(layer)
    return prior
    
    
def non_tt_prior(layer):
    non_tt_weight_prior= lambda x:tf.reduce_sum(tfd.Normal(0.0,50.0).log_prob(x))
    layer_prior = tf.zeros([1])
    for weight in layer.trainable_weights:
        layer_prior+=non_tt_weight_prior(weight)
        
    return layer_prior

def make_low_rank_variables(tt_rank):
    
    global_low_rank = []
    local_low_rank = []
    
    for i in range(6):    
        global_low_rank.append(tf.Variable(5.0, dtype=tf_precision))
    global_low_rank_placeholder = tf.placeholder(dtype = tf_precision,shape = global_low_rank.shape)
    local_low_rank = tf.Variable(np.arange(0,1,1/tt_rank),dtype = tf_precision)
    local_low_rank_placeholder = tf.placeholder(dtype = tf_precision,shape = local_low_rank.shape)
    low_rank_initializers = [global_low_rank.initializer, local_low_rank.initializer]
    assignments = [global_low_rank.assign_add(global_low_rank_placeholder),local_low_rank.assign_add(local_low_rank_placeholder)]
    placeholders = [global_low_rank_placeholder,local_low_rank_placeholder]
    return local_low_rank, global_low_rank, low_rank_initializers, assignments,placeholders

def tt_prior(layer,pre_local_low_rank,pre_global_low_rank):
    local_low_rank = tf.nn.softmax(pre_local_low_rank)/tf.reduce_sum(tf.nn.softmax(pre_local_low_rank))
    global_low_rank = tf.exp(pre_global_low_rank)
    cores = layer.matrix.tt_cores
    bias = layer.b
    
    config = layer.get_config()
    tt_rank = config['tt_rank']
    dirichlet_parameter = tf.cast(np.repeat(np.sqrt(tt_rank)/tt_rank,tt_rank),tf_precision)
    
    a_lambda = 10.*tf.ones([1],dtype=tf_precision)
    b_lambda = .1*tf.ones([1],dtype=tf_precision)
    
    local_low_rank_prior = tfd.Dirichlet(dirichlet_parameter).log_prob(local_low_rank)
    
    global_low_rank_prior = tfd.Gamma(a_lambda,b_lambda).log_prob(global_low_rank)
    
    
    #local low rank
        
    def tt_core_prior(left_core,right_core):
        left_core_shape = left_core.shape.as_list()
        right_core_shape = right_core.shape.as_list()
        prior_normal = tfd.MultivariateNormalDiag(loc = tf.zeros(shape = [tt_rank],dtype = tf_precision),scale_diag = local_low_rank*global_low_rank)
        core_prior = tf.zeros([1],dtype = tf_precision)
        for j in range(0,left_core_shape[1]):   
            for k in range(0,left_core_shape[2]):
                core_prior += tf.reduce_sum(prior_normal.log_prob(left_core[:,j,k,:]))
        for j in range(0,right_core_shape[1]):   
            for k in range(0,right_core_shape[2]):
                core_prior += tf.reduce_sum(prior_normal.log_prob(tf.transpose(right_core[:,j,:])))
        return core_prior
    
    core_prior_list = []
    
    for i in range(0,len(cores)-1):
        core_prior_list.append(tt_core_prior(cores[i],cores[i+1]))    

    bias_prior = tf.reduce_sum(tfd.Normal(0.0,50.0).log_prob(bias))
    
    return tf.reduce_sum(core_prior_list)+bias_prior+local_low_rank_prior+global_low_rank_prior


#%%
    
#temp_layer = model.layers[1]
#temp_weight = model.weights[1]



#%%
    
