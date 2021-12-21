#most code here is from the t3f package
from itertools import count
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Activation
from keras.utils import conv_utils
import t3f
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class reshapedBayesKerasConv(Layer):
    _counter = count(0)

    def __init__(self, input_channels, input_channel_modes, output_channels, 
                 output_channel_modes, kernel_size,tt_rank, strides=1,
               padding='valid', data_format='channels_last', 
               dilation_rate=1, activation=None, use_bias=False, 
               kernel_initializer='cvpr_2018',bias_initializer=0.0, low_rank_init = None, base = 1.2, **kwargs):
        """Creates a TT based Conv Keras layer.
        
        Args:
          input_dims: an array, tensor shape of the matrix row index
          ouput_dims: an array, tensor shape of the matrix column index
          tt_rank: a number or an array, desired tt-rank of the TT-Matrix
          activation: [None] string or None, specifies the activation function.
          use_bias: bool, whether to use bias
          kernel_initializer: string specifying initializer for the TT-Matrix.
              Possible values are 'glorot', 'he', and 'lecun'.
          bias_initializer: a number, initialization value of the bias
        
        Returns:
          Layer object corresponding to multiplication by a TT-Matrix
              followed by addition of a bias and applying
              an elementwise activation
        
        Raises:
            ValueError if the provided activation or kernel_initializer is
            unknown.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = output_channels
        self.input_channel_modes= input_channel_modes
        self.output_channel_modes= output_channel_modes
        self.kernel_size = kernel_size
        self.counter = next(self._counter)
        self.filter_shape = kernel_size+[input_channels, output_channels]
        self.tt_shape = [np.prod(kernel_size)]+input_channel_modes+output_channel_modes
        self.tt_rank = tt_rank
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.padding = padding
        self.strides = [strides,strides]
        self.data_format = data_format
        self.low_rank_init = low_rank_init
        self.base = base
        
        super(reshapedBayesKerasConv, self).__init__(**kwargs)


    def build(self, input_shape):
#        self.output_shape = [input_shape[1]-self.kernel_size[0]+1,input_shape[2]-self.kernel_size[1]+1,self.output_channels]

        if self.kernel_initializer =='cvpr_2018':        
            N = (np.prod(self.kernel_size)*self.output_channels*self.input_channels)/self.tt_rank
            self.target_variance = np.sqrt(2/N)

        elif self.kernel_initializer =='he_normal':        
            N = (np.prod(self.kernel_size)*self.input_channels)/self.tt_rank
            self.target_variance = 2/N
                        
        d = len(self.tt_shape)
        R = self.tt_rank
        self.sigma = np.sqrt( np.power( R*self.target_variance  , 1/d )/R )
        
        initializer = t3f.tensor_with_random_cores(shape=self.tt_shape,tt_rank=self.tt_rank,mean=0,stddev=self.sigma)

        name = 'tt_conv_{}'.format(self.counter)
        with tf.variable_scope(name):
          self.tt_filter = t3f.get_variable('filter', initializer=initializer)
          self.filter = tf.reshape(t3f.full(self.tt_filter),self.filter_shape)
          self.b = None
          if self.use_bias:
            b_init = tf.constant_initializer(self.bias_initializer)
            self.b = tf.get_variable('bias', shape=self.output_channels,
                                     initializer=b_init)
        self.trainable_weights = list(self.tt_filter.tt_cores)
        if self.b is not None:
          self.trainable_weights.append(self.b)
    
#    def round_matrix(self,max_rank):.
#        rounded_matrix = t3f.round(self.matrix, max_rank)
#        if self.use_bias:
#            self.trainable_weights=list(rounded_matrix.tt_cores)+[self.b]
#        else:
#            self.trainable_weights=rounded_matrix.tt_cores
#        self.matrix = rounded_matrix
#        self.tt_rank = max_rank
    
    def call(self, x):
        res = K.conv2d(x, self.filter,padding=self.padding,strides=self.strides)
        if self.use_bias:
          res += self.b
        if self.activation is not None:
          res = Activation(self.activation)(res)
        return res

    
    def create_layer_prior(self,bias_variance = 100,tf_precision=tf.float32,invert=False):
        #self.build(self.input_shape)
        prior = tf.zeros([1])          
        layer_1_local_prior_variables = []
        self.processed_layer_1_local_prior_variables = []
        self.prior_variances = []          

            
    
        for i in range(len(self.tt_shape)):
            
            if self.low_rank_init is None:                
                a = -self.tt_rank
                b = self.tt_rank
                step = (b-a)/self.tt_rank
                random_init = -np.arange(a,b,step)

            else:
                a = -self.tt_rank-self.low_rank_init
                b = self.tt_rank-self.low_rank_init
                step = (b-a)/self.tt_rank
                random_init = -np.arange(a,b,step)
                
#            flat_init = np.zeros(self.tt_rank)
            layer_1_local_prior_variables.append(tf.Variable(random_init,dtype=tf_precision))
            self.processed_layer_1_local_prior_variables.append(tf.exp(layer_1_local_prior_variables[i]))
            self.prior_variances.append(self.processed_layer_1_local_prior_variables[i])

        self.trainable_weights = self.trainable_weights+layer_1_local_prior_variables
        
        
        a_lambda = 1.*tf.ones([1],dtype=tf_precision)
        b_lambda = 5.*tf.ones([1],dtype=tf_precision)
            
        gamma_dist = tfd.Gamma(a_lambda,b_lambda)
        
        local_prior = tf.reduce_sum(gamma_dist.log_prob(self.processed_layer_1_local_prior_variables))
        core_priors = []
        def new_tt_core_prior(core,left_prior_variance,right_prior_variance, last_core = False,first_core = False):
           # left_core_shape = left_core.shape.as_list()
           # right_core_shape = right_core.shape.as_list()
    
            if first_core:                
                diag = tf.pow(right_prior_variance,2)
                prior_normal = tfd.MultivariateNormalDiag(loc = tf.zeros(shape = [self.tt_rank],dtype = tf_precision),scale_diag = diag)
                #                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(left_core)))
                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [0,1,2])))
            
            elif last_core:                
#                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(left_core)))
                diag = tf.pow(left_prior_variance,2)
                prior_normal = tfd.MultivariateNormalDiag(loc = tf.zeros(shape = [self.tt_rank],dtype = tf_precision),scale_diag = diag)
                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [2,1,0])))
            else:
#                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(left_core,perm = [3,0,1,2])))
                outer = tf.einsum('i,j->ij', left_prior_variance, right_prior_variance)
                prior_normal = tfd.Normal(loc = tf.zeros(shape = [self.tt_rank,self.tt_rank],dtype = tf_precision),scale = outer)

                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [1,0,2])))


        bias_prior = tf.reduce_sum(tfd.Normal(loc = 0,scale = bias_variance).log_prob(self.trainable_weights[i]))

        for i in range(len(self.tt_shape)-1):
           # print(i)
           # print(self.trainable_weights[i])

            if i==0:
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],[],self.prior_variances[i],first_core = True,last_core=False))    
            elif i==(len(self.tt_shape)-1):
                #last core
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],self.prior_variances[i-1],[],first_core = False,last_core = True))    
            else:
                #middle core
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],self.prior_variances[i-1],self.prior_variances[i]))
        
        prior = tf.reduce_sum(core_priors+[local_prior,bias_prior])
        
        return prior


#    def count_params(self):
#        
#        num_params = 0
#        num_params+= np.prod(self.b.get_shape().as_list())
#        
#        for core in self.tt_filter.tt_cores:   
#            num_params += np.prod(core.get_shape().as_list())
#        
#        return num_params
    
    def count_params(self,new_tt_rank=None):
        
        if new_tt_rank is None:
            new_tt_rank= (len(self.tt_shape)+1)*[self.tt_rank]
            new_tt_rank[0] = 1
            new_tt_rank[-1] = 1
            
        num_params = 0
        num_params+= np.prod(self.b.get_shape().as_list())
        
        for i in range(len(self.tt_filter.tt_cores)):   
            core = self.tt_filter.tt_cores[i]
            core_shape = core.get_shape().as_list()
            core_shape[0] = new_tt_rank[i]
            core_shape[-1] = new_tt_rank[i+1]
            
            num_params += np.prod(core_shape)        

        return num_params


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1)
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)

          
#    def compute_output_shape(self, input_shape):
#        if self.padding == 'valid':
#            assert(self.strides ==1)
#            return (input_shape[0],input_shape[1]-self.kernel_size[0]+1,input_shape[2]-self.kernel_size[1]+1,self.output_channels)
#        elif self.padding == 'same':
#            if self.strides == 1:
#                return (input_shape[0],input_shape[1],input_shape[2],self.output_channels)
#            elif self.strides == 2:
#                return (input_shape[0],int(input_shape[1]/2),int(input_shape[2]/2),self.output_channels)
    def get_config(self):
        config = {
            'input_channels':self.input_channels,
            'output_channels': self.output_channels,
            'input_channel_modes': self.input_channel_modes,
            'output_channel_modes': self.output_channel_modes,
            'kernel_size':self.kernel_size,
            'tt_rank': self.tt_rank,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'padding': self.padding,
            'strides': self.strides[0],
            'low_rank_init': self.low_rank_init
        }
    
        return config