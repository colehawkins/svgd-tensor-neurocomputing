#most code here is from the t3f package
from itertools import count
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Activation,Dense
import t3f
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class BayesKerasDense(Layer):
    _counter = count(0)

    def __init__(self, input_dims, output_dims, tt_rank=2,
               activation=None, use_bias=True, kernel_initializer='glorot',
               bias_initializer=0.1,low_rank_init=None, **kwargs):
        """Creates a TT-Matrix based Dense Keras layer.
        
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
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.counter = next(self._counter)
        self.tt_shape = [input_dims, output_dims]
        self.output_dim = np.prod(output_dims)
        self.tt_rank = tt_rank
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.low_rank_init = low_rank_init
        super(BayesKerasDense, self).__init__(**kwargs)


    def build(self, input_shape):
        if self.kernel_initializer == 'glorot':
          initializer = t3f.glorot_initializer(self.tt_shape,
                                               tt_rank=self.tt_rank)
        elif self.kernel_initializer == 'he':
          initializer = t3f.he_initializer(self.tt_shape,
                                           tt_rank=self.tt_rank)
        elif self.kernel_initializer == 'lecun':
          initializer = t3f.lecun_initializer(self.tt_shape,
                                              tt_rank=self.tt_rank)
        else:
          raise ValueError('Unknown kernel_initializer "%s", only "glorot",'
                           '"he", and "lecun"  are supported'
                           % self.kernel_initializer)
        name = 'tt_dense_{}'.format(self.counter)
        with tf.variable_scope(name):
          self.matrix = t3f.get_variable('matrix', initializer=initializer)
          self.b = None
          if self.use_bias:
            b_init = tf.constant_initializer(self.bias_initializer)
            self.b = tf.get_variable('bias', shape=self.output_dim,
                                     initializer=b_init)
        self.trainable_weights = list(self.matrix.tt_cores)
        if self.b is not None:
          self.trainable_weights.append(self.b)
    
    def round_matrix(self,max_rank):
        rounded_matrix = t3f.round(self.matrix, max_rank)
        if self.use_bias:
            self.trainable_weights=list(rounded_matrix.tt_cores)+[self.b]
        else:
            self.trainable_weights=rounded_matrix.tt_cores
        self.matrix = rounded_matrix
        self.tt_rank = max_rank
    
    def call(self, x):
        res = t3f.matmul(x, self.matrix)
        if self.use_bias:
          res += self.b
        if self.activation is not None:
          res = Activation(self.activation)(res)
        return res

    
    def create_layer_prior(self,bias_variance = 1000,tf_precision=tf.float32,invert=False):
        #self.build(self.input_shape)
        prior = tf.zeros([1])          
        layer_1_local_prior_variables = []
        self.processed_layer_1_local_prior_variables = []
        self.prior_variances = []          

            
    
        for i in range(len(self.input_dims)-1):
            
            if self.low_rank_init is None:                
                a = -self.tt_rank
                b = self.tt_rank
                step = (b-a)/(self.tt_rank)
                random_init = -np.arange(a,b,step)

            else:
                a = -self.tt_rank-self.low_rank_init
                b = self.tt_rank-self.low_rank_init
                step = (b-a)/(self.tt_rank)
                random_init = -np.arange(a,b,step)
            
            
            
            if invert:
                random_init = -random_init                

            layer_1_local_prior_variables.append(tf.Variable(random_init,dtype=tf_precision))
            self.processed_layer_1_local_prior_variables.append(tf.exp(.5*layer_1_local_prior_variables[i]))    
#            self.processed_layer_1_local_prior_variables.append(tf.square(layer_1_local_prior_variables[i]+1e-10))    
            
            if invert:
                self.prior_variances.append(1/self.processed_layer_1_local_prior_variables[i])
            else:
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
                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [1,2,0,3])))
            
            elif last_core:                
#                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(left_core)))
                diag = tf.pow(left_prior_variance,2)
                prior_normal = tfd.MultivariateNormalDiag(loc = tf.zeros(shape = [self.tt_rank],dtype = tf_precision),scale_diag = diag)
                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [1,2,0,3])))
            else:
#                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(left_core,perm = [3,0,1,2])))
                outer = tf.einsum('i,j->ij', left_prior_variance, right_prior_variance)
                prior_normal = tfd.Normal(loc = tf.zeros(shape = [self.tt_rank,self.tt_rank],dtype = tf_precision),scale = outer)
                return tf.reduce_sum(prior_normal.log_prob(tf.transpose(core,perm = [1,2,0,3])))


        bias_prior = tf.reduce_sum(tfd.Normal(loc = 0,scale = bias_variance).log_prob(self.b))

        for i in range(len(self.tt_shape[0])-1):
           # print(i)
           # print(self.trainable_weights[i])

            if i==0:
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],[],self.prior_variances[i],first_core = True,last_core=False))    
            elif i==(len(self.tt_shape[0])-1):
                #last core
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],self.prior_variances[i-1],[],first_core = False,last_core = True))    
            else:
                #middle core
                core_priors.append(new_tt_core_prior(self.trainable_weights[i],self.prior_variances[i-1],self.prior_variances[i]))
        
        prior = tf.reduce_sum(core_priors+[local_prior,bias_prior])
        
        return prior        
    
    def count_params(self,new_tt_rank=None):
        
        if new_tt_rank is None:
            new_tt_rank= (len(self.tt_shape[0])+1)*[self.tt_rank]
            new_tt_rank[0] = 1
            new_tt_rank[-1] = 1
            
        num_params = 0
        num_params+= np.prod(self.b.get_shape().as_list())
        
        for i in range(len(self.matrix.tt_cores)):   
            core = self.matrix.tt_cores[i]
            core_shape = core.get_shape().as_list()
            core_shape[0] = new_tt_rank[i]
            core_shape[-1] = new_tt_rank[i+1]
            
            num_params += np.prod(core_shape)        
        
        return num_params
          
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'input_dims':self.tt_shape[0],
            'output_dims':self.tt_shape[1],
            'tt_rank': self.tt_rank,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'low_rank_init': self.low_rank_init
        }
    
        return config