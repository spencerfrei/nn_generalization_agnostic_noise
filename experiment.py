#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Add
import numpy as np
from scipy import stats
from datetime import datetime
import pandas as pd
from plotnine import ggplot, geom_point, aes, scale_shape_manual, geom_vline
from plotnine import scale_x_continuous, scale_y_continuous, theme

# Set directory to output training results.
bp = '/home/ubuntu/'


# Set data parameters and hyperparameters.

# In[2]:


rcn = 0.1
margin = 0.5
lr = 1e-2
sample_size = 20000 
width = 1e3
activation = 'leaky'
init_sd = 1/(width**0.5)
# init_sd = 1/(width)
use_bias_inner = False
use_bias_outer = False
trainable_outer = False
train_batch_size = 1
epochs = 1

# Prepare filename prefix for exporting CSV of results and h5 model files
date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Use a suggestive description based on whatever hyperparam choices
# you are making.  You want to be able to filter filenames in 
# the directory to search for the given experiment for plotting later on.
experiment_description = 'onelayerfc' + str(int(width)) + '_rcn' + str(rcn)
base_filestring = date + '_' + experiment_description


# Set experimental parameters.

# In[3]:


'''Simple experiment setup -- use on first try
This will do optlin=0.4, 0.25, 0.1, with 4 different init's each'''
num_runs_per_error = 4
num_optlin = 3

'''Paper experiment setup -- parameters used to create figures in paper
This will do 25 equally spaced optlin between 0.1 & 0.4,
with 10 different weight inits per optlin.'''
# num_runs_per_error = 10
# num_optlin = 25

# Use different seeds for first layer weight initialization
seeds = 100 * (np.arange(num_runs_per_error) + 1)

# Consider optlin in an equally spaced grid from 0.4 to 0.1
errors = np.linspace(0.4, 0.1, num = num_optlin)


# First create a special layer class for using the +/- constant outer layer initialization.

# In[4]:


# Create outer layer initializer of all +/- constant.
class UniformPlusMinusConstant(keras.initializers.Initializer):
    def __init__(self, const, seed = None):
        self.const = tf.constant(const, dtype = keras.backend.floatx())
        self.seed = seed
    def __call__(self, shape, dtype=None):
        k = int(shape[0] / 2) 
        if shape[0] % 2 == 0:
            return self.const * tf.concat([tf.constant(1.0, shape = (k,1)), tf.constant(-1.0, shape = (k,1))], axis = 0)
        else:
            return self.const * tf.concat([tf.constant(1.0, shape = (k+1,1)), tf.constant(-1.0, shape = (k,1))], axis = 0)
    def get_config(self):  # To support serialization
        return {'const': self.const}


# Create a neural network model class to be used.

# In[5]:


def one_hlayer_learner(d, # Input dimension
                       m, # Width of network
                       activation = 'leaky', # Either leaky, relu, tanh, sigmoid, elu 
                       init_sd = 0.001, # Initialization SD for std normal initializer
                       use_bias_inner = False,
                       use_bias_outer = False,
                       trainable_outer = False, 
                       seed = 123): # Seed for random initialization of weights
    inputs = Input(shape = (d,), name = 'input')
    if activation == 'leaky':
        x = Dense(m, 
                  kernel_initializer = keras.initializers.RandomNormal(stddev = init_sd, seed = seed),
                  activation = None, 
                  use_bias = use_bias_inner)(inputs)
        x = LeakyReLU(alpha = 0.1)(x)
    else:
        x = Dense(m, 
                  kernel_initializer = keras.initializers.RandomNormal(stddev = init_sd, seed = seed),
                  activation = activation, 
                  use_bias = use_bias_inner)(inputs)
    output = Dense(1, 
                   kernel_initializer = UniformPlusMinusConstant(1.0/np.sqrt(m), seed),
                   activation = 'linear', 
                   use_bias = use_bias_outer,
                   trainable = trainable_outer)(x)
    return keras.Model(inputs = inputs, outputs = output)


# Create the dataset $\mathcal{D}_{\gamma_0, b}$.

# In[6]:


def prep_dataset(n_train, x_loc = 3.0, margin = 0.0, rcn = 0.0, boundary_factor = 1.75, seed = 123):
    # Set seed for data
    np.random.seed(seed)
    n_per_class = int(n_train // 2)
    # Create two clusters of data, centered at (+/- x_loc, 0)
    clust1 = np.random.normal(loc = (-1*x_loc, 0), size = (n_per_class, 2))
    clust2 = np.random.normal(loc = (x_loc, 0), size = (n_per_class, 2))
    
    # Introduce RCN and deterministic label noise.
    # Ensure that rcn fraction of |x|>b have labels flipped,
    # and that 100% of labels with |x|<b have been flipped.
    
    # Initially make rcn fraction of labels with x>0 have label 0,
    # and rcn fraction of labels with x<0 have label 1.
    flip = int(n_per_class * rcn)
    class1 = np.array([0.0] * (n_per_class - flip) + [1] * flip)
    class2 = np.array([1] * (n_per_class - flip) + [0] * flip)
    
    # Then deterministically set label to 0 if 0<x<b and label to 1 if -b<x<0
    class1[(clust1[:,0] <= boundary_factor) & (clust1[:,0] >= 0)] = 0
    class2[(clust2[:,0] <= boundary_factor) & (clust2[:,0] >= 0)] = 0
    class1[(clust1[:,0] >= -1*boundary_factor) & (clust1[:,0] <= 0)] = 1
    class2[(clust2[:,0] >= -1*boundary_factor) & (clust2[:,0] <= 0)] = 1
    
    # Note that the ordering of the data does not matter here as we will shuffle
    # the data randomly before running SGD.
    
    # Create pandas df for easy plotting later on.
    data = pd.DataFrame.from_dict({
        'x': np.concatenate([clust1[:,0], clust2[:,0]], axis = 0),
        'y': np.concatenate([clust1[:,1], clust2[:,1]], axis = 0),
        'group': np.concatenate([class1, class2])})
    data['group'] = data['group'].astype('category')
    
    # If using a margin, remove all data with |x| < margin.
    if margin > 0:
        data = data[np.abs(data.x) > margin]
    
    # Extract numpy arrays of data to be used for creating tf dataset.
    features = np.array((data.x, data.y)).T
    response = np.array(data.group)
    return data, features, response


# Derive formulas for going between the boundary factor $b$ and $\mathsf{OPT}_{\mathsf{lin}}$, and for calculating the Bayes-optimal classifier accuracy.

# In[7]:


def error_from_b(boundary_factor, margin = 0.5, rcn = 0.1):
    rcn_error = rcn * stats.norm.cdf(3 - boundary_factor)/ stats.norm.cdf(3 - margin)
    deterministic_error = (stats.norm.cdf(3-margin) - stats.norm.cdf(3 - boundary_factor) )/ stats.norm.cdf(3 - margin)
    return rcn_error + deterministic_error

def b_from_error(error, margin = 0.5, rcn = 0.1):
    Z = stats.norm.cdf(3 - margin)
    return 3 - stats.norm.ppf( (1 - error)*Z/(1 - rcn))

def best_classifier_error(boundary_factor, margin = 0.5, rcn = 0.1):
    rcn_error = rcn * stats.norm.cdf(3 - boundary_factor)/ stats.norm.cdf(3 - margin)
    return rcn_error


# Visualize the dataset to make sure everything looks alright.  Note: this uses the Python library `plotnine`, which mimics `ggplot2` behavior from R. Note: comment this out if you are running this as a Python script.

# In[8]:


to_plot = 150
margin = 0.5
x_loc = 3
rcn = 0.1
optlin = 0.4
boundary_factor = b_from_error(optlin)

train, _, _ = prep_dataset(to_plot, margin = margin, x_loc = x_loc, rcn = rcn, boundary_factor = boundary_factor)
(ggplot(train,
       aes(x = 'x', y = 'y', color = 'group', shape = 'group')) 
        + geom_point(size = 4, fill = 'none') 
        + scale_shape_manual(values = ('o', 'P')) 
        + geom_vline(xintercept = [-1*boundary_factor, 0, boundary_factor], linetype = 'dotted')
        + scale_x_continuous(breaks = np.arange(-6, 7, 1), name = '') 
        + scale_y_continuous(name = '') 
        + theme(legend_position='none')
)


# Create a function which runs the experiment for a given learner and hyperparameter configuration. 

# In[9]:


def experiment(learner, n_train, optlin, rcn, 
               lr = 1e-2, train_batch_size = 1, epochs = 1, data_seed = 123,
               filestring = 'na', sgd_shuffle_seed = 123):
    boundary_factor = b_from_error(optlin)
    
    # Note that training batch size is important as it influences learned model.
    # For online SGD, we use batch size 1 and 1 epoch.
    # Validation and test batch size don't matter since we only want to compute summary stats.
    valid_batch_size = 1000 
    n_validation = 1e4
    test_batch_size = valid_batch_size
    n_test = 1e5

    # Create training data.  Note use of data_seed ensure consistent training set across experiments.
    train, train_features, train_response = prep_dataset(
        n_train, margin = margin, x_loc = 3.0, rcn = rcn, boundary_factor = boundary_factor, seed = data_seed)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_response))
    
    # Note: be sure to use shuffle with buffer size > n_train, so that we are 
    # randomly shuffling the data.
    
    # Also of note: we are seeding the shuffle here, so every time experiment() is called,
    # we will use an ordering of the data determined by this seed.  This is important for online SGD.
    train_dataset = train_dataset.batch(train_batch_size).shuffle(buffer_size = 2 * n_train, seed = sgd_shuffle_seed)
    train_loss_metric = keras.metrics.BinaryCrossentropy(from_logits = True)
    train_acc_metric = keras.metrics.BinaryAccuracy()

    # Create validation data.  Note use of data_seed+1.
    valid, valid_features, valid_response = prep_dataset(
        n_validation, margin = margin, x_loc = 3.0, rcn = rcn, boundary_factor = boundary_factor, seed = data_seed + 1)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_features, valid_response))
    valid_dataset = valid_dataset.batch(valid_batch_size)
    valid_loss_metric = keras.metrics.BinaryCrossentropy(from_logits = True)
    valid_acc_metric = keras.metrics.BinaryAccuracy()

    # Prepare optimizer, model, and loss
    optimizer = keras.optimizers.SGD(learning_rate = lr)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits = True)
    student = learner

    # Prepare arrays for saving model outputs
    train_loss = list()
    valid_loss = list()
    train_acc = list()
    valid_acc = list()

    # Begin model training
    eval_every_steps = 100
    best_model = student
    best_valid_acc = 0.0
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = student(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, student.trainable_weights)
            optimizer.apply_gradients(zip(grads, student.trainable_weights))
            # Update MSE metric
            train_loss_metric.update_state(y_batch_train, logits)
            train_acc_metric.update_state(y_batch_train, logits)

            # Run evaluation metrics after eval_every_steps gradient updates
            if step % eval_every_steps == 0:
                train_loss_result = train_loss_metric.result()
                train_loss_metric.reset_states()
                train_loss.append(train_loss_result.numpy())

                train_acc_result = train_acc_metric.result()
                train_acc_metric.reset_states()
                train_acc.append(train_acc_result.numpy())
                # Run validation metrics every gradient update
                for x_batch_valid, y_batch_valid in valid_dataset:
                    valid_output = student(x_batch_valid, training = False)
                    valid_loss_metric.update_state(y_batch_valid, valid_output)
                    valid_acc_metric.update_state(y_batch_valid, valid_output)
                valid_loss_result = valid_loss_metric.result()
                valid_loss_metric.reset_states()
                valid_loss.append(valid_loss_result.numpy())

                valid_acc_result = valid_acc_metric.result()
                valid_acc_metric.reset_states()
                valid_acc.append(valid_acc_result.numpy())
                
                # If the validation accuracy is better, save this model as the best model.
                if valid_acc_result > best_valid_acc:
                    print('Previous best valid accuracy: %.4f, new best valid accuracy: %.4f' % (best_valid_acc, valid_acc_result))
                    best_model = student
                    best_valid_acc = valid_acc_result.numpy()
                    best_step = step
                    train_loss_at_best = train_loss_result.numpy()
                    train_acc_at_best = train_acc_result.numpy()
                    valid_loss_at_best = valid_loss_result.numpy()
                
                print("\nStep %d, train loss: %.4f, train acc: %.4f" % (step, float(train_loss_result),float(train_acc_result)))
                print("Step %d, valid loss: %.4f, valid acc: %.4f" % (step, float(valid_loss_result),float(valid_acc_result)))
                train_loss_metric.reset_states()
                train_acc_metric.reset_states()
        
    # Done with training; create fresh test set and compare performance on test set.
    # Note use of data_seed+2.
    final_test, final_test_features, final_test_response = prep_dataset(
        n_test, margin = margin, x_loc = 3.0, rcn = rcn, boundary_factor = boundary_factor, seed = data_seed + 2)
    final_test_dataset = tf.data.Dataset.from_tensor_slices((final_test_features, final_test_response))
    final_test_dataset = final_test_dataset.batch(test_batch_size)
    final_test_acc_metric = keras.metrics.BinaryAccuracy()
    final_test_loss_metric = keras.metrics.BinaryCrossentropy(from_logits = True)

    for x_batch_test, y_batch_test in final_test_dataset:
        test_output = best_model(x_batch_test, training = False)
        final_test_acc_metric.update_state(y_batch_test, test_output)
        final_test_loss_metric.update_state(y_batch_test, test_output)
    best_acc = final_test_acc_metric.result()
    best_loss = final_test_loss_metric.result()
    out_str = bp + 'weights_' + filestring + '_lr' + str(lr) + '_optlin' + str(optlin) +            '_acc%.4f' % best_acc.numpy() + '.h5' 
    best_model.save_weights(out_str)
    print('\nAt best iterate, had the following metrics.')
    print('Valid acc: %.4f, valid loss: %.4f' % (best_valid_acc, valid_loss_at_best))
    print('Best test accuracy: %.4f, best test loss: %.4f' % (best_acc,best_loss))
    return best_acc.numpy(), best_loss.numpy(), best_step, best_valid_acc, valid_loss_at_best


# Run the experiment.

# In[10]:


# Prepare lists for experiment outputs
seed_list = list()
errors_list = list()
test_acc = list()
test_loss = list()
step = list()
valid_acc = list()
valid_loss = list()

# Begin experiment
for error in errors:
    for seed in seeds:
        print('\n**********************************************')
        print('Running experiment for error %.4f, seed %d' % (error, seed))
        print('**********************************************\n')
        learner = one_hlayer_learner(2, 
                       width, 
                       activation = activation, 
                       init_sd = init_sd, 
                       use_bias_inner = use_bias_inner,
                       use_bias_outer = use_bias_outer,
                       trainable_outer = trainable_outer, 
                       seed = seed)
        filestring = base_filestring + '_seed' + str(seed)
        res = experiment(learner, sample_size, error, rcn, 
                         lr = lr, train_batch_size = train_batch_size, epochs = epochs,
                         data_seed = 123, filestring = filestring, sgd_shuffle_seed = seed + 5)
        
        # Save results into a dataframe, save after every experiment
        test_acc.append(res[0])
        test_loss.append(res[1])
        step.append(res[2])
        valid_acc.append(res[3])
        valid_loss.append(res[4])
        seed_list.append(seed)
        errors_list.append(error)
        boundary_factors = b_from_error(np.array(errors_list))
        best_poss_errors = best_classifier_error(boundary_factors, margin, rcn)
        results = pd.DataFrame({
            'optlin': np.array(errors_list),
            'best_poss': best_poss_errors, 
            'seed': seed_list,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'best_step': step
        })
        results.to_csv(bp + 'results_' + base_filestring + '_lr' + str(lr) + '.csv')

