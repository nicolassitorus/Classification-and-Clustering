# Artificial Neural Network - Clustering
# Nicolas Kornelius Sitorus - 2201825586

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import csv
import random
import math
#--------------------------------------------------------
from tensorflow.python.ops import control_flow_ops
from IPython.display import clear_output
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import normalized_mutual_info_score
from math import sqrt

def standardization(X):
    return normalize(X, axis=0)
def laplacian(A):
    S = np.sum(A, 0)
    D = np.diag(S)
    D = LA.matrix_power(D, -1)
    L = np.dot(D, A)
    return L
def normalization(V):
    return (V - min(V)) / (max(V) - min(V))

class Cosine_Similarity:
    def get_matrix(self, Data):
        X = standardization(Data)
        X = pdist(X, 'cosine')
        X = squareform(X)
        L = laplacian(X)
        Y = np.apply_along_axis(normalization, 1, L)
        return Y

class Similarity_Dataset_Iterator():
    def __init__(self, data, labels, similarity):
        self.data = data
        self.labels = labels
        self.matrix = similarity.get_matrix(data)
        self.data_size = self.matrix.shape[0]
        self.current_index = 0
    def next_batch(self, num):
        data=self.matrix.transpose()
        labels=self.labels
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
        return data_shuffle, labels_shuffle
    def whole_dataset(self):
        return (self.matrix.transpose(), self.labels)

# Fetching dataset: 
def read_iris_data(similarity):
    dataset = pd.read_csv('clustering_data.csv')
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values
   
    # Encoding categorical data
    # Label Encoding the "Gender" column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:, 9] = le.fit_transform(X[:, 9])
    X[:, 14] = le.fit_transform(X[:, 14])
    X[:, 15] = le.fit_transform(X[:, 15])
    X[:, 16] = le.fit_transform(X[:, 16])
    y=le.fit_transform(y)
    data=X
    labels=y
    
    return Similarity_Dataset_Iterator(data, labels, similarity)

def k_means_(X, n_clusters):
    kmeans_centroids,_ =  kmeans(X,n_clusters)
    kmeans_, _ = vq(X, kmeans_centroids)
    return kmeans_
def encoder(x, n_code, mode_train):    
    with tf.variable_scope("encoder"):        
        with tf.variable_scope("hidden-layer-1"):
            hidden_1 = layer(x, [n_input, n_hidden_1], [n_hidden_1], mode_train)
        with tf.variable_scope("hidden-layer-2"):
            hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2], mode_train)
        with tf.variable_scope("hidden-layer-3"):
            hidden_3 = layer(hidden_2, [n_hidden_2, n_hidden_3], [n_hidden_3], mode_train)        
        with tf.variable_scope("embedded"):
            code = layer(hidden_3, [n_hidden_3, n_code], [n_code], mode_train)
    return code

def decoder(code, n_code, mode_train):
    with tf.variable_scope("decoder"):
        with tf.variable_scope("hidden-layer-1"):
            hidden_1 = layer(code, [n_code, n_hidden_3], [n_hidden_3], mode_train)
        with tf.variable_scope("hidden-layer-2"):
            hidden_2 = layer(hidden_1, [n_hidden_3, n_hidden_2], [n_hidden_2], mode_train)
        with tf.variable_scope("hidden-layer-3"):
            hidden_3 = layer(hidden_2, [n_hidden_2, n_hidden_1], [n_hidden_1], mode_train)              
        with tf.variable_scope("reconstructed"):
            output = layer(hidden_3, [n_hidden_1, n_input], [n_input], mode_train)
    return output

def batch_norm(x, n_out, mode_train):
    beta_initialize = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_initialize = tf.constant_initializer(value=1.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_initialize)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_initialize)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(mode_train, mean_var, lambda: (ema_mean, ema_var))
    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])
def layer(input, weight_shape, bias_shape, mode_train):
    value_initialize = (1.0 / weight_shape[0] ** 0.5)
    weight_initialize = tf.random_normal_initializer(stddev = value_initialize, seed = None)
    bias_initialize = tf.constant_initializer(value=0.0, dtype=tf.float32)
    w = tf.get_variable("w", weight_shape, initializer=weight_initialize)
    b = tf.get_variable("b", bias_shape, initializer=bias_initialize)
    return tf.nn.sigmoid(batch_norm((tf.matmul(input, w) + b), weight_shape[1], mode_train))

def loss(reconstructed, x):
    with tf.variable_scope("train"):
        train_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reconstructed, x)), 1))
        return train_loss

def training(cost, learning_rate, beta1, beta2, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon=1e-08, use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

trainSet_cosine = read_iris_data(Cosine_Similarity())
n_input = trainSet_cosine.data_size #Number of input data.

# Define the number of hidden layer. 
if n_input >= 256:
    Nn = int(512)
elif n_input >= 128:
    Nn = int(256)

n_hidden_1 = int(Nn/2) #The autoencoder hidden layer 1.
n_hidden_2 = int(n_hidden_1/2) #The autoencoder hidden layer 2.
n_hidden_3 = int(n_hidden_2/2) #The autoencoder hidden layer 3.
n_hidden_4 = int(n_hidden_3/2) #The autoencoder hidden layer 4.
n_code = str(2) #The number of output dimension value.

print('Layer 1: ', n_input)
print('Layer 2: ', n_hidden_1)
print('Layer 3: ', n_hidden_2)
print('Layer 4: ', n_hidden_3)
print('Layer 5: ', int(n_code))

# Parameters
n_layers = 5 #Number of Neural Networks Layers.
beta1 = 0.9 #The decay rate 1.  
beta2 = 0.999 #The decay rate 2.
learning_rate = (beta1 / n_input) #The learning rate.
n_batch = math.ceil(n_input / n_layers) #Number of selection data in per step.
n_clusters = 3 #Number of clusters.
data_cos, labels_cos = trainSet_cosine.whole_dataset() #Allocation of data and labels

results_cos=[] #A list to keep all NMI scores.
loss_cost_cos=[] #A list to keep all training evaluations.
steps_cos=[] #A list to keep all steps.

with tf.Graph().as_default():    
    with tf.variable_scope("autoencoder_architecture"):
        x = tf.placeholder("float", [None, n_input])   
        mode_train = tf.placeholder(tf.bool)
        code = encoder(x, int(n_code), mode_train)
        reconstructed = decoder(code, int(n_code), mode_train)
        cost = loss(reconstructed, x)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_optimizer = training(cost, learning_rate, beta1, beta2, global_step)
        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
# Training cycle
# Fit training with Backpropagation using batch data.
for epoch in range(n_layers):
    miniData, _ = trainSet_cosine.next_batch(n_batch)
    _, new_cost = sess.run([train_optimizer,cost], feed_dict={x: miniData,
                                                              mode_train: True})       
    #------------------------- End of the Optimization ------------------------------
    # Save the results after per 10 epochs.    
    # Getting embedded codes and running K-Means.
    ae_codes_cos = sess.run(code, feed_dict={x: data_cos, mode_train: False})        
    idx_cos = k_means_(ae_codes_cos, n_clusters)
    ae_nmi_cos = normalized_mutual_info_score(labels_cos, idx_cos)
    ae_nmi_cos = ae_nmi_cos*100
    results_cos.append(ae_nmi_cos)    
    steps_cos.append(epoch)
    loss_cost_cos.append(new_cost)    
    print("NMI Score for AE is: {:0.2f} and new cost is: {:0.2f} in {:d} step. "
          .format(ae_nmi_cos, new_cost, epoch))
warnings.filterwarnings('ignore')
plt.figure(figsize=(12,3.5))
plt.subplot(1,2,1)
plt.ylim(-0.5, 100)
plt.plot(steps_cos, loss_cost_cos, label='Cost Trianing for Cosine Distance ', marker='o')
plt.xlabel('Number of Epochs.')
plt.ylabel('Cost')
plt.grid()
plt.title('Cost Function Trianing')
plt.legend(loc='best')
plt.subplot(1,2,2)
plt.ylim(30, 109)
plt.plot(steps_cos, results_cos, label='AE Normalized Cosine Distance ', marker='o')
plt.xlabel('Number of Epochs.')
plt.ylabel('NMI')
plt.grid()
plt.title(('NMI of AE Cosine is {:0.2f}').format(ae_nmi_cos))
plt.legend(loc='best')
plt.show()
k_means_indx_cos = k_means_(data_cos, n_clusters)
k_means_nmi_cos = (normalized_mutual_info_score(labels_cos, k_means_indx_cos))
print("KMeans on Similarity Matrix: --------- {:0.2f}".format(k_means_nmi_cos*100))
print("KMeans on Embedded Graph: ------------ {:0.2f}".format(ae_nmi_cos))

origin_label_cos = np.array(trainSet_cosine.whole_dataset()[1]).astype(int)

colors = [('g', 'x'),('r', 'x'),('b','x')]
plt.figure(figsize=(14, 5))
for num in range(3):
    plt.subplot(1,2,1)
    plt.scatter([ae_codes_cos[:,0][i] for i in range(len(idx_cos)) if idx_cos[i] == num],
                [ae_codes_cos[:,1][i] for i in range(len(idx_cos)) if idx_cos[i] == num],
                50, label=str(num+1), color = colors[num][0], marker=colors[num][1])
    plt.title(('NMI of AE on Cosine is {:0.2f}').format(ae_nmi_cos))
    plt.xlabel('Runs for the representation by AE in 2 dimensions.')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter([ae_codes_cos[:,0][i] for i in range(len(origin_label_cos)) if origin_label_cos[i] == num],
                [ae_codes_cos[:,1][i] for i in range(len(origin_label_cos)) if origin_label_cos[i] == num],
                50, label=str(num+1), color = colors[num][0], marker=colors[num][1])
    plt.title('Cosine Distance')
    plt.xlabel('Check the Original Lables on the representation by AE in 2 Dimensions.')
    plt.legend()
    
trainSet_cosine = read_iris_data(Cosine_Similarity())