
"""
Created on Mon Dec 12 15:13:16 2016

@author: sbanijam
"""

import numpy as np
import pickle
from keras.models import Model , Sequential
from keras.layers import Dense, Input, Reshape, Lambda
from keras import backend as K
import tensorflow as tf
from keras import objectives , optimizers, callbacks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from keras.callbacks import Callback
import os
from keras import initializers

np.random.seed(1)  # for reproducibility


aa = str(datetime.datetime.now())
aa = aa[0:10] + '_'+aa[11:13] + '.' +aa[14:16]

os.mkdir(aa)

enc_dim = 300
decod_dim = 300
trans_dim = 100
latent_dim = 2
action_dimension = 2

net1_size = 100
net2_size = 10

batch_size = 100
image_size = 40

input_size = image_size**2
epsilon_std  = 1
num_epochs = 2

kernel_stdv= 0.02



x = Input(shape=(input_size,))
x_1 = Input(shape = (input_size,))


encoder_p_1 = Dense(enc_dim, kernel_initializer=initializers.random_normal(stddev=kernel_stdv), activation='relu')
encoder_p_2 = Dense(enc_dim, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation='relu')

z_mean_p = Dense(latent_dim,kernel_initializer=initializers.random_normal(stddev=kernel_stdv))
z_log_var_p = Dense(latent_dim,kernel_initializer=initializers.random_normal(stddev=kernel_stdv))

_encoder_p_1 = encoder_p_1(x)
_encoder_p_2 = encoder_p_2(_encoder_p_1)

_z_mean_p = z_mean_p(_encoder_p_2)
_z_log_var_p = z_log_var_p(_encoder_p_2)


encoder_q_1 = encoder_p_1(x_1)
encoder_q_2 = encoder_p_2(encoder_q_1)

z_mean_q = z_mean_p(encoder_q_2)
z_log_var_q = z_log_var_p(encoder_q_2)




def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var/ 2) * epsilon 
    
z_hat_t1 = Lambda(sampling, output_shape=(latent_dim,))([z_mean_q, z_log_var_q])


net1_1 = Dense(net1_size, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation='relu')(x)
#net1_2 = Dense(net1_size, activation='relu')(net1_1)

net2_1 = Dense(net2_size, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation='relu')(z_hat_t1)
#net2_2 = Dense(net2_size, activation='relu')(net2_1)

def Concat_net(args):
    
    net1_2,net2_2 = args
    
    return tf.concat([net1_2,net2_2],1)


net_3 = Lambda(Concat_net)([net1_1,net2_1])
net_out = Dense(net1_size,kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation= 'relu')(net_3)

z_mean_bar = Dense(latent_dim)(net_out)
z_log_var_bar = Dense(latent_dim)(net_out)

z_bar_t = Lambda(sampling, output_shape=(latent_dim,))([z_mean_bar, z_log_var_bar])


u_t = Input(shape = (action_dimension,))

def Concat_net_2(args):
    
    net1_2,net2_2 = args
    
    return tf.concat([net1_2,net2_2],1)
    
ZU = Lambda(Concat_net_2)([z_bar_t,u_t])

trans_1 = Dense(trans_dim, activation = 'relu')(z_bar_t)
trans_2 = Dense(trans_dim, activation = 'relu')(trans_1)

v_t = Dense(latent_dim)(trans_2)
r_t = Dense(latent_dim)(trans_2)
B_t = Dense(latent_dim*action_dimension)(trans_2)
o_t = Dense(latent_dim)(trans_2)


def Transition_model(args):
    
    v_t, r_t, B_t, o_t, z_hat_t1,u_t = args
    M = tf.matmul(tf.expand_dims(v_t,-1),tf.expand_dims(r_t,1)) + tf.diag([1.]*latent_dim)
    B = tf.reshape(B_t,(batch_size,latent_dim,action_dimension))
    z = tf.expand_dims(z_hat_t1,-1)
    u = tf.expand_dims(u_t,-1)
    o = tf.expand_dims(o_t,-1) 
    tot_mat = z - tf.matmul(B,u) - o
    return tf.squeeze(tf.matmul(M,tot_mat) ,[-1])
   

    


z_t = Lambda(Transition_model)([ v_t, r_t, B_t, o_t, z_hat_t1,u_t])


decoder_1 = Dense(decod_dim, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation = 'relu')(z_hat_t1)
decoder_2 = Dense(decod_dim, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation = 'relu')(decoder_1)
Output = Dense(input_size, kernel_initializer=initializers.random_normal(stddev=kernel_stdv),activation = 'sigmoid')(decoder_2)


## Prediction network
Hidden_model = Model(inputs  = [x,u_t,x_1],outputs = [Output,z_t])

Our_Model = Model(inputs  = [x,u_t,x_1],outputs = [Output])
Our_Model.layers = Hidden_model.layers




lamm = 1.
    
def Cost_Functions_E2C(x_1,Output):
    
    xent_loss =  input_size * objectives.binary_crossentropy(x_1, Output)
    
    Entropy = 0.5 * (latent_dim*np.log(2*np.pi*np.exp(1))  + tf.log(K.sum(K.exp(z_log_var_q),axis=-1)))
                                      
    p_z = - 0.5 * (latent_dim*tf.log(2*np.pi)  +tf.log(K.sum(K.exp(_z_log_var_p),axis=-1)))  - 0.5*K.sum((z_t- _z_mean_p)*(z_t- _z_mean_p)/(K.exp(_z_log_var_p)),axis= -1 )                             

    KL_loss = 0.5 * (K.sum(K.exp(z_log_var_bar)/K.exp(_z_log_var_p),axis = -1) + K.sum((_z_mean_p- z_mean_bar)*(_z_mean_p- z_mean_bar)/(K.exp(_z_log_var_p)),axis= -1 ) - latent_dim + K.sum(_z_log_var_p,axis=-1) -  K.sum(z_log_var_bar,axis=-1) )                                    

    return xent_loss - Entropy - lamm*p_z + KL_loss

    
#hidden_model = Model(inputs, outputs + hidden_outputs)
#Our_M.layers = self.hidden_model.layers
learning_rate = 0.000000001
my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)
Our_Model.compile(optimizer=my_adam, loss=Cost_Functions_E2C)


### plane dataset
(X,U,X_1) = pickle.load(open('plane_dataset_train', 'rb'))

num_datapoints = 5000
X = X[5000:5000+num_datapoints,:]
U = U[5000:5000+num_datapoints,:]
X_1 = X_1[5000:5000+num_datapoints,:]

x_train = X.reshape((num_datapoints,image_size**2)).astype('float32')
x_train_1 = X_1.reshape((num_datapoints,image_size**2)).astype('float32')
U_train = U.astype('float32')

(X,U,X_1) = pickle.load(open('plane_dataset_validation', 'rb'))
num_datapoints = np.shape(X)[0]

validation_size = 2000

x_validation = X[0:validation_size,:,:].reshape((validation_size,image_size**2)).astype('float32')
x_validation_1 = X_1[0:validation_size,:,:].reshape((validation_size,image_size**2)).astype('float32')
U_validation = U[0:validation_size,:].astype('float32')

test_size = num_datapoints - validation_size

x_test = X[validation_size:,:,:].reshape((test_size,image_size**2)).astype('float32')
x_test_1 = X_1[validation_size:,:,:].reshape((test_size,image_size**2)).astype('float32')
U_test = U[validation_size:,:].astype('float32')




### Training the model
plane_model_weights = pickle.load(open('plane_model_weights', 'rb'))
Our_Model.set_weights(plane_model_weights)


KK = 1
model_paramter_list = {'encoder size': enc_dim, 'decoder size': decod_dim, 'trans size': trans_dim,
                        'net1 size': net1_size, 'net2 size' : net2_size, 'learning rate' : learning_rate,
                        'lambda' : lamm, 'num epochs' : num_epochs}

                        
with open('./' + aa + '/model_parameters.txt', 'w') as f:
    for key, value in model_paramter_list.items():
        f.write('%s:%s\n' % (key, value))                        
f.close()


#for ii in range(KK):
my_data = np.genfromtxt('map_plane.csv', delimiter=',')
my_data = my_data[:,0:1600]
    
Our_Model_encoder = Model(x,_z_mean_p)
ii=0
pickle.dump((ii),open('counter','wb'))
class DRAW(Callback):

    def on_epoch_end(self,batch,logs = {}):
#        ii= pickle.load(open('counter','rb'))
#        ii= ii+1
#        pickle.dump((ii),open('counter','wb'))
        x_encoded = Our_Model_encoder.predict(my_data,batch_size = batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_encoded[:,0], x_encoded[:, 1], c= np.arange(1201),linewidth = 0)
        plt.show()
#        plt.savefig('./' + aa+ '/latent epoch-' + str(ii))
#        plt.close('all')


        
    
drawmaps = DRAW()

Our_Model.fit([x_train,U_train,x_train_1], x_train_1,
    shuffle=True,
    nb_epoch= num_epochs,
    batch_size=batch_size,
    validation_data= ([x_validation,U_validation,x_validation_1],x_validation_1), callbacks = [drawmaps])
    
   
    


#    x_predicted = Our_Model.predict([x_train,U_train,x_train_1], batch_size = batch_size)
#    img_numer_to_plot = 100    
#    f, ([[ax1, ax2],[ax3,ax4]]) = plt.subplots(2, 2)
#    ax1.imshow(x_train_1[img_numer_to_plot,:].reshape((image_size,image_size)),cmap='Greys_r')
#    ax2.imshow(x_predicted[img_numer_to_plot,:].reshape((image_size,image_size)),cmap='Greys_r')   
#    ax3.imshow(x_train_1[img_numer_to_plot+1,:].reshape((image_size,image_size)),cmap='Greys_r')
#    ax4.imshow(x_predicted[img_numer_to_plot+1,:].reshape((image_size,image_size)),cmap='Greys_r') 
#    
#    
#    plt.savefig('./' + aa+ '/reconstruction epoch-' + str(ii+1))
    
    
    
    
    
#    plane_model_weights = Our_Model.get_weights()
#    my_filename = 'plane_model_parameters_'+str(ii+1)
#    pickle.dump((plane_model_weights,model_paramter_list), open('./' + aa + '/' +my_filename, 'wb'))







my_data = np.genfromtxt('map_plane.csv', delimiter=',')
my_data = my_data[:,0:1600]
Our_Model_encoder = Model(x,_z_mean_p)
x_encoded = Our_Model_encoder.predict(my_data,batch_size = batch_size)

plt.figure(figsize=(6, 6))
plt.scatter(x_encoded[:, 0], x_encoded[:, 1], c= np.arange(1201),linewidth = 0)
plt.axis('off')
plt.show()
plt.savefig('rcemapnonoise.eps', format='eps', dpi=300)



plt.figure()
#
#(X,U,X_1,TS,TSN) = pickle.load(open('plane_random_trajectory', 'rb'))
#num_datapoints = np.shape(X)[0]
#x_train = X.reshape((num_datapoints,image_size**2)).astype('float32')
#x_train_1 = X_1.reshape((num_datapoints,image_size**2)).astype('float32')
#U_train = U.astype('float32')
#
#x_predicted = Our_Model.predict([x_train,U_train,x_train_1], batch_size = batch_size)
#
#for i in range(6):
#    subplot(1,6,i+1)
#    imshow(x_predicted[2*i+1,:].reshape((40,40)),cmap='Greys_r')
#    plt.axis('off')
#    plt.savefig(str(i))
#    plt.close('all')
#    
#plt.savefig('rcepredictedtrajnonoise.eps', format='eps', dpi=300)
#
#
#
#x_train_1.tofile('a_p.csv',sep=',',format='%10.5f')
#x_predicted.tofile('b_p.csv',sep=',',format='%10.5f')