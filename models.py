import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Layer, LeakyReLU, Lambda, LSTM, TimeDistributed, Dropout, Flatten, RepeatVector
from tensorflow.keras.optimizers import Adam
from functions import *

def define_generator(num_steps=5, num_params=5):    
    A1 = Input(shape=(num_steps,num_params), name='A1')
    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), name='A2')(A1)
    A21 = LeakyReLU(name='A21')(A2)     
    A3 = LSTM(num_params, return_sequences=False, input_shape=(num_steps,num_params), name='A3')(A21)
    A31 = LeakyReLU(name='A31')(A3) 
    A32 = Dropout(0.5, name='A32')(A31)
#    A4 = TimeDistributed(Dense(1, activation='sigmoid'), name='A4')(A3)
#    A4 = TimeDistributed(Dense(5, activation='relu'), name='A4')(A3)
    A5 = Layer(Dense(1), name='A5')(A32)
    A51 = LeakyReLU(name='A51')(A5) 
    A6 = tf.keras.layers.Reshape((1,5))(A51)
    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A6])
    merged = Model(inputs=[A1], outputs=[C])
    merged.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])        
    return merged
    

#define the standalone discriminator model
#in order to perfect loss function, we need to pass input to output and concatenate with prediction
#input is list of 6 days, we need to concatenate it wih 6 times discriminator prediction
#we need to use functional model for this
def define_discriminator(num_steps=6, num_params=5):
    
    A1 = Input(shape=(num_steps,num_params), name='A1')
    A2 = Dense(7, name='A2')(A1)
    A21 = LeakyReLU(name='A21')(A2)   
    A3 = Dense(10, name='A3')(A21)
    A31 = LeakyReLU(name='A31')(A3)
    A4 = Dense(10, name='A4')(A31)
    A41 = LeakyReLU(name='A41')(A4)
    
    #Ok, here it gets messy, so I'll explain a bit
    #We have some computation going on with, but our input shape is num_steps*num_params
    #That means, that each node's output is array of num_steps elements
    #But for prediction, we only want one!
    #So we use Flatten. This takes the outputs and make a single straight array, that we densely connect to single neuron
    A42 = Flatten(input_shape=(num_steps, 1))(A41)
    A5 = Dense(1, activation='sigmoid', name='A5')(A42)
    #Ok, we have our predicted value in A5, however, we can't glue it to the original values,
    #Tensorflow wants element of array to have the same dimmension
    #Our input array, that we want to pass is num_steps arrays of num_params elements
    #In order to glue our prediction, we need to make a fake num_params array
    #To make it a num_params array, we can use RepeatVector to just repeat the predicted value
    #Don't forget, later in loss function, we need to make the prediction back to single value    
    A51 = RepeatVector(num_params)(A5)

    #Reshaping, cause tf is stupid (or too smart actually)
    A6 = tf.keras.layers.Reshape((1,num_params))(A51)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A6])
    
    merged = Model(inputs=[A1], outputs=[C])
    merged.compile(loss=my_dloss, optimizer='adam',  metrics=[my_dacc])
    return merged


    
#define GAN by sequentialy combining generator and discriminator
#We freeze the discriminator model, so it is updated only when we train it alone
#When we train GAN, we want to update only generator
def define_GAN(g_model, d_model):
    #make weights in discriminator not trainable
    d_model.trainable = False
    #connect
    model = Sequential()
    #add generator
    model.add(g_model)
    #add discriminator
    model.add(d_model)
    #compile
    #maybe this Adam settings is not good
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=my_aloss, optimizer='adam')
    return model
