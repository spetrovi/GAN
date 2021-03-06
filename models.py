import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Layer, LeakyReLU, Lambda, LSTM, TimeDistributed, Dropout, Flatten, RepeatVector
from tensorflow.keras.optimizers import Adam
from functions import *

def define_generator(num_steps, num_params):    
    A1 = Input(shape=(num_steps,num_params))

    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='relu')(A1)
    A3 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='relu')(A2)
    A4 = Dropout(0.0)(A3)

    A5 = TimeDistributed(Dense(num_params))(A4)
    
    A6 = Flatten(input_shape=(num_steps, num_params))(A5)

    A7 = Dense(num_params)(A6)

    A8 = tf.keras.layers.Reshape((1,num_params))(A7)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    merged = Model(inputs=[A1], outputs=[C])
    merged.compile(loss=my_gMSE, optimizer='adam', metrics=[my_gMSE, my_gMAE_l, my_gMAE_h])        

    return merged
    

def define_discriminator(num_steps, num_params):
    num_steps = num_steps + 1
    A1 = Input(shape=(num_steps,num_params))
    A2 = Dense(72, activation='relu')(A1)
    A3 = Dense(100, activation='relu')(A2)
    A4 = Dense(10, activation='relu')(A3)

    A5 = Dropout(0.0)(A4)
    A6 = Flatten(input_shape=(num_steps, 1))(A5)
    A7 = Dense(1, activation='sigmoid')(A6)
    A8 = RepeatVector(num_params)(A7)

    #Reshaping, cause tf is stupid (or too smart actually)
    A9 = tf.keras.layers.Reshape((1,num_params))(A8)

    B1 = tf.keras.layers.Lambda(lambda x: x)(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A6])
    
    merged = Model(inputs=[A1], outputs=[C])
    opt = Adam(lr=0.006, beta_1=0.9)
    merged.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
    return merged

def define_discriminator_LSTM(num_steps, num_params):
    num_steps = num_steps + 1
    A1 = Input(shape=(num_steps,num_params))

    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='softmax')(A1)    
    A3 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='softmax')(A2)
    A31 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='softmax')(A3)    

    A4 = Dropout(0.0)(A31)
    A5 = Flatten(input_shape=(num_steps, 1))(A4)
    
    A6 = Dense(1, activation='sigmoid')(A5)
    A7 = RepeatVector(num_params)(A6)

    #Reshaping, cause tf is stupid (or too smart actually)
    A8 = tf.keras.layers.Reshape((1,num_params))(A7)

    B1 = tf.keras.layers.Lambda(lambda x: x)(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    
    merged = Model(inputs=[A1], outputs=[C])
    opt = Adam(lr=0.006, beta_1=0.9)
    merged.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
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
    opt = Adam(lr=0.006, beta_1=0.9)
    model.compile(loss=my_aloss, optimizer=opt)
    return model
