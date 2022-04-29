import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Layer, LeakyReLU, Lambda, LSTM, TimeDistributed, Dropout, Flatten, RepeatVector, Conv1D, ReLU, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from functions import *

def define_generator(num_steps, num_params, opt):    
    A1 = Input(shape=(num_steps, num_params))

    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A1)
    A3 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A2)
    A4 = Dropout(0.1)(A3)

    A5 = TimeDistributed(Dense(num_params))(A4)
    
    A6 = Flatten(input_shape=(num_steps, num_params))(A5)

    A7 = Dense(num_params)(A6)

    A8 = tf.keras.layers.Reshape((1,num_params))(A7)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    merged = Model(inputs=[A1], outputs=[C])
#    opt = Adam(lr=0.001, beta_1=0.999)    
    merged.compile(loss=my_gMAE, optimizer=opt, metrics=[my_gMAE]) 
    return merged
    
def define_generator_OHLC(num_steps, num_params):    
    A1 = Input(shape=(num_steps, num_params))

    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A1)
    A3 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A2)
    A4 = Dropout(0.1)(A3)

    A5 = TimeDistributed(Dense(num_params))(A4)
    
    A6 = Flatten(input_shape=(num_steps, num_params))(A5)

    A7 = Dense(4)(A6)

    A8 = tf.keras.layers.Reshape((1, 4))(A7)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    merged = Model(inputs=[A1], outputs=[C])
    opt = Adam(lr=0.001, beta_1=0.999)    
    merged.compile(loss=my_gMSE_o_c, optimizer=opt, metrics=[my_gMSE, my_gMSE_o_c])        

    return merged    
    

def define_discriminator(num_steps, num_params, opt):
    num_steps = num_steps + 1
    A1 = Input(shape=(num_steps,num_params))
    A2 = Dense(72, activation=LeakyReLU(alpha=0.01))(A1)
    A3 = Dense(100, activation=LeakyReLU(alpha=0.01))(A2)
    A4 = Dense(10, activation=LeakyReLU(alpha=0.01))(A3)

    A5 = Dropout(0.0)(A4)
    A6 = Flatten(input_shape=(num_steps, 1))(A5)
    A7 = Dense(1, activation='sigmoid')(A6)
    A8 = RepeatVector(num_params)(A7)

    #Reshaping, cause tf is stupid (or too smart actually)
    A9 = tf.keras.layers.Reshape((1,num_params))(A8)

    B1 = tf.keras.layers.Lambda(lambda x: x)(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A9])
    
    merged = Model(inputs=[A1], outputs=[C])
#    opt = Adam(lr=0.001, beta_1=0.999)
    merged.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
    return merged

def generator_orig(num_steps, num_params, opt):
    A1 = Input(shape=(num_steps,num_params))
    A2 = GRU(units=256, return_sequences = True, input_shape=(num_steps, num_params), recurrent_dropout=0.2, recurrent_regularizer=regularizers.l2(1e-3))(A1)
    A3 = GRU(units=128, recurrent_dropout=0.2, recurrent_regularizer=regularizers.l2(1e-3))(A2)
#    A4 = GRU(units=64, recurrent_dropout=0.2)(A3)
    A5 = Dense(32, kernel_regularizer=regularizers.l2(1e-3))(A3)
    A6 = Dense(16, kernel_regularizer=regularizers.l2(1e-3))(A5)
    A7 = Dense(num_params)(A6)
    A8 = tf.keras.layers.Reshape((1,num_params))(A7)
    
    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    merged = Model(inputs=[A1], outputs=[C])
#    opt = Adam(lr=0.001, beta_1=0.999)
    merged.compile(loss=my_gMAE, optimizer=opt, metrics=[my_gMAE])
    return merged 

def discriminator_orig(num_steps, num_params, opt):
    num_steps = num_steps + 1
    A1 = Input(shape=(num_steps,num_params))
    A2 = Conv1D(72, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))(A1)
    A3 = Conv1D(100, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))(A2)
    A4 = Conv1D(10, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01))(A3)

    A5 = Dropout(0.0)(A4)
    A6 = Flatten(input_shape=(num_steps, 1))(A5)
#    A61 = Dense(220, use_bias=True)(A6)
#    A7 = LeakyReLU()(A61)
#    A8 = Dense(220, use_bias=True)(A7)
#    A9 = ReLU()(A8)
    
    A10 = Dense(1, activation='sigmoid')(A6)
    A11 = RepeatVector(num_params)(A10)

    #Reshaping, cause tf is stupid (or too smart actually)
    A12 = tf.keras.layers.Reshape((1,num_params))(A11)

    B1 = tf.keras.layers.Lambda(lambda x: x)(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A12])
    
    merged = Model(inputs=[A1], outputs=[C])
#    opt = Adam(lr=0.001, beta_1=0.999)
    merged.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
    return merged


def define_discriminator_LSTM(num_steps, num_params, opt):
    num_steps = num_steps + 1
    A1 = Input(shape=(num_steps,num_params))

    A2 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A1)    
    A3 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='tanh')(A2)
#    A31 = LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params), activation='softmax')(A3)    
    A31 = TimeDistributed(Dense(num_params))(A3)
    A4 = Dropout(0.1)(A31)
    A5 = Flatten(input_shape=(num_steps, 1))(A4)
    
    A6 = Dense(1, activation='sigmoid')(A5)
    A7 = RepeatVector(num_params)(A6)

    #Reshaping, cause tf is stupid (or too smart actually)
    A8 = tf.keras.layers.Reshape((1,num_params))(A7)

    B1 = tf.keras.layers.Lambda(lambda x: x)(A1)
    C = tf.keras.layers.Concatenate(axis=1)([B1, A8])
    
    merged = Model(inputs=[A1], outputs=[C])
#    opt = Adam(lr=0.001, beta_1=0.999)
    merged.compile(loss=my_dloss, optimizer=opt,  metrics=[my_dacc])
    return merged    

    
#define GAN by sequentialy combining generator and discriminator
#We freeze the discriminator model, so it is updated only when we train it alone
#When we train GAN, we want to update only generator
def define_GAN(g_model, d_model, opt):
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
#    opt = Adam(lr=0.001, beta_1=0.999)
    model.compile(loss=my_aloss, optimizer=opt)
    return model
