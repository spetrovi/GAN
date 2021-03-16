import csv
import glob
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.utils import *
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import random
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as K
import statistics 
# train the discriminator model
def train_discriminator(model, data_generator, g_model, n_iter=1000, n_batch=256):
    half_batch = int(n_batch / 2)
    #manualy enumerate epochos
    for i in range(n_iter):
        #get a batch of real data
        #true_data_generator = KerasBatchGenerator(dataset, num_steps, batch_size, latent_dim, num_params, skip_step=num_steps)
        data = next(data_generator.generate_for_dmodel())
        X_real, y_real = data[0], data[1]
        
        #update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
	    
        #generate 'fake' examples
        data = next(data_generator.generate_pred(g_model))
        X_fake, y_fake = data[0], data[1]
        y_fake = np.zeros((len(y_fake)))
        #update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
 
        #summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
        
def shuffle_inputs(X, y):

    X_new = np.zeros((len(X), len(X[0]), len(X[0][0])))
    y_new = []

    
    positions = [i for i in range(len(X))]

    random.shuffle(positions)

    for i, pos in enumerate(positions):
        X_new[i,:] = X[pos]
        y_new.append(y[pos])
        
        
    return X_new, np.array(y_new)


# train the discriminator model
def train_d_better(model, data_generator, g_model, n_iter=1000, n_batch=256):
    half_batch = int(n_batch / 2)
    #manualy enumerate epochos
    for i in range(n_iter):

        #get a batch of real data
        #true_data_generator = KerasBatchGenerator(dataset, num_steps, batch_size, latent_dim, num_params, skip_step=num_steps)
        data = next(data_generator.generate_for_dmodel())
        X_real, y_real = data[0], data[1]

        #generate 'fake' examples
        data = next(data_generator.generate_pred(g_model))
        X_fake, y_fake = data[0], data[1]
        y_fake = np.zeros((len(y_fake)))
        #update discriminator on fake samples

        X = np.concatenate((X_real, X_fake), axis=0)
        y = np.concatenate((y_real, y_fake), axis=0)

        X, y = shuffle_inputs(X, y)

        loss, acc = model.train_on_batch(X, y)
 
        #summarize performance
        print('>%d d_loss=%.10f, acc=%.0f%%' % (i+1, loss, acc*100))        

def my_mse(y_true,y_pred):
#    y_true = tf.transpose(tf.reshape(y_true,(10,5)))[-1]
#    y_pred = tf.transpose(tf.reshape(y_pred,(10,5)))[-1]
    y_true = tf.transpose(y_true)[-1]
    y_pred = tf.transpose(y_pred)[-1]
    return K.mean(K.square(y_pred-y_true))
    
# train the generator model
#g_model input = 5 days (X)
#g_model output = 6 days (y)
def train_g_better(g_model, data_generator, n_iter=10000, n_batch=256):
    half_batch = int(n_batch / 2)
    g_model.compile(loss='mse', optimizer='adam',  metrics=['mse', 'mae', 'mape'])    
#    g_model.fit(data_generator.generate_for_gmodel(), steps_per_epoch = 100, batch_size=10, epochs=100)
    #manualy enumerate epochos
    for i in range(n_iter):

        #get a batch of  data
        (X, y) = next(data_generator.generate_for_gmodel())
    
    
        loss, mse, mae, mape = g_model.train_on_batch(X, y)
 
        #summarize performance
        print('>%d g_loss=%.10f, mse=%.10f , mae=%.10f, mape=%.10f' % (i+1, loss, mse, mae, mape))        
        
        
#Normalise prices to [0,1] interval
def normalise(price, high, low):
    return (price - low) / (high - low)

#86.716 15.965
def denormalise(price, high=86.716, low=15.965):
    return price * (high - low) + low
    
    
def process_original(name, output, ma_days=5):
    orig = glob.glob(name)[0]
    bars = []
    closes = []
    with open(orig, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(csvreader):
            if i > 0 and float(row[5]) > 0.0:           
                open_p = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                
                if i == 1:
                    max_p = open_p
                    min_p = open_p
                
                
                if i > ma_days:
                    #moving average for past X days
                    s = 0
                    for j in range(1, 1+ma_days):
                        s += closes[-j]
                    moving_avg = s / ma_days
                    bars.append(np.array([open_p,high,low,close,moving_avg]))
                    
                    #For normalisation purposes
                    if max([open_p,high,low,close]) > max_p:
                        max_p = max([open_p,high,low,close])
                    if min([open_p,high,low,close]) < min_p:
                        min_p = min([open_p,high,low,close])
                closes.append(close)
    out = []
    print(max_p, min_p)
    for l in bars:
        out.append(list(map(lambda x: x, l)))

    return np.array(out)
	
class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, num_params, latent_dim, skip_step=1):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_params = num_params
        self.latent_dim = latent_dim
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate_for_dmodel(self):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.ones((self.batch_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))
                    

                x[i, :] = norm_days
                self.current_idx += self.skip_step

            yield (x, y)
            
    def generate_fake(self):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.zeros((self.batch_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    
                x[i, :] =[[random.uniform(0, 1) for k in range(self.num_params)] for l in range(self.num_steps+1)]
                self.current_idx += self.skip_step

            yield (x, y)
            
    #Generate training data for adversary model
    #X is batches of 5 days (inputs for generator)
    #y is ones
    def generate_for_amodel(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.ones((self.batch_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))            
                x[i, :] = norm_days
                self.current_idx += self.skip_step

            yield (x, y)
    
    #Generate training data for generator model
    #X is batches of 5 days (inputs)
    #y are the same days but with one more day (outputs)
    def generate_for_gmodel(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                y_days = self.data[self.current_idx:self.current_idx + self.num_steps + 1]
                
                mean = np.mean(y_days)
                std = np.std(y_days)
                
                x_norm_days = []
                y_norm_days = []
                for d in x_days:
                    x_norm_days.append(list((d - mean) / std))
                    
                for d in y_days:
                    y_norm_days.append(list((d - mean) / std))                    
                x[i, :] = x_norm_days
                y[i, :] = y_norm_days
                self.current_idx += self.skip_step

            yield (x, y)
            
    #Generate for training d model
    def generate_pred(self, g_model):
        x = np.zeros((self.batch_size, self.num_steps + 1, self.num_params))
        y = np.zeros((self.batch_size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    
                predict_day = g_model.predict(np.array(([self.data[self.current_idx:self.current_idx + self.num_steps]])) )[0][-1]

                x_days = self.data[self.current_idx:self.current_idx + self.num_steps]
                mean = np.mean(x_days)
                std = np.std(x_days)

                norm_days = []
                for d in x_days:
                    norm_days.append(list((d - mean) / std))     

                predict_day = (predict_day - mean ) / std
                x[i, :] = np.append(norm_days, [predict_day], 0)
                self.current_idx += self.skip_step

            yield (x, y)        
            
    def generate_eval(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_params))
        y = np.zeros((self.batch_size, self.num_steps+1, self.num_params))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]                
                y[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps+1]

                self.current_idx += self.skip_step

            yield (x, y)

"""
# define the standalone generator model
def define_generator():
    model = Sequential()
    num_steps = 5
    use_dropout = True
    num_params = 5
#	model.add(Dense(num_steps*num_params, input_dim=latent_dim))
    model.add(LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params)))
    model.add(LeakyReLU())    
    model.add(LSTM(num_params, return_sequences=True, input_shape=(num_steps,num_params)))
    model.add(LeakyReLU())    
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_params, activation='sigmoid')))
    return model
"""
 
def define_generator():    
    num_steps = 5
    num_params = 5
    A1 = Input(shape=(num_steps,num_params), name='A1')
    A2 = LSTM(32, return_sequences=True, input_shape=(num_steps,num_params), activation=LeakyReLU(), name='A2')(A1)
    A3 = LSTM(32, return_sequences=True, input_shape=(num_steps,num_params), activation=LeakyReLU(), name='A3')(A2)
    A4 = TimeDistributed(Dense(1, activation='sigmoid'), name='A4')(A3)

    B1 = tf.keras.layers.Lambda(lambda x: x, name='B1')(A1)

    A41 = tf.keras.layers.Reshape((1,5))(A4)

    C = tf.keras.layers.Concatenate(axis=1)([B1, A41])

    merged = Model(inputs=[A1], outputs=[C])
    return merged

#define the standalone discriminator model
def define_discriminator():
    num_params = 5
    num_steps = 6
    model = Sequential()
#    model.add(LSTM(64, return_sequences=True, input_shape=(num_steps,num_params)))
#    model.add(LSTM(64, return_sequences=True, input_shape=(num_steps,num_params)))
#    model.add(TimeDistributed(Dense(num_params, activation=LeakyReLU())))
    model.add(Dense(72))
    model.add(LeakyReLU(0.7))
    model.add(Dense(100))
    model.add(LeakyReLU(0.7))    
    model.add(Dense(10))
    model.add(LeakyReLU(0.7))    
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
    return model


def my_binary_crossentropy(y_true, y_pred):
    print(K.cast(y_true, 'float'))
    cce = tf.keras.losses.BinaryCrossentropy()
    return cce(y_true, y_pred)
    
    
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
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model




def train(g_model, d_model, a_model, data_generator, test_generator, n_epochs=100, n_batch=256):
    batch_size = 5
    for i in range(n_epochs):
        for j in range(batch_size):
            #Train discriminator with true and fake data
#            data = next(data_generator.generate_true())
#            X_real, y_real = data[0], data[1]
       
#            data = next(data_generator.generate_pred(g_model))
#            X_fake, y_fake = data[0], data[1]

#            X = np.concatenate((X_real, X_fake), axis=0)
#            y = np.concatenate((y_real, y_fake), axis=0)

#            d_loss, d_acc = d_model.train_on_batch(X, y)
            
            
            #Generate real examples
            data = next(data_generator.generate_for_dmodel())
            X_real, y_real = data[0], data[1]

            #generate 'fake' examples
            data = next(data_generator.generate_pred(g_model))
            X_fake, y_fake = data[0], data[1]
            y_fake = np.zeros((len(y_fake)))

            #Merge and shuffle
            X = np.concatenate((X_real, X_fake), axis=0)
            y = np.concatenate((y_real, y_fake), axis=0)
    
            X, y = shuffle_inputs(X, y)
            
            #update discriminator on fake samples
            d_loss, d_acc = d_model.train_on_batch(X, y)
            
            
            #Update generator via discriminator's error
            data = next(data_generator.generate_for_amodel())
  
            a_loss = a_model.train_on_batch(data[0], data[1])
            
        #summary
        print('>%d, %d/%d, d_loss=%.10f, d_acc=%.10f, a_loss=%.10f' % (i+1, j+1, batch_size, d_loss, d_acc, a_loss))
        g_model.compile(loss=my_mse, optimizer='adam',  metrics=['mse', 'mae', 'mape'])            
        g_model.evaluate(test_generator.generate_for_gmodel(), steps=100)





#dataset = process_original('eurusd_all_days.csv', 'eurusd_daily_processed.csv')
dataset = process_original('brent_all_days.csv', 'brent_daily_processed.csv')
train_percent = 0.9
train_vol = int(len(dataset)*train_percent)

#train_data = dataset[:train_vol]
train_data = dataset
test_data = dataset[train_vol:]




# fit the model
#train_discriminator(model, data_generator)

num_steps = 5
batch_size = 10
num_params = 5
latent_dim = 100

g_model = define_generator()
d_model = define_discriminator()
a_model = define_GAN(g_model, d_model)

data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)
test_generator = KerasBatchGenerator(test_data, num_steps, batch_size, num_params, latent_dim, skip_step=1)

#Train discriminator
#d_model.summary()
#train_d_better(d_model, data_generator, g_model)
#train_g_better(g_model, data_generator)
train(g_model, d_model, a_model, data_generator, test_generator)

#g_model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])

#save the generator model
#g_model.summary()
data_path_g = "gan_proper/1_generator"
data_path_d = "gan_proper/1_discriminator"
g_model.save(data_path_g)
d_model.save(data_path_d)


#g_model = tf.keras.models.load_model(data_path_g)


recent_data = np.array([dataset[len(dataset)-5 : len(dataset)]])
mean = np.mean(recent_data)
std = np.std(recent_data)

#normalise
norm_r = (recent_data-mean)/std


prediction = g_model.predict(norm_r)
print('Actual day:')
print(recent_data)

print('Predicted next day:')
print(mean + (prediction*std))
#for i in range(len(recent_data[0])):
#    print(list(map(lambda x: denormalise(x), recent_data[0][i])))
#    print(recent_data[0][i])

#print('Predicted next day:')
#for i in range(len(prediction)):
 #   print(prediction[i])
#    print(list(map(lambda x: (x*std)+mean, prediction[i])))






























